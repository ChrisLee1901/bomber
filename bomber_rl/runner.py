"""Headless training, demonstration, and reproducible evaluation commands."""

import csv
import json
import logging
import os
import shutil
from argparse import Namespace
from itertools import product
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import Adam
from tqdm import tqdm

import settings as game_settings
from environment import BombeRLeWorld
from .agent import _save, finish_training
from .core import ACTION_TO_ID, BoardDQN, append_csv, device_for, encode_state, load_checkpoint, observation_shape, safe_action_mask, save_checkpoint, seed_everything, shaped_reward

EVAL_FIELDS = ("step", "opponent_set", "mean_return", "std_return", "win_rate", "death_rate", "returns")


def load_config(path, gamma=None, total_steps=None):
    config = OmegaConf.load(path)
    if gamma is not None: config.train.gamma = float(gamma)
    if total_steps is not None: config.train.total_steps = int(total_steps)
    return config


def run_dir_for(config, seed): return Path(config.output.root) / "dqn" / f"gamma_{float(config.train.gamma):.2f}" / f"seed_{seed}"


def _option(config, path, default):
    value = OmegaConf.select(config, path, default=default)
    return default if value is None else value


def _world_args(config, seed, log_dir):
    return Namespace(no_gui=True, fps=0, turn_based=False, update_interval=0.0, save_replay=False, replay=None, make_video=False, continue_without_training=False, log_dir=str(log_dir), save_stats=False, match_name=None, seed=int(seed), silence_errors=False, scenario=str(config.environment.scenario))


def _new_world(config, seed, run_dir, training=False, agent_name="dqn_agent", agents=None):
    game_settings.LOG_GAME = game_settings.LOG_AGENT_WRAPPER = game_settings.LOG_AGENT_CODE = logging.WARNING
    log_dir = Path(run_dir) / "logs"; log_dir.mkdir(parents=True, exist_ok=True)
    agents = agents or [(agent_name, training), *[(name, False) for name in config.environment.opponents]]
    return BombeRLeWorld(_world_args(config, seed, log_dir), agents)


def _play_one_round(world):
    world.new_round()
    while world.running: world.do_step()


def _metrics(world):
    learner = world.agents[0]
    return {"return": float(learner.score), "win": int(learner.score > max(agent.score for agent in world.agents[1:])), "death": int(learner.dead), "coins": int(learner.statistics["coins"]), "crates": int(learner.statistics["crates"]), "length": int(learner.statistics["steps"])}


def _opponent_sets(config, override=None):
    if override: return {"custom": list(override)}
    configured = _option(config, "eval.opponent_sets", None)
    if configured: return {str(name): list(opponents) for name, opponents in configured.items()}
    return {"default": list(config.environment.opponents)}


def _seeds(config, final=False):
    return list(_option(config, "eval.test_seeds" if final else "eval.validation_seeds", _option(config, "eval.seeds", [])))


def _evaluation_config(config, path, opponents):
    copied = OmegaConf.create(OmegaConf.to_container(config, resolve=True)); copied.environment.opponents = list(opponents)
    OmegaConf.save(copied, path, resolve=True); return copied


def evaluate_policy(config, checkpoint, episodes=None, agent_name="dqn_agent", opponents=None, final=False, label="evaluate"):
    checkpoint = Path(checkpoint).resolve(); episodes = int(episodes or (config.eval.final_episodes if final else config.eval.periodic_episodes))
    previous = {key: os.environ.get(key) for key in ("BOMBER_RL_CONFIG", "BOMBER_RL_RUN_DIR", "BOMBER_RL_CHECKPOINT", "BOMBER_RL_SEED", "BOMBER_RL_RESUME")}
    eval_config_path = checkpoint.parent / ".evaluation_config.yaml"; active = _evaluation_config(config, eval_config_path, opponents or config.environment.opponents)
    os.environ.update({"BOMBER_RL_CONFIG": str(eval_config_path), "BOMBER_RL_CHECKPOINT": str(checkpoint), "BOMBER_RL_RUN_DIR": str(checkpoint.parent), "BOMBER_RL_SEED": "0", "BOMBER_RL_RESUME": "0"})
    results, seeds = [], _seeds(active, final)[:episodes]
    try:
        for episode_seed in tqdm(seeds, desc=label, unit="game"):
            seed_everything(int(episode_seed), deterministic=True); world = _new_world(active, episode_seed, checkpoint.parent, False, agent_name)
            _play_one_round(world); results.append(_metrics(world)); world.end()
    finally:
        for key, value in previous.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
    returns = [item["return"] for item in results]
    return {"mean_return": float(np.mean(returns)) if returns else 0.0, "std_return": float(np.std(returns)) if returns else 0.0, "win_rate": float(np.mean([item["win"] for item in results])) if results else 0.0, "death_rate": float(np.mean([item["death"] for item in results])) if results else 0.0, "returns": returns, "episodes": results, "seeds": seeds}


def evaluate_sets(config, checkpoint, episodes=None, final=False, opponents=None):
    return {name: evaluate_policy(config, checkpoint, episodes, opponents=opponent_set, final=final, label=f"evaluate:{name}") for name, opponent_set in _opponent_sets(config, opponents).items()}


def _metric(result): return (result["win_rate"], result["mean_return"], -result["death_rate"])


def _stage_config(config, step):
    stages = _option(config, "curriculum.stages", [])
    for index, stage in enumerate(stages):
        if int(step) < int(stage.until_step):
            copied = OmegaConf.create(OmegaConf.to_container(config, resolve=True)); copied.environment.opponents = list(stage.opponents); return index, copied
    return max(len(stages) - 1, 0), config


def train(config, seed, run_dir=None, resume=True, pretrained=None, demos=None):
    run_dir = (Path(run_dir) if run_dir else run_dir_for(config, seed)).resolve(); latest = run_dir / "latest.pt"
    if not resume and (latest.exists() or (run_dir / "train.csv").exists()): raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True); config_path = run_dir / "resolved_config.yaml"; OmegaConf.save(config, config_path, resolve=True)
    os.environ.update({"BOMBER_RL_CONFIG": str(config_path), "BOMBER_RL_RUN_DIR": str(run_dir), "BOMBER_RL_SEED": str(seed), "BOMBER_RL_RESUME": "1" if resume else "0"})
    if pretrained: os.environ["BOMBER_RL_PRETRAINED"] = str(Path(pretrained).resolve())
    if demos: os.environ["BOMBER_RL_DEMOS"] = str(Path(demos).resolve())
    seed_everything(seed, deterministic=True); stage, active = _stage_config(config, 0); os.environ["BOMBER_RL_STAGE"] = str(stage)
    OmegaConf.save(active, config_path, resolve=True); world = _new_world(active, seed, run_dir, True); learner = world.agents[0].backend.runner.fake_self
    next_eval = ((learner.steps // int(config.eval.frequency)) + 1) * int(config.eval.frequency); target_steps = int(config.train.total_steps); progress = tqdm(total=target_steps, initial=min(learner.steps, target_steps), desc="train", unit="step")
    try:
        while learner.steps < target_steps:
            desired_stage, desired = _stage_config(config, learner.steps)
            if desired_stage != stage:
                _save(learner); world.end(); stage = desired_stage; active = desired; os.environ["BOMBER_RL_STAGE"] = str(stage); OmegaConf.save(active, config_path, resolve=True)
                world = _new_world(active, seed + stage, run_dir, True); learner = world.agents[0].backend.runner.fake_self
            world.new_round()
            while world.running and learner.steps < target_steps:
                previous_step = learner.steps; world.do_step(); progress.update(learner.steps - previous_step)
                progress.set_postfix(stage=stage, loss="" if learner.last_loss is None else f"{learner.last_loss:.4f}")
            if world.running: world.end_round()
            if learner.steps >= next_eval:
                _save(learner); periodic_sets = evaluate_sets(config, latest, config.eval.periodic_episodes)
                for name, periodic in periodic_sets.items():
                    append_csv(run_dir / "periodic_eval.csv", EVAL_FIELDS, {"step": learner.steps, "opponent_set": name, "mean_return": f"{periodic['mean_return']:.6f}", "std_return": f"{periodic['std_return']:.6f}", "win_rate": f"{periodic['win_rate']:.6f}", "death_rate": f"{periodic['death_rate']:.6f}", "returns": json.dumps(periodic["returns"])})
                selected = periodic_sets.get("default", next(iter(periodic_sets.values())))
                if _metric(selected) > tuple(learner.best_metric): learner.best_metric = _metric(selected); _save(learner); shutil.copy2(latest, run_dir / "best.pt")
                next_eval += int(config.eval.frequency)
        finish_training(learner)
        if not (run_dir / "best.pt").exists(): shutil.copy2(run_dir / "final.pt", run_dir / "best.pt")
        final_sets = evaluate_sets(config, run_dir / "best.pt", config.eval.final_episodes, final=True)
        final = {"algorithm": "dqn", "training_seed": int(seed), "gamma": float(config.train.gamma), "checkpoint": "best.pt", "selection_metric": list(learner.best_metric), "opponent_sets": final_sets}
        (run_dir / "final_evaluation.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
        with (run_dir / "final_evaluation.csv").open("w", newline="", encoding="utf-8") as handle:
            rows = [{"opponent_set": name, **{key: value for key, value in values.items() if key not in ("episodes", "returns")}} for name, values in final_sets.items()]
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
        return final
    finally:
        progress.close(); world.end(); os.environ.pop("BOMBER_RL_PRETRAINED", None); os.environ.pop("BOMBER_RL_DEMOS", None)


def evaluate(config, checkpoint, episodes=None, output=None, final=False, opponents=None):
    result = {"checkpoint": str(checkpoint), "opponent_sets": evaluate_sets(config, checkpoint, episodes, final, opponents)}
    if output:
        output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        rows = [{"opponent_set": name, **{key: value for key, value in values.items() if key not in ("episodes", "returns")}} for name, values in result["opponent_sets"].items()]
        with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    return result


def write_demo_shard(path, transitions):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle: np.savez_compressed(handle, **{key: np.asarray([item[key] for item in transitions]) for key in transitions[0]})
    os.replace(temporary, path)


def _demo_progress(output_dir):
    shards = sorted(output_dir.glob("demo_*.npz"))
    if not shards: return 0, 0
    try: indices = [int(path.stem.removeprefix("demo_")) for path in shards]
    except ValueError as error: raise ValueError("demo shard names must use demo_00000.npz") from error
    last_episode = -1
    for path in shards:
        with np.load(path) as shard:
            required = {"observation", "action", "reward", "next_observation", "terminated", "next_action_mask", "episode_id", "opponent_set"}
            if not required.issubset(shard.files): raise ValueError(f"invalid demo shard: {path}")
            last_episode = max(last_episode, int(shard["episode_id"].max(initial=-1)))
    return last_episode + 1, max(indices) + 1


def _write_demo_manifest(output_dir, payload):
    path = output_dir / "manifest.json"; temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8"); os.replace(temporary, path)


def collect_demos(config, episodes, output_dir, agents=None, shard_size=50000, resume=False, seed_start=0):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); agents = list(agents or ["rule_based_agent"] * 4); transitions = []
    start_episode, shard = _demo_progress(output_dir)
    if start_episode and not resume: raise FileExistsError(f"demo shards already exist in {output_dir}; rerun with --resume")
    start_episode = max(start_episode, int(seed_start)); end_episode = int(seed_start) + int(episodes)
    if start_episode > end_episode: raise ValueError("existing demo episodes exceed the requested total; increase --episodes")
    include_danger, opponent_set = bool(_option(config, "state.include_danger", False)), "|".join(agents)
    metadata = {"agents": agents, "include_danger": include_danger, "seed_start": int(seed_start), "shard_size": int(shard_size)}; manifest = output_dir / "manifest.json"
    if resume and manifest.exists():
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if any(previous.get(key) != value for key, value in metadata.items()): raise ValueError("demo resume config differs from manifest")
    _write_demo_manifest(output_dir, metadata | {"target_episodes": end_episode})
    progress = tqdm(total=end_episode, initial=start_episode, desc="collect demos", unit="game")
    try:
        for episode_id in range(start_episode, end_episode):
            world = _new_world(config, episode_id, output_dir, agents=[(name, False) for name in agents])
            try:
                world.new_round(); world.user_input = "WAIT"
                while world.running:
                    before = {agent.name: world.get_state_for_agent(agent) for agent in world.active_agents}; world.do_step()
                    for agent in world.agents:
                        old = before.get(agent.name); actions = world.replay["actions"].get(agent.name, [])
                        if old is None or not actions or actions[-1] not in ACTION_TO_ID: continue
                        new = None if agent.dead or not world.running else world.get_state_for_agent(agent); _, reward = shaped_reward(agent.events); observation = encode_state(old, include_danger)
                        transitions.append({"observation": observation, "action": np.int64(ACTION_TO_ID[actions[-1]]), "reward": np.float32(reward), "next_observation": np.zeros_like(observation) if new is None else encode_state(new, include_danger), "terminated": np.bool_(new is None), "next_action_mask": np.ones(6, dtype=np.bool_) if new is None else safe_action_mask(new), "episode_id": np.int64(episode_id), "opponent_set": np.str_(opponent_set)})
                        if len(transitions) >= int(shard_size): write_demo_shard(output_dir / f"demo_{shard:05d}.npz", transitions); transitions.clear(); shard += 1
            finally:
                world.end()
            progress.update(1); progress.set_postfix(shards=shard)
    finally:
        progress.close()
    if transitions: write_demo_shard(output_dir / f"demo_{shard:05d}.npz", transitions)
    completed, _ = _demo_progress(output_dir); _write_demo_manifest(output_dir, metadata | {"target_episodes": end_episode, "completed_episodes": completed})
    return {"shards": len(list(output_dir.glob("demo_*.npz"))), "completed_episodes": completed, "output_dir": str(output_dir)}


def pretrain(config, demos, output, epochs=20, batch_size=128, patience=4, resume=True, seed=42):
    paths = sorted(Path(demos).glob("*.npz")) if Path(demos).is_dir() else [Path(demos)]
    if not paths: raise FileNotFoundError("no demonstration shards found")
    output = Path(output); latest = output.with_name(f"{output.stem}.latest{output.suffix}")
    include_danger = bool(_option(config, "state.include_danger", False)); shape = observation_shape(include_danger); device = device_for(config.device); options = {"include_danger": include_danger, "dueling": bool(_option(config, "algorithm.dueling", False))}
    model = BoardDQN(input_channels=shape[0], dueling=options["dueling"]).to(device); optimizer = Adam(model.parameters(), lr=float(config.train.learning_rate)); start_epoch = 0; best, stale = float("inf"), 0
    if resume and latest.exists():
        checkpoint = load_checkpoint(latest, device)
        if tuple(checkpoint.get("observation", {}).get("shape", ())) != shape: raise ValueError("pretrain checkpoint observation shape does not match config")
        model.load_state_dict(checkpoint["model_state_dict"]); optimizer.load_state_dict(checkpoint["optimizer_state_dict"]); start_epoch, best, stale = int(checkpoint["pretrain_epoch"]), float(checkpoint["best_validation_loss"]), int(checkpoint["pretrain_stale"])
        np.random.set_state(checkpoint["numpy_rng_state"]); torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
    elif resume and output.exists():
        raise FileNotFoundError(f"missing pretrain resume checkpoint: {latest}")
    elif not resume and (output.exists() or latest.exists()):
        raise FileExistsError(f"pretrain checkpoint already exists: {output}")
    else:
        seed_everything(seed, deterministic=True)
    for epoch in tqdm(range(start_epoch, int(epochs)), desc="pretrain", unit="epoch"):
        validation_losses = []
        for path in tqdm(paths, desc=f"pretrain {epoch + 1}/{epochs}", unit="shard", leave=False):
            with np.load(path) as shard:
                observations, actions, episode_ids = shard["observation"], shard["action"], shard["episode_id"]
                unique_episodes = np.unique(episode_ids); validation_episodes = set(unique_episodes[::10]) if len(unique_episodes) > 1 else set()
                for validation in (False, True):
                    indices = np.asarray([index for index, episode_id in enumerate(episode_ids) if (episode_id in validation_episodes) == validation])
                    if not len(indices): continue
                    if not validation: np.random.shuffle(indices)
                    for start in range(0, len(indices), int(batch_size)):
                        batch = indices[start:start + int(batch_size)]; x = torch.as_tensor(observations[batch], device=device); y = torch.as_tensor(actions[batch], device=device)
                        if validation:
                            with torch.no_grad(): validation_losses.append(float(torch.nn.functional.cross_entropy(model(x), y).cpu()))
                        else:
                            optimizer.zero_grad(set_to_none=True); loss = torch.nn.functional.cross_entropy(model(x), y); loss.backward(); optimizer.step()
        validation_loss = float(np.mean(validation_losses)) if validation_losses else 0.0
        tqdm.write(f"pretrain epoch {epoch + 1}/{epochs}: validation_loss={validation_loss:.6f}")
        if validation_loss < best:
            best, stale = validation_loss, 0; save_checkpoint(output, {"algorithm": "dqn", "model_state_dict": model.state_dict(), "target_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "step": 0, "observation": {"shape": list(shape), "dtype": "uint8"}, "model_options": options, "pretrain_validation_loss": best})
        else:
            stale += 1
        save_checkpoint(latest, {"algorithm": "dqn", "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "observation": {"shape": list(shape)}, "pretrain_epoch": epoch + 1, "best_validation_loss": best, "pretrain_stale": stale, "numpy_rng_state": np.random.get_state(), "torch_rng_state": torch.get_rng_state()})
        if stale >= int(patience): break
    return {"checkpoint": str(output), "latest_checkpoint": str(latest), "validation_loss": best}


def run_experiment(experiment_path):
    experiment = OmegaConf.load(experiment_path); base_path = Path(experiment_path).parent.parent / str(experiment.base_config)
    for gamma, seed in tqdm(list(product(experiment.gammas, experiment.seeds)), desc="experiment", unit="run"):
        config = load_config(base_path, gamma=gamma, total_steps=experiment.total_steps); config.output.root = str(experiment.output_root); run_dir = run_dir_for(config, seed)
        if not (run_dir / "final_evaluation.json").exists(): train(config, int(seed), run_dir=run_dir, resume=True)


def aggregate(experiment_path):
    experiment = OmegaConf.load(experiment_path); base_path = Path(experiment_path).parent.parent / str(experiment.base_config); result_dir = Path(experiment.results_dir); result_dir.mkdir(parents=True, exist_ok=True); rows = []
    for gamma in tqdm(experiment.gammas, desc="aggregate", unit="gamma"):
        values = []
        for seed in experiment.seeds:
            path = run_dir_for(load_config(base_path, gamma=gamma, total_steps=experiment.total_steps), seed) / "final_evaluation.json"; evaluation = json.loads(path.read_text(encoding="utf-8")); values.append(evaluation["opponent_sets"].get("default", next(iter(evaluation["opponent_sets"].values()))))
        rows.append({"algorithm": "dqn", "gamma": float(gamma), "n_seeds": len(values), "mean_return": float(np.mean([item["mean_return"] for item in values])), "win_rate": float(np.mean([item["win_rate"] for item in values]))})
    with (result_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle: writer = csv.DictWriter(handle, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
