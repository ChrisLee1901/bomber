"""Callback implementation used by ``agent_code/dqn_agent``."""

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.optim import Adam

from .core import ACTIONS, ACTION_TO_ID, BoardDQN, ReplayBuffer, action_mask, append_csv, device_for, encode_state, epsilon, load_checkpoint, observation_shape, optimize, safe_action_mask, save_checkpoint, seed_everything, shaped_reward, sync

TRAIN_FIELDS = ("step", "episode", "environment_return", "shaped_return", "length", "epsilon", "loss", "win", "death", "coins", "crates", "elapsed_seconds")


def _required_environment(name):
    value = os.environ.get(name)
    if not value: raise RuntimeError(f"{name} is required; start this agent through `python main.py train` or `evaluate`.")
    return value


def _option(config, path, default):
    value = OmegaConf.select(config, path, default=default)
    return default if value is None else value


def _model_options(config):
    return {"include_danger": bool(_option(config, "state.include_danger", False)), "dueling": bool(_option(config, "algorithm.dueling", False))}


def _preload_demos(replay, directory):
    paths = [Path(directory)] if Path(directory).is_file() else sorted(Path(directory).glob("*.npz"))
    for path in paths:
        with np.load(path) as shard:
            for index in range(len(shard["actions"])):
                if len(replay) >= replay.capacity: return
                replay.add(shard["observation"][index], int(shard["action"][index]), float(shard["reward"][index]), shard["next_observation"][index], bool(shard["terminated"][index]), shard["next_action_mask"][index])


def _validate_checkpoint(checkpoint, options):
    observed = tuple(checkpoint.get("observation", {}).get("shape", ()))
    expected = observation_shape(options["include_danger"])
    if observed and observed != expected: raise ValueError(f"checkpoint observation shape {observed} does not match configured {expected}")


def _checkpoint(self):
    return {
        "algorithm": "dqn", "model_state_dict": self.model.state_dict(), "target_state_dict": self.target.state_dict(), "optimizer_state_dict": self.optimizer.state_dict(), "scaler_state_dict": self.scaler.state_dict(),
        "step": self.steps, "episode": self.episode, "seed": self.seed, "config": OmegaConf.to_container(self.config, resolve=True), "action_names": list(ACTIONS),
        "observation": {"shape": list(self.observation_shape), "dtype": "uint8", "scale": 4.0}, "model_options": self.model_options,
        "replay_state": self.replay.state_dict() if self.train else None, "best_metric": self.best_metric, "curriculum_stage": self.curriculum_stage,
        "rng_state": self.rng.bit_generator.state, "python_rng_state": random.getstate(), "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _save(self, final=False):
    save_checkpoint(self.run_dir / "latest.pt", _checkpoint(self))
    if final: save_checkpoint(self.run_dir / "final.pt", _checkpoint(self))


def _restore(self, checkpoint):
    _validate_checkpoint(checkpoint, self.model_options)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    if checkpoint.get("target_state_dict"): self.target.load_state_dict(checkpoint["target_state_dict"])
    else: sync(self.model, self.target)
    if checkpoint.get("optimizer_state_dict"): self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if checkpoint.get("scaler_state_dict"): self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    self.steps, self.episode = int(checkpoint.get("step", 0)), int(checkpoint.get("episode", 0))
    self.best_metric = tuple(checkpoint.get("best_metric", (-float("inf"), -float("inf"), -float("inf"))))
    self.curriculum_stage = int(checkpoint.get("curriculum_stage", 0))
    if checkpoint.get("replay_state"): self.replay.load_state_dict(checkpoint["replay_state"])
    if checkpoint.get("rng_state"):
        self.rng.bit_generator.state = checkpoint["rng_state"]; random.setstate(checkpoint["python_rng_state"]); torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state"): torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])


def setup(self):
    self.config = OmegaConf.load(_required_environment("BOMBER_RL_CONFIG")); self.run_dir = Path(_required_environment("BOMBER_RL_RUN_DIR")); self.run_dir.mkdir(parents=True, exist_ok=True)
    self.seed, self.device, self.model_options = int(os.environ.get("BOMBER_RL_SEED", "42")), device_for(self.config.device), _model_options(self.config)
    self.observation_shape = observation_shape(self.model_options["include_danger"]); seed_everything(self.seed); self.rng = np.random.default_rng(self.seed)
    self.double_dqn = bool(_option(self.config, "algorithm.double_dqn", False)); self.safe_actions = bool(_option(self.config, "safety.enabled", False)); self.augment = bool(_option(self.config, "augmentation.enabled", False))
    self.prioritized = bool(_option(self.config, "per.enabled", False)); self.per_beta_start = float(_option(self.config, "per.beta_start", .4)); self.per_beta_steps = int(_option(self.config, "per.beta_steps", 1))
    self.model = BoardDQN(input_channels=self.observation_shape[0], dueling=self.model_options["dueling"]).to(self.device); self.target = BoardDQN(input_channels=self.observation_shape[0], dueling=self.model_options["dueling"]).to(self.device)
    self.optimizer = Adam(self.model.parameters(), lr=float(self.config.train.learning_rate)); self.scaler = torch.amp.GradScaler("cuda", enabled=self.device.type == "cuda")
    self.steps = self.episode = self.saved_checkpoint_bucket = 0; self.last_loss = None; self._last_transition = None; self.best_metric = (-float("inf"), -float("inf"), -float("inf")); self.curriculum_stage = int(os.environ.get("BOMBER_RL_STAGE", 0))
    self._example_state = {"field": np.zeros((17, 17), dtype=np.int8), "coins": [], "bombs": [], "explosion_map": np.zeros((17, 17)), "self": ("dqn_agent", 0, True, (1, 1)), "others": []}
    if self.train:
        self.replay = ReplayBuffer(self.config.train.replay_capacity, self.rng, self.observation_shape, self.prioritized, float(_option(self.config, "per.alpha", .6)))
        latest = self.run_dir / "latest.pt"
        if os.environ.get("BOMBER_RL_RESUME") == "1" and latest.exists(): _restore(self, load_checkpoint(latest, self.device))
        elif os.environ.get("BOMBER_RL_PRETRAINED"):
            checkpoint = load_checkpoint(os.environ["BOMBER_RL_PRETRAINED"], self.device); _validate_checkpoint(checkpoint, self.model_options); self.model.load_state_dict(checkpoint["model_state_dict"]); sync(self.model, self.target)
        else: sync(self.model, self.target)
        if os.environ.get("BOMBER_RL_DEMOS"): _preload_demos(self.replay, os.environ["BOMBER_RL_DEMOS"])
        self.started_at = time.time(); self.model.train()
    else:
        checkpoint = load_checkpoint(_required_environment("BOMBER_RL_CHECKPOINT"), self.device)
        if checkpoint.get("algorithm") != "dqn": raise ValueError("only DQN checkpoints are supported")
        _validate_checkpoint(checkpoint, self.model_options); self.model.load_state_dict(checkpoint["model_state_dict"]); self.model.eval()


def state_to_features(game_state): return encode_state(game_state)


def act(self, game_state):
    observation = encode_state(game_state, self.model_options["include_danger"]); self._example_state = game_state
    legal_actions = safe_action_mask(game_state) if self.safe_actions else action_mask(game_state)
    current_epsilon = epsilon(self.steps, self.config.train.epsilon_start, self.config.train.epsilon_final, self.config.train.epsilon_decay_steps) if self.train else 0.0
    if self.train and self.rng.random() < current_epsilon: return ACTIONS[int(self.rng.choice(np.flatnonzero(legal_actions)))]
    with torch.no_grad():
        values = self.model(torch.as_tensor(observation[None], device=self.device)); values.masked_fill_(~torch.as_tensor(legal_actions, device=self.device)[None], -torch.inf)
        return ACTIONS[values.argmax(dim=1).item()]


def _record_transition(self, old_game_state, action, new_game_state, events, terminated):
    if old_game_state is None or action not in ACTION_TO_ID: return
    observation = encode_state(old_game_state, self.model_options["include_danger"]); next_observation = np.zeros_like(observation) if new_game_state is None else encode_state(new_game_state, self.model_options["include_danger"])
    next_mask = np.ones(len(ACTIONS), dtype=np.bool_) if new_game_state is None else (safe_action_mask(new_game_state) if self.safe_actions else action_mask(new_game_state))
    original_reward, training_reward = shaped_reward(events); self.replay.add(observation, ACTION_TO_ID[action], training_reward, next_observation, terminated, next_mask)
    self.steps += 1; self.episode_original_return += original_reward; self.episode_shaped_return += training_reward; self.episode_length += 1; self.episode_coins += events.count("COIN_COLLECTED"); self.episode_crates += events.count("CRATE_DESTROYED")
    if len(self.replay) >= max(int(self.config.train.learning_starts), int(self.config.train.batch_size)) and self.steps % int(self.config.train.train_frequency) == 0:
        beta = min(1.0, self.per_beta_start + (1.0 - self.per_beta_start) * self.steps / max(self.per_beta_steps, 1)); self.last_loss, indices, priorities = optimize(self.model, self.target, self.optimizer, self.replay, self.config.train.batch_size, self.config.train.gamma, self.device, self.scaler, self.double_dqn, beta, self.augment); self.replay.update_priorities(indices, priorities)
    if self.steps >= int(self.config.train.learning_starts) and self.steps % int(self.config.train.target_update_frequency) == 0: sync(self.model, self.target)
    self._last_transition = (old_game_state["round"], old_game_state["step"], action)


def setup_training(self): self.episode_original_return = self.episode_shaped_return = 0.0; self.episode_length = self.episode_coins = self.episode_crates = 0
def game_events_occurred(self, old_game_state, self_action, new_game_state, events): _record_transition(self, old_game_state, self_action, new_game_state, events, terminated=False)


def end_of_round(self, last_game_state, last_action, events):
    key = None if last_game_state is None else (last_game_state["round"], last_game_state["step"], last_action); death = "KILLED_SELF" in events or "GOT_KILLED" in events
    if key != self._last_transition: _record_transition(self, last_game_state, last_action, None if death else last_game_state, events, terminated=death)
    elif "SURVIVED_ROUND" in events and len(self.replay): self.replay.rewards[(self.replay.position - 1) % self.replay.capacity] += .20; self.episode_shaped_return += .20
    self.episode += 1; score = 0 if last_game_state is None else last_game_state["self"][1]
    append_csv(self.run_dir / "train.csv", TRAIN_FIELDS, {"step": self.steps, "episode": self.episode, "environment_return": f"{self.episode_original_return:.6f}", "shaped_return": f"{self.episode_shaped_return:.6f}", "length": self.episode_length, "epsilon": f"{epsilon(self.steps, self.config.train.epsilon_start, self.config.train.epsilon_final, self.config.train.epsilon_decay_steps):.6f}", "loss": "" if self.last_loss is None else f"{self.last_loss:.8f}", "win": int(score > 0 and not death), "death": int(death), "coins": self.episode_coins, "crates": self.episode_crates, "elapsed_seconds": f"{time.time() - self.started_at:.3f}"})
    bucket = self.steps // int(self.config.train.checkpoint_frequency)
    if bucket > self.saved_checkpoint_bucket: self.saved_checkpoint_bucket = bucket; _save(self)
    setup_training(self)


def finish_training(self): _save(self, final=True)
