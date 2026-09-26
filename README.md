# DQN Bomberman Agent

This repository trains a Bomberman DQN agent in the upstream engine. `config/dqn.yaml` remains the seven-channel, vanilla-DQN baseline; `config/dqn_enhanced.yaml` enables the stronger, resumable training pipeline.

## Performance design

- Baseline: seven compact board channels. Enhanced mode adds an eighth danger channel whose blast rays stop at walls and crates.
- Six actions: `UP`, `RIGHT`, `DOWN`, `LEFT`, `WAIT`, `BOMB`.
- Legal-action masks in baseline; enhanced mode masks provably unsafe actions and falls back to legal actions when necessary.
- Replay stays on CPU as `uint8`. Enhanced mode uses prioritized replay, importance sampling, and TD-error priority updates.
- Enhanced mode supports Double DQN, Dueling DQN, one random board symmetry per sampled batch, behavior-cloning warm starts, demos, curriculum, atomic checkpoints, and reproducible validation/test splits.

No configuration is claimed to be stronger without a measured multi-seed experiment.

## Setup

```powershell
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Training

```powershell
# Short functional check
uv run python -m unittest discover -s tests -v
uv run python -m scripts.smoke

# One resumable production run
uv run python main.py train --algo dqn --config config/dqn.yaml --seed 42 --gamma 0.99

# Enhanced curriculum run; use --demos and/or --pretrained after the two commands below
uv run python main.py train --config config/dqn_enhanced.yaml --seed 42 --demos data/demos --pretrained runs/pretrain.pt

# Three independent production seeds
uv run python main.py experiment --config config/experiment.yaml

# Aggregate runs and run the random-policy baseline
uv run python main.py aggregate --config config/experiment.yaml
```

Training is headless. Each run stores `resolved_config.yaml`, episode logs, periodic validation evaluations, `latest.pt`, `best.pt`, `final.pt`, and `final_evaluation.json` under `runs/dqn/gamma_0.99/seed_<seed>/`. `latest.pt` contains model, target, optimizer, AMP scaler, replay/PER state, RNG, current curriculum stage, best validation metric, and step. Resume requires matching observation shape and replay capacity; incompatible checkpoints fail explicitly.

Long-running commands show `tqdm` progress bars: games for demo collection/evaluation, agent steps for training, and epochs plus shards for pretraining.

## Demonstrations and pretraining

```powershell
# Use engine-controlled agents; shard fields are observation, action, reward,
# next_observation, terminated, next_action_mask, episode_id, opponent_set.
uv run python main.py collect-demos --config config/dqn_enhanced.yaml --episodes 500 --output-dir data/demos --agents hard_rule_based_agent rule_based_agent rule_based_agent rule_based_agent

# Continue an interrupted collection without overwriting completed shards.
uv run python main.py collect-demos --config config/dqn_enhanced.yaml --episodes 500 --output-dir data/demos --agents hard_rule_based_agent rule_based_agent rule_based_agent rule_based_agent --resume

# Behavior cloning uses an episode-level split (not random transitions) and early stopping.
uv run python main.py pretrain --config config/dqn_enhanced.yaml --demos data/demos --output runs/pretrain.pt --epochs 40 --patience 6

# Pretraining resumes from runs/pretrain.latest.pt by default.
```

`hard_rule_based_agent` targets `dqn_agent` first, avoids predicted blast paths, and only drops a bomb when its safe-action search finds an exit. The enhanced curriculum progresses through random/rule, all-rule, hard/rule, then mixed opponents.

## Evaluation

```powershell
uv run python main.py evaluate --checkpoint runs/dqn/gamma_0.99/seed_42/best.pt --episodes 50

# Held-out final test seeds and a JSON plus per-opponent-set CSV report
uv run python main.py evaluate --config config/dqn_enhanced.yaml --checkpoint runs/dqn/gamma_0.99/seed_42/best.pt --test --output results/evaluation.json
```

Enhanced periodic evaluation uses only `validation_seeds`; `--test` uses separate held-out `test_seeds`. Best checkpoints are selected lexicographically by validation win rate, mean return, then lower death rate. Evaluation writes one result per configured opponent set.

## Play against a trained model

Start a GUI match with a keyboard-controlled `user_agent`, one trained `dqn_agent`, and two rule-based opponents. Train a model first so the checkpoint exists.

```powershell
$env:BOMBER_RL_CONFIG = (Resolve-Path 'config/dqn.yaml')
$env:BOMBER_RL_RUN_DIR = "$PWD\runs\human-vs-model"
$env:BOMBER_RL_CHECKPOINT = (Resolve-Path 'runs\dqn\gamma_0.99\seed_42\best.pt')

uv run python main.py play --agents user_agent dqn_agent rule_based_agent rule_based_agent --n-rounds 1
```

Use the arrow keys to move, `Enter` to wait, and `Space` to place a bomb. Replace the checkpoint path to play against another trained run.

## Upstream attribution

The game engine comes from [anacarsi/ss24_ReinforcementLearning_BomberMan](https://github.com/anacarsi/ss24_ReinforcementLearning_BomberMan), commit `b133290`, under its MIT license.
