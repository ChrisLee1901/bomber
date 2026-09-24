# Training a DQN Agent for Discrete-Action Bomberman

**Status:** implementation complete; production results pending.

## Research question

Can a DQN policy outperform the upstream random agent against three fixed rule-based Bomberman opponents, and how stable are return and win rate across three independently seeded runs?

## Method

The agent sees seven `17×17` channels: walls, crates, coins, bomb timers, explosions, its own position/bomb availability, and opponents. It selects one of six discrete actions. Invalid movement and unavailable bombs are action-masked. Replay is stored as `uint8` on CPU; GPU optimization uses mixed precision when CUDA is available.

The DQN target is `r + gamma * (1 - terminated) * max_a Q_target(s', a)`. The reward version, seeds, hyperparameters, opponent policy, and evaluation maps are captured in each resolved configuration.

## Protocol

Train DQN with gamma 0.99 on seeds 42, 43, and 44. Evaluate `best.pt` greedily for 50 fixed episodes (seeds 1000–1049) against the same rule-based opponents. Compare every seed with the random-policy baseline; report means and standard deviations without a significance claim.

## Results

Pending generated files: `results/summary.csv`, `learning_curves.png`, `final_returns.png`, `win_rates.png`, and `random_baseline.json`.

## Limitations

The policy can overfit to fixed opponents and the reward function. Board features do not eliminate strategic partial observability, and three seeds quantify variability but are not a significance test.
