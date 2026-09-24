# Bomberman DQN-to-Double-DQN Restore Guide

The current final-project implementation intentionally supports **DQN only**. Its production protocol is one gamma (`0.99`) and three training seeds (`42`, `43`, `44`). Do not mix those DQN-only outputs with a later DQN-vs-Double-DQN comparison.

To restore the comparison, retain the same board encoding, six actions, reward version, scripted opponents, evaluation maps/seeds, network, and all DQN hyperparameters. Add `double_dqn` as a second CLI/config algorithm, then change only the bootstrap target:

```text
DQN:        max_a Q_target(next_state, a)
Double DQN: Q_target(next_state, argmax_a Q_policy(next_state, a))
```

Use gamma values `0.95` and `0.99`, seeds `42`, `43`, and `44` for 12 production runs (2 algorithms × 2 gammas × 3 seeds). Keep 50 greedy final episodes with identical evaluation seeds and fixed rule-based opponents. Add a numerical unit test proving the two target equations differ, then report every seed plus mean and standard deviation without significance claims.
