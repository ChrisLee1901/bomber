"""A conservative opponent: hunts the learner, but never bombs without an exit."""

from collections import deque

import numpy as np

from agent_code.rule_based_agent.callbacks import look_for_targets
from bomber_rl.core import ACTIONS, ACTION_TO_ID, safe_action_mask


def setup(self):
    self.bomb_history = deque([], 5)
    self.current_round = 0


def act(self, game_state):
    if game_state["round"] != self.current_round:
        self.bomb_history.clear(); self.current_round = game_state["round"]
    _, _, bombs_left, start = game_state["self"]
    safe = safe_action_mask(game_state)
    targets = [position for name, _, _, position in game_state["others"] if name.startswith("dqn_agent")]
    if not targets:
        targets = [position for _, _, _, position in game_state["others"]]
    field = game_state["field"]
    step = look_for_targets(field == 0, start, targets, self.logger)
    moves = {(start[0], start[1] - 1): "UP", (start[0] + 1, start[1]): "RIGHT", (start[0], start[1] + 1): "DOWN", (start[0] - 1, start[1]): "LEFT"}
    if step in moves and safe[ACTION_TO_ID[moves[step]]]:
        return moves[step]
    close = any(abs(x - start[0]) + abs(y - start[1]) <= 1 for x, y in targets)
    if close and bombs_left and start not in self.bomb_history and safe[ACTION_TO_ID["BOMB"]]:
        self.bomb_history.append(start); return "BOMB"
    for action in ACTIONS:
        if safe[ACTION_TO_ID[action]]:
            return action
    return "WAIT"
