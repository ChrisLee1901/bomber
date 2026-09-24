"""Shared, environment-independent pieces for Bomberman DQN experiments."""

import csv
import os
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ACTIONS = ("UP", "RIGHT", "DOWN", "LEFT", "WAIT", "BOMB")
ACTION_TO_ID = {action: index for index, action in enumerate(ACTIONS)}
BASE_OBSERVATION_SHAPE = (7, 17, 17)
OBSERVATION_SHAPE = BASE_OBSERVATION_SHAPE  # Backwards-compatible baseline alias.
OBSERVATION_SCALE = 4.0
_ROTATION_ACTIONS = ((0, 1, 2, 3, 4, 5), (1, 2, 3, 0, 4, 5), (2, 3, 0, 1, 4, 5), (3, 0, 1, 2, 4, 5))
_FLIP_ACTIONS = (2, 1, 0, 3, 4, 5)


def observation_shape(include_danger=False):
    return (8 if include_danger else 7, 17, 17)


def seed_everything(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(min(4, max(1, os.cpu_count() or 1)))
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic; torch.backends.cudnn.benchmark = not deterministic
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.set_float32_matmul_precision("high")


def device_for(requested="auto"):
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else "cpu" if requested == "auto" else requested)


def danger_map(game_state, power=3):
    """Earliest bomb timer affecting each tile; blast rays stop at walls and crates."""
    field, danger = game_state["field"], np.zeros(game_state["field"].shape, dtype=np.uint8)
    for (x, y), timer in game_state["bombs"]:
        deadline = min(int(timer) + 1, 255)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)):
            for distance in range(power + 1):
                nx, ny = x + dx * distance, y + dy * distance
                if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]) or field[nx, ny] == -1: break
                danger[nx, ny] = deadline if danger[nx, ny] == 0 else min(danger[nx, ny], deadline)
                if distance and field[nx, ny] == 1: break
    return danger


def encode_state(game_state, include_danger=False):
    if game_state is None: return None
    field, features = game_state["field"], np.zeros(observation_shape(include_danger), dtype=np.uint8)
    features[0] = field == -1; features[1] = field == 1
    for x, y in game_state["coins"]: features[2, x, y] = 1
    for (x, y), timer in game_state["bombs"]: features[3, x, y] = min(int(timer) + 1, 4)
    features[4] = np.clip(game_state["explosion_map"], 0, 3).astype(np.uint8)
    _, _, bombs_left, (x, y) = game_state["self"]; features[5, x, y] = 2 if bombs_left else 1
    for _, _, _, (x, y) in game_state["others"]: features[6, x, y] = 1
    if include_danger: features[7] = danger_map(game_state)
    return features


def action_mask(game_state):
    field = game_state["field"]; _, _, bombs_left, (x, y) = game_state["self"]
    occupied = {position for position, _ in game_state["bombs"]}; occupied.update(position for _, _, _, position in game_state["others"])
    def free(nx, ny): return field[nx, ny] == 0 and (nx, ny) not in occupied
    return np.asarray([free(x, y - 1), free(x + 1, y), free(x, y + 1), free(x - 1, y), True, bool(bombs_left)], dtype=np.bool_)


def _can_escape(field, danger, start, blocked):
    queue, visited, max_time = deque([(start, 0)]), {(start, 0)}, max(int(danger.max()), 1)
    while queue:
        (x, y), elapsed = queue.popleft(); deadline = int(danger[x, y])
        if deadline == 0: return True
        if elapsed >= deadline: continue
        for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
            nx, ny, next_elapsed = x + dx, y + dy, elapsed + 1
            if not (0 <= nx < field.shape[0] and 0 <= ny < field.shape[1]) or field[nx, ny] != 0 or (nx, ny) in blocked: continue
            key = ((nx, ny), next_elapsed)
            if next_elapsed <= max_time and key not in visited:
                visited.add(key); queue.append(key)
    return False


def safe_action_mask(game_state):
    legal, field = action_mask(game_state), game_state["field"]
    _, _, _, start = game_state["self"]
    blocked = {position for position, _ in game_state["bombs"]}; blocked.update(position for _, _, _, position in game_state["others"])
    destinations = ((start[0], start[1] - 1), (start[0] + 1, start[1]), (start[0], start[1] + 1), (start[0] - 1, start[1]), start, start)
    result = legal.copy()
    for action_id, destination in enumerate(destinations):
        if not legal[action_id]: continue
        state = dict(game_state); state["bombs"] = list(game_state["bombs"]) + ([(start, 4)] if action_id == ACTION_TO_ID["BOMB"] else [])
        result[action_id] = _can_escape(field, danger_map(state), destination, blocked - {destination})
    return result if result.any() else legal


def shaped_reward(events):
    original = events.count("COIN_COLLECTED") + 5.0 * events.count("KILLED_OPPONENT")
    return original, original + .20 * events.count("CRATE_DESTROYED") - .20 * events.count("INVALID_ACTION") - 2.0 * (events.count("KILLED_SELF") + events.count("GOT_KILLED")) + .20 * events.count("SURVIVED_ROUND")


class BoardDQN(nn.Module):
    def __init__(self, action_dim=len(ACTIONS), input_channels=7, dueling=False):
        super().__init__(); self.dueling = bool(dueling)
        self.features = nn.Sequential(nn.Conv2d(input_channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((5, 5)), nn.Flatten(), nn.Linear(64 * 5 * 5, 256), nn.ReLU())
        self.head = nn.Linear(256, action_dim) if not self.dueling else None
        if self.dueling: self.value, self.advantage = nn.Linear(256, 1), nn.Linear(256, action_dim)

    def forward(self, observations):
        features = self.features(observations.float() / OBSERVATION_SCALE)
        if not self.dueling: return self.head(features)
        advantage = self.advantage(features)
        return self.value(features) + advantage - advantage.mean(dim=1, keepdim=True)


def transform_id_action_map(transform_id):
    rotation, flipped = divmod(int(transform_id) % 8, 2); mapping = list(_ROTATION_ACTIONS[rotation])
    return tuple(_FLIP_ACTIONS[action] for action in mapping) if flipped else tuple(mapping)


def transform_observation(observations, transform_id):
    rotation, flipped = divmod(int(transform_id) % 8, 2); result = torch.rot90(observations, rotation, dims=(-2, -1))
    return torch.flip(result, dims=(-1,)) if flipped else result


def augment_batch(observations, actions, next_observations, masks, transform_id):
    mapping = torch.as_tensor(transform_id_action_map(transform_id), device=actions.device)
    remapped_masks = torch.zeros_like(masks); remapped_masks[:, mapping] = masks
    return transform_observation(observations, transform_id), mapping[actions], transform_observation(next_observations, transform_id), remapped_masks


class ReplayBuffer:
    def __init__(self, capacity, rng, observation_shape_=BASE_OBSERVATION_SHAPE, prioritized=False, alpha=.6):
        self.capacity, self.rng, self.prioritized, self.alpha, self.observation_shape = int(capacity), rng, bool(prioritized), float(alpha), tuple(observation_shape_)
        self.observations = np.empty((self.capacity, *self.observation_shape), dtype=np.uint8); self.next_observations = np.empty_like(self.observations)
        self.actions = np.empty(self.capacity, dtype=np.int64); self.rewards = np.empty(self.capacity, dtype=np.float32); self.terminated = np.empty(self.capacity, dtype=np.bool_)
        self.next_action_masks = np.empty((self.capacity, len(ACTIONS)), dtype=np.bool_); self.priorities = np.ones(self.capacity, dtype=np.float32)
        self.position = self.size = 0; self.max_priority = 1.0

    def __len__(self): return self.size

    def add(self, observation, action, reward, next_observation, terminated, next_action_mask=None):
        index = self.position; self.observations[index], self.next_observations[index] = observation, next_observation
        self.actions[index], self.rewards[index], self.terminated[index] = action, reward, terminated; self.next_action_masks[index] = True if next_action_mask is None else next_action_mask
        self.priorities[index] = self.max_priority; self.position = (index + 1) % self.capacity; self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=1.0):
        if self.prioritized:
            probabilities = self.priorities[:self.size] ** self.alpha; probabilities /= probabilities.sum(); indices = self.rng.choice(self.size, size=int(batch_size), p=probabilities)
            weights = (self.size * probabilities[indices]) ** -float(beta); weights /= weights.max()
        else: indices, weights = self.rng.integers(0, self.size, size=int(batch_size)), np.ones(int(batch_size), dtype=np.float32)
        return (self.observations[indices], self.actions[indices], self.rewards[indices], self.next_observations[indices], self.terminated[indices], self.next_action_masks[indices], indices, weights.astype(np.float32))

    def update_priorities(self, indices, priorities):
        if not self.prioritized: return
        values = np.asarray(priorities, dtype=np.float32) + 1e-6; self.priorities[np.asarray(indices)] = values; self.max_priority = max(self.max_priority, float(values.max()))

    def state_dict(self):
        arrays = {name: getattr(self, name).copy() for name in ("observations", "next_observations", "actions", "rewards", "terminated", "next_action_masks", "priorities")}
        return arrays | {"capacity": self.capacity, "position": self.position, "size": self.size, "max_priority": self.max_priority, "prioritized": self.prioritized, "alpha": self.alpha, "observation_shape": self.observation_shape}

    def load_state_dict(self, state):
        if tuple(state["observation_shape"]) != self.observation_shape or int(state["capacity"]) != self.capacity: raise ValueError("replay checkpoint does not match the configured observation shape/capacity")
        for name in ("observations", "next_observations", "actions", "rewards", "terminated", "next_action_masks", "priorities"): getattr(self, name)[:] = state[name]
        self.position, self.size, self.max_priority = int(state["position"]), int(state["size"]), float(state["max_priority"])


def epsilon(step, start, final, decay_steps):
    return float(final) if int(step) >= int(decay_steps) else float(start + min(max(int(step), 0) / int(decay_steps), 1.0) * (final - start))


def dqn_target(rewards, terminated, target_q, gamma, next_action_masks=None, online_q=None, double_dqn=False):
    if next_action_masks is not None:
        target_q = target_q.masked_fill(~next_action_masks.bool(), -torch.inf)
        if online_q is not None: online_q = online_q.masked_fill(~next_action_masks.bool(), -torch.inf)
    next_values = target_q.gather(1, online_q.argmax(dim=1, keepdim=True)).squeeze(1) if double_dqn else target_q.max(dim=1).values
    return rewards + float(gamma) * (~terminated).float() * next_values


def optimize(model, target, optimizer, replay, batch_size, gamma, device, scaler=None, double_dqn=False, beta=1.0, augment=False):
    observations, actions, rewards, next_observations, terminated, next_masks, indices, weights = replay.sample(batch_size, beta)
    observations, actions, rewards = torch.as_tensor(observations, device=device), torch.as_tensor(actions, device=device), torch.as_tensor(rewards, device=device)
    next_observations, terminated, next_masks, weights = torch.as_tensor(next_observations, device=device), torch.as_tensor(terminated, device=device), torch.as_tensor(next_masks, device=device), torch.as_tensor(weights, device=device)
    if augment: observations, actions, next_observations, next_masks = augment_batch(observations, actions, next_observations, next_masks, int(replay.rng.integers(8)))
    amp_enabled = scaler is not None and scaler.is_enabled()
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        q_values = model(observations).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad(): targets = dqn_target(rewards, terminated, target(next_observations), gamma, next_masks, model(next_observations) if double_dqn else None, double_dqn)
        td_errors = targets - q_values; loss = (F.smooth_l1_loss(q_values, targets, reduction="none") * weights).mean()
    optimizer.zero_grad(set_to_none=True)
    if amp_enabled:
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); scaler.step(optimizer); scaler.update()
    else: loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0); optimizer.step()
    return float(loss.detach().cpu()), indices, td_errors.detach().abs().cpu().numpy()


def sync(source, destination): destination.load_state_dict(source.state_dict())


def append_csv(path, fields, row):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if handle.tell() == 0: writer.writeheader()
        writer.writerow(row)


def save_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(".tmp"); torch.save(payload, temporary)
    for attempt in range(10):
        try:
            os.replace(temporary, path); return
        except PermissionError:
            if attempt == 9: raise
            time.sleep(.05)


def load_checkpoint(path, device): return torch.load(path, map_location=device, weights_only=False)
