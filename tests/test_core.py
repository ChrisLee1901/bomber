import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from agents import SequentialAgentBackend
from bomber_rl.core import ACTIONS, BoardDQN, ReplayBuffer, action_mask, augment_batch, danger_map, dqn_target, encode_state, epsilon, load_checkpoint, safe_action_mask, save_checkpoint, transform_id_action_map, transform_observation
from bomber_rl.runner import _demo_progress, pretrain, write_demo_shard


def example_state():
    field = np.zeros((17, 17), dtype=np.int8)
    field[0, :] = field[-1, :] = field[:, 0] = field[:, -1] = -1
    field[3, 3] = 1
    return {
        "field": field, "coins": [(2, 2)], "bombs": [((4, 4), 3)], "explosion_map": np.zeros((17, 17)),
        "self": ("dqn_agent", 0, True, (1, 1)), "others": [("enemy", 0, True, (5, 5))],
    }


class CoreTests(unittest.TestCase):
    def test_action_mapping_and_state_encoding(self):
        observation = encode_state(example_state())
        self.assertEqual(len(ACTIONS), 6)
        self.assertEqual(observation.shape, (7, 17, 17))
        self.assertEqual(observation.dtype, np.uint8)
        self.assertEqual(observation[0, 0, 0], 1)
        self.assertEqual(observation[3, 4, 4], 4)
        mask = action_mask(example_state())
        self.assertTrue(mask[4])
        self.assertTrue(mask[5])
        self.assertFalse(mask[0])

    def test_schedule_and_dqn_target(self):
        self.assertEqual(epsilon(0, 1.0, 0.05, 100), 1.0)
        self.assertAlmostEqual(epsilon(50, 1.0, 0.05, 100), 0.525)
        self.assertEqual(epsilon(200, 1.0, 0.05, 100), 0.05)
        target = dqn_target(torch.tensor([1.0, 2.0]), torch.tensor([False, True]), torch.tensor([[3.0, 4.0], [8.0, 9.0]]), 0.5)
        self.assertTrue(torch.allclose(target, torch.tensor([3.0, 2.0])))

    def test_double_dqn_uses_online_choice_and_target_value(self):
        target = dqn_target(torch.tensor([1.0]), torch.tensor([False]), torch.tensor([[2.0, 9.0]]), .5, online_q=torch.tensor([[8.0, 1.0]]), double_dqn=True)
        self.assertTrue(torch.allclose(target, torch.tensor([2.0])))

    def test_dueling_and_all_symmetry_action_masks(self):
        self.assertEqual(BoardDQN(dueling=True)(torch.zeros(3, 7, 17, 17)).shape, (3, 6))
        observations = torch.arange(7 * 17 * 17).reshape(1, 7, 17, 17)
        actions, masks = torch.tensor([0]), torch.tensor([[True, False, False, False, False, False]])
        for transform in range(8):
            mapped = transform_id_action_map(transform)
            x, action, next_x, mask = augment_batch(observations, actions, observations, masks, transform)
            self.assertEqual(action.item(), mapped[0])
            self.assertTrue(mask[0, mapped[0]])
            self.assertEqual(mask.sum().item(), 1)
            self.assertTrue(torch.equal(x, transform_observation(observations, transform)))

    def test_danger_stops_at_crates_and_walls(self):
        state = example_state(); state["bombs"] = [((3, 3), 2), ((10, 10), 1)]
        state["field"][5, 3] = 1; state["field"][3, 5] = -1
        danger = danger_map(state)
        self.assertGreater(danger[5, 3], 0)
        self.assertEqual(danger[6, 3], 0)
        self.assertEqual(danger[3, 5], 0)
        self.assertGreater(danger[10, 10], 0)

    def test_safe_mask_falls_back_to_legal_actions(self):
        state = example_state(); state["self"] = ("dqn_agent", 0, False, (2, 2)); state["bombs"] = [((2, 3), 1)]
        state["field"][1, 2] = state["field"][3, 2] = state["field"][2, 1] = -1
        self.assertTrue(safe_action_mask(state).any())

    def test_replay_wraparound_and_checkpoint(self):
        rng = np.random.default_rng(7)
        replay = ReplayBuffer(2, rng)
        observation = encode_state(example_state())
        for action in range(3):
            replay.add(observation, action, float(action), observation, action == 2)
        self.assertEqual(len(replay), 2)
        batch = replay.sample(2)
        self.assertEqual(batch[0].dtype, np.uint8)
        self.assertEqual(batch[1].dtype, np.int64)
        model = BoardDQN()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, {"algorithm": "dqn", "model": model.state_dict()})
            restored = load_checkpoint(path, "cpu")
        self.assertEqual(restored["algorithm"], "dqn")
        self.assertEqual(set(restored["model"]), set(model.state_dict()))

    def test_per_checkpoint_and_demo_round_trip(self):
        observation = encode_state(example_state()); replay = ReplayBuffer(4, np.random.default_rng(3), prioritized=True)
        for index in range(4): replay.add(observation, index, index, observation, False)
        replay.update_priorities([0, 1], [100.0, .01])
        self.assertGreater(replay.priorities[0], replay.priorities[1])
        restored = ReplayBuffer(4, np.random.default_rng(4), prioritized=True); restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.position, replay.position); self.assertTrue(np.array_equal(restored.observations, replay.observations))
        transition = {"observation": observation, "action": np.int64(1), "reward": np.float32(.5), "next_observation": observation, "terminated": np.bool_(False), "next_action_mask": np.ones(6, dtype=np.bool_), "episode_id": np.int64(7), "opponent_set": np.str_("rule")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.npz"; write_demo_shard(path, [transition])
            with np.load(path) as shard:
                self.assertEqual(shard["observation"].dtype, np.uint8); self.assertEqual(shard["observation"].shape, (1, 7, 17, 17)); self.assertEqual(shard["episode_id"][0], 7)

    def test_demo_resume_uses_next_episode_and_shard(self):
        observation = encode_state(example_state())
        def transition(episode_id): return {"observation": observation, "action": np.int64(1), "reward": np.float32(.5), "next_observation": observation, "terminated": np.bool_(False), "next_action_mask": np.ones(6, dtype=np.bool_), "episode_id": np.int64(episode_id), "opponent_set": np.str_("rule")}
        with tempfile.TemporaryDirectory() as directory:
            write_demo_shard(Path(directory) / "demo_00000.npz", [transition(0)])
            write_demo_shard(Path(directory) / "demo_00002.npz", [transition(5)])
            self.assertEqual(_demo_progress(Path(directory)), (6, 3))

    def test_agent_backend_closes_its_log_handler(self):
        backend = SequentialAgentBackend(False, "close_test", "rule_based_agent")
        backend.start(); handler = backend.runner.handler; backend.close()
        self.assertIsNone(handler.stream)

    def test_pretrain_resume_restores_epoch(self):
        config = OmegaConf.create({"state": {"include_danger": False}, "algorithm": {"dueling": False}, "device": "cpu", "train": {"learning_rate": .0001}})
        observation = encode_state(example_state()); transition = {"observation": observation, "action": np.int64(1), "reward": np.float32(.5), "next_observation": observation, "terminated": np.bool_(False), "next_action_mask": np.ones(6, dtype=np.bool_), "episode_id": np.int64(0), "opponent_set": np.str_("rule")}
        second = dict(transition, episode_id=np.int64(1))
        with tempfile.TemporaryDirectory() as directory:
            demos, output = Path(directory) / "demo.npz", Path(directory) / "pretrain.pt"; write_demo_shard(demos, [transition, second])
            pretrain(config, demos, output, epochs=1, batch_size=2, patience=3, resume=False)
            pretrain(config, demos, output, epochs=2, batch_size=2, patience=3, resume=True)
            self.assertEqual(load_checkpoint(output.with_name("pretrain.latest.pt"), "cpu")["pretrain_epoch"], 2)


if __name__ == "__main__":
    unittest.main()
