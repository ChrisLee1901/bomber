from pathlib import Path
import shutil

from bomber_rl.runner import load_config, train


def main():
    root = Path(".smoke")
    if root.exists():
        shutil.rmtree(root)
    config = load_config("config/smoke.yaml")
    train(config, seed=42, run_dir=root / "dqn", resume=False)
    required = ("train.csv", "latest.pt", "final.pt", "final_evaluation.json")
    missing = [name for name in required if not (root / "dqn" / name).exists()]
    assert not missing, f"missing smoke artifacts: {missing}"
    print("DQN smoke training completed.")


if __name__ == "__main__":
    main()
