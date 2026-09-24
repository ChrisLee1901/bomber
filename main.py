import os
from argparse import ArgumentParser
from pathlib import Path
from time import sleep, time
from tqdm import tqdm

import settings as s
from environment import BombeRLeWorld, GUI
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld
from bomber_rl.runner import aggregate, collect_demos, evaluate, load_config, pretrain, run_experiment, train

ESCAPE_KEYS = (pygame.K_q, pygame.K_ESCAPE)


class Timekeeper:
    def __init__(self, interval):
        self.interval = interval
        self.next_time = None

    def is_due(self):
        return self.next_time is None or time() >= self.next_time

    def note(self):
        self.next_time = time() + self.interval

    def wait(self):
        if not self.is_due():
            duration = self.next_time - time()
            sleep(duration)


def world_controller(world, n_rounds, *,
                     gui, every_step, turn_based, make_video, update_interval):
    if make_video and not gui.screenshot_dir.exists():
        gui.screenshot_dir.mkdir()

    gui_timekeeper = Timekeeper(update_interval)

    def render(wait_until_due):
        # If every step should be displayed, wait until it is due to be shown
        if wait_until_due:
            gui_timekeeper.wait()

        if gui_timekeeper.is_due():
            gui_timekeeper.note()
            # Render (which takes time)
            gui.render()
            pygame.display.flip()

    user_input = None
    for _ in tqdm(range(n_rounds)):
        world.new_round()
        while world.running:
            # Only render when the last frame is not too old
            if gui is not None:
                render(every_step)

                # Check GUI events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in ESCAPE_KEYS:
                            world.end_round()
                        elif key_pressed in s.INPUT_MAP:
                            user_input = s.INPUT_MAP[key_pressed]

            # Advances step (for turn based: only if user input is available)
            if world.running and not (turn_based and user_input is None):
                world.do_step(user_input)
                user_input = None
            else:
                # Might want to wait
                pass

        # Save video of last game
        if make_video:
            gui.make_video()

        # Render end screen until next round is queried
        if gui is not None:
            do_continue = False
            while not do_continue:
                render(True)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                    elif event.type == pygame.KEYDOWN:
                        key_pressed = event.key
                        if key_pressed in s.INPUT_MAP or key_pressed in ESCAPE_KEYS:
                            do_continue = True

    world.end()


def main(argv = None):
    parser = ArgumentParser()

    subparsers = parser.add_subparsers(dest='command_name', required=True)

    # Run arguments
    play_parser = subparsers.add_parser("play")
    agent_group = play_parser.add_mutually_exclusive_group()
    agent_group.add_argument("--my-agent", type=str, help="Play agent of name ... against three rule_based_agents")
    agent_group.add_argument("--agents", type=str, nargs="+", default=["rule_based_agent"] * s.MAX_AGENTS, help="Explicitly set the agent names in the game")
    play_parser.add_argument("--train", default=0, type=int, choices=[0, 1, 2, 3, 4],
                             help="First … agents should be set to training mode")
    play_parser.add_argument("--continue-without-training", default=False, action="store_true")
    # play_parser.add_argument("--single-process", default=False, action="store_true")

    play_parser.add_argument("--scenario", default="classic", choices=s.SCENARIOS)

    play_parser.add_argument("--seed", type=int, help="Reset the world's random number generator to a known number for reproducibility")

    play_parser.add_argument("--n-rounds", type=int, default=10, help="How many rounds to play")
    play_parser.add_argument("--save-replay", const=True, default=False, action='store', nargs='?', help='Store the game as .pt for a replay')
    play_parser.add_argument("--match-name", help="Give the match a name")

    play_parser.add_argument("--silence-errors", default=False, action="store_true", help="Ignore errors from agents")

    group = play_parser.add_mutually_exclusive_group()
    group.add_argument("--skip-frames", default=False, action="store_true", help="Play several steps per GUI render.")
    group.add_argument("--no-gui", default=False, action="store_true", help="Deactivate the user interface and play as fast as possible.")

    train_parser = subparsers.add_parser("train", help="Train the DQN Bomberman agent headlessly")
    train_parser.add_argument("--algo", choices=["dqn"], default="dqn")
    train_parser.add_argument("--config", default="config/dqn.yaml")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--gamma", type=float)
    train_parser.add_argument("--steps", type=int)
    train_parser.add_argument("--run-dir")
    train_parser.add_argument("--no-resume", action="store_true")
    train_parser.add_argument("--pretrained", help="Behavior-cloned checkpoint to initialize online RL")
    train_parser.add_argument("--demos", help="NPZ shard directory to preload into replay")

    evaluate_parser = subparsers.add_parser("evaluate", help="Greedily evaluate a DQN checkpoint")
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--config", default="config/dqn.yaml")
    evaluate_parser.add_argument("--episodes", type=int)
    evaluate_parser.add_argument("--output")
    evaluate_parser.add_argument("--test", action="store_true", help="Use held-out test seeds (not validation seeds)")
    evaluate_parser.add_argument("--opponents", nargs=3, help="Evaluate one explicit three-opponent set")

    demos_parser = subparsers.add_parser("collect-demos", help="Collect expert transitions from engine agents")
    demos_parser.add_argument("--config", default="config/dqn_enhanced.yaml")
    demos_parser.add_argument("--episodes", type=int, default=100)
    demos_parser.add_argument("--output-dir", default="data/demos")
    demos_parser.add_argument("--agents", nargs=4, default=["rule_based_agent"] * 4)
    demos_parser.add_argument("--shard-size", type=int, default=50000)

    pretrain_parser = subparsers.add_parser("pretrain", help="Behavior-clone DQN from NPZ demonstration shards")
    pretrain_parser.add_argument("--config", default="config/dqn_enhanced.yaml")
    pretrain_parser.add_argument("--demos", required=True)
    pretrain_parser.add_argument("--output", default="runs/pretrain.pt")
    pretrain_parser.add_argument("--epochs", type=int, default=20)
    pretrain_parser.add_argument("--batch-size", type=int, default=128)
    pretrain_parser.add_argument("--patience", type=int, default=4)

    experiment_parser = subparsers.add_parser("experiment", help="Run the DQN seed and gamma matrix")
    experiment_parser.add_argument("--config", default="config/experiment.yaml")
    aggregate_parser = subparsers.add_parser("aggregate", help="Create plots and summaries from completed DQN runs")
    aggregate_parser.add_argument("--config", default="config/experiment.yaml")

    # Replay arguments
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("replay", help="File to load replay from")

    # Interaction
    for sub in [play_parser, replay_parser]:
        sub.add_argument("--turn-based", default=False, action="store_true",
                         help="Wait for key press until next movement")
        sub.add_argument("--update-interval", type=float, default=0.1,
                         help="How often agents take steps (ignored without GUI)")
        sub.add_argument("--log-dir", default=os.path.dirname(os.path.abspath(__file__)) + "/logs")
        sub.add_argument("--save-stats", const=True, default=False, action='store', nargs='?', help='Store the game results as .json for evaluation')

        # Video?
        sub.add_argument("--make-video", const=True, default=False, action='store', nargs='?',
                         help="Make a video from the game")

    args = parser.parse_args(argv)
    if args.command_name == "train":
        config = load_config(args.config, gamma=args.gamma, total_steps=args.steps)
        print(train(config, args.seed, run_dir=args.run_dir, resume=not args.no_resume, pretrained=args.pretrained, demos=args.demos))
        return
    if args.command_name == "evaluate":
        os.environ["BOMBER_RL_CONFIG"] = str(Path(args.config).resolve())
        print(evaluate(load_config(args.config), args.checkpoint, args.episodes, args.output, args.test, args.opponents))
        return
    if args.command_name == "collect-demos":
        print([str(path) for path in collect_demos(load_config(args.config), args.episodes, args.output_dir, args.agents, args.shard_size)])
        return
    if args.command_name == "pretrain":
        print(pretrain(load_config(args.config), args.demos, args.output, args.epochs, args.batch_size, args.patience))
        return
    if args.command_name == "experiment":
        run_experiment(args.config)
        return
    if args.command_name == "aggregate":
        aggregate(args.config)
        return
    if args.command_name == "replay":
        args.no_gui = False
        args.n_rounds = 1
        args.match_name = Path(args.replay).name

    has_gui = not args.no_gui
    if has_gui:
        if not LOADED_PYGAME:
            raise ValueError("pygame could not loaded, cannot run with GUI")

    # Initialize environment and agents
    if args.command_name == "play":
        agents = []
        if args.train == 0 and not args.continue_without_training:
            args.continue_without_training = True
        if args.my_agent:
            agents.append((args.my_agent, len(agents) < args.train))
            args.agents = ["rule_based_agent"] * (s.MAX_AGENTS - 1)
        for agent_name in args.agents:
            agents.append((agent_name, len(agents) < args.train))

        world = BombeRLeWorld(args, agents)
        every_step = not args.skip_frames
    elif args.command_name == "replay":
        world = ReplayWorld(args)
        every_step = True
    else:
        raise ValueError(f"Unknown command {args.command_name}")

    # Launch GUI
    if has_gui:
        gui = GUI(world)
    else:
        gui = None
    world_controller(world, args.n_rounds,
                     gui=gui, every_step=every_step, turn_based=args.turn_based,
                     make_video=args.make_video, update_interval=args.update_interval)


if __name__ == '__main__':
    main()
