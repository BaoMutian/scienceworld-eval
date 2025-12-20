#!/usr/bin/env python3
"""Main entry point for ScienceWorld evaluation."""

import argparse
import sys

from src.config import load_config, Config
from src.evaluator import run_evaluation
from src.logging_utils import setup_logging


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ScienceWorld LLM Agent Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python run_eval.py
  
  # Run with custom config
  python run_eval.py --config config/my_config.yaml
  
  # Override specific settings
  python run_eval.py --model qwen/qwen3-8b --split dev
  
  # Run specific tasks
  python run_eval.py --task-ids 1-1 4-1 4-2 --num-episodes 3
  
  # Debug mode
  python run_eval.py --debug --task-ids 4-1 --num-episodes 1

  # Memory modes
  python run_eval.py --memory-mode retrieve_and_extract
  python run_eval.py --memory-mode retrieve_only
  python run_eval.py --memory-mode baseline
""",
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/default.yaml)",
    )

    # LLM settings
    parser.add_argument("--model", "-m", type=str, help="Model name")
    parser.add_argument("--api-base", type=str, help="API base URL")
    parser.add_argument("--api-key", type=str, help="API key")
    parser.add_argument("--temperature", type=float, help="Sampling temperature")

    # Test settings
    parser.add_argument("--num-episodes", "-n", type=int, help="Variations to test per task")
    parser.add_argument(
        "--split", "-s",
        type=str,
        choices=["train", "dev", "test"],
        help="Dataset split",
    )
    parser.add_argument("--task-ids", "-t", type=str, nargs="+", help="Task IDs (e.g., 1-1 4-1)")
    parser.add_argument("--max-steps", type=int, help="Max steps per episode")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--simplifications",
        type=str,
        help="Simplifications preset or comma-separated list",
    )

    # Prompt settings
    parser.add_argument("--no-few-shot", action="store_true", help="Disable few-shot examples")
    parser.add_argument("--history-length", type=int, help="History entries to include")

    # Memory settings
    parser.add_argument(
        "--memory-mode",
        type=str,
        choices=["baseline", "retrieve_only", "retrieve_and_extract"],
        help="Memory mode",
    )
    parser.add_argument("--memory-dir", type=str, help="Memory bank directory")

    # Runtime settings
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")

    return parser.parse_args()


def apply_overrides(config: Config, args) -> Config:
    """Apply command line overrides to config."""
    # LLM overrides
    if args.model:
        config.llm.model = args.model
    if args.api_base:
        config.llm.api_base_url = args.api_base
    if args.api_key:
        config.llm.api_key = args.api_key
    if args.temperature is not None:
        config.llm.temperature = args.temperature

    # Test overrides
    if args.num_episodes is not None:
        config.test.num_episodes = args.num_episodes
    if args.split:
        config.test.split = args.split
    if args.task_ids:
        config.test.task_ids = args.task_ids
    if args.max_steps is not None:
        config.test.max_steps = args.max_steps
    if args.seed is not None:
        config.test.seed = args.seed
    if args.simplifications:
        config.test.simplifications = args.simplifications

    # Prompt overrides
    if args.no_few_shot:
        config.prompt.use_few_shot = False
    if args.history_length is not None:
        config.prompt.history_length = args.history_length

    # Memory overrides
    if args.memory_mode:
        config.memory.enabled = args.memory_mode != "baseline"
        config.memory.mode = args.memory_mode
    if args.memory_dir:
        config.memory.memory_dir = args.memory_dir

    # Runtime overrides
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.debug:
        config.runtime.debug = True

    return config


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = Config()

    config = apply_overrides(config, args)

    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)

    # Setup basic logging
    setup_logging(debug=config.runtime.debug)

    try:
        run_evaluation(config)
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")
        print("Progress has been saved to checkpoint.")
        sys.exit(130)
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        if config.runtime.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

