#!/usr/bin/env python3
"""
CLI for running Coup Bench benchmarks.

Examples:
    # Quick test - 2 models, 5 games each
    python run_bench.py --quick

    # Standard benchmark - default models, 10 games
    python run_bench.py --standard

    # Custom models
    python run_bench.py --models "Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-14B-Instruct"

    # Chat disabled (faster)
    python run_bench.py --no-chat

    # Verbose output
    python run_bench.py --verbose
"""

import argparse
from pathlib import Path

from coup_bench import CoupBench, BenchmarkConfig
from chat_system import ChatConfig


def main():
    parser = argparse.ArgumentParser(
        description="Coup Bench - Benchmark LLMs on strategic Coup gameplay",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Preset modes
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test: 2 models, 3 games each, minimal chat"
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Standard benchmark: default models, 10 games each"
    )
    parser.add_argument(
        "--extensive",
        action="store_true",
        help="Extensive benchmark: all models, 20 games each"
    )

    # Custom configuration
    parser.add_argument(
        "--models",
        nargs="+",
        help="List of HuggingFace model names to benchmark"
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Games per matchup (default: 10)"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Benchmark name (default: auto-generated)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/bench",
        help="Output directory (default: logs/bench)"
    )

    # Chat configuration
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Disable chat (faster)"
    )
    parser.add_argument(
        "--chat-mode",
        choices=["minimal", "default", "verbose"],
        default="default",
        help="Chat verbosity (default: default)"
    )

    # Model configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for model inference (default: cuda)"
    )

    # Output options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during benchmark"
    )

    args = parser.parse_args()

    # Determine configuration
    if args.quick:
        models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct"
        ]
        games = 3
        chat_config = ChatConfig.minimal()
        name = "quick_test"
    elif args.standard:
        models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct"
        ]
        games = 10
        chat_config = ChatConfig.default()
        name = "standard_bench"
    elif args.extensive:
        models = [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-3B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3"
        ]
        games = 20
        chat_config = ChatConfig.default()
        name = "extensive_bench"
    else:
        # Custom configuration
        if args.models:
            models = args.models
        else:
            # Default models
            models = [
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct"
            ]

        games = args.games

        if args.no_chat:
            chat_config = ChatConfig.disabled()
        else:
            chat_modes = {
                "minimal": ChatConfig.minimal(),
                "default": ChatConfig.default(),
                "verbose": ChatConfig.verbose()
            }
            chat_config = chat_modes[args.chat_mode]

        name = args.name or "custom_bench"

    # Validate at least 2 models
    if len(models) < 2:
        print("Error: Need at least 2 models for a benchmark")
        return

    # Create config
    config = BenchmarkConfig(
        name=name,
        models=models,
        games_per_matchup=games,
        num_players=2,
        chat_enabled=not args.no_chat,
        chat_config=chat_config,
        output_dir=args.output,
        temperature=args.temperature,
        device=args.device
    )

    # Run benchmark
    bench = CoupBench(config)
    bench.run_tournament(verbose=args.verbose)


if __name__ == "__main__":
    main()
