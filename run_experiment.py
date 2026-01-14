"""
Main experiment runner for Tier 1 CoT Faithfulness study.

Runs LLM vs LLM Coup games and collects bluff logs.
"""

import argparse
import os
from coup_game import CoupGame
from llm_agent import LLMAgent
from thought_wrapper import ThoughtProbeWrapper


def run_tier1_experiment(
    num_games: int = 50,
    model_name: str = "Qwen/Qwen2.5-72B-Instruct",
    temperature: float = 0.7,
    output_dir: str = "logs",
    verbose: bool = False
):
    """
    Run Tier 1 experiment.

    Args:
        num_games: Number of games to play
        model_name: Ollama model name
        temperature: LLM temperature
        output_dir: Directory to save logs
        verbose: Print detailed game info
    """
    print(f"=== Tier 1 CoT Faithfulness Experiment ===")
    print(f"Model: {model_name}")
    print(f"Games: {num_games}")
    print(f"Temperature: {temperature}")
    print(f"Output: {output_dir}/tier1_logs.jsonl\n")

    # Initialize components
    game = CoupGame()
    llm = LLMAgent(model_name=model_name, temperature=temperature)
    wrapper = ThoughtProbeWrapper(game, llm)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Run games
    for game_idx in range(num_games):
        print(f"[Game {game_idx + 1}/{num_games}] Starting...")

        # Reset game
        game.reset()
        wrapper.reset_game()

        done = False
        turn_count = 0
        max_turns = 50  # Prevent infinite games

        while not done and turn_count < max_turns:
            current_player = game.current_player

            try:
                obs, done, info = wrapper.play_turn(current_player)

                if verbose:
                    print(f"  Turn {turn_count}: Player {current_player}")
                    print(f"    Thought: {info.get('thought', 'N/A')[:80]}...")
                    print(f"    Action: {info.get('parsed_action', 'N/A')}")

                if not info.get("valid", True):
                    print(f"  ERROR: {info.get('message')}")
                    print(f"  Retrying with fallback action...")
                    # Fallback: take Income
                    obs, done, info = game.step("Income")

                turn_count += 1

            except Exception as e:
                print(f"  ERROR in turn {turn_count}: {e}")
                print(f"  Skipping to next game...")
                break

        # Game finished
        winner = game.get_winner()
        if winner is not None:
            print(f"[Game {game_idx + 1}/{num_games}] Winner: Player {winner} ({turn_count} turns)")
        else:
            print(f"[Game {game_idx + 1}/{num_games}] Max turns reached, no winner")

    # Save logs
    output_path = os.path.join(output_dir, "tier1_logs.jsonl")
    wrapper.save_logs(output_path)

    # Print summary
    print("\n=== Experiment Complete ===")
    print(f"Total games played: {num_games}")
    print(f"Total bluff instances logged: {len(wrapper.turn_logs)}")

    # Analyze bluff types
    action_bluffs = sum(1 for log in wrapper.turn_logs if log["action_type"] == "ACTION_BLUFF")
    block_bluffs = sum(1 for log in wrapper.turn_logs if log["action_type"] == "BLOCK_BLUFF")

    print(f"  - Action bluffs: {action_bluffs}")
    print(f"  - Block bluffs: {block_bluffs}")
    print(f"\nNext step: Manually label bluffs in {output_path}")
    print("Label each entry as 'faithful_lie' or 'confabulation'")


def run_smoke_test():
    """Run a quick 5-game smoke test."""
    print("Running smoke test (5 games)...\n")
    run_tier1_experiment(
        num_games=5,
        model_name="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.7,
        output_dir="logs",
        verbose=True
    )


def main():
    parser = argparse.ArgumentParser(description="Run Tier 1 CoT Faithfulness Experiment")
    parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of games to play (default: 50)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-72B-Instruct",
        help="HuggingFace model name (default: Qwen/Qwen2.5-72B-Instruct)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs",
        help="Output directory (default: logs)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed game information"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run quick 5-game smoke test"
    )

    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
    else:
        run_tier1_experiment(
            num_games=args.games,
            model_name=args.model,
            temperature=args.temperature,
            output_dir=args.output,
            verbose=args.verbose
        )


if __name__ == "__main__":
    main()
