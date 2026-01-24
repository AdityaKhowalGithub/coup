"""
Coup Bench - Benchmarking system for LLM Coup performance.

Features:
- Round-robin tournaments
- ELO rating system
- Multi-dimensional metrics
- Leaderboards
- Model comparison
"""

import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict
import itertools

from coup_game import CoupGame
from llm_agent import LLMAgent
from chat_wrapper import ChatEnabledWrapper
from chat_system import ChatConfig


@dataclass
class PlayerStats:
    """Statistics for a single player/model."""
    model_name: str
    games_played: int = 0
    games_won: int = 0
    total_turns: int = 0
    total_bluffs: int = 0
    successful_bluffs: int = 0
    failed_bluffs: int = 0
    challenges_made: int = 0
    successful_challenges: int = 0
    failed_challenges: int = 0
    elo_rating: float = 1500.0

    def win_rate(self) -> float:
        """Calculate win rate."""
        return self.games_won / self.games_played if self.games_played > 0 else 0.0

    def bluff_success_rate(self) -> float:
        """Calculate bluff success rate."""
        total = self.successful_bluffs + self.failed_bluffs
        return self.successful_bluffs / total if total > 0 else 0.0

    def challenge_success_rate(self) -> float:
        """Calculate challenge success rate."""
        total = self.successful_challenges + self.failed_challenges
        return self.successful_challenges / total if total > 0 else 0.0

    def avg_turns_per_game(self) -> float:
        """Average turns survived per game."""
        return self.total_turns / self.games_played if self.games_played > 0 else 0.0


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""
    name: str
    models: List[str]
    games_per_matchup: int = 10
    num_players: int = 2
    chat_enabled: bool = True
    chat_config: Optional[ChatConfig] = None
    output_dir: str = "logs/bench"
    temperature: float = 0.7
    device: str = "cuda"

    def __post_init__(self):
        """Set defaults."""
        if self.chat_config is None:
            self.chat_config = ChatConfig.default() if self.chat_enabled else ChatConfig.disabled()


@dataclass
class MatchResult:
    """Result of a single match."""
    match_id: int
    model_a: str
    model_b: str
    winner: str
    loser: str
    turns: int
    duration_seconds: float
    bluffs_a: int = 0
    bluffs_b: int = 0


class ELOSystem:
    """Simple ELO rating system for Coup."""

    def __init__(self, k_factor: int = 32):
        """
        Initialize ELO system.

        Args:
            k_factor: ELO K-factor (sensitivity to individual games)
        """
        self.k_factor = k_factor

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for player A."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(
        self,
        rating_a: float,
        rating_b: float,
        score_a: float
    ) -> Tuple[float, float]:
        """
        Update ratings after a match.

        Args:
            rating_a: Current rating of player A
            rating_b: Current rating of player B
            score_a: Actual score for A (1.0 = win, 0.5 = draw, 0.0 = loss)

        Returns:
            (new_rating_a, new_rating_b)
        """
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        score_b = 1 - score_a

        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * (score_b - expected_b)

        return new_rating_a, new_rating_b


class CoupBench:
    """
    Benchmark system for evaluating LLM Coup performance.

    Runs tournaments, tracks metrics, generates leaderboards.
    """

    def __init__(self, config: BenchmarkConfig):
        """
        Initialize benchmark system.

        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.stats: Dict[str, PlayerStats] = {}
        self.match_history: List[MatchResult] = []
        self.elo_system = ELOSystem()

        # Initialize stats for each model
        for model in config.models:
            self.stats[model] = PlayerStats(model_name=model)

        # Create output directory
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def run_tournament(self, verbose: bool = False):
        """
        Run a round-robin tournament.

        Each model plays against every other model for N games.

        Args:
            verbose: Print detailed progress
        """
        print(f"\n{'='*60}")
        print(f"COUP BENCH: {self.config.name}")
        print(f"{'='*60}")
        print(f"Models: {', '.join(self.config.models)}")
        print(f"Games per matchup: {self.config.games_per_matchup}")
        print(f"Chat enabled: {self.config.chat_enabled}")
        print(f"{'='*60}\n")

        # Generate all matchups (round-robin)
        matchups = list(itertools.combinations(self.config.models, 2))
        total_games = len(matchups) * self.config.games_per_matchup

        print(f"Total matchups: {len(matchups)}")
        print(f"Total games: {total_games}\n")

        match_id = 0
        for model_a, model_b in matchups:
            print(f"\n--- {model_a} vs {model_b} ---")

            for game_num in range(self.config.games_per_matchup):
                match_id += 1
                if verbose:
                    print(f"  Game {game_num + 1}/{self.config.games_per_matchup}...", end=" ")

                result = self._play_match(match_id, model_a, model_b)
                self._update_stats(result)
                self.match_history.append(result)

                if verbose:
                    print(f"Winner: {result.winner} ({result.turns} turns)")

            # Print interim results
            self._print_matchup_summary(model_a, model_b)

        print(f"\n{'='*60}")
        print("TOURNAMENT COMPLETE")
        print(f"{'='*60}\n")

        self._print_final_results()
        self._save_results()

    def _play_match(
        self,
        match_id: int,
        model_a_name: str,
        model_b_name: str
    ) -> MatchResult:
        """
        Play a single match between two models.

        Args:
            match_id: Unique match identifier
            model_a_name: Name of model A
            model_b_name: Name of model B

        Returns:
            MatchResult with game outcome
        """
        start_time = time.time()

        # Initialize game
        game = CoupGame(num_players=2)
        game.reset()

        # Initialize LLM agents
        # Note: In production, you'd want to cache loaded models
        # For now, we'll use a simplified approach
        llm_a = LLMAgent(
            model_name=model_a_name,
            temperature=self.config.temperature,
            device=self.config.device
        )
        llm_b = LLMAgent(
            model_name=model_b_name,
            temperature=self.config.temperature,
            device=self.config.device
        )

        # Create wrappers
        wrapper_a = ChatEnabledWrapper(game, llm_a, self.config.chat_config)
        wrapper_b = ChatEnabledWrapper(game, llm_b, self.config.chat_config)

        wrappers = {0: wrapper_a, 1: wrapper_b}
        model_names = {0: model_a_name, 1: model_b_name}

        # Play game
        done = False
        turns = 0
        max_turns = 100  # Prevent infinite games

        while not done and turns < max_turns:
            current_player = game.current_player
            wrapper = wrappers[current_player]

            obs, done, info = wrapper.play_turn(current_player)
            turns += 1

            if not info["valid"]:
                # Invalid action - skip turn
                continue

        # Determine winner
        winner_id = game.get_winner()
        if winner_id is not None:
            winner_name = model_names[winner_id]
            loser_name = model_names[1 - winner_id]
        else:
            # Draw or max turns reached
            winner_name = model_a_name  # Arbitrary
            loser_name = model_b_name

        duration = time.time() - start_time

        # Count bluffs
        bluffs_a = len([log for log in wrapper_a.get_bluff_logs() if log["is_bluff"]])
        bluffs_b = len([log for log in wrapper_b.get_bluff_logs() if log["is_bluff"]])

        return MatchResult(
            match_id=match_id,
            model_a=model_a_name,
            model_b=model_b_name,
            winner=winner_name,
            loser=loser_name,
            turns=turns,
            duration_seconds=duration,
            bluffs_a=bluffs_a,
            bluffs_b=bluffs_b
        )

    def _update_stats(self, result: MatchResult):
        """Update player stats based on match result."""
        stats_a = self.stats[result.model_a]
        stats_b = self.stats[result.model_b]

        # Update games played
        stats_a.games_played += 1
        stats_b.games_played += 1

        # Update wins
        if result.winner == result.model_a:
            stats_a.games_won += 1
        else:
            stats_b.games_won += 1

        # Update turns
        stats_a.total_turns += result.turns
        stats_b.total_turns += result.turns

        # Update bluffs
        stats_a.total_bluffs += result.bluffs_a
        stats_b.total_bluffs += result.bluffs_b

        # Update ELO ratings
        score_a = 1.0 if result.winner == result.model_a else 0.0
        new_rating_a, new_rating_b = self.elo_system.update_ratings(
            stats_a.elo_rating,
            stats_b.elo_rating,
            score_a
        )
        stats_a.elo_rating = new_rating_a
        stats_b.elo_rating = new_rating_b

    def _print_matchup_summary(self, model_a: str, model_b: str):
        """Print summary for a specific matchup."""
        a_wins = sum(1 for m in self.match_history
                     if m.model_a == model_a and m.model_b == model_b and m.winner == model_a)
        a_wins += sum(1 for m in self.match_history
                      if m.model_b == model_a and m.model_a == model_b and m.winner == model_a)

        total = self.config.games_per_matchup
        b_wins = total - a_wins

        print(f"  Results: {model_a} {a_wins}-{b_wins} {model_b}")

    def _print_final_results(self):
        """Print final leaderboard and statistics."""
        print("LEADERBOARD (by ELO):")
        print(f"{'Rank':<6}{'Model':<40}{'ELO':<8}{'W-L':<12}{'Win%':<8}{'Bluff%':<8}")
        print("-" * 80)

        # Sort by ELO rating
        sorted_stats = sorted(
            self.stats.values(),
            key=lambda s: s.elo_rating,
            reverse=True
        )

        for rank, stats in enumerate(sorted_stats, 1):
            wins = stats.games_won
            losses = stats.games_played - stats.games_won
            win_rate = stats.win_rate() * 100
            bluff_rate = stats.bluff_success_rate() * 100 if stats.total_bluffs > 0 else 0

            print(f"{rank:<6}{stats.model_name:<40}{stats.elo_rating:<8.0f}"
                  f"{wins}-{losses:<10}{win_rate:<7.1f}%{bluff_rate:<7.1f}%")

        print("\n" + "="*80 + "\n")

    def _save_results(self):
        """Save benchmark results to files."""
        output_dir = Path(self.config.output_dir)

        # Save stats
        stats_file = output_dir / f"{self.config.name}_stats.json"
        with open(stats_file, 'w') as f:
            stats_data = {
                model: {
                    "games_played": s.games_played,
                    "games_won": s.games_won,
                    "win_rate": s.win_rate(),
                    "elo_rating": s.elo_rating,
                    "bluff_success_rate": s.bluff_success_rate(),
                    "total_bluffs": s.total_bluffs,
                    "avg_turns_per_game": s.avg_turns_per_game()
                }
                for model, s in self.stats.items()
            }
            json.dump(stats_data, f, indent=2)

        # Save match history
        matches_file = output_dir / f"{self.config.name}_matches.jsonl"
        with open(matches_file, 'w') as f:
            for match in self.match_history:
                f.write(json.dumps({
                    "match_id": match.match_id,
                    "model_a": match.model_a,
                    "model_b": match.model_b,
                    "winner": match.winner,
                    "turns": match.turns,
                    "duration_seconds": match.duration_seconds,
                    "bluffs_a": match.bluffs_a,
                    "bluffs_b": match.bluffs_b
                }) + '\n')

        print(f"Results saved to {output_dir}/")
        print(f"  - {stats_file.name}")
        print(f"  - {matches_file.name}")

    def compare_models(self, model_a: str, model_b: str):
        """
        Generate detailed comparison between two models.

        Args:
            model_a: First model name
            model_b: Second model name
        """
        matches = [m for m in self.match_history
                   if (m.model_a == model_a and m.model_b == model_b) or
                      (m.model_a == model_b and m.model_b == model_a)]

        if not matches:
            print(f"No matches found between {model_a} and {model_b}")
            return

        a_wins = sum(1 for m in matches if m.winner == model_a)
        b_wins = len(matches) - a_wins

        avg_turns = sum(m.turns for m in matches) / len(matches)

        print(f"\n{model_a} vs {model_b}")
        print(f"{'='*60}")
        print(f"Head-to-head: {a_wins}-{b_wins}")
        print(f"Average game length: {avg_turns:.1f} turns")
        print(f"\nELO Ratings:")
        print(f"  {model_a}: {self.stats[model_a].elo_rating:.0f}")
        print(f"  {model_b}: {self.stats[model_b].elo_rating:.0f}")
