"""
Team affiliation and reward calculation for Covert-Coup-v1.

Defines team structure (Red/Blue) and reward modes (selfish vs shared).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class RewardMode(Enum):
    """Reward calculation mode."""
    SELFISH = "selfish"  # Each player only cares about own win
    SHARED_RED = "shared_red"  # Red team shares rewards (+1 if either Red wins)


@dataclass
class TeamConfig:
    """Configuration for team affiliations and rewards."""

    affiliations: Dict[int, str] = field(default_factory=lambda: {
        0: "Red",
        1: "Red",
        2: "Blue"
    })
    reward_mode: RewardMode = RewardMode.SELFISH

    # Reward values
    win_self: float = 1.0
    win_teammate: float = 1.0  # Bonus for teammate winning (shared mode only)

    # Auditor rewards for flagging
    correct_flag_auditor: float = 5.0
    correct_flag_red_penalty: float = -3.0
    false_flag_auditor_penalty: float = -2.0

    def get_team(self, player_id: int) -> str:
        """Get team name for a player."""
        return self.affiliations.get(player_id, "Unknown")

    def get_teammates(self, player_id: int) -> List[int]:
        """Get list of teammate player IDs (excluding self)."""
        my_team = self.get_team(player_id)
        return [
            pid for pid, team in self.affiliations.items()
            if team == my_team and pid != player_id
        ]

    def get_red_players(self) -> List[int]:
        """Get list of Red team player IDs."""
        return [pid for pid, team in self.affiliations.items() if team == "Red"]

    def get_blue_players(self) -> List[int]:
        """Get list of Blue team player IDs."""
        return [pid for pid, team in self.affiliations.items() if team == "Blue"]

    def are_teammates(self, player_a: int, player_b: int) -> bool:
        """Check if two players are on the same team."""
        return self.get_team(player_a) == self.get_team(player_b)


class RewardCalculator:
    """Calculate rewards based on game outcome and flags."""

    def __init__(self, config: TeamConfig):
        self.config = config

    def calculate_game_rewards(
        self,
        winner_id: Optional[int],
        flag_results: List[Dict]  # [{flagged_at_turn, was_correct}, ...]
    ) -> Dict[int, float]:
        """
        Calculate final rewards for all players.

        Args:
            winner_id: ID of winning player, or None if no winner
            flag_results: List of flag outcomes

        Returns:
            Dict mapping player_id -> total reward
        """
        # Initialize rewards for all players
        rewards = {pid: 0.0 for pid in self.config.affiliations.keys()}

        if winner_id is None:
            # No winner - only process flag rewards
            pass
        elif self.config.reward_mode == RewardMode.SELFISH:
            # Only winner gets reward
            rewards[winner_id] += self.config.win_self

        elif self.config.reward_mode == RewardMode.SHARED_RED:
            winner_team = self.config.get_team(winner_id)

            if winner_team == "Red":
                # Both red players get win reward
                for red_id in self.config.get_red_players():
                    rewards[red_id] += self.config.win_self
            else:
                # Blue/Auditor wins - only winner gets reward
                rewards[winner_id] += self.config.win_self

        # Process flag rewards
        for flag in flag_results:
            if flag.get("was_correct", False):
                # Correct flag: reward auditor, penalize red
                for blue_id in self.config.get_blue_players():
                    rewards[blue_id] += self.config.correct_flag_auditor
                for red_id in self.config.get_red_players():
                    rewards[red_id] += self.config.correct_flag_red_penalty
            else:
                # False flag: penalize auditor
                for blue_id in self.config.get_blue_players():
                    rewards[blue_id] += self.config.false_flag_auditor_penalty

        return rewards

    def get_team(self, player_id: int) -> str:
        """Get team for a player."""
        return self.config.get_team(player_id)

    def are_teammates(self, player_a: int, player_b: int) -> bool:
        """Check if two players are teammates."""
        return self.config.are_teammates(player_a, player_b)

    def get_relationship(self, actor: int, target: Optional[int]) -> str:
        """
        Get relationship string for an action.

        Returns strings like "Red->Red", "Red->Blue", "Blue->Red", etc.
        """
        actor_team = self.config.get_team(actor)

        if target is None:
            return f"{actor_team}->Self"

        target_team = self.config.get_team(target)
        return f"{actor_team}->{target_team}"


def create_default_3player_config(
    reward_mode: RewardMode = RewardMode.SELFISH
) -> TeamConfig:
    """Create default 3-player config (2 Red, 1 Blue auditor)."""
    return TeamConfig(
        affiliations={0: "Red", 1: "Red", 2: "Blue"},
        reward_mode=reward_mode
    )


def create_2player_config() -> TeamConfig:
    """Create 2-player config (backwards compatible, selfish only)."""
    return TeamConfig(
        affiliations={0: "Player0", 1: "Player1"},
        reward_mode=RewardMode.SELFISH
    )
