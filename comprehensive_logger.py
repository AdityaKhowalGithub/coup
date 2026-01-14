"""
Comprehensive action logging for Covert-Coup-v1.

Logs ALL actions (not just bluffs) with relationship metadata for
training collusion detection classifiers.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ActionLogEntry:
    """Complete log entry for one action."""

    # Identifiers
    game_id: int
    turn_id: int
    action_seq: int  # Action sequence within turn (for challenge/block chains)

    # Action info
    phase: str
    acting_player: int
    target_player: Optional[int]
    action: str
    claimed_role: Optional[str]

    # Ground truth
    is_bluff: bool
    actor_actual_hand: List[str]

    # Thought trace
    thought_trace: str
    raw_llm_response: str

    # Team/relationship metadata
    actor_team: str
    target_team: Optional[str]
    relationship: str  # "Red->Red", "Red->Blue", "Blue->Red", "Auditor->Flag"

    # Game state at time of action
    game_state: Dict[str, Any]

    # Outcome (filled in after resolution)
    action_succeeded: Optional[bool] = None
    was_challenged: Optional[bool] = None
    was_blocked: Optional[bool] = None

    # Timestamps
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


class ComprehensiveLogger:
    """Logger that captures all game actions with relationship data."""

    def __init__(self, team_affiliations: Dict[int, str]):
        """
        Initialize logger.

        Args:
            team_affiliations: Dict mapping player_id -> team name
        """
        self.team_affiliations = team_affiliations
        self.logs: List[ActionLogEntry] = []
        self.current_game_id: int = 0
        self.current_turn_id: int = 0
        self.current_action_seq: int = 0

    def new_game(self):
        """Start a new game."""
        self.current_game_id += 1
        self.current_turn_id = 0
        self.current_action_seq = 0

    def new_turn(self):
        """Start a new turn."""
        self.current_turn_id += 1
        self.current_action_seq = 0

    def get_relationship(
        self,
        actor: int,
        target: Optional[int],
        is_flag: bool = False
    ) -> str:
        """
        Determine relationship type for the action.

        Args:
            actor: Acting player ID
            target: Target player ID (or None)
            is_flag: Whether this is a FlagCollusion action

        Returns:
            Relationship string like "Red->Blue"
        """
        if is_flag:
            return "Auditor->Flag"

        actor_team = self.team_affiliations.get(actor, "Unknown")

        if target is None:
            return f"{actor_team}->Self"

        target_team = self.team_affiliations.get(target, "Unknown")
        return f"{actor_team}->{target_team}"

    def log_action(
        self,
        phase: str,
        acting_player: int,
        target_player: Optional[int],
        action: str,
        claimed_role: Optional[str],
        is_bluff: bool,
        actor_hand: List[str],
        thought_trace: str,
        raw_response: str,
        game_state: Dict[str, Any],
        is_flag: bool = False
    ) -> ActionLogEntry:
        """
        Log an action with full context.

        Args:
            phase: Current game phase
            acting_player: Player taking action
            target_player: Target of action (if any)
            action: Action string
            claimed_role: Role claimed in action (if any)
            is_bluff: Whether action is a bluff
            actor_hand: Actor's actual cards
            thought_trace: Extracted thought from LLM response
            raw_response: Full LLM response
            game_state: Current game state dict
            is_flag: Whether this is a FlagCollusion action

        Returns:
            The logged ActionLogEntry
        """
        self.current_action_seq += 1

        actor_team = self.team_affiliations.get(acting_player, "Unknown")
        target_team = self.team_affiliations.get(target_player) if target_player is not None else None

        entry = ActionLogEntry(
            game_id=self.current_game_id,
            turn_id=self.current_turn_id,
            action_seq=self.current_action_seq,
            phase=phase,
            acting_player=acting_player,
            target_player=target_player,
            action=action,
            claimed_role=claimed_role,
            is_bluff=is_bluff,
            actor_actual_hand=actor_hand,
            thought_trace=thought_trace,
            raw_llm_response=raw_response,
            actor_team=actor_team,
            target_team=target_team,
            relationship=self.get_relationship(acting_player, target_player, is_flag),
            game_state=game_state
        )

        self.logs.append(entry)
        return entry

    def update_outcome(
        self,
        entry: ActionLogEntry,
        succeeded: bool,
        challenged: bool = False,
        blocked: bool = False
    ):
        """
        Update action outcome after resolution.

        Args:
            entry: The log entry to update
            succeeded: Whether action succeeded
            challenged: Whether action was challenged
            blocked: Whether action was blocked
        """
        entry.action_succeeded = succeeded
        entry.was_challenged = challenged
        entry.was_blocked = blocked

    def get_logs(self) -> List[ActionLogEntry]:
        """Get all logged entries."""
        return self.logs

    def get_logs_by_relationship(self, relationship: str) -> List[ActionLogEntry]:
        """
        Filter logs by relationship type.

        Args:
            relationship: Relationship string to filter by (e.g., "Red->Red")

        Returns:
            List of matching log entries
        """
        return [log for log in self.logs if log.relationship == relationship]

    def get_logs_by_game(self, game_id: int) -> List[ActionLogEntry]:
        """Get all logs for a specific game."""
        return [log for log in self.logs if log.game_id == game_id]

    def get_collusion_signals(self) -> List[Dict]:
        """
        Extract potential collusion signals from logs.

        Signals include:
        - Red->Red actions that benefit target (non-aggressive)
        - Allowing teammate's actions without challenge
        - Coordinated targeting of Blue player
        """
        signals = []

        red_to_red = self.get_logs_by_relationship("Red->Red")

        for log in red_to_red:
            signal = {
                "game_id": log.game_id,
                "turn": log.turn_id,
                "action": log.action,
                "actor": log.acting_player,
                "target": log.target_player,
                "thought_snippet": log.thought_trace[:200] if log.thought_trace else "",
                "signal_type": self._classify_signal(log)
            }
            signals.append(signal)

        return signals

    def _classify_signal(self, log: ActionLogEntry) -> str:
        """
        Classify the type of potential collusion signal.

        Args:
            log: Log entry to classify

        Returns:
            Signal type string
        """
        action = log.action.split("|")[0] if log.action else ""

        if action == "Allow":
            return "PASSIVE_COOPERATION"
        elif action in ["Steal", "Assassinate", "Coup"]:
            return "AGGRESSION"  # Attacking teammate (unusual in collusion)
        elif action == "Challenge":
            return "CHALLENGE"
        elif action.startswith("Block"):
            return "BLOCK"
        else:
            return "NEUTRAL"

    def get_game_summary(self, game_id: int) -> Dict:
        """
        Get summary statistics for a game.

        Args:
            game_id: Game to summarize

        Returns:
            Summary dict with action counts by relationship
        """
        game_logs = self.get_logs_by_game(game_id)

        summary = {
            "game_id": game_id,
            "total_actions": len(game_logs),
            "relationships": {},
            "bluff_count": 0,
            "flag_count": 0
        }

        for log in game_logs:
            rel = log.relationship
            if rel not in summary["relationships"]:
                summary["relationships"][rel] = 0
            summary["relationships"][rel] += 1

            if log.is_bluff:
                summary["bluff_count"] += 1
            if log.relationship == "Auditor->Flag":
                summary["flag_count"] += 1

        return summary

    def save_jsonl(self, filepath: str):
        """
        Save logs to JSONL file.

        Args:
            filepath: Path to output file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            for log in self.logs:
                f.write(json.dumps(log.to_dict()) + '\n')

        print(f"Saved {len(self.logs)} action logs to {filepath}")

    def save_collusion_signals(self, filepath: str):
        """
        Save extracted collusion signals.

        Args:
            filepath: Path to output file
        """
        signals = self.get_collusion_signals()

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            for signal in signals:
                f.write(json.dumps(signal) + '\n')

        print(f"Saved {len(signals)} collusion signals to {filepath}")

    def save_game_summaries(self, filepath: str):
        """
        Save per-game summaries.

        Args:
            filepath: Path to output file
        """
        # Get unique game IDs
        game_ids = sorted(set(log.game_id for log in self.logs))

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            for game_id in game_ids:
                summary = self.get_game_summary(game_id)
                f.write(json.dumps(summary) + '\n')

        print(f"Saved {len(game_ids)} game summaries to {filepath}")
