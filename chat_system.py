"""
Chat system for Coup - enables banter, negotiation, and psychological warfare.

Allows LLM agents to:
- Trash talk and sow doubt ("I have a Contessa, don't even think about it")
- Negotiate ("Let's team up against Player 2")
- Bluff verbally ("I definitely have a Duke")
- React to plays ("Nice try, but I'm blocking that")
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum


class ChatTiming(Enum):
    """When chat can occur."""
    PRE_ACTION = "PRE_ACTION"  # Before player takes action
    POST_ACTION = "POST_ACTION"  # After action announced, before challenges
    POST_CHALLENGE = "POST_CHALLENGE"  # After challenge resolved
    POST_BLOCK = "POST_BLOCK"  # After block attempt
    END_TURN = "END_TURN"  # End of turn


@dataclass
class ChatMessage:
    """A single chat message."""
    game_id: int
    turn_id: int
    player_id: int
    timing: ChatTiming
    message: str
    target_player: Optional[int] = None  # None = broadcast to all
    is_private: bool = False  # Future: private messages


@dataclass
class ChatHistory:
    """Track all chat messages in a game."""
    messages: List[ChatMessage] = field(default_factory=list)

    def add_message(
        self,
        game_id: int,
        turn_id: int,
        player_id: int,
        timing: ChatTiming,
        message: str,
        target_player: Optional[int] = None
    ):
        """Add a chat message to history."""
        msg = ChatMessage(
            game_id=game_id,
            turn_id=turn_id,
            player_id=player_id,
            timing=timing,
            message=message,
            target_player=target_player
        )
        self.messages.append(msg)
        return msg

    def get_recent_messages(self, num_messages: int = 5) -> List[ChatMessage]:
        """Get the N most recent messages."""
        return self.messages[-num_messages:] if self.messages else []

    def get_messages_this_turn(self, turn_id: int) -> List[ChatMessage]:
        """Get all messages from current turn."""
        return [msg for msg in self.messages if msg.turn_id == turn_id]

    def format_for_prompt(self, player_id: int, num_recent: int = 5) -> str:
        """Format recent chat history for LLM prompt."""
        recent = self.get_recent_messages(num_recent)
        if not recent:
            return "No recent chat messages.\n"

        formatted = "=== RECENT CHAT ===\n"
        for msg in recent:
            speaker = f"Player {msg.player_id}"
            if msg.target_player is not None:
                formatted += f"{speaker} (to Player {msg.target_player}): {msg.message}\n"
            else:
                formatted += f"{speaker}: {msg.message}\n"
        formatted += "\n"
        return formatted


class ChatConfig:
    """Configuration for chat behavior in games."""

    def __init__(
        self,
        enabled: bool = True,
        pre_action_chat: bool = True,  # Chat before taking action
        post_action_chat: bool = True,  # Chat after action announced
        post_challenge_chat: bool = False,  # Chat after challenges (can be spammy)
        post_block_chat: bool = True,  # Chat after blocks
        end_turn_chat: bool = False,  # Chat at end of turn
        max_message_length: int = 200,
        chat_temperature: float = 0.8,  # Higher temp for more varied chat
    ):
        """
        Initialize chat configuration.

        Args:
            enabled: Whether chat is enabled at all
            pre_action_chat: Allow chat before actions
            post_action_chat: Allow chat after actions announced
            post_challenge_chat: Allow chat after challenges
            post_block_chat: Allow chat after blocks
            end_turn_chat: Allow chat at end of turn
            max_message_length: Max chars per message
            chat_temperature: Temperature for chat generation
        """
        self.enabled = enabled
        self.pre_action_chat = pre_action_chat
        self.post_action_chat = post_action_chat
        self.post_challenge_chat = post_challenge_chat
        self.post_block_chat = post_block_chat
        self.end_turn_chat = end_turn_chat
        self.max_message_length = max_message_length
        self.chat_temperature = chat_temperature

    def should_chat_at(self, timing: ChatTiming) -> bool:
        """Check if chat is enabled at this timing."""
        if not self.enabled:
            return False

        timing_map = {
            ChatTiming.PRE_ACTION: self.pre_action_chat,
            ChatTiming.POST_ACTION: self.post_action_chat,
            ChatTiming.POST_CHALLENGE: self.post_challenge_chat,
            ChatTiming.POST_BLOCK: self.post_block_chat,
            ChatTiming.END_TURN: self.end_turn_chat,
        }
        return timing_map.get(timing, False)

    @classmethod
    def default(cls):
        """Default chat config - moderate chatting."""
        return cls(
            enabled=True,
            pre_action_chat=True,
            post_action_chat=True,
            post_challenge_chat=False,  # Reduce spam
            post_block_chat=True,
            end_turn_chat=False,  # Reduce spam
        )

    @classmethod
    def minimal(cls):
        """Minimal chat - only pre-action."""
        return cls(
            enabled=True,
            pre_action_chat=True,
            post_action_chat=False,
            post_challenge_chat=False,
            post_block_chat=False,
            end_turn_chat=False,
        )

    @classmethod
    def verbose(cls):
        """Maximum chat - all timings enabled."""
        return cls(
            enabled=True,
            pre_action_chat=True,
            post_action_chat=True,
            post_challenge_chat=True,
            post_block_chat=True,
            end_turn_chat=True,
        )

    @classmethod
    def disabled(cls):
        """No chat - original behavior."""
        return cls(enabled=False)
