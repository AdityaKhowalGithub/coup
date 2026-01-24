#!/usr/bin/env python3
"""
Demo script for chat-enabled Coup games.

Shows how LLMs banter, negotiate, and bluff verbally during gameplay.
"""

from coup_game import CoupGame
from llm_agent import LLMAgent
from chat_wrapper import ChatEnabledWrapper
from chat_system import ChatConfig


def print_chat_message(msg):
    """Pretty print a chat message."""
    print(f"  💬 Player {msg.player_id}: \"{msg.message}\"")


def main():
    print("="*60)
    print("COUP CHAT DEMO")
    print("="*60)
    print("\nWatch LLMs trash talk, bluff, and negotiate in Coup!\n")

    # Initialize game
    game = CoupGame(num_players=2)
    game.reset()

    # Initialize LLM (using smaller model for demo)
    print("Loading model...")
    llm = LLMAgent(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.8,  # Higher temp for more varied chat
        device="cuda"
    )

    # Create chat-enabled wrapper with verbose chat
    chat_config = ChatConfig.verbose()
    wrapper = ChatEnabledWrapper(game, llm, chat_config)

    print("Starting game with chat enabled!\n")
    print("="*60 + "\n")

    # Play a few turns
    done = False
    turn = 0
    max_turns = 20  # Just a demo

    while not done and turn < max_turns:
        current_player = game.current_player

        print(f"\n--- Turn {turn + 1} (Player {current_player}) ---")

        # Show game state
        obs = game.serialize_observation(current_player)
        print(f"Player {current_player} has {game.players[current_player].coins} coins, "
              f"{len(game.players[current_player].cards)} cards")

        # Play turn (includes chat)
        obs_result, done, info = wrapper.play_turn(current_player)

        # Show any chat messages from this turn
        turn_chats = wrapper.chat_history.get_messages_this_turn(turn + 1)
        for msg in turn_chats:
            print_chat_message(msg)

        # Show action taken
        if info.get("parsed_action"):
            print(f"  🎯 Action: {info['parsed_action']}")

        # Show thought (CoT faithfulness research)
        if info.get("thought"):
            print(f"  💭 Thought: {info['thought'][:100]}...")

        turn += 1

        if done:
            winner = game.get_winner()
            print(f"\n{'='*60}")
            print(f"🏆 GAME OVER! Player {winner} wins!")
            print(f"{'='*60}\n")

    # Show chat history
    print("\n" + "="*60)
    print("FULL CHAT HISTORY")
    print("="*60 + "\n")
    for msg in wrapper.chat_history.messages:
        print(f"Turn {msg.turn_id} ({msg.timing.value})")
        print_chat_message(msg)
        print()

    # Show bluff analysis
    bluffs = wrapper.get_bluff_logs()
    if bluffs:
        print("\n" + "="*60)
        print("BLUFF ANALYSIS (CoT Faithfulness)")
        print("="*60 + "\n")
        for bluff in bluffs:
            print(f"Turn {bluff['turn_id']}: Player {bluff['player_id']}")
            print(f"  Claimed: {bluff['claimed_role']}")
            print(f"  Actual hand: {bluff['ground_truth_hand']}")
            print(f"  Thought: {bluff['model_thought_trace'][:150]}...")
            print()


if __name__ == "__main__":
    main()
