"""
Quick test script to verify setup before running full experiment.

Tests:
1. Environment variables loaded
2. Model can be loaded
3. Basic generation works
4. Game engine works
"""

import os
from dotenv import load_dotenv

def test_env():
    """Test environment setup."""
    print("=== Testing Environment ===")
    load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")

    if token:
        print(f"✓ HuggingFace token found: {token[:10]}...")
    else:
        print("✗ HuggingFace token NOT found in .env")
        return False
    return True


def test_game():
    """Test game engine."""
    print("\n=== Testing Game Engine ===")
    try:
        from coup_game import CoupGame

        game = CoupGame()
        obs = game.reset()

        print(f"✓ Game initialized")
        print(f"  Player 0 hand: {game.get_ground_truth_hand(0)}")
        print(f"  Player 1 hand: {game.get_ground_truth_hand(1)}")

        # Test a basic action
        obs, done, info = game.step("Income")
        print(f"✓ Basic action (Income) works")

        return True
    except Exception as e:
        print(f"✗ Game engine failed: {e}")
        return False


def test_llm():
    """Test LLM loading and generation."""
    print("\n=== Testing LLM ===")
    print("WARNING: This will download the model (~4-15GB) on first run")

    response = input("Continue with model download? (y/n): ")
    if response.lower() != 'y':
        print("Skipping LLM test")
        return True

    try:
        from llm_agent import LLMAgent

        # Use smallest Qwen model for testing
        print("\nLoading Qwen/Qwen2.5-3B-Instruct (smallest model)...")
        llm = LLMAgent(model_name="Qwen/Qwen2.5-3B-Instruct", temperature=0.7)

        print("\nTesting generation...")
        prompt = "Say 'Hello' and nothing else."
        response = llm.generate(prompt, max_tokens=10)

        print(f"✓ Generation works")
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response}")

        return True
    except Exception as e:
        print(f"✗ LLM failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wrapper():
    """Test thought wrapper."""
    print("\n=== Testing ThoughtProbe Wrapper ===")
    try:
        from coup_game import CoupGame
        from llm_agent import LLMAgent
        from thought_wrapper import ThoughtProbeWrapper

        # Mock LLM for testing (don't actually load model)
        class MockLLM:
            model_name = "test"
            def generate(self, prompt, **kwargs):
                return "THOUGHT: Test thought\nACTION: Income"

        game = CoupGame()
        game.reset()

        wrapper = ThoughtProbeWrapper(game, MockLLM())

        # Test parsing
        thought, action = wrapper._parse_response("THOUGHT: I need coins\nACTION: Income")
        assert thought == "I need coins", f"Expected 'I need coins', got '{thought}'"
        assert action == "Income", f"Expected 'Income', got '{action}'"

        print(f"✓ Wrapper parsing works")

        return True
    except Exception as e:
        print(f"✗ Wrapper failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("CoT Faithfulness - Setup Test\n")

    results = []

    results.append(("Environment", test_env()))
    results.append(("Game Engine", test_game()))
    results.append(("Wrapper", test_wrapper()))
    results.append(("LLM", test_llm()))

    print("\n" + "="*50)
    print("RESULTS:")
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")

    all_passed = all(r[1] for r in results)

    if all_passed:
        print("\n✓ All tests passed! Ready to run experiment.")
        print("\nRun smoke test with:")
        print("  python run_experiment.py --smoke-test")
    else:
        print("\n✗ Some tests failed. Fix issues before running experiment.")


if __name__ == "__main__":
    main()
