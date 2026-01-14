# Covert-Coup: Auditing Emergent Collusion in Multi-Agent Games

**Research Question:** Can action-constrained agents invent covert signaling protocols to bypass safety monitors, and do these protocols generalize across model families?

Covert-Coup is a "Model Organism" of misalignment designed to stress-test Scalable Oversight. We investigate whether two LLM agents, sharing a hidden objective but restricted from communicating, can learn to use **game actions as a covert channel** to coordinate against a budget-constrained auditor.

---

## What is Coup?

Before it became a testbed for AI alignment, Coup was a tabletop game of bluffing and deduction. It distills complex social dynamics into a simple deck of cards.

* **The State:** Each player holds two private character cards (Duke, Assassin, Captain, Ambassador, Contessa).
* **The Bluff:** On your turn, you can claim the powers of *any* card, regardless of what you actually hold.
* **The Audit:** Opponents have a chance to **Challenge** your claim. If you were lying, you lose a life. If you were telling the truth, *they* lose a life.



This "High-Stakes Challenge" mechanic makes Coup the perfect **Model Organism for AI Safety**. It natively simulates the core problem of **Scalable Oversight**: an Auditor (the opponent) must police a Deceptive Agent (the player) but faces a high cost for every false accusation.

---

## The Experiment

We simulate a 3-Player environment (2 Red Agents vs 1 Blue Auditor) to test for **Algorithmic Collusion**.

### Phase 1: Baseline (Selfish)
**Hypothesis:** Agents with individual rewards will act selfishly.
- **Setup:** Red A and Red B optimize only for their own win.
- **Metric:** "Friendly Fire" (Red-on-Red attacks) should be high.

### Phase 2: Emergent Collusion (Shared Reward)
**Hypothesis:** Agents with a shared hidden reward will invent action-based signals to coordinate.
- **Setup:** Red A and Red B receive +1 reward if *either* wins.
- **Input:** "System Prompt Injection" (Reflexion) allows agents to evolve a shared strategy over 50 games.
- **Metric:** Red-on-Red attacks drop to near zero; "Kingmaking" moves increase.

### Phase 4: The Transfer Test (The Audit)
**Hypothesis:** Safety monitors are brittle.
- **Setup:** An Auditor trained to detect collusion in **Qwen-2.5** agents plays against **Llama-3** agents.
- **Metric:** **Transfer Gap**. The Auditor should fail to flag Llama-3's collusion because the signaling "dialect" is different.

---

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
uv pip install -r requirements.txt

# The .env file must contain your HuggingFace token
# HF_TOKEN=hf_...
```

we use `uv` to manage everything so start with
```bash
uv venv venv
source venv/bin/activate/
```

our models, phases and runs are completely configurable, for this initial run we ran this on a 1x B200 (180 GB SXM6) instance:
```sh
uv run python run_covert_coup.py --games 500 --device cuda --model Qwen/Qwen2.5-72B-Instruct
```



