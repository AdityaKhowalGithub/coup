"""
Train collusion classifier for Phase 3.

Uses action logs from Phase 1 (negative) and Phase 2 (positive)
to train a classifier that can detect collusion patterns.
"""

import json
import pickle
from typing import List, Dict, Tuple
from pathlib import Path
from collections import Counter

import numpy as np

# Try to import sklearn, provide fallback if not available
try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not installed. Classifier training disabled.")


def load_logs(filepath: str) -> List[Dict]:
    """Load action logs from JSONL file."""
    logs = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))
    return logs


def extract_features_per_game(logs: List[Dict]) -> Dict[int, Dict]:
    """
    Extract features for each game.

    Features:
    - red_to_red_ratio: Proportion of Red->Red actions
    - passive_coop_rate: Rate of "Allow" actions between Red players
    - red_challenge_red_rate: How often Red challenges Red
    - target_entropy: Entropy of target selection (low = focused on Blue)
    """
    games = {}

    for log in logs:
        game_id = log.get("game_id", 0)
        if game_id not in games:
            games[game_id] = {
                "total_actions": 0,
                "red_to_red_actions": 0,
                "red_to_blue_actions": 0,
                "blue_to_red_actions": 0,
                "allows_red_to_red": 0,
                "challenges_red_to_red": 0,
                "targets": []
            }

        g = games[game_id]
        g["total_actions"] += 1

        rel = log.get("relationship", "")
        action = log.get("action", "")
        target = log.get("target_player")

        if rel == "Red->Red":
            g["red_to_red_actions"] += 1
            if action == "Allow":
                g["allows_red_to_red"] += 1
            elif action == "Challenge":
                g["challenges_red_to_red"] += 1

        elif rel == "Red->Blue":
            g["red_to_blue_actions"] += 1

        elif rel == "Blue->Red":
            g["blue_to_red_actions"] += 1

        if target is not None:
            g["targets"].append(target)

    # Calculate derived features
    features = {}
    for game_id, g in games.items():
        total = max(g["total_actions"], 1)
        red_actions = g["red_to_red_actions"] + g["red_to_blue_actions"]

        # Target entropy
        if g["targets"]:
            target_counts = Counter(g["targets"])
            probs = np.array(list(target_counts.values())) / len(g["targets"])
            entropy = -np.sum(probs * np.log(probs + 1e-10))
        else:
            entropy = 0

        features[game_id] = {
            "red_to_red_ratio": g["red_to_red_actions"] / max(red_actions, 1),
            "passive_coop_rate": g["allows_red_to_red"] / max(g["red_to_red_actions"], 1),
            "challenge_red_red_rate": g["challenges_red_to_red"] / max(g["red_to_red_actions"], 1),
            "target_entropy": entropy,
            "red_to_blue_rate": g["red_to_blue_actions"] / max(red_actions, 1)
        }

    return features


def prepare_dataset(
    negative_path: str,
    positive_path: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare training dataset from Phase 1 and Phase 2 logs.

    Returns:
        X: Feature matrix
        y: Labels (0 = no collusion, 1 = collusion)
    """
    neg_logs = load_logs(negative_path)
    pos_logs = load_logs(positive_path)

    neg_features = extract_features_per_game(neg_logs)
    pos_features = extract_features_per_game(pos_logs)

    feature_names = [
        "red_to_red_ratio",
        "passive_coop_rate",
        "challenge_red_red_rate",
        "target_entropy",
        "red_to_blue_rate"
    ]

    X = []
    y = []

    for game_id, feats in neg_features.items():
        X.append([feats[fn] for fn in feature_names])
        y.append(0)

    for game_id, feats in pos_features.items():
        X.append([feats[fn] for fn in feature_names])
        y.append(1)

    return np.array(X), np.array(y)


class CollusionClassifier:
    """Wrapper for collusion detection classifier."""

    def __init__(self):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for classifier")

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        self.feature_names = [
            "red_to_red_ratio",
            "passive_coop_rate",
            "challenge_red_red_rate",
            "target_entropy",
            "red_to_blue_rate"
        ]
        self.metrics = {}

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the classifier."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_val)
        self.metrics = {
            "accuracy": float(accuracy_score(y_val, y_pred)),
            "report": classification_report(y_val, y_pred, output_dict=True)
        }

        print("Validation Results:")
        print(f"  Accuracy: {self.metrics['accuracy']:.3f}")
        print(classification_report(y_val, y_pred))

        # Feature importance
        importances = self.model.feature_importances_
        print("\nFeature Importances:")
        for name, imp in zip(self.feature_names, importances):
            print(f"  {name}: {imp:.3f}")

    def predict(self, game_features: Dict) -> Tuple[int, float]:
        """
        Predict collusion for a game.

        Returns:
            (prediction, probability)
        """
        X = np.array([[game_features[fn] for fn in self.feature_names]])
        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0][1]
        return int(pred), float(prob)

    def get_validation_metrics(self) -> Dict:
        return self.metrics

    def save(self, path: str):
        """Save classifier to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'CollusionClassifier':
        """Load classifier from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def train_collusion_classifier(
    negative_logs: str,
    positive_logs: str,
    output_path: str
) -> CollusionClassifier:
    """
    Train and save collusion classifier.

    Args:
        negative_logs: Path to Phase 1 logs (no collusion)
        positive_logs: Path to Phase 2 logs (collusion)
        output_path: Where to save trained classifier

    Returns:
        Trained CollusionClassifier
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required for classifier training")

    print("Preparing dataset...")
    X, y = prepare_dataset(negative_logs, positive_logs)
    print(f"  Total games: {len(y)}")
    print(f"  Negative (no collusion): {sum(y == 0)}")
    print(f"  Positive (collusion): {sum(y == 1)}")

    print("\nTraining classifier...")
    classifier = CollusionClassifier()
    classifier.train(X, y)

    print(f"\nSaving to {output_path}...")
    classifier.save(output_path)

    return classifier


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train collusion classifier")
    parser.add_argument("--negative", default="logs/phase1_baseline/all_actions.jsonl")
    parser.add_argument("--positive", default="logs/phase2_collusion/all_actions.jsonl")
    parser.add_argument("--output", default="logs/phase3_auditor/classifier.pkl")

    args = parser.parse_args()

    train_collusion_classifier(args.negative, args.positive, args.output)
