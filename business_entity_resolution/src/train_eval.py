"""
Model Training and Threshold Optimization Module.
Trains a LightGBM Binary Classifier on candidate pairs and performs a fine-grained
threshold sweep (0.70 to 0.95) to maximize the official Macro F_0.5 evaluation metric.
"""

import numpy as np
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score


def calculate_macro_f05(predicted_dict: dict, ground_truth_dict: dict, s1_ids: set) -> tuple:
    """
    Computes exact competition Macro F_0.5 score across all Source 1 entities in s1_ids.
    Formula:
        F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)

    Singletons (empty ground truth):
        - Predict empty list -> score = 1.0
        - Predict any matches -> score = 0.0

    Non-singletons:
        - Predict empty list -> score = 0.0
        - Otherwise standard F_0.5 per entity
    Returns: (macro_f05, singleton_accuracy, non_singleton_f05)
    """
    scores = []
    singleton_scores = []
    non_singleton_scores = []

    for sid in s1_ids:
        gt_matches = ground_truth_dict.get(sid, set())
        pred_matches = set(predicted_dict.get(sid, []))

        # Case 1: Singleton entity in Ground Truth
        if len(gt_matches) == 0:
            if len(pred_matches) == 0:
                score = 1.0
            else:
                score = 0.0
            singleton_scores.append(score)
            scores.append(score)
            continue

        # Case 2: Entity with true matches
        if len(pred_matches) == 0:
            score = 0.0
            non_singleton_scores.append(score)
            scores.append(score)
            continue

        tp = len(gt_matches & pred_matches)
        precision = tp / len(pred_matches)
        recall = tp / len(gt_matches)

        if precision + recall == 0 or tp == 0:
            score = 0.0
        else:
            score = (1.25 * precision * recall) / (0.25 * precision + recall)

        non_singleton_scores.append(score)
        scores.append(score)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 1.0
    non_singleton_f05 = float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0

    return macro_f05, singleton_acc, non_singleton_f05


class LightGBMMatcher:
    """
    LightGBM classifier tailored for high-precision entity resolution ranking.
    """
    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, random_state: int = 42):
        self.model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
        self.optimal_threshold = 0.85

    def train(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None):
        """Train the LightGBM model and print feature importances."""
        print(f"\nTraining LightGBM Classifier on {len(X_train):,} candidate pairs...")
        pos_count = int(np.sum(y_train == 1))
        neg_count = int(np.sum(y_train == 0))
        print(f"  Class Distribution: {pos_count:,} Matches (1), {neg_count:,} Non-matches (0)")

        self.model.fit(X_train, y_train)

        # Print top feature importances
        if feature_names is not None:
            importances = self.model.feature_importances_
            sorted_idx = np.argsort(-importances)
            print("\nFeature Importances (Split Count):")
            for rank, idx in enumerate(sorted_idx[:8], 1):
                print(f"  {rank}. {feature_names[idx]:<25}: {importances[idx]}")

    def predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Returns array of match probabilities (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def optimize_threshold(self, val_pairs: list, val_probs: np.ndarray,
                           gt_dict: dict, val_s1_ids: set,
                           threshold_range=(0.70, 0.96, 0.01)) -> float:
        """
        Sweeps decision threshold from 0.70 to 0.95 in 0.01 increments
        to find the global peak for Macro F_0.5.
        """
        print("\n" + "=" * 65)
        print("SWEEPING THRESHOLDS FOR MACRO F_0.5 OPTIMIZATION:")
        print(f"{'Threshold':<12} | {'Macro F_0.5':<14} | {'Singleton Acc':<16} | {'Match F_0.5':<12}")
        print("-" * 65)

        start, stop, step = threshold_range
        thresholds = np.arange(start, stop, step)

        best_thresh = 0.85
        best_score = -1.0
        best_singleton_acc = 0.0

        for thresh in thresholds:
            thresh = round(float(thresh), 3)

            # Build prediction mapping at this threshold
            pred_dict = {sid: [] for sid in val_s1_ids}
            for (s1_id, cand_id), prob in zip(val_pairs, val_probs):
                if prob >= thresh:
                    pred_dict[s1_id].append(cand_id)

            macro_f, s_acc, m_f = calculate_macro_f05(pred_dict, gt_dict, val_s1_ids)

            marker = " *" if macro_f > best_score else ""
            print(f"{thresh:<12.2f} | {macro_f:<14.4f} | {s_acc:<16.2%} | {m_f:<12.4f}{marker}")

            if macro_f > best_score:
                best_score = macro_f
                best_thresh = thresh
                best_singleton_acc = s_acc

        print("=" * 65)
        print(f"OPTIMAL THRESHOLD FOUND: {best_thresh:.2f}")
        print(f"  Peak Validation Macro F_0.5: {best_score:.4f}")
        print(f"  Singleton Accuracy:          {best_singleton_acc:.2%}")
        print("=" * 65)

        self.optimal_threshold = best_thresh
        return best_thresh
