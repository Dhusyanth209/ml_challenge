"""
LightGBM Training & Macro F_0.5 Threshold Optimizer Module (Method 2).
Trains LightGBM Binary Classifier on candidate pairs and sweeps thresholds
from 0.70 to 0.95 in 0.01 increments to strictly maximize Macro F_0.5.
"""

import numpy as np
import lightgbm as lgb


def evaluate_macro_f05(pred_dict: dict, gt_dict: dict, eval_s1_ids: set) -> tuple:
    """
    Computes exact Macro F_0.5, Macro Precision, Macro Recall, and Singleton Accuracy.
    """
    scores = []
    precisions = []
    recalls = []
    singleton_scores = []

    for sid in eval_s1_ids:
        gt_matches = gt_dict.get(sid, set())
        preds = set(pred_dict.get(sid, []))

        # Ground truth singleton
        if len(gt_matches) == 0:
            if len(preds) == 0:
                s_score = 1.0
                prec = 1.0
                rec = 1.0
            else:
                s_score = 0.0
                prec = 0.0
                rec = 1.0
            singleton_scores.append(s_score)
            scores.append(s_score)
            precisions.append(prec)
            recalls.append(rec)
            continue

        # Non-singleton entity
        if len(preds) == 0:
            scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        tp = len(gt_matches & preds)
        prec = tp / len(preds)
        rec = tp / len(gt_matches)

        precisions.append(prec)
        recalls.append(rec)

        if tp == 0 or (prec + rec) == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            scores.append(f05)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 1.0

    return macro_f05, macro_prec, macro_rec, singleton_acc


class LightGBMEntityMatcher:
    """LightGBM Classifier wrapper for candidate pair ranking and classification."""

    def __init__(self, n_estimators: int = 300, learning_rate: float = 0.05, num_leaves: int = 31):
        self.model = lgb.LGBMClassifier(
            objective="binary",
            metric="binary_logloss",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        self.best_threshold = 0.85

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, feature_names: list = None):
        """Fits model and displays feature importances."""
        pos = int(np.sum(y_train == 1))
        neg = int(np.sum(y_train == 0))
        print(f"\nFitting LightGBM on {len(X_train):,} training pairs ({pos:,} matches, {neg:,} non-matches)...")

        self.model.fit(X_train, y_train)

        if feature_names:
            importances = self.model.feature_importances_
            sorted_idx = np.argsort(-importances)
            print("\nLightGBM Feature Importance (Top Features):")
            for rank, idx in enumerate(sorted_idx[:8], 1):
                print(f"  {rank}. {feature_names[idx]:<25}: {importances[idx]}")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns match probability (class 1)."""
        return self.model.predict_proba(X)[:, 1]

    def sweep_thresholds(self, val_pairs: list, val_probs: np.ndarray,
                         gt_dict: dict, val_s1_ids: set,
                         threshold_range=None) -> tuple:
        """
        Sweeps decision threshold from 0.70 to 0.95 in 0.01 increments
        optimizing strictly for Macro F_0.5.
        """
        if threshold_range is None:
            thresholds = [round(t, 2) for t in np.arange(0.70, 0.96, 0.01)]
        else:
            thresholds = threshold_range

        print("\n" + "=" * 70)
        print("FINE-GRAINED THRESHOLD SWEEP (Macro F_0.5 Optimization):")
        print(f"{'Threshold':<12} | {'Macro F_0.5':<14} | {'Precision':<12} | {'Recall':<12} | {'Singleton Acc'}")
        print("-" * 70)

        best_score = -1.0
        best_thresh = 0.85
        best_metrics = (0.0, 0.0, 0.0, 0.0)

        for thresh in thresholds:
            # Build prediction dictionary
            pred_dict = {sid: [] for sid in val_s1_ids}
            for (sid, cid), prob in zip(val_pairs, val_probs):
                if prob >= thresh:
                    pred_dict[sid].append(cid)

            f05, prec, rec, s_acc = evaluate_macro_f05(pred_dict, gt_dict, val_s1_ids)

            marker = " *" if f05 > best_score else ""
            print(f"{thresh:<12.2f} | {f05:<14.4f} | {prec:<12.4f} | {rec:<12.4f} | {s_acc:<13.2%}{marker}")

            if f05 > best_score:
                best_score = f05
                best_thresh = thresh
                best_metrics = (f05, prec, rec, s_acc)

        print("=" * 70)
        print(f"Optimal Threshold:       {best_thresh:.2f}")
        print(f"Peak Validation Macro F_0.5: {best_metrics[0]:.4f}")
        print(f"Macro Precision at Peak: {best_metrics[1]:.4f}")
        print(f"Macro Recall at Peak:    {best_metrics[2]:.4f}")
        print(f"Singleton Accuracy:      {best_metrics[3]:.2%}")
        print("=" * 70)

        self.best_threshold = best_thresh
        return best_thresh, best_metrics


def build_final_matches(pairs: list, probs: np.ndarray, threshold: float, all_s1_ids: list) -> dict:
    """
    Applies calibrated threshold to candidate pairs. Singletons are preserved as empty lists.
    Returns: {s1_id: [cand_id, ...]}
    """
    final_dict = {sid: [] for sid in all_s1_ids}
    for (sid, cid), prob in zip(pairs, probs):
        if prob >= threshold:
            final_dict[sid].append(cid)
    return final_dict
