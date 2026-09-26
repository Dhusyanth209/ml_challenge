import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from feature_engineering import FEATURES

logger = logging.getLogger(__name__)


def macro_f05(s1_ids, labels, preds, gt_counts):
    """
    Exact challenge metric over the S1 ids in `gt_counts` (a Series: s1_id -> #true matches).
    True matches missed by blocking still count against recall.
    """
    d = pd.DataFrame({'s1': s1_ids, 'tp': labels & preds, 'pp': preds})
    agg = d.groupby('s1', sort=False)[['tp', 'pp']].sum().reindex(gt_counts.index, fill_value=0)
    tp, pp, ap = agg['tp'].values, agg['pp'].values, gt_counts.values
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(pp > 0, tp / pp, 0)
        r = np.where(ap > 0, tp / ap, 0)
        f = np.where(tp > 0, 1.25 * p * r / (0.25 * p + r), 0)
    f = np.where(ap == 0, (pp == 0).astype(float), f)
    return f.mean()


class MatchingModel:
    def __init__(self):
        self.model = HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=500, max_leaf_nodes=63, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=30, random_state=42)
        self.threshold = 0.5

    def fit(self, feats, labels):
        logger.info(f"Training HistGradientBoosting on {len(feats):,} pairs "
                    f"({labels.mean():.3%} positive)...")
        self.model.fit(feats[FEATURES].values, labels)
        logger.info(f"  stopped at {self.model.n_iter_} iterations")

    def predict(self, feats):
        return self.model.predict_proba(feats[FEATURES].values)[:, 1]

    def tune_threshold(self, s1_ids, labels, probs, gt_counts):
        best = (0.0, 0.5)
        for t in np.arange(0.2, 0.96, 0.02):
            score = macro_f05(s1_ids, labels, probs > t, gt_counts)
            best = max(best, (score, t))
        ceiling = macro_f05(s1_ids, labels, labels.astype(bool), gt_counts)
        logger.info(f"  validation macro F0.5 = {best[0]:.4f} at threshold {best[1]:.2f} "
                    f"(blocking ceiling = {ceiling:.4f})")
        self.threshold = best[1]
        return best
