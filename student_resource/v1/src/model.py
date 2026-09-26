import lightgbm as lgb
from sklearn.model_selection import GroupKFold, KFold
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class MatchingModel:
    def __init__(self):
        self.models = []
        self.features = ['name_jw', 'name_ts', 'addr_jw', 'addr_tset',
                         'num_conflict', 'name_subset', 'len_ratio', 'blocking_score']

    def prepare_training_data(self, df_features, gt_df):
        """Label each candidate pair as match (1) / non-match (0).

        Vectorized via an explode + merge instead of iterating over 2.2M GT
        rows and applying a per-row set lookup across ~20M candidate pairs.
        """
        logger.info("Preparing training labels...")

        gt = gt_df[['source1_entity_id', 'matched_entity_ids']].copy()
        gt['matched_entity_ids'] = gt['matched_entity_ids'].fillna('').astype(str)
        gt = gt[gt['matched_entity_ids'].str.strip() != '']
        gt = gt.assign(candidate_entity_id=gt['matched_entity_ids'].str.split(','))
        gt = gt.explode('candidate_entity_id')
        gt['candidate_entity_id'] = gt['candidate_entity_id'].str.strip()
        gt = gt[gt['candidate_entity_id'] != '']

        pos = gt[['source1_entity_id', 'candidate_entity_id']].drop_duplicates()
        pos['label'] = 1

        df_features = df_features.merge(
            pos, on=['source1_entity_id', 'candidate_entity_id'], how='left'
        )
        df_features['label'] = df_features['label'].fillna(0).astype(int)
        return df_features

    def train_cv(self, df_train):
        logger.info("Training LightGBM model with GroupKFold on Country...")

        X = df_train[self.features]
        y = df_train['label']
        groups = df_train['country']
        n_groups = groups.nunique()

        # GroupKFold requires n_splits <= number of groups. Training only has
        # US + India (2 groups), so a hard-coded n_splits=5 crashes. Cap it and
        # fall back to plain KFold if there is only a single group.
        if n_groups >= 2:
            n_splits = min(5, n_groups)
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(X, y, groups)
            logger.info(f"  Using GroupKFold with n_splits={n_splits} ({n_groups} countries)")
        else:
            splitter = KFold(n_splits=5, shuffle=True, random_state=42)
            split_iter = splitter.split(X, y)
            logger.info("  Only one country group; falling back to KFold(5)")

        oof_preds = np.zeros(len(df_train))

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.05,
            'max_depth': 6,
            'num_leaves': 31,
            'n_estimators': 300,
            'verbose': -1
        }

        self.models = []
        for fold, (train_idx, val_idx) in enumerate(split_iter):
            logger.info(f"  Fold {fold + 1}")
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(50, verbose=False)])

            oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
            self.models.append(model)

        return oof_preds

    def predict(self, df_test):
        # Ensemble every fold instead of relying on a single fold's model.
        if not self.models:
            raise RuntimeError("Model has not been trained; call train_cv first.")
        preds = np.mean(
            [m.predict_proba(df_test[self.features])[:, 1] for m in self.models],
            axis=0
        )
        return preds

    def optimize_threshold(self, df_train, oof_preds, n_total_s1, n_missing_singletons):
        """Grid-search the threshold that maximises macro-averaged F0.5.

        The metric is macro-averaged over *every* S1 entity, so entities that
        blocking never produced a candidate for still count:
          - missing singletons (no true match)  -> score 1.0 (empty is correct)
          - missing entities that DO have matches -> score 0.0 (recall 0)
        These contribute a fixed amount (n_missing_singletons) independent of
        the threshold; ignoring them (as the original code did) inflates the
        local score and mis-tunes the threshold.
        """
        logger.info("Optimizing threshold for F0.5...")

        s1 = df_train['source1_entity_id'].values
        label = df_train['label'].values.astype(np.int64)
        prob = np.asarray(oof_preds)

        best_thresh, best_score = 0.5, -1.0

        for thresh in np.arange(0.5, 0.95, 0.05):
            pred = (prob > thresh).astype(np.int64)
            agg = pd.DataFrame({
                'e': s1,
                'pred': pred,
                'tp': pred * label,
                'actual': label,
            }).groupby('e', sort=False).agg(
                tp=('tp', 'sum'),
                pred_pos=('pred', 'sum'),
                actual_pos=('actual', 'sum'),
            )

            tp = agg['tp'].to_numpy(dtype=float)
            pred_pos = agg['pred_pos'].to_numpy(dtype=float)
            actual_pos = agg['actual_pos'].to_numpy(dtype=float)

            precision = np.where(pred_pos > 0, tp / np.maximum(pred_pos, 1), 0.0)
            recall = np.where(actual_pos > 0, tp / np.maximum(actual_pos, 1), 0.0)
            denom = 0.25 * precision + recall
            f05 = np.where(denom > 0, (1.25 * precision * recall) / denom, 0.0)

            # Singletons that had candidates: 1.0 iff we predicted nothing.
            singleton = actual_pos == 0
            f05 = np.where(singleton, np.where(pred_pos == 0, 1.0, 0.0), f05)

            macro_sum = f05.sum() + n_missing_singletons  # missing matched -> +0
            macro_f05 = macro_sum / n_total_s1
            logger.info(f"  Threshold {thresh:.2f}: F0.5 = {macro_f05:.4f}")

            if macro_f05 > best_score:
                best_score = macro_f05
                best_thresh = thresh

        logger.info(f"Best Threshold: {best_thresh:.2f} (Score: {best_score:.4f})")
        return best_thresh
