import os
import sys
import logging
import pandas as pd
from data_loader import load_data
from preprocessing import preprocess_dataframe
from blocking import SemanticBlocker
from feature_engineering import compute_features
from model import MatchingModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    dataset_dir = os.path.join(base_dir, 'dataset')
    output_dir = os.path.join(base_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    s1_train, s2_train, s3_train, gt_train = load_data(dataset_dir, split="train")
    
    # 2. Preprocess
    logger.info("Preprocessing Train data...")
    s1_train = preprocess_dataframe(s1_train)
    s2_train = preprocess_dataframe(s2_train)
    s3_train = preprocess_dataframe(s3_train)
    
    # 3. Blocking
    blocker = SemanticBlocker(top_k=10)
    train_candidates = blocker.fit_and_search(s1_train, s2_train, s3_train)
    
    # 4. Feature Engineering
    s23_pool_train = pd.concat([s2_train, s3_train], ignore_index=True)
    train_features = compute_features(train_candidates, s1_train, s23_pool_train)
    
    # 5. Model Training & Optimization
    model = MatchingModel()
    train_features = model.prepare_training_data(train_features, gt_train)
    oof_preds = model.train_cv(train_features)

    # The F0.5 metric is macro-averaged over *every* S1 entity, including those
    # blocking produced no candidate for. Those absent entities never appear in
    # train_features, so we account for them explicitly during tuning: a missing
    # singleton scores 1.0 (empty prediction is correct), a missing entity with
    # true matches scores 0.0.
    n_total_s1 = s1_train['entity_id'].nunique()
    present_ids = set(train_features['source1_entity_id'].unique())
    has_match = gt_train['matched_entity_ids'].fillna('').astype(str).str.strip() != ''
    matched_s1_ids = set(gt_train.loc[has_match, 'source1_entity_id'])
    missing_ids = set(s1_train['entity_id']) - present_ids
    n_missing_singletons = sum(1 for x in missing_ids if x not in matched_s1_ids)
    logger.info(f"S1 entities: {n_total_s1} total, {len(missing_ids)} without candidates "
                f"({n_missing_singletons} of them singletons)")

    best_thresh = model.optimize_threshold(
        train_features, oof_preds, n_total_s1, n_missing_singletons
    )
    
    # ==========================
    # TEST INFERENCE
    # ==========================
    s1_test, s2_test, s3_test = load_data(dataset_dir, split="test")
    
    logger.info("Preprocessing Test data...")
    s1_test = preprocess_dataframe(s1_test)
    s2_test = preprocess_dataframe(s2_test)
    s3_test = preprocess_dataframe(s3_test)
    
    # We re-initialize the blocker for test data pool
    test_blocker = SemanticBlocker(top_k=10)
    test_candidates = test_blocker.fit_and_search(s1_test, s2_test, s3_test)
    
    # Save candidate_pairs.tsv
    test_cand_out = test_candidates.groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    # Add empty rows for singletons
    test_cand_out = pd.merge(s1_test[['entity_id']], test_cand_out, left_on='entity_id', right_on='source1_entity_id', how='left')
    test_cand_out['candidate_entity_ids'] = test_cand_out['candidate_entity_id'].fillna('')
    test_cand_out[['entity_id', 'candidate_entity_ids']].rename(columns={'entity_id': 'source1_entity_id'}).to_csv(
        os.path.join(output_dir, 'candidate_pairs.tsv'), sep='\t', index=False
    )
    logger.info("Saved candidate_pairs.tsv")
    
    s23_pool_test = pd.concat([s2_test, s3_test], ignore_index=True)
    test_features = compute_features(test_candidates, s1_test, s23_pool_test)
    
    test_features['pred_prob'] = model.predict(test_features)
    test_features['pred_label'] = (test_features['pred_prob'] > best_thresh).astype(int)
    
    # Generate final matches
    final_matches = test_features[test_features['pred_label'] == 1].groupby('source1_entity_id')['candidate_entity_id'].apply(lambda x: ','.join(x)).reset_index()
    
    # Add singletons
    final_out = pd.merge(s1_test[['entity_id']], final_matches, left_on='entity_id', right_on='source1_entity_id', how='left')
    final_out['matched_entity_ids'] = final_out['candidate_entity_id'].fillna('')
    
    out_path = os.path.join(output_dir, 'matching_results.tsv')
    final_out[['entity_id', 'matched_entity_ids']].rename(columns={'entity_id': 'source1_entity_id'}).to_csv(
        out_path, sep='\t', index=False
    )
    logger.info(f"Saved matching_results.tsv to {out_path}")
    logger.info("Pipeline Complete!")

if __name__ == "__main__":
    main()
