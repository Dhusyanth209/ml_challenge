"""
Method 2: Multi-Pass Union Blocking + RapidFuzz Feature Engineering + LightGBM Classifier
Top-level runner script for Amazon ML Challenge 2026: Business Entity Resolution.
"""

import os
import sys
import argparse
import time

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baseline_loader import load_train_data, create_stratified_split
from src.blocking_m2 import (
    run_multipass_union_blocking,
    save_candidate_pairs,
    evaluate_blocking_m2
)
from src.features_m2 import build_m2_feature_dataset, FEATURE_NAMES_M2
from src.train_lgbm import LightGBMEntityMatcher, build_final_matches
from src.baseline_matcher import save_matching_results
from src.evaluate_baseline import validate_tsv_structure


def find_default_dataset_dir():
    """Detect dataset path."""
    candidate_paths = [
        r"D:\ML challenge\student_resource\dataset",
        os.path.join("..", "student_resource", "dataset"),
        os.path.join("student_resource", "dataset"),
        "dataset"
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "train")):
            return os.path.abspath(p)
    return r"D:\ML challenge\student_resource\dataset"


def build_quick_lookup(df):
    """Builds fast O(1) dictionary mapping entity_id to fields."""
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.entity_id] = {
            "name": getattr(row, "business_name", ""),
            "addr": getattr(row, "business_address", "")
        }
    return lookup


def print_method_comparison(m2_metrics: dict, m1_f05: float = 0.8520, m1_recall_ceiling: float = 0.9356):
    """Prints side-by-side comparison between Method 1 Baseline and Method 2."""
    print("\n" + "=" * 70)
    print("        METHOD COMPARISON SCORECARD (METHOD 1 vs METHOD 2)")
    print("=" * 70)
    print(f"{'Metric':<30} | {'Method 1 (Baseline)':<20} | {'Method 2 (Upgraded)':<18}")
    print("-" * 70)
    print(f"{'Candidate Recall Ceiling':<30} | {m1_recall_ceiling:<20.2%} | {m2_metrics['recall_ceiling']:<18.2%}")
    print(f"{'Reduction Ratio':<30} | {'99.9839%':<20} | {m2_metrics['reduction_ratio']:<18.6%}")
    print(f"{'Optimal Probability Cutoff':<30} | {'0.70':<20} | {m2_metrics['optimal_threshold']:<18.2f}")
    print(f"{'Singleton Accuracy':<30} | {'98.18%':<20} | {m2_metrics['singleton_acc']:<18.2%}")
    print(f"{'Macro Precision':<30} | {'0.9351':<20} | {m2_metrics['macro_prec']:<18.4f}")
    print(f"{'Macro Recall':<30} | {'0.6986':<20} | {m2_metrics['macro_rec']:<18.4f}")
    print("-" * 70)
    gain = m2_metrics['macro_f05'] - m1_f05
    print(f"{'FINAL MACRO F_0.5':<30} | {m1_f05:<20.4f} | {m2_metrics['macro_f05']:<18.4f} (+{gain:.4f})")
    print("=" * 70)
    print(f"Total Execution Time: {m2_metrics['elapsed_time']:.2f} seconds ({m2_metrics['elapsed_time']/60:.2f} min)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Method 2: Multi-Pass Union Blocking + RapidFuzz Feature Engineering + LightGBM Classifier"
    )
    parser.add_argument("--dataset-dir", type=str, default=find_default_dataset_dir(),
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save candidate_pairs.tsv and matching_results.tsv")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Sample size for evaluation (default: 5000; set 0 for full data)")
    parser.add_argument("--max-candidates", type=int, default=5,
                        help="Maximum candidates per S1 entity (default: 5)")

    args = parser.parse_args()
    sample_size = args.sample_size if args.sample_size and args.sample_size > 0 else None

    start_time = time.time()
    print("=" * 75)
    print("   AMAZON ML CHALLENGE 2026: METHOD 2 (OPTIMIZED PIPELINE)")
    print("   Multi-Pass Union Blocking + RapidFuzz Features + LightGBM Classifier")
    print("=" * 75)
    print(f"Dataset Path:       {args.dataset_dir}")
    print(f"Output Path:        {args.output_dir}")
    print(f"Sample Size:        {sample_size if sample_size else 'FULL DATASET'}")
    print(f"Max Candidates / S1:{args.max_candidates}")
    print("=" * 75)

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD TRAINING DATA & CREATE STRATIFIED VAL SPLIT
    # -------------------------------------------------------------------------
    print("\n>>> STEP 1: Loading Dataset & Constructing Stratified 80/20 Validation Split")
    s1_df, s2_df, s3_df, gt_dict = load_train_data(args.dataset_dir, sample_size=sample_size)

    s1_lookup = build_quick_lookup(s1_df)
    target_lookup = {}
    target_lookup.update(build_quick_lookup(s2_df))
    target_lookup.update(build_quick_lookup(s3_df))

    train_s1_ids, val_s1_ids = create_stratified_split(s1_df, gt_dict, val_ratio=0.20, random_state=42)

    # -------------------------------------------------------------------------
    # STEP 2: MULTI-PASS UNION BLOCKING (CANDIDATE GENERATION)
    # -------------------------------------------------------------------------
    print("\n>>> STEP 2: Executing Multi-Pass Union Blocking (TF-IDF + Prefix/Token Key)")
    candidates_dict = run_multipass_union_blocking(
        s1_df, s2_df, s3_df,
        max_candidates_per_s1=args.max_candidates
    )

    all_s1_ids = list(s1_df["entity_id"].values)
    candidate_output_file = os.path.join(args.output_dir, "candidate_pairs.tsv")
    save_candidate_pairs(candidates_dict, candidate_output_file, all_s1_ids)

    # Audit blocking metrics on validation set
    recall_ceiling, avg_cands, reduction_ratio = evaluate_blocking_m2(
        candidates_dict, gt_dict, val_s1_ids,
        num_s2=len(s2_df), num_s3=len(s3_df)
    )

    # -------------------------------------------------------------------------
    # STEP 3: PAIRWISE FEATURE ENGINEERING WITH RAPIDFUZZ
    # -------------------------------------------------------------------------
    print("\n>>> STEP 3: Feature Engineering with RapidFuzz & Jaro-Winkler Metrics")
    print("  Extracting feature vectors for training candidate pairs...")
    X_train, y_train, train_pairs = build_m2_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=train_s1_ids, gt_dict=gt_dict
    )
    print(f"  Train set: {X_train.shape[0]:,} pairs, {X_train.shape[1]} features")

    print("  Extracting feature vectors for validation candidate pairs...")
    X_val, y_val, val_pairs = build_m2_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=val_s1_ids, gt_dict=gt_dict
    )
    print(f"  Val set:   {X_val.shape[0]:,} pairs, {X_val.shape[1]} features")

    # -------------------------------------------------------------------------
    # STEP 4: TRAIN LIGHTGBM CLASSIFIER & SWEEP THRESHOLD
    # -------------------------------------------------------------------------
    print("\n>>> STEP 4: Training LightGBM Classifier & Sweeping Threshold")
    matcher = LightGBMEntityMatcher(n_estimators=300, learning_rate=0.05, num_leaves=31)
    matcher.fit(X_train, y_train, feature_names=FEATURE_NAMES_M2)

    val_probs = matcher.predict_proba(X_val)
    best_threshold, best_metrics = matcher.sweep_thresholds(
        val_pairs, val_probs, gt_dict, val_s1_ids,
        threshold_range=None
    )
    opt_f05, opt_prec, opt_rec, opt_singleton_acc = best_metrics

    # -------------------------------------------------------------------------
    # STEP 5: PREDICTION EXPORT & FORMAT VALIDATION
    # -------------------------------------------------------------------------
    print("\n>>> STEP 5: Applying Optimal Threshold & Exporting Matching Results")
    # Score all candidate pairs (train + val)
    X_all, _, all_pairs = build_m2_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=set(all_s1_ids), gt_dict=None
    )
    all_probs = matcher.predict_proba(X_all)
    final_matches = build_final_matches(all_pairs, all_probs, best_threshold, all_s1_ids)

    matching_output_file = os.path.join(args.output_dir, "matching_results.tsv")
    save_matching_results(final_matches, matching_output_file, all_s1_ids)

    # Validate output submission structure
    validate_tsv_structure(matching_output_file, candidate_output_file, set(all_s1_ids))

    elapsed = time.time() - start_time
    m2_scorecard = {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "optimal_threshold": best_threshold,
        "singleton_acc": opt_singleton_acc,
        "macro_prec": opt_prec,
        "macro_rec": opt_rec,
        "macro_f05": opt_f05,
        "elapsed_time": elapsed
    }
    print_method_comparison(m2_scorecard, m1_f05=0.8520, m1_recall_ceiling=0.9356)


if __name__ == "__main__":
    main()
