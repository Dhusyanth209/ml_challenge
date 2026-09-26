"""
Method 3: Multi-Index BM25/LSH Blocking + GBDT + Global Bipartite Conflict Resolution
Top-level runner script for Amazon ML Challenge 2026: Business Entity Resolution.
"""

import os
import sys
import argparse
import time

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baseline_loader import load_train_data, create_stratified_split
from src.blocking_m3 import (
    run_adaptive_dual_blocking,
    save_candidate_pairs_m3,
    evaluate_blocking_m3
)
from src.features_m3 import build_m3_feature_dataset, FEATURE_NAMES_M3
from src.train_lgbm import LightGBMEntityMatcher, evaluate_macro_f05
from src.postprocess_m3 import resolve_global_bipartite_matching, save_bipartite_matches
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
    """Builds O(1) field lookup dictionary."""
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.entity_id] = {
            "name": getattr(row, "business_name", ""),
            "addr": getattr(row, "business_address", "")
        }
    return lookup


def print_full_evolution_scorecard(m3_metrics: dict,
                                   m1_metrics: tuple = (0.8520, 0.9356, 0.9351, 0.6986, 0.70),
                                   m2_metrics: tuple = (0.9732, 0.9169, 0.9945, 0.9211, 0.93)):
    """Displays 3-generation progress scorecard."""
    print("\n" + "=" * 80)
    print("      AMAZON ML CHALLENGE 2026: 3-GENERATION BENCHMARK EVOLUTION")
    print("=" * 80)
    print(f"{'Evaluation Metric':<28} | {'Method 1 (Baseline)':<20} | {'Method 2 (Upgraded)':<20} | {'Method 3 (SOTA)':<18}")
    print("-" * 80)
    print(f"{'Candidate Recall Ceiling':<28} | {m1_metrics[1]:<20.2%} | {m2_metrics[1]:<20.2%} | {m3_metrics['recall_ceiling']:<18.2%}")
    print(f"{'Reduction Ratio':<28} | {'99.9839%':<20} | {'99.9866%':<20} | {m3_metrics['reduction_ratio']:<18.6%}")
    print(f"{'Optimal Cutoff Threshold':<28} | {m1_metrics[4]:<20.2f} | {m2_metrics[4]:<20.2f} | {m3_metrics['optimal_threshold']:<18.2f}")
    print(f"{'Singleton Accuracy':<28} | {'98.18%':<20} | {'98.18%':<20} | {m3_metrics['singleton_acc']:<18.2%}")
    print(f"{'Macro Precision':<28} | {m1_metrics[2]:<20.4f} | {m2_metrics[2]:<20.4f} | {m3_metrics['macro_prec']:<18.4f}")
    print(f"{'Macro Recall':<28} | {m1_metrics[3]:<20.4f} | {m2_metrics[3]:<20.4f} | {m3_metrics['macro_rec']:<18.4f}")
    print("-" * 80)
    print(f"{'FINAL MACRO F_0.5 SCORE':<28} | {m1_metrics[0]:<20.4f} | {m2_metrics[0]:<20.4f} | {m3_metrics['macro_f05']:<18.4f}")
    print("=" * 80)
    print(f"Total Execution Time: {m3_metrics['elapsed_time']:.2f} seconds ({m3_metrics['elapsed_time']/60:.2f} min)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Method 3: Multi-Index BM25/LSH Blocking + GBDT + Global Bipartite Conflict Resolution"
    )
    parser.add_argument("--dataset-dir", type=str, default=find_default_dataset_dir(),
                        help="Path to dataset directory")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save candidate_pairs.tsv and matching_results.tsv")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Sample size for evaluation (default: 5000; set 0 for full data)")

    args = parser.parse_args()
    sample_size = args.sample_size if args.sample_size and args.sample_size > 0 else None

    start_time = time.time()
    print("=" * 75)
    print("  AMAZON ML CHALLENGE 2026: METHOD 3 (LEADERBOARD ARCHITECTURE)")
    print("  Multi-Index BM25/LSH Blocking + GBDT + Global Bipartite Conflict Resolution")
    print("=" * 75)
    print(f"Dataset Path:  {args.dataset_dir}")
    print(f"Output Path:   {args.output_dir}")
    print(f"Sample Size:   {sample_size if sample_size else 'FULL DATASET'}")
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
    # STEP 2: ADAPTIVE DUAL-INDEX BLOCKING (TF-IDF + SHINGLING JACCARD)
    # -------------------------------------------------------------------------
    print("\n>>> STEP 2: Adaptive Dual-Index Blocking (Char 3-Gram TF-IDF + Token Shingling)")
    candidates_dict = run_adaptive_dual_blocking(s1_df, s2_df, s3_df)

    all_s1_ids = list(s1_df["entity_id"].values)
    candidate_output_file = os.path.join(args.output_dir, "candidate_pairs.tsv")
    save_candidate_pairs_m3(candidates_dict, candidate_output_file, all_s1_ids)

    # Audit blocking metrics on validation set
    recall_ceiling, avg_cands, reduction_ratio = evaluate_blocking_m3(
        candidates_dict, gt_dict, val_s1_ids,
        num_s2=len(s2_df), num_s3=len(s3_df)
    )

    # -------------------------------------------------------------------------
    # STEP 3: ADVANCED 15-FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n>>> STEP 3: Advanced Pairwise Feature Engineering (Domain Expansions)")
    print("  Extracting 15 feature signals for training candidate pairs...")
    X_train, y_train, train_pairs = build_m3_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=train_s1_ids, gt_dict=gt_dict
    )
    print(f"  Train set: {X_train.shape[0]:,} pairs, {X_train.shape[1]} features")

    print("  Extracting 15 feature signals for validation candidate pairs...")
    X_val, y_val, val_pairs = build_m3_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=val_s1_ids, gt_dict=gt_dict
    )
    print(f"  Val set:   {X_val.shape[0]:,} pairs, {X_val.shape[1]} features")

    # -------------------------------------------------------------------------
    # STEP 4: TRAIN LIGHTGBM CLASSIFIER & SWEEP THRESHOLD
    # -------------------------------------------------------------------------
    print("\n>>> STEP 4: Training LightGBM Binary Classifier & Sweeping Threshold")
    matcher = LightGBMEntityMatcher(n_estimators=300, learning_rate=0.05, num_leaves=31)
    matcher.fit(X_train, y_train, feature_names=FEATURE_NAMES_M3)

    val_probs = matcher.predict_proba(X_val)
    best_threshold, best_metrics = matcher.sweep_thresholds(
        val_pairs, val_probs, gt_dict, val_s1_ids,
        threshold_range=None
    )

    # -------------------------------------------------------------------------
    # STEP 5: GLOBAL BIPARTITE CONFLICT RESOLUTION
    # -------------------------------------------------------------------------
    print(f"\n>>> STEP 5: Global Bipartite Conflict Resolution at Threshold {best_threshold:.2f}")
    # Score all candidate pairs (train + val)
    X_all, _, all_pairs = build_m3_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=set(all_s1_ids), gt_dict=None
    )
    all_probs = matcher.predict_proba(X_all)

    # Enforce bipartite 1-to-1 constraint
    final_matches = resolve_global_bipartite_matching(all_pairs, all_probs, best_threshold, all_s1_ids)

    # Re-evaluate postprocessed validation metrics
    val_bipartite = {sid: final_matches.get(sid, []) for sid in val_s1_ids}
    post_f05, post_prec, post_rec, post_s_acc = evaluate_macro_f05(val_bipartite, gt_dict, val_s1_ids)

    matching_output_file = os.path.join(args.output_dir, "matching_results.tsv")
    save_bipartite_matches(final_matches, matching_output_file, all_s1_ids)

    # Validate output submission structure
    validate_tsv_structure(matching_output_file, candidate_output_file, set(all_s1_ids))

    elapsed = time.time() - start_time
    m3_scorecard = {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "optimal_threshold": best_threshold,
        "singleton_acc": post_s_acc,
        "macro_prec": post_prec,
        "macro_rec": post_rec,
        "macro_f05": post_f05,
        "elapsed_time": elapsed
    }
    print_full_evolution_scorecard(m3_scorecard)


if __name__ == "__main__":
    main()
