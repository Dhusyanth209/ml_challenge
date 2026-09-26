"""
Method 5: High-Recall Multi-Key Lexical Cascade + LightGBM + Global Hungarian Assignment
Top-level runner script for Amazon ML Challenge 2026: Business Entity Resolution.
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baseline_loader import load_train_data, create_stratified_split
from src.blocking_m5 import (
    run_high_recall_cascade_blocking,
    save_candidate_pairs_m5,
    evaluate_blocking_m5
)
from src.features_m5 import build_m5_feature_dataset, FEATURE_NAMES_M5
from src.train_lgbm import LightGBMEntityMatcher, evaluate_macro_f05
from src.postprocess_m5 import resolve_hungarian_bipartite_matching, save_hungarian_matches
from src.evaluate_baseline import validate_tsv_structure


def find_default_dataset_dir():
    candidates = [
        r"D:\ML challenge\student_resource\dataset",
        os.path.join("..", "student_resource", "dataset"),
        os.path.join("student_resource", "dataset"),
        "dataset"
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "train")):
            return os.path.abspath(p)
    return r"D:\ML challenge\student_resource\dataset"


def build_quick_lookup(df):
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.entity_id] = {
            "name": getattr(row, "business_name", ""),
            "addr": getattr(row, "business_address", "")
        }
    return lookup


def print_5gen_scorecard(m5_metrics: dict,
                         m1=(0.8520, 0.9356, 0.9351, 0.6986, 0.70, "30.95s", "99.9839%", "98.18%"),
                         m2=(0.9732, 0.9169, 0.9945, 0.9211, 0.93, "32.17s", "99.9866%", "98.18%"),
                         m3=(0.9773, 0.9240, 0.9937, 0.9349, 0.70, "29.99s", "99.9839%", "98.18%"),
                         m4=(0.9626, 0.8893, 0.9877, 0.9037, 0.70, "136.41s", "99.9839%", "100.00%")):
    """Prints the comprehensive 5-generation benchmark scorecard."""
    print("\n" + "=" * 115)
    print("                 AMAZON ML CHALLENGE 2026: 5-GENERATION BENCHMARK EVOLUTION")
    print("=" * 115)
    hdr = f"{'Evaluation Metric':<26} | {'M1 Baseline':<14} | {'M2 LightGBM':<14} | {'M3 Bipartite':<14} | {'M4 Dense+FAISS':<15} | {'M5 Multi-Cascade':<16}"
    print(hdr)
    print("-" * 115)

    print(f"{'Candidate Recall Ceiling':<26} | {m1[1]:<14.2%} | {m2[1]:<14.2%} | {m3[1]:<14.2%} | {m4[1]:<15.2%} | {m5_metrics['recall_ceiling']:<16.2%}")
    print(f"{'Reduction Ratio':<26} | {m1[6]:<14} | {m2[6]:<14} | {m3[6]:<14} | {m4[6]:<15} | {m5_metrics['reduction_ratio']:<16.6%}")
    print(f"{'Optimal Cutoff Threshold':<26} | {m1[4]:<14.2f} | {m2[4]:<14.2f} | {m3[4]:<14.2f} | {m4[4]:<15.2f} | {m5_metrics['optimal_threshold']:<16.2f}")
    print(f"{'Singleton Accuracy':<26} | {m1[7]:<14} | {m2[7]:<14} | {m3[7]:<14} | {m4[7]:<15} | {m5_metrics['singleton_acc']:<16.2%}")
    print(f"{'Macro Precision':<26} | {m1[2]:<14.4f} | {m2[2]:<14.4f} | {m3[2]:<14.4f} | {m4[2]:<15.4f} | {m5_metrics['macro_prec']:<16.4f}")
    print(f"{'Macro Recall':<26} | {m1[3]:<14.4f} | {m2[3]:<14.4f} | {m3[3]:<14.4f} | {m4[3]:<15.4f} | {m5_metrics['macro_rec']:<16.4f}")
    print("-" * 115)
    print(f"{'FINAL MACRO F_0.5 SCORE':<26} | {m1[0]:<14.4f} | {m2[0]:<14.4f} | {m3[0]:<14.4f} | {m4[0]:<15.4f} | {m5_metrics['macro_f05']:<16.4f}")
    print("=" * 115)
    print(f"{'Total Execution Time':<26} | {m1[5]:<14} | {m2[5]:<14} | {m3[5]:<14} | {m4[5]:<15} | {m5_metrics['elapsed_time']:.2f}s ({m5_metrics['elapsed_time']/60:.1f} min)")
    print("=" * 115)


def main():
    parser = argparse.ArgumentParser(
        description="Method 5: High-Recall Multi-Key Lexical Cascade + LightGBM + Global Hungarian Assignment"
    )
    parser.add_argument("--dataset-dir", type=str, default=find_default_dataset_dir())
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--top-k", type=int, default=3, help="Top candidates per source (default: 3)")

    args = parser.parse_args()
    sample_size = args.sample_size if args.sample_size > 0 else None
    start_time = time.time()

    print("=" * 85)
    print("  AMAZON ML CHALLENGE 2026: METHOD 5 (LEADERBOARD PRODUCTION PIPELINE)")
    print("  High-Recall Multi-Key Cascade + 18-Feature LightGBM + Global Hungarian Assignment")
    print("=" * 85)
    print(f"Dataset Path:  {args.dataset_dir}")
    print(f"Output Path:   {args.output_dir}")
    print(f"Sample Size:   {sample_size if sample_size else 'FULL DATASET'}")
    print(f"Top-K/Source:  {args.top_k}")
    print("=" * 85)

    os.makedirs(args.output_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: LOAD DATA & STRATIFIED SPLIT
    # -------------------------------------------------------------------------
    print("\n>>> STEP 1: Loading Dataset & Constructing Stratified 80/20 Validation Split")
    s1_df, s2_df, s3_df, gt_dict = load_train_data(args.dataset_dir, sample_size=sample_size)

    s1_lookup = build_quick_lookup(s1_df)
    target_lookup = {}
    target_lookup.update(build_quick_lookup(s2_df))
    target_lookup.update(build_quick_lookup(s3_df))

    train_s1_ids, val_s1_ids = create_stratified_split(s1_df, gt_dict, val_ratio=0.20, random_state=42)

    # -------------------------------------------------------------------------
    # STEP 2: HIGH-RECALL MULTI-KEY CASCADE BLOCKING
    # -------------------------------------------------------------------------
    print("\n>>> STEP 2: High-Recall Multi-Key Cascade Blocking (TF-IDF + Brand Core + Address Pin/Num)")
    candidates_dict = run_high_recall_cascade_blocking(s1_df, s2_df, s3_df, top_k_per_source=args.top_k)

    all_s1_ids = list(s1_df["entity_id"].values)
    cand_file = os.path.join(args.output_dir, "candidate_pairs.tsv")
    save_candidate_pairs_m5(candidates_dict, cand_file, all_s1_ids)

    recall_ceiling, avg_cands, reduction_ratio = evaluate_blocking_m5(
        candidates_dict, gt_dict, val_s1_ids,
        num_s2=len(s2_df), num_s3=len(s3_df)
    )

    # -------------------------------------------------------------------------
    # STEP 3: ADVANCED 18-FEATURE ENGINEERING
    # -------------------------------------------------------------------------
    print("\n>>> STEP 3: Domain Feature Engineering (RapidFuzz + Pin Code + Containment + Number Overlap)")
    print(f"  Extracting {len(FEATURE_NAMES_M5)} features for training candidate pairs...")
    X_train, y_train, train_pairs = build_m5_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=train_s1_ids, gt_dict=gt_dict
    )
    print(f"  Train set: {X_train.shape[0]:,} pairs, {X_train.shape[1]} features")

    print(f"  Extracting {len(FEATURE_NAMES_M5)} features for validation candidate pairs...")
    X_val, y_val, val_pairs = build_m5_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=val_s1_ids, gt_dict=gt_dict
    )
    print(f"  Val set:   {X_val.shape[0]:,} pairs, {X_val.shape[1]} features")

    # -------------------------------------------------------------------------
    # STEP 4: TRAIN LIGHTGBM CLASSIFIER & SWEEP THRESHOLD
    # -------------------------------------------------------------------------
    print("\n>>> STEP 4: Training LightGBM Binary Classifier & Sweeping Threshold")
    matcher = LightGBMEntityMatcher(n_estimators=400, learning_rate=0.04, num_leaves=35)
    matcher.fit(X_train, y_train, feature_names=FEATURE_NAMES_M5)

    val_probs = matcher.predict_proba(X_val)
    best_threshold, best_metrics = matcher.sweep_thresholds(
        val_pairs, val_probs, gt_dict, val_s1_ids
    )

    # -------------------------------------------------------------------------
    # STEP 5: GLOBAL HUNGARIAN BIPARTITE CONFLICT RESOLUTION
    # -------------------------------------------------------------------------
    print(f"\n>>> STEP 5: Global Hungarian Assignment at Threshold {best_threshold:.2f}")
    X_all, _, all_pairs = build_m5_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=set(all_s1_ids), gt_dict=None
    )
    all_probs = matcher.predict_proba(X_all)

    # Enforce global 1-to-1 bipartite constraint via Hungarian algorithm
    final_matches = resolve_hungarian_bipartite_matching(all_pairs, all_probs, best_threshold, all_s1_ids)

    # Re-evaluate validation set metrics after Hungarian resolution
    val_hungarian = {sid: final_matches.get(sid, []) for sid in val_s1_ids}
    post_f05, post_prec, post_rec, post_s_acc = evaluate_macro_f05(val_hungarian, gt_dict, val_s1_ids)

    match_file = os.path.join(args.output_dir, "matching_results.tsv")
    save_hungarian_matches(final_matches, match_file, all_s1_ids)

    validate_tsv_structure(match_file, cand_file, set(all_s1_ids))

    elapsed = time.time() - start_time
    m5_scorecard = {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "optimal_threshold": best_threshold,
        "singleton_acc": post_s_acc,
        "macro_prec": post_prec,
        "macro_rec": post_rec,
        "macro_f05": post_f05,
        "elapsed_time": elapsed
    }
    print_5gen_scorecard(m5_scorecard)


if __name__ == "__main__":
    main()
