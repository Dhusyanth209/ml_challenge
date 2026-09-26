"""
Method 1: Character 3-Gram TF-IDF Blocking + Heuristic Rule Matching (Baseline)
Top-level runner script for Amazon ML Challenge 2026: Business Entity Resolution.
"""

import os
import sys
import argparse
import time

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baseline_loader import load_train_data, create_stratified_split
from src.baseline_blocking import (
    run_country_partitioned_blocking,
    save_candidate_pairs,
    evaluate_blocking_metrics
)
from src.baseline_matcher import (
    score_candidate_pairs,
    sweep_optimal_threshold,
    apply_matching_threshold,
    save_matching_results
)
from src.evaluate_baseline import (
    validate_tsv_structure,
    print_validation_summary
)


def find_default_dataset_dir():
    """Detect dataset path whether running inside business_entity_resolution or repository root."""
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
    """Builds O(1) dictionary mapping entity_id to {'name': ..., 'addr': ...}."""
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.entity_id] = {
            "name": getattr(row, "business_name", ""),
            "addr": getattr(row, "business_address", "")
        }
    return lookup


def main():
    parser = argparse.ArgumentParser(
        description="Method 1: Character 3-Gram TF-IDF Blocking + Heuristic Rule Matching (Baseline)"
    )
    parser.add_argument("--dataset-dir", type=str, default=find_default_dataset_dir(),
                        help="Path to dataset directory containing train/ and test/")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save candidate_pairs.tsv and matching_results.tsv")
    parser.add_argument("--sample-size", type=int, default=5000,
                        help="Number of S1 sample entities to evaluate (default: 5000; set to 0 or omit for full data)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of nearest candidates per source (default: 3)")

    args = parser.parse_args()
    sample_size = args.sample_size if args.sample_size and args.sample_size > 0 else None

    start_time = time.time()
    print("=" * 70)
    print("  AMAZON ML CHALLENGE 2026: METHOD 1 (BASELINE)")
    print("  Character 3-Gram TF-IDF Blocking + Heuristic Rule Matching")
    print("=" * 70)
    print(f"Dataset Path:    {args.dataset_dir}")
    print(f"Output Path:     {args.output_dir}")
    print(f"Sample Size:     {sample_size if sample_size else 'FULL DATASET'}")
    print(f"Top-K / Source:  {args.top_k}")
    print("=" * 70)

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
    # STEP 2: CHARACTER 3-GRAM TF-IDF BLOCKING (CANDIDATE GENERATION)
    # -------------------------------------------------------------------------
    print("\n>>> STEP 2: Country-Partitioned Character 3-Gram TF-IDF Blocking")
    candidates_dict = run_country_partitioned_blocking(
        s1_df, s2_df, s3_df,
        top_k_per_source=args.top_k,
        min_similarity=0.05
    )

    all_s1_ids = list(s1_df["entity_id"].values)
    candidate_output_file = os.path.join(args.output_dir, "candidate_pairs.tsv")
    save_candidate_pairs(candidates_dict, candidate_output_file, all_s1_ids)

    # Audit blocking metrics on validation set
    recall_ceiling, avg_cands, reduction_ratio = evaluate_blocking_metrics(
        candidates_dict, gt_dict, val_s1_ids,
        num_s2=len(s2_df), num_s3=len(s3_df)
    )

    # -------------------------------------------------------------------------
    # STEP 3: HEURISTIC RULE MATCHING & THRESHOLD GRID SEARCH
    # -------------------------------------------------------------------------
    print("\n>>> STEP 3: Computing Composite Scores & Grid Searching Threshold")
    print("  Calculating pairwise Name & Address Character 3-Gram Cosine Similarities...")
    scored_dict = score_candidate_pairs(candidates_dict, s1_lookup, target_lookup)

    # Threshold grid search on holdout validation split
    best_thresh, best_stats = sweep_optimal_threshold(scored_dict, gt_dict, val_s1_ids)
    opt_f05, opt_prec, opt_rec, opt_singleton_acc = best_stats

    # Apply optimal threshold to generate final predictions
    matched_dict = apply_matching_threshold(scored_dict, best_thresh, all_s1_ids)
    matching_output_file = os.path.join(args.output_dir, "matching_results.tsv")
    save_matching_results(matched_dict, matching_output_file, all_s1_ids)

    # -------------------------------------------------------------------------
    # STEP 4: SUBMISSION VALIDATION & BENCHMARK SCORECARD
    # -------------------------------------------------------------------------
    print("\n>>> STEP 4: Running Format Verification & Generating Benchmark Scorecard")
    validate_tsv_structure(matching_output_file, candidate_output_file, set(all_s1_ids))

    elapsed = time.time() - start_time
    print_validation_summary(
        macro_f05=opt_f05,
        macro_prec=opt_prec,
        macro_recall=opt_rec,
        singleton_acc=opt_singleton_acc,
        optimal_threshold=best_thresh,
        elapsed_seconds=elapsed
    )


if __name__ == "__main__":
    main()
