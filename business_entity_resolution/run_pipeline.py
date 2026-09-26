"""
End-to-End Runner Script for Business Entity Resolution.
Orchestrates:
  1. Stratified 80/20 train/validation split
  2. Country-partitioned 3-gram character TF-IDF sparse blocking
  3. Pairwise RapidFuzz feature extraction
  4. LightGBM classifier training
  5. Macro F_0.5 threshold sweep (0.70 - 0.95)
  6. Bipartite conflict resolution & singleton isolation
  7. Exporting candidate_pairs.tsv and matching_results.tsv
  8. Output verification against competition validator & protocol checks
"""

import os
import sys
import argparse
import time

# Ensure src is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_dataset, load_all_s1_ids, create_stratified_val_split, build_entity_lookup
from src.blocking import CountryPartitionedBlocker, export_candidate_pairs, evaluate_blocking
from src.features import build_pair_dataset, FEATURE_NAMES
from src.train_eval import LightGBMMatcher, calculate_macro_f05
from src.postprocess import resolve_bipartite_conflicts, export_matching_results, validate_results_format


def find_default_dataset_dir():
    """Detect dataset path whether running inside business_entity_resolution or repository root."""
    candidate_paths = [
        "dataset",
        os.path.join("..", "student_resource", "dataset"),
        os.path.join("student_resource", "dataset"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "student_resource", "dataset")
    ]
    for p in candidate_paths:
        if os.path.exists(p) and (os.path.exists(os.path.join(p, "train")) or os.path.exists(os.path.join(p, "test"))):
            return os.path.abspath(p)
    return "dataset"


def main():
    parser = argparse.ArgumentParser(description="End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--dataset-dir", type=str, default=find_default_dataset_dir(),
                        help="Path to dataset directory containing train/ and test/")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Path to directory where candidate_pairs.tsv and matching_results.tsv are saved")
    parser.add_argument("--mode", type=str, choices=["val", "test", "full"], default="full",
                        help="'val' (hold-out evaluation), 'test' (test predictions), or 'full' (both)")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Subsample N records for rapid prototyping/debugging (default: None for full data)")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of nearest candidates to retrieve per source file (default: 3)")

    args = parser.parse_args()

    start_time = time.time()
    print("=" * 70)
    print("      AMAZON ML CHALLENGE 2026: BUSINESS ENTITY RESOLUTION")
    print("=" * 70)
    print(f"Dataset Directory: {args.dataset_dir}")
    print(f"Output Directory:  {args.output_dir}")
    print(f"Pipeline Mode:     {args.mode}")
    print(f"Sample Size:       {args.sample_size or 'FULL DATASET'}")
    print(f"Top-K per source:  {args.top_k}")
    print("=" * 70)

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # STEP 1: LOAD TRAINING DATA & CREATE STRATIFIED VAL SPLIT
    # =========================================================================
    print("\n>>> STEP 1: Loading Training Data & Building Stratified Hold-out Split")
    s1_train, s2_train, s3_train, gt_dict, gt_df = load_dataset(
        args.dataset_dir, split="train", nrows=args.sample_size
    )

    # Build fast O(1) lookups
    s1_lookup = build_entity_lookup(s1_train)
    target_lookup_train = {}
    target_lookup_train.update(build_entity_lookup(s2_train))
    target_lookup_train.update(build_entity_lookup(s3_train))

    train_s1_ids, val_s1_ids = create_stratified_val_split(
        s1_train, gt_dict, val_ratio=0.20, random_state=42
    )

    # =========================================================================
    # STEP 2: BLOCKING (CANDIDATE GENERATION)
    # =========================================================================
    print("\n>>> STEP 2: Executing Country-Partitioned Character 3-Gram Sparse Indexing")
    blocker = CountryPartitionedBlocker(
        top_k_per_source=args.top_k,
        min_similarity=0.05
    )
    train_candidates = blocker.generate_candidates(s1_train, s2_train, s3_train)

    # Evaluate blocking quality on validation subset
    val_gt_dict = {sid: gt_dict.get(sid, set()) for sid in val_s1_ids}
    recall_ceiling, avg_cands, reduction_ratio = evaluate_blocking(
        train_candidates, val_gt_dict,
        num_s1=len(val_s1_ids),
        num_s2=len(s2_train),
        num_s3=len(s3_train)
    )

    # =========================================================================
    # STEP 3: PAIRWISE FEATURE EXTRACTION
    # =========================================================================
    print("\n>>> STEP 3: Feature Engineering with RapidFuzz Metrics")
    # For massive datasets (1.7M train entities), sample up to 100k for training & 25k for val
    # to achieve instant LightGBM convergence without spending hours on CPU string comparisons
    max_train_entities = 100000
    if len(train_s1_ids) > max_train_entities:
        import random
        random.seed(42)
        train_feat_ids = set(random.sample(list(train_s1_ids), max_train_entities))
        print(f"  Subsampling {len(train_feat_ids):,} / {len(train_s1_ids):,} S1 entities for rapid model training...")
    else:
        train_feat_ids = train_s1_ids

    max_val_entities = 25000
    if len(val_s1_ids) > max_val_entities:
        import random
        random.seed(42)
        val_feat_ids = set(random.sample(list(val_s1_ids), max_val_entities))
        print(f"  Subsampling {len(val_feat_ids):,} / {len(val_s1_ids):,} S1 entities for validation threshold sweep...")
    else:
        val_feat_ids = val_s1_ids

    print("  Extracting features for training pairs...")
    X_train, y_train, train_pairs = build_pair_dataset(
        train_candidates, s1_lookup, target_lookup_train,
        s1_ids_subset=train_feat_ids, gt_dict=gt_dict
    )
    print(f"  Train set: {X_train.shape[0]:,} pairs, {X_train.shape[1]} features")

    print("  Extracting features for validation pairs...")
    X_val, y_val, val_pairs = build_pair_dataset(
        train_candidates, s1_lookup, target_lookup_train,
        s1_ids_subset=val_feat_ids, gt_dict=gt_dict
    )
    print(f"  Val set:   {X_val.shape[0]:,} pairs, {X_val.shape[1]} features")

    # =========================================================================
    # STEP 4: TRAIN LIGHTGBM CLASSIFIER & SWEEP THRESHOLD
    # =========================================================================
    print("\n>>> STEP 4: Training LightGBM Binary Classifier & Calibrating Threshold")
    matcher = LightGBMMatcher(n_estimators=300, learning_rate=0.05, num_leaves=31)
    matcher.train(X_train, y_train, feature_names=FEATURE_NAMES)

    val_probs = matcher.predict_probabilities(X_val)
    best_threshold = matcher.optimize_threshold(
        val_pairs, val_probs, gt_dict, val_s1_ids,
        threshold_range=(0.70, 0.96, 0.01)
    )

    # =========================================================================
    # STEP 5: VALIDATION POSTPROCESSING & SCORE AUDIT
    # =========================================================================
    print("\n>>> STEP 5: Bipartite Conflict Resolution on Hold-out Validation Split")
    val_resolved = resolve_bipartite_conflicts(val_pairs, val_probs, best_threshold, list(val_s1_ids))
    val_macro_f05, val_singleton_acc, val_match_f05 = calculate_macro_f05(
        val_resolved, gt_dict, val_s1_ids
    )

    print("\n" + "=" * 65)
    print("GOAL VERIFICATION PROTOCOL AUDIT:")
    print(f"  1. Candidate Recall Ceiling:   {recall_ceiling:.2%} (Target >= 98.5%)")
    print(f"     Average Candidates / S1:    {avg_cands:.2f} (Target <= 4.0)")
    print(f"  2. Threshold Peak Range:       {best_threshold:.2f} (Standard: 0.82 - 0.89)")
    print(f"  3. Singleton Accuracy:         {val_singleton_acc:.2%} (Target >= 97.0%)")
    print(f"  --> Final Local Macro F_0.5:   {val_macro_f05:.4f}")
    print("=" * 65)

    # =========================================================================
    # STEP 6: TEST SET INFERENCE & SUBMISSION GENERATION
    # =========================================================================
    if args.mode in ["test", "full"]:
        test_dir = os.path.join(args.dataset_dir, "test")
        test_s1_file = os.path.join(test_dir, "test_source1.tsv")
        if os.path.exists(test_dir) and os.path.exists(test_s1_file):
            print("\n>>> STEP 6: Generating Final Test Submissions for Leaderboard")
            # Load all required S1 IDs to guarantee complete submission rows
            all_required_s1_ids = load_all_s1_ids(test_s1_file)
            print(f"  Total required test S1 entities: {len(all_required_s1_ids):,}")

            s1_test, s2_test, s3_test, _, _ = load_dataset(
                args.dataset_dir, split="test", nrows=args.sample_size
            )

            test_s1_lookup = build_entity_lookup(s1_test)
            target_lookup_test = {}
            target_lookup_test.update(build_entity_lookup(s2_test))
            target_lookup_test.update(build_entity_lookup(s3_test))

            # 6A. Generate test candidates
            print("  Generating candidate pairs for test set...")
            test_candidates = blocker.generate_candidates(s1_test, s2_test, s3_test)

            candidate_output_path = os.path.join(args.output_dir, "candidate_pairs.tsv")
            export_candidate_pairs(test_candidates, candidate_output_path, all_required_s1_ids)

            # 6B. Feature extraction on test candidates
            print("  Extracting features on test candidate pairs...")
            X_test, _, test_pairs = build_pair_dataset(
                test_candidates, test_s1_lookup, target_lookup_test
            )
            print(f"  Extracted features for {len(test_pairs):,} test pairs.")

            # 6C. Predict match probabilities
            if len(test_pairs) > 0:
                print("  Scoring test pairs using trained LightGBM model...")
                test_probs = matcher.predict_probabilities(X_test)
            else:
                test_probs = []

            # 6D. Bipartite conflict resolution & final matches export
            print(f"  Resolving conflicts at optimal threshold {best_threshold:.2f}...")
            test_resolved = resolve_bipartite_conflicts(
                test_pairs, test_probs, best_threshold, all_required_s1_ids
            )

            matching_output_path = os.path.join(args.output_dir, "matching_results.tsv")
            export_matching_results(test_resolved, matching_output_path, all_required_s1_ids)

            # 6E. Validate final submission format
            validate_results_format(matching_output_path, set(all_required_s1_ids))
        else:
            print(f"\nTest directory not found at: {test_dir}. Skipping test set generation.")

    elapsed = time.time() - start_time
    print(f"\nPipeline execution completed in {elapsed/60:.2f} minutes.")


if __name__ == "__main__":
    main()
