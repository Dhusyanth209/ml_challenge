"""
Method 4: Dense Transformer Bi-Encoder (MiniLM) + FAISS ANN + Calibrated LightGBM Matching.
Top-level runner for Amazon ML Challenge 2026: Business Entity Resolution.
"""

import os
os.environ["HF_HOME"] = r"D:\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"D:\hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"D:\hf_cache"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TEMP"] = r"D:\temp"
os.environ["TMP"] = r"D:\temp"
os.makedirs(r"D:\hf_cache", exist_ok=True)
os.makedirs(r"D:\temp", exist_ok=True)

import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.baseline_loader import load_train_data, create_stratified_split
from src.blocking_m4 import (
    run_dense_blocking,
    save_candidate_pairs_m4,
    evaluate_blocking_m4
)
from src.matcher_m4 import build_m4_feature_dataset, FEATURE_NAMES_M4
from src.train_lgbm import LightGBMEntityMatcher, evaluate_macro_f05
from src.postprocess_m3 import resolve_global_bipartite_matching, save_bipartite_matches
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


def print_4gen_scorecard(m4_metrics: dict,
                         m1=(0.8520, 0.9356, 0.9351, 0.6986, 0.70),
                         m2=(0.9732, 0.9169, 0.9945, 0.9211, 0.93),
                         m3=(0.9773, 0.9240, 0.9937, 0.9349, 0.70)):
    print("\n" + "=" * 100)
    print("        AMAZON ML CHALLENGE 2026: 4-GENERATION BENCHMARK EVOLUTION")
    print("=" * 100)
    hdr = f"{'Metric':<30} | {'M1 Baseline':<14} | {'M2 LightGBM':<14} | {'M3 Bipartite':<14} | {'M4 Dense+FAISS':<14}"
    print(hdr)
    print("-" * 100)

    print(f"{'Candidate Recall Ceiling':<30} | {m1[1]:<14.2%} | {m2[1]:<14.2%} | {m3[1]:<14.2%} | {m4_metrics['recall_ceiling']:<14.2%}")
    print(f"{'Reduction Ratio':<30} | {'99.9839%':<14} | {'99.9866%':<14} | {'99.9839%':<14} | {m4_metrics['reduction_ratio']:<14.6%}")
    print(f"{'Optimal Threshold':<30} | {m1[4]:<14.2f} | {m2[4]:<14.2f} | {m3[4]:<14.2f} | {m4_metrics['optimal_threshold']:<14.2f}")
    print(f"{'Singleton Accuracy':<30} | {'98.18%':<14} | {'98.18%':<14} | {'98.18%':<14} | {m4_metrics['singleton_acc']:<14.2%}")
    print(f"{'Macro Precision':<30} | {m1[2]:<14.4f} | {m2[2]:<14.4f} | {m3[2]:<14.4f} | {m4_metrics['macro_prec']:<14.4f}")
    print(f"{'Macro Recall':<30} | {m1[3]:<14.4f} | {m2[3]:<14.4f} | {m3[3]:<14.4f} | {m4_metrics['macro_rec']:<14.4f}")
    print("-" * 100)
    print(f"{'FINAL MACRO F_0.5':<30} | {m1[0]:<14.4f} | {m2[0]:<14.4f} | {m3[0]:<14.4f} | {m4_metrics['macro_f05']:<14.4f}")
    print("=" * 100)
    print(f"Total Execution Time: {m4_metrics['elapsed_time']:.2f}s ({m4_metrics['elapsed_time']/60:.1f} min)")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Method 4: Dense Bi-Encoder (MiniLM) + FAISS ANN + Calibrated LightGBM"
    )
    parser.add_argument("--dataset-dir", type=str, default=find_default_dataset_dir())
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--model-name", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--top-k", type=int, default=3, help="Top K candidates per source")
    parser.add_argument("--batch-size", type=int, default=256, help="Embedding batch size")
    args = parser.parse_args()

    sample_size = args.sample_size if args.sample_size > 0 else None
    start_time = time.time()

    print("=" * 80)
    print("  AMAZON ML CHALLENGE 2026: METHOD 4 (DENSE TRANSFORMER ARCHITECTURE)")
    print("  Dense Bi-Encoder (MiniLM-L6-v2) + FAISS ANN + LightGBM + Bipartite")
    print("=" * 80)
    print(f"Dataset Path:  {args.dataset_dir}")
    print(f"Output Path:   {args.output_dir}")
    print(f"Sample Size:   {sample_size if sample_size else 'FULL DATASET'}")
    print(f"Model:         {args.model_name}")
    print(f"Top-K/Source:  {args.top_k}")
    print(f"Batch Size:    {args.batch_size}")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # STEP 1: LOAD DATA & STRATIFIED SPLIT
    # =========================================================================
    print("\n>>> STEP 1: Loading Dataset & Constructing Stratified 80/20 Validation Split")
    s1_df, s2_df, s3_df, gt_dict = load_train_data(args.dataset_dir, sample_size=sample_size)

    s1_lookup = build_quick_lookup(s1_df)
    target_lookup = {}
    target_lookup.update(build_quick_lookup(s2_df))
    target_lookup.update(build_quick_lookup(s3_df))

    train_s1_ids, val_s1_ids = create_stratified_split(s1_df, gt_dict, val_ratio=0.20, random_state=42)

    # =========================================================================
    # STEP 2: DENSE BI-ENCODER FAISS BLOCKING
    # =========================================================================
    print("\n>>> STEP 2: Dense Bi-Encoder Blocking (MiniLM-L6-v2 + FAISS IndexFlatIP)")
    candidates_dict = run_dense_blocking(
        s1_df, s2_df, s3_df,
        model_name=args.model_name,
        top_k_per_source=args.top_k,
        batch_size=args.batch_size
    )

    all_s1_ids = list(s1_df["entity_id"].values)
    cand_file = os.path.join(args.output_dir, "candidate_pairs.tsv")
    save_candidate_pairs_m4(candidates_dict, cand_file, all_s1_ids)

    recall_ceiling, avg_cands, reduction_ratio = evaluate_blocking_m4(
        candidates_dict, gt_dict, val_s1_ids,
        num_s2=len(s2_df), num_s3=len(s3_df)
    )

    # =========================================================================
    # STEP 3: HYBRID FEATURE ENGINEERING (DENSE + LEXICAL)
    # =========================================================================
    print("\n>>> STEP 3: Hybrid Feature Engineering (Dense Cosine + RapidFuzz Lexical)")
    print(f"  Extracting {len(FEATURE_NAMES_M4)} features for training pairs...")
    X_train, y_train, train_pairs = build_m4_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=train_s1_ids, gt_dict=gt_dict
    )
    print(f"  Train set: {X_train.shape[0]:,} pairs, {X_train.shape[1]} features")

    print(f"  Extracting {len(FEATURE_NAMES_M4)} features for validation pairs...")
    X_val, y_val, val_pairs = build_m4_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=val_s1_ids, gt_dict=gt_dict
    )
    print(f"  Val set:   {X_val.shape[0]:,} pairs, {X_val.shape[1]} features")

    # =========================================================================
    # STEP 4: TRAIN LIGHTGBM & THRESHOLD SWEEP
    # =========================================================================
    print("\n>>> STEP 4: Training LightGBM Classifier & Sweeping Threshold")
    matcher = LightGBMEntityMatcher(n_estimators=400, learning_rate=0.05, num_leaves=31)
    matcher.fit(X_train, y_train, feature_names=FEATURE_NAMES_M4)

    val_probs = matcher.predict_proba(X_val)
    best_threshold, best_metrics = matcher.sweep_thresholds(
        val_pairs, val_probs, gt_dict, val_s1_ids
    )

    # =========================================================================
    # STEP 5: GLOBAL BIPARTITE CONFLICT RESOLUTION
    # =========================================================================
    print(f"\n>>> STEP 5: Global Bipartite Conflict Resolution at Threshold {best_threshold:.2f}")
    X_all, _, all_pairs = build_m4_feature_dataset(
        candidates_dict, s1_lookup, target_lookup,
        s1_ids_subset=set(all_s1_ids), gt_dict=None
    )
    all_probs = matcher.predict_proba(X_all)

    final_matches = resolve_global_bipartite_matching(all_pairs, all_probs, best_threshold, all_s1_ids)

    # Re-evaluate on validation set after bipartite resolution
    val_bipartite = {sid: final_matches.get(sid, []) for sid in val_s1_ids}
    post_f05, post_prec, post_rec, post_s_acc = evaluate_macro_f05(val_bipartite, gt_dict, val_s1_ids)

    match_file = os.path.join(args.output_dir, "matching_results.tsv")
    save_bipartite_matches(final_matches, match_file, all_s1_ids)

    validate_tsv_structure(match_file, cand_file, set(all_s1_ids))

    elapsed = time.time() - start_time
    m4_scorecard = {
        "recall_ceiling": recall_ceiling,
        "reduction_ratio": reduction_ratio,
        "optimal_threshold": best_threshold,
        "singleton_acc": post_s_acc,
        "macro_prec": post_prec,
        "macro_rec": post_rec,
        "macro_f05": post_f05,
        "elapsed_time": elapsed
    }
    print_4gen_scorecard(m4_scorecard)


if __name__ == "__main__":
    main()
