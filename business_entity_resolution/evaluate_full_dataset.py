"""
Comprehensive Full-Dataset Evaluation Script for Amazon ML Challenge 2026.
Evaluates entity resolution on the official training ground truth across all 2.2M records
(or configurable large sample size) comparing:
  1. Production Pipeline Current (tau = 0.92, max 1 per source)
  2. Balanced Precision (tau = 0.80, multi-match)
  3. Calibrated F_0.5 Optimal (tau = 0.70, multi-match)
"""

import os
import sys
import gc
import time
import argparse
from collections import defaultdict
import numpy as np
import polars as pl

# Ensure business_entity_resolution is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

from src.production_blocking import ProductionBlocker
from src.production_matcher import PrecisionProductionScorer
from src.train_eval import calculate_macro_f05


def load_ground_truth_map(gt_path: str, filter_s1_ids: set = None) -> dict:
    """Loads ground truth into {s1_id: set(matched_ids)}."""
    print(f"[Loader] Reading ground truth from {gt_path}...")
    t0 = time.time()
    gt_df = pl.read_csv(
        gt_path,
        separator="\t",
        schema_overrides={
            "source1_entity_id": pl.String,
            "matched_entity_ids": pl.String
        }
    )
    if filter_s1_ids:
        gt_df = gt_df.filter(pl.col("source1_entity_id").is_in(list(filter_s1_ids)))

    s1_ids = gt_df["source1_entity_id"].to_list()
    matches_raw = gt_df["matched_entity_ids"].to_list()

    gt_dict = {}
    for sid, m_str in zip(s1_ids, matches_raw):
        if m_str and m_str.strip():
            gt_dict[sid] = set(x.strip() for x in m_str.split(",") if x.strip())
        else:
            gt_dict[sid] = set()

    print(f"  Loaded {len(gt_dict):,} ground truth mappings in {time.time() - t0:.2f}s")
    return gt_dict


def compute_metrics(pred_dict: dict, gt_dict: dict, eval_ids: set):
    """Computes Macro Precision, Macro Recall, Singleton Accuracy, and Macro F_0.5."""
    macro_f05, s_acc, non_s_f05 = calculate_macro_f05(pred_dict, gt_dict, eval_ids)
    precs, recs = [], []
    for sid in eval_ids:
        gt_m = gt_dict.get(sid, set())
        pr_m = set(pred_dict.get(sid, []))
        if len(gt_m) == 0:
            continue
        if len(pr_m) == 0:
            recs.append(0.0)
        else:
            tp = len(gt_m & pr_m)
            precs.append(tp / len(pr_m))
            recs.append(tp / len(gt_m))

    mp = float(np.mean(precs)) if precs else 0.0
    mr = float(np.mean(recs)) if recs else 0.0
    return {
        "macro_f05": macro_f05,
        "macro_precision": mp,
        "macro_recall": mr,
        "singleton_accuracy": s_acc,
        "non_singleton_f05": non_s_f05
    }


def main():
    parser = argparse.ArgumentParser(description="Full Dataset Evaluation")
    parser.add_argument("--dataset-dir", type=str, default="/home/raki/Downloads/dataset",
                        help="Path to dataset directory containing train/")
    parser.add_argument("--sample-size", type=int, default=0,
                        help="Number of S1 records to evaluate (0 for full 2.2M dataset)")
    parser.add_argument("--bucket-cap", type=int, default=40,
                        help="Bucket cap for inverted index blocking")
    args = parser.parse_args()

    train_dir = os.path.join(args.dataset_dir, "train")
    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    print("=" * 80)
    print("      AMAZON ML CHALLENGE 2026: FULL-DATASET EVALUATION")
    print("=" * 80)
    print(f"Dataset Path:  {args.dataset_dir}")
    print(f"Sample Size:   {'FULL DATASET (2,206,821 records)' if args.sample_size <= 0 else f'{args.sample_size:,} records'}")
    print("=" * 80)

    # 1. Load Ground Truth
    gt_dict = load_ground_truth_map(gt_file)

    countries = ["India", "US"]
    blocker = ProductionBlocker(bucket_cap=args.bucket_cap)
    scorer = PrecisionProductionScorer()

    # Track predictions for 3 candidate configurations
    # Config 1: Production (tau = 0.92, max 1 per source)
    preds_prod = {}
    # Config 2: Balanced (tau = 0.80, multi-match)
    preds_bal = {}
    # Config 3: Calibrated Optimal (tau = 0.70, multi-match)
    preds_opt = {}

    all_evaluated_s1_ids = set()
    total_processed = 0
    t_global_start = time.time()

    for country in countries:
        print(f"\n{'='*30} PROCESSING COUNTRY: {country} {'='*30}")
        t_country_start = time.time()

        # Load S1
        s1_pl = pl.read_csv(
            s1_file, separator="\t",
            schema_overrides={"entity_id": pl.String, "business_name": pl.String, "business_address": pl.String, "country": pl.String}
        ).filter(pl.col("country") == country)

        if args.sample_size > 0:
            country_sample = int(args.sample_size * (len(s1_pl) / 2206821))
            s1_pl = s1_pl.head(country_sample)

        n_s1 = len(s1_pl)
        s1_ids = s1_pl["entity_id"].to_list()
        s1_names = s1_pl["business_name"].to_list()
        s1_addrs = s1_pl["business_address"].to_list()
        del s1_pl
        gc.collect()

        # Load S2 and S3 for this country
        print(f"[Loader] Streaming S2 & S3 for country: {country}...")
        s2_pl = pl.read_csv(
            s2_file, separator="\t",
            schema_overrides={"entity_id": pl.String, "business_name": pl.String, "business_address": pl.String, "country": pl.String}
        ).filter(pl.col("country") == country)

        s3_pl = pl.read_csv(
            s3_file, separator="\t",
            schema_overrides={"entity_id": pl.String, "business_name": pl.String, "business_address": pl.String, "country": pl.String}
        ).filter(pl.col("country") == country)

        s2_ids = s2_pl["entity_id"].to_list()
        s2_names = s2_pl["business_name"].to_list()
        s2_addrs = s2_pl["business_address"].to_list()
        del s2_pl

        s3_ids = s3_pl["entity_id"].to_list()
        s3_names = s3_pl["business_name"].to_list()
        s3_addrs = s3_pl["business_address"].to_list()
        del s3_pl
        gc.collect()

        # Build Inverted Indexes
        print(f"  [Index] Building S2 dual-index ({len(s2_ids):,} entities)...")
        t0 = time.time()
        s2_index = blocker.build_index(s2_ids, s2_names, s2_addrs)
        print(f"  [Index] S2 indexed in {time.time() - t0:.2f}s ({len(s2_index):,} keys)")

        print(f"  [Index] Building S3 dual-index ({len(s3_ids):,} entities)...")
        t0 = time.time()
        s3_index = blocker.build_index(s3_ids, s3_names, s3_addrs)
        print(f"  [Index] S3 indexed in {time.time() - t0:.2f}s ({len(s3_index):,} keys)")

        print(f"  [Evaluate] Scoring {n_s1:,} S1 entities for {country}...")
        t_query0 = time.time()

        CHUNK_SIZE = 10000
        for start_idx in range(0, n_s1, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, n_s1)

            for i in range(start_idx, end_idx):
                sid = s1_ids[i]
                s_name = s1_names[i]
                s_addr = s1_addrs[i]
                all_evaluated_s1_ids.add(sid)

                s2_cands = blocker.retrieve_top_k(s_name, s_addr, s2_index, s2_ids, s2_names, s2_addrs, top_k=5)
                s3_cands = blocker.retrieve_top_k(s_name, s_addr, s3_index, s3_ids, s3_names, s3_addrs, top_k=5)
                all_cands = s2_cands + s3_cands

                # Score all candidates
                scored_cands = []
                for cid, base_sc, c_idx in all_cands:
                    if cid.startswith("S2-"):
                        t_n, t_a = s2_names[c_idx], s2_addrs[c_idx]
                    else:
                        t_n, t_a = s3_names[c_idx], s3_addrs[c_idx]
                    sc = scorer.compute_match_score(s_name, s_addr, t_n, t_a, base_sc)
                    scored_cands.append((cid, sc))

                # Config 1: Production (tau = 0.92, max 1 per source)
                c1_matches = scorer.prune_and_select_matches([(c, s) for c, s in scored_cands if s >= 0.92])
                preds_prod[sid] = [c for c, _ in c1_matches]

                # Config 2: Balanced (tau = 0.80, multi-match)
                preds_bal[sid] = [c for c, s in scored_cands if s >= 0.80]

                # Config 3: Calibrated Optimal (tau = 0.70, multi-match)
                preds_opt[sid] = [c for c, s in scored_cands if s >= 0.70]

            total_processed += (end_idx - start_idx)
            rate = total_processed / (time.time() - t_global_start)
            pct = (end_idx / n_s1) * 100.0
            print(f"    Processed {end_idx:,}/{n_s1:,} ({pct:.1f}%) | Overall: {total_processed:,} @ {rate:,.0f} queries/sec")

        del s2_ids, s2_names, s2_addrs, s2_index
        del s3_ids, s3_names, s3_addrs, s3_index
        gc.collect()

    print("\n" + "=" * 80)
    print("      EXACT OVERALL BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total S1 Entities Evaluated: {len(all_evaluated_s1_ids):,}")
    print(f"Total Execution Time:        {time.time() - t_global_start:.2f}s ({(time.time() - t_global_start)/60:.2f} min)")
    print("-" * 80)

    m_prod = compute_metrics(preds_prod, gt_dict, all_evaluated_s1_ids)
    m_bal = compute_metrics(preds_bal, gt_dict, all_evaluated_s1_ids)
    m_opt = compute_metrics(preds_opt, gt_dict, all_evaluated_s1_ids)

    print(f"{'Metric':<25} | {'Production (tau=0.92, max 1)':<28} | {'Balanced (tau=0.80, multi)':<26} | {'Calibrated (tau=0.70, multi)':<28}")
    print("-" * 80)
    print(f"{'Macro Precision':<25} | {m_prod['macro_precision']:<28.4f} | {m_bal['macro_precision']:<26.4f} | {m_opt['macro_precision']:<28.4f}")
    print(f"{'Macro Recall':<25} | {m_prod['macro_recall']:<28.4f} | {m_bal['macro_recall']:<26.4f} | {m_opt['macro_recall']:<28.4f}")
    print(f"{'Singleton Accuracy':<25} | {m_prod['singleton_accuracy']:<28.4%} | {m_bal['singleton_accuracy']:<26.4%} | {m_opt['singleton_accuracy']:<28.4%}")
    print(f"{'Non-Singleton F0.5':<25} | {m_prod['non_singleton_f05']:<28.4f} | {m_bal['non_singleton_f05']:<26.4f} | {m_opt['non_singleton_f05']:<28.4f}")
    print("-" * 80)
    print(f"{'FINAL MACRO F_0.5':<25} | {m_prod['macro_f05']:<28.4f} | {m_bal['macro_f05']:<26.4f} | {m_opt['macro_f05']:<28.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
