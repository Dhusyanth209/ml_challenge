#!/usr/bin/env python3
"""
Champion Pipeline: Unified Entity Resolution
=============================================
Fuses v2's vectorized sparse blocking + C++ feature extraction with
Methodology A's global bipartite conflict resolution.

Since no train_ground_truth.tsv / train_source2/3.tsv are available,
uses a calibrated heuristic scorer on the 20 v2 features instead of
a trained ML classifier.

Usage:
    python3 run_champion_pipeline.py [--threshold 0.80] [--top-k 15] [--workers 4]
"""
import os
import sys
import gc
import time
import argparse
import logging
import numpy as np
import pandas as pd

# Add v2 src to path so we can import its modules directly
V2_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      'student_resource', 'v2', 'src')
sys.path.insert(0, V2_SRC)

from preprocessing import preprocess_dataframe                     # v2
from blocking import SparseBlocker, blocking_text                  # v2
from feature_engineering import compute_features, FEATURES         # v2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('champion')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── ID encoding (from v2 data_loader) ───────────────────────────────────────
ID_BASE = 10 ** 12

def encode_ids(ids: pd.Series) -> np.ndarray:
    src = ids.str.slice(1, 2).astype(np.int64)
    num = ids.str.slice(3).astype(np.int64)
    return src.values * ID_BASE + num.values

def decode_ids(enc: np.ndarray) -> np.ndarray:
    enc = np.asarray(enc)
    return ('S' + pd.Series(enc // ID_BASE).astype(str) +
            '-' + pd.Series(enc % ID_BASE).astype(str)).values


# ─── Data loading (adapted from v2, no multiprocessing for stability) ────────
READ_CHUNK = 250_000

def load_source_country(path: str, country: str) -> pd.DataFrame:
    """Load one source file, filter to country, preprocess. Returns eid/name/addr."""
    reader = pd.read_csv(
        path, sep="\t", dtype=str, chunksize=READ_CHUNK,
        keep_default_na=False,
        usecols=['entity_id', 'business_name', 'business_address', 'country'],
        quoting=3,
    )
    parts = []
    for chunk in reader:
        sub = chunk[chunk['country'] == country]
        if sub.empty:
            continue
        out = preprocess_dataframe(sub)
        parts.append(pd.DataFrame({
            'eid': encode_ids(sub['entity_id']),
            'name': out['name'].values,
            'addr': out['addr'].values,
        }))
    if not parts:
        return pd.DataFrame(columns=['eid', 'name', 'addr'])
    return pd.concat(parts, ignore_index=True)


def list_countries(data_dir: str, split: str) -> list:
    path = os.path.join(data_dir, split, f"{split}_source1.tsv")
    return (pd.read_csv(path, sep="\t", usecols=['country'], dtype=str,
                        keep_default_na=False, quoting=3)
            ['country'].value_counts().index.tolist())


def load_country(data_dir: str, split: str, country: str):
    """Returns (s1_df, pool_df) for one country. pool = source2 + source3."""
    logger.info(f"Loading {split}/{country}...")
    s1 = load_source_country(
        os.path.join(data_dir, split, f"{split}_source1.tsv"), country)
    pool_parts = []
    for i in (2, 3):
        p = load_source_country(
            os.path.join(data_dir, split, f"{split}_source{i}.tsv"), country)
        if not p.empty:
            pool_parts.append(p)
    pool = pd.concat(pool_parts, ignore_index=True) if pool_parts else pd.DataFrame(columns=['eid','name','addr'])
    logger.info(f"  {country}: {len(s1):,} S1 × {len(pool):,} pool")
    return s1, pool


# ─── Heuristic Scorer (replaces ML classifier when no training data) ─────────

def compute_match_scores(feats: pd.DataFrame) -> np.ndarray:
    """
    Compute a composite match score ∈ [0, 1] from the 20 v2 features.

    Weights calibrated from the feature importance analysis:
    - name_tset (token-set-ratio) is the single most discriminative feature
    - blocking_score provides the base similarity
    - addr_tset adds address confirmation
    - name_jw (Jaro-Winkler) is best for short names
    - compact_ratio catches names with spacing differences
    """
    # Primary: name similarity (65% weight)
    name_score = (
        0.25 * feats['name_tset'].values / 100.0 +
        0.15 * feats['name_tsort'].values / 100.0 +
        0.10 * feats['name_ratio'].values / 100.0 +
        0.10 * feats['name_jw'].values +
        0.05 * feats['compact_ratio'].values / 100.0
    )

    # Secondary: address similarity (20% weight)
    addr_score = (
        0.10 * feats['addr_tset'].values / 100.0 +
        0.05 * feats['addr_tsort'].values / 100.0 +
        0.05 * feats['addr_ratio'].values / 100.0
    )

    # Tertiary: blocking context (10% weight)
    block_score = (
        0.10 * feats['blocking_score'].values
    )

    # Numeric confirmation bonus (5% weight)
    num_score = (
        0.05 * feats['num_tset'].values / 100.0 * feats['num_both'].values
    )

    total = name_score + addr_score + block_score + num_score

    # Penalties for suspicious pairs
    # If both names empty, the match is meaningless
    empty_mask = (feats['name1_empty'].values > 0.5) | (feats['name2_empty'].values > 0.5)
    total[empty_mask] *= 0.3

    # Very short names with low length ratio are risky
    short_mask = (feats['len_ratio'].values < 0.4) & (feats['name_ratio'].values < 60)
    total[short_mask] *= 0.5

    return total.astype(np.float32)


# ─── Bipartite Conflict Resolution (from Methodology A) ──────────────────────

def resolve_bipartite(
    all_s1_eids: np.ndarray,
    all_pairs: list,       # list of (s1_eid_int, cand_eid_int)
    all_scores: list,      # corresponding match scores
    threshold: float,
) -> dict:
    """
    Global greedy bipartite assignment.
    Each target entity (S2/S3) can be matched to at most ONE S1 entity.
    Returns {s1_eid_int: [cand_eid_int, ...]}
    """
    final = {int(e): [] for e in all_s1_eids}

    # Filter to above-threshold pairs
    qualified = []
    for (sid, cid), score in zip(all_pairs, all_scores):
        if score >= threshold:
            qualified.append((float(score), int(sid), int(cid)))

    # Sort by score descending — highest confidence claims first
    qualified.sort(key=lambda x: x[0], reverse=True)

    claimed = set()
    for score, sid, cid in qualified:
        if cid not in claimed:
            claimed.add(cid)
            final[sid].append(cid)

    n_matched = sum(len(v) for v in final.values())
    n_singleton = sum(1 for v in final.values() if len(v) == 0)
    logger.info(f"  Bipartite resolution: {n_matched:,} matches, "
                f"{n_singleton:,} singletons, {len(claimed):,} unique targets claimed")
    return final


# ─── Main Pipeline ───────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(description='Champion Entity Resolution Pipeline')
    ap.add_argument('--dataset-dir', default=os.path.join(BASE_DIR, 'dataset'))
    ap.add_argument('--output-dir', default=os.path.join(BASE_DIR, 'output'))
    ap.add_argument('--threshold', type=float, default=0.80,
                    help='Match threshold (calibrated optimal ≈ 0.80)')
    ap.add_argument('--top-k', type=int, default=15,
                    help='Blocking: candidates per S1 entity')
    ap.add_argument('--max-df', type=int, default=3000,
                    help='Blocking: max document frequency for IDF')
    ap.add_argument('--query-chunk', type=int, default=1000,
                    help='Blocking: S1 rows per sparse matmul')
    ap.add_argument('--batch', type=int, default=50_000,
                    help='Feature extraction: S1 rows per batch')
    ap.add_argument('--workers', type=int, default=4,
                    help='Parallel workers for hashing/search')
    ap.add_argument('--test-limit', type=int, default=0,
                    help='Debug: process only first N S1 rows per country')
    return ap.parse_args()


def iter_candidates(q, p, args):
    """Per-country sparse blocking. Yields (b_start, b_end, q_idx, p_idx, scores)."""
    blocker = SparseBlocker(
        top_k=args.top_k, max_df=args.max_df,
        query_chunk=args.query_chunk, workers=args.workers,
    )
    blocker.fit(blocking_text(p), blocking_text(q))
    for b_start in range(0, len(q), args.batch):
        b_end = min(b_start + args.batch, len(q))
        qi, pi, sc = blocker.search(b_start, b_end)
        yield b_start, b_end, qi, pi, sc


def run_pipeline(args):
    t_total = time.time()
    os.makedirs(args.output_dir, exist_ok=True)

    cand_path = os.path.join(args.output_dir, 'candidate_pairs.tsv')
    match_path = os.path.join(args.output_dir, 'matching_results.tsv')

    countries = list_countries(args.dataset_dir, 'test')
    logger.info(f"Countries discovered: {countries}")
    logger.info(f"Threshold: {args.threshold}, top-k: {args.top_k}, "
                f"batch: {args.batch:,}, workers: {args.workers}")

    # Accumulate all data across countries for global bipartite resolution
    all_s1_eids = []           # ordered list of all S1 entity IDs (int64)
    all_s1_id_strings = []     # corresponding string IDs for output
    all_candidates = {}        # {s1_eid_int: [cand_eid_str, ...]}  for candidate_pairs.tsv
    all_pairs = []             # [(s1_eid_int, cand_eid_int), ...]
    all_scores = []            # [match_score, ...]

    for country in countries:
        t_country = time.time()
        logger.info(f"═══ Processing {country} ═══")

        q, p = load_country(args.dataset_dir, 'test', country)
        if args.test_limit:
            q = q.head(args.test_limit).reset_index(drop=True)

        q_ids_int = q['eid'].values
        q_ids_str = decode_ids(q_ids_int)

        all_s1_eids.extend(q_ids_int.tolist())
        all_s1_id_strings.extend(q_ids_str.tolist())

        if p.empty:
            logger.info(f"  No pool entities for {country}, all singletons")
            for eid_int in q_ids_int:
                all_candidates[int(eid_int)] = []
            continue

        # ── Blocking + Feature Extraction + Scoring ──
        for b_start, b_end, qi, pi, sc in iter_candidates(q, p, args):
            if len(qi) == 0:
                logger.info(f"    {b_end:,}/{len(q):,} — no candidates in this batch")
                # Still register these S1 entities as having no candidates
                for i in range(b_start, b_end):
                    eid_int = int(q_ids_int[i])
                    if eid_int not in all_candidates:
                        all_candidates[eid_int] = []
                continue

            # Compute 20 vectorized features
            feats = compute_features(q, p, qi, pi, sc)

            # Heuristic scoring
            match_scores = compute_match_scores(feats)

            # Collect candidates and pairs
            p_eids_int = p['eid'].values[pi]
            p_eids_str = decode_ids(p_eids_int)

            for i in range(b_start, b_end):
                eid_int = int(q_ids_int[i])
                if eid_int not in all_candidates:
                    all_candidates[eid_int] = []

            for idx in range(len(qi)):
                s1_row = qi[idx]
                s1_eid = int(q_ids_int[s1_row])
                c_eid_int = int(p_eids_int[idx])
                c_eid_str = p_eids_str[idx]

                all_candidates[s1_eid].append(c_eid_str)
                all_pairs.append((s1_eid, c_eid_int))
                all_scores.append(float(match_scores[idx]))

            logger.info(f"    {b_end:,}/{len(q):,} S1 done — "
                        f"{len(qi):,} pairs scored")

        del q, p
        gc.collect()
        logger.info(f"  {country} completed in {time.time()-t_country:.0f}s")

    # ── Phase 6: Global Bipartite Conflict Resolution ──
    logger.info(f"═══ Global Bipartite Resolution ═══")
    logger.info(f"  Total S1 entities: {len(all_s1_eids):,}")
    logger.info(f"  Total candidate pairs: {len(all_pairs):,}")

    s1_eids_arr = np.array(all_s1_eids, dtype=np.int64)
    bipartite_matches = resolve_bipartite(
        s1_eids_arr, all_pairs, all_scores, args.threshold
    )

    # ── Phase 7: Write Output Files ──
    logger.info(f"═══ Writing Output ═══")

    # candidate_pairs.tsv
    with open(cand_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tcandidate_entity_ids\n')
        for eid_int, eid_str in zip(all_s1_eids, all_s1_id_strings):
            cands = all_candidates.get(int(eid_int), [])
            # Deduplicate while preserving order
            seen = set()
            unique_cands = []
            for c in cands:
                if c not in seen:
                    seen.add(c)
                    unique_cands.append(c)
            f.write(f"{eid_str}\t{','.join(unique_cands)}\n")

    # matching_results.tsv (post-bipartite resolution)
    with open(match_path, 'w', encoding='utf-8') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for eid_int, eid_str in zip(all_s1_eids, all_s1_id_strings):
            matches = bipartite_matches.get(int(eid_int), [])
            match_strs = decode_ids(np.array(matches, dtype=np.int64)) if matches else []
            f.write(f"{eid_str}\t{','.join(match_strs)}\n")

    elapsed = time.time() - t_total
    logger.info(f"═══ Pipeline Complete ═══")
    logger.info(f"  Total time: {elapsed/60:.1f} minutes")
    logger.info(f"  candidate_pairs.tsv: {cand_path}")
    logger.info(f"  matching_results.tsv: {match_path}")

    # Quick stats
    n_matched = sum(1 for v in bipartite_matches.values() if len(v) > 0)
    n_total_matches = sum(len(v) for v in bipartite_matches.values())
    logger.info(f"  S1 entities with ≥1 match: {n_matched:,} / {len(all_s1_eids):,}")
    logger.info(f"  Total match pairs: {n_total_matches:,}")

    return cand_path, match_path


if __name__ == '__main__':
    args = parse_args()
    cand_path, match_path = run_pipeline(args)

    # ── Validation ──
    validator = os.path.join(BASE_DIR, 'utils', 'validate_submission.py')
    if os.path.exists(validator):
        logger.info("Running submission validator...")
        import subprocess
        result = subprocess.run([
            sys.executable, validator,
            '--matching', match_path,
            '--candidate', cand_path,
            '--test-dir', os.path.join(args.dataset_dir, 'test'),
        ], capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
