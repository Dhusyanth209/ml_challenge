"""
Multi-Pass Union Blocking Module (Method 2).
Combines:
  Pass 1: Country-Partitioned Character 3-Gram TF-IDF Sparse Indexing (Top 3 from S2, Top 3 from S3)
  Pass 2: Name Prefix/Token Inverted Index (First 2 tokens / 6-char prefix)
Merges candidates, deduplicates, and caps at top 4-5 candidates per S1 entity
to achieve >= 98% Candidate Recall Ceiling while preserving > 99.9% Reduction Ratio.
"""

import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """Standard cleaner: lowercase, strip legal entity suffixes and special chars."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\b(pvt|ltd|inc|corp|llc|limited|private|incorporated|the|co)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_prefix_key(name: str) -> str:
    """Extracts first 2 tokens or first 6 characters as blocking key."""
    cleaned = clean_text(name)
    words = cleaned.split()
    if len(words) >= 2:
        return f"{words[0][:5]}_{words[1][:5]}"
    elif len(words) == 1:
        return words[0][:6]
    return ""


def build_composite_text(names, addresses):
    """Combines cleaned name (weighted) and cleaned address."""
    c_names = [clean_text(n) for n in names]
    c_addrs = [clean_text(a) for a in addresses]
    return [f"{n} {n} {a}".strip() for n, a in zip(c_names, c_addrs)]


def query_top_k_sparse(s1_vecs: csr_matrix, target_vecs: csr_matrix, target_ids: np.ndarray,
                       top_k: int = 3, min_similarity: float = 0.05,
                       q_batch_size: int = 2000, t_chunk_size: int = 250000):
    """Memory-safe batched sparse dot-product with C-level sparsity pruning."""
    num_queries = s1_vecs.shape[0]
    num_targets = target_vecs.shape[0]
    results = [[] for _ in range(num_queries)]

    if num_targets == 0 or num_queries == 0:
        return results

    for q_start in range(0, num_queries, q_batch_size):
        q_end = min(q_start + q_batch_size, num_queries)
        batch_q = s1_vecs[q_start:q_end]
        q_len = q_end - q_start

        batch_candidates = [[] for _ in range(q_len)]

        for t_start in range(0, num_targets, t_chunk_size):
            t_end = min(t_start + t_chunk_size, num_targets)
            sub_target_T = target_vecs[t_start:t_end].T.tocsc()

            sim_chunk = batch_q.dot(sub_target_T)
            sim_chunk.data[sim_chunk.data < min_similarity] = 0.0
            sim_chunk.eliminate_zeros()

            if sim_chunk.nnz == 0:
                continue

            indptr = sim_chunk.indptr
            indices = sim_chunk.indices
            data = sim_chunk.data

            for i in range(q_len):
                p_start = indptr[i]
                p_end = indptr[i + 1]
                if p_start < p_end:
                    cols = indices[p_start:p_end]
                    vals = data[p_start:p_end]
                    for col_idx, val in zip(cols, vals):
                        cand_id = target_ids[t_start + col_idx]
                        batch_candidates[i].append((cand_id, float(val)))

        for i in range(q_len):
            cands = batch_candidates[i]
            if len(cands) <= top_k:
                cands.sort(key=lambda x: x[1], reverse=True)
                results[q_start + i] = cands
            else:
                cands.sort(key=lambda x: x[1], reverse=True)
                results[q_start + i] = cands[:top_k]

    return results


def run_multipass_union_blocking(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                                 max_candidates_per_s1: int = 5):
    """
    Executes Multi-Pass Union Blocking:
      1. Pass 1: Character 3-Gram TF-IDF (top 3 from S2, top 3 from S3)
      2. Pass 2: Prefix/Token inverted index
    Merges candidates and caps at max_candidates_per_s1 (default: 5).
    """
    merged_candidates = {sid: {} for sid in s1_df["entity_id"]}
    countries = s1_df["country"].fillna("UNKNOWN").unique()
    print(f"\nExecuting Multi-Pass Union Blocking across {len(countries)} countries: {list(countries)}")

    for country in countries:
        s1_sub = s1_df[s1_df["country"] == country]
        s2_sub = s2_df[s2_df["country"] == country]
        s3_sub = s3_df[s3_df["country"] == country]

        n1, n2, n3 = len(s1_sub), len(s2_sub), len(s3_sub)
        print(f"\n--- Partition '{country}': S1={n1:,}, S2={n2:,}, S3={n3:,} ---")
        if n1 == 0:
            continue

        s1_ids = s1_sub["entity_id"].values
        s1_names = s1_sub["business_name"].values
        s1_addrs = s1_sub["business_address"].values
        s1_texts = build_composite_text(s1_names, s1_addrs)

        # -------------------------------------------------------------
        # PASS 1A: TF-IDF Blocking against Source 2
        # -------------------------------------------------------------
        if n2 > 0:
            s2_ids = s2_sub["entity_id"].values
            s2_names = s2_sub["business_name"].values
            s2_addrs = s2_sub["business_address"].values
            s2_texts = build_composite_text(s2_names, s2_addrs)

            print(f"  [Pass 1A] TF-IDF 3-Gram Vectorizer for Source 2...")
            vec2 = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                sublinear_tf=True, max_features=100000, dtype=np.float32
            )
            s2_vecs = vec2.fit_transform(s2_texts)
            s1_vecs_2 = vec2.transform(s1_texts)

            print(f"  [Pass 1A] Querying top 3 candidates from Source 2...")
            s2_top = query_top_k_sparse(s1_vecs_2, s2_vecs, s2_ids, top_k=3, min_similarity=0.05)
            for sid, cands in zip(s1_ids, s2_top):
                for cid, score in cands:
                    merged_candidates[sid][cid] = max(merged_candidates[sid].get(cid, 0.0), score)

            # ---------------------------------------------------------
            # PASS 2A: Prefix/Token Key Index for Source 2
            # ---------------------------------------------------------
            print(f"  [Pass 2A] Prefix/Token inverted indexing for Source 2...")
            s2_prefix_index = defaultdict(list)
            for cid, name in zip(s2_ids, s2_names):
                k = extract_prefix_key(name)
                if k and len(s2_prefix_index[k]) < 5:  # cap index postings to prevent rare giant buckets
                    s2_prefix_index[k].append(cid)

            for sid, name in zip(s1_ids, s1_names):
                k = extract_prefix_key(name)
                if k in s2_prefix_index:
                    for cid in s2_prefix_index[k]:
                        if cid not in merged_candidates[sid]:
                            merged_candidates[sid][cid] = 0.50  # baseline heuristic score for prefix hit

        # -------------------------------------------------------------
        # PASS 1B: TF-IDF Blocking against Source 3
        # -------------------------------------------------------------
        if n3 > 0:
            s3_ids = s3_sub["entity_id"].values
            s3_names = s3_sub["business_name"].values
            s3_addrs = s3_sub["business_address"].values
            s3_texts = build_composite_text(s3_names, s3_addrs)

            print(f"  [Pass 1B] TF-IDF 3-Gram Vectorizer for Source 3...")
            vec3 = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 3), min_df=2,
                sublinear_tf=True, max_features=100000, dtype=np.float32
            )
            s3_vecs = vec3.fit_transform(s3_texts)
            s1_vecs_3 = vec3.transform(s1_texts)

            print(f"  [Pass 1B] Querying top 3 candidates from Source 3...")
            s3_top = query_top_k_sparse(s1_vecs_3, s3_vecs, s3_ids, top_k=3, min_similarity=0.05)
            for sid, cands in zip(s1_ids, s3_top):
                for cid, score in cands:
                    merged_candidates[sid][cid] = max(merged_candidates[sid].get(cid, 0.0), score)

            # ---------------------------------------------------------
            # PASS 2B: Prefix/Token Key Index for Source 3
            # ---------------------------------------------------------
            print(f"  [Pass 2B] Prefix/Token inverted indexing for Source 3...")
            s3_prefix_index = defaultdict(list)
            for cid, name in zip(s3_ids, s3_names):
                k = extract_prefix_key(name)
                if k and len(s3_prefix_index[k]) < 5:
                    s3_prefix_index[k].append(cid)

            for sid, name in zip(s1_ids, s1_names):
                k = extract_prefix_key(name)
                if k in s3_prefix_index:
                    for cid in s3_prefix_index[k]:
                        if cid not in merged_candidates[sid]:
                            merged_candidates[sid][cid] = 0.50

    # Format and cap at max_candidates_per_s1
    final_candidates = {}
    for sid, cand_map in merged_candidates.items():
        sorted_cands = sorted(cand_map.items(), key=lambda x: x[1], reverse=True)
        final_candidates[sid] = sorted_cands[:max_candidates_per_s1]

    return final_candidates


def save_candidate_pairs(candidates_dict: dict, output_file: str, all_s1_ids: list):
    """Saves candidate pairs to candidate_pairs.tsv."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            cands = candidates_dict.get(sid, [])
            cand_ids = [c[0] for c in cands]
            f.write(f"{sid}\t{','.join(cand_ids)}\n")
    print(f"Exported candidate pairs to: {output_file}")


def evaluate_blocking_m2(candidates_dict: dict, gt_dict: dict, val_s1_ids: set, num_s2: int, num_s3: int):
    """Computes Candidate Recall Ceiling and Reduction Ratio."""
    total_true_matches = 0
    captured_matches = 0
    total_candidates = 0

    for sid in val_s1_ids:
        true_matches = gt_dict.get(sid, set())
        total_true_matches += len(true_matches)

        cands = candidates_dict.get(sid, [])
        cand_ids = {c[0] for c in cands}
        total_candidates += len(cand_ids)

        if len(true_matches) > 0:
            captured_matches += len(true_matches & cand_ids)

    recall_ceiling = captured_matches / total_true_matches if total_true_matches > 0 else 1.0
    avg_cands = total_candidates / len(val_s1_ids) if len(val_s1_ids) > 0 else 0.0
    total_possible_pairs = len(val_s1_ids) * (num_s2 + num_s3)
    reduction_ratio = 1.0 - (total_candidates / total_possible_pairs) if total_possible_pairs > 0 else 1.0

    print("\n" + "=" * 65)
    print("METHOD 2: MULTI-PASS UNION BLOCKING AUDIT:")
    print(f"  Total True Matches:           {total_true_matches:,}")
    print(f"  Captured in Candidate Pairs:  {captured_matches:,}")
    print(f"  Candidate Recall Ceiling:     {recall_ceiling:.2%} (Target >= 98.0%)")
    print(f"  Total Candidates Generated:   {total_candidates:,}")
    print(f"  Avg Candidates / S1 Entity:   {avg_cands:.2f} (Target <= 5.0)")
    print(f"  Reduction Ratio:              {reduction_ratio:.6%}")
    print("=" * 65)

    return recall_ceiling, avg_cands, reduction_ratio
