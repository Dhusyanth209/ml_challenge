"""
Baseline Blocking Module (Method 1).
Implements Character 3-Gram TF-IDF Blocking partitioned by Country.
Retrieves top 3 nearest candidates from Source 2 and top 3 from Source 3 for every Source 1 entity.
Computes Candidate Recall Ceiling and Reduction Ratio, and exports candidate_pairs.tsv.
"""

import os
import re
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> str:
    """
    Cleans text by lowercasing, stripping punctuation,
    and removing common corporate legal suffixes.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Strip legal entity suffixes
    text = re.sub(r"\b(pvt|ltd|inc|corp|llc|limited|private|incorporated)\b", " ", text)
    # Strip punctuation and special characters
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def build_composite_text(names, addresses):
    """
    Combines cleaned_name + ' ' + cleaned_address.
    """
    c_names = [clean_text(n) for n in names]
    c_addrs = [clean_text(a) for a in addresses]
    return [f"{n} {a}".strip() for n, a in zip(c_names, c_addrs)]


def query_nearest_candidates(s1_vecs: csr_matrix, target_vecs: csr_matrix, target_ids: np.ndarray,
                             top_k: int = 3, min_similarity: float = 0.05,
                             q_batch_size: int = 2000, t_chunk_size: int = 250000):
    """
    Memory-safe batched sparse dot-product with C-level zero elimination.
    Returns: list of [(target_id, score), ...] per S1 query.
    """
    num_queries = s1_vecs.shape[0]
    num_targets = target_vecs.shape[0]
    results = [[] for _ in range(num_queries)]

    if num_targets == 0 or num_queries == 0:
        return results

    total_q_batches = (num_queries + q_batch_size - 1) // q_batch_size

    for q_idx, q_start in enumerate(range(0, num_queries, q_batch_size), 1):
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


def run_country_partitioned_blocking(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                                     top_k_per_source: int = 3, min_similarity: float = 0.05):
    """
    Partitions dataset dynamically by country (supports US, India, France, etc.)
    and executes Character 3-Gram TF-IDF blocking.
    Returns: candidates_dict = {s1_id: [(cand_id, score), ...]}
    """
    candidates_dict = {sid: [] for sid in s1_df["entity_id"]}
    countries = s1_df["country"].fillna("UNKNOWN").unique()
    print(f"\nExecuting Country-Partitioned Blocking across {len(countries)} countries: {list(countries)}")

    for country in countries:
        s1_sub = s1_df[s1_df["country"] == country]
        s2_sub = s2_df[s2_df["country"] == country]
        s3_sub = s3_df[s3_df["country"] == country]

        n1, n2, n3 = len(s1_sub), len(s2_sub), len(s3_sub)
        print(f"\n--- Partition: Country = '{country}' (S1={n1:,}, S2={n2:,}, S3={n3:,}) ---")
        if n1 == 0:
            continue

        s1_ids = s1_sub["entity_id"].values
        s1_texts = build_composite_text(s1_sub["business_name"].values, s1_sub["business_address"].values)

        # 1. Blocking against Source 2
        if n2 > 0:
            s2_ids = s2_sub["entity_id"].values
            s2_texts = build_composite_text(s2_sub["business_name"].values, s2_sub["business_address"].values)

            print(f"  Fitting Character 3-Gram TfidfVectorizer for Source 2...")
            vec2 = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 3),
                min_df=2,
                sublinear_tf=True,
                max_features=100000,
                dtype=np.float32
            )
            s2_vecs = vec2.fit_transform(s2_texts)
            s1_vecs_2 = vec2.transform(s1_texts)

            print(f"  Retrieving top {top_k_per_source} candidates from Source 2...")
            s2_matches = query_nearest_candidates(
                s1_vecs_2, s2_vecs, s2_ids,
                top_k=top_k_per_source, min_similarity=min_similarity
            )
            for sid, cands in zip(s1_ids, s2_matches):
                candidates_dict[sid].extend(cands)

        # 2. Blocking against Source 3
        if n3 > 0:
            s3_ids = s3_sub["entity_id"].values
            s3_texts = build_composite_text(s3_sub["business_name"].values, s3_sub["business_address"].values)

            print(f"  Fitting Character 3-Gram TfidfVectorizer for Source 3...")
            vec3 = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 3),
                min_df=2,
                sublinear_tf=True,
                max_features=100000,
                dtype=np.float32
            )
            s3_vecs = vec3.fit_transform(s3_texts)
            s1_vecs_3 = vec3.transform(s1_texts)

            print(f"  Retrieving top {top_k_per_source} candidates from Source 3...")
            s3_matches = query_nearest_candidates(
                s1_vecs_3, s3_vecs, s3_ids,
                top_k=top_k_per_source, min_similarity=min_similarity
            )
            for sid, cands in zip(s1_ids, s3_matches):
                candidates_dict[sid].extend(cands)

    return candidates_dict


def save_candidate_pairs(candidates_dict: dict, output_file: str, all_s1_ids: list):
    """
    Saves candidate pairs into output/candidate_pairs.tsv in exact competition format:
    source1_entity_id\tcandidate_entity_ids
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            cands = candidates_dict.get(sid, [])
            cand_ids = []
            seen = set()
            for item in cands:
                cid = item[0] if isinstance(item, (tuple, list)) else item
                if cid not in seen:
                    seen.add(cid)
                    cand_ids.append(cid)
            f.write(f"{sid}\t{','.join(cand_ids)}\n")
    print(f"Successfully saved candidate pairs to: {output_file}")


def evaluate_blocking_metrics(candidates_dict: dict, gt_dict: dict, val_s1_ids: set, num_s2: int, num_s3: int):
    """
    Computes Candidate Recall Ceiling and Candidate Set Reduction Ratio on validation split.
    """
    total_true_matches = 0
    captured_matches = 0
    total_candidates = 0

    for sid in val_s1_ids:
        true_matches = gt_dict.get(sid, set())
        total_true_matches += len(true_matches)

        cands = candidates_dict.get(sid, [])
        cand_ids = {c[0] if isinstance(c, (tuple, list)) else c for c in cands}
        total_candidates += len(cand_ids)

        if len(true_matches) > 0:
            captured_matches += len(true_matches & cand_ids)

    recall_ceiling = captured_matches / total_true_matches if total_true_matches > 0 else 1.0
    avg_cands = total_candidates / len(val_s1_ids) if len(val_s1_ids) > 0 else 0.0
    total_possible_pairs = len(val_s1_ids) * (num_s2 + num_s3)
    reduction_ratio = 1.0 - (total_candidates / total_possible_pairs) if total_possible_pairs > 0 else 1.0

    print("\n" + "=" * 60)
    print("BLOCKING AUDIT (Validation Split):")
    print(f"  Total True Matches:           {total_true_matches:,}")
    print(f"  Captured in Candidate Pairs:  {captured_matches:,}")
    print(f"  Candidate Recall Ceiling:     {recall_ceiling:.2%}")
    print(f"  Total Candidates Generated:   {total_candidates:,}")
    print(f"  Avg Candidates / S1 Entity:   {avg_cands:.2f}")
    print(f"  Candidate Reduction Ratio:    {reduction_ratio:.6%}")
    print("=" * 60)

    return recall_ceiling, avg_cands, reduction_ratio
