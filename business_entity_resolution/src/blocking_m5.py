"""
High-Recall Multi-Key Lexical Cascade Blocking Module (Method 5).
Combines:
  Key A: Character 3-Gram TF-IDF Cosine Index (Top 2 from S2, Top 2 from S3)
  Key B: Brand Core Token Exact Match (First 2 significant words, ignoring legal entities & stopwords)
  Key C: Normalized House Number & Postal Code Inverted Index with Brand Anchor
Unions streams and ranks candidates by cumulative lexical overlap.
Retains strictly Top 3 from S2 and Top 3 from S3 per Source 1 entity (Total max 6 per S1).
"""

import os
import gc
import re
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from rapidfuzz import fuzz

STOPWORDS = {
    'pvt', 'ltd', 'inc', 'corp', 'llc', 'limited', 'private', 'incorporated',
    'the', 'co', 'and', 'of', 'in', 'at', 'a', 'an', 'sa', 'sarl', 'gmbh',
    'enterprise', 'enterprises', 'company', 'services', 'group'
}


def clean_text(text: str) -> str:
    """Standard cleaner: lowercase, strip legal entity suffixes and special characters."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\b(pvt|ltd|inc|corp|llc|limited|private|incorporated|the|co|sa|sarl|gmbh)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def get_significant_tokens(name: str) -> list:
    """Extracts non-stopword tokens of length > 1."""
    cleaned = clean_text(name)
    return [w for w in cleaned.split() if w not in STOPWORDS and len(w) > 1]


def extract_brand_keys(name: str) -> list:
    """
    Extracts brand core keys:
    1. First 2 tokens joined (e.g., 'starbucks coffee')
    2. First token if len >= 4 (e.g., 'starbucks')
    """
    tokens = get_significant_tokens(name)
    keys = []
    if len(tokens) >= 2:
        keys.append(f"{tokens[0]} {tokens[1]}")
    if tokens and len(tokens[0]) >= 3:
        keys.append(tokens[0])
    return keys


def extract_address_keys(address: str, brand_first_token: str = "") -> list:
    """
    Extracts address number and postal code keys:
    1. Postal code (5+ digits)
    2. (street_number, brand_first_token)
    """
    keys = []
    addr_str = str(address) if address else ""
    
    # Postal code
    pins = re.findall(r"\b\d{5,}\b", addr_str)
    if pins:
        keys.append(f"pin_{pins[-1]}")
    
    # House/Street number
    nums = re.findall(r"\b\d+\b", addr_str)
    if nums and brand_first_token:
        keys.append(f"num_{nums[0]}_{brand_first_token[:4]}")
        
    return keys


def build_composite_text(names, addresses):
    """Weighted composite string emphasizing name."""
    c_names = [clean_text(n) for n in names]
    c_addrs = [clean_text(a) for a in addresses]
    return [f"{n} {n} {a}".strip() for n, a in zip(c_names, c_addrs)]


def query_top_k_sparse(s1_vecs: csr_matrix, target_vecs: csr_matrix, target_ids: np.ndarray,
                        top_k: int = 2, min_similarity: float = 0.03,
                        q_batch_size: int = 500, t_chunk_size: int = 5000):
    """
    Memory-safe batched sparse dot-product with C-level sparsity pruning.
    Avoids large intermediate dense allocations on constrained systems.
    """
    num_queries = s1_vecs.shape[0]
    num_targets = target_vecs.shape[0]
    results = [[] for _ in range(num_queries)]

    if num_targets == 0 or num_queries == 0:
        return results

    for q_start in range(0, num_queries, q_batch_size):
        q_end = min(q_start + q_batch_size, num_queries)
        batch_q = s1_vecs[q_start:q_end]

        batch_best_scores = np.zeros((q_end - q_start, top_k), dtype=np.float32)
        batch_best_indices = np.full((q_end - q_start, top_k), -1, dtype=np.int32)

        for t_start in range(0, num_targets, t_chunk_size):
            t_end = min(t_start + t_chunk_size, num_targets)
            sub_target_T = target_vecs[t_start:t_end].T

            sim_chunk = batch_q.dot(sub_target_T)
            dense_chunk = sim_chunk.toarray()

            for b_idx in range(q_end - q_start):
                row = dense_chunk[b_idx]
                if np.max(row) < min_similarity:
                    continue

                valid_mask = row >= min_similarity
                if not np.any(valid_mask):
                    continue

                valid_col_indices = np.where(valid_mask)[0]
                valid_scores = row[valid_col_indices]

                curr_scores = batch_best_scores[b_idx]
                curr_indices = batch_best_indices[b_idx]

                all_scores = np.concatenate([curr_scores, valid_scores])
                all_indices = np.concatenate([curr_indices, valid_col_indices + t_start])

                top_part = np.argpartition(-all_scores, min(top_k - 1, len(all_scores) - 1))[:top_k]
                sorted_top = top_part[np.argsort(-all_scores[top_part])]

                batch_best_scores[b_idx] = all_scores[sorted_top]
                batch_best_indices[b_idx] = all_indices[sorted_top]

        for b_idx in range(q_end - q_start):
            q_global_idx = q_start + b_idx
            cand_list = []
            for k in range(top_k):
                idx = batch_best_indices[b_idx, k]
                sc = batch_best_scores[b_idx, k]
                if idx != -1 and sc >= min_similarity:
                    cand_list.append((target_ids[idx], float(sc)))
            results[q_global_idx] = cand_list

    return results


def run_high_recall_cascade_blocking(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                                     top_k_per_source: int = 3):
    """
    Executes Method 5 High-Recall Multi-Key Lexical Cascade Blocking:
      1. Country dynamic partitioning
      2. Key A: Char-3gram TF-IDF top 2
      3. Key B: Brand Core Inverted Index
      4. Key C: House Number & Postal Code Inverted Index
      5. Cumulative score ranking and retention of top 3 from S2 + top 3 from S3.
    """
    final_candidates = {sid: [] for sid in s1_df["entity_id"]}
    countries = s1_df["country"].fillna("UNKNOWN").unique()
    print(f"\nExecuting Method 5 Multi-Key Cascade Blocking across {len(countries)} countries: {list(countries)}")

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
        s1_first_tokens = [get_significant_tokens(n)[0] if get_significant_tokens(n) else "" for n in s1_names]

        # ---------------------------------------------------------------------
        # SOURCE 2 MULTI-KEY CASCADE
        # ---------------------------------------------------------------------
        s2_merged = [defaultdict(float) for _ in range(n1)]
        if n2 > 0:
            s2_ids = s2_sub["entity_id"].values
            s2_names = s2_sub["business_name"].values
            s2_addrs = s2_sub["business_address"].values
            s2_first_tokens = [get_significant_tokens(n)[0] if get_significant_tokens(n) else "" for n in s2_names]

            # Key A: TF-IDF Char 3-Gram Index (Top 2)
            print("  [Key A - S2] Building Char 3-Gram TF-IDF Index...")
            s1_comp = build_composite_text(s1_names, s1_addrs)
            s2_comp = build_composite_text(s2_names, s2_addrs)
            vec2 = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, dtype=np.float32)
            vec2.fit(s2_comp)
            s1_v2 = vec2.transform(s1_comp)
            s2_v = vec2.transform(s2_comp)

            print("  [Key A - S2] Querying Top 2 TF-IDF Candidates...")
            s2_tfidf_cands = query_top_k_sparse(s1_v2, s2_v, s2_ids, top_k=2, min_similarity=0.03)

            for i in range(n1):
                for cid, score in s2_tfidf_cands[i]:
                    s2_merged[i][cid] += float(score)

            del vec2, s1_v2, s2_v, s2_tfidf_cands
            gc.collect()

            # Key B & C: Inverted Indexes for Brand & Address Keys
            print("  [Key B & C - S2] Building Inverted Brand & Pin/Number Keys...")
            s2_brand_idx = defaultdict(list)
            s2_addr_idx = defaultdict(list)

            for idx, (cid, cname, caddr, ctok) in enumerate(zip(s2_ids, s2_names, s2_addrs, s2_first_tokens)):
                for b_key in extract_brand_keys(cname):
                    s2_brand_idx[b_key].append((cid, cname, caddr))
                for a_key in extract_address_keys(caddr, ctok):
                    s2_addr_idx[a_key].append((cid, cname, caddr))

            # Query Brand & Address keys for S1 queries
            print("  [Key B & C - S2] Querying Brand & Pin/Number Keys...")
            for i, (s_name, s_addr, s_tok) in enumerate(zip(s1_names, s1_addrs, s1_first_tokens)):
                # Brand core match
                for b_key in extract_brand_keys(s_name):
                    if b_key in s2_brand_idx:
                        bucket = s2_brand_idx[b_key]
                        if len(bucket) <= 25:  # filter overly common buckets
                            for cid, cname, caddr in bucket:
                                sim = fuzz.token_sort_ratio(clean_text(s_name), clean_text(cname)) / 100.0
                                s2_merged[i][cid] += 0.35 + 0.15 * sim

                # Pin & Number match
                for a_key in extract_address_keys(s_addr, s_tok):
                    if a_key in s2_addr_idx:
                        bucket = s2_addr_idx[a_key]
                        if len(bucket) <= 20:
                            for cid, cname, caddr in bucket:
                                s2_merged[i][cid] += 0.25

            del s2_brand_idx, s2_addr_idx
            gc.collect()

        # ---------------------------------------------------------------------
        # SOURCE 3 MULTI-KEY CASCADE
        # ---------------------------------------------------------------------
        s3_merged = [defaultdict(float) for _ in range(n1)]
        if n3 > 0:
            s3_ids = s3_sub["entity_id"].values
            s3_names = s3_sub["business_name"].values
            s3_addrs = s3_sub["business_address"].values
            s3_first_tokens = [get_significant_tokens(n)[0] if get_significant_tokens(n) else "" for n in s3_names]

            # Key A: TF-IDF Char 3-Gram Index (Top 2)
            print("  [Key A - S3] Building Char 3-Gram TF-IDF Index...")
            s1_comp = build_composite_text(s1_names, s1_addrs)
            s3_comp = build_composite_text(s3_names, s3_addrs)
            vec3 = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, dtype=np.float32)
            vec3.fit(s3_comp)
            s1_v3 = vec3.transform(s1_comp)
            s3_v = vec3.transform(s3_comp)

            print("  [Key A - S3] Querying Top 2 TF-IDF Candidates...")
            s3_tfidf_cands = query_top_k_sparse(s1_v3, s3_v, s3_ids, top_k=2, min_similarity=0.03)

            for i in range(n1):
                for cid, score in s3_tfidf_cands[i]:
                    s3_merged[i][cid] += float(score)

            del vec3, s1_v3, s3_v, s3_tfidf_cands
            gc.collect()

            # Key B & C: Inverted Indexes for Brand & Address Keys
            print("  [Key B & C - S3] Building Inverted Brand & Pin/Number Keys...")
            s3_brand_idx = defaultdict(list)
            s3_addr_idx = defaultdict(list)

            for idx, (cid, cname, caddr, ctok) in enumerate(zip(s3_ids, s3_names, s3_addrs, s3_first_tokens)):
                for b_key in extract_brand_keys(cname):
                    s3_brand_idx[b_key].append((cid, cname, caddr))
                for a_key in extract_address_keys(caddr, ctok):
                    s3_addr_idx[a_key].append((cid, cname, caddr))

            # Query Brand & Address keys for S1 queries
            print("  [Key B & C - S3] Querying Brand & Pin/Number Keys...")
            for i, (s_name, s_addr, s_tok) in enumerate(zip(s1_names, s1_addrs, s1_first_tokens)):
                # Brand core match
                for b_key in extract_brand_keys(s_name):
                    if b_key in s3_brand_idx:
                        bucket = s3_brand_idx[b_key]
                        if len(bucket) <= 25:
                            for cid, cname, caddr in bucket:
                                sim = fuzz.token_sort_ratio(clean_text(s_name), clean_text(cname)) / 100.0
                                s3_merged[i][cid] += 0.35 + 0.15 * sim

                # Pin & Number match
                for a_key in extract_address_keys(s_addr, s_tok):
                    if a_key in s3_addr_idx:
                        bucket = s3_addr_idx[a_key]
                        if len(bucket) <= 20:
                            for cid, cname, caddr in bucket:
                                s3_merged[i][cid] += 0.25

            del s3_brand_idx, s3_addr_idx
            gc.collect()

        # ---------------------------------------------------------------------
        # RANK AND RETAIN TOP 3 FROM S2 AND TOP 3 FROM S3
        # ---------------------------------------------------------------------
        for i, sid in enumerate(s1_ids):
            # Top 3 from S2
            top_s2 = sorted(s2_merged[i].items(), key=lambda x: x[1], reverse=True)[:top_k_per_source]
            # Top 3 from S3
            top_s3 = sorted(s3_merged[i].items(), key=lambda x: x[1], reverse=True)[:top_k_per_source]
            final_candidates[sid] = top_s2 + top_s3

    return final_candidates


def save_candidate_pairs_m5(candidates_dict: dict, output_file: str, all_s1_ids: list):
    """Exports candidate pairs to candidate_pairs.tsv."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            cands = candidates_dict.get(sid, [])
            cand_ids = [c[0] for c in cands]
            f.write(f"{sid}\t{','.join(cand_ids)}\n")
    print(f"\nExported candidate pairs to: {output_file}")


def evaluate_blocking_m5(candidates_dict: dict, gt_dict: dict, val_s1_ids: set, num_s2: int, num_s3: int):
    """Evaluates Candidate Recall Ceiling and Reduction Ratio."""
    total_true = 0
    captured = 0
    total_cands = 0

    for sid in val_s1_ids:
        true_matches = gt_dict.get(sid, set())
        total_true += len(true_matches)
        cand_ids = {c[0] for c in candidates_dict.get(sid, [])}
        total_cands += len(cand_ids)
        if len(true_matches) > 0:
            captured += len(true_matches & cand_ids)

    recall_ceiling = captured / total_true if total_true > 0 else 1.0
    avg_cands = total_cands / len(val_s1_ids) if len(val_s1_ids) > 0 else 0.0
    total_possible = len(val_s1_ids) * (num_s2 + num_s3)
    reduction_ratio = 1.0 - (total_cands / total_possible) if total_possible > 0 else 1.0

    print("\n" + "=" * 65)
    print("METHOD 5: MULTI-KEY CASCADE BLOCKING AUDIT:")
    print(f"  Total True Matches:           {total_true:,}")
    print(f"  Captured in Candidate Pairs:  {captured:,}")
    print(f"  Candidate Recall Ceiling:     {recall_ceiling:.2%}")
    print(f"  Total Candidates Generated:   {total_cands:,}")
    print(f"  Avg Candidates / S1 Entity:   {avg_cands:.2f} (Target <= 6.0)")
    print(f"  Reduction Ratio:              {reduction_ratio:.6%}")
    print("=" * 65)

    return recall_ceiling, avg_cands, reduction_ratio
