"""
Dense Transformer Bi-Encoder Blocking Module (Method 4).
Uses sentence-transformers/all-MiniLM-L6-v2 for dense embeddings
and FAISS IndexFlatIP (cosine similarity after L2 normalization)
for fast approximate nearest-neighbor candidate retrieval.
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

import gc
import re
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\b(pvt|ltd|inc|corp|llc|limited|private|incorporated|the|co)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def format_entity_text(names, addresses):
    """Format input: 'name: {cleaned_name} | address: {cleaned_address}'"""
    texts = []
    for n, a in zip(names, addresses):
        cn = clean_text(n)
        ca = clean_text(a)
        texts.append(f"name: {cn} | address: {ca}")
    return texts


def encode_batch(model, texts, batch_size=256, show_progress=False):
    """Encode texts in batches and L2-normalize for cosine similarity."""
    embeddings = model.encode(
        texts, batch_size=batch_size, show_progress_bar=show_progress,
        normalize_embeddings=True, convert_to_numpy=True
    )
    return embeddings.astype(np.float32)


def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS inner-product index (cosine sim on L2-normed vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def query_faiss_top_k(index, query_embeddings, target_ids, top_k=3):
    """Query FAISS index and return [(cand_id, score), ...] per query."""
    scores, indices = index.search(query_embeddings, top_k)
    results = []
    for i in range(len(query_embeddings)):
        row = []
        for j in range(top_k):
            idx = indices[i, j]
            if idx >= 0:  # valid result
                row.append((target_ids[idx], float(scores[i, j])))
        results.append(row)
    return results


def run_dense_blocking(s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame,
                       model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                       top_k_per_source: int = 3, batch_size: int = 256):
    """
    Dense bi-encoder blocking with FAISS cosine ANN.
    Returns: {s1_id: [(cand_id, cosine_score), ...]} max 2*top_k per S1.
    """
    print(f"  Loading model: {model_name}")
    model = SentenceTransformer(model_name)

    final_candidates = {sid: [] for sid in s1_df["entity_id"]}
    countries = s1_df["country"].fillna("UNKNOWN").unique()
    print(f"  Partitioning by {len(countries)} countries: {list(countries)}")

    for country in countries:
        s1_sub = s1_df[s1_df["country"] == country]
        s2_sub = s2_df[s2_df["country"] == country]
        s3_sub = s3_df[s3_df["country"] == country]

        n1, n2, n3 = len(s1_sub), len(s2_sub), len(s3_sub)
        print(f"\n  --- Partition '{country}': S1={n1:,}, S2={n2:,}, S3={n3:,} ---")
        if n1 == 0:
            continue

        s1_ids = s1_sub["entity_id"].values
        s1_texts = format_entity_text(s1_sub["business_name"].values, s1_sub["business_address"].values)

        print(f"    Encoding {n1:,} S1 entities...")
        s1_embs = encode_batch(model, s1_texts, batch_size=batch_size)

        # --- Source 2 ---
        if n2 > 0:
            s2_ids = s2_sub["entity_id"].values
            s2_texts = format_entity_text(s2_sub["business_name"].values, s2_sub["business_address"].values)
            print(f"    Encoding {n2:,} S2 entities...")
            s2_embs = encode_batch(model, s2_texts, batch_size=batch_size)
            print(f"    Building FAISS index for S2...")
            s2_index = build_faiss_index(s2_embs)
            print(f"    Querying top {top_k_per_source} S2 candidates...")
            s2_results = query_faiss_top_k(s2_index, s1_embs, s2_ids, top_k=top_k_per_source)
            del s2_embs, s2_index
            gc.collect()
        else:
            s2_results = [[] for _ in range(n1)]

        # --- Source 3 ---
        if n3 > 0:
            s3_ids = s3_sub["entity_id"].values
            s3_texts = format_entity_text(s3_sub["business_name"].values, s3_sub["business_address"].values)
            print(f"    Encoding {n3:,} S3 entities...")
            s3_embs = encode_batch(model, s3_texts, batch_size=batch_size)
            print(f"    Building FAISS index for S3...")
            s3_index = build_faiss_index(s3_embs)
            print(f"    Querying top {top_k_per_source} S3 candidates...")
            s3_results = query_faiss_top_k(s3_index, s1_embs, s3_ids, top_k=top_k_per_source)
            del s3_embs, s3_index
            gc.collect()
        else:
            s3_results = [[] for _ in range(n1)]

        # Merge
        for i, sid in enumerate(s1_ids):
            final_candidates[sid] = s2_results[i] + s3_results[i]

        del s1_embs
        gc.collect()

    del model
    gc.collect()

    return final_candidates


def save_candidate_pairs_m4(candidates_dict: dict, output_file: str, all_s1_ids: list):
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            cands = candidates_dict.get(sid, [])
            cand_ids = [c[0] for c in cands]
            f.write(f"{sid}\t{','.join(cand_ids)}\n")
    print(f"  Exported candidate pairs to: {output_file}")


def evaluate_blocking_m4(candidates_dict: dict, gt_dict: dict, val_s1_ids: set,
                         num_s2: int, num_s3: int):
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
    print("METHOD 4: DENSE BI-ENCODER FAISS BLOCKING AUDIT:")
    print(f"  Total True Matches:           {total_true:,}")
    print(f"  Captured in Candidate Pairs:  {captured:,}")
    print(f"  Candidate Recall Ceiling:     {recall_ceiling:.2%}")
    print(f"  Total Candidates Generated:   {total_cands:,}")
    print(f"  Avg Candidates / S1 Entity:   {avg_cands:.2f} (Target <= 6.0)")
    print(f"  Reduction Ratio:              {reduction_ratio:.6%}")
    print("=" * 65)

    return recall_ceiling, avg_cands, reduction_ratio
