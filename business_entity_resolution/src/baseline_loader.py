"""
Baseline Data Loader Module (Method 1).
Handles robust tab-separated loading of business records and ground truth,
and generates an 80/20 stratified validation split strictly preserving singleton proportions.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_tsv(filepath: str, columns: list = None, nrows: int = None, chunksize: int = None):
    """
    Load a tab-separated file with robust NA handling and memory efficiency.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    read_kwargs = {
        "sep": "\t",
        "dtype": str,
        "keep_default_na": False,
        "na_values": [],
        "usecols": columns,
        "nrows": nrows,
        "on_bad_lines": "skip",
        "encoding": "utf-8"
    }

    if chunksize is not None and chunksize > 0:
        return pd.read_csv(filepath, chunksize=chunksize, **read_kwargs)
    return pd.read_csv(filepath, **read_kwargs)


def load_ground_truth(gt_filepath: str, nrows: int = None):
    """
    Load ground truth mapping:
    s1_to_matches = {s1_id: set([matched_id, ...])}
    """
    df = load_tsv(gt_filepath, columns=["source1_entity_id", "matched_entity_ids"], nrows=nrows)
    s1_to_matches = {}
    for s1_id, matches_str in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        s1_id = s1_id.strip()
        matches_str = matches_str.strip()
        if matches_str:
            s1_to_matches[s1_id] = set(m.strip() for m in matches_str.split(",") if m.strip())
        else:
            s1_to_matches[s1_id] = set()
    return s1_to_matches, df


def load_coherent_sample(dataset_dir: str, sample_size: int = 5000):
    """
    Loads a coherent sample of S1 records with their true matching S2 and S3 pairs intact,
    plus negative distractor pool for realistic candidate generation and matching.
    """
    train_dir = os.path.join(dataset_dir, "train")
    gt_file = os.path.join(train_dir, "train_ground_truth.tsv")
    s1_file = os.path.join(train_dir, "train_source1.tsv")
    s2_file = os.path.join(train_dir, "train_source2.tsv")
    s3_file = os.path.join(train_dir, "train_source3.tsv")

    print(f"Loading coherent sample of {sample_size:,} S1 records...")
    gt_dict, _ = load_ground_truth(gt_file, nrows=sample_size)
    sample_s1_ids = set(gt_dict.keys())

    target_s2_needed = set()
    target_s3_needed = set()
    for matches in gt_dict.values():
        for mid in matches:
            if mid.startswith("S2-"):
                target_s2_needed.add(mid)
            elif mid.startswith("S3-"):
                target_s3_needed.add(mid)

    # 1. Load S1
    s1_chunks = []
    for chunk in pd.read_csv(s1_file, sep="\t", chunksize=200000, dtype=str, keep_default_na=False):
        sub = chunk[chunk["entity_id"].isin(sample_s1_ids)]
        if len(sub) > 0:
            s1_chunks.append(sub)
        if sum(len(c) for c in s1_chunks) >= len(sample_s1_ids):
            break
    s1_df = pd.concat(s1_chunks, ignore_index=True) if s1_chunks else pd.DataFrame()

    # 2. Load S2 (needed matches + distractors)
    s2_chunks = []
    neg_quota_s2 = sample_size * 2
    for chunk in pd.read_csv(s2_file, sep="\t", chunksize=300000, dtype=str, keep_default_na=False):
        needed = chunk[chunk["entity_id"].isin(target_s2_needed)]
        if len(needed) > 0:
            s2_chunks.append(needed)
            target_s2_needed -= set(needed["entity_id"])
        if neg_quota_s2 > 0:
            distractors = chunk[~chunk["entity_id"].isin(target_s2_needed)].head(neg_quota_s2)
            s2_chunks.append(distractors)
            neg_quota_s2 -= len(distractors)
        if len(target_s2_needed) == 0 and neg_quota_s2 <= 0:
            break
    s2_df = pd.concat(s2_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])

    # 3. Load S3 (needed matches + distractors)
    s3_chunks = []
    neg_quota_s3 = sample_size * 2
    for chunk in pd.read_csv(s3_file, sep="\t", chunksize=300000, dtype=str, keep_default_na=False):
        needed = chunk[chunk["entity_id"].isin(target_s3_needed)]
        if len(needed) > 0:
            s3_chunks.append(needed)
            target_s3_needed -= set(needed["entity_id"])
        if neg_quota_s3 > 0:
            distractors = chunk[~chunk["entity_id"].isin(target_s3_needed)].head(neg_quota_s3)
            s3_chunks.append(distractors)
            neg_quota_s3 -= len(distractors)
        if len(target_s3_needed) == 0 and neg_quota_s3 <= 0:
            break
    s3_df = pd.concat(s3_chunks, ignore_index=True).drop_duplicates(subset=["entity_id"])

    singletons = sum(1 for m in gt_dict.values() if len(m) == 0)
    print(f"Sample loaded successfully:")
    print(f"  S1: {len(s1_df):,} records ({singletons:,} singletons, {singletons/len(s1_df):.2%})")
    print(f"  S2: {len(s2_df):,} records")
    print(f"  S3: {len(s3_df):,} records")
    return s1_df, s2_df, s3_df, gt_dict


def load_train_data(dataset_dir: str, sample_size: int = None):
    """
    Loads train_source1, train_source2, train_source3, and train_ground_truth.
    """
    if sample_size is not None and sample_size > 0:
        return load_coherent_sample(dataset_dir, sample_size=sample_size)

    train_dir = os.path.join(dataset_dir, "train")
    print(f"Loading full training datasets from: {train_dir}")
    s1_df = load_tsv(os.path.join(train_dir, "train_source1.tsv"))
    s2_df = load_tsv(os.path.join(train_dir, "train_source2.tsv"))
    s3_df = load_tsv(os.path.join(train_dir, "train_source3.tsv"))
    gt_dict, _ = load_ground_truth(os.path.join(train_dir, "train_ground_truth.tsv"))

    singletons = sum(1 for m in gt_dict.values() if len(m) == 0)
    print(f"  Loaded Source 1: {len(s1_df):,} records")
    print(f"  Loaded Source 2: {len(s2_df):,} records")
    print(f"  Loaded Source 3: {len(s3_df):,} records")
    print(f"  Loaded Ground Truth: {len(gt_dict):,} S1 entities ({singletons:,} singletons, {singletons/len(gt_dict):.2%})")
    return s1_df, s2_df, s3_df, gt_dict


def create_stratified_split(s1_df: pd.DataFrame, gt_dict: dict, val_ratio: float = 0.20, random_state: int = 42):
    """
    Splits Source 1 records into 80% train and 20% holdout validation,
    stratifying by (country + is_singleton) to preserve singleton and country proportions.
    """
    s1_ids = s1_df["entity_id"].values
    countries = s1_df["country"].fillna("UNKNOWN").values
    is_singleton = np.array([len(gt_dict.get(sid, set())) == 0 for sid in s1_ids])

    strat_key = [f"{c}_{'singleton' if s else 'matched'}" for c, s in zip(countries, is_singleton)]

    train_ids, val_ids = train_test_split(
        s1_ids,
        test_size=val_ratio,
        random_state=random_state,
        stratify=strat_key
    )

    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    val_singletons = sum(1 for sid in val_ids if len(gt_dict.get(sid, set())) == 0)
    print(f"\nStratified 80/20 Split Created:")
    print(f"  Train S1 records: {len(train_ids):,} ({len(train_ids)/len(s1_ids):.1%})")
    print(f"  Val S1 records:   {len(val_ids):,} ({len(val_ids)/len(s1_ids):.1%})")
    print(f"  Val Singleton Ratio: {val_singletons/len(val_ids):.2%}")

    return train_ids_set, val_ids_set
