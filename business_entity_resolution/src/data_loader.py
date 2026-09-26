"""
Data Loader Module for Business Entity Resolution.
Handles loading large TSVs with proper tab-separation, chunking/memory-efficiency,
and generates stratified train/validation splits preserving country and singleton distribution.
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


def load_all_s1_ids(filepath: str) -> list:
    """Quickly loads all Source 1 entity IDs from test_source1.tsv."""
    df = pd.read_csv(
        filepath, sep="\t", usecols=["entity_id"],
        dtype=str, keep_default_na=False, on_bad_lines="skip"
    )
    return df["entity_id"].tolist()


def load_ground_truth(gt_filepath: str, nrows: int = None):
    """
    Load ground truth and parse into:
    1. s1_to_matches: dict mapping S1 ID to set of matching S2/S3 IDs.
    2. gt_df: raw DataFrame.
    """
    df = load_tsv(gt_filepath, columns=["source1_entity_id", "matched_entity_ids"], nrows=nrows)
    s1_to_matches = {}
    for s1_id, matches_str in zip(df["source1_entity_id"], df["matched_entity_ids"]):
        if matches_str.strip():
            s1_to_matches[s1_id] = set(m.strip() for m in matches_str.split(",") if m.strip())
        else:
            s1_to_matches[s1_id] = set()
    return s1_to_matches, df


def load_dataset_sample(dataset_dir: str, split: str = "train", sample_size: int = 5000):
    """
    Loads a coherent sample where Source 1 entities have their true corresponding
    matches preserved from Source 2 and Source 3, plus realistic negative distractors.
    """
    split_dir = os.path.join(dataset_dir, split)
    s1_file = os.path.join(split_dir, f"{split}_source1.tsv")
    s2_file = os.path.join(split_dir, f"{split}_source2.tsv")
    s3_file = os.path.join(split_dir, f"{split}_source3.tsv")

    if split == "train":
        gt_file = os.path.join(split_dir, "train_ground_truth.tsv")
        gt_dict, _ = load_ground_truth(gt_file, nrows=sample_size)
        sample_s1_ids = set(gt_dict.keys())

        # Collect all true matched S2 and S3 IDs
        target_s2_needed = set()
        target_s3_needed = set()
        for matches in gt_dict.values():
            for mid in matches:
                if mid.startswith("S2-"):
                    target_s2_needed.add(mid)
                elif mid.startswith("S3-"):
                    target_s3_needed.add(mid)

        # Load S1 records corresponding to sample_s1_ids
        s1_chunks = []
        for chunk in pd.read_csv(s1_file, sep="\t", chunksize=200000, dtype=str, keep_default_na=False):
            sub = chunk[chunk["entity_id"].isin(sample_s1_ids)]
            if len(sub) > 0:
                s1_chunks.append(sub)
            if sum(len(c) for c in s1_chunks) >= len(sample_s1_ids):
                break
        s1_df = pd.concat(s1_chunks, ignore_index=True) if s1_chunks else pd.DataFrame()

        # Load S2 records: needed true matches + distractors
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

        # Load S3 records: needed true matches + distractors
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
        print(f"Loaded Coherent Sample ({split}):")
        print(f"  S1: {len(s1_df):,} records ({singletons:,} singletons, {singletons/len(s1_df):.2%})")
        print(f"  S2: {len(s2_df):,} records")
        print(f"  S3: {len(s3_df):,} records")
        return s1_df, s2_df, s3_df, gt_dict, None
    else:
        # Test split sample
        s1_df = load_tsv(s1_file, nrows=sample_size)
        s2_df = load_tsv(s2_file, nrows=sample_size * 2)
        s3_df = load_tsv(s3_file, nrows=sample_size * 2)
        return s1_df, s2_df, s3_df, None, None


def load_dataset(dataset_dir: str, split: str = "train", nrows: int = None):
    """
    Loads source1, source2, source3 for a given split ('train' or 'test').
    If nrows is specified, uses load_dataset_sample to ensure ground truth matches are intact.
    """
    if nrows is not None and nrows > 0:
        return load_dataset_sample(dataset_dir, split=split, sample_size=nrows)

    split_dir = os.path.join(dataset_dir, split)
    s1_file = os.path.join(split_dir, f"{split}_source1.tsv")
    s2_file = os.path.join(split_dir, f"{split}_source2.tsv")
    s3_file = os.path.join(split_dir, f"{split}_source3.tsv")

    print(f"Loading {split} datasets from: {split_dir}")
    s1_df = load_tsv(s1_file)
    s2_df = load_tsv(s2_file)
    s3_df = load_tsv(s3_file)

    print(f"  Loaded Source 1: {len(s1_df):,} records")
    print(f"  Loaded Source 2: {len(s2_df):,} records")
    print(f"  Loaded Source 3: {len(s3_df):,} records")

    gt_dict, gt_df = None, None
    if split == "train":
        gt_file = os.path.join(split_dir, "train_ground_truth.tsv")
        if os.path.exists(gt_file):
            gt_dict, gt_df = load_ground_truth(gt_file)
            singletons = sum(1 for m in gt_dict.values() if len(m) == 0)
            print(f"  Loaded Ground Truth: {len(gt_dict):,} S1 entities ({singletons:,} singletons, {singletons/len(gt_dict):.2%})")

    return s1_df, s2_df, s3_df, gt_dict, gt_df


def create_stratified_val_split(s1_df: pd.DataFrame, gt_dict: dict, val_ratio: float = 0.2, random_state: int = 42):
    """
    Creates an 80/20 train/validation split on Source 1 entities stratified by:
    (country + is_singleton), ensuring validation accurately reflects test conditions.
    """
    s1_ids = s1_df["entity_id"].values
    countries = s1_df["country"].fillna("UNKNOWN").values
    is_singleton = np.array([len(gt_dict.get(sid, set())) == 0 for sid in s1_ids])

    strat_key = [f"{c}_{'singleton' if s else 'has_match'}" for c, s in zip(countries, is_singleton)]

    train_ids, val_ids = train_test_split(
        s1_ids,
        test_size=val_ratio,
        random_state=random_state,
        stratify=strat_key
    )

    train_ids_set = set(train_ids)
    val_ids_set = set(val_ids)

    print(f"Stratified Split Created:")
    print(f"  Train S1 count: {len(train_ids):,} ({len(train_ids)/len(s1_ids):.1%})")
    print(f"  Val S1 count:   {len(val_ids):,} ({len(val_ids)/len(s1_ids):.1%})")

    val_singletons = sum(1 for sid in val_ids if len(gt_dict.get(sid, set())) == 0)
    print(f"  Val Singleton Ratio: {val_singletons/len(val_ids):.2%}")

    return train_ids_set, val_ids_set


def build_entity_lookup(df: pd.DataFrame):
    """
    Converts DataFrame to dictionary for ultra-fast O(1) field lookup:
    {entity_id: {'name': business_name, 'addr': business_address, 'country': country}}
    """
    lookup = {}
    for row in df.itertuples(index=False):
        lookup[row.entity_id] = {
            "name": getattr(row, "business_name", ""),
            "addr": getattr(row, "business_address", ""),
            "country": getattr(row, "country", "")
        }
    return lookup
