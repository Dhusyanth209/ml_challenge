import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import logging

logger = logging.getLogger(__name__)


def get_number_conflict(num_str1, num_str2):
    if not num_str1 or not num_str2:
        return 0  # Neutral, one or both missing

    set1 = set(num_str1.split())
    set2 = set(num_str2.split())

    if set1 & set2:
        return 1  # Positive signal: they share at least one number
    return -1  # Negative signal: both have numbers but they don't overlap


def compute_features(candidates_df, s1_df, s23_pool_df):
    logger.info("Computing features for candidates...")

    # Merge data to get strings for both sides
    df = candidates_df.merge(
        s1_df[['entity_id', 'name_core', 'address_clean', 'address_numbers', 'country']],
        left_on='source1_entity_id',
        right_on='entity_id'
    ).rename(columns={
        'name_core': 'name_1',
        'address_clean': 'address_1',
        'address_numbers': 'num_1'
    }).drop(columns=['entity_id'])

    df = df.merge(
        s23_pool_df[['entity_id', 'name_core', 'address_clean', 'address_numbers']],
        left_on='candidate_entity_id',
        right_on='entity_id'
    ).rename(columns={
        'name_core': 'name_2',
        'address_clean': 'address_2',
        'address_numbers': 'num_2'
    }).drop(columns=['entity_id'])

    # Pull columns out to plain Python lists once. rapidfuzz is C-level, so
    # list comprehensions over millions of pairs are far faster than the
    # previous per-row DataFrame.apply(axis=1) calls (which ran the Python
    # interpreter once per row, per feature).
    n1 = df['name_1'].fillna('').astype(str).tolist()
    n2 = df['name_2'].fillna('').astype(str).tolist()
    a1 = df['address_1'].fillna('').astype(str).tolist()
    a2 = df['address_2'].fillna('').astype(str).tolist()
    num1 = df['num_1'].fillna('').astype(str).tolist()
    num2 = df['num_2'].fillna('').astype(str).tolist()

    # Feature 1: Name Jaro-Winkler (prefix-weighted, good for names)
    df['name_jw'] = [JaroWinkler.similarity(x, y) for x, y in zip(n1, n2)]

    # Feature 2: Name Token Sort Ratio (handles word-order transpositions)
    df['name_ts'] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]

    # Feature 3: Address Jaro-Winkler
    df['addr_jw'] = [JaroWinkler.similarity(x, y) for x, y in zip(a1, a2)]

    # Feature 4: Address Token Set Ratio
    df['addr_tset'] = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(a1, a2)]

    # Feature 5: Numerical conflict (shared/conflicting street numbers etc.)
    df['num_conflict'] = [get_number_conflict(x, y) for x, y in zip(num1, num2)]

    # Feature 6: Substring match in name (common with DBA / trade names)
    df['name_subset'] = [
        1 if len(x) > 3 and (x in y or y in x) else 0
        for x, y in zip(n1, n2)
    ]

    # Feature 7: Length ratio of names
    l1 = np.fromiter((len(x) for x in n1), dtype=np.int64, count=len(n1))
    l2 = np.fromiter((len(x) for x in n2), dtype=np.int64, count=len(n2))
    df['len_ratio'] = np.minimum(l1, l2) / np.maximum(1, np.maximum(l1, l2))

    return df
