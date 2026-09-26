"""
Feature Engineering Module for Business Entity Resolution.
Computes fast pairwise string, token, and numerical similarity metrics
using RapidFuzz and regex extraction.
"""

import re
import numpy as np
from rapidfuzz import fuzz


NUM_REGEX = re.compile(r"\b\d+\b")
WORD_REGEX = re.compile(r"\b[a-z0-9]+\b")


def extract_numbers(text: str) -> set:
    """Extract set of discrete numeric tokens (PIN codes, street numbers, door numbers)."""
    if not text:
        return set()
    return set(NUM_REGEX.findall(text))


def extract_words(text: str) -> set:
    """Extract set of alphanumeric word tokens."""
    if not text:
        return set()
    return set(WORD_REGEX.findall(text.lower()))


def extract_pairwise_features(s1_record: dict, cand_record: dict, tfidf_score: float = 0.0) -> list:
    """
    Extracts a dense feature vector comparing an S1 record and an S2/S3 candidate record.
    Returns: list of floats representing feature values.
    """
    name1 = s1_record.get("name", "")
    addr1 = s1_record.get("addr", "")
    name2 = cand_record.get("name", "")
    addr2 = cand_record.get("addr", "")

    # Cleaned strings for rapidfuzz
    n1_clean = name1.lower().strip()
    n2_clean = name2.lower().strip()
    a1_clean = addr1.lower().strip()
    a2_clean = addr2.lower().strip()

    # --- 1. Business Name Features ---
    name_token_sort = fuzz.token_sort_ratio(n1_clean, n2_clean) / 100.0
    name_partial = fuzz.partial_ratio(n1_clean, n2_clean) / 100.0
    name_w_ratio = fuzz.WRatio(n1_clean, n2_clean) / 100.0
    name_ratio = fuzz.ratio(n1_clean, n2_clean) / 100.0
    name_token_set = fuzz.token_set_ratio(n1_clean, n2_clean) / 100.0

    len1, len2 = len(n1_clean), len(n2_clean)
    name_len_diff = abs(len1 - len2) / max(len1, len2, 1)

    # --- 2. Address Features ---
    addr_token_set = fuzz.token_set_ratio(a1_clean, a2_clean) / 100.0
    addr_token_sort = fuzz.token_sort_ratio(a1_clean, a2_clean) / 100.0
    addr_partial = fuzz.partial_ratio(a1_clean, a2_clean) / 100.0

    words1 = extract_words(a1_clean)
    words2 = extract_words(a2_clean)
    common_words = words1 & words2
    addr_token_match_count = len(common_words)
    addr_jaccard = len(common_words) / len(words1 | words2) if (words1 or words2) else 0.0

    # --- 3. Number / PIN / ZIP Overlap Features ---
    nums1 = extract_numbers(addr1) | extract_numbers(name1)
    nums2 = extract_numbers(addr2) | extract_numbers(name2)
    common_nums = nums1 & nums2

    has_num_overlap = 1.0 if len(common_nums) > 0 else 0.0
    num_jaccard = len(common_nums) / len(nums1 | nums2) if (nums1 or nums2) else 0.0

    # --- 4. Cross & Composite Features ---
    combined_1 = f"{n1_clean} {a1_clean}"
    combined_2 = f"{n2_clean} {a2_clean}"
    comb_token_sort = fuzz.token_sort_ratio(combined_1, combined_2) / 100.0

    # Harmonic mean of name and address similarity
    harmonic_name_addr = (
        2.0 * name_token_sort * addr_token_sort / (name_token_sort + addr_token_sort)
        if (name_token_sort + addr_token_sort) > 0 else 0.0
    )

    return [
        name_token_sort,          # 0
        name_partial,             # 1
        name_w_ratio,             # 2
        name_ratio,               # 3
        name_token_set,           # 4
        name_len_diff,            # 5
        addr_token_set,           # 6
        addr_token_sort,          # 7
        addr_partial,             # 8
        float(addr_token_match_count), # 9
        addr_jaccard,             # 10
        has_num_overlap,          # 11
        num_jaccard,              # 12
        comb_token_sort,          # 13
        harmonic_name_addr,       # 14
        float(tfidf_score)        # 15
    ]


FEATURE_NAMES = [
    "name_token_sort",
    "name_partial_ratio",
    "name_w_ratio",
    "name_levenshtein_ratio",
    "name_token_set",
    "name_len_diff",
    "addr_token_set",
    "addr_token_sort",
    "addr_partial_ratio",
    "addr_token_match_count",
    "addr_jaccard",
    "has_num_overlap",
    "num_jaccard",
    "comb_token_sort",
    "harmonic_name_addr",
    "tfidf_blocking_score"
]


def build_pair_dataset(candidates_dict: dict, s1_lookup: dict, target_lookup: dict,
                       s1_ids_subset: set = None, gt_dict: dict = None):
    """
    Builds feature matrix X and label vector y for candidate pairs.
    If gt_dict is provided, generates binary ground truth labels (1 = match, 0 = non-match).
    Returns: X (np.ndarray), y (np.ndarray or None), pair_metadata (list of (s1_id, cand_id))
    """
    pairs_list = []
    features_list = []
    labels_list = []

    s1_keys = s1_ids_subset if s1_ids_subset is not None else candidates_dict.keys()

    for s1_id in s1_keys:
        cands = candidates_dict.get(s1_id, [])
        s1_rec = s1_lookup.get(s1_id)
        if not s1_rec:
            continue

        true_matches = gt_dict.get(s1_id, set()) if gt_dict is not None else set()

        for item in cands:
            if isinstance(item, (tuple, list)):
                cand_id, tfidf_score = item[0], item[1]
            else:
                cand_id, tfidf_score = item, 0.0

            cand_rec = target_lookup.get(cand_id)
            if not cand_rec:
                continue

            feats = extract_pairwise_features(s1_rec, cand_rec, tfidf_score)
            features_list.append(feats)
            pairs_list.append((s1_id, cand_id))

            if gt_dict is not None:
                labels_list.append(1 if cand_id in true_matches else 0)

    X = np.array(features_list, dtype=np.float32) if features_list else np.empty((0, len(FEATURE_NAMES)))
    y = np.array(labels_list, dtype=np.int32) if labels_list else None

    return X, y, pairs_list
