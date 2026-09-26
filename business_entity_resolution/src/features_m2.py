"""
Feature Engineering Module (Method 2).
Extracts RapidFuzz string similarities, Jaro-Winkler distances, token ratios,
number sequence overlaps, and structural heuristics for pairwise candidate classification.
"""

import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


NUM_REGEX = re.compile(r"\b\d+\b")
WORD_REGEX = re.compile(r"\b[a-z0-9]+\b")


def extract_numbers(text: str) -> set:
    """Extract set of numerical tokens from text (house numbers, postal codes)."""
    if not text:
        return set()
    return set(NUM_REGEX.findall(text))


def extract_first_token(text: str) -> str:
    """Extracts first alphanumeric token."""
    if not text:
        return ""
    words = WORD_REGEX.findall(text.lower())
    return words[0] if words else ""


def extract_m2_features(s1_record: dict, cand_record: dict, tfidf_score: float = 0.0) -> list:
    """
    Computes exact Method 2 pairwise feature vector:
      1. name_ratio
      2. name_token_sort_ratio
      3. name_token_set_ratio
      4. name_w_ratio
      5. address_ratio
      6. address_token_set_ratio
      7. name_jaro_winkler
      8. num_overlap (count of exact matching digits/numbers)
      9. name_len_diff (absolute length difference)
      10. exact_first_token_match (binary 1/0)
      11. comb_token_sort (composite name+addr similarity)
      12. tfidf_blocking_score
    """
    n1 = s1_record.get("name", "").lower().strip()
    a1 = s1_record.get("addr", "").lower().strip()
    n2 = cand_record.get("name", "").lower().strip()
    a2 = cand_record.get("addr", "").lower().strip()

    # 1. Name features
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_w_ratio = fuzz.WRatio(n1, n2) / 100.0
    name_jaro_winkler = float(JaroWinkler.similarity(n1, n2))
    name_len_diff = float(abs(len(n1) - len(n2)))

    # First token match
    w1_first = extract_first_token(n1)
    w2_first = extract_first_token(n2)
    exact_first_token_match = 1.0 if (w1_first and w1_first == w2_first) else 0.0

    # 2. Address features
    address_ratio = fuzz.ratio(a1, a2) / 100.0
    address_token_set = fuzz.token_set_ratio(a1, a2) / 100.0

    # 3. Number overlap count
    nums1 = extract_numbers(a1) | extract_numbers(n1)
    nums2 = extract_numbers(a2) | extract_numbers(n2)
    common_nums = nums1 & nums2
    num_overlap = float(len(common_nums))

    # 4. Composite features
    comb_1 = f"{n1} {a1}"
    comb_2 = f"{n2} {a2}"
    comb_token_sort = fuzz.token_sort_ratio(comb_1, comb_2) / 100.0

    return [
        name_ratio,                 # 0
        name_token_sort,            # 1
        name_token_set,             # 2
        name_w_ratio,               # 3
        address_ratio,              # 4
        address_token_set,          # 5
        name_jaro_winkler,          # 6
        num_overlap,                # 7
        name_len_diff,              # 8
        exact_first_token_match,    # 9
        comb_token_sort,            # 10
        float(tfidf_score)          # 11
    ]


FEATURE_NAMES_M2 = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_w_ratio",
    "address_ratio",
    "address_token_set_ratio",
    "name_jaro_winkler",
    "num_overlap",
    "name_len_diff",
    "exact_first_token_match",
    "comb_token_sort",
    "tfidf_blocking_score"
]


def build_m2_feature_dataset(candidates_dict: dict, s1_lookup: dict, target_lookup: dict,
                             s1_ids_subset: set = None, gt_dict: dict = None):
    """
    Extracts features for all candidate pairs in s1_ids_subset.
    Returns: X (np.ndarray), y (np.ndarray or None), pair_keys (list of (s1_id, cand_id))
    """
    keys = s1_ids_subset if s1_ids_subset is not None else candidates_dict.keys()
    X_list = []
    y_list = []
    pair_keys = []

    for sid in keys:
        s1_rec = s1_lookup.get(sid)
        if not s1_rec:
            continue

        true_matches = gt_dict.get(sid, set()) if gt_dict is not None else set()
        cands = candidates_dict.get(sid, [])

        for cid, score in cands:
            t_rec = target_lookup.get(cid)
            if not t_rec:
                continue

            feats = extract_m2_features(s1_rec, t_rec, score)
            X_list.append(feats)
            pair_keys.append((sid, cid))

            if gt_dict is not None:
                y_list.append(1 if cid in true_matches else 0)

    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, len(FEATURE_NAMES_M2)))
    y = np.array(y_list, dtype=np.int32) if y_list else None

    return X, y, pair_keys
