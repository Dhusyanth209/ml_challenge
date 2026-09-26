"""
Advanced Feature Engineering Module (Method 3).
Extends Method 2 features with domain-specific signals:
  1. exact_postal_code_match (1: match, -1: mismatch, 0: missing)
  2. token_containment_ratio (core name subset containment)
  3. address_number_match (leading street/door number match: 1, -1, 0)
"""

import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


NUM_REGEX = re.compile(r"\b\d+\b")
WORD_REGEX = re.compile(r"\b[a-z0-9]+\b")
POSTAL_REGEX = re.compile(r"\b(\d{5}|\d{6})\b")
LEADING_NUM_REGEX = re.compile(r"^\D*(\d+)")


def extract_numbers(text: str) -> set:
    """Extracts set of numeric tokens."""
    if not text:
        return set()
    return set(NUM_REGEX.findall(text))


def extract_first_token(text: str) -> str:
    """Extracts first alphanumeric token."""
    if not text:
        return ""
    words = WORD_REGEX.findall(text.lower())
    return words[0] if words else ""


def extract_postal_code(text: str) -> str:
    """Extracts 5 or 6 digit postal / ZIP / PIN code."""
    if not text:
        return ""
    matches = POSTAL_REGEX.findall(text)
    return matches[0] if matches else ""


def extract_leading_number(text: str) -> str:
    """Extracts leading street/door number."""
    if not text:
        return ""
    m = LEADING_NUM_REGEX.search(text)
    return m.group(1) if m else ""


def extract_m3_features(s1_record: dict, cand_record: dict, tfidf_score: float = 0.0) -> list:
    """
    Computes 15 advanced features for Method 3.
    """
    n1 = s1_record.get("name", "").lower().strip()
    a1 = s1_record.get("addr", "").lower().strip()
    n2 = cand_record.get("name", "").lower().strip()
    a2 = cand_record.get("addr", "").lower().strip()

    # ---------------------------------------------------------
    # 1. Base RapidFuzz Name & Address Metrics
    # ---------------------------------------------------------
    name_ratio = fuzz.ratio(n1, n2) / 100.0
    name_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    name_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    name_w_ratio = fuzz.WRatio(n1, n2) / 100.0
    name_jaro_winkler = float(JaroWinkler.similarity(n1, n2))
    name_len_diff = float(abs(len(n1) - len(n2)))

    w1_first = extract_first_token(n1)
    w2_first = extract_first_token(n2)
    exact_first_token_match = 1.0 if (w1_first and w1_first == w2_first) else 0.0

    address_ratio = fuzz.ratio(a1, a2) / 100.0
    address_token_set = fuzz.token_set_ratio(a1, a2) / 100.0

    nums1 = extract_numbers(a1) | extract_numbers(n1)
    nums2 = extract_numbers(a2) | extract_numbers(n2)
    num_overlap = float(len(nums1 & nums2))

    comb_1 = f"{n1} {a1}"
    comb_2 = f"{n2} {a2}"
    comb_token_sort = fuzz.token_sort_ratio(comb_1, comb_2) / 100.0

    # ---------------------------------------------------------
    # 2. Domain-Specific Method 3 Expansions
    # ---------------------------------------------------------
    # A. Exact postal code match (1 = match, -1 = mismatch, 0 = missing)
    post1 = extract_postal_code(a1)
    post2 = extract_postal_code(a2)
    if post1 and post2:
        exact_postal_code_match = 1.0 if post1 == post2 else -1.0
    else:
        exact_postal_code_match = 0.0

    # B. Token containment ratio (core name of S1 inside S2/S3)
    words1 = set(WORD_REGEX.findall(n1))
    words2 = set(WORD_REGEX.findall(n2))
    if words1:
        token_containment_ratio = float(len(words1 & words2) / len(words1))
    else:
        token_containment_ratio = 0.0

    # C. Address leading house/street number match (1 = match, -1 = mismatch, 0 = missing)
    num_addr1 = extract_leading_number(a1)
    num_addr2 = extract_leading_number(a2)
    if num_addr1 and num_addr2:
        address_number_match = 1.0 if num_addr1 == num_addr2 else -1.0
    else:
        address_number_match = 0.0

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
        exact_postal_code_match,    # 11
        token_containment_ratio,    # 12
        address_number_match,       # 13
        float(tfidf_score)          # 14
    ]


FEATURE_NAMES_M3 = [
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
    "exact_postal_code_match",
    "token_containment_ratio",
    "address_number_match",
    "tfidf_blocking_score"
]


def build_m3_feature_dataset(candidates_dict: dict, s1_lookup: dict, target_lookup: dict,
                             s1_ids_subset: set = None, gt_dict: dict = None):
    """
    Builds (X, y, pair_keys) for candidate pairs in s1_ids_subset.
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

            feats = extract_m3_features(s1_rec, t_rec, score)
            X_list.append(feats)
            pair_keys.append((sid, cid))

            if gt_dict is not None:
                y_list.append(1 if cid in true_matches else 0)

    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, len(FEATURE_NAMES_M3)))
    y = np.array(y_list, dtype=np.int32) if y_list else None

    return X, y, pair_keys
