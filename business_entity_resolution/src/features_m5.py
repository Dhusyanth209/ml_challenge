"""
Domain Feature Extraction Module (Method 5).
Combines:
  1. High-precision RapidFuzz string metrics (Token Sort, WRatio, Token Set, Jaro-Winkler, Levenshtein)
  2. Domain features:
     - exact_pincode_match (1: match, -1: mismatch, 0: missing)
     - core_name_containment (1: inside, 0: otherwise)
     - street_number_overlap (count of matching numeric tokens)
     - brand_first_token_match (1: match, 0: otherwise)
  3. Blocking cumulative score
"""

import re
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

STOPWORDS = {
    'pvt', 'ltd', 'inc', 'corp', 'llc', 'limited', 'private', 'incorporated',
    'the', 'co', 'and', 'of', 'in', 'at', 'a', 'an'
}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\b(pvt|ltd|inc|corp|llc|limited|private|incorporated|the|co)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_postal_code(text: str):
    """Extracts 5+ digit postal code."""
    codes = re.findall(r"\b\d{5,}\b", str(text))
    return codes[-1] if codes else None


def extract_street_numbers(text: str):
    """Extracts house/street numbers."""
    return set(re.findall(r"\b\d+\b", str(text)))


def get_first_significant_token(name: str) -> str:
    cleaned = clean_text(name)
    tokens = [w for w in cleaned.split() if w not in STOPWORDS and len(w) > 1]
    return tokens[0] if tokens else ""


FEATURE_NAMES_M5 = [
    "blocking_score",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_w_ratio",
    "name_ratio",
    "name_jaro_winkler",
    "name_levenshtein_sim",
    "name_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_ratio",
    "addr_w_ratio",
    "comb_token_sort",
    "name_len_diff",
    "exact_pincode_match",
    "core_name_containment",
    "street_number_overlap",
    "brand_first_token_match"
]


def extract_m5_features(s1_name: str, s1_addr: str, t_name: str, t_addr: str, blocking_score: float = 0.0) -> list:
    cn1 = clean_text(s1_name)
    cn2 = clean_text(t_name)
    ca1 = clean_text(s1_addr)
    ca2 = clean_text(t_addr)

    # 1. Blocking score
    f_blocking = float(blocking_score)

    # 2. Name similarities
    f_name_tsr = fuzz.token_sort_ratio(cn1, cn2) / 100.0
    f_name_tsetr = fuzz.token_set_ratio(cn1, cn2) / 100.0
    f_name_wr = fuzz.WRatio(cn1, cn2) / 100.0
    f_name_r = fuzz.ratio(cn1, cn2) / 100.0
    f_name_jw = float(JaroWinkler.similarity(cn1, cn2))
    f_name_lev = float(Levenshtein.normalized_similarity(cn1, cn2))
    f_name_pr = fuzz.partial_ratio(cn1, cn2) / 100.0

    # 3. Address similarities
    f_addr_tsr = fuzz.token_sort_ratio(ca1, ca2) / 100.0
    f_addr_tsetr = fuzz.token_set_ratio(ca1, ca2) / 100.0
    f_addr_r = fuzz.ratio(ca1, ca2) / 100.0
    f_addr_wr = fuzz.WRatio(ca1, ca2) / 100.0

    # 4. Composite
    comb1 = f"{cn1} {ca1}"
    comb2 = f"{cn2} {ca2}"
    f_comb_tsr = fuzz.token_sort_ratio(comb1, comb2) / 100.0

    # 5. Length diff
    f_name_len_diff = abs(len(cn1) - len(cn2)) / max(len(cn1), len(cn2), 1)

    # 6. Exact pincode match
    pin1 = extract_postal_code(ca1)
    pin2 = extract_postal_code(ca2)
    if pin1 and pin2:
        f_pincode = 1.0 if pin1 == pin2 else -1.0
    else:
        f_pincode = 0.0

    # 7. Core name containment
    if cn1 and cn2:
        f_containment = 1.0 if (cn1 in cn2 or cn2 in cn1) else 0.0
    else:
        f_containment = 0.0

    # 8. Street number overlap
    nums1 = extract_street_numbers(ca1)
    nums2 = extract_street_numbers(ca2)
    if nums1 and nums2:
        overlap = len(nums1 & nums2)
        f_num_overlap = float(overlap) if overlap > 0 else -1.0
    else:
        f_num_overlap = 0.0

    # 9. Brand first token match
    tok1 = get_first_significant_token(s1_name)
    tok2 = get_first_significant_token(t_name)
    f_brand_tok = 1.0 if (tok1 and tok2 and tok1 == tok2) else 0.0

    return [
        f_blocking,
        f_name_tsr,
        f_name_tsetr,
        f_name_wr,
        f_name_r,
        f_name_jw,
        f_name_lev,
        f_name_pr,
        f_addr_tsr,
        f_addr_tsetr,
        f_addr_r,
        f_addr_wr,
        f_comb_tsr,
        f_name_len_diff,
        f_pincode,
        f_containment,
        f_num_overlap,
        f_brand_tok
    ]


def build_m5_feature_dataset(candidates_dict: dict, s1_lookup: dict, target_lookup: dict,
                             s1_ids_subset: set = None, gt_dict: dict = None):
    """
    Builds feature matrix from candidate pairs.
    Returns: X (np.ndarray), y (np.ndarray or None), pair_keys (list of tuples)
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

        for cid, sc in cands:
            t_rec = target_lookup.get(cid)
            if not t_rec:
                continue

            feats = extract_m5_features(
                s1_rec["name"], s1_rec["addr"],
                t_rec["name"], t_rec["addr"],
                blocking_score=sc
            )
            X_list.append(feats)
            pair_keys.append((sid, cid))

            if gt_dict is not None:
                y_list.append(1 if cid in true_matches else 0)

    X = np.array(X_list, dtype=np.float32) if X_list else np.empty((0, len(FEATURE_NAMES_M5)), dtype=np.float32)
    y = np.array(y_list, dtype=np.int32) if y_list else None

    return X, y, pair_keys
