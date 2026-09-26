"""
Dense + Lexical Hybrid Feature Extraction & Scoring Module (Method 4).
Combines dense cosine similarity from the bi-encoder blocking stage
with RapidFuzz lexical features, then trains a LightGBM classifier.
"""

import re
import numpy as np
from rapidfuzz import fuzz


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\b(pvt|ltd|inc|corp|llc|limited|private|incorporated|the|co)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def extract_street_numbers(text: str):
    return set(re.findall(r"\b\d+\b", text))


FEATURE_NAMES_M4 = [
    "dense_cosine_sim",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_w_ratio",
    "name_ratio",
    "name_jaro_winkler",
    "name_partial_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_ratio",
    "addr_w_ratio",
    "comb_token_sort",
    "name_len_diff",
    "street_number_match",
    "token_containment_ratio",
    "exact_postal_code_match",
]


def extract_postal_code(text: str):
    """Extract likely postal/zip codes (5+ digit sequences)."""
    codes = re.findall(r"\b\d{5,}\b", text)
    return codes[-1] if codes else None


def compute_pairwise_features(s1_name, s1_addr, t_name, t_addr, dense_score):
    """Compute 16 features for a single (S1, candidate) pair."""
    cn1 = clean_text(s1_name)
    cn2 = clean_text(t_name)
    ca1 = clean_text(s1_addr)
    ca2 = clean_text(t_addr)

    # Dense cosine similarity from FAISS blocking
    f_dense = dense_score

    # Name features
    f_name_tsr = fuzz.token_sort_ratio(cn1, cn2) / 100.0
    f_name_tsetr = fuzz.token_set_ratio(cn1, cn2) / 100.0
    f_name_wr = fuzz.WRatio(cn1, cn2) / 100.0
    f_name_r = fuzz.ratio(cn1, cn2) / 100.0
    f_name_jw = fuzz.ratio(cn1[:20], cn2[:20]) / 100.0  # prefix-weighted proxy
    f_name_pr = fuzz.partial_ratio(cn1, cn2) / 100.0

    # Address features
    f_addr_tsr = fuzz.token_sort_ratio(ca1, ca2) / 100.0
    f_addr_tsetr = fuzz.token_set_ratio(ca1, ca2) / 100.0
    f_addr_r = fuzz.ratio(ca1, ca2) / 100.0
    f_addr_wr = fuzz.WRatio(ca1, ca2) / 100.0

    # Combined
    comb1 = f"{cn1} {ca1}"
    comb2 = f"{cn2} {ca2}"
    f_comb_tsr = fuzz.token_sort_ratio(comb1, comb2) / 100.0

    # Name length difference (normalized)
    f_name_len_diff = abs(len(cn1) - len(cn2)) / max(len(cn1), len(cn2), 1)

    # Street number match
    nums1 = extract_street_numbers(ca1)
    nums2 = extract_street_numbers(ca2)
    if nums1 and nums2:
        f_street_num = 1.0 if nums1 & nums2 else -1.0
    else:
        f_street_num = 0.0

    # Token containment ratio
    tokens1 = set(cn1.split())
    tokens2 = set(cn2.split())
    if tokens1:
        f_containment = len(tokens1 & tokens2) / len(tokens1)
    else:
        f_containment = 0.0

    # Exact postal code match
    pc1 = extract_postal_code(ca1)
    pc2 = extract_postal_code(ca2)
    if pc1 and pc2:
        f_postal = 1.0 if pc1 == pc2 else -1.0
    else:
        f_postal = 0.0

    return [
        f_dense,
        f_name_tsr, f_name_tsetr, f_name_wr, f_name_r, f_name_jw, f_name_pr,
        f_addr_tsr, f_addr_tsetr, f_addr_r, f_addr_wr,
        f_comb_tsr, f_name_len_diff, f_street_num, f_containment, f_postal
    ]


def build_m4_feature_dataset(candidates_dict, s1_lookup, target_lookup,
                             s1_ids_subset=None, gt_dict=None):
    """
    Build feature matrix from candidate pairs.
    Returns: X (np.array), y (np.array or None), pairs list [(s1_id, cand_id), ...]
    """
    all_features = []
    all_labels = []
    all_pairs = []

    s1_ids = s1_ids_subset if s1_ids_subset else list(candidates_dict.keys())

    for sid in s1_ids:
        cands = candidates_dict.get(sid, [])
        if not cands:
            continue

        s1_info = s1_lookup.get(sid, {"name": "", "addr": ""})

        for cid, dense_score in cands:
            t_info = target_lookup.get(cid, {"name": "", "addr": ""})
            feats = compute_pairwise_features(
                s1_info["name"], s1_info["addr"],
                t_info["name"], t_info["addr"],
                dense_score
            )
            all_features.append(feats)
            all_pairs.append((sid, cid))

            if gt_dict is not None:
                true_matches = gt_dict.get(sid, set())
                all_labels.append(1 if cid in true_matches else 0)

    X = np.array(all_features, dtype=np.float32) if all_features else np.empty((0, len(FEATURE_NAMES_M4)), dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32) if all_labels else None
    return X, y, all_pairs
