"""
Production Dual-Index Blocking Module (Upgraded Method 3).
Features:
  - Unicode NFKD accent and diacritic stripping for international datasets (France, US, India).
  - Dynamic legal noise stripping ('SAS', 'SARL', 'SA', 'Pvt Ltd', 'Inc', 'LLC', etc.).
  - Dual-Index: Core Token Inverted Index + Primary Prefix Index + Postal/House Number Anchors.
  - Generates strictly Top 5 candidates from Source 2 and Top 5 candidates from Source 3 (Total max 10).
"""

import re
import unicodedata
from collections import defaultdict
from rapidfuzz import fuzz

STOPWORDS = {
    'pvt', 'ltd', 'inc', 'corp', 'llc', 'limited', 'private', 'incorporated',
    'the', 'co', 'company', 'and', 'of', 'in', 'at', 'a', 'an', 'services',
    'group', 'enterprises', 'sa', 'sarl', 'sas', 'eurl', 'sci', 'ste',
    'societe', 'france', 'paris', 'solutions', 'industries', 'de', 'la', 'le',
    'les', 'du', 'des', 'en', 'et'
}


def strip_accents(text: str) -> str:
    """Strips accents and diacritics using Unicode NFKD decomposition."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    return "".join(c for c in text if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Fast text normalizer with Unicode accent stripping and legal noise removal."""
    if not isinstance(text, str):
        return ""
    text = strip_accents(text).lower()
    text = re.sub(
        r"\b(pvt|ltd|inc|corp|llc|limited|private|incorporated|the|co|sa|sarl|sas|eurl|sci|ste|societe|solutions|services|group|enterprises|enterprise)\b",
        " ",
        text
    )
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def get_tokens(name: str) -> list:
    """Extracts non-stopword tokens."""
    cleaned = clean_text(name)
    return [w for w in cleaned.split() if w not in STOPWORDS and len(w) > 1]


def extract_blocking_keys(name: str, addr: str) -> list:
    """
    Extracts high-precision blocking keys:
    1. Core 2-token key: 'b2_{w1}_{w2}'
    2. Primary token prefix: 'w1_{w1[:5]}'
    3. Postal code + brand token prefix: 'pin_{pincode}_{w1[:3]}'
    4. Street number + brand token prefix: 'num_{street_num}_{w1[:3]}'
    """
    keys = []
    toks = get_tokens(name)

    if len(toks) >= 2:
        keys.append(f"b2_{toks[0]}_{toks[1]}")
    if toks and len(toks[0]) >= 3:
        keys.append(f"w1_{toks[0][:5]}")

    addr_str = str(addr) if addr else ""
    if toks:
        w_pfx = toks[0][:3]
        # Postal code
        pins = re.findall(r"\b\d{5,}\b", addr_str)
        if pins:
            keys.append(f"pin_{pins[-1]}_{w_pfx}")
        # House/Street number
        nums = re.findall(r"\b\d+\b", addr_str)
        if nums:
            keys.append(f"num_{nums[0]}_{w_pfx}")

    return keys


class ProductionBlocker:
    """Inverted index dual-key blocker optimized for high throughput and recall."""

    def __init__(self, bucket_cap: int = 40):
        self.bucket_cap = bucket_cap

    def build_index(self, ids: list, names: list, addrs: list):
        """Builds an inverted index mapping key -> list of row indices."""
        inv = defaultdict(list)
        for i, (sid, name, addr) in enumerate(zip(ids, names, addrs)):
            for k in extract_blocking_keys(name, addr):
                inv[k].append(i)

        # Cap overly broad buckets to maintain precision and speed
        for k in inv:
            if len(inv[k]) > self.bucket_cap:
                inv[k] = inv[k][:self.bucket_cap]

        return inv

    def retrieve_top_k(self, s1_name: str, s1_addr: str,
                       target_index: dict, target_ids: list,
                       target_names: list, target_addrs: list,
                       top_k: int = 5) -> list:
        """
        Retrieves top K candidate IDs from a target source for an S1 entity.
        Scores candidate matches using rapid token overlap and lexical similarities.
        """
        keys = extract_blocking_keys(s1_name, s1_addr)
        candidate_indices = set()
        for k in keys:
            if k in target_index:
                candidate_indices.update(target_index[k])

        if not candidate_indices:
            return []

        clean_s1_name = clean_text(s1_name)
        clean_s1_addr = clean_text(s1_addr)
        s1_comb = f"{clean_s1_name} {clean_s1_addr}"

        scored = []
        for idx in candidate_indices:
            t_name = clean_text(target_names[idx])
            t_addr = clean_text(target_addrs[idx])
            t_comb = f"{t_name} {t_addr}"

            # Rapid lexical scoring
            sim_name = fuzz.token_sort_ratio(clean_s1_name, t_name) / 100.0
            sim_comb = fuzz.token_sort_ratio(s1_comb, t_comb) / 100.0
            score = 0.60 * sim_name + 0.40 * sim_comb

            scored.append((target_ids[idx], score, idx))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
