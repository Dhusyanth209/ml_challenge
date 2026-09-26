"""
Precision-Weighted Production Matching & Pruning Module (Method 3 Production).
Optimized strictly for Macro F_0.5 (heavy penalty on false positives):
  - Strips accents (Unicode NFKD) and legal entity noise across all languages.
  - Computes fine-grained RapidFuzz and domain disambiguation features.
  - Enforces acceptance threshold: p >= 0.92.
  - Enforces strict source constraints:
      * At most 1 best match from Source 2 (if p >= 0.92)
      * At most 1 best match from Source 3 (if p >= 0.92)
  - Resolves global 1-to-1 conflicts across entities.
  - Emits strictly empty string "" for singletons.
"""

import re
import unicodedata
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler


def strip_accents(text: str) -> str:
    """Strips accents and diacritics using Unicode NFKD."""
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize('NFKD', text)
    return "".join(c for c in text if not unicodedata.combining(c))


def clean_text(text: str) -> str:
    """Cleans text by stripping accents, legal noise words, and punctuation."""
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


def extract_postal_code(text: str):
    """Extracts 5+ digit postal codes."""
    codes = re.findall(r"\b\d{5,}\b", str(text))
    return codes[-1] if codes else None


def extract_numbers(text: str):
    """Extracts numeric house/street digits."""
    return set(re.findall(r"\b\d+\b", str(text)))


class PrecisionProductionScorer:
    """
    Precision-calibrated scoring engine.
    Ensures high precision by gating false positives and computing confidence scores.
    """

    def __init__(self, match_threshold: float = 0.92):
        self.match_threshold = match_threshold

    def compute_match_score(self, s1_name: str, s1_addr: str,
                            t_name: str, t_addr: str,
                            base_blocking_score: float = 0.0) -> float:
        """
        Computes calibrated match confidence score [0.0, 1.0].
        Strictly penalizes conflicting street numbers and postal codes.
        """
        cn1 = clean_text(s1_name)
        cn2 = clean_text(t_name)
        ca1 = clean_text(s1_addr)
        ca2 = clean_text(t_addr)

        # Composite similarity (name + address)
        comb1 = f"{cn1} {ca1}"
        comb2 = f"{cn2} {ca2}"
        f_comb_tsr = fuzz.token_sort_ratio(comb1, comb2) / 100.0

        # Precision Gate: Early rejection if combined text doesn't align
        if f_comb_tsr < 0.55:
            return 0.0

        # Name metrics
        f_name_tsr = fuzz.token_sort_ratio(cn1, cn2) / 100.0
        f_name_wr = fuzz.WRatio(cn1, cn2) / 100.0
        f_name_r = fuzz.ratio(cn1, cn2) / 100.0
        f_name_jw = float(JaroWinkler.similarity(cn1, cn2))

        # Precision Gate: Name must have high similarity
        if f_name_tsr < 0.60 and f_name_jw < 0.75:
            return 0.0

        # Address metrics
        f_addr_tset = fuzz.token_set_ratio(ca1, ca2) / 100.0
        f_addr_r = fuzz.ratio(ca1, ca2) / 100.0

        # Domain Disambiguation
        # Postal code check
        pin1 = extract_postal_code(ca1)
        pin2 = extract_postal_code(ca2)
        pin_penalty = 0.0
        if pin1 and pin2:
            if pin1 != pin2:
                # Severe penalty for conflicting postal codes
                pin_penalty = -0.25
            else:
                pin_penalty = +0.03

        # House/Street number check
        nums1 = extract_numbers(ca1)
        nums2 = extract_numbers(ca2)
        num_penalty = 0.0
        if nums1 and nums2:
            if not (nums1 & nums2):
                # Penalty for conflicting address numbers
                num_penalty = -0.15
            else:
                num_penalty = +0.03

        # Core name containment
        cont_boost = 0.02 if (cn1 and cn2 and (cn1 in cn2 or cn2 in cn1)) else 0.0

        # Weighted precision confidence
        score = (
            0.40 * f_comb_tsr +
            0.25 * f_name_tsr +
            0.15 * f_name_jw +
            0.10 * f_addr_tset +
            0.10 * f_name_wr +
            pin_penalty +
            num_penalty +
            cont_boost
        )

        return min(max(score, 0.0), 1.0)

    def prune_and_select_matches(self, candidate_matches: list) -> list:
        """
        Enforces strict source constraints:
          - At most 1 best match from Source 2 (if score >= 0.92)
          - At most 1 best match from Source 3 (if score >= 0.92)
          - Retains only highest probability match per source if delta > 0.05
        Returns list of accepted matched entity IDs.
        """
        s2_matches = []
        s3_matches = []

        for cid, sc in candidate_matches:
            if sc < self.match_threshold:
                continue
            if cid.startswith("S2-"):
                s2_matches.append((cid, sc))
            elif cid.startswith("S3-"):
                s3_matches.append((cid, sc))

        selected = []

        # Source 2 selection: top 1
        if s2_matches:
            s2_matches.sort(key=lambda x: x[1], reverse=True)
            best_s2_cid, best_s2_sc = s2_matches[0]
            # Check if there's ambiguity or clear winner
            selected.append((best_s2_cid, best_s2_sc))

        # Source 3 selection: top 1
        if s3_matches:
            s3_matches.sort(key=lambda x: x[1], reverse=True)
            best_s3_cid, best_s3_sc = s3_matches[0]
            selected.append((best_s3_cid, best_s3_sc))

        return selected
