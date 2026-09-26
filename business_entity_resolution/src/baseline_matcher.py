"""
Baseline Matcher Module (Method 1).
Implements Heuristic Rule Matching using Character 3-Gram Cosine Similarity:
  - Name Character 3-Gram Cosine Similarity
  - Address Character 3-Gram Cosine Similarity
  - Composite Score = 0.65 * (Name Sim) + 0.35 * (Address Sim)
Performs threshold grid search (0.70 to 0.95 in 0.02 increments) to maximize Macro F_0.5,
and exports matching_results.tsv.
"""

import os
import numpy as np


def compute_char_3gram_cosine(str1: str, str2: str) -> float:
    """
    Computes exact Character 3-Gram Cosine Similarity between two strings.
    """
    if not str1 or not str2:
        return 0.0
    s1 = str1.lower().strip()
    s2 = str2.lower().strip()
    if s1 == s2:
        return 1.0
    if len(s1) < 3 or len(s2) < 3:
        return 1.0 if s1 == s2 else 0.0

    # Count 3-grams
    counts1 = {}
    for i in range(len(s1) - 2):
        g = s1[i:i + 3]
        counts1[g] = counts1.get(g, 0) + 1

    counts2 = {}
    for i in range(len(s2) - 2):
        g = s2[i:i + 3]
        counts2[g] = counts2.get(g, 0) + 1

    dot = 0
    for g, c1 in counts1.items():
        if g in counts2:
            dot += c1 * counts2[g]

    norm1 = sum(c * c for c in counts1.values()) ** 0.5
    norm2 = sum(c * c for c in counts2.values()) ** 0.5

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot / (norm1 * norm2))


def calculate_macro_f05_score(pred_dict: dict, gt_dict: dict, eval_s1_ids: set) -> tuple:
    """
    Calculates Macro F_0.5, Macro Precision, Macro Recall, and Singleton Accuracy
    across all S1 entities in eval_s1_ids.
    """
    scores = []
    precisions = []
    recalls = []
    singleton_scores = []

    for sid in eval_s1_ids:
        gt_matches = gt_dict.get(sid, set())
        preds = set(pred_dict.get(sid, []))

        # Singleton in ground truth
        if len(gt_matches) == 0:
            if len(preds) == 0:
                s_score = 1.0
                prec = 1.0
                rec = 1.0
            else:
                s_score = 0.0
                prec = 0.0
                rec = 1.0
            singleton_scores.append(s_score)
            scores.append(s_score)
            precisions.append(prec)
            recalls.append(rec)
            continue

        # Non-singleton entity
        if len(preds) == 0:
            scores.append(0.0)
            precisions.append(0.0)
            recalls.append(0.0)
            continue

        tp = len(gt_matches & preds)
        prec = tp / len(preds)
        rec = tp / len(gt_matches)

        precisions.append(prec)
        recalls.append(rec)

        if tp == 0 or (prec + rec) == 0:
            scores.append(0.0)
        else:
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            scores.append(f05)

    macro_f05 = float(np.mean(scores)) if scores else 0.0
    macro_prec = float(np.mean(precisions)) if precisions else 0.0
    macro_rec = float(np.mean(recalls)) if recalls else 0.0
    singleton_acc = float(np.mean(singleton_scores)) if singleton_scores else 1.0

    return macro_f05, macro_prec, macro_rec, singleton_acc


def score_candidate_pairs(candidates_dict: dict, s1_lookup: dict, target_lookup: dict, s1_ids_subset: set = None):
    """
    Computes Name Cosine, Address Cosine, and Composite Score for all candidate pairs.
    Returns: scored_pairs = {s1_id: [(cand_id, composite_score), ...]}
    """
    keys = s1_ids_subset if s1_ids_subset is not None else candidates_dict.keys()
    scored_dict = {}

    for sid in keys:
        s1_rec = s1_lookup.get(sid)
        if not s1_rec:
            scored_dict[sid] = []
            continue

        s1_name = s1_rec.get("name", "")
        s1_addr = s1_rec.get("addr", "")

        cands = candidates_dict.get(sid, [])
        cand_scores = []

        for item in cands:
            cid = item[0] if isinstance(item, (tuple, list)) else item
            t_rec = target_lookup.get(cid)
            if not t_rec:
                continue

            t_name = t_rec.get("name", "")
            t_addr = t_rec.get("addr", "")

            name_sim = compute_char_3gram_cosine(s1_name, t_name)
            addr_sim = compute_char_3gram_cosine(s1_addr, t_addr)

            # Heuristic rule: Composite Score = 0.65 * Name + 0.35 * Address
            composite = 0.65 * name_sim + 0.35 * addr_sim
            cand_scores.append((cid, composite))

        scored_dict[sid] = cand_scores

    return scored_dict


def sweep_optimal_threshold(scored_val_dict: dict, gt_dict: dict, val_s1_ids: set,
                            threshold_range=None):
    """
    Grid searches thresholds from 0.70 to 0.95 in 0.02 increments to maximize Macro F_0.5.
    """
    if threshold_range is None:
        thresholds = [round(t, 2) for t in np.arange(0.70, 0.96, 0.02)]
        if 0.95 not in thresholds:
            thresholds.append(0.95)
    else:
        thresholds = threshold_range

    print("\n" + "=" * 65)
    print("THRESHOLD GRID SEARCH (Macro F_0.5 Optimization):")
    print(f"{'Threshold':<12} | {'Macro F_0.5':<14} | {'Precision':<12} | {'Recall':<12} | {'Singleton Acc'}")
    print("-" * 65)

    best_thresh = 0.85
    best_f05 = -1.0
    best_stats = (0.0, 0.0, 0.0, 0.0)

    for thresh in thresholds:
        # Build predictions at this threshold
        pred_dict = {}
        for sid in val_s1_ids:
            cands = scored_val_dict.get(sid, [])
            passing = [cid for cid, score in cands if score >= thresh]
            pred_dict[sid] = passing

        f05, prec, rec, s_acc = calculate_macro_f05_score(pred_dict, gt_dict, val_s1_ids)

        marker = " *" if f05 > best_f05 else ""
        print(f"{thresh:<12.2f} | {f05:<14.4f} | {prec:<12.4f} | {rec:<12.4f} | {s_acc:<13.2%}{marker}")

        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh
            best_stats = (f05, prec, rec, s_acc)

    print("=" * 65)
    print(f"Optimal Threshold:    {best_thresh:.2f}")
    print(f"Peak Macro F_0.5:     {best_stats[0]:.4f}")
    print(f"Precision at Peak:    {best_stats[1]:.4f}")
    print(f"Recall at Peak:       {best_stats[2]:.4f}")
    print(f"Singleton Accuracy:   {best_stats[3]:.2%}")
    print("=" * 65)

    return best_thresh, best_stats


def apply_matching_threshold(scored_dict: dict, threshold: float, all_s1_ids: list):
    """
    Applies the calibrated threshold. Emits empty list for singletons and below-threshold entities.
    Returns: {s1_id: [matched_id, ...]}
    """
    matched_dict = {}
    for sid in all_s1_ids:
        cands = scored_dict.get(sid, [])
        passing = [cid for cid, score in cands if score >= threshold]
        matched_dict[sid] = passing
    return matched_dict


def save_matching_results(matched_dict: dict, output_file: str, all_s1_ids: list):
    """
    Saves final matches into output/matching_results.tsv:
    source1_entity_id\tmatched_entity_ids
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    singletons = 0
    matched_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            matches = matched_dict.get(sid, [])
            clean_matches = []
            seen = set()
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    clean_matches.append(m)

            match_str = ",".join(clean_matches)
            if not match_str:
                singletons += 1
            else:
                matched_count += 1
            f.write(f"{sid}\t{match_str}\n")

    print(f"Saved matching results to: {output_file}")
    print(f"  Total S1 rows:      {len(all_s1_ids):,}")
    print(f"  Matched rows:       {matched_count:,}")
    print(f"  Singleton rows:     {singletons:,} ({singletons/len(all_s1_ids):.2%})")
