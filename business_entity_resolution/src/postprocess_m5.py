"""
Global Bipartite Linear Assignment & Submission Generator Module (Method 5).
Applies scipy.optimize.linear_sum_assignment to resolve multi-source candidate conflicts
so that every S2/S3 entity is assigned to AT MOST ONE Source 1 entity globally.
Preserves singletons with empty string "".
"""

import os
from collections import defaultdict
import numpy as np
from scipy.optimize import linear_sum_assignment


def resolve_hungarian_bipartite_matching(pairs: list, probs: np.ndarray,
                                         threshold: float, all_s1_ids: list) -> dict:
    """
    Enforces the global 1-to-1 target constraint:
    Each target entity (S2/S3) is matched to AT MOST ONE Source 1 entity.
    Contested candidates (claimed by >=2 S1 entities with prob >= threshold)
    are resolved via scipy.optimize.linear_sum_assignment to maximize total score.
    """
    # Filter by threshold
    valid_pairs = []
    s1_to_cands = defaultdict(dict)
    cand_to_s1s = defaultdict(dict)

    for (sid, cid), prob in zip(pairs, probs):
        if prob >= threshold:
            valid_pairs.append((sid, cid, float(prob)))
            s1_to_cands[sid][cid] = float(prob)
            cand_to_s1s[cid][sid] = float(prob)

    final_matches = {sid: [] for sid in all_s1_ids}

    # Separate uncontested vs contested target candidates
    uncontested_cands = set()
    contested_cands = set()

    for cid, s1_dict in cand_to_s1s.items():
        if len(s1_dict) == 1:
            uncontested_cands.add(cid)
        else:
            contested_cands.add(cid)

    # 1. Directly assign uncontested candidates
    for cid in uncontested_cands:
        sid = list(cand_to_s1s[cid].keys())[0]
        final_matches[sid].append(cid)

    # 2. For contested candidates, solve via Hungarian Linear Sum Assignment
    if contested_cands:
        # Find all S1 entities involved in contested candidates
        contested_s1s = set()
        for cid in contested_cands:
            for sid in cand_to_s1s[cid]:
                contested_s1s.add(sid)

        contested_s1_list = sorted(list(contested_s1s))
        contested_cand_list = sorted(list(contested_cands))

        s1_idx_map = {sid: i for i, sid in enumerate(contested_s1_list)}
        cand_idx_map = {cid: j for j, cid in enumerate(contested_cand_list)}

        n_rows = len(contested_s1_list)
        n_cols = len(contested_cand_list)

        # Cost matrix: minimize negative probability (maximize probability)
        # Unconnected edges get large cost
        BIG_COST = 100.0
        cost_matrix = np.full((n_rows, n_cols), BIG_COST, dtype=np.float64)

        for cid in contested_cand_list:
            j = cand_idx_map[cid]
            for sid, p in cand_to_s1s[cid].items():
                i = s1_idx_map[sid]
                cost_matrix[i, j] = -p

        # Run Hungarian / Linear Sum Assignment
        # If n_rows < n_cols or n_rows > n_cols, rectangular assignment matches min(n_rows, n_cols)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_cands = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 0:  # Valid matched edge
                sid = contested_s1_list[r]
                cid = contested_cand_list[c]
                final_matches[sid].append(cid)
                assigned_cands.add(cid)

        # If any contested candidate wasn't assigned due to rectangular matrix constraints (n_rows < n_cols),
        # greedy assign remaining candidates to their highest-scoring available S1
        remaining_cands = contested_cands - assigned_cands
        if remaining_cands:
            rem_pairs = []
            for cid in remaining_cands:
                for sid, p in cand_to_s1s[cid].items():
                    rem_pairs.append((sid, cid, p))
            rem_pairs.sort(key=lambda x: x[2], reverse=True)
            for sid, cid, p in rem_pairs:
                if cid not in assigned_cands:
                    final_matches[sid].append(cid)
                    assigned_cands.add(cid)

    # Sort candidates for deterministic output
    for sid in all_s1_ids:
        final_matches[sid] = sorted(list(set(final_matches[sid])))

    return final_matches


def save_hungarian_matches(final_matches: dict, output_file: str, all_s1_ids: list):
    """
    Saves matching results to matching_results.tsv.
    Singletons strictly emit an empty string in matched_entity_ids.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        matched_count = 0
        singleton_count = 0
        for sid in all_s1_ids:
            matches = final_matches.get(sid, [])
            if matches:
                f.write(f"{sid}\t{','.join(matches)}\n")
                matched_count += 1
            else:
                f.write(f"{sid}\t\n")
                singleton_count += 1

    print(f"\nExported Hungarian bipartite matches to: {output_file}")
    print(f"  Total S1 rows:      {len(all_s1_ids):,}")
    print(f"  Matched rows:       {matched_count:,}")
    print(f"  Singleton rows:     {singleton_count:,} ({singleton_count/len(all_s1_ids):.2%})")
