"""
Global Bipartite Conflict Resolution Module (Method 3).
Enforces the 1-to-1 matching constraint across the deduplicated Source 1 reference catalog:
Each target candidate (S2 or S3 entity) can be assigned to at most ONE Source 1 entity.
Solves conflicts using confidence-weighted bipartite assignment, cleanly isolating singletons.
"""

import os


def resolve_global_bipartite_matching(pairs: list, probs: list, threshold: float, all_s1_ids: list) -> dict:
    """
    Solves bipartite matching conflicts globally across all pairs >= threshold.
    Guarantees no target candidate is allocated to multiple S1 records.
    Returns: {s1_id: [matched_id, ...]}
    """
    final_matches = {sid: [] for sid in all_s1_ids}

    # Filter pairs that meet or exceed threshold
    qualified_pairs = []
    for (sid, cid), prob in zip(pairs, probs):
        if prob >= threshold:
            qualified_pairs.append((float(prob), sid, cid))

    # Sort descending by model confidence
    qualified_pairs.sort(key=lambda x: x[0], reverse=True)

    claimed_targets = set()

    for prob, sid, cid in qualified_pairs:
        # Bipartite constraint: target candidate can only link to one master S1 entity
        if cid not in claimed_targets:
            claimed_targets.add(cid)
            final_matches[sid].append(cid)

    return final_matches


def save_bipartite_matches(final_matches: dict, output_file: str, all_s1_ids: list):
    """
    Saves results to matching_results.tsv adhering strictly to competition specifications.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    singletons = 0
    matched_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            matches = final_matches.get(sid, [])
            clean_matches = []
            seen = set()
            for m in matches:
                if m not in seen:
                    seen.add(m)
                    clean_matches.append(m)

            m_str = ",".join(clean_matches)
            if not m_str:
                singletons += 1
            else:
                matched_count += 1
            f.write(f"{sid}\t{m_str}\n")

    print(f"Exported global bipartite matches to: {output_file}")
    print(f"  Total S1 rows:      {len(all_s1_ids):,}")
    print(f"  Matched rows:       {matched_count:,}")
    print(f"  Singleton rows:     {singletons:,} ({singletons/len(all_s1_ids):.2%})")
