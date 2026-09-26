"""
Postprocessing & Global Bipartite Conflict Resolution Module.
Applies the calibrated threshold, resolves multi-assignment conflicts across S1 entities
via greedy maximum-weight bipartite matching, cleanly isolates singletons,
and exports matching_results.tsv.
"""

import os
import pandas as pd


def resolve_bipartite_conflicts(pairs: list, probs: list, threshold: float, all_s1_ids: list) -> dict:
    """
    Greedy maximum-confidence bipartite conflict resolution:
    - Filters candidate pairs with probability >= threshold.
    - Sorts all passing pairs by confidence probability in descending order.
    - Ensures each target entity (S2/S3 record) is linked to at most ONE S1 entity.
    - Returns: {s1_id: [matched_id, ...]} for all s1_ids in all_s1_ids.
    """
    # Initialize all S1 records as empty (singletons by default)
    final_matches = {sid: [] for sid in all_s1_ids}

    # Filter by calibrated threshold
    filtered_pairs = []
    for (s1_id, cand_id), prob in zip(pairs, probs):
        if prob >= threshold:
            filtered_pairs.append((prob, s1_id, cand_id))

    # Sort descending by model confidence
    filtered_pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_candidates = set()

    for prob, s1_id, cand_id in filtered_pairs:
        # Bipartite constraint: target candidate can only link to one reference S1 entity
        if cand_id not in assigned_candidates:
            assigned_candidates.add(cand_id)
            final_matches[s1_id].append(cand_id)

    return final_matches


def export_matching_results(matching_dict: dict, output_file: str, all_s1_ids: list):
    """
    Exports final matching results in the exact schema:
    source1_entity_id\tmatched_entity_ids
    Ensuring tab delimiter, no duplicates, and exact ordering.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

    singletons_count = 0
    matched_count = 0

    with open(output_file, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            matches = matching_dict.get(sid, [])
            # Deduplicate preserving order
            seen = set()
            clean_matches = []
            for mid in matches:
                if mid not in seen:
                    seen.add(mid)
                    clean_matches.append(mid)

            match_str = ",".join(clean_matches)
            if not match_str:
                singletons_count += 1
            else:
                matched_count += 1

            f.write(f"{sid}\t{match_str}\n")

    print(f"\nFinal Matching Results Exported:")
    print(f"  Target File:        {output_file}")
    print(f"  Total S1 Entities:  {len(all_s1_ids):,}")
    print(f"  Entities w/ Matches:{matched_count:,} ({matched_count/len(all_s1_ids):.2%})")
    print(f"  Singletons (Empty): {singletons_count:,} ({singletons_count/len(all_s1_ids):.2%})")


def validate_results_format(filepath: str, expected_s1_ids: set):
    """
    Validates formatting locally before submission:
    - Tab delimiter verification
    - Exact header
    - Exactly one row per expected S1 ID
    - No self-matches or invalid prefixes
    """
    errors = []
    seen = set()
    intra_dupes = 0
    self_matches = 0

    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            errors.append(f"Invalid header: {header}. Expected ['source1_entity_id', 'matched_entity_ids']")

        for line_num, line in enumerate(f, start=2):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 2:
                errors.append(f"Line {line_num}: malformed row (must have exactly 1 tab).")
                continue

            sid, match_str = parts[0], parts[1]
            if sid in seen:
                errors.append(f"Line {line_num}: duplicate source1_entity_id: {sid}")
            seen.add(sid)

            if match_str:
                ids = match_str.split(",")
                if len(ids) != len(set(ids)):
                    intra_dupes += 1
                for mid in ids:
                    if mid.startswith("S1-"):
                        self_matches += 1
                    elif not (mid.startswith("S2-") or mid.startswith("S3-")):
                        errors.append(f"Line {line_num}: invalid prefix in matched ID: {mid}")

    missing = expected_s1_ids - seen
    if missing:
        errors.append(f"Missing {len(missing)} S1 entities in submission.")
    extra = seen - expected_s1_ids
    if extra:
        errors.append(f"Found {len(extra)} unexpected S1 IDs.")
    if intra_dupes > 0:
        errors.append(f"Found {intra_dupes} rows with duplicate IDs within matched_entity_ids.")
    if self_matches > 0:
        errors.append(f"Found {self_matches} self-matches (S1 IDs in matched_entity_ids).")

    if errors:
        print("\nLOCAL VALIDATION FAILED:")
        for e in errors[:5]:
            print(f"  - {e}")
        return False

    print("\nLOCAL VALIDATION PASSED: Output file strictly adheres to competition schema! [OK]")
    return True
