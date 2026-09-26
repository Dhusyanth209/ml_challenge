import argparse
import sys
from pathlib import Path

def validate(matching_path, candidate_path, test_dir):
    print("Checking files...")
    test_dir = Path(test_dir)
    s1_file = test_dir / "test_source1.tsv"
    
    # 1. Read S1 test IDs
    with open(s1_file, "r", encoding="utf-8") as f:
        s1_lines = [line.strip().split("\t")[0] for line in f if line.strip()]
    expected_s1 = set(s1_lines[1:]) # skip header
    total_expected = len(expected_s1)
    
    # 2. Check candidate_pairs.tsv
    cand_dict = {}
    with open(candidate_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        header = lines[0].strip().split("\t")
        assert header == ["source1_entity_id", "candidate_entity_ids"], "Invalid header in candidate_pairs.tsv"
        for line in lines[1:]:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            cands = set(parts[1].split(",")) if len(parts) > 1 and parts[1] else set()
            cand_dict[s1_id] = cands

    assert len(cand_dict) == total_expected, f"Row count mismatch in candidates: got {len(cand_dict)}, expected {total_expected}"
    
    # 3. Check matching_results.tsv
    match_count = 0
    with open(matching_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        header = lines[0].strip().split("\t")
        assert header == ["source1_entity_id", "matched_entity_ids"], "Invalid header in matching_results.tsv"
        for line in lines[1:]:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            matches = parts[1].split(",") if len(parts) > 1 and parts[1] else []
            match_set = set(matches)
            assert len(matches) == len(match_set), f"Duplicate IDs in match for {s1_id}"
            
            # Must be subset of candidates
            assert match_set.issubset(cand_dict.get(s1_id, set())), f"Matched ID not in candidate_pairs for {s1_id}"
            match_count += 1
            
    assert match_count == total_expected, f"Row count mismatch in matching: got {match_count}, expected {total_expected}"
    print("PASS (exit 0): All validation constraints met perfectly!")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matching", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--test-dir", required=True)
    args = parser.parse_args()
    sys.exit(validate(args.matching, args.candidate, args.test_dir))
