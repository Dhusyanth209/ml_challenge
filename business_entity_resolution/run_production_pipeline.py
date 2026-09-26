"""
End-to-End Production Pipeline for Amazon ML Challenge 2026 (Precision-Optimized).
Key Upgrades:
  - Unicode NFKD normalization for French accents and diacritics.
  - Strict Macro F_0.5 precision threshold: p >= 0.92.
  - Strict source constraint: At most 1 best match from S2, at most 1 best match from S3.
  - Full candidate pairs retained (K=5 per source, max 10 per S1).
  - Validates with official validator using --check-ids.
  - Packages final submission zip.
"""

import os
import sys
import gc
import time
import zipfile
import subprocess
import argparse
from collections import defaultdict

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.full_data_loader import load_partitioned_test_data, get_all_test_countries
from src.production_blocking import ProductionBlocker
from src.production_matcher import PrecisionProductionScorer


def find_default_dataset_dir():
    candidates = [
        r"D:\ML challenge\student_resource\dataset",
        os.path.join("..", "student_resource", "dataset"),
        os.path.join("student_resource", "dataset"),
        "dataset"
    ]
    for p in candidates:
        if os.path.exists(p) and os.path.exists(os.path.join(p, "test")):
            return os.path.abspath(p)
    return r"D:\ML challenge\student_resource\dataset"


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026: Precision Production Pipeline")
    parser.add_argument("--dataset-dir", type=str, default=find_default_dataset_dir(),
                        help="Path to dataset directory containing test/ and train/")
    parser.add_argument("--output-dir", type=str, default="output",
                        help="Directory to save candidate_pairs.tsv and matching_results.tsv")
    parser.add_argument("--team-name", type=str, default="Antigravity_Team",
                        help="Team name for packaging submission zip")
    parser.add_argument("--threshold", type=float, default=0.92,
                        help="Calibrated match threshold strictly optimizing Macro F_0.5 (default: 0.92)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Candidate allocation per source (default: 5, total max 10)")
    args = parser.parse_args()

    pipeline_start = time.time()
    test_dir = os.path.join(args.dataset_dir, "test")
    os.makedirs(args.output_dir, exist_ok=True)

    cand_file = os.path.join(args.output_dir, "candidate_pairs.tsv")
    match_file = os.path.join(args.output_dir, "matching_results.tsv")

    print("=" * 85)
    print("  AMAZON ML CHALLENGE 2026: PRECISION PRODUCTION PIPELINE (TARGET F_0.5 >= 0.99)")
    print("  Accent Normalization + Strict Threshold (p >= 0.92) + Max 1 Match/Source Constraint")
    print("=" * 85)
    print(f"Dataset Path:       {args.dataset_dir}")
    print(f"Test Directory:     {test_dir}")
    print(f"Output Directory:   {args.output_dir}")
    print(f"Candidate Cap (K):  {args.top_k} per source (Max {args.top_k * 2} per S1 entity)")
    print(f"Decision Cutoff:    {args.threshold:.2f} (Strict Precision Gate)")
    print(f"Team Name:          {args.team_name}")
    print("=" * 85)

    countries = get_all_test_countries(test_dir)
    print(f"\nDetected {len(countries)} country partitions in test set: {countries}")

    blocker = ProductionBlocker(bucket_cap=40)
    scorer = PrecisionProductionScorer(match_threshold=args.threshold)

    total_s1_processed = 0
    total_candidates_written = 0
    total_matches_written = 0
    total_singletons = 0
    match_cardinality_counts = defaultdict(int)

    # Open output streams
    with open(cand_file, "w", encoding="utf-8") as f_cand, \
         open(match_file, "w", encoding="utf-8") as f_match:

        # Write official competition headers
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_match.write("source1_entity_id\tmatched_entity_ids\n")

        for country in countries:
            c_start = time.time()
            print(f"\n{'='*35} COUNTRY: {country} {'='*35}")

            # 1. Load Country Partition via Polars
            s1_df, s2_df, s3_df = load_partitioned_test_data(test_dir, country)
            n_s1 = s1_df.height
            total_s1_processed += n_s1

            s1_ids = s1_df["entity_id"].to_list()
            s1_names = s1_df["business_name"].to_list()
            s1_addrs = s1_df["business_address"].to_list()

            s2_ids = s2_df["entity_id"].to_list()
            s2_names = s2_df["business_name"].to_list()
            s2_addrs = s2_df["business_address"].to_list()

            s3_ids = s3_df["entity_id"].to_list()
            s3_names = s3_df["business_name"].to_list()
            s3_addrs = s3_df["business_address"].to_list()

            # Free Polars DataFrames from memory
            del s1_df, s2_df, s3_df
            gc.collect()

            # 2. Build Inverted Dual-Indexes for Source 2 and Source 3
            print(f"  [Index] Building Dual-Index for Source 2 ({len(s2_ids):,} entities)...")
            t_idx0 = time.time()
            s2_index = blocker.build_index(s2_ids, s2_names, s2_addrs)
            print(f"  [Index] S2 indexed in {time.time() - t_idx0:.2f}s ({len(s2_index):,} unique keys)")

            print(f"  [Index] Building Dual-Index for Source 3 ({len(s3_ids):,} entities)...")
            t_idx1 = time.time()
            s3_index = blocker.build_index(s3_ids, s3_names, s3_addrs)
            print(f"  [Index] S3 indexed in {time.time() - t_idx1:.2f}s ({len(s3_index):,} unique keys)")

            # 3. Stream Candidates & Compute Match Scores
            print(f"  [Query & Match] Processing {n_s1:,} Source 1 entities for '{country}'...")
            t_query0 = time.time()

            country_matches = defaultdict(list)
            # Track candidate claims for 1-to-1 global constraint: cand_id -> (s1_id, score)
            cand_claimed_by = {}

            # Process in streaming chunks of 25,000
            CHUNK_SIZE = 25000
            for start_idx in range(0, n_s1, CHUNK_SIZE):
                end_idx = min(start_idx + CHUNK_SIZE, n_s1)

                for i in range(start_idx, end_idx):
                    s_id = s1_ids[i]
                    s_name = s1_names[i]
                    s_addr = s1_addrs[i]

                    # Retrieve Top 5 from S2
                    s2_cands = blocker.retrieve_top_k(
                        s_name, s_addr, s2_index, s2_ids, s2_names, s2_addrs, top_k=args.top_k
                    )

                    # Retrieve Top 5 from S3
                    s3_cands = blocker.retrieve_top_k(
                        s_name, s_addr, s3_index, s3_ids, s3_names, s3_addrs, top_k=args.top_k
                    )

                    # Total candidate list
                    all_cands = s2_cands + s3_cands
                    cand_ids_list = [c[0] for c in all_cands]
                    total_candidates_written += len(cand_ids_list)

                    # Stream directly to candidate_pairs.tsv
                    f_cand.write(f"{s_id}\t{','.join(cand_ids_list)}\n")

                    # Score candidates and apply precision gating
                    candidate_matches = []
                    for cid, base_sc, c_idx in all_cands:
                        if cid.startswith("S2-"):
                            t_n, t_a = s2_names[c_idx], s2_addrs[c_idx]
                        else:
                            t_n, t_a = s3_names[c_idx], s3_addrs[c_idx]

                        match_sc = scorer.compute_match_score(s_name, s_addr, t_n, t_a, base_sc)
                        if match_sc >= args.threshold:
                            candidate_matches.append((cid, match_sc))

                    # Strict source constraint: At most 1 best match from S2, at most 1 from S3
                    selected = scorer.prune_and_select_matches(candidate_matches)

                    for cid, match_sc in selected:
                        # 1-to-1 global conflict check: highest confidence claimant wins
                        if cid in cand_claimed_by:
                            prev_sid, prev_sc = cand_claimed_by[cid]
                            if match_sc > prev_sc:
                                cand_claimed_by[cid] = (s_id, match_sc)
                                if cid in country_matches[prev_sid]:
                                    country_matches[prev_sid].remove(cid)
                                country_matches[s_id].append(cid)
                        else:
                            cand_claimed_by[cid] = (s_id, match_sc)
                            country_matches[s_id].append(cid)

                progress_pct = (end_idx / n_s1) * 100.0
                rate = end_idx / (time.time() - t_query0)
                print(f"    Processed {end_idx:,}/{n_s1:,} ({progress_pct:.1f}%) @ {rate:,.0f} queries/sec")

            # 4. Stream Final Matches for this Country
            for sid in s1_ids:
                matches = country_matches.get(sid, [])
                match_count = len(matches)
                match_cardinality_counts[match_count] += 1

                if matches:
                    f_match.write(f"{sid}\t{','.join(matches)}\n")
                    total_matches_written += 1
                else:
                    f_match.write(f"{sid}\t\n")
                    total_singletons += 1

            c_elapsed = time.time() - c_start
            print(f"  Partition '{country}' completed in {c_elapsed:.2f}s ({c_elapsed/60:.2f} min)")

            # Free all country data and run garbage collection
            del s1_ids, s1_names, s1_addrs
            del s2_ids, s2_names, s2_addrs, s2_index
            del s3_ids, s3_names, s3_addrs, s3_index
            del country_matches, cand_claimed_by
            gc.collect()

    f_cand.close()
    f_match.close()

    total_elapsed = time.time() - pipeline_start
    print("\n" + "=" * 85)
    print("      PRECISION PRODUCTION PIPELINE EXECUTION SUMMARY")
    print("=" * 85)
    print(f"Total Source 1 Entities Processed:  {total_s1_processed:,}")
    print(f"Total Candidate Pairs Generated:     {total_candidates_written:,}")
    print(f"Average Candidates per S1 Entity:    {total_candidates_written / total_s1_processed:.2f}")
    print(f"Total Matched Entities:              {total_matches_written:,} ({total_matches_written / total_s1_processed:.2%})")
    print(f"Total Singletons Isolated:           {total_singletons:,} ({total_singletons / total_s1_processed:.2%})")
    print("\nMatch Count Distribution per S1 Entity:")
    for k in sorted(match_cardinality_counts.keys()):
        cnt = match_cardinality_counts[k]
        print(f"  {k} matches: {cnt:,} ({cnt / total_s1_processed:.2%})")
    print(f"\nTotal Pipeline Runtime:              {total_elapsed:.2f}s ({total_elapsed / 60:.2f} min)")
    print("=" * 85)

    # -------------------------------------------------------------------------
    # STEP 5: RUN OFFICIAL SUBMISSION VALIDATOR WITH --check-ids
    # -------------------------------------------------------------------------
    print("\n>>> STEP 5: Running Official Competition Submission Validator (with --check-ids)...")
    validator_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "student_resource", "utils", "validate_submission.py")
    if not os.path.exists(validator_script):
        validator_script = os.path.join("student_resource", "utils", "validate_submission.py")

    cmd = [
        sys.executable,
        validator_script,
        "--matching", match_file,
        "--candidate", cand_file,
        "--test-dir", test_dir,
        "--check-ids"
    ]
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print("[ERROR] Submission validation failed! Please check the error log above.")
        return False
    else:
        print("[SUCCESS] Submission validation PASSED (Exit code 0)!")

    # -------------------------------------------------------------------------
    # STEP 6: ASSEMBLE SUBMISSION ZIP
    # -------------------------------------------------------------------------
    print("\n>>> STEP 6: Packaging Final Competition Submission ZIP...")
    zip_name = f"{args.team_name}_submission.zip"
    zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), zip_name)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    doc_template = os.path.join(base_dir, "..", "student_resource", "Documentation_template.md")

    if os.path.exists(zip_path):
        os.remove(zip_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        zf.write(match_file, arcname="output/matching_results.tsv")
        zf.write(cand_file, arcname="output/candidate_pairs.tsv")

        # 2. Source code files
        src_dir = os.path.join(base_dir, "src")
        for root, _, files in os.walk(src_dir):
            for file in files:
                if file.endswith(".py"):
                    full_p = os.path.join(root, file)
                    rel_p = os.path.relpath(full_p, base_dir)
                    zf.write(full_p, arcname=f"code/business_entity_resolution/{rel_p}")

        # 3. Top-level code files
        for extra_f in ["run_production_pipeline.py", "run_method3.py", "README.md", "requirements.txt"]:
            f_p = os.path.join(base_dir, extra_f)
            if os.path.exists(f_p):
                zf.write(f_p, arcname=f"code/business_entity_resolution/{extra_f}")

        # 4. Documentation
        if os.path.exists(doc_template):
            zf.write(doc_template, arcname="Documentation_template.md")

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"\nSubmission ZIP Package Created Successfully: {zip_path}")
    print(f"Zip File Size: {zip_size_mb:.2f} MB")
    print("=" * 85)
    return True


if __name__ == "__main__":
    main()
