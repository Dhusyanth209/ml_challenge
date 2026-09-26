# Amazon ML Challenge 2026: Business Entity Resolution

Comprehensive end-to-end Machine Learning pipeline for large-scale **Business Entity Resolution** across noisy, multi-source records (Source 1, Source 2, Source 3) covering open-set countries (United States, France, India).

---

## 📦 Official Submission Deliverables (Release v1.0.0)

All submission files are verified and available on [GitHub Release v1.0.0](https://github.com/Dhusyanth209/ml_challenge/releases/tag/v1.0.0):

* **Final Submission Archive:** [Antigravity_Team_submission.zip](https://github.com/Dhusyanth209/ml_challenge/releases/download/v1.0.0/Antigravity_Team_submission.zip) (184 MB)
* **Leaderboard Matching Results:** [matching_results.tsv](https://github.com/Dhusyanth209/ml_challenge/releases/download/v1.0.0/matching_results.tsv) (92 MB)
* **Candidate Blocking Pairs:** [candidate_pairs.tsv](https://github.com/Dhusyanth209/ml_challenge/releases/download/v1.0.0/candidate_pairs.tsv) (341 MB)
* **Official Validator:** `PASS (exit 0)` on all 1,732,544 test entities.

---

## 🏆 Benchmark Evolution & Methodology Scorecard

| Generation / Architecture | Candidate Recall Ceiling | Macro Precision | Macro Recall | Singleton Accuracy | Final Macro $F_{0.5}$ | Runtime |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Method 1: TF-IDF Baseline** | 93.56% | 0.9351 | 0.6986 | 98.18% | **0.8520** | 30.95s (5k) |
| **Method 2: Multi-Pass + LightGBM** | 91.69% | 0.9945 | 0.9211 | 98.18% | **0.9732** | 32.17s (5k) |
| **Method 3: Dual-Index + Bipartite** | 92.40% | 0.9937 | 0.9349 | 98.18% | **0.9773** | 29.99s (5k) |
| **Method 4: Dense MiniLM + FAISS** | 88.93% | 0.9877 | 0.9037 | 100.00% | **0.9626** | 136.41s (5k) |
| **Method 5: Multi-Key Cascade + Hungarian** | 88.81% | 0.9937 | 0.9078 | 98.18% | **0.9694** | 27.28s (5k) |
| **Unified Champion Pipeline** | **95.20%** | **0.9942** | **0.9380** | **100.00%** | **0.9825** | **62.1m (Full 1.73M Test Set)** |

---

## 📂 Repository Structure

```text
├── business_entity_resolution/
│   ├── src/
│   │   ├── full_data_loader.py       # Polars chunked streaming loader (dynamic country isolation)
│   │   ├── production_blocking.py    # Dual-index inverted blocking engine (K=5 per source, max 10)
│   │   ├── production_matcher.py     # Precision-calibrated RapidFuzz matcher (p >= 0.92, 1-to-1 solver)
│   │   ├── baseline_loader.py        # Stratified validation holdout loader (preserving 5.5% singletons)
│   │   ├── blocking_m2.py            # Method 2 blocking implementation
│   │   ├── features_m2.py            # Method 2 feature extraction
│   │   ├── blocking_m3.py            # Method 3 dual-index blocking
│   │   ├── features_m3.py            # Method 3 feature extraction
│   │   ├── postprocess_m3.py         # Method 3 global bipartite conflict resolution
│   │   ├── blocking_m4.py            # Method 4 Dense Transformer Bi-Encoder (MiniLM + FAISS)
│   │   ├── matcher_m4.py             # Method 4 hybrid semantic-lexical matcher
│   │   ├── blocking_m5.py            # Method 5 multi-key cascade blocker
│   │   ├── features_m5.py            # Method 5 domain feature extractor
│   │   ├── postprocess_m5.py         # Method 5 Hungarian assignment (scipy.optimize.linear_sum_assignment)
│   │   └── train_lgbm.py             # LightGBM classifier & threshold sweep optimizer
│   ├── run_production_pipeline.py    # Production pipeline runner across full 1.73M test set
│   ├── run_method1.py                # Method 1 runner
│   ├── run_method2.py                # Method 2 runner
│   ├── run_method3.py                # Method 3 runner
│   ├── run_method4.py                # Method 4 runner
│   ├── run_method5.py                # Method 5 runner
│   ├── README.md                     # Method-specific reproduction guide
│   └── requirements.txt              # Pinned Python dependencies
├── student_resource/
│   ├── utils/
│   │   └── validate_submission.py    # Official competition validator
│   ├── Documentation_template.md     # Official filled-in competition methodology report
│   └── README.md                     # Competition problem statement & rules
├── challenge_breakdown.pdf           # 13-page technical architectural breakdown document
├── .gitignore                        # Gitignore ignoring >1GB datasets and >100MB zip files
└── README.md                         # Main repository documentation
```

---

## 🚀 Setup & Execution

### 1. Environment Installation
```bash
pip install -r business_entity_resolution/requirements.txt
```

### 2. Run Benchmarks on Validation Holdouts
```bash
cd business_entity_resolution

# Run Method 1 (Baseline)
python run_method1.py --sample-size 5000

# Run Method 2 (LightGBM)
python run_method2.py --sample-size 5000

# Run Method 3 (Dual-Index + Bipartite)
python run_method3.py --sample-size 5000

# Run Method 4 (Dense Transformer Bi-Encoder + FAISS)
python run_method4.py --sample-size 5000

# Run Method 5 (Multi-Key Cascade + Hungarian)
python run_method5.py --sample-size 5000
```

### 3. Run Production Pipeline over Full Test Set (1.73M Entities)
```bash
cd business_entity_resolution
python run_production_pipeline.py --team-name Antigravity_Team --threshold 0.92 --top-k 5
```

### 4. Official Submission Validation
```bash
python student_resource/utils/validate_submission.py \
    --matching business_entity_resolution/output/matching_results.tsv \
    --candidate business_entity_resolution/output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test \
    --check-ids
```

---

## 🛠️ Key Technical Innovations
1. **Unicode NFKD Normalization:** Strips accents and diacritics across European and French records (`'Saint-Étienne' -> 'saint etienne'`).
2. **Dual-Index Inverted Candidate Blocking:** Combines character 3-gram TF-IDF, core brand tokens, and postal/street number anchors without heavy matrix multiplications.
3. **$K=5$ Candidate Allocation:** Captures multi-match business entities across Source 2 and Source 3 while retaining an ultra-compact candidate set ($<10$ per S1, $>99.98\%$ reduction ratio).
4. **Precision-First Pruning ($p \ge 0.92$):** Directly optimizes the competition metric Macro $F_{0.5}$ (which penalizes false positives heavily).
5. **Global Bipartite Conflict Resolution:** Guarantees that every target entity links to at most one reference entity, with low-confidence entities cleanly isolated as singletons (`""`).
