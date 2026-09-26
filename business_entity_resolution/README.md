# Business Entity Resolution Pipeline — ML Challenge 2026

An end-to-end, high-precision Entity Resolution and Record Linkage pipeline designed for large-scale multi-source business data matching.

## Architecture

1. **Preprocessor & Stratified Hold-out Splitter (`src/data_loader.py`)**:
   - Explicit tab-separated loading with memory-efficient parsing.
   - Stratified 80/20 train/validation split strictly preserving the country distribution and singleton ratio.

2. **Country-Partitioned Character 3-Gram Sparse Indexing (`src/blocking.py`)**:
   - Dynamically partitions records by country (supporting US, India, France, etc. without hardcoded constraints).
   - Generates character 3-gram TF-IDF representations.
   - Retrieves top 3 nearest candidate pairs from Source 2 and Source 3 via batched sparse matrix multiplication.
   - Outputs `output/candidate_pairs.tsv` achieving >99.9% reduction ratio and high recall ceiling.

3. **Pairwise Feature Engineering (`src/features.py`)**:
   - RapidFuzz string metrics: Token Sort Ratio, Partial Ratio, WRatio, Levenshtein Ratio, Token Set Ratio.
   - Address token overlaps, word Jaccards, exact token match count.
   - Numerical / PIN code / ZIP overlap extraction.
   - Composite harmonic means and TF-IDF blocking scores.

4. **LightGBM Binary Ranker & Threshold Optimizer (`src/train_eval.py`)**:
   - LightGBM binary classifier trained on candidate pairs.
   - Custom threshold sweep (0.70 to 0.95 in 0.01 steps) explicitly optimizing the official macro $F_{0.5}$ metric:
     $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
   - Handles singletons (empty ground truth) scoring 1.0 if empty and 0.0 if any match is predicted.

5. **Global Bipartite Conflict Resolution (`src/postprocess.py`)**:
   - Greedy confidence-weighted bipartite assignment to ensure no target entity is incorrectly linked to multiple Source 1 entities.
   - Clean singleton isolation.
   - Generates `output/matching_results.tsv` strictly adhering to the competition schema.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Execution

### 1. Run Complete Pipeline (Hold-out Validation + Test Inference):
```bash
python run_pipeline.py --mode full
```

### 2. Run Only on Validation Split:
```bash
python run_pipeline.py --mode val
```

### 3. Rapid Prototype / Smoke Test (e.g. 5,000 records):
```bash
python run_pipeline.py --sample-size 5000
```

---

## Validation

Verify that generated submission files pass all checks using the official validator:
```bash
python ../student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir ../student_resource/dataset/test
```
