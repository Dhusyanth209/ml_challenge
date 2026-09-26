# Entity Resolution ML Challenge - Advanced Strategic Plan (v1.1)

## Executive Summary: The Winning Architecture

To win this challenge, we must optimize heavily for the precision-biased $F_{0.5}$ metric while avoiding overfitting to the US and India, ensuring our model generalizes perfectly to the unseen test country (France). 

Given the massive dataset size (~12.5M records total), Deep Learning approaches (like cross-encoders) for all-pairs comparison are computationally impossible. **The optimal architecture is a highly robust Multi-Strategy Blocking pipeline feeding into a LightGBM/XGBoost classifier that operates purely on language-agnostic distance features.**

Below is the elaborated, champion pipeline, selected from the best possible methods.

---

## 1. Preprocessing & Normalization: The Hybrid Approach

*Relying purely on dictionaries overfits to specific countries. Using local LLMs is too slow for 12.5M rows. The best method is a lightweight hybrid.*

### Champion Method: 
**A. Universal Text Cleaning (Fast & Scalable)**
1. **Garbage Collection:** Detect and strip non-ASCII and garbled unicode (crucial for Indian transliteration noise).
2. **Standardization:** Lowercase, remove punctuation (except `&`), and remove URL domains (e.g., strip `| www.shivshakti.com`).
3. **Legal Entity Stripping:** Extract and remove legal suffixes (`ltd, limited, pvt, private, inc, llc`) from the core `business_name` to prevent false similarities based purely on entity types. Store these suffixes as a separate categorical feature.

**B. Lightweight Structural Parsing**
Instead of rigid Regex, use statistical parsers like `libpostal` (which is trained globally and will natively understand French addresses) to decompose `business_address` into:
- `house_number`
- `road`
- `city`
- `postcode`

If `libpostal` is too difficult to set up, fallback to a heuristic Regex parser that looks for numbers (postcodes/house numbers) and extracts them as independent features.

---

## 2. Candidate Generation (Blocking): High-Recall Dual-Pass

*Blocking determines the maximum possible recall. A single strategy will fail. We need a dual-pass approach that captures both exact lexical overlaps and fuzzy semantic overlaps.*

### Champion Method: Dual-Pass FAISS + TF-IDF
To reduce trillions of pairs to a few million highly probable candidates, we union the results of two fast blocking strategies:

**Pass 1: Semantic Blocking via Multilingual Embeddings (Generalization King)**
- **Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Fits the <8B MIT/Apache rule, exceptionally fast, supports English, Hindi, and French).
- **Execution:** 
  1. Concatenate `cleaned_name + " " + cleaned_address`.
  2. Embed all 12.5M records.
  3. Index S2 and S3 vectors in a **FAISS IVFFlat or HNSW** index.
  4. Query each S1 vector to retrieve the Top-K (e.g., Top 15) closest matches.
- **Why it's best:** It effortlessly handles word transpositions, French translations, and severe typos without relying on exact string matches.

**Pass 2: TF-IDF Token Blocking (Precision Lexical Match)**
- **Execution:** Create a TF-IDF matrix for `business_name`. For each S1 entity, find S2/S3 entities that share highly weighted (rare) tokens.
- **Why it's best:** Embeddings sometimes cluster generic names together. TF-IDF ensures that if two businesses share a very rare, specific word (e.g., "Ectolumdrex"), they are paired.

**Candidate Pool:** Union the results of Pass 1 and Pass 2. Ensure we filter out any candidate where `country` does not match exactly (as per challenge rules, though France is unseen, matches only happen within the same country).

---

## 3. Feature Engineering: The Distinguishing Signals

*The ML model cannot read text; it only understands the numerical differences between the S1 entity and the S2/S3 candidate. We must engineered language-agnostic features to ensure France is handled correctly.*

### Champion Feature Set:
For every (S1, Candidate) pair, compute the following features:

**A. Lexical Name Distances**
1. **Jaro-Winkler Distance:** Excellent for names because it heavily weights prefix matches (e.g., `Consulting Nyasa` vs `Consulting`).
2. **Token Sort Ratio (FuzzyWuzzy):** Handles word-order transpositions (`Consulting Nyasa` vs `Nyasa Consulting`).
3. **TF-IDF Weighted Intersection:** The sum of TF-IDF weights of words present in both names. (Sharing the word "Pharmacy" is a weak signal; sharing "Nyasa" is a strong signal).

**B. Address & Location Features**
1. **Address Levenshtein Ratio:** Normalized edit distance between the full addresses.
2. **Numerical Conflict Flag:** Extract all numbers from both addresses. If they share a number (e.g., street number `1795`), `match=1`. If both have numbers but they differ (e.g., `1795` vs `204`), `conflict=-1`. If missing, `neutral=0`. This is highly predictive!
3. **City/Postcode Jaro-Winkler:** Compare the parsed city/postcode strings.

**C. Cross-Features & Semantic Signals**
1. **Embedding Cosine Similarity:** The exact dot-product score from the FAISS index.
2. **Name Substring Flag:** Is the S1 name completely contained within the S2 name (common in DBA/Trade names)?
3. **Length Ratios:** `len(name_1) / len(name_2)`.

---

## 4. The ML Matching Model: Optimizing for $F_{0.5}$

*We need an interpretable, fast model that handles missing values natively and can be regularized against country bias.*

### Champion Model: LightGBM / XGBoost
- **Why:** Tree-based models dominate tabular distance data. They naturally handle missing feature values (e.g., when a parsed postcode is missing) without imputation hacks.
- **Generalization Strategy (Crucial):** 
  - Do **not** include `country` as a feature.
  - Train using `GroupKFold` cross-validation where the grouping variable is `country`. This forces the LightGBM model to perform well on a country it hasn't trained on in that fold, perfectly mimicking the hidden "France" test set.

---

## 5. Threshold Tuning & Post-Processing

*The $F_{0.5}$ metric weights precision 2x over recall. We must heavily penalize False Positives.*

### Champion Tuning Strategy:
1. **Probability Calibration:** Output the raw probabilities from LightGBM for every candidate pair.
2. **Grid Search Threshold:** On our out-of-fold validation set, calculate the macro-averaged $F_{0.5}$ for classification thresholds ranging from `0.50` to `0.95` (in steps of 0.05).
3. **High Threshold Selection:** The optimal threshold will likely be very high (e.g., `0.75` or `0.85`).
4. **Singleton Handling:** If an S1 entity's highest-scoring candidate is below this strict threshold, we return an empty list `[]`. 
   - *Why this is genius:* Correctly identifying a singleton yields a full 1.0 score for that entity. By setting a high threshold, we avoid false merges (which destroy the $F_{0.5}$ score) and confidently capitalize on the 5.6% of entities that are true singletons.

---

## Next Steps for Implementation
1. Develop the **TF-IDF + FAISS Blocking Module** and measure the *Recall Ceiling* (Target: > 97% of true matches found).
2. Build the **Feature Engineering Pipeline** to generate the lexical and semantic distance vectors.
3. Train the **LightGBM Classifier** using `GroupKFold` on the generated pairs.
4. Optimize the classification threshold specifically against the $F_{0.5}$ metric.
