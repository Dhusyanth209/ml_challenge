import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import logging

logger = logging.getLogger(__name__)

CAND_COLUMNS = ['source1_entity_id', 'candidate_entity_id', 'blocking_score']


class SemanticBlocker:
    def __init__(self, model_name='paraphrase-multilingual-MiniLM-L12-v2', top_k=15):
        self.model = SentenceTransformer(model_name)
        self.top_k = top_k

    def _build_index(self, embs):
        """Build a cosine (inner-product) FAISS index.

        Uses a brute-force flat index for small pools and an IVF index for
        large ones, so search does not degrade into an exhaustive scan over
        millions of vectors (the pools here are ~5M records each).
        """
        dim = embs.shape[1]
        n = embs.shape[0]

        # IVF needs enough training points to pay off; a flat index is already
        # cheap and exact for small pools.
        if n <= 50000:
            index = faiss.IndexFlatIP(dim)
            index.add(embs)
            return index

        # nlist between ~sqrt(n) and 4*sqrt(n); keep enough points per cell.
        nlist = int(max(1, min(4 * np.sqrt(n), n // 39)))
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embs)
        index.add(embs)
        index.nprobe = int(max(1, min(nlist, 24)))
        logger.info(f"  Built IVFFlat index (n={n}, nlist={nlist}, nprobe={index.nprobe})")
        return index

    def fit_and_search(self, s1_df, s2_df, s3_df):
        logger.info("Encoding S2 and S3 for blocking...")
        # Combine S2 and S3 pool
        s23_pool = pd.concat([s2_df, s3_df], ignore_index=True)

        # We only match within the same country
        frames = []

        for country in s1_df['country'].unique():
            logger.info(f"Processing blocking for country: {country}")

            s1_sub = s1_df[s1_df['country'] == country]
            s23_sub = s23_pool[s23_pool['country'] == country]

            if s23_sub.empty or s1_sub.empty:
                continue

            s23_sub_ids = s23_sub['entity_id'].values

            logger.info("  Encoding candidates...")
            s23_embs = self.model.encode(
                s23_sub['blocking_text'].tolist(),
                show_progress_bar=True, convert_to_numpy=True, batch_size=256
            ).astype(np.float32)

            # Normalize for cosine similarity in FAISS (inner product == cosine).
            faiss.normalize_L2(s23_embs)
            index = self._build_index(s23_embs)

            logger.info("  Encoding queries...")
            s1_embs = self.model.encode(
                s1_sub['blocking_text'].tolist(),
                show_progress_bar=True, convert_to_numpy=True, batch_size=256
            ).astype(np.float32)
            faiss.normalize_L2(s1_embs)

            logger.info("  Searching index...")
            actual_k = min(self.top_k, len(s23_sub_ids))
            distances, indices = index.search(s1_embs, actual_k)

            # Vectorized candidate extraction (avoids appending millions of dicts).
            s1_ids = s1_sub['entity_id'].values
            flat_idx = indices.reshape(-1)
            flat_dist = distances.reshape(-1)
            rep_s1 = np.repeat(s1_ids, actual_k)

            # IVF can return -1 when fewer than k neighbours are found; mask them.
            mask = flat_idx != -1
            safe_idx = np.where(mask, flat_idx, 0)
            cand_ids = s23_sub_ids[safe_idx]

            frames.append(pd.DataFrame({
                'source1_entity_id': rep_s1[mask],
                'candidate_entity_id': cand_ids[mask],
                'blocking_score': flat_dist[mask],
            }))

        if not frames:
            return pd.DataFrame(columns=CAND_COLUMNS)

        result = pd.concat(frames, ignore_index=True)
        # Guard against duplicate (S1, candidate) pairs.
        result = result.drop_duplicates(subset=['source1_entity_id', 'candidate_entity_id'])
        return result.reset_index(drop=True)
