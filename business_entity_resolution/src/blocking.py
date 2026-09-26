import logging
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)

N_FEATURES = 2 ** 22
VEC_CHUNK = 200_000
VEC = HashingVectorizer(analyzer='word', token_pattern=r'\S+', ngram_range=(1, 2),
                        n_features=N_FEATURES, alternate_sign=False, norm=None,
                        binary=True, dtype=np.float32)


def blocking_text(df):
    # compact name as its own token lets "millenniumallied.com" hit "Millennium Allied"
    return (df['name'] + ' ' + df['name'].str.replace(' ', '', regex=False) + ' ' + df['addr']).to_numpy(object)


def _doc_freq(texts):
    return np.bincount(VEC.transform(texts).indices, minlength=N_FEATURES).astype(np.int32)


def _weighted(args):
    texts, idf = args
    X = VEC.transform(texts)
    X.data *= idf[X.indices]
    X.eliminate_zeros()
    X = sp.csr_matrix((X.data.copy(), X.indices.copy(), X.indptr.copy()), shape=X.shape)  # compact
    return normalize(X, copy=False)


def _chunks(texts):
    return [texts[i:i + VEC_CHUNK] for i in range(0, len(texts), VEC_CHUNK)]


class SparseBlocker:
    """
    Hashed word (1,2)-gram TF-IDF + sparse top-k cosine search, per country.

    Memory stays bounded because:
      * HashingVectorizer keeps no vocabulary dict,
      * tokens that are too common (df > max_df) or unique (df < 2) are dropped
        before the matrix is assembled, so the similarity product stays sparse,
      * queries are processed in small chunks and never densified.
    Hashing runs in `workers` processes; search runs in `workers` threads
    (scipy releases the GIL in sparse matmul).
    """

    def __init__(self, top_k=15, max_df=3000, min_score=0.1, query_chunk=1000, workers=4):
        self.top_k = top_k
        self.max_df = max_df
        self.min_score = min_score
        self.query_chunk = query_chunk
        self.workers = workers

    def fit(self, pool_texts, query_texts):
        with Pool(self.workers) as pool:
            df = np.zeros(N_FEATURES, np.int64)
            for d in pool.imap_unordered(_doc_freq, _chunks(pool_texts) + _chunks(query_texts)):
                df += d
            n = len(pool_texts) + len(query_texts)
            idf = (np.log((1 + n) / (1 + df)) + 1).astype(np.float32)
            idf[(df > self.max_df) | (df < 2)] = 0
            del df
            self.query = sp.vstack(pool.map(_weighted, [(c, idf) for c in _chunks(query_texts)])).tocsr()
            P = sp.vstack(pool.map(_weighted, [(c, idf) for c in _chunks(pool_texts)])).tocsr()
        # pool stored transposed (features x docs) so query @ pool_T is a csr @ csr product
        self.pool_T = P.T.tocsr()
        del P
        logger.info(f"    pool nnz={self.pool_T.nnz:,}  query nnz={self.query.nnz:,}")

    def _search_chunk(self, q_start, q_end):
        S = (self.query[q_start:q_end] @ self.pool_T).tocsr()
        if S.nnz == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
        rows = np.repeat(np.arange(S.shape[0]), np.diff(S.indptr))
        order = np.lexsort((-S.data, rows))
        rank = np.arange(len(order)) - S.indptr[rows[order]]
        keep = order[(rank < self.top_k) & (S.data[order] >= self.min_score)]
        return rows[keep] + q_start, S.indices[keep].astype(np.int64), S.data[keep]

    def search(self, q_start, q_end):
        """Returns (query_row_idx, pool_idx, score) for queries [q_start, q_end),
        grouped by query row and sorted by score desc within each row."""
        starts = range(q_start, q_end, self.query_chunk)
        with ThreadPoolExecutor(self.workers) as ex:
            parts = list(ex.map(lambda s: self._search_chunk(s, min(s + self.query_chunk, q_end)), starts))
        return tuple(np.concatenate(x) for x in zip(*parts))
