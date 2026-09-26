import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

FEATURES = [
    'blocking_score', 'block_rank', 'block_gap', 'n_cands', 'is_s2',
    'name_ratio', 'name_tsort', 'name_tset', 'name_partial', 'name_jw', 'compact_ratio',
    'addr_ratio', 'addr_tsort', 'addr_tset', 'addr_partial',
    'num_tset', 'num_both', 'name1_empty', 'name2_empty', 'len_ratio',
]


def _pairwise(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


NUM_RE = r'\d+'


def _nums(addrs):
    return pd.Series(addrs).str.findall(NUM_RE).map(lambda xs: ' '.join(sorted(set(xs)))).values


def compute_features(q, p, q_idx, p_idx, scores):
    """
    q, p: preprocessed query (S1) / pool frames for one country.
    q_idx, p_idx, scores: candidate pairs from the blocker (row positions).
    Returns a float32 feature frame (one row per pair).
    """
    # object ndarrays: iterating pandas string arrays element-wise is ~10x slower
    qn, pn = q['name'].to_numpy(object)[q_idx], p['name'].to_numpy(object)[p_idx]
    qa, pa = q['addr'].to_numpy(object)[q_idx], p['addr'].to_numpy(object)[p_idx]
    qc = pd.Series(qn).str.replace(' ', '', regex=False).values
    pc = pd.Series(pn).str.replace(' ', '', regex=False).values
    qnum, pnum = _nums(qa), _nums(pa)

    f = pd.DataFrame({'blocking_score': scores.astype(np.float32)})

    # candidate-list context (pairs are grouped by q_idx, sorted by score desc)
    grp = pd.Series(scores).groupby(q_idx)
    f['block_rank'] = grp.cumcount().values.astype(np.float32)
    f['block_gap'] = (grp.transform('max').values - scores).astype(np.float32)
    f['n_cands'] = grp.transform('size').values.astype(np.float32)
    f['is_s2'] = (p['eid'].values[p_idx] // 10 ** 12 == 2).astype(np.float32)

    f['name_ratio'] = _pairwise(qn, pn, fuzz.ratio)
    f['name_tsort'] = _pairwise(qn, pn, fuzz.token_sort_ratio)
    f['name_tset'] = _pairwise(qn, pn, fuzz.token_set_ratio)
    f['name_partial'] = _pairwise(qn, pn, fuzz.partial_ratio)
    f['name_jw'] = _pairwise(qn, pn, JaroWinkler.normalized_similarity)
    f['compact_ratio'] = _pairwise(qc, pc, fuzz.ratio)
    f['addr_ratio'] = _pairwise(qa, pa, fuzz.ratio)
    f['addr_tsort'] = _pairwise(qa, pa, fuzz.token_sort_ratio)
    f['addr_tset'] = _pairwise(qa, pa, fuzz.token_set_ratio)
    f['addr_partial'] = _pairwise(qa, pa, fuzz.partial_ratio)
    f['num_tset'] = _pairwise(qnum, pnum, fuzz.token_set_ratio)

    q_has_num = pd.Series(qnum).str.len().values > 0
    p_has_num = pd.Series(pnum).str.len().values > 0
    f['num_both'] = (q_has_num & p_has_num).astype(np.float32)
    ql = pd.Series(qn).str.len().values
    pl = pd.Series(pn).str.len().values
    f['name1_empty'] = (ql == 0).astype(np.float32)
    f['name2_empty'] = (pl == 0).astype(np.float32)
    f['len_ratio'] = (np.minimum(ql, pl) / np.maximum(1, np.maximum(ql, pl))).astype(np.float32)
    return f
