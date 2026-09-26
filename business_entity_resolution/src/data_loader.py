import os
import logging
from multiprocessing import Pool
import numpy as np
import pandas as pd
from preprocessing import preprocess_dataframe

logger = logging.getLogger(__name__)

READ_CHUNK = 250_000
ID_BASE = 10 ** 12   # entity ids are stored as int64: source * ID_BASE + number


def encode_ids(ids: pd.Series) -> np.ndarray:
    src = ids.str.slice(1, 2).astype(np.int64)
    num = ids.str.slice(3).astype(np.int64)
    enc = src.values * ID_BASE + num.values
    assert (ids.values == decode_ids(enc)).all(), "entity_id not round-trippable"
    return enc


def decode_ids(enc: np.ndarray) -> np.ndarray:
    enc = np.asarray(enc)
    return ('S' + pd.Series(enc // ID_BASE).astype(str) + '-' + pd.Series(enc % ID_BASE).astype(str)).values


def _prep_chunk(chunk):
    out = preprocess_dataframe(chunk)
    return pd.DataFrame({'eid': encode_ids(chunk['entity_id']), 'name': out['name'].values,
                         'addr': out['addr'].values})


def _load_source(path, country, workers):
    reader = pd.read_csv(path, sep="\t", dtype=str, chunksize=READ_CHUNK, keep_default_na=False,
                         usecols=['entity_id', 'business_name', 'business_address', 'country'],
                         quoting=3)
    parts, pending = [], []
    with Pool(workers) as pool:
        # bounded: at most `workers` raw chunks in flight, filtered to the country first
        for chunk in reader:
            pending.append(chunk[chunk['country'] == country])
            if len(pending) == workers:
                parts += pool.map(_prep_chunk, pending)
                pending = []
        if pending:
            parts += pool.map(_prep_chunk, pending)
    return pd.concat(parts, ignore_index=True)


def list_countries(data_dir, split):
    path = os.path.join(data_dir, split, f"{split}_source1.tsv")
    return pd.read_csv(path, sep="\t", usecols=['country'], dtype=str, keep_default_na=False,
                       quoting=3)['country'].value_counts().index.tolist()


def load_country(data_dir, split, country, workers=4):
    """Returns (s1, pool) for one country; pool = source2 + source3. Columns: eid, name, addr."""
    logger.info(f"Loading + preprocessing {split}/{country}...")
    s1 = _load_source(os.path.join(data_dir, split, f"{split}_source1.tsv"), country, workers)
    pool = pd.concat([_load_source(os.path.join(data_dir, split, f"{split}_source{i}.tsv"), country, workers)
                      for i in (2, 3)], ignore_index=True)
    return s1, pool


def load_ground_truth(data_dir, s1_eids):
    """Returns {s1_eid: np.array of matched eids} for the requested S1 entities only."""
    gt = pd.read_csv(os.path.join(data_dir, "train", "train_ground_truth.tsv"),
                     sep="\t", dtype=str, keep_default_na=False, quoting=3)
    gt['eid'] = encode_ids(gt['source1_entity_id'])
    gt = gt[gt['eid'].isin(s1_eids)]
    return {e: (encode_ids(pd.Series(m.split(','))) if m else np.empty(0, np.int64))
            for e, m in zip(gt['eid'], gt['matched_entity_ids'])}
