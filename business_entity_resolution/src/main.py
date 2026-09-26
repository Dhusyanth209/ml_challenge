import os
import gc
import time
import pickle
import argparse
import logging
import numpy as np
import pandas as pd
from data_loader import list_countries, load_country, load_ground_truth, decode_ids
from blocking import SparseBlocker, blocking_text
from feature_engineering import compute_features
from model import MatchingModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-dir', default=os.path.join(BASE_DIR, 'dataset'))
    ap.add_argument('--output-dir', default=os.path.join(BASE_DIR, 'output'))
    ap.add_argument('--model-path', default=os.path.join(BASE_DIR, 'v2', 'model.pkl'))
    ap.add_argument('--train-frac', type=float, default=0.07,
                    help='fraction of S1 train entities used for training (full set is not needed)')
    ap.add_argument('--top-k', type=int, default=15)
    ap.add_argument('--max-df', type=int, default=3000)
    ap.add_argument('--query-chunk', type=int, default=1000, help='S1 rows per sparse matmul')
    ap.add_argument('--batch', type=int, default=50_000, help='S1 rows per feature/predict batch')
    ap.add_argument('--workers', type=int, default=4, help='processes/threads for preprocessing, hashing and search')
    ap.add_argument('--test-limit', type=int, default=0, help='debug: only first N test S1 rows per country')
    ap.add_argument('--retrain', action='store_true', help='ignore saved model')
    return ap.parse_args()


def iter_candidates(q, p, args):
    """Per-country blocking; yields (b_start, b_end, q_idx, p_idx, score) per batch of S1 rows."""
    blocker = SparseBlocker(top_k=args.top_k, max_df=args.max_df,
                            query_chunk=args.query_chunk, workers=args.workers)
    blocker.fit(blocking_text(p), blocking_text(q))
    for b_start in range(0, len(q), args.batch):
        b_end = min(b_start + args.batch, len(q))
        yield b_start, b_end, *blocker.search(b_start, b_end)


def build_training_set(args):
    feats, labels, s1_ids, gt_counts = [], [], [], []
    for country in list_countries(args.dataset_dir, 'train'):
        q, p = load_country(args.dataset_dir, 'train', country, args.workers)
        q = q.sample(frac=args.train_frac, random_state=42).reset_index(drop=True)
        gt = load_ground_truth(args.dataset_dir, q['eid'].values)
        gt_counts.append(pd.Series([len(gt.get(e, ())) for e in q['eid']], index=q['eid'].values))
        gt_pairs = {(a, m) for a, ms in gt.items() for m in ms}
        del gt
        logger.info(f"[train] {country}: {len(q):,} S1 vs {len(p):,} pool")

        for _, _, qi, pi, sc in iter_candidates(q, p, args):
            a_ids, c_ids = q['eid'].values[qi], p['eid'].values[pi]
            feats.append(compute_features(q, p, qi, pi, sc))
            labels.append(np.fromiter(((a, c) in gt_pairs for a, c in zip(a_ids, c_ids)),
                                      dtype=bool, count=len(qi)))
            s1_ids.append(a_ids)
        del q, p, gt_pairs
        gc.collect()
    return (pd.concat(feats, ignore_index=True), np.concatenate(labels),
            np.concatenate(s1_ids), pd.concat(gt_counts))


def train(args):
    t0 = time.time()
    feats, labels, s1_ids, gt_counts = build_training_set(args)
    logger.info(f"Training pairs built in {time.time() - t0:.0f}s")

    # hold out 20% of S1 entities to pick the threshold on the exact metric
    rng = np.random.default_rng(0)
    val_ids = rng.choice(gt_counts.index.values, size=len(gt_counts) // 5, replace=False)
    is_val = np.isin(s1_ids, val_ids)

    model = MatchingModel()
    model.fit(feats[~is_val], labels[~is_val])
    probs = model.predict(feats[is_val])
    model.tune_threshold(s1_ids[is_val], labels[is_val], probs, gt_counts[gt_counts.index.isin(val_ids)])

    with open(args.model_path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {args.model_path}")
    return model


def _join_groups(qi, eids):
    s = pd.Series(decode_ids(eids)).groupby(qi).agg(','.join)
    return dict(zip(s.index, s.values))


def predict_test(args, model):
    os.makedirs(args.output_dir, exist_ok=True)
    cand_path = os.path.join(args.output_dir, 'candidate_pairs.tsv')
    match_path = os.path.join(args.output_dir, 'matching_results.tsv')

    n_matched = 0
    with open(cand_path, 'w') as fc, open(match_path, 'w') as fm:
        fc.write('source1_entity_id\tcandidate_entity_ids\n')
        fm.write('source1_entity_id\tmatched_entity_ids\n')
        for country in list_countries(args.dataset_dir, 'test'):
            q, p = load_country(args.dataset_dir, 'test', country, args.workers)
            if args.test_limit:
                q = q.head(args.test_limit)
            logger.info(f"[test] {country}: {len(q):,} S1 vs {len(p):,} pool")
            q_ids = decode_ids(q['eid'].values)
            if p.empty:
                for a in q_ids:
                    fc.write(f'{a}\t\n')
                    fm.write(f'{a}\t\n')
                continue
            for b_start, b_end, qi, pi, sc in iter_candidates(q, p, args):
                f = compute_features(q, p, qi, pi, sc)
                keep = model.predict(f) > model.threshold
                cands = _join_groups(qi, p['eid'].values[pi])
                matches = _join_groups(qi[keep], p['eid'].values[pi][keep])
                n_matched += keep.sum()
                fc.writelines(f'{q_ids[i]}\t{cands.get(i, "")}\n' for i in range(b_start, b_end))
                fm.writelines(f'{q_ids[i]}\t{matches.get(i, "")}\n' for i in range(b_start, b_end))
                logger.info(f"    {b_end:,}/{len(q):,} S1 done")
            del q, p
            gc.collect()
    logger.info(f"Wrote {cand_path} and {match_path} ({n_matched:,} matched pairs)")


def main():
    args = parse_args()
    if os.path.exists(args.model_path) and not args.retrain:
        logger.info(f"Loading saved model from {args.model_path} (use --retrain to rebuild)")
        with open(args.model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        model = train(args)
    gc.collect()

    predict_test(args, model)
    logger.info("Pipeline Complete!")


if __name__ == "__main__":
    main()
