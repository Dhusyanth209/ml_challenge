import pandas as pd
import os
import logging

logger = logging.getLogger(__name__)

def load_data(data_dir, split="train"):
    """
    Load data from the specified directory and split (train or test).
    """
    logger.info(f"Loading {split} data from {data_dir}...")
    
    s1_path = os.path.join(data_dir, split, f"{split}_source1.tsv")
    s2_path = os.path.join(data_dir, split, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, split, f"{split}_source3.tsv")
    
    s1 = pd.read_csv(s1_path, sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(s2_path, sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(s3_path, sep="\t", dtype=str).fillna("")
    
    logger.info(f"Loaded {len(s1)} rows from Source 1")
    logger.info(f"Loaded {len(s2)} rows from Source 2")
    logger.info(f"Loaded {len(s3)} rows from Source 3")
    
    if split == "train":
        gt_path = os.path.join(data_dir, split, f"{split}_ground_truth.tsv")
        gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
        logger.info(f"Loaded {len(gt)} rows from Ground Truth")
        return s1, s2, s3, gt
    
    return s1, s2, s3
