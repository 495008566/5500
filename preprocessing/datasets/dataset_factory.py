#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from .simple_dataset import SimpleDataset
from .casia_b_dataset import CasiaBDataset

def get_dataset(dataset_name, split='train', **kwargs):
    """
    Factory function to get a dataset by name
    
    Args:
        dataset_name: Name of the dataset
        split: 'train' or 'test' split
        **kwargs: Additional arguments to pass to the dataset constructor
    
    Returns:
        Dataset object
    """
    if dataset_name == 'simple':
        return SimpleDataset(split=split, **kwargs)
    elif dataset_name == 'synthetic_casia_b':
        logging.info(f"Creating synthetic_casia_b dataset...")
        return SimpleDataset(root_dir='data/synthetic_casia_b', split=split, **kwargs)
    elif dataset_name == 'casia_b':
        logging.info(f"Loading Real CASIA-B dataset from data/casia_b...")
        return CasiaBDataset(root_dir='data/casia_b', split=split, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
