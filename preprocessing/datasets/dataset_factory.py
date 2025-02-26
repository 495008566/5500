#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .simple_dataset import SimpleDataset

def get_dataset(name, split='train'):
    """
    Factory function to create a dataset based on the dataset name.
    """
    # For testing purposes, use the simple dataset
    return SimpleDataset(name, split)
