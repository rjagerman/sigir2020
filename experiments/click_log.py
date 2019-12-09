"""Utility for loading click logs."""
import logging
import pickle

import numpy as np
import torch
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn


class ClicklogDataset(torch.utils.data.Dataset):
    """A click log dataset."""
    def __init__(self, ranking_dataset, click_log, clip=None):
        self._ranking_dataset = ranking_dataset
        self._clicked_docs = click_log["clicked_docs"]
        self._qids = click_log["qids"]
        self._propensities = click_log["propensities"]
        if clip is not None:
            self._propensities = np.clip(
                self._propensities, a_min=clip, a_max=None)
        self._clip = clip

    @property
    def propensities(self):
        return self._propensities

    def __len__(self):
        return self._clicked_docs.shape[0]

    def __getitem__(self, index):
        qid = self._qids[index]
        rd_index = self._ranking_dataset.get_index(qid)
        sample = self._ranking_dataset[rd_index]
        sample["clicks"] = torch.zeros_like(sample["relevance"])
        sample["clicks"][self._clicked_docs[index]] = 1
        sample["propensity"] = self._propensities[index]
        return sample


def create_clicklog_collate_fn(rng=np.random.RandomState(42),
                               max_list_size=None):
    """Creates a collate_fn for click log datasets."""
    svmrank_collate_fn = create_svmranking_collate_fn(
        rng, max_list_size)
    def _collate_fn(batch):
        out = svmrank_collate_fn(batch)
        out["clicks"] = torch.zeros_like(out["relevance"])
        for i, sample in enumerate(batch):
            out["clicks"][i, 0:sample["clicks"].shape[0]] = sample["clicks"]
        out["propensity"] = torch.FloatTensor(
            [sample["propensity"] for sample in batch])
        return out
    return _collate_fn


def clicklog_dataset(ranking_dataset, click_log_file_path, clip=None):
    """Loads a click log dataset from given file path.

    Arguments:
        ranking_dataset: The svmranking dataset that was used to generate
            clicks.
        click_log_file_path: Path to the generated click log file.
        clip: Value to clip propensities at (if None, apply no clipping).

    Returns:
        A ClicklogDataset used for ranking experiments.
    """
    with open(click_log_file_path, "rb") as f:
        click_log = pickle.load(f)
    return ClicklogDataset(ranking_dataset, click_log, clip)
