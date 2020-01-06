import numpy as np
import torch
from pytorchltr.dataset.svmrank import svmranking_dataset as _load
from joblib.memory import Memory as _Memory

from experiments.click_log import clicklog_dataset
from experiments.click_log import create_clicklog_collate_fn


load_ranking_dataset = _Memory(".cache", compress=6).cache(_load)


def load_click_dataset(data_file, click_log_file, ips_strategy, ips_clip=None,
                       n_clicks=None, batch_size=50, max_list_size=None,
                       normalize=True, filter_queries=False):
    """Loads a click dataset from file.

    Arguments:
        data_file: The data file to load.
        click_log_file: The corresponding click log to load.
        ips_strategy: A string indicating the IPS strategy to use.
        ips_clip: (Optional) whether to apply IPS clipping.
        n_clicks: (Optional) number of clicks to include (None includes all).
        batch_size: (Optional) the batch size to use (default: 50).
        max_list_size: (Optional) a cut-off for list size padding, default:
            None.
        normalize: (Optional) whether to normalize input data, default: True
        filter_queries: (Optional) whether to filter queries that have no
            relevant documents from the training data (default: False).

    Returns:
        A tuple containing a `torch.utils.data.DataLoader` for loading data and
        an int indicating the dimensionality of input data.
    """
    # Load training data.
    data = load_ranking_dataset(
        data_file, normalize=True, filter_queries=filter_queries)

    # Load click log from file.
    click_log = clicklog_dataset(data, click_log_file, clip=ips_clip,
                                 n_clicks=n_clicks)

    # Construct sampling strategy.
    if ips_strategy == "sample":
        propensities = torch.FloatTensor(click_log.propensities)
        weights = (1.0 / propensities)
        probabilities = weights / torch.sum(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            probabilities, len(click_log))
    else:
        sampler = torch.utils.data.sampler.RandomSampler(click_log)

    # Return a data loader.
    return (torch.utils.data.DataLoader(
        click_log, batch_size=batch_size, sampler=sampler,
        collate_fn=create_clicklog_collate_fn(max_list_size=max_list_size)),
        data[0]["features"].shape[1])
