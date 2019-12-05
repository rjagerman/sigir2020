"""Simulates a click log from supervised ranking data."""
import logging
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
from pytorchltr.dataset.svmrank import svmranking_dataset
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from pytorchltr.util import mask_padded_values
from pytorchltr.util import rank_by_score


LOGGER = logging.getLogger(__name__)


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
        sample["relevance"] = torch.zeros_like(sample["relevance"])
        sample["relevance"][self._clicked_docs[index]] = 1
        sample["propensity"] = self._propensities[index]
        return sample


def create_clicklog_collate_fn(rng=np.random.RandomState(42),
                               max_list_size=None):
    """Creates a collate_fn for click log datasets."""
    svmrank_collate_fn = create_svmranking_collate_fn(
        rng, max_list_size)
    def _collate_fn(batch):
        out = svmrank_collate_fn(batch)
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


def simulate_perfect(rankings, n, ys, cutoff=None):
    """Simulates perfect clicks.

    Arguments:
        rankings: A tensor of size (batch_size, list_size) containing indices
            of the ranked list
        n: A tensor of size (batch_size) containing the per-query list size.
        ys: A tensor of size (batch_size, list_size) indicating the relevance
            labels of each document.
        cutoff: (optional) An integer indicating the cutoff for the simulation.

    Returns:
        A tuple of two tensors of size (batch_size, list_size), where the first
        indicates the clicks with 0.0 and 1.0 and the second indicates the
        propensity of observing each document.
    """
    if cutoff is not None:
        n = torch.min(torch.ones_like(n) * cutoff, n)
    obs_prob = torch.ones_like(rankings, dtype=torch.float)
    obs_prob = mask_padded_values(obs_prob, n, mask_value=0.0, mutate=True)

    ranked_ys = torch.gather(ys, 1, rankings)
    click_prob = torch.zeros_like(ranked_ys, dtype=torch.float)
    click_prob[ranked_ys == 0] = 0.0
    click_prob[ranked_ys == 1] = 0.2
    click_prob[ranked_ys == 2] = 0.4
    click_prob[ranked_ys == 3] = 0.8
    click_prob[ranked_ys == 4] = 1.0

    clicks = torch.bernoulli(click_prob * obs_prob)
    invert_ranking = torch.argsort(rankings, dim=1)
    return (
        torch.gather(clicks, 1, invert_ranking).to(dtype=torch.long),
        torch.gather(obs_prob, 1, invert_ranking)
    )


def simulate_position(rankings, n, ys, cutoff=None, eta=1.0, pos_prob=1.0,
                      neg_prob=0.1):
    """Simulates position-biased clicks.

    Arguments:
        rankings: A tensor of size (batch_size, list_size) containing indices
            of the ranked list
        n: A tensor of size (batch_size) containing the per-query list size.
        ys: A tensor of size (batch_size, list_size) indicating the relevance
            labels of each document.
        cutoff: (optional) An integer indicating the cutoff for the simulation.
        eta: A float >= 0.0, indicating the severity of click bias.
        pos_prob: A float in [0.0, 1.0] indicating the probability of clicking
            a relevant document.
        neg_prob: A float in [0.0, 1.0] indicating the probability of clicking
            a non-relevant document.

    Returns:
        A tuple of two tensors of size (batch_size, list_size), where the first
        indicates the clicks with 0.0 and 1.0 and the second indicates the
        propensity of observing each document.
    """
    if cutoff is not None:
        n = torch.min(torch.ones_like(n) * cutoff, n)
    obs_prob = 1.0 / (1.0 + torch.arange(rankings.shape[1])) ** eta
    obs_prob = torch.repeat_interleave(obs_prob[None, :], rankings.shape[0], dim=0)
    obs_prob = mask_padded_values(obs_prob, n, mask_value=0.0, mutate=True)

    ranked_ys = torch.gather(ys, 1, rankings)
    click_prob = torch.zeros_like(ranked_ys, dtype=torch.float)
    click_prob[ranked_ys == 0] = neg_prob
    click_prob[ranked_ys == 1] = neg_prob
    click_prob[ranked_ys == 2] = neg_prob
    click_prob[ranked_ys == 3] = pos_prob
    click_prob[ranked_ys == 4] = pos_prob

    clicks = torch.bernoulli(click_prob * obs_prob)
    invert_ranking = torch.argsort(rankings, dim=1)
    return (
        torch.gather(clicks, 1, invert_ranking).to(dtype=torch.long),
        torch.gather(obs_prob, 1, invert_ranking)
    )


def simulate_nearrandom(rankings, n, ys, cutoff=None, eta=1.0):
    """Simulates near-random position-biased clicks.

    Arguments:
        rankings: A tensor of size (batch_size, list_size) containing indices
            of the ranked list
        n: A tensor of size (batch_size) containing the per-query list size.
        ys: A tensor of size (batch_size, list_size) indicating the relevance
            labels of each document.
        cutoff: (optional) An integer indicating the cutoff for the simulation.
        eta: A float >= 0.0, indicating the severity of click bias.

    Returns:
        A tuple of two tensors of size (batch_size, list_size), where the first
        indicates the clicks with 0.0 and 1.0 and the second indicates the
        propensity of observing each document.
    """
    if cutoff is not None:
        n = torch.min(torch.ones_like(n) * cutoff, n)
    obs_prob = 1.0 / (1.0 + torch.arange(rankings.shape[1])) ** eta
    obs_prob = torch.repeat_interleave(obs_prob[None, :], rankings.shape[0], dim=0)
    obs_prob = mask_padded_values(obs_prob, n, mask_value=0.0, mutate=True)

    ranked_ys = torch.gather(ys, 1, rankings)
    click_prob = torch.zeros_like(ranked_ys, dtype=torch.float)
    click_prob[ranked_ys == 0] = 0.40
    click_prob[ranked_ys == 1] = 0.45
    click_prob[ranked_ys == 2] = 0.50
    click_prob[ranked_ys == 3] = 0.55
    click_prob[ranked_ys == 4] = 0.60

    clicks = torch.bernoulli(click_prob * obs_prob)
    invert_ranking = torch.argsort(rankings, dim=1)
    return (
        torch.gather(clicks, 1, invert_ranking).to(dtype=torch.long),
        torch.gather(obs_prob, 1, invert_ranking)
    )


def get_parser():
    """Gets the parser to create arguments for `main`."""
    parser = ArgumentParser()
    parser.add_argument("--input_data", type=str, required=True)
    parser.add_argument("--ranker", type=str, required=True)
    parser.add_argument("--output_log", type=str, required=True)
    parser.add_argument("--vali_data", type=str, default=None)
    parser.add_argument('--behavior', default="perfect",
                        choices=["perfect", "position", "nearrandom"])
    parser.add_argument("--cutoff", type=int, default=None)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--pos_prob", type=float, default=1.0)
    parser.add_argument("--neg_prob", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sessions", type=int, default=1_000_000)
    parser.add_argument("--max_clicks", type=int, default=None)
    return parser


def main(args):
    """Runs the click simulation with given arguments."""
    torch.manual_seed(args.seed)
    LOGGER.info("Loading input data")
    dataset = svmranking_dataset(args.input_data, normalize=True)
    indices = torch.randint(len(dataset), (args.sessions,))
    dataset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, collate_fn=create_svmranking_collate_fn(),
        drop_last=False, shuffle=False)

    LOGGER.info("Loading ranker")
    ranker = torch.load(args.ranker)

    simulator = {
        "perfect": lambda rankings, n, ys: simulate_perfect(
            rankings, n, ys, args.cutoff),
        "position": lambda rankings, n, ys: simulate_position(
            rankings, n, ys, args.cutoff, args.eta, args.pos_prob,
            args.neg_prob),
        "nearrandom": lambda rankings, n, ys: simulate_nearrandom(
            rankings, n, ys, args.cutoff, args.eta)
    }[args.behavior]

    LOGGER.info("Generating rankings and clicks (%d sessions)", args.sessions)
    count = 0
    clicked_docs = []
    propensities = []
    qids = []
    for batch in loader:
        xs, ys, qid, n = (
            batch["features"], batch["relevance"], batch["qid"], batch["n"])
        scores = ranker(xs)
        rankings = rank_by_score(scores, n)
        clicks, obs_probs = simulator(rankings, n, ys)
        count += xs.shape[0]
        for row, col in zip(*torch.where(clicks == 1)):
            if args.max_clicks is None or len(clicked_docs) < args.max_clicks:
                clicked_docs.append(int(col))
                propensities.append(float(obs_probs[row, col]))
                qids.append(int(qid[row]))
        LOGGER.info(
            "%d clicks, %d sessions (+%d), current list_size=%d",
            len(clicked_docs), count, xs.shape[0], xs.shape[1])

        if args.max_clicks is not None and len(clicked_docs) >= args.max_clicks:
            LOGGER.info("reached %d maximum clicks, stopping simulation.",
                        args.max_clicks)
            break

    LOGGER.info("Compressing click log to numpy arrays")
    click_log = {
        "svmrank_dataset": args.input_data,
        "clicked_docs": np.array(clicked_docs, dtype=np.int32),
        "propensities": np.array(propensities, dtype=np.float32),
        "qids": np.array(qids, dtype=np.int32)
    }

    LOGGER.info("Writing click log to file")
    with open(args.output_log, "wb") as f:
        pickle.dump(click_log, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s, %(levelname)s, %(module)s, %(threadName)s] %(message)s",
        level=logging.INFO)
    main(get_parser().parse_args())
