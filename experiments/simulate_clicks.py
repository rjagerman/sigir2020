"""Simulates a click log from supervised ranking data."""
import logging
import pickle
from argparse import ArgumentParser

import numpy as np
import torch
from pytorchltr.click_simulation import simulate_perfect
from pytorchltr.click_simulation import simulate_position
from pytorchltr.click_simulation import simulate_nearrandom
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from pytorchltr.util import rank_by_score

from experiments.dataset import load_ranking_dataset


LOGGER = logging.getLogger(__name__)


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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sessions", type=int, default=1_000_000)
    parser.add_argument("--max_clicks", type=int, default=None)
    return parser


def main(args):
    """Runs the click simulation with given arguments."""
    torch.manual_seed(args.seed)
    LOGGER.info("Loading input data")
    dataset = load_ranking_dataset(args.input_data, normalize=True)
    indices = torch.randint(len(dataset), (args.sessions,))
    dataset = torch.utils.data.Subset(dataset, indices)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1000, collate_fn=create_svmranking_collate_fn(),
        drop_last=False, shuffle=False)

    LOGGER.info("Loading ranker")
    ranker = torch.load(args.ranker)

    simulator = {
        "perfect": lambda rankings, n, ys: simulate_perfect(
            rankings, n, ys, cutoff=args.cutoff),
        "position": lambda rankings, n, ys: simulate_position(
            rankings, n, ys, cutoff=args.cutoff, eta=args.eta),
        "nearrandom": lambda rankings, n, ys: simulate_nearrandom(
            rankings, n, ys, cutoff=args.cutoff, eta=args.eta)
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
        clicks, obs_probs = simulator(rankings, ys, n)
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
        "args": args,
        "clicked_docs": np.array(clicked_docs, dtype=np.int32),
        "propensities": np.array(propensities, dtype=np.float32),
        "qids": np.array(qids, dtype=np.int32)
    }

    LOGGER.info("Writing click log to file")
    with open(args.output_log, "wb") as f:
        pickle.dump(click_log, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s, %(levelname)s, %(module)s] %(message)s",
        level=logging.INFO)
    main(get_parser().parse_args())
