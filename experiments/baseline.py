"""Trains a baseline ranker on a fraction of supervised training data."""
import logging
from argparse import ArgumentParser

import torch
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from pytorchltr.loss.pairwise import AdditivePairwiseLoss
from pytorchltr.evaluation.dcg import ndcg
from pytorchltr.evaluation.arp import arp

from experiments.evaluate import evaluate
from experiments.util import load_dataset


LOGGER = logging.getLogger(__name__)


def get_parser():
    """Gets the parser to create arguments for `main`."""
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--vali_data", type=str, default=None)
    parser.add_argument('--optimizer', default="sgd",
                        choices=['sgd', 'adam', 'adagrad'])
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    return parser


def main(args):
    """Trains the baseline ranker using given arguments."""
    torch.manual_seed(args.seed)

    LOGGER.info("Loading train data %s", args.train_data)
    train = load_dataset(args.train_data, normalize=True)

    if args.vali_data is not None:
        LOGGER.info("Loading vali data %s", args.vali_data)
        vali = load_dataset(
            args.vali_data, normalize=True, filter_queries=True)

    LOGGER.info("Subsampling train data by %.3f", args.fraction)
    indices = torch.randperm(len(train))[0:int(args.fraction * len(train))]
    train = torch.utils.data.Subset(train, indices)

    LOGGER.info("Creating linear model")
    linear_model = torch.nn.Linear(train[0]["features"].shape[1], 1)

    LOGGER.info("Creating optimizer and loss function")
    optimizer = {
        "sgd": lambda: torch.optim.SGD(linear_model.parameters(), args.lr),
        "adam": lambda: torch.optim.Adam(linear_model.parameters(), args.lr),
        "adagrad": lambda: torch.optim.Adagrad(linear_model.parameters(), args.lr)
    }[args.optimizer]()
    loss_fn = AdditivePairwiseLoss("rank")

    LOGGER.info("Start training")
    for e in range(1, 1 + args.epochs):
        loader = torch.utils.data.DataLoader(
            train, batch_size=args.batch_size, shuffle=True,
            collate_fn=create_svmranking_collate_fn())
        for i, batch in enumerate(loader):
            linear_model.train()
            xs, ys, n = batch["features"], batch["relevance"], batch["n"]
            scores = linear_model(xs)
            loss = torch.mean(loss_fn(scores, ys, n))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        LOGGER.info("Finished epoch %d", e)

        if args.vali_data is not None:
            results = evaluate(vali, linear_model, {
                "arp": lambda scores, ys, n: arp(scores, ys, n),
                "ndcg@10": lambda scores, ys, n: ndcg(scores, ys, n, k=10)
            })
            LOGGER.info("arp    : %.4f", results["arp"])
            LOGGER.info("ndcg@10: %.4f", results["ndcg@10"])

    LOGGER.info("Saving model to %s", args.output)
    torch.save(linear_model, args.output)
    LOGGER.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s, %(levelname)s, %(module)s, %(threadName)s] %(message)s",
        level=logging.INFO)
    main(get_parser().parse_args())
