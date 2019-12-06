"""Counterfactual Learning to Rank training procedure."""
import logging
import json
from argparse import ArgumentParser

import torch
from pytorchltr.dataset.svmrank import svmranking_dataset
from pytorchltr.loss.pairwise import AdditivePairwiseLoss
from pytorchltr.evaluation.dcg import ndcg
from pytorchltr.evaluation.arp import arp
from joblib import Memory

from experiments.evaluate import evaluate
from experiments.simulate_clicks import clicklog_dataset
from experiments.simulate_clicks import create_clicklog_collate_fn


LOGGER = logging.getLogger(__name__)
METRICS = {
    "arp": lambda scores, ys, n: arp(scores, ys, n),
    "ndcg@10": lambda scores, ys, n: ndcg(scores, ys, n, k=10)
}
memory = Memory("./.cache", compress=6)
svmranking_dataset = memory.cache(svmranking_dataset)


def get_parser():
    """Gets the parser to create arguments for `main`."""
    parser = ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--click_log", type=str, required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--vali_data", type=str, default=None)
    parser.add_argument("--test_data", type=str, default=None)
    parser.add_argument("--optimizer", default="sgd",
                        choices=["sgd", "adam", "adagrad"])
    parser.add_argument("--ips_strategy", default="none",
                        choices=["none", "weight", "sample"])
    parser.add_argument("--ips_clip", type=float, default=None)
    parser.add_argument("--objective", default="rank",
                        choices=["rank", "normrank", "dcg"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enable_cuda", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_list_size", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=1000)
    return parser


def main(args):
    """Trains the baseline ranker using given arguments."""
    if args.enable_cuda and torch.cuda.is_available():
        args.device = torch.device("cuda")
        LOGGER.info("Using device %s", args.device)
    else:
        args.device = torch.device("cpu")
    torch.manual_seed(args.seed)

    LOGGER.info("Loading train data %s", args.train_data)
    train = svmranking_dataset(args.train_data, normalize=True)

    LOGGER.info("Loading click log %s", args.click_log)
    click_log = clicklog_dataset(train, args.click_log, clip=args.ips_clip)

    if args.vali_data is not None:
        LOGGER.info("Loading vali data %s", args.vali_data)
        vali = svmranking_dataset(
            args.vali_data, normalize=True, filter_queries=True)

    if args.test_data is not None:
        LOGGER.info("Loading test data %s", args.test_data)
        test = svmranking_dataset(
            args.test_data, normalize=True, filter_queries=True)

    LOGGER.info("Creating linear model")
    linear_model = torch.nn.Linear(train[0]["features"].shape[1], 1)
    linear_model = linear_model.to(device=args.device)

    LOGGER.info("Creating optimizer and loss function")
    optimizer = {
        "sgd": lambda: torch.optim.SGD(linear_model.parameters(), args.lr),
        "adam": lambda: torch.optim.Adam(linear_model.parameters(), args.lr),
        "adagrad": lambda: torch.optim.Adagrad(linear_model.parameters(), args.lr)
    }[args.optimizer]()
    loss_fn = AdditivePairwiseLoss(args.objective)

    if args.ips_strategy == "sample":
        propensities = torch.FloatTensor(click_log.propensities)
        weights = (1.0 / propensities)
        probabilities = weights / torch.sum(weights)
        sampler = torch.utils.data.sampler.WeightedRandomSampler(
            probabilities, len(click_log))
    else:
        sampler = torch.utils.data.sampler.RandomSampler(click_log)

    LOGGER.info("Start training")
    count = 0
    sample_count = 0
    batch_count = 0
    out_results = {
        "args": vars(args)
    }
    if args.vali_data is not None:
        out_results["vali"] = {key: [] for key in METRICS.keys()}
        out_results["vali"]["x"] = []
    if args.test_data is not None:
        out_results["test"] = {key: [] for key in METRICS.keys()}
        out_results["test"]["x"] = []

    if args.vali_data is not None:
        results = evaluate(
            vali, linear_model, METRICS,
            batch_size=args.batch_size, device=args.device)
        record_results(out_results["vali"], sample_count, results)
    if args.test_data is not None:
        results = evaluate(
            test, linear_model, METRICS,
            batch_size=args.batch_size, device=args.device)
        record_results(out_results["test"], sample_count, results)

    for e in range(1, 1 + args.epochs):
        loader = torch.utils.data.DataLoader(
            click_log, batch_size=args.batch_size, sampler=sampler,
            collate_fn=create_clicklog_collate_fn(
                max_list_size=args.max_list_size))

        for i, batch in enumerate(loader):
            linear_model.train()
            xs, ys, n, p = batch["features"], batch["relevance"], batch["n"], batch["propensity"]
            xs, ys, n, p = xs.to(args.device), ys.to(args.device), n.to(args.device), p.to(args.device)
            scores = linear_model(xs)
            loss = loss_fn(scores, ys, n)
            if args.ips_strategy == "weight":
                loss = loss / p
            loss = torch.mean(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            count += batch["features"].shape[0]
            sample_count += batch["features"].shape[0]
            batch_count += 1
            if count >= args.log_every:
                if args.vali_data is not None:
                    results = evaluate(
                        vali, linear_model, METRICS,
                        batch_size=args.batch_size, device=args.device)
                    record_results(out_results["vali"], sample_count, results)
                if args.test_data is not None:
                    results = evaluate(
                        test, linear_model, METRICS,
                        batch_size=args.batch_size, device=args.device)
                    record_results(out_results["test"], sample_count, results)
                count = count % args.log_every
        LOGGER.info("Finished epoch %d", e)

    if args.output is not None:
        LOGGER.info("Writing results to output")
        with open(args.output, "wt") as f:
            json.dump(out_results, f, indent=2)

    LOGGER.info("Done")


def record_results(out_results, x, results):
    for key in METRICS.keys():
        LOGGER.info("[%7d] %-7s : %.4f", x, key, results[key])
        out_results[key].append(results[key])
    out_results["x"].append(x)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s, %(levelname)s, %(module)s, %(threadName)s] %(message)s",
        level=logging.INFO)
    main(get_parser().parse_args())
