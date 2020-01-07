"""Counterfactual Learning to Rank training procedure."""
import logging
import json
from argparse import ArgumentParser

import torch
from torchcontrib.optim import SWA
from ignite.engine import Engine
from ignite.engine import Events
from pytorchltr.loss.pairwise import AdditivePairwiseLoss
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn

from experiments.evaluate import evaluate
from experiments.evaluate import NDCG
from experiments.evaluate import ARP
from experiments.click_log import clicklog_dataset
from experiments.click_log import create_clicklog_collate_fn
from experiments.dataset import load_click_dataset
from experiments.dataset import load_ranking_dataset
from experiments.models import LinearScorer
from experiments.util import get_torch_device
from experiments.util import JsonLogger
from experiments.util import every_n_iteration


LOGGER = logging.getLogger(__name__)


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
    parser.add_argument("--disable_swa", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=500)
    parser.add_argument("--n_clicks", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_list_size", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=1000)
    return parser


def create_pairwise_loss(objective, ips_strategy, bcf=1.0):
    """Creates a pairwise loss objective

    Arguments:
        objective: A string indicating the AdditivePairwiseLoss objective.
        ips_strategy: A string indicating the IPS strategy to use.
        bcf: Bias correction factor for sample-based learning.

    Returns:
        A loss function that takes 4 arguments: (scores, ys, n, p).
    """
    pairwise_loss_fn = AdditivePairwiseLoss(objective)
    if ips_strategy == "weight":
        def _loss_fn(scores, ys, n, p):
            return pairwise_loss_fn(scores, ys, n) / p
    else:
        def _loss_fn(scores, ys, n, p):
            return pairwise_loss_fn(scores, ys, n) * bcf
    return _loss_fn


def create_cfltr_trainer(optimizer, loss_fn, model, device, metrics={}):
    """Creates a training `Engine` for counterfactual LTR.

    Arguments:
        optimizer: The optimizer to use.
        loss_fn: The loss function to call.
        model: The model to predict scores with.
        device: The device to move batches to.
        metrics: Metrics to compute per step.

    Returns:
        An `Engine` used for training.
    """
    def _update_fn(engine, batch):
        model.train()
        xs, clicks, n, relevance, p = (
            batch["features"], batch["clicks"], batch["n"], batch["relevance"],
            batch["propensity"])
        xs, clicks, n, relevance, p = (
            xs.to(device), clicks.to(device), n.to(device),
            relevance.to(device), p.to(device))
        model.train()
        optimizer.zero_grad()
        scores = model(xs)
        loss = torch.mean(loss_fn(scores, clicks, n, p))
        loss.backward()
        optimizer.step()
        return scores, relevance, n

    engine = Engine(_update_fn)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def create_ltr_evaluator(model, device, metrics):
    """Creates an evaluation `Engine` for LTR.

    Arguments:
        model: The model to predict scores with.
        device: The device to move batches to.
        metrics: Metrics to compute per step.

    Returns:
        An `Engine` used for evaluation.
    """
    def _inference_fn(engine, batch):
        model.eval()
        with torch.no_grad():
            xs, relevance, n = batch["features"], batch["relevance"], batch["n"]
            xs, relevance, n = xs.to(device), relevance.to(device), n.to(device)
            scores = model(xs)
            return scores, relevance, n

    engine = Engine(_inference_fn)
    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def main(args):
    """Trains the baseline ranker using given arguments."""

    LOGGER.info("Setting device and seeding RNG")
    device = get_torch_device(args.enable_cuda)
    torch.manual_seed(args.seed)

    LOGGER.info("Loading click log for training")
    train_data_loader, input_dimensionality, bcf = load_click_dataset(
        args.train_data, args.click_log, args.ips_strategy, args.ips_clip,
        args.n_clicks, args.batch_size, args.max_list_size)

    eval_data_loaders = {}
    if args.vali_data is not None:
        LOGGER.info("Loading vali data")
        eval_data_loaders["vali"] = torch.utils.data.DataLoader(
            load_ranking_dataset(args.vali_data, normalize=True,
                                 filter_queries=True),
            shuffle=False, batch_size=args.eval_batch_size,
            collate_fn=create_svmranking_collate_fn())

    if args.test_data is not None:
        LOGGER.info("Loading test data")
        eval_data_loaders["vali"] = torch.utils.data.DataLoader(
            load_ranking_dataset(args.test_data, normalize=True,
                                 filter_queries=True),
            shuffle=False, batch_size=args.eval_batch_size,
            collate_fn=create_svmranking_collate_fn())

    LOGGER.info("Creating linear model")
    model = LinearScorer(input_dimensionality)
    model = model.to(device=device)

    LOGGER.info("Creating loss function (bcf=%f)", bcf)
    loss_fn = create_pairwise_loss(args.objective, args.ips_strategy, bcf)
    args.bias_correction_factor = bcf

    LOGGER.info("Creating optimizer")
    optimizer = {
        "sgd": lambda: torch.optim.SGD(model.parameters(), args.lr),
        "adam": lambda: torch.optim.Adam(model.parameters(), args.lr),
        "adagrad": lambda: torch.optim.Adagrad(model.parameters(), args.lr)
    }[args.optimizer]()

    if not args.disable_swa:
        optimizer = SWA(optimizer, swa_start=0, swa_freq=1, swa_lr=args.lr)

    if args.output is not None:
        LOGGER.info("Creating result logger")
        json_logger = JsonLogger(args.output, indent=1, args=args)

    LOGGER.info("Setup training engine")
    trainer = create_cfltr_trainer(
        optimizer, loss_fn, model, device)

    for eval_name, eval_data_loader in eval_data_loaders.items():
        LOGGER.info("Setup %s engine", eval_name)
        metrics = {"ndcg@10": NDCG(k=10), "arp": ARP()}
        evaluator = create_ltr_evaluator(model, device, metrics)
        if not args.disable_swa:
            swa_evaluator = create_ltr_evaluator(model, device, metrics)

        # Run evaluation
        def run_evaluation(trainer):
            if not args.disable_swa:
                optim_state = optimizer.state_dict()
                if "swa_state" in optim_state and len(optim_state["swa_state"]) > 0:
                    optimizer.swap_swa_sgd()
                swa_evaluator.run(eval_data_loader)
                if "swa_state" in optim_state and len(optim_state["swa_state"]) > 0:
                    optimizer.swap_swa_sgd()
            evaluator.run(eval_data_loader)

        trainer.add_event_handler(Events.STARTED, run_evaluation)
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED,
            every_n_iteration(trainer, args.eval_every, run_evaluation))

        # Write results to file when evaluation finishes.
        if args.output is not None:
            if not args.disable_swa:
                @swa_evaluator.on(Events.COMPLETED)
                def log_results(evaluator):
                    json_logger.append_all(
                        "%s/avgmodel" % eval_name, trainer.state.iteration,
                        swa_evaluator.state.metrics)
                    json_logger.write_to_disk()

            @evaluator.on(Events.COMPLETED)
            def log_results(evaluator):
                json_logger.append_all(
                    "%s/latestmodel" % eval_name, trainer.state.iteration,
                    evaluator.state.metrics)
                json_logger.write_to_disk()

    LOGGER.info("Run train loop")
    trainer.run(train_data_loader, args.epochs)

    LOGGER.info("Writing final results to disk.")
    json_logger.write_to_disk()

    LOGGER.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s, %(levelname)s, %(module)s] %(message)s",
        level=logging.INFO)
    main(get_parser().parse_args())
