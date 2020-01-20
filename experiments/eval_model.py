from argparse import ArgumentParser
import logging
import json

import torch
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from experiments.dataset import load_ranking_dataset
from experiments.evaluate import evaluate
from experiments.evaluate import ARP
from experiments.evaluate import NDCG
from experiments.train import create_ltr_evaluator
from experiments.util import get_torch_device


LOGGER = logging.getLogger(__name__)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--eval_batch_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(args):
    torch.manual_seed(args.seed)
    device = get_torch_device(enable_cuda=False)

    LOGGER.info("Loading dataset")
    data_loader = torch.utils.data.DataLoader(
        load_ranking_dataset(args.data, normalize=True, filter_queries=True),
        shuffle=False, batch_size=args.eval_batch_size,
        collate_fn=create_svmranking_collate_fn())

    LOGGER.info("Loading model")
    model = torch.load(args.model)

    LOGGER.info("Starting evaluation")
    metrics = {"ndcg@10": NDCG(k=10), "arp": ARP()}
    evaluator = create_ltr_evaluator(model, device, metrics)
    evaluator.run(data_loader)
    with open(args.output, "wt") as f:
        json.dump(evaluator.state.metrics, f)

    LOGGER.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s, %(levelname)s, %(module)s] %(message)s",
        level=logging.INFO)
    main(get_parser().parse_args())
