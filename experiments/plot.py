import logging
import json
import os
from time import sleep
from argparse import ArgumentParser
from argparse import FileType

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as anim
plt.rcParams["figure.figsize"] = [12,6]


def get_parser():
    """Gets the parser to create arguments for `main`."""
    parser = ArgumentParser()
    parser.add_argument("--json_files", type=FileType("rt"), nargs="+")
    parser.add_argument("--color_by", type=str, required=False, default="none",
                        choices=["none", "ips_strategy", "optimizer"])
    parser.add_argument("--out", type=FileType("wb"), required=False, default=None)
    parser.add_argument("--dataset", type=str, required=False, default="vali")
    parser.add_argument("--model", type=str, required=False, default="avgmodel")
    return parser


def color_by_ips_strategy(name):
    if "none" in name:
        return "C0"
    elif "weight" in name:
        return "C1"
    elif "sample" in name:
        return "C2"
    else:
        return None


def color_by_optimizer(name):
    if "adam" in name:
        return "C0"
    elif "sgd" in name:
        return "C1"
    elif "adagrad" in name:
        return "C2"
    else:
        return None


def main(args):

    fig = plt.figure()

    def plot_data(args, fig):
        fig.clear()
        ax = fig.add_subplot(111)
        data = {}
        for json_file in args.json_files:
            data[os.path.basename(json_file.name)] = json.load(json_file)
            json_file.seek(0)

        color = {
            "optimizer": color_by_optimizer,
            "ips_strategy": color_by_ips_strategy,
            "none": lambda x: None
        }[args.color_by]

        for name, results in data.items():
            if args.dataset in results and args.model in results[args.dataset]:
                ys = np.array(results[args.dataset][args.model]["arp"])
                xs = np.array(results[args.dataset][args.model]["iteration"])
                ax.plot(xs, ys, label=name, color=color(name))
        #ax.set_ylim([0.68, 0.75])
        ax.set_ylim([11.0, 12.0])
        ax.set_ylabel("ndcg@10")
        ax.set_xlabel("iterations (batch size=100)")
        fig.legend()

        print_results = []
        for name, results in data.items():
            if args.dataset in results and args.model in results[args.dataset]:
                y = results[args.dataset][args.model]["ndcg@10"][-1]
                print_results.append((y, name))
        for value, name in sorted(print_results, key=lambda e: e[0]):
            print(f"{name:12s} {value:.4f}")

    plot_data(args, fig)

    if args.out is not None:
        plt.savefig(args.out)
    else:
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s, %(levelname)s, %(module)s, %(threadName)s] %(message)s",
        level=logging.INFO)
    main(get_parser().parse_args())
