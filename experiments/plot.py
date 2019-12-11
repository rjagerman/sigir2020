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
            ys = np.array(results["vali"]["ndcg@10"])
            xs = np.array(results["vali"]["iteration"])
            ax.plot(xs, ys, label=name, color=color(name))
        ax.set_ylim([0.68, 0.75])
        fig.legend()

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
