import logging
import json
import os
from copy import deepcopy
from time import sleep
from argparse import ArgumentParser
from argparse import FileType

import numpy as np
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{biolinum}\n\\usepackage{sfmath}\n\\usepackage[T1]{fontenc}\n\\usepackage[libertine]{newtxmath}' #\\usepackage{libertine}\n
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 13})
import tikzplotlib
from matplotlib import pyplot as plt
from matplotlib import animation as anim
plt.rcParams["figure.figsize"] = [12,3]


def get_parser():
    """Gets the parser to create arguments for `main`."""
    parser = ArgumentParser()
    parser.add_argument("--json_files", type=FileType("rt"), nargs="+")
    parser.add_argument("--color_by", type=str, required=False, default="ips_strategy",
                        choices=["none", "ips_strategy", "optimizer"])
    parser.add_argument("--out", type=FileType("wb"), required=False, default=None)
    parser.add_argument("--dataset", type=str, required=False, default="vali")
    parser.add_argument("--model", type=str, required=False, default="avgmodel")
    parser.add_argument("--metric", type=str, default="ndcg@10")
    parser.add_argument("--legend", action="store_true", default=False)
    parser.add_argument("--width", type=float, default=12.0)
    parser.add_argument("--height", type=float, default=3.0)
    parser.add_argument("--format", type=str, default=None)
    parser.add_argument("--points", type=int, default=None)
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


def compress_list_of_arrays(list_of_arrays):
    min_shape = np.min([np.array(a).shape for a in list_of_arrays])
    mean = np.mean([a[:min_shape] for a in list_of_arrays], axis=0)
    std = np.std([a[:min_shape] for a in list_of_arrays], axis=0)
    n = len(list_of_arrays)
    return mean, std, n


def filter_arg(dictionary, key, value):
    return {k: v for k, v in dictionary.items() if v["args"][key] == value}


def sample_points(xs, points):
    if points is None:
        return xs
    else:
        # Plot the first 20% of the points perfectly (where convergence mostly takes place),
        # subsample the remainder of the points to prevent very slow PDF viewing
        start_points = points // 5
        end_points = points - start_points
        sample = np.hstack([
            np.arange(start_points, dtype=np.int32),
            np.linspace(start_points, xs.shape[0] - 1, num=end_points, dtype=np.int32)])
        return xs[sample]


def main(args):

    fig = plt.figure(figsize=(args.width, args.height))

    def plot_data(args, fig):
        fig.clear()
        data = {}
        for json_file in args.json_files:
            data[os.path.basename(json_file.name)] = json.load(json_file)
            json_file.seek(0)

        color = {
            "optimizer": color_by_optimizer,
            "ips_strategy": color_by_ips_strategy,
            "none": lambda x: None
        }[args.color_by]

        # Compute error bars for identical runs with multiple seeds
        avg_runs = {}
        for name, results in data.items():
            if args.dataset in results and args.model in results[args.dataset]:
                x = results[args.dataset][args.model]["iteration"]
                y = results[args.dataset][args.model][args.metric]
                a = deepcopy(results['args'])
                del a["seed"]
                del a["output"]
                key = tuple(sorted(a.items()))
                if key not in avg_runs:
                    avg_runs[key] = {
                        'name': name,
                        'results': {args.dataset: {args.model: {args.metric: [], "iteration": []}}}
                    }
                avg_runs[key]['results'][args.dataset][args.model][args.metric].append(y)
                avg_runs[key]['results'][args.dataset][args.model]["iteration"].append(x)
                avg_runs[key]['args'] = results['args']
        data = {}
        for run in avg_runs.values():
            ys_mean, ys_std, ys_n = compress_list_of_arrays(run['results'][args.dataset][args.model][args.metric])
            xs_mean, _, _  = compress_list_of_arrays(run['results'][args.dataset][args.model]["iteration"])
            data[run['name']] = {
                args.dataset: {args.model: {
                    args.metric: ys_mean,
                    f"{args.metric}/std": ys_std,
                    f"{args.metric}/n": ys_n,
                    "iteration": xs_mean
                }},
                "args": run['args']
            }

        # Plot actual results
        labels = {
            "none": "Biased-SGD",
            "weight": "IPS-SGD",
            "sample": "\\textsc{CounterSample}"
        }
        inv_labels = {k: v for v, k in labels.items()}
        sorting = {
            "none": 3,
            "weight": 2,
            "sample": 1
        }
        metrics = {
            "ndcg@10": "nDCG@10",
            "arp": "ARP"
        }
        markers = {
            "none": "s",
            "weight": "o",
            "sample": "^"
        }
        optimizers = {
            "sgd": "SGD",
            "adam": "Adam",
            "adagrad": "Adagrad"
        }
        for j, dataset in enumerate(["yahoo", "istella"]):
            # Compute y limits
            max_last_y = 0.0
            min_last_y = 1e30
            for name, results in data.items():
                if args.dataset in results and args.model in results[args.dataset] and dataset in results["args"]["train_data"]:
                    y = results[args.dataset][args.model][args.metric][-1]
                    min_last_y = min(y, min_last_y)
                    max_last_y = max(y, max_last_y)

            # Plot subplots
            for i, optimizer in enumerate(["sgd", "adam", "adagrad"]):
                ax = fig.add_subplot(2, 3, 1 + i + j * 3)
                for name, results in filter_arg(data, "optimizer", optimizer).items():
                    if args.dataset in results and args.model in results[args.dataset] and dataset in results["args"]["train_data"]:
                        ys = np.array(results[args.dataset][args.model][args.metric])
                        xs = np.array(results[args.dataset][args.model]["iteration"])
                        xs = (xs * int(results['args']['batch_size'])) / 1_000_000
                        xs = sample_points(xs, args.points)
                        ys = sample_points(ys, args.points)
                        label = f"{results['args']['ips_strategy']}"
                        ax.plot(xs, ys, label=labels[label], color=color(name), marker=markers[label], markevery=0.1, markersize=4.5)
                        if f"{args.metric}/std" in results[args.dataset][args.model]:
                            ys_std = np.array(results[args.dataset][args.model][f"{args.metric}/std"])
                            ys_std = sample_points(ys_std, args.points)
                            ax.fill_between(xs, ys - ys_std, ys + ys_std, color=color(name), alpha=0.35)
                ax.set_ylim([0.99 * min_last_y, 1.01 * max_last_y])
                if dataset == "yahoo":
                    ax.set_title(f"Optimizer = {optimizers[optimizer]}")
                if i == 0:
                    ax.set_ylabel(metrics[args.metric])
                else:
                    ax.set_yticklabels([])
                if dataset == "istella":
                    ax.set_xlabel(r"Iterations ($\times 10^6$)")
                else:
                    ax.set_xticklabels([])
                ax.set_xticks(np.arange(0, 5 + 1, step=1.0))

        # Legend
        if args.legend:
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            keys = sorted(by_label.keys(), key=lambda key: sorting[inv_labels[key]])
            values = [by_label[key] for key in keys]
            fig.legend(values, keys, loc='upper center',
                       bbox_to_anchor=(0.5 + 0.02, 1.0 + 0.05), ncol=3)

    plot_data(args, fig)
    plt.tight_layout()

    if args.out is not None:
        if args.out.name.endswith(".tex"):
            tikzplotlib.save(args.out.name)
        else:
            plt.savefig(args.out, bbox_inches="tight", format=args.format)
    else:
        plt.show()


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s, %(levelname)s, %(module)s, %(threadName)s] %(message)s",
        level=logging.INFO)
    main(get_parser().parse_args())
