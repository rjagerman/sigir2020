import logging
import json
import os
from copy import deepcopy
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
    parser.add_argument("--color_by", type=str, required=False, default="ips_strategy",
                        choices=["none", "ips_strategy", "optimizer"])
    parser.add_argument("--out", type=FileType("wb"), required=False, default=None)
    parser.add_argument("--dataset", type=str, required=False, default="vali")
    parser.add_argument("--model", type=str, required=False, default="avgmodel")
    parser.add_argument("--metric", type=str, default="ndcg@10")
    parser.add_argument("--select_best_lr", action="store_true", default=False)
    parser.add_argument("--seed_error_bars", action="store_true", default=False)
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

        # Print all loaded results
        print_results = []
        for name, results in data.items():
            if args.dataset in results and args.model in results[args.dataset]:
                y = results[args.dataset][args.model][args.metric][-1]
                print_results.append((y, name))
        for value, name in sorted(print_results, key=lambda e: e[0]):
            print(f"{name:12s} {value:.4f}")

        # Select only best LR per setting
        if args.select_best_lr:
            print("============== selecting only best LR ============")
            best = {}
            for name, results in data.items():
                if args.dataset in results and args.model in results[args.dataset]:
                    y = np.mean(results[args.dataset][args.model][args.metric])
                    if args.metric == "arp":
                        y = -y
                    a = deepcopy(results['args'])
                    del a["lr"]
                    del a["output"]
                    key = tuple(sorted(a.items()))
                    if key not in best or best[key]['y'] < y:
                        best[key] = {}
                        best[key]['y'] = y
                        best[key]['name'] = name
                        best[key]['results'] = results
            data = {v['name']: v['results'] for v in best.values()}
            print_results = []
            for name, results in data.items():
                if args.dataset in results and args.model in results[args.dataset]:
                    y = np.mean(results[args.dataset][args.model][args.metric])
                    x = results[args.dataset][args.model]["iteration"][-1]
                    print_results.append((y, x, name))
            for value, x, name in sorted(print_results, key=lambda e: e[0]):
                print(f"{name:12s} {value:.4f} [{x:6d}]")

        # Compute error bars for identical runs with multiple seeds
        if args.seed_error_bars:
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

        # Compute y limits
        max_last_y = 0.0
        min_last_y = 1e30
        for name, results in data.items():
            if args.dataset in results and args.model in results[args.dataset]:
                y = results[args.dataset][args.model][args.metric][-1]
                min_last_y = min(y, min_last_y)
                max_last_y = max(y, max_last_y)

        # Plot actual results
        for name, results in data.items():
            if args.dataset in results and args.model in results[args.dataset]:
                ys = np.array(results[args.dataset][args.model][args.metric])
                xs = np.array(results[args.dataset][args.model]["iteration"])
                xs = (xs * int(results['args']['batch_size'])) / 1_000_000
                ax.plot(xs, ys, label=name, color=color(name))
                if f"{args.metric}/std" in results[args.dataset][args.model]:
                    ys_std = np.array(results[args.dataset][args.model][f"{args.metric}/std"])
                    ax.fill_between(xs, ys - ys_std, ys + ys_std, color=color(name), alpha=0.35)
        ax.set_ylim([0.96 * max_last_y, 1.01 * max_last_y])
        ax.set_ylabel(args.metric)
        ax.set_xlabel("Epochs")
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
