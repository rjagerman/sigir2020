import logging
import json
import os
from collections import defaultdict
from copy import deepcopy
from time import sleep
from argparse import ArgumentParser
from argparse import FileType
from scipy.stats import ttest_ind_from_stats

import numpy as np


def get_parser():
    """Gets the parser to create arguments for `main`."""
    parser = ArgumentParser()
    parser.add_argument("--json_files", type=FileType("rt"), nargs="+")
    parser.add_argument("--out", type=FileType("wb"), required=False, default=None)
    parser.add_argument("--dataset", type=str, required=False, default="vali")
    parser.add_argument("--model", type=str, required=False, default="avgmodel")
    parser.add_argument("--metric", type=str, default="ndcg@10")
    return parser


def compress_list_of_arrays(list_of_arrays):
    min_shape = np.min([np.array(a).shape for a in list_of_arrays])
    mean = np.mean([a[:min_shape] for a in list_of_arrays], axis=0)
    std = np.std([a[:min_shape] for a in list_of_arrays], axis=0)
    n = len(list_of_arrays)
    return mean, std, n


def filter_arg(dictionary, key, value):
    return {k: v for k, v in dictionary.items() if v["args"][key] == value}


def main(args):

    def create_table(args):
        data = {}
        for json_file in args.json_files:
            data[os.path.basename(json_file.name)] = json.load(json_file)
            json_file.seek(0)

        # Compute error bars for identical runs with multiple seeds
        max_y = defaultdict(float)
        avg_runs = {}
        for name, results in data.items():
            if args.dataset in results and args.model in results[args.dataset]:
                dataset = "istella" if "istella" in results["args"]["train_data"] else "yahoo"
                x = results[args.dataset][args.model]["iteration"]
                y = results[args.dataset][args.model][args.metric]
                for e in y:
                    max_y[dataset] = max(e, max_y[dataset])
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
            dataset = "istella" if "istella" in run["args"]["train_data"] else "yahoo"
            regret = np.mean(max_y[dataset] - np.array(run['results'][args.dataset][args.model][args.metric]), axis=1)
            ys_mean, ys_std, ys_n = np.mean(regret), np.std(regret), regret.shape[0]
            method = run["args"]["ips_strategy"]
            optimizer = run["args"]["optimizer"]
            if method not in data:
                data[method] = {}
            if dataset not in data[method]:
                data[method][dataset] = {}

            data[method][dataset][optimizer] ={
                "mean": ys_mean,
                "std": ys_std,
                "n": ys_n
            }

        methods = {
            "none": r"\BiasedSGD{}",
            "weight": r"\IPSSGD{}",
            "sample": r"\OurMethod{}"
        }
        datasets = {
            "yahoo": r"\Yahoo{}",
            "istella": r"\Istella{}"
        }
        print(r"\begin{tabular}{l@{\hspace{6mm}}rrr}")
        print(r"\toprule")
        print(r"Optimizer: & SGD & \textsc{Adam} & \textsc{Adagrad} \\")
        for dataset in ["yahoo", "istella"]:
            print(r"\midrule")
            print(f"{datasets[dataset]} \\\\")
            for method in ["none", "weight", "sample"]:
                result_row = []
                print(f"\quad{methods[method]}", end = " & ")
                for optimizer in ["sgd", "adam", "adagrad"]:
                    statsig = ""
                    if method != "weight":
                        d1 = data[method][dataset][optimizer]
                        d2 = data["weight"][dataset][optimizer]
                        res = ttest_ind_from_stats(d1["mean"], d1["std"], d1["n"], d2["mean"], d2["std"], d2["n"])
                        if res.pvalue < 0.01:
                            statsig = r"\rlap{$^{\triangledown}$}" if res.statistic < 0.0 else r"\rlap{$^{\triangle}$}"
                    result_row.append(f"{100.0 * data[method][dataset][optimizer]['mean']:.2f}{statsig}")
                print(" & ".join(result_row) + r" \\")
        print(r"\bottomrule")
        print(r"\end{tabular}")

    create_table(args)


if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s, %(levelname)s, %(module)s, %(threadName)s] %(message)s",
        level=logging.INFO)
    main(get_parser().parse_args())
