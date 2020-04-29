import logging
import json
import os
from argparse import ArgumentParser
from argparse import FileType
from scipy.stats import ttest_ind_from_stats

import numpy as np

from experiments.dataset import load_ranking_dataset
from experiments.click_log import clicklog_dataset


def get_parser():
    """Gets the parser to create arguments for `main`."""
    parser = ArgumentParser()
    parser.add_argument("--istella_click_logs", type=FileType("rt"), nargs="+")
    parser.add_argument("--istella_dataset", type=FileType("rt"))
    parser.add_argument("--yahoo_click_logs", type=FileType("rt"), nargs="+")
    parser.add_argument("--yahoo_dataset", type=FileType("rt"))
    return parser


def load_propensity_stats(dataset, click_logs):
    r = load_ranking_dataset(dataset.name)
    clicklogs = []
    for c in click_logs:
        clicklogs.append(clicklog_dataset(r, c.name))

    out = {}
    for c, clicklog in zip(click_logs, clicklogs):
        out[c.name] = {
            'max': np.max(1.0 / clicklog.propensities),
            'mean': np.mean(1.0 / clicklog.propensities)
        }
    return out


def main(args):
    # Load propensity distribution
    datasets = {
        "yahoo": load_propensity_stats(args.yahoo_dataset, args.yahoo_click_logs),
        "istella": load_propensity_stats(args.istella_dataset, args.istella_click_logs)
    }

    # Print
    latex_metrics = {
        "mean": r"$\bar{M}$",
        "max": r"$M$"
    }
    latex_datasets = {
        "yahoo": r"\Yahoo{}",
        "istella": r"\Istella{}"
    }
    print(r"%!TEX root = ../sigir2020.tex")
    print(r"\begin{tabular}{l@{\hspace{6mm}}rrrrr}")
    print(r"\toprule")
    print(r"Position bias ($\gamma$): & 0.5 & 0.75 & 1.0 & 1.25 & 1.5 \\")
    for dataset in datasets.keys():
        print(r"\midrule")
        for metric in ["mean", "max"]:
            print(f"{latex_datasets[dataset]}: {latex_metrics[metric]} ", end='')
            # row dataset+metric
            for eta in ["0.5", "0.75", "1.0", "1.25", "1.5"]:
                # col eta
                d = list(filter(lambda d: ("_eta_" + eta) in d, datasets[dataset].keys()))[0]
                print(f"& {datasets[dataset][d][metric]:.2f}", end='')
            print(r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

if __name__ == "__main__":
    main(get_parser().parse_args())
