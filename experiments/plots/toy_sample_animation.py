from argparse import ArgumentParser
from argparse import FileType

import json
import numpy as np
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{biolinum}\n\\usepackage{sfmath}\n\\usepackage[T1]{fontenc}' #\\usepackage{libertine}\n
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 14})
import torch
from torchcontrib.optim import SWA
from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation


def get_parser():
    """Gets the parser to create arguments for `main`."""
    parser = ArgumentParser()
    parser.add_argument("--out", type=str, required=True, default=None)
    parser.add_argument("--format", type=str, default=None)
    return parser


def main(args):

    # Parameters for toy data and experiments
    plt.style.use('dark_background')
    rng_seed = 4200
    np.random.seed(rng_seed)

    wstar = torch.tensor([[0.973, 1.144]], dtype=torch.float)
    xs = torch.tensor(np.random.randn(50, 2), dtype=torch.float)
    labels = torch.mm(xs, wstar.T)

    p = torch.tensor(np.random.uniform(0.05, 1.0, xs.shape[0]), dtype=torch.float)
    ips = 1.0 / p
    n_iters = 500
    plot_every = n_iters // 10
    arrow_width = 0.012 * 0.01
    legend = {}
    data = {}
    colors = {}
    interpsize = 30

    # Figure
    fig = plt.figure(figsize=(8.4,5.4), dpi=150)
    ax = fig.add_subplot(111)
    ax.set_xlim([-0.3, 1.8])
    ax.set_ylim([-0.3, 1.8])

    # The loss function we want to optimize
    def loss_fn(out, y, mult):
        l2loss = (out - y) ** 2.0
        logl2loss = torch.log(1.0 + (out - y) ** 2.0)
        return torch.mean(mult * l2loss)

    # True IPS-weighted loss over all datapoints, used for plotting contour
    def f(x1, x2):
        w = torch.tensor([[x1], [x2]], dtype=torch.float)
        o = torch.mm(xs, w)
        return float(loss_fn(o, torch.mm(xs, wstar.reshape((2, 1))), ips))

    # Plot contour
    true_x1 = np.linspace(float(wstar[0, 0]) - 1.5, float(wstar[0, 0]) + 0.8) # - 1.5 / + 1.0
    true_x2 = np.linspace(float(wstar[0, 1]) - 1.5, float(wstar[0, 1]) + 1.2) # - 1.5 / + 1.0
    true_x1, true_x2 = np.meshgrid(true_x1, true_x2)
    true_y = np.array([
        [f(true_x1[i1, i2], true_x2[i1, i2]) for i2 in range(len(true_x2))]
        for i1 in range(len(true_x1))
    ])
    ax.contour(true_x1, true_x2, true_y, levels=10, colors='white', alpha=0.45)
    plt.plot(0.0, 0.0, marker='o', markersize=6, color="white")
    plt.plot(wstar[0, 0], wstar[0, 1], marker='*', markersize=6, color="white")

    # IPS-weighted approach
    for color_index, lr in enumerate([0.1, 0.01, 0.03]):  # 0.01, 0.03, 0.05, 0.1, 0.3
        if lr == 0.01:
            color = "violet"
        elif lr == 0.03:
            color = "orange"
        else:
            color = "lightgreen"
        #color = "C%d" % (color_index + 2)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer = SWA(optimizer, swa_start=0, swa_freq=1, swa_lr=lr)
        with torch.no_grad():
            model.bias.zero_()
            model.weight.zero_()
        old_weights = np.copy(model.weight.data.numpy())
        np.random.seed(rng_seed + color_index + 1)# + color_index + 1)
        label = f"IPS-SGD ($\\eta={lr}$)"
        data[label] = np.zeros((3, n_iters * interpsize + 1))
        colors[label] = color
        for t in range(n_iters):
            i = np.random.randint(xs.shape[0])
            x = xs[i, :]
            y = labels[i]
            optimizer.zero_grad()
            o = model(x)
            l = loss_fn(o, y, ips[i])
            l.backward()
            optimizer.step()

            # Record current iteration performance and location
            optimizer.swap_swa_sgd()
            x, y = model.weight.data.numpy()[0]
            optimizer.swap_swa_sgd()
            old_x, old_y, old_z = data[label][:, t * interpsize]
            xr = np.linspace(old_x, x, num=interpsize)
            yr = np.linspace(old_y, y, num=interpsize)
            for i in range(interpsize):
                data[label][:, 1 + t * interpsize + i] = np.array([
                    xr[i], yr[i], f(xr[i], yr[i])])
            #data[label][:, t] = np.array([x, y, f(x, y)])


    # Sample based approach
    lr = 10.0 # 1.0
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = SWA(optimizer, swa_start=0, swa_freq=1, swa_lr=lr)
    with torch.no_grad():
        model.bias.zero_()
        model.weight.zero_()
    old_weights = np.copy(model.weight.data.numpy())
    sample_probs = np.array(ips / torch.sum(ips))
    Mbar = float(np.mean(sample_probs))
    np.random.seed(rng_seed - 1)
    label = f"\\textsc{{CounterSample}} ($\\eta={lr}$)"
    data[label] = np.zeros((3, n_iters * interpsize + 1))
    for t in range(n_iters):
        i = np.argwhere(np.random.multinomial(1, sample_probs) == 1.0)[0, 0]
        x = xs[i, :]
        y = labels[i]
        optimizer.zero_grad()
        o = model(x)
        l = loss_fn(o, y, Mbar)
        l.backward()
        optimizer.step()

        # Record current iteration location and performance
        optimizer.swap_swa_sgd()
        x, y = model.weight.data.numpy()[0]
        optimizer.swap_swa_sgd()
        old_x, old_y, old_z = data[label][:, t * interpsize]
        xr = np.linspace(old_x, x, num=interpsize)
        yr = np.linspace(old_y, y, num=interpsize)
        for i in range(interpsize):
            data[label][:, 1 + t * interpsize + i] = np.array([
                xr[i], yr[i], f(xr[i], yr[i])])
        colors[label] = "deepskyblue"

    # Print summary to quickly find performance at convergence
    for label in data.keys():
        print(f"{label}: {data[label][2, -1]}")

    # Create legend
    lines = {}
    for label in data.keys():
        line = data[label]
        lines[label] = ax.plot(line[:, 0], line[:, 1], color=colors[label], label=label, linewidth=2.0)
        legend[colors[label]] = label

    ax.set_xlabel('$w_1$')
    ax.set_ylabel('$w_2$')
    legend_lines = [
        lines[legend["deepskyblue"]][0],
        lines[legend["orange"]][0],
        lines[legend["violet"]][0],
        lines[legend["lightgreen"]][0]]
    legend_labels = [
        "\\textsc{CounterSample}",
        "IPS-SGD (best learning rate)",
        "IPS-SGD (learning rate too small)",
        "IPS-SGD (learning rate too large)"]
    legend_artist = ax.legend(legend_lines, legend_labels, loc='lower right') #, bbox_to_anchor=(1.0 - 0.3, 0.25))

    # Update function for animation
    n_frames = n_iters * 2
    n_data = n_iters * interpsize
    from math import floor
    def transform_num(num):
        x = 3 * ((1.0 * num) / n_frames)
        y = (((x + 0.5)**2 - 0.5**2) / 12.0)
        return floor(y * n_data)

    def update_lines(num):
        print(f"frame {num:4d} / {n_frames:4d} [{num / n_frames * 100:.0f}%]", end="\r")
        num = transform_num(num)
        out = [legend_artist]
        for label in data.keys():
            line = lines[label][0]
            d = data[label]
            line.set_data(d[0:2, :num])
            out.append(line)
        return out

    # Write animation to file
    line_ani = animation.FuncAnimation(fig, update_lines, n_frames, interval=100, blit=False, repeat_delay=3000)
    writer = animation.FFMpegWriter(fps=60, codec='h264')   # for keynote
    line_ani.save(args.out, writer=writer)
    print("\033[K\n")

if __name__ == "__main__":
    main(get_parser().parse_args())
