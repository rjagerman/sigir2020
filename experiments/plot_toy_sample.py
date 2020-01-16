import numpy as np
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{biolinum}\n\\usepackage{sfmath}\n\\usepackage[T1]{fontenc}\n\\usepackage[libertine]{newtxmath}' #\\usepackage{libertine}\n
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


# Parameters for toy data and experiments
rng_seed = 42
np.random.seed(rng_seed)

wstar = torch.tensor([[0.973, 1.144]], dtype=torch.float)
xs = torch.tensor(np.random.randn(50, 2), dtype=torch.float)
labels = torch.mm(xs, wstar.T)

p = torch.tensor(np.random.uniform(0.05, 1.0, xs.shape[0]), dtype=torch.float)
ips = 1.0 / p
n_iters = 50
plot_every = n_iters // 10
arrow_width = 0.012
legends = {}

# The loss function we want to optimize
def loss_fn(out, y, mult):
    l2loss = (out - y) ** 2.0
    logl2loss = torch.log(1.0 + (out - y) ** 2.0)
    return torch.mean(mult * l2loss)

# IPS-weighted approach
for color_index, lr in enumerate([0.01, 0.02, 0.03, 0.05, 0.1]):  # 0.01, 0.03, 0.05, 0.1, 0.3
    color = "C%d" % (color_index + 2)
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = SWA(optimizer, swa_start=0, swa_freq=1, swa_lr=lr)
    with torch.no_grad():
        model.bias.zero_()
        model.weight.zero_()
    old_weights = np.copy(model.weight.data.numpy())
    np.random.seed(rng_seed + color_index + 1)
    for t in range(n_iters):
        i = np.random.randint(xs.shape[0])
        x = xs[i, :]
        y = labels[i]
        optimizer.zero_grad()
        o = model(x)
        l = loss_fn(o, y, ips[i])
        l.backward()
        optimizer.step()
        if t % plot_every == 0:
            optimizer.swap_swa_sgd()
            x, y = model.weight.data.numpy()[0]
            ox, oy = old_weights[0]
            label = f"IPS SGD ($\\eta={lr}$)"
            arr = plt.arrow(ox, oy, x - ox, y - oy, width=arrow_width, length_includes_head=True,
                      color=color, label=label)
            old_weights = np.copy(model.weight.data.numpy())
            optimizer.swap_swa_sgd()
            legends[label] = arr

# Sample based approach
lr = 3.0 # 1.0
model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
optimizer = SWA(optimizer, swa_start=0, swa_freq=1, swa_lr=lr)
with torch.no_grad():
    model.bias.zero_()
    model.weight.zero_()
old_weights = np.copy(model.weight.data.numpy())
sample_probs = np.array(ips / torch.sum(ips))
Mbar = float(np.mean(sample_probs))
np.random.seed(rng_seed)
for t in range(n_iters):
    i = np.argwhere(np.random.multinomial(1, sample_probs) == 1.0)[0, 0]
    x = xs[i, :]
    y = labels[i]
    optimizer.zero_grad()
    o = model(x)
    l = loss_fn(o, y, Mbar)
    l.backward()
    optimizer.step()
    if t % plot_every == 0:
        x, y = model.weight.data.numpy()[0]
        ox, oy = old_weights[0]
        label = f"CounterSample ($\\eta={lr}$)"
        arr = plt.arrow(ox, oy, x - ox, y - oy, width=arrow_width, length_includes_head=True,
                  color="C1", label=label)
        old_weights = np.copy(model.weight.data.numpy())
        legends[label] = arr

# True IPS-weighted loss over all datapoints, used for plotting contour
def f(x1, x2):
    w = torch.tensor([[x1], [x2]])
    o = torch.mm(xs, w)
    return float(loss_fn(o, torch.mm(xs, wstar.reshape((2, 1))), ips))

# Compute all useful combinations of weights and compute true loss for each one
# This will be used to compute a contour plot
true_x1 = np.linspace(float(wstar[0, 0]) - 1.5, float(wstar[0, 0]) + 0.8) # - 1.5 / + 1.0
true_x2 = np.linspace(float(wstar[0, 1]) - 1.5, float(wstar[0, 1]) + 1.6) # - 1.5 / + 1.0
true_x1, true_x2 = np.meshgrid(true_x1, true_x2)
true_y = np.array([
    [f(true_x1[i1, i2], true_x2[i1, i2]) for i2 in range(len(true_x2))]
    for i1 in range(len(true_x1))
])

# Contour plot with optimum
plt.plot(wstar[0, 0], wstar[0, 1], marker='o', markersize=3, color="black")
plt.contour(true_x1, true_x2, true_y, 10, colors="black", alpha=0.35)

# Generate legends from arrows and make figure
def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True,
                            head_width=0.75*height)
    return p

labels = [key for key in sorted(legends.keys())]
arrows = [legends[key] for key in sorted(legends.keys())]
plt.legend(arrows, labels, ncol=2, loc='upper center', framealpha=0.95, handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})
plt.xlabel("$w_1$")
plt.ylabel("$w_2$")
plt.tight_layout()
plt.savefig("test.pdf")
