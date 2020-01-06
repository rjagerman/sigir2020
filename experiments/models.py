import torch


class DeepScorer(torch.nn.Module):
    """A deep feed-forward neural network with non-linear activations."""
    def __init__(self, input_size, layers=[], activation_fn=torch.nn.ReLU):
        super().__init__()
        current_size = input_size
        self.layers = torch.nn.ModuleList([])
        self.activations = torch.nn.ModuleList([])
        for layer in layers:
            self.layers.append(torch.nn.Linear(input_size, layer))
            self.activations.append(activation_fn())
            current_size = layer
        self.output = torch.nn.Linear(current_size, 1)

    def forward(self, xs):
        current = xs
        for layer, activation in zip(self.layers, self.activations):
            current = activation(layer(current))
        return self.output(current)


class LinearScorer(torch.nn.Module):
    """A linear model."""
    def __init__(self, input_size):
        super().__init__()
        self.output = torch.nn.Linear(input_size, 1)

    def forward(self, xs):
        return self.output(xs)
