import torch
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn
from pytorchltr.evaluation.dcg import ndcg
from pytorchltr.evaluation.arp import arp
from ignite.metrics.metric import Metric
from ignite.exceptions import NotComputableError


def evaluate(dataset, model, metrics,
             collate_fn=create_svmranking_collate_fn(), batch_size=16,
             device=None):
    model.eval()
    out_metrics = {
        metric: []
        for metric in metrics.keys()
    }
    with torch.no_grad():
        eval_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn)
        for batch in eval_loader:
            xs, ys, n = batch["features"], batch["relevance"], batch["n"]
            xs, ys, n = xs.to(device), ys.to(device), n.to(device)
            scores = model(xs)
            for metric in metrics.keys():
                values = metrics[metric](scores, ys, n)
                out_metrics[metric].extend(values.tolist())
    return {
        metric: float(torch.mean(torch.FloatTensor(values)))
        for metric, values in out_metrics.items()
    }


class OnlineAverageMetric(Metric):
    """
    Calculates an online average metric.
    """
    def __init__(self, output_transform=lambda x: x):
        super().__init__(output_transform)
        self._avg = 0.0

    def reset(self):
        self._avg = 0.0
        self._n = 0

    def update(self, output):
        for e in output:
            self._avg = (self._avg * self._n + e.item()) / (self._n + 1)
            self._n += 1

    def compute(self):
        if self._n == 0:
            raise NotComputableError("OnlineAverageMetric must have at least one example before it can be computed.")
        return self._avg


class LTRMetric(OnlineAverageMetric):
    """
    Calculates an LTR metric online.
    """
    def __init__(self, ltr_metric_fn, output_transform=lambda x: x):
        super().__init__(self._compute_metric)
        self._ltr_metric_fn = ltr_metric_fn
        self._output_fn = output_transform

    def _compute_metric(self, output):
        scores, ys, n = self._output_fn(output)
        return self._ltr_metric_fn(scores, ys, n)


class NDCG(LTRMetric):
    def __init__(self, k=10, exp=True, output_transform=lambda x: x):
        super().__init__(lambda scores, ys, n: ndcg(scores, ys, n, k, exp),
                         output_transform=output_transform)


class ARP(LTRMetric):
    def __init__(self, output_transform=lambda x: x):
        super().__init__(arp, output_transform=output_transform)
