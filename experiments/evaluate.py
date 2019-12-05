import torch
from pytorchltr.dataset.svmrank import create_svmranking_collate_fn


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
