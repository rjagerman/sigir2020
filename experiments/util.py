from pytorchltr.dataset.svmrank import svmranking_dataset as _load
from joblib.memory import Memory as _Memory

load_dataset = _Memory(".cache", compress=6).cache(_load)
