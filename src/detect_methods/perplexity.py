import numpy as np
from torch import Tensor

from utils import time_performance_decorator


@time_performance_decorator(enable=False)
def calcu_perplexity(_probs: Tensor) -> float:
    # Ensure dtype is float64
    probs = _probs.to("cpu").numpy()
    probs = probs.astype(np.float64)
    log_probs = np.log(np.clip(probs, a_min=1e-9, a_max=None))
    mean_log_probs = np.mean(log_probs, axis=1)

    perplexity = np.exp(-mean_log_probs).mean()

    return perplexity.item()
