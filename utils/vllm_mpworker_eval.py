"""Review-only vLLM evaluation worker facade.

The executable vLLM distributed bootstrap is intentionally omitted from the
anonymous review artifact. Evaluation scripts may keep importing this module,
but invoking the worker is not supported in the public review version.
"""

from utils.review_only import review_only_runtime_unavailable


REVIEW_ONLY_ARTIFACT = True


def torch_dist_worker(*args, **kwargs):
    review_only_runtime_unavailable("vLLM evaluation worker")


def update(*args, **kwargs):
    review_only_runtime_unavailable("vLLM evaluation model update")


def update_statedict(*args, **kwargs):
    review_only_runtime_unavailable("vLLM evaluation state update")

