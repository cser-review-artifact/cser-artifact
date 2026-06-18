"""Review-only vLLM actor worker facade.

The executable worker bootstrap is intentionally omitted from the anonymous
review artifact. The method-facing sampling and replay logic remains in
``run.py``; this module only preserves the public import surface.
"""

from utils.review_only import review_only_runtime_unavailable


REVIEW_ONLY_ARTIFACT = True


def torch_dist_worker(*args, **kwargs):
    review_only_runtime_unavailable("actor worker")


def update(*args, **kwargs):
    review_only_runtime_unavailable("actor model update")


def update_statedict(*args, **kwargs):
    review_only_runtime_unavailable("actor state update")


def check_params(*args, **kwargs):
    review_only_runtime_unavailable("actor parameter check")


def validate_update(*args, **kwargs):
    review_only_runtime_unavailable("actor update validation")

