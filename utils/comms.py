"""Review-only MPI communication facade.

The executable MPI communicator setup, rank routing, message tags, request
lifecycle, and payload sharding details are intentionally omitted from the
anonymous review artifact. These function names are kept so the public code
still shows the actor-buffer-learner dataflow boundaries.
"""

from utils.review_only import review_only_runtime_unavailable


REVIEW_ONLY_ARTIFACT = True


def actor_send(*args, **kwargs):
    review_only_runtime_unavailable("MPI actor-to-buffer channel")


def buffer_recv(*args, **kwargs):
    review_only_runtime_unavailable("MPI buffer receive channel")


def buffer_send(*args, **kwargs):
    review_only_runtime_unavailable("MPI buffer-to-learner channel")


def learner_recv(*args, **kwargs):
    review_only_runtime_unavailable("MPI learner receive channel")


def learner_send(*args, **kwargs):
    review_only_runtime_unavailable("MPI learner-to-actor channel")


def actor_recv(*args, **kwargs):
    review_only_runtime_unavailable("MPI actor receive channel")
