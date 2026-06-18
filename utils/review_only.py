REVIEW_ONLY_ARTIFACT = True

REVIEW_ONLY_MESSAGE = (
    "This anonymous review artifact intentionally omits executable distributed "
    "worker bootstrap code. The public release keeps the CSER replay, sampling, "
    "and objective logic for review, but it is not runnable as a training system."
)


class ReviewOnlyRuntimeError(RuntimeError):
    """Raised when omitted runtime-only worker code is invoked."""


def review_only_runtime_unavailable(component="worker"):
    raise ReviewOnlyRuntimeError(f"{component}: {REVIEW_ONLY_MESSAGE}")

