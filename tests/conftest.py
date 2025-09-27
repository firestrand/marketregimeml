import os
import tempfile


def pytest_sessionstart(session):
    # Ensure Matplotlib has a writable config dir in constrained environments
    if "MPLCONFIGDIR" not in os.environ:
        os.environ["MPLCONFIGDIR"] = tempfile.mkdtemp(prefix="mplconfig_")
