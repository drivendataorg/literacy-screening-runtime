import importlib
import subprocess

import pytest

packages = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "torch",
    "torchaudio",
    "transformers",
    "whisper",
    "speechbrain",
]


def is_gpu_available():
    try:
        return subprocess.check_call(["nvidia-smi"]) == 0

    except FileNotFoundError:
        return False


GPU_AVAILABLE = is_gpu_available()


@pytest.mark.parametrize("package_name", packages, ids=packages)
def test_import(package_name):
    """Test that certain dependencies are importable."""
    importlib.import_module(package_name)


@pytest.mark.skipif(not GPU_AVAILABLE, reason="No GPU available")
def test_allocate_torch():
    import torch

    assert torch.cuda.is_available()

    torch.zeros(1).cuda()
