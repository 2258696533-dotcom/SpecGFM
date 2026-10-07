"""Check the SpecGFM software stack (Python, PyTorch, CUDA 11.3, PyG, DGL)."""

import sys
import numpy as np
import torch


def _mark(ok: bool) -> str:
    return "ok" if ok else "mismatch"


def check_version():
    print("===== SpecGFM environment =====")
    python_ver = sys.version.split()[0]
    python_req = "3.9.20"
    print(f"Python: {python_ver} (expected {python_req}) -> {_mark(python_ver == python_req)}")

    numpy_ver = np.__version__
    print(
        f"NumPy: {numpy_ver} (expected 1.x, e.g. 1.26.4) -> "
        f"{_mark(numpy_ver.startswith('1.'))}"
    )

    torch_ver = torch.__version__
    torch_req = "1.10.1+cu113"
    print(f"PyTorch: {torch_ver} (expected {torch_req}) -> {_mark(torch_ver == torch_req)}")

    cuda_available = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_available else "unavailable"
    cuda_req = "11.3"
    print(f"CUDA available: {cuda_available}")
    print(f"CUDA: {cuda_ver} (expected {cuda_req}) -> {_mark(cuda_ver == cuda_req)}")

    try:
        import torch_cluster
        import torch_geometric
        import torch_scatter
        import torch_sparse
        import torch_spline_conv
        print("\n===== PyG =====")
        print(f"torch_cluster: {torch_cluster.__version__} (expected 1.6.0) -> {_mark(torch_cluster.__version__ == '1.6.0')}")
        print(f"torch_geometric: {torch_geometric.__version__} (expected 2.1.0) -> {_mark(torch_geometric.__version__ == '2.1.0')}")
        print(f"torch_scatter: {torch_scatter.__version__} (expected 2.0.9) -> {_mark(torch_scatter.__version__ == '2.0.9')}")
        print(f"torch_sparse: {torch_sparse.__version__} (expected 0.6.13) -> {_mark(torch_sparse.__version__ == '0.6.13')}")
        print(f"torch_spline_conv: {torch_spline_conv.__version__} (expected 1.2.1) -> {_mark(torch_spline_conv.__version__ == '1.2.1')}")
    except ImportError as e:
        print(f"\nPyG import failed: {e}")

    try:
        import dgl
        dgl_req = "0.9.1"
        print(f"\nDGL: {dgl.__version__} (expected {dgl_req}) -> {_mark(dgl.__version__ == dgl_req)}")
    except ImportError as e:
        print(f"\nDGL import failed: {e}. Install dgl==0.9.1.")


if __name__ == "__main__":
    check_version()
