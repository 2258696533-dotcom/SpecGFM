"""环境版本检查脚本。

校验 Python、PyTorch、CUDA、PyG 及 DGL 版本是否符合项目预期，
用于运行实验前的依赖一致性确认。
"""

import sys
import numpy as np
import torch

def check_version():
    print("===== MDGFM 环境版本验证 =====")
    # 1. Python版本（从sys获取，而非torch）
    python_ver = sys.version.split()[0]
    python_req = "3.9.20"
    print(f"Python 版本: {python_ver} (要求: {python_req}) → {'✅ 符合' if python_ver == python_req else '❌ 不符'}")
    
    # 2. NumPy版本（需1.x，避免2.x兼容问题）
    numpy_ver = np.__version__
    numpy_req = "1.x (如1.26.4)"
    print(f"NumPy 版本: {numpy_ver} (要求: {numpy_req}) → {'✅ 符合' if numpy_ver.startswith('1.') else '❌ 不符 (需降级到1.x)'}")
    
    # 3. PyTorch版本（要求1.10.1+cu113）
    torch_ver = torch.__version__
    torch_req = "1.10.1+cu113"
    print(f"PyTorch 版本: {torch_ver} (要求: {torch_req}) → {'✅ 符合' if torch_ver == torch_req else '❌ 不符'}")
    
    # 4. CUDA可用性及版本
    cuda_available = torch.cuda.is_available()
    cuda_ver = torch.version.cuda if cuda_available else "无CUDA"
    cuda_req = "11.3"
    print(f"CUDA 可用: {cuda_available} → {'✅ 是' if cuda_available else '❌ 否'}")
    print(f"CUDA 版本: {cuda_ver} (要求: {cuda_req}) → {'✅ 符合' if cuda_ver == cuda_req else '❌ 不符'}")

    # 5. 验证PyG系列库（按需检查，MDGFM核心依赖）
    try:
        import torch_cluster
        import torch_geometric
        import torch_scatter
        import torch_sparse
        import torch_spline_conv
        print("\n===== PyG 系列库验证 =====")
        print(f"torch_cluster: {torch_cluster.__version__} (要求: 1.6.0) → {'✅ 符合' if torch_cluster.__version__ == '1.6.0' else '❌ 不符'}")
        print(f"torch_geometric: {torch_geometric.__version__} (要求: 2.1.0) → {'✅ 符合' if torch_geometric.__version__ == '2.1.0' else '❌ 不符'}")
        print(f"torch_scatter: {torch_scatter.__version__} (要求: 2.0.9) → {'✅ 符合' if torch_scatter.__version__ == '2.0.9' else '❌ 不符'}")
        print(f"torch_sparse: {torch_sparse.__version__} (要求: 0.6.13) → {'✅ 符合' if torch_sparse.__version__ == '0.6.13' else '❌ 不符'}")
        print(f"torch_spline_conv: {torch_spline_conv.__version__} (要求: 1.2.1) → {'✅ 符合' if torch_spline_conv.__version__ == '1.2.1' else '❌ 不符'}")
    except ImportError as e:
        print(f"\n❌ PyG 库导入失败: {e} → 请安装对应版本")

    # 6. DGL版本（要求0.9.1）
    try:
        import dgl
        dgl_ver = dgl.__version__
        dgl_req = "0.9.1"
        print(f"\nDGL 版本: {dgl_ver} (要求: {dgl_req}) → {'✅ 符合' if dgl_ver == dgl_req else '❌ 不符'}")
    except ImportError as e:
        print(f"\n❌ DGL 导入失败: {e} → 请安装 dgl==0.9.1")

if __name__ == "__main__":
    check_version()