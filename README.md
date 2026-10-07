# SpecGFM

论文代码：**Spectral Graph Foundation Model with Homophily-Guided Dual-Branch Adaptation**（IEEE TKDE）。

协议与正文一致：六个目标域轮流作为 **unseen target**，其余五个图做预训练。报告 1-shot / 5-shot 节点分类。RQ2 是四个组件变体，协议与 1-shot 主表相同。

本仓库只包含 **SpecGFM 主实验** 和 **RQ2 四个变体**。对比基线（GCN、GAT、DGI、GraphCL、GPPT、MTG、MDGFM、SCGFM）不在这里。

---

## 1. 硬件

训练入口使用 CUDA（代码里调用 `.cuda()`）。CPU 不能跑完整实验。

| 项目 | 要求 |
|------|------|
| GPU | NVIDIA，**CUDA 11.3**，显存 **≥ 24 GB** |
| 内存 | **≥ 64 GB** |
| CPU | x86_64 多核即可。开发机为 Intel Xeon Gold 6348（112 线程） |
| 磁盘 | 六个 PyG 图 + checkpoint，预留约 20 GB |

显存来自 BandGSL：低通/高通候选邻接是稠密矩阵。Pubmed 有 19,717 个节点，一张 float32 的 \(n \times n\) 矩阵大约 1.6 GB，预训练会同时保留多份。Cora、Cornell 本身不大，但 leave-one-out 时 Pubmed 经常是源域，所以按 24 GB 准备。11 GB 卡在 Pubmed 上会不够。

---

## 2. 软件环境

与论文实验相同：

| 组件 | 版本 |
|------|------|
| Python | 3.9.20 |
| PyTorch | 1.10.1+cu113 |
| CUDA | 11.3 |
| NumPy | 1.x（例如 1.26.4，不要 2.x） |
| PyTorch Geometric | 2.1.0 |
| torch-scatter / sparse / cluster / spline-conv | 2.0.9 / 0.6.13 / 1.6.0 / 1.2.1 |
| DGL | 0.9.1 |
| SciPy, scikit-learn, tqdm | 当前稳定版即可 |

```bash
conda create -n specgfm python=3.9 -y && conda activate specgfm
pip install torch==1.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
pip install numpy==1.26.4 scipy scikit-learn tqdm
pip install torch-scatter==2.0.9 torch-sparse==0.6.13 torch-cluster==1.6.0 torch-spline-conv==1.2.1 \
  -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
pip install torch-geometric==2.1.0
pip install dgl==0.9.1 -f https://data.dgl.ai/wheels/repo.html
python check_env.py
```

目录：`data/`（图和 few-shot 划分）、`checkpoints/`、`logs/`。数据不随仓库发布。

---

## 3. 数据集

放到 `data/` 下，用 PyG 的离线目录，不要在运行时重新下载。

| 数据集 | 论文中的类型 | PyG |
|--------|----------------|-----|
| Cora / Citeseer / Pubmed | homophilic | `Planetoid` |
| Cornell | heterophilic | `WebKB` |
| Chameleon / Squirrel | heterophilic | `WikipediaNetwork` |

节点数与正文数据集统计表一致：Cora 2,708；Citeseer 3,327；Pubmed 19,717；Cornell 183；Chameleon 2,277；Squirrel 5,201。

Few-shot 划分：`data/fewshot_<name>/<k>-shot_<name>/<episode>/{idx,labels}.pt`。没有这些文件时，用 `generate_idx.py` 在本地生成，不要改公开划分的节点文件。

---

## 4. 怎么跑

主实验（BandGSL + 结构坐标对齐 + 按 \(h_e\) 选择同配/异配分支 + 同配分支上的 support-GEE）：

```bash
python run_specgfm.py --mode specgfm --dataset Cora --seeds 1024 --shot_num 1
python run_specgfm.py --mode specgfm --dataset Cora --seeds 1024 --shot_num 5
```

六个目标域、1-shot（正文默认五个随机种子；这里先给一个种子，其余种子换 `--seeds`）：

```bash
for ds in Cora Citeseer Pubmed Cornell Chameleon Squirrel; do
  python run_specgfm.py --mode specgfm --dataset "$ds" --seeds 1024 --shot_num 1
done
```

RQ2（都是 1-shot，和主表同一套 unseen-target 协议）：

| `--mode` | 论文名称 | 去掉什么 |
|----------|----------|----------|
| `rq2_wo_band` | w/o-Band | BandGSL（结构坐标对齐一起去掉） |
| `rq2_wo_he` | w/o-He | 异配分支，所有 episode 走同配分支 |
| `rq2_wo_ho` | w/o-Ho | 同配分支，等价于把路由阈值 \(\tau\) 锁成 1 |
| `rq2_wo_gee` | w/o-GEE | 同配分支上的 support-GEE |

```bash
python run_specgfm.py --mode rq2_wo_band --dataset Cora --seeds 1024 --shot_num 1
python run_specgfm.py --mode rq2_wo_he   --dataset Cora --seeds 1024 --shot_num 1
python run_specgfm.py --mode rq2_wo_ho   --dataset Cora --seeds 1024 --shot_num 1
python run_specgfm.py --mode rq2_wo_gee  --dataset Cora --seeds 1024 --shot_num 1
```

同配分支会把三个独立初始化的线性头的 logits 取平均（`--dual_ensemble 3`）。路由阈值 \(\tau\) 对应 `--homo_bypass_thresh`（主实验 0.52）：\(h_e>\tau\) 走同配分支，否则走异配分支。support-GEE 只加在同配分支（`--f2_gee_branch dual`）。

---

## 5. 文件

| 文件 | 作用 |
|------|------|
| `run_specgfm.py` | 主实验和 RQ2 的启动器 |
| `SpecGFM.py` | 预训练与 few-shot 评估 |
| `preprompt.py` | BandGSL 与多域预训练 |
| `scgw_utils.py` | 两个 BandGSL 邻接的结构坐标对齐 |
| `dual_training.py` | 同配分支（线性头 + 原型头 + support-GEE） |
| `downprompt_bikt.py` | 异配分支 |
| `downprompt.py` | 异配分支用到的 prompt / 原型 |
| `downstream_encoder.py` | episode 内的 prompt 与 GSL 编码 |
| `models/f2_downstream_plugins.py` | support-GEE |
| `models/branch_utils.py` | 按 \(h_e\) 选分支 |
| `aug.py` `tools.py` `generate_idx.py` | 增强、图工具、few-shot 划分 |
| `layers/` `models/` `utils/` | GCN 主干与预训练损失 |
| `check_env.py` | 检查上面的软件版本 |
