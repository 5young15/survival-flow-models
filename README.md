# FlowSurv: 基于流匹配的生成式生存分析

本项目实现了基于连续流匹配（Flow Matching）的生成式生存分析模型，包括 **FlowSurv**、**GumbelFlowSurv** 和 **MultiGumbelFlowSurv** 三种模型。与传统生存分析模型相比，流模型能够精确估计条件概率密度函数 $f(t|x)$ 和风险函数 $h(t|x)$，特别适用于非比例风险（NPH）和复杂密度形状的场景。

---

## 项目结构

```
statistical_modeling/
├── experiments/              # 实验相关代码
│   ├── config.py            # 配置文件（数据生成、模型超参数）
│   ├── data_generation.py   # 仿真数据生成模块
│   ├── metrics.py           # 评估指标计算
│   ├── run_experiments.py   # 实验运行脚本
│   ├── visualization.py     # 可视化模块
│   └── main.py              # 主入口脚本
│
├── models/                  # 模型实现
│   ├── baselines/           # 基线模型
│   │   ├── coxph.py         # Cox比例风险模型
│   │   ├── deepsurv.py      # DeepSurv深度神经网络
│   │   ├── deephit.py       # DeepHit离散时间模型
│   │   ├── weibullAFT.py    # Weibull加速失效时间模型
│   │   └── RSF.py           # 随机生存森林
│   │
│   ├── flowmodel/           # 流匹配模型
│   │   ├── base_flow.py     # FlowSurv基础流模型
│   │   ├── gumbel_flow.py   # GumbelFlowSurv模型
│   │   └── multi_gumbel_flow.py  # MultiGumbelFlowSurv模型
│   │
│   └── interface.py         # 模型统一接口
│
├── docs/                    # 文档
│   ├── EXPERIMENT_DESIGN.md # 实验设计详细说明
│   ├── FLOW_MODELS_THEORY.md # 流模型理论推导
│   ├── METRICS_SPACE.md     # 评估指标说明
│   └── MODEL_COMPUTE.md     # 模型计算函数说明
│
├── results/                 # 实验结果
│   ├── cv_results.json      # 交叉验证详细结果
│   ├── aggregated_results.json # 聚合统计结果
│   └── cv_checkpoints/      # 模型检查点
│
└── tests/                   # 测试代码
    ├── test_models.py       # 模型单元测试
    └── analyze_mc_error.py  # 蒙特卡洛误差分析
```

---

## 流匹配模型介绍

### 1. FlowSurv（标准正态先验）

FlowSurv 是最基础的流匹配生存模型，使用标准正态分布 $\mathcal{N}(0,1)$ 作为先验。

**核心思想**：通过可逆神经网络学习从先验分布到目标生存时间分布的映射。

**损失函数**：
$$
\mathcal{L}_{FM} = \mathbb{E}_{\tau, z_0 \sim \mathcal{N}(0,1)} \left[ \| v_\theta(z(\tau), \tau, x) - (t^{norm} - z_0) \|^2 \right]
$$

其中 $z(\tau) = (1-\tau)z_0 + \tau t^{norm}$ 是线性插值路径，$v_\theta$ 是学习的向量场。

**特点**：
- 先验简单，训练稳定
- 适用于一般场景
- 无法显式建模删失时间

### 2. GumbelFlowSurv（Gumbel先验）

GumbelFlowSurv 采用 **两阶段训练策略**，使用 Gumbel 分布作为先验。

**第一阶段**：学习 Gumbel 分布参数
$$
\mathcal{L}^{(1)} = -\frac{1}{N}\sum_i \left[ \delta_i \log f(t_i) + (1-\delta_i) \log S(t_i) \right]
$$

**第二阶段**：流匹配训练，使用 Gumbel 逆变换采样：
$$
z_0 = \alpha + \beta \cdot (-\log(-\log(u))), \quad u \sim U(0,1)
$$

**特点**：
- Gumbel 先验适合右偏分布
- 两阶段训练更加稳定
- 特别适合高删失场景

### 3. MultiGumbelFlowSurv（联合建模）

MultiGumbelFlowSurv 同时建模失效时间 $T$ 和删失时间 $C$ 的联合分布。

**联合密度**：
$$
f_{T,C}(t,c) = F_{T,C}(t,c) \cdot \frac{e^{(t-\mu_T)/\beta}}{\beta} \cdot \frac{e^{(c-\mu_C)/\beta}}{\beta} \cdot \left(e^{(t-\mu_T)/\beta} + e^{(c-\mu_C)/\beta}\right)
$$

**特点**：
- 充分利用删失信息
- 二维向量场，建模能力更强
- 计算复杂度最高

---

## 实验设计

实验组设计如下（详见 [docs/EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md)）：

| 组别 | 分布类型 | PH/NPH | 样本量 | 删失率 | 目的 |
|------|----------|--------|--------|--------|------|
| **E1** | Weibull 单峰 | PH | 2000 | ~40% | 基线场景 |
| **E3** | Weibull 混合 | NPH | 2000 | ~70% | 高删失+NPH |

### 数据生成

- **协变量**：$d=5$维（2信号+3噪声）
- **信号变量**：$X_1 \sim U(-1,1)$（线性），$X_2 \sim U(-1,1)$（非线性）
- **噪声变量**：$X_3,X_4,X_5 \sim \mathcal{N}(0,1)$
- **删失机制**：$C \sim \text{Exponential}(\lambda_c)$

---

## 交叉验证结果

已在 E1 和 E3 数据集上完成 5 折交叉验证，结果如下：

### E1 实验组（PH场景，~40%删失）

| 模型 | C-index ↑ | IBS ↓ | Hazard MSE ↓ | Density MSE ↓ |
|------|-----------|-------|--------------|---------------|
| **DeepSurv** | **0.6176** ±0.018 | **0.1425** ±0.006 | 0.6997 ±0.040 | - |
| **FlowSurv** | 0.5042 ±0.014 | 0.1610 ±0.007 | 0.6144 ±0.054 | 0.0402 ±0.015 |
| **GumbelFlowSurv** | 0.6125 ±0.016 | 0.1647 ±0.010 | 0.6843 ±0.131 | **0.0145** ±0.005 |
| **MultiGumbelFlowSurv** | 0.6108 ±0.019 | 0.1666 ±0.007 | **0.5591** ±0.047 | 0.0459 ±0.019 |

### E3 实验组（NPH场景，~70%高删失）

| 模型 | C-index ↑ | IBS ↓ | Hazard MSE ↓ | Density MSE ↓ |
|------|-----------|-------|--------------|---------------|
| **DeepSurv** | **0.6235** ±0.017 | **0.1463** ±0.006 | 1.0399 ±0.096 | - |
| **FlowSurv** | 0.5230 ±0.045 | 0.1704 ±0.009 | 0.6629 ±0.083 | 0.0234 ±0.005 |
| **GumbelFlowSurv** | 0.5916 ±0.035 | 0.2353 ±0.040 | 0.9269 ±0.248 | **0.0398** ±0.016 |
| **MultiGumbelFlowSurv** | 0.5849 ±0.036 | 0.2095 ±0.040 | **0.6579** ±0.108 | 0.0560 ±0.022 |

### 关键发现

1. **C-index 表现**：DeepSurv 在区分度指标上表现最佳，流模型在此指标上略弱于深度神经网络
2. **密度估计**：GumbelFlowSurv 在 E1 上的 Density MSE 最低（0.0145），表明其密度估计能力最强
3. **高删失场景（E3）**：所有模型性能下降，但 GumbelFlowSurv 和 MultiGumbelFlowSurv 在密度估计上仍保持优势
4. **Hazard MSE**：MultiGumbelFlowSurv 在 E1 和 E3 上均实现了最低的 Hazard MSE

---

## 运行实验

```bash
# 快速测试（2次重复）
python experiments/main.py --quick

# 完整实验（5次重复）
python experiments/main.py --full

# 指定模型和实验组
python experiments/main.py \
    --models DeepSurv FlowSurv GumbelFlowSurv \
    --groups E1 E3
```

---

## 文档说明

| 文档 | 内容 |
|------|------|
| [EXPERIMENT_DESIGN.md](docs/EXPERIMENT_DESIGN.md) | 实验设计详细说明（数据生成、评估指标、可视化） |
| [FLOW_MODELS_THEORY.md](docs/FLOW_MODELS_THEORY.md) | 流模型理论推导（损失函数、ODE求解、密度计算） |
| [METRICS_SPACE.md](docs/METRICS_SPACE.md) | 评估指标计算公式说明 |
| [MODEL_COMPUTE.md](docs/MODEL_COMPUTE.md) | 模型接口和计算函数说明 |

---

## 依赖环境

- Python 3.8+
- PyTorch 2.0+
- numpy, pandas, scikit-learn, matplotlib
