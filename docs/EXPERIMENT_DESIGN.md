# FlowSurv / GumbelFlowSurv 仿真实验设计

## 1. 实验目的与动机

本研究旨在验证基于连续流匹配的生成式生存分析模型（FlowSurv 与 GumbelFlowSurv）在以下两个层面的优势：

### 1.1 传统预测性能层面

验证流模型在区分度（C-index）和校准度上与传统方法具有竞争力。

### 1.2 密度与风险曲面刻画层面

这是流模型的**核心优势**所在：

| 模型类型 | 密度估计能力 | 局限性 |
|---------|-------------|--------|
| Cox 模型 | ❌ 无法显式建模 | 仅提供相对风险 |
| DeepHit | ⚠️ 离散化近似 | 精度受限于bin数量 |
| Weibull AFT | ⚠️ 单峰假设 | 无法捕捉多峰、复杂形状 |
| **FlowSurv** | ✅ 连续精确 | 标准正态先验，通用场景 |
| **GumbelFlowSurv** | ✅✅ 鲁棒稳定 | **两阶段训练**，Gumbel先验，适合右偏/高删失 |

流模型通过连续可逆变换，能够精确拟合任意形状的 $f(t|x)$、$h(t|x)$ 和 $S(t|x)$。

### 1.3 核心假设

- 在简单 PH 场景下，所有模型表现接近
- 在 NPH（非比例风险）、高删失、复杂密度场景下，流模型显著优于传统方法
- **GumbelFlowSurv** 通过两阶段训练，在处理具有强 Weibull 倾向的真实数据时比 FlowSurv 更稳定

---

## 2. 数据生成方式

### 2.1 协变量设计

$$X = (X_1, X_2, X_3, \ldots, X_d), \quad d \in [5, 10]$$

**信号变量**（真正影响生存）：
- $X_1 \sim \text{Uniform}(-1, 1)$：线性效应
- $X_2 \sim \text{Uniform}(-1, 1)$：非线性效应（如 $X_2^2$）

**噪声变量**：
- $X_3, \ldots, X_d \sim \mathcal{N}(0, 1)$：纯噪声

### 2.2 两种基础分布

#### 分布 A：Weibull 混合

$$f(t|x) = \pi_1(x) \cdot f_{\text{Weibull}}(t; k_1, \lambda_1(x)) + \pi_2(x) \cdot f_{\text{Weibull}}(t; k_2, \lambda_2(x))$$

**参数设置**：
- 形状参数：$k_1 = 1.5$, $k_2 = 3.5$（产生非比例风险）
- 尺度参数：$\lambda_j(x) = \exp(\alpha_j + \beta_j^T X)$
- 混合权重：$\pi_1(x) = \sigma(\gamma^T X)$

**特点**：hazard 交叉、多峰密度、NPH

#### 分布 B：高斯混合（对数时间域）

$$\log T | X \sim \pi_1 \cdot \mathcal{N}(\mu_1(x), \sigma_1^2) + \pi_2 \cdot \mathcal{N}(\mu_2(x), \sigma_2^2)$$

**参数设置**：
- $\mu_1 = 1.5$, $\mu_2 = 4.0$
- $\sigma_1 = 0.4$, $\sigma_2 = 0.6$
- $\pi_1 = 0.5$

**特点**：可控多峰、密度形状多样

### 2.3 删失机制

$$C \sim \text{Exponential}(\lambda_c)$$

观测时间 $Y = \min(T, C)$，事件指示 $\delta = \mathbb{I}(T \leq C)$

**删失率控制**：
- 常规删失：$P(\delta=0) \approx 35-45\%$
- 高删失：$P(\delta=0) \approx 60-75\%$

---

## 3. 对比模型列表

| 模型 | 类型 | 密度估计 | Hazard估计 | 主要特点 |
|------|------|----------|------------|----------|
| **LinearCoxPH** | 半参数 | ❌ | ❌ | 基线，假设PH |
| **DeepSurv** | 半参数神经网络 | ❌ | ❌ | 非线性PH |
| **WeibullAFT** | 全参数 | ✅ | ✅ | 单峰、假设Weibull |
| **RSF** | 非参数 | ❌ | 近似 | 无分布假设，无显式密度 |
| **DeepHit** | 离散时间 | 近似 | 近似 | 离散化损失 |
| **FlowSurv** | 生成式流 | ✅✅ | ✅✅ | 连续密度，标准正态先验 |
| **GumbelFlowSurv** | 生成式流 | ✅✅ | ✅✅ | Gumbel先验，适合右偏数据 |

---

## 4. 实验分组

| 组别 | 分布类型 | PH/NPH | 样本量 | 删失率 | 噪声维度 | 噪声强度 | 目的 |
|------|----------|--------|--------|--------|----------|----------|------|
| **E1** | Weibull 单峰 | PH | 2000 | 40% | 5维 (2信号+3噪声) | 低 | 基线场景 |
| **E2** | Weibull 混合 | NPH | 2000 | 40% | 5维 | 低 | 非比例风险 |
| **E3** | Weibull 混合 | NPH | 2000 | **70%** | 5维 | 低 | **高删失+NPH** |
| **E4** | 高斯混合 | NPH | 2000 | 40% | 5维 | 低 | 多峰密度 |
| **E5** | Weibull 混合 | NPH | **500** | 50% | 5维 | 低 | 小样本场景 |
| **E6** | Weibull 混合 | NPH | 2000 | 50% | **10维** | **高** | 高维噪声 |
| **E7** | Weibull 混合 | NPH | **5000** | 40% | 5维 | 低 | 大样本场景 |
| **E8** | Weibull 混合 | NPH | **10000** | 40% | 5维 | 低 | 超大样本场景 |
| **E9** | Weibull 混合 | NPH | **5000** | **70%** | 5维 | 低 | 大样本+高删失 |
| **E10** | Weibull 混合 | NPH | **10000** | **70%** | 5维 | 低 | 超大样本+高删失 |

---

## 5. 评估指标

### 5.1 A层指标：所有模型报告

| 指标 | 公式 | 说明 |
|------|------|------|
| **C-index** | $P(\hat{r}_i > \hat{r}_j \mid T_i < T_j)$ | 区分度，越高越好 |
| **Integrated Brier Score** | $\frac{1}{t_{\max}} \int_0^{t_{\max}} BS(t) dt$ | 校准度，越低越好 |
| **BS@25%/50%/75%** | $BS(t) = \mathbb{E}[(\mathbb{I}(T > t) - \hat{S}(t))^2]$ | 时间依赖Brier Score |
| **Median MAE** | $\frac{1}{n}\sum \|T_{med} - \hat{T}_{med}\|$ | 中位时间误差 |
| **Median RMSE** | $\sqrt{\frac{1}{n}\sum (T_{med} - \hat{T}_{med})^2}$ | 中位时间误差 |

### 5.2 B层指标：仅密度模型参与

| 指标 | 公式 | 适用模型 |
|------|------|----------|
| **Hazard MSE** | $\frac{1}{nG} \sum_{i,g} (h_{true}(t_g) - \hat{h}(t_g))^2$ | Weibull, FlowSurv, GumbelFlowSurv |
| **Hazard IAE** | $\int \|h_{true}(t) - \hat{h}(t)\| dt$ | 同上 |
| **Density MSE** | $\frac{1}{nG} \sum_{i,g} (f_{true}(t_g) - \hat{f}(t_g))^2$ | 同上 |
| **Wasserstein-1** | $\int \|F_{true}(t) - \hat{F}(t)\| dt$ | 同上 |

---

## 6. 可视化计划

### 6.1 每组必画图形

| 图号 | 图形类型 | 内容 | 目的 |
|------|----------|------|------|
| **Fig-1** | Hazard曲线对比 | 4-6个代表性个体的真实h(t) vs 各模型估计 | 展示复杂hazard捕捉能力 |
| **Fig-2** | 风险曲面图 | x轴=时间t，y轴=$X_1$，颜色=h(t) | 展示hazard时空变化 |
| **Fig-3** | 密度曲线对比 | 4-5个协变量值的真实f(t) vs 各模型估计 | 展示多峰、偏态拟合 |
| **Fig-4** | Survival曲线 | 真实S(t) vs 各模型估计 | 验证生存函数一致性 |

### 6.2 高删失场景额外图形

| 图号 | 图形类型 | 内容 |
|------|----------|------|
| **Fig-5** | 高删失S(t)对比 | E3组，验证高删失下的鲁棒性 |
| **Fig-6** | 尾部密度对比 | 长尾部分的密度估计 |

### 6.3 汇总图形

| 图号 | 图形类型 | 内容 |
|------|----------|------|
| **Fig-Summary-1** | 热力图 | 各模型在各场景的C-index矩阵 |
| **Fig-Summary-2** | 柱状图 | Hazard MSE / Density MSE 对比 |
| **Fig-Summary-3** | 箱线图 | 10次重复实验的指标分布 |

---

## 7. 预期结果假设

### 7.1 传统指标（C-index, IBS）

| 场景 | 预期排名 |
|------|----------|
| E1 (PH, 低删失) | 所有模型接近，Cox/DeepSurv 略优 |
| E2 (NPH, 低删失) | FlowSurv ≈ GumbelFlowSurv > DeepHit > RSF > Cox |
| E3 (NPH, 高删失) | **GumbelFlowSurv > FlowSurv >> 其他** |
| E4 (多峰) | FlowSurv > GumbelFlowSurv > DeepHit > Weibull |
| E5 (小样本) | RSF ≈ Cox > FlowSurv (流模型需更多数据) |
| E6 (高噪声) | 正则化好的模型表现更稳定 |
| E7-E10 (大样本) | 流模型优势进一步扩大，收敛更稳定 |

### 7.2 密度/Hazard指标

**流模型大幅领先场景**：E2, E3, E4（非比例风险 + 复杂密度）

**预期差距**：
- Hazard MSE：FlowSurv 比传统方法低 **30-50%**
- Density MSE：FlowSurv 比传统方法低 **40-60%**

---

## 8. 可重复性说明

| 项目 | 设置 |
|------|------|
| **随机种子** | `seed = base_seed + group_id × 10 + rep_id` |
| **重复次数** | 每组 10 次，报告均值 ± 标准差 |
| **超参数搜索** | 每个模型使用验证集早停 |
| **硬件环境** | 记录 GPU 型号、PyTorch 版本 |
| **代码版本** | Git commit hash 记录 |

---

## 9. 代码使用说明

### 9.1 目录结构

```
experiments/
├── config.py            # 配置文件（dataclass风格）
├── data_generation.py   # 数据生成模块
├── metrics.py           # 评估指标模块
├── run_experiments.py   # 实验运行脚本
├── visualization.py     # 可视化模块
└── main.py              # 主入口脚本
```

### 9.2 运行命令

```bash
# 快速测试（2次重复）
python experiments/main.py --quick

# 完整实验（10次重复）
python experiments/main.py --full

# 指定模型和实验组
python experiments/main.py \
    --models LinearCoxPH DeepSurv FlowSurv GumbelFlowSurv \
    --groups E1_PH_Baseline E3_NPH_HighCensoring

# 自定义输出目录
python experiments/main.py --output my_results
```

### 9.3 输出文件

```
results/
├── experiment_results.json    # 详细结果
├── aggregated_results.json    # 聚合统计
└── figures/                   # 可视化图形
    ├── hazard_curves.png
    ├── density_curves.png
    └── ...
```

---

## 10. 参考文献

1. **Flow Matching**: Lipman et al. "Flow Matching for Generative Modeling" (ICLR 2023)
2. **DeepSurv**: Katzman et al. "DeepSurv: Personalized Treatment Recommender System Using A Cox Proportional Hazards Deep Neural Network" (2018)
3. **DeepHit**: Lee et al. "DeepHit: A Deep Learning Approach to Survival Analysis with Competing Risks" (AAAI 2018)
4. **Random Survival Forest**: Ishwaran et al. "Random Survival Forests" (2008)
