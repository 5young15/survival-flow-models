# FlowSurv / GumbelFlowSurv

基于流匹配的生存分析模型实现，支持复杂生存时间分布的精确密度估计。

## 项目结构

```
statistical_modeling/
├── models/
│   ├── flowmodel/           # 流模型实现
│   │   ├── base_flow.py     # FlowSurv基础模型
│   │   ├── gumbel_flow.py   # GumbelFlowSurv扩展
│   │   └── components.py    # 网络组件
│   ├── baselines/           # 基线模型
│   │   ├── coxph.py         # Cox比例风险
│   │   ├── deepsurv.py      # DeepSurv
│   │   ├── deephit.py       # DeepHit
│   │   ├── weibullAFT.py    # Weibull AFT
│   │   └── RSF.py           # 随机生存森林
│   └── interface.py         # 统一模型接口
├── experiments/
│   ├── config.py            # 实验配置
│   ├── main.py              # 实验入口
│   ├── run_experiments.py   # 实验运行器
│   ├── data_generation.py   # 数据生成
│   ├── metrics.py           # 评估指标
│   └── visualization.py     # 可视化
├── docs/
│   ├── FLOW_MODEL.md        # 流模型详解
│   └── EXPERIMENT_DESIGN.md # 实验设计
├── tests/
│   └── test_models.py       # 单元测试
└── run_experiments.py       # 启动脚本
```

## 快速开始

### 安装依赖

```bash
pip install torch numpy pandas scikit-learn lifelines matplotlib tqdm
```

### 运行实验

```bash
# 快速测试（2次重复）
python run_experiments.py --quick --groups E1_PH_Baseline --models FlowSurv GumbelFlowSurv

# 完整实验
python run_experiments.py --groups E1 E2 E3 --models DeepSurv FlowSurv GumbelFlowSurv

# 所有实验组
python run_experiments.py
```

### 命令行参数

| 参数 | 说明 |
|------|------|
| `--quick` | 快速模式，减少重复次数 |
| `--groups` | 指定实验组（E1-E10） |
| `--models` | 指定模型列表 |
| `--output` | 输出目录 |

## 模型列表

### 流模型

| 模型 | 先验分布 | 适用场景 |
|------|----------|----------|
| FlowSurv | 标准正态 | 通用NPH场景 |
| GumbelFlowSurv | Gumbel | 右偏分布、高删失 |

### 基线模型

| 模型 | 类型 | 特点 |
|------|------|------|
| LinearCoxPH | 半参数 | PH假设，无密度估计 |
| DeepSurv | 神经网络 | PH假设，非线性特征 |
| WeibullAFT | 参数 | 强分布假设 |
| DeepHit | 神经网络 | 离散化近似 |
| RSF | 集成学习 | 无密度估计 |

## 实验组设计

| 组别 | 样本量 | 删失率 | 分布类型 | 目的 |
|------|--------|--------|----------|------|
| E1 | 2000 | 40% | Weibull单峰 | PH基准 |
| E2 | 2000 | 40% | Weibull混合 | NPH标准 |
| E3 | 2000 | 70% | Weibull混合 | 高删失 |
| E4 | 2000 | 40% | Gaussian混合 | 多峰分布 |
| E5 | 500 | 50% | Weibull混合 | 小样本 |
| E6 | 2000 | 50% | Weibull混合 | 高噪声 |
| E7 | 5000 | 40% | Weibull混合 | 大样本 |
| E8 | 10000 | 40% | Weibull混合 | 超大样本 |
| E9 | 5000 | 70% | Weibull混合 | 大样本+高删失 |
| E10 | 10000 | 70% | Weibull混合 | 超大样本+高删失 |

## 评估指标

- **C-index**：一致性指数，衡量风险排序能力
- **IBS**：积分Brier分数，衡量校准能力
- **Hazard MSE**：风险函数MSE
- **Density MSE**：密度函数MSE

## 文档

- [流模型详解](docs/FLOW_MODEL.md)：设计理念、架构、数学推导
- [实验设计](docs/EXPERIMENT_DESIGN.md)：实验组设计、评估方法

## 核心特性

1. **两阶段训练框架**：GumbelFlowSurv 引入两阶段训练流程（Weibull AFT 预训练 + 流匹配微调），显著提升高删失场景下的模型稳定性。
2. **精确密度估计**：通过连续正规化流 (CNF) 实现生存时间分布的精确建模，不依赖比例风险 (PH) 假设。
3. **灵活先验分布**：支持标准正态先验 (FlowSurv) 和动态 Gumbel 先验 (GumbelFlowSurv)，适应不同偏度的生存数据。
4. **高效 ODE 求解**：内置基于 `torchdiffeq` 的批处理散度计算，优化反向传播性能与内存占用。
5. **统一评估体系**：集成 C-index, IBS, Hazard MSE, Density MSE 等多维生存分析指标。
