# FlowSurv / GumbelFlowSurv 流模型详解

## 1. 设计理念

### 1.1 传统生存分析的局限

传统生存分析方法存在以下局限：

| 方法 | 局限性 |
|------|--------|
| Cox PH | 仅提供相对风险，无法估计绝对风险和密度函数 |
| Weibull AFT | 强分布假设，无法拟合多峰或复杂形状 |
| DeepHit | 离散化近似，精度受限于时间bin数量 |
| RSF | 无显式密度估计，难以进行概率推断 |

### 1.2 流模型的核心思想

**FlowSurv** 基于连续正规化流的思想，通过学习一个可逆变换，将复杂的生存时间分布映射到简单的先验分布：

```
复杂生存分布 T ~ p_T(t|x)  <--可逆变换-->  简单先验分布 Z ~ p_Z(z)
```

核心优势：
- **精确密度估计**：通过变量变换定理精确计算任意形状的密度函数
- **灵活性强**：可拟合任意复杂的时间分布（多峰、偏态、厚尾等）
- **端到端训练**：无需分步估计，直接优化似然

---

## 2. 模型架构

### 2.1 整体架构图

```
输入特征 X
    │
    ▼
┌─────────────────┐
│  特征编码器      │  ResidualEncoderBlock × N
│  Encoder(X)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FiLM调制头      │  生成调制参数 (γ, β)
│  FilmHead(Z)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  向量场网络      │  条件流匹配核心
│  V_θ(τ, h, z)   │  τ: 流时间, h: 隐状态, z: 调制参数
└────────┬────────┘
         │
         ▼
    ODE积分求解
         │
         ▼
┌─────────────────┐
│  生存时间预测    │
│  T = ψ^{-1}(Z)  │
└─────────────────┘
```

### 2.2 核心组件

#### 2.2.1 时间嵌入 (SinusoidalEmbedding)

将流时间 τ ∈ [0,1] 编码为高维向量：

$$\text{Emb}(\tau)_i = \begin{cases} \sin(\tau \cdot \omega_i) & i < d/2 \\ \cos(\tau \cdot \omega_i) & i \geq d/2 \end{cases}$$

其中 $\omega_i = 10000^{-2i/d}$

#### 2.2.2 FiLM调制层 (Feature-wise Linear Modulation)

通过特征调制实现条件生成：

$$h_{\text{mod}} = \gamma(z) \odot h + \beta(z)$$

其中 $\gamma$ 和 $\beta$ 由调制网络生成。

#### 2.2.3 向量场网络 (Vector Field Network)

核心组件，学习从先验到目标分布的变换路径：

$$v_\theta(\tau, h, z) = \text{MLP}([\text{TimeEmb}(\tau); h], z)$$

---

## 3. 数学推导

### 3.1 流匹配基础

#### 3.1.1 连续正规化流

给定源分布 $p_0$ 和目标分布 $p_1$，通过ODE定义变换：

$$\frac{dh}{d\tau} = v_\theta(\tau, h), \quad h(0) \sim p_0, \quad h(1) \sim p_1$$

#### 3.1.2 变量变换定理

通过变换的雅可比行列式计算密度：

$$p_1(h_1) = p_0(h_0) \cdot \left| \det \frac{\partial h_0}{\partial h_1} \right|$$

对数密度：

$$\log p_1(h_1) = \log p_0(h_0) - \int_0^1 \nabla \cdot v_\theta(\tau, h(\tau)) \, d\tau$$

### 3.2 FlowSurv 训练目标

#### 3.2.1 事件样本损失

对于观察到事件的样本 $(X_i, T_i)$：

$$\mathcal{L}_{\text{event}} = \mathbb{E}_{\tau \sim U(0,1), Z_0 \sim \mathcal{N}(0,1)} \left[ \| v_\theta(\tau, h_\tau, z) - (T_i - Z_0) \|^2 \right]$$

其中 $h_\tau = (1-\tau) Z_0 + \tau T_i$（线性插值路径）

#### 3.2.2 删失样本损失

对于删失样本，使用截断采样：

$$\mathcal{L}_{\text{censored}} = \mathbb{E}_{T^* > T_{\text{obs}}} \left[ \| v_\theta(\tau, h_\tau, z) - (T^* - Z_0) \|^2 \right]$$

其中 $T^*$ 从截断指数分布采样。

### 3.3 GumbelFlowSurv 扩展 (两阶段训练框架)

#### 3.3.1 两阶段训练逻辑

为了解决流模型与先验分布预测之间的训练冲突，GumbelFlowSurv 采用了**两阶段训练框架**：

**第一阶段：Weibull AFT 预训练**
- **目标**：通过 Encoder 和一个轻量级的 Weibull-MLP (WeibullHead) 学习生存时间的基准分布。
- **损失函数**：Weibull 负对数似然 (Negative Log-Likelihood)。
- **状态**：冻结流模型相关模块 (FiLMHead, VectorField)，仅更新 Encoder 和 WeibullHead。
- **意义**：为流模型提供一个高质量的、基于协变量的初始先验分布。

**第二阶段：流匹配微调**
- **目标**：在固定的先验分布基础上，利用流模型学习更复杂的残差分布和非线性特征。
- **先验转换**：将第一阶段学到的 Weibull 参数 $(k, \lambda)$ 通过数学变换转换为 Gumbel 最小值的参数 $(\alpha, \beta)$。
- **损失函数**：仅包含流匹配损失 ($\mathcal{L}_{\text{flow}}$)，排除似然损失以避免模型对抗。
- **状态**：冻结 WeibullHead，更新 Encoder、FiLMHead 和 VectorField。

#### 3.3.2 Weibull 到 Gumbel 的参数转换

为了确保两阶段分布的一致性，通过中位数对齐和方差匹配进行参数转换：

1. **中位数对齐**：
   $$T_{\text{median}} = \lambda (\ln 2)^{1/k}$$
   $$\alpha = \text{Norm}(T_{\text{median}}) - \beta \cdot \ln(\ln 2)$$

2. **尺度匹配**：
   $$\beta = \frac{1/k}{\sigma_{\text{time\_scaler}}}$$

#### 3.3.3 数值稳定性增强

- **采样保护**：对 Gumbel 采样过程进行双重截断处理，防止极端概率导致 `inf`。
- **梯度裁剪**：对向量场输出和 ODE 积分过程进行数值限制，确保长序列求解的稳定性。

---

## 4. 推断过程

### 4.1 生存时间预测

#### 4.1.1 采样法（推荐）

```python
def predict_time(features, mode='median', n_samples=128):
    # 1. 从先验采样
    z0 = sample_prior((B * n_samples, 1))  # FlowSurv: N(0,1), GumbelFlowSurv: Gumbel(α,β)
    
    # 2. ODE积分
    z1 = odeint(vector_field, z0, τ_span=[0, 1])
    
    # 3. 转换到原始时间尺度
    t_pred = to_original_time(z1)
    
    # 4. 聚合（中位数或均值）
    return t_pred.view(B, n_samples).median(dim=1)
```

#### 4.1.2 确定性ODE法

```python
def predict_time_deterministic(features):
    # 使用先验中位数作为起点
    z0 = prior_median()  # FlowSurv: 0, GumbelFlowSurv: α - β*log(log(2))
    z1 = odeint(vector_field, z0, τ_span=[0, 1])
    return to_original_time(z1)
```

### 4.2 密度函数计算

通过逆流和变量变换定理：

```python
def compute_density(features, time_grid):
    # 1. 逆流：从目标时间回到先验
    z0, log_det_jacobian = inverse_flow(time_grid)
    
    # 2. 计算先验对数密度
    log_p0 = prior.log_prob(z0)
    
    # 3. 变量变换
    log_p1 = log_p0 - log_det_jacobian
    
    # 4. 雅可比校正（时间尺度变换）
    density = exp(log_p1) * jacobian_correction
    return density
```

### 4.3 生存函数计算

通过密度积分：

$$S(t|x) = 1 - \int_0^t f(s|x) \, ds$$

数值实现使用梯形法则：

```python
def predict_survival(features, time_grid):
    density = compute_density(features, time_grid)
    dt = diff(time_grid)
    cdf = cumsum(0.5 * (density[:, :-1] + density[:, 1:]) * dt)
    survival = 1 - cdf
    return survival
```

---

## 5. 关键实现细节

### 5.1 时间尺度变换

为提高数值稳定性，在归一化时间空间操作：

```python
def to_normalized_time(t_raw):
    t_log = log(t_raw + 1)
    return (t_log - mean) / std

def to_original_time(t_norm):
    t_log = t_norm * std + mean
    return exp(t_log) - 1
```

### 5.2 ODE求解器

支持两种求解器：

| 求解器 | 精度 | 速度 | 适用场景 |
|--------|------|------|----------|
| Euler | 低 | 快 | 快速测试 |
| RK4 | 高 | 中 | 生产环境 |

### 5.3 梯度计算

逆流需要计算向量场的散度：

```python
def inverse_flow_with_integral(t1, mod_params, steps):
    h = t1.clone()
    integral = 0
    for τ in linspace(1, 0, steps):
        h.requires_grad_(True)
        v = vector_field(τ, h, mod_params)
        div_v = grad(v.sum(), h)[0]  # 散度
        h = h + v * dτ
        integral = integral + div_v * dτ
    return h, integral
```

---

## 6. 与基线模型对比

### 6.1 密度估计能力

| 模型 | 单峰分布 | 多峰分布 | 偏态分布 | 厚尾分布 |
|------|----------|----------|----------|----------|
| Cox PH | ❌ | ❌ | ❌ | ❌ |
| Weibull AFT | ✅ | ❌ | ⚠️ | ❌ |
| DeepHit | ⚠️ | ⚠️ | ⚠️ | ⚠️ |
| **FlowSurv** | ✅ | ✅ | ✅ | ✅ |
| **GumbelFlowSurv** | ✅ | ✅ | ✅✅ | ✅ |

### 6.2 计算复杂度

| 模型 | 训练复杂度 | 推断复杂度 |
|------|------------|------------|
| Cox PH | O(n log n) | O(1) |
| DeepHit | O(n × bins) | O(bins) |
| **FlowSurv** | O(n × steps) | O(samples × steps) |

---

## 7. 使用建议

### 7.1 模型选择

- **PH场景**：Cox/DeepSurv 足够
- **NPH + 简单分布**：FlowSurv
- **NPH + 右偏分布**：GumbelFlowSurv
- **高删失场景**：GumbelFlowSurv（先验更匹配）

### 7.2 超参数调优

关键超参数：

| 参数 | 默认值 | 调优范围 |
|------|--------|----------|
| `ode_steps` | 50 | 20-100 |
| `n_samples` | 128 | 64-256 |
| `weight_gumbel` | 2.0 | 0.5-5.0 |
| `truncated_samples` | 32 | 16-64 |

### 7.3 常见问题

**Q: 训练不稳定？**
- 减小学习率
- 增加ODE步数
- 检查时间尺度变换

**Q: 密度估计不准？**
- 增加采样数
- 使用RK4求解器
- 调整先验参数初始化

**Q: 推断速度慢？**
- 减少采样数
- 使用Euler求解器
- 考虑模型蒸馏

---

## 8. 参考文献

1. Lipman, Y., et al. "Flow Matching for Generative Modeling." ICLR 2023.
2. Chen, R.T.Q., et al. "Neural Ordinary Differential Equations." NeurIPS 2018.
3. Katzman, J.L., et al. "DeepSurv: Personalized Treatment Recommender System." 2018.
4. Lee, C., et al. "DeepHit: A Deep Learning Approach to Survival Analysis." AAAI 2018.
