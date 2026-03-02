# 流匹配生存模型理论

本文档介绍三种流匹配生存模型：**FlowSurv**、**GumbelFlowSurv** 和 **MultiGumbelFlowSurv**，从数学推导、损失函数设计、求解方法三个维度进行详细阐述。

---

## 目录

1. [流匹配基础理论](#1-流匹配基础理论)
2. [FlowSurv：标准正态先验模型](#2-flowsurv标准正态先验模型)
3. [GumbelFlowSurv：Gumbel先验模型](#3-gumbelflowsurvgumbel先验模型)
4. [MultiGumbelFlowSurv：二维Min-Gumbel模型](#4-multigumbelflowsurv二维min-gumbel模型)
5. [三种模型对比总结](#5-三种模型对比总结)

---

## 1. 流匹配基础理论

### 1.1 连续正规化流 (CNF)

流匹配是一种基于连续正规化流的生成模型。核心思想是通过一个时间相关的向量场 $v_\tau(z)$ 将简单先验分布 $p_0(z_0)$ 变换为复杂目标分布 $p_1(z_1)$。

**ODE 形式**：
$$\frac{dz}{d\tau} = v_\tau(z), \quad \tau \in [0, 1]$$

**变量代换公式**：
$$\log p_1(z_1) = \log p_0(z_0) - \int_0^1 \nabla \cdot v_\tau(z(\tau)) \, d\tau$$

其中 $z_0$ 是通过逆流从 $z_1$ 积分得到的先验样本。

### 1.2 流匹配损失

流匹配的核心训练目标是学习向量场 $v_\theta(z, \tau, x)$，使其能够将先验样本映射到目标分布。

**最优传输路径**（线性插值）：
$$z(\tau) = (1-\tau) z_0 + \tau z_1$$

**目标向量场**：
$$v^*(z(\tau)) = \frac{dz}{d\tau} = z_1 - z_0$$

**损失函数**：
$$\mathcal{L}_{FM} = \mathbb{E}_{\tau, z_0, z_1} \left[ \| v_\theta(z(\tau), \tau, x) - (z_1 - z_0) \|^2 \right]$$

### 1.3 生存分析中的应用

在生存分析中，目标是建模条件分布 $p(t|x)$，其中 $t$ 是生存时间，$x$ 是协变量。

**关键映射**：
- 先验空间 $z_0$：简单分布（如标准正态或 Gumbel）
- 目标空间 $z_1$：归一化后的生存时间
- 条件信息 $x$：通过 FiLM 调制机制注入向量场

---

## 2. FlowSurv：标准正态先验模型

### 2.1 模型架构

FlowSurv 是最基础的流匹配生存模型，使用标准正态分布作为先验。

**先验分布**：
$$p_0(z_0) = \mathcal{N}(z_0 | 0, 1) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z_0^2}{2}\right)$$

**网络结构**：
```
输入 x → Encoder → FiLM参数 (γ, β)
              ↓
         Vector Field Network
              ↓
    v_θ(τ, z, γ, β) → 生存时间
```

### 2.2 损失函数设计

#### 2.2.1 事件样本损失

对于观测到失效事件的样本 $(t_i, x_i, \delta_i=1)$：

$$\mathcal{L}_{event} = \mathbb{E}_{\tau, z_0} \left[ \| v_\theta(z(\tau), \tau, x_i) - (t_i^{norm} - z_0) \|^2 \right]$$

其中：
- $t_i^{norm}$ 是归一化后的生存时间
- $z_0 \sim \mathcal{N}(0, 1)$ 是先验采样
- $z(\tau) = (1-\tau) z_0 + \tau t_i^{norm}$

**代码实现**：
```python
tau = torch.rand(n_event, device=device)
t0 = torch.randn_like(t1_event)  # 标准正态采样
xt = (1 - tau.unsqueeze(-1)) * t0 + tau.unsqueeze(-1) * t1_event
target_v = t1_event - t0
pred_v = self.vf_forward(tau, xt, mod_event)
event_loss = F.mse_loss(pred_v, target_v)
```

#### 2.2.2 删失样本损失

对于删失样本 $(t_i, x_i, \delta_i=0)$，真实的失效时间 $T > t_i$。采用**截断指数采样**策略：

$$\mathcal{L}_{censored} = \mathbb{E}_{\tau, z_0, \tilde{t}} \left[ \| v_\theta(z(\tau), \tau, x_i) - (\tilde{t} - z_0) \|^2 \right]$$

其中 $\tilde{t} \sim \text{TruncExp}(t_i, \lambda)$ 是从 $t_i$ 开始的截断指数分布采样。

**截断指数采样公式**：
$$\tilde{t} = t_i - \frac{\log(1-u)}{\lambda}, \quad u \sim \text{Uniform}(0, 1)$$

### 2.3 密度计算

通过逆流积分计算对数密度：

$$\log f(t|x) = \log p_0(z_0) - \int_1^0 \nabla \cdot v_\tau(z(\tau), x) \, d\tau + \log\left|\frac{dt^{norm}}{dt}\right|$$

**代码实现**：
```python
def _inverse_flow_with_integral(self, t1, mod_params, ode_steps=100):
    tau_span = torch.linspace(1.0, 0.0, ode_steps, device=t1.device)
    curr_t = t1.clone()
    integral = torch.zeros(t1.size(0), 1, device=t1.device)
    
    for i in range(len(tau_span) - 1):
        v = self.vf_forward(tau, curr_t, mod_params)
        div_v = torch.autograd.grad(v, curr_t, ...)  # 散度计算
        curr_t = curr_t + v * dt
        integral = integral + div_v * dt
    
    return curr_t, integral  # z0 和 log Jacobian
```

### 2.4 ODE 求解方法

#### 2.4.1 Euler 方法

$$z_{n+1} = z_n + \Delta\tau \cdot v_\tau(z_n)$$

**优点**：简单快速  
**缺点**：精度较低，需要更多步数

#### 2.4.2 RK4 方法

$$\begin{aligned}
k_1 &= v_\tau(z_n) \\
k_2 &= v_{\tau+\Delta\tau/2}(z_n + \frac{\Delta\tau}{2} k_1) \\
k_3 &= v_{\tau+\Delta\tau/2}(z_n + \frac{\Delta\tau}{2} k_2) \\
k_4 &= v_{\tau+\Delta\tau}(z_n + \Delta\tau \cdot k_3) \\
z_{n+1} &= z_n + \frac{\Delta\tau}{6}(k_1 + 2k_2 + 2k_3 + k_4)
\end{aligned}$$

**优点**：四阶精度，更准确  
**推荐**：默认使用 RK4

### 2.5 蒙特卡洛采样法（可选）

除了密度积分法，还可以通过蒙特卡洛采样估计生存函数。

**数学原理**：
$$S(t|x) = P(T > t|x) = \mathbb{E}_{z_0 \sim p_0}[\mathbb{I}(\text{Flow}(z_0; x) > t)]$$

**估计方法**：
$$S(t|x) \approx \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}(t_i > t)$$

其中 $t_i = \text{Flow}(z_0^{(i)}; x)$，$z_0^{(i)} \sim p_0(z_0)$。

**代码实现**：
```python
def predict_survival_function_mc(self, features, time_grid, n_samples=1000):
    z0 = self.sample_prior((B * n_samples, 1), device)  # 先验采样
    t_samples = self._forward_flow_samples(z0, mod_params)  # 正向ODE
    S = (t_samples > time_grid).float().mean(dim=2)  # 指示函数均值
    return S
```

**两种方法对比**：

| 方法 | 密度积分法 | 蒙特卡洛法 |
|------|-----------|-----------|
| **原理** | 逆流积分 + 变量代换 | 正向采样 + 统计估计 |
| **精度** | 确定性，依赖ODE步数 | 随机性，依赖采样数 |
| **计算量** | 每个时间点需积分 | 一次采样，多次使用 |
| **适用场景** | 时间点较少时 | 时间点较多时 |
| **风险函数** | 直接计算 | 数值微分，不稳定 |

---

## 3. GumbelFlowSurv：Gumbel先验模型

### 3.1 Gumbel 分布

Gumbel 分布是极值分布的一种，非常适合建模生存时间。

**概率密度函数**：
$$f(z|\alpha, \beta) = \frac{1}{\beta} \exp\left(\frac{z-\alpha}{\beta} - e^{\frac{z-\alpha}{\beta}}\right)$$

**累积分布函数**：
$$F(z|\alpha, \beta) = 1 - \exp\left(-e^{\frac{z-\alpha}{\beta}}\right)$$

**生存函数**：
$$S(z|\alpha, \beta) = \exp\left(-e^{\frac{z-\alpha}{\beta}}\right)$$

**参数含义**：
- $\alpha$：位置参数（众数）
- $\beta$：尺度参数（控制分布宽度）

### 3.2 两阶段训练策略

GumbelFlowSurv 采用两阶段训练：

#### 阶段一：Gumbel 预训练

学习 Gumbel 分布参数 $(\alpha, \beta)$，使用最大似然估计：

**事件样本**：
$$\mathcal{L}_{event}^{(1)} = -\log f(t_i | \alpha_i, \beta_i)$$

**删失样本**：
$$\mathcal{L}_{censored}^{(1)} = -\log S(t_i | \alpha_i, \beta_i)$$

**总损失**：
$$\mathcal{L}^{(1)} = -\frac{1}{N}\sum_{i=1}^N \left[ \delta_i \log f(t_i) + (1-\delta_i) \log S(t_i) \right]$$

**代码实现**：
```python
if self._stage == 'gumbel':
    log_prob = self.log_prob_prior(t1, alpha, beta)  # log f(t)
    log_surv = -torch.exp((t1 - alpha) / beta)        # log S(t)
    log_lik = events * log_prob + (1 - events) * log_surv
    return -log_lik.mean()
```

#### 阶段二：Flow 训练

固定 Gumbel 参数，训练向量场：

**先验采样**（逆变换采样）：
$$z_0 = \alpha + \beta \cdot (-\log(-\log(u))), \quad u \sim \text{Uniform}(0,1)$$

**流匹配损失**：
$$\mathcal{L}^{(2)} = \mathbb{E}_{\tau, z_0 \sim \text{Gumbel}} \left[ \| v_\theta(z(\tau), \tau, x) - (t^{norm} - z_0) \|^2 \right]$$

### 3.3 参数初始化

使用数据统计量初始化 Gumbel 参数：

$$\begin{aligned}
\beta_{init} &= \frac{\sigma \sqrt{6}}{\pi} \\
\alpha_{init} &= \mu - \beta_{init} \cdot \gamma
\end{aligned}$$

其中 $\gamma \approx 0.5772$ 是欧拉-马歇罗尼常数，$\mu$ 和 $\sigma$ 是归一化时间的均值和标准差。

### 3.4 密度计算

$$\log f(t|x) = \log p_0^{Gumbel}(z_0|\alpha, \beta) - \int_1^0 \nabla \cdot v_\tau \, d\tau$$

其中：
$$\log p_0^{Gumbel}(z_0) = -\log\beta + \frac{z_0-\alpha}{\beta} - \exp\left(\frac{z_0-\alpha}{\beta}\right)$$

---

## 4. MultiGumbelFlowSurv：二维Min-Gumbel模型

### 4.1 动机：联合建模失效时间与删失时间

传统生存分析只建模失效时间 $T$，忽略删失时间 $C$ 的信息。MultiGumbelFlowSurv 同时建模 $(T, C)$ 的联合分布。

**观测数据**：
$$Y = \min(T, C), \quad \delta = \mathbb{I}(T \leq C)$$

### 4.2 Min-Gumbel 联合分布

**联合 CDF**：
$$F_{T,C}(t, c) = \exp\left(-\left(e^{\frac{t-\mu_T}{\beta}} + e^{\frac{c-\mu_C}{\beta}}\right)\right)$$

**联合 PDF**（推导）：

对 $F(t,c)$ 求偏导：
$$\begin{aligned}
f_{T,C}(t,c) &= \frac{\partial^2 F}{\partial t \partial c} \\
&= F(t,c) \cdot \frac{e^{(t-\mu_T)/\beta}}{\beta} \cdot \frac{e^{(c-\mu_C)/\beta}}{\beta} \cdot \left(e^{(t-\mu_T)/\beta} + e^{(c-\mu_C)/\beta}\right)
\end{aligned}$$

**对数密度**：
$$\log f_{T,C}(t,c) = -\text{sum\_exp} - \text{std}_T - \log\beta - \text{std}_C - \log\beta + \log(\text{sum\_exp})$$

其中 $\text{std}_T = \frac{t-\mu_T}{\beta}$，$\text{sum\_exp} = e^{\text{std}_T} + e^{\text{std}_C}$。

**代码实现**：
```python
def log_prob_min_gumbel(self, z_T, z_C, mu_T, mu_C, beta):
    std_T = (z_T - mu_T) / beta
    std_C = (z_C - mu_C) / beta
    exp_std_T = torch.exp(std_T)
    exp_std_C = torch.exp(std_C)
    sum_exp = exp_std_T + exp_std_C
    log_F = -sum_exp
    log_f = log_F + (-std_T - log_beta) + (-std_C - log_beta) + log(sum_exp)
    return log_f
```

### 4.3 边缘分布性质

**失效时间 $T$ 的边缘分布**：
$$f_T(t) = \int_{-\infty}^{\infty} f_{T,C}(t,c) \, dc = \text{Gumbel}(t|\mu_T, \beta)$$

**删失时间 $C$ 的边缘分布**：
$$f_C(c) = \text{Gumbel}(c|\mu_C, \beta)$$

### 4.4 两阶段训练

#### 阶段一：Min-Gumbel 预训练

**事件样本** ($\delta=1, T=t, C>t$)：

方法一（边缘分解）：
$$\mathcal{L}_{event} = -\log f_T(t) - \log S_C(t)$$

方法二（联合采样）：
$$\mathcal{L}_{event} = -\mathbb{E}_{c \sim p(C|C>t)}[\log f_{T,C}(t, c)]$$

**删失样本** ($\delta=0, C=t, T>t$)：

$$\mathcal{L}_{censored} = -\mathbb{E}_{t' \sim p(T|T>t)}[\log f_{T,C}(t', t)]$$

#### 阶段二：二维流匹配

**向量场维度**：$v_\theta \in \mathbb{R}^2$，同时建模 $(z_T, z_C)$

**先验采样**：
$$z_0 = [z_T^0, z_C^0] \sim \text{Min-Gumbel}(\mu_T, \mu_C, \beta)$$

**目标构造**：

对于事件样本 ($\delta=1$)：
$$z_1 = [t^{norm}, c_{sample}]$$
其中 $c_{sample} \sim p(C|C > t^{norm})$

对于删失样本 ($\delta=0$)：
$$z_1 = [t_{sample}, t^{norm}]$$
其中 $t_{sample} \sim p(T|T > t^{norm})$

**流匹配损失**：
$$\mathcal{L}_{FM} = \mathbb{E}\left[ \| v_\theta(z(\tau), \tau, x) - (z_1 - z_0) \|^2 \right]$$

### 4.5 条件分布采样

**从 $p(C|C>t)$ 采样**：

$$c = \mu_C - \beta \cdot \log(-\log(1 - S_C(t) \cdot v))$$

其中 $v \sim \text{Uniform}(0,1)$，$S_C(t) = \exp(-e^{(t-\mu_C)/\beta})$。

**代码实现**：
```python
def sample_gumbel_truncated_above(self, t, mu, beta):
    std_t = (t - mu) / beta
    S_t = torch.exp(-torch.exp(std_t))  # 生存函数
    v = torch.rand_like(t)
    inner = 1.0 - S_t * v
    c = mu - beta * torch.log(-torch.log(inner))
    return torch.max(c, t)  # 确保 c > t
```

### 4.6 边缘密度计算

计算 $f_T(t|x)$ 时，需要对 $C$ 积分。实践中使用**中位数近似**：

$$\log f_T(t|x) \approx \log p_0^{Gumbel}(z_0^T|\mu_T, \beta) - \int_1^0 \nabla \cdot v_\tau \, d\tau$$

其中逆流时固定 $c = \text{median}(C|C>t)$。

---

## 5. 三种模型对比总结

| 特性 | FlowSurv | GumbelFlowSurv | MultiGumbelFlowSurv |
|------|----------|----------------|---------------------|
| **先验分布** | $\mathcal{N}(0,1)$ | $\text{Gumbel}(\alpha,\beta)$ | $\text{Min-Gumbel}(\mu_T,\mu_C,\beta)$ |
| **向量场维度** | 1D | 1D | 2D |
| **训练阶段** | 单阶段 | 两阶段 | 两阶段 |
| **参数数量** | 最少 | 中等 | 最多 |
| **删失建模** | 截断采样 | 截断采样 | 联合分布 |
| **计算复杂度** | 低 | 中 | 高 |
| **适用场景** | 简单数据 | 一般生存数据 | 复杂删失机制 |

### 5.1 数学公式对比

**损失函数**：

| 模型 | 事件样本损失 | 删失样本损失 |
|------|-------------|-------------|
| FlowSurv | $\|v_\theta - (t-z_0)\|^2$ | $\|v_\theta - (\tilde{t}-z_0)\|^2$ |
| GumbelFlowSurv | $\|v_\theta - (t-z_0^{Gumbel})\|^2$ | $\|v_\theta - (\tilde{t}-z_0^{Gumbel})\|^2$ |
| MultiGumbelFlowSurv | $\|v_\theta - (z_1-z_0)\|^2$ | $\|v_\theta - (z_1-z_0)\|^2$ |

**密度计算**：

| 模型 | 先验对数密度 |
|------|-------------|
| FlowSurv | $-\frac{z_0^2}{2} - \frac{1}{2}\log(2\pi)$ |
| GumbelFlowSurv | $-\log\beta + \frac{z_0-\alpha}{\beta} - e^{\frac{z_0-\alpha}{\beta}}$ |
| MultiGumbelFlowSurv | $-\log\beta + \frac{z_0^T-\mu_T}{\beta} - e^{\frac{z_0^T-\mu_T}{\beta}}$ (边缘) |

### 5.2 选择建议

1. **FlowSurv**：适合快速实验、数据量较小、删失比例较低的场景
2. **GumbelFlowSurv**：适合一般生存分析任务，平衡性能与复杂度
3. **MultiGumbelFlowSurv**：适合删失机制复杂、需要充分利用删失信息的场景

### 5.3 关键参数配置

```python
config = {
    # 网络结构
    'tau_dim': 32,              # 时间嵌入维度
    'encoder_hidden': [32, 16], # 编码器隐藏层
    'vf_hidden_dims': [16, 16], # 向量场网络隐藏层
    'film_hidden': [16],        # FiLM调制网络隐藏层
    'gumbel_head_hidden': [16], # Gumbel参数预测头 (仅GumbelFlowSurv/MultiGumbelFlowSurv)
    'dropout': 0.1,             # Dropout比例
    'sigma': 0.1,               # 时间归一化参数
    
    # 训练参数
    'LR': 8e-5,                 # 流训练学习率
    'BATCH_SIZE': 64,           # 批大小
    'EPOCHS': 200,              # 最大训练轮数
    'PATIENCE': 10,             # 早停耐心值
    'WEIGHT_DECAY': 1e-5,       # 权重衰减
    'weight_event': 1.0,        # 事件样本权重
    'weight_censored': 1.0,     # 删失样本权重
    
    # Gumbel预训练参数 (仅GumbelFlowSurv/MultiGumbelFlowSurv)
    'GUMBEL_LR': 4e-5,          # Gumbel阶段学习率
    'GUMBEL_EPOCHS': 200,       # Gumbel阶段最大轮数
    'GUMBEL_BATCH_SIZE': 64,    # Gumbel阶段批大小
    'GUMBEL_PATIENCE': 15,      # Gumbel阶段早停耐心值
    
    # ODE求解
    'ode_steps': 100,           # ODE积分步数
    'solver': 'euler',          # 求解器: 'euler' 或 'rk4'
    
    # 采样参数
    'truncated_samples': 32,    # 删失样本截断采样数
    'n_samples': 256,           # 预测时采样数
    
    # MultiGumbelFlowSurv特有
    'use_joint_loss': False,    # 是否使用联合损失
}
```

---

## 附录：FiLM 调制机制

**Feature-wise Linear Modulation (FiLM)** 是一种条件注入机制：

$$\text{FiLM}(h, z) = \gamma(z) \odot h + \beta(z)$$

其中 $\gamma$ 和 $\beta$ 由协变量 $x$ 通过神经网络生成。

**时间调制扩展**：
$$h_{mod} = (1-g) \cdot h_{film} + g \cdot (h_{film} \odot (1+\tanh(s)) + b)$$

其中 $g, s, b$ 由时间嵌入 $\tau$ 生成，实现时间相关的动态调制。
