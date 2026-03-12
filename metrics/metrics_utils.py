from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


def _safe_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def c_index_score(time, event, risk) -> float:
    time = _safe_numpy(time).reshape(-1)
    event = _safe_numpy(event).reshape(-1).astype(bool)
    risk = _safe_numpy(risk).reshape(-1)
    try:
        from sksurv.metrics import concordance_index_censored

        return float(concordance_index_censored(event, time, risk)[0])
    except Exception:
        try:
            from lifelines.utils import concordance_index

            return float(concordance_index(time, -risk, event))
        except Exception:
            return _c_index_fallback(time, event.astype(np.float32), risk)


def _c_index_fallback(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    n = len(time)
    concordant = 0.0
    comparable = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            ti, tj = time[i], time[j]
            ei, ej = event[i], event[j]
            ri, rj = risk[i], risk[j]
            if ei == 1 and ti < tj:
                comparable += 1
                concordant += 1 if ri > rj else 0.5 if ri == rj else 0
            elif ej == 1 and tj < ti:
                comparable += 1
                concordant += 1 if rj > ri else 0.5 if ri == rj else 0
    if comparable == 0:
        return 0.5
    return float(concordant / comparable)


def _simple_brier_at_t(time, event, survival_prob, t):
    time = _safe_numpy(time).reshape(-1)
    event = _safe_numpy(event).reshape(-1).astype(np.int32)
    s = _safe_numpy(survival_prob).reshape(-1)
    y = (time > t).astype(np.float32)
    return float(np.mean((y - s) ** 2))


def _to_structured(time, event):
    try:
        from sksurv.util import Surv
        return Surv.from_arrays(event=_safe_numpy(event).astype(bool), time=_safe_numpy(time))
    except ImportError:
        return None


def dynamic_auc_score(
    train_time,
    train_event,
    test_time,
    test_event,
    survival_matrix,
    eval_times: Iterable[float],
) -> tuple[float, Optional[np.ndarray]]:
    """
    计算累积动态 AUC (Cumulative Dynamic AUC)
    使用 scikit-survival 的 cumulative_dynamic_auc
    """
    try:
        from sksurv.metrics import cumulative_dynamic_auc
    except ImportError:
        return float("nan"), None

    y_train = _to_structured(train_time, train_event)
    y_test = _to_structured(test_time, test_event)
    
    if y_train is None or y_test is None:
        return float("nan"), None

    # Risk score = 1 - Survival Probability
    # survival_matrix shape: (n_samples, n_times)
    risk_matrix = 1.0 - _safe_numpy(survival_matrix)
    
    eval_times = np.asarray(eval_times)
    
    # 确保 eval_times 在训练和测试数据的范围内
    # cumulative_dynamic_auc 要求评估时间点必须在测试集的事件时间范围内
    # 并且要能通过 IPCW 估计 (通常要在训练集时间范围内)
    
    min_t = max(train_time.min(), test_time.min())
    max_t = min(train_time.max(), test_time.max()) # IPCW 需要在最大随访时间内
    
    valid_mask = (eval_times > min_t) & (eval_times < max_t)
    full_aucs = np.full(len(eval_times), np.nan)
    
    if not np.any(valid_mask):
        return float("nan"), full_aucs
        
    valid_times = eval_times[valid_mask]
    valid_risk = risk_matrix[:, valid_mask]
    
    try:
        aucs, mean_auc = cumulative_dynamic_auc(y_train, y_test, valid_risk, valid_times)
        full_aucs[valid_mask] = aucs
        return float(mean_auc), full_aucs
    except Exception as e:
        # print(f"Warning: cumulative_dynamic_auc failed: {e}")
        return float("nan"), full_aucs


def dynamic_c_index_score(
    train_time,
    train_event,
    test_time,
    test_event,
    survival_matrix,
    eval_times: Iterable[float],
) -> tuple[float, Optional[np.ndarray]]:
    """
    计算动态 C-index (Time-dependent Concordance Index, IPCW)
    使用 scikit-survival 的 concordance_index_ipcw
    """
    try:
        from sksurv.metrics import concordance_index_ipcw
    except ImportError:
        return float("nan"), None

    y_train = _to_structured(train_time, train_event)
    y_test = _to_structured(test_time, test_event)

    if y_train is None or y_test is None:
        return float("nan"), None

    risk_matrix = 1.0 - _safe_numpy(survival_matrix)
    eval_times = np.asarray(eval_times)
    
    c_indices = []
    
    # 同样需要范围检查，但这里我们可以逐个处理
    min_t = train_time.min() # IPCW 训练集支持
    max_t = train_time.max() 
    
    for i, t in enumerate(eval_times):
        # 检查时间 t 是否在有效范围内 (必须在训练集范围内以计算 censoring 分布)
        # 且在测试集中有意义
        if t <= min_t or t >= max_t:
            c_indices.append(float("nan"))
            continue
            
        try:
            # concordance_index_ipcw 返回 (cindex, concordant, discordant, tied_risk, tied_time)
            # 我们只取第一个 cindex
            # risk_matrix[:, i] 是在时间 t 的累积风险估计
            c_index = concordance_index_ipcw(y_train, y_test, risk_matrix[:, i], tau=t)[0]
            c_indices.append(c_index)
        except Exception:
            c_indices.append(float("nan"))
            
    c_indices = np.array(c_indices)
    # 计算平均值时忽略 nan
    if np.all(np.isnan(c_indices)):
        return float("nan"), c_indices
    return float(np.nanmean(c_indices)), c_indices


def ibs_score(
    train_time,
    train_event,
    test_time,
    test_event,
    survival_matrix,
    eval_times: Iterable[float],
) -> float:
    eval_times = np.asarray(list(eval_times), dtype=np.float64)
    surv = _safe_numpy(survival_matrix)
    if surv.shape[1] != len(eval_times):
        raise ValueError("survival_matrix列数必须与eval_times长度一致。")
    try:
        from sksurv.metrics import integrated_brier_score
        from sksurv.util import Surv

        y_train = Surv.from_arrays(event=_safe_numpy(train_event).astype(bool), time=_safe_numpy(train_time))
        y_test = Surv.from_arrays(event=_safe_numpy(test_event).astype(bool), time=_safe_numpy(test_time))
        return float(integrated_brier_score(y_train, y_test, surv, eval_times))
    except Exception:
        scores = []
        for i, t in enumerate(eval_times):
            scores.append(_simple_brier_at_t(test_time, test_event, surv[:, i], t))
        return float(np.trapz(np.asarray(scores), eval_times) / max(eval_times[-1] - eval_times[0], 1e-8))


def evaluate_all_metrics(
    train_time,
    train_event,
    test_time,
    test_event,
    risk,
    survival_matrix,
    eval_times,
) -> Dict[str, float]:
    metrics = {
        "c_index": c_index_score(test_time, test_event, risk),
        "ibs": ibs_score(train_time, train_event, test_time, test_event, survival_matrix, eval_times),
    }
    
    # 动态指标
    dyn_auc_mean, _ = dynamic_auc_score(train_time, train_event, test_time, test_event, survival_matrix, eval_times)
    metrics["dynamic_auc"] = dyn_auc_mean
    
    dyn_c_mean, _ = dynamic_c_index_score(train_time, train_event, test_time, test_event, survival_matrix, eval_times)
    metrics["dynamic_c_index"] = dyn_c_mean
    
    return metrics
