import torch
import numpy as np
from typing import Optional, Union
from models.interface import TorchSurvivalModel

try:
    from sksurv.ensemble import RandomSurvivalForest
    SKSURV_AVAILABLE = True
except ImportError:
    SKSURV_AVAILABLE = False
    RandomSurvivalForest = None


class RandomSurvivalForestWrapper(TorchSurvivalModel):
    def __init__(self, in_dim: int, config: Optional[dict] = None, **kwargs):
        super().__init__()
        if not SKSURV_AVAILABLE:
            raise ImportError("scikit-survival is required for RSF. Install with: pip install scikit-survival")
        self.config = config or {}
        self.model = RandomSurvivalForest(
            n_estimators=self.config.get('n_estimators', 100),
            max_depth=self.config.get('max_depth', None),
            min_samples_split=self.config.get('min_samples_split', 6),
            min_samples_leaf=self.config.get('min_samples_leaf', 3),
            random_state=self.config.get('random_state', 42),
            n_jobs=self.config.get('n_jobs', -1)
        )
        self.is_fitted = False

    def forward_loss(self, features, times, events, **kwargs):
        raise NotImplementedError("RSF uses .fit() instead of forward_loss")

    def fit(self, features: Union[torch.Tensor, np.ndarray],
            times: Union[torch.Tensor, np.ndarray],
            events: Union[torch.Tensor, np.ndarray]) -> None:
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if isinstance(times, torch.Tensor):
            times = times.detach().cpu().numpy()
        if isinstance(events, torch.Tensor):
            events = events.detach().cpu().numpy()
        y = np.empty(len(events), dtype=[('event', bool), ('time', float)])
        y['event'] = events.astype(bool)
        y['time'] = times.astype(float)
        self.model.fit(features, y)
        self.is_fitted = True

    def predict_risk(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        device = features.device
        features_np = features.detach().cpu().numpy()
        risk_scores = self.model.predict(features_np)
        return torch.from_numpy(risk_scores).to(device=device, dtype=torch.float32)

    def predict_survival_function(self, features: torch.Tensor, time_grid: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        device = features.device
        features_np = features.detach().cpu().numpy()
        surv_funcs = self.model.predict_survival_function(features_np)
        if time_grid is None:
            time_grid_np = np.linspace(0, 10, 100)
        else:
            time_grid = time_grid.to(device) if isinstance(time_grid, torch.Tensor) else torch.tensor(time_grid, device=device)
            time_grid_np = time_grid.cpu().numpy()
        S = np.zeros((features_np.shape[0], len(time_grid_np)))
        for i, fn in enumerate(surv_funcs):
            S[i] = fn(time_grid_np)
        return torch.from_numpy(S).to(device=device, dtype=torch.float32)

    def predict_time(self, features: torch.Tensor, **kwargs) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        device = features.device
        features_np = features.detach().cpu().numpy()
        surv_funcs = self.model.predict_survival_function(features_np)
        medians = []
        for fn in surv_funcs:
            idx = np.searchsorted(-fn.y, -0.5)
            median = fn.x[idx] if idx < len(fn.x) else (fn.x[-1] if len(fn.x) > 0 else np.nan)
            medians.append(median)
        return torch.tensor(medians, device=device, dtype=torch.float32)