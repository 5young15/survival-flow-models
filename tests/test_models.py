
import sys
import os
import torch
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baselines.weibullAFT import WeibullAFT
from models.baselines.coxph import LinearCoxPH
from models.baselines.deepsurv import DeepSurv
from models.baselines.deephit import DeepHit
from models.baselines.RSF import RandomSurvivalForestWrapper
from models.flowmodel.base_flow import FlowSurv
from models.flowmodel.gumbel_flow import GumbelFlowSurv

class TestSurvivalModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(42)
        np.random.seed(42)
    
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Testing on device: {self.device}")
        
        torch.manual_seed(42)
        
        self.batch_size = 32
        self.in_dim = 10
        self.features = torch.randn(self.batch_size, self.in_dim).to(self.device)
        self.times = torch.rand(self.batch_size).to(self.device) * 10.0
        self.events = torch.randint(0, 2, (self.batch_size,)).float().to(self.device)
        self.time_grid = torch.linspace(0, 10, 50).to(self.device)
        self.times = torch.clamp(self.times, min=0.1)

    def _fit_baseline_hazard(self, model):
        model = model.to(self.device)
        model.train()
        _ = model.forward_loss(self.features, self.times, self.events)
        with torch.no_grad():
            log_haz = model.predict_risk(self.features)
            model._fit_baseline_hazard(self.times, self.events, log_haz)
        return model

    def _test_gradient_model(self, model, model_name):
        print(f"\nTesting {model_name}...")
        model = model.to(self.device)
        
        # 1. Test forward_loss
        loss, loss_dict = model.forward_loss(self.features, self.times, self.events)
        self.assertTrue(torch.is_tensor(loss), f"{model_name} loss should be a tensor")
        self.assertFalse(torch.isnan(loss).item(), f"{model_name} loss should not be NaN")
        print(f"  Forward Loss: {loss.item():.4f}")
        
        # 2. Test predict_risk
        risk = model.predict_risk(self.features)
        self.assertEqual(risk.shape, (self.batch_size,), f"{model_name} risk shape mismatch")
        
        # 3. Test predict_survival_function
        surv = model.predict_survival_function(self.features, self.time_grid)
        self.assertEqual(surv.shape, (self.batch_size, len(self.time_grid)), f"{model_name} survival shape mismatch")
        self.assertTrue((surv >= 0).all() and (surv <= 1.05).all(), f"{model_name} survival probabilities out of range [0, 1]")
        
        # 4. Test predict_time
        pred_time = model.predict_time(self.features)
        self.assertEqual(pred_time.shape, (self.batch_size,), f"{model_name} predicted time shape mismatch")
        
        # 5. Test compute_hazard_rate
        haz = model.compute_hazard_rate(self.features, self.time_grid)
        self.assertEqual(haz.shape, (self.batch_size, len(self.time_grid)), f"{model_name} hazard rate shape mismatch")
        self.assertFalse(torch.isnan(haz).any(), f"{model_name} hazard rate contains NaNs")

        print(f"  All checks passed for {model_name}")

    def test_weibull_aft(self):
        model = WeibullAFT(in_dim=self.in_dim)
        self._test_gradient_model(model, "WeibullAFT")

    def test_coxph(self):
        model = LinearCoxPH(in_dim=self.in_dim)
        model = self._fit_baseline_hazard(model)
        self._test_gradient_model(model, "LinearCoxPH")

    def test_deepsurv(self):
        model = DeepSurv(in_dim=self.in_dim)
        model = self._fit_baseline_hazard(model)
        self._test_gradient_model(model, "DeepSurv")

    def test_deephit(self):
        # DeepHit requires num_duration_bins
        config = {'num_duration_bins': 50, 'duration_index': self.time_grid.cpu().numpy()}
        model = DeepHit(in_dim=self.in_dim, config=config)
        self._test_gradient_model(model, "DeepHit")

    def test_flowsurv(self):
        model = FlowSurv(in_dim=self.in_dim)
        self._test_gradient_model(model, "FlowSurv")

    def test_gumbel_flow_surv(self):
        model = GumbelFlowSurv(in_dim=self.in_dim)
        self._test_gradient_model(model, "GumbelFlowSurv")

    def test_rsf(self):
        print("\nTesting RandomSurvivalForestWrapper...")
        try:
            from sksurv.ensemble import RandomSurvivalForest
        except ImportError:
            self.skipTest("scikit-survival not installed")

        model = RandomSurvivalForestWrapper(in_dim=self.in_dim)
        model.fit(self.features, self.times, self.events)
        
        # Test predict_risk
        risk = model.predict_risk(self.features)
        self.assertEqual(risk.shape, (self.batch_size,), "RSF risk shape mismatch")
        
        # Test predict_survival_function
        surv = model.predict_survival_function(self.features, self.time_grid)
        self.assertEqual(surv.shape, (self.batch_size, len(self.time_grid)), "RSF survival shape mismatch")
        
        # Test predict_time
        if hasattr(model, 'predict_time') and callable(getattr(model, 'predict_time', None)):
            pred_time = model.predict_time(self.features)
            self.assertEqual(pred_time.shape, (self.batch_size,), "RSF predicted time shape mismatch")
        else:
            print("  RSF predict_time not implemented, skipping")

        # Test compute_hazard_rate (default implementation)
        haz = model.compute_hazard_rate(self.features, self.time_grid)
        self.assertEqual(haz.shape, (self.batch_size, len(self.time_grid)), "RSF hazard rate shape mismatch")
        
        print("  All checks passed for RandomSurvivalForestWrapper")

if __name__ == '__main__':
    unittest.main()
