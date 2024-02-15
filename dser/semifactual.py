import warnings
import numpy as np
from scipy.optimize import minimize

from utils import non_zero_threshold, non_zero_threshold_sparsity


class SemifactualExplanation():
    def __init__(self, reject_option, C_simple=.1, C_reg=1., C_diversity=1., C_feasibility=1., C_sf=1., sparsity_upper_bound=2., solver="Nelder-Mead", max_iter=None, **kwds):
        self.reject_option = reject_option
        self.solver = solver
        self.max_iter = max_iter
        self.C_simple = C_simple
        self.C_reg = C_reg
        self.C_diversity = C_diversity
        self.C_feasibility = C_feasibility
        self.C_sf = C_sf
        self.sparsity_upper_bound = sparsity_upper_bound

        super().__init__(**kwds)

    def __compute_semifactual(self, x_orig, features_blacklist=[]):
        # Loss function
        low_complexity_expl_loss = lambda x: self.C_simple * max([np.sum(np.abs(x[i] - x_orig[i]) > non_zero_threshold_sparsity for i in range(x_orig.shape[0])) - self.sparsity_upper_bound, 0])
        similarity_orig_loss = lambda x: -1. * self.C_reg * np.linalg.norm(x - x_orig, 2)
        feasibility_loss = lambda x: self.C_feasibility * max([self.reject_option.criterion(x) - self.reject_option.threshold, 0])
        sf_loss = lambda x: self.C_sf * max([self.reject_option.criterion(x_orig) - self.reject_option.criterion(x), 0])
        diversity_loss = lambda x: self.C_diversity * np.sum([np.abs(x[idx] - x_orig[idx]) > non_zero_threshold for idx in features_blacklist])
        
        loss = lambda x: low_complexity_expl_loss(x) + similarity_orig_loss(x) + feasibility_loss(x) + sf_loss(x) + diversity_loss(x)

        # Minimize loss function
        res = minimize(fun=loss, x0=x_orig, method=self.solver, options={'maxiter': self.max_iter})
        x_sf = res["x"]

        return x_sf

    def _compute_diverse_semifactual(self, x_orig, X_sf):
        features_blacklist = [] # Diversity: Already used features must not be used again
        for x_sf in X_sf:
            delta_cf = np.abs(x_orig - x_sf)
            features_blacklist += [int(idx) for idx in np.argwhere(delta_cf > non_zero_threshold)]
        features_blacklist = list(set(features_blacklist))

        return self.__compute_semifactual(x_orig, features_blacklist)

    def compute_diverse_explanations(self, x_orig, n_explanations=3):
        X_sf = []
        
        # First semifactual can be computed using the standard approach
        X_sf.append(self.compute_explanation(x_orig))

        # Compute more & diverse counterfactuals
        for _ in range(n_explanations - 1):
            x_sf = self._compute_diverse_semifactual(x_orig, X_sf)
            if x_sf is None:
                break

            X_sf.append(x_sf)

        return X_sf

    def compute_explanation(self, x_orig):
        return self.__compute_semifactual(x_orig)
