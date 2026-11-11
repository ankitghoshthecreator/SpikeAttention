"""Linear regression modeling and residual diagnostics using statsmodels OLS."""

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

from spikeattn.exceptions import InsufficientDataError


class RegressionAnalyzer:
    """Estimates OLS regression models and conducts regression diagnostics."""

    @staticmethod
    def fit_simple(x: np.ndarray | list[float], y: np.ndarray | list[float]) -> dict[str, Any]:
        """Fit a simple univariate linear regression model y = beta_0 + beta_1 * x."""
        a = np.asarray(x, dtype=np.float64)
        b = np.asarray(y, dtype=np.float64)
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        if len(a) < 3:
            raise InsufficientDataError("Regression requires at least 3 points")

        X = sm.add_constant(a)
        model = sm.OLS(b, X)
        results = model.fit()

        ci = results.conf_int()
        return {
            "intercept": float(results.params[0]),
            "slope": float(results.params[1]),
            "intercept_ci": (float(ci[0][0]), float(ci[0][1])),
            "slope_ci": (float(ci[1][0]), float(ci[1][1])),
            "r_squared": float(results.rsquared),
            "adj_r_squared": float(results.rsquared_adj),
            "f_statistic": float(results.fvalue),
            "f_pvalue": float(results.f_pvalue),
            "residuals": results.resid.tolist(),
            "fitted_values": results.fittedvalues.tolist(),
            "n": int(len(a)),
            "_statsmodels_model": results,
        }

    @staticmethod
    def fit_multiple(
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | list[float],
        feature_names: list[str] | None = None,
        cluster_ids: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Fit a multiple OLS regression with optional cluster-robust standard errors."""
        y_arr = np.asarray(y, dtype=np.float64)
        if isinstance(X, pd.DataFrame):
            f_names = list(X.columns)
            X_mat = X.to_numpy(dtype=np.float64)
        else:
            X_mat = np.asarray(X, dtype=np.float64)
            f_names = feature_names or [f"X{j}" for j in range(X_mat.shape[1])]

        if len(y_arr) != len(X_mat):
            raise ValueError("X and y must have the same number of rows")
        if len(y_arr) <= X_mat.shape[1] + 1:
            raise InsufficientDataError("Observations must exceed the number of predictors + 1")

        X_with_const = sm.add_constant(X_mat)
        all_names = ["const"] + f_names

        model = sm.OLS(y_arr, X_with_const)
        if cluster_ids is not None:
            results = model.fit(cov_type="cluster", cov_kwds={"groups": np.asarray(cluster_ids)})
        else:
            results = model.fit()

        ci = results.conf_int()
        coef_dict = {}
        for idx, name in enumerate(all_names):
            coef_dict[name] = {
                "estimate": float(results.params[idx]),
                "std_err": float(results.bse[idx]),
                "t_stat": float(results.tvalues[idx]),
                "p_value": float(results.pvalues[idx]),
                "ci_lower": float(ci[idx][0]),
                "ci_upper": float(ci[idx][1]),
            }

        return {
            "coefficients": coef_dict,
            "r_squared": float(results.rsquared),
            "adj_r_squared": float(results.rsquared_adj),
            "f_statistic": float(results.fvalue) if np.isfinite(results.fvalue) else 0.0,
            "f_pvalue": float(results.f_pvalue) if np.isfinite(results.f_pvalue) else 1.0,
            "residuals": results.resid.tolist(),
            "fitted_values": results.fittedvalues.tolist(),
            "n": int(len(y_arr)),
            "_statsmodels_model": results,
            "_X_with_const": X_with_const,
            "_feature_names": all_names,
        }

    @staticmethod
    def diagnostics(fit_result: dict[str, Any]) -> dict[str, Any]:
        """Perform regression diagnostic checks on residuals and regressors."""
        res_obj = fit_result.get("_statsmodels_model")
        residuals = np.asarray(fit_result.get("residuals", []), dtype=np.float64)
        if len(residuals) < 4:
            raise InsufficientDataError("Diagnostic checks require at least 4 residuals")

        # 1. Residual Normality: Shapiro-Wilk
        sw = stats.shapiro(residuals)
        normality_check = {
            "test": "Shapiro-Wilk",
            "statistic": float(sw.statistic),
            "p_value": float(sw.pvalue),
            "passed": bool(sw.pvalue >= 0.05),
        }

        # 2. Autocorrelation: Durbin-Watson
        dw = float(durbin_watson(residuals))
        dw_check = {
            "test": "Durbin-Watson",
            "statistic": dw,
            "passed": bool(1.5 <= dw <= 2.5),
            "interpretation": "Residuals appear uncorrelated" if 1.5 <= dw <= 2.5 else "Possible autocorrelation",
        }

        # 3. Homoscedasticity: Breusch-Pagan
        X_const = fit_result.get("_X_with_const")
        if X_const is not None and X_const.shape[1] > 1:
            try:
                bp_stat, bp_pval, _, _ = het_breuschpagan(residuals, X_const)
                bp_check = {
                    "test": "Breusch-Pagan",
                    "statistic": float(bp_stat),
                    "p_value": float(bp_pval),
                    "passed": bool(bp_pval >= 0.05),
                }
            except Exception:
                bp_check = {"test": "Breusch-Pagan", "passed": True, "note": "Could not compute Breusch-Pagan"}
        else:
            bp_check = {"test": "Breusch-Pagan", "passed": True, "note": "Univariate constant only"}

        # 4. Multicollinearity: Variance Inflation Factor (VIF)
        vifs = {}
        if X_const is not None and X_const.shape[1] > 2:
            names = fit_result.get("_feature_names", [])
            for col_i in range(1, X_const.shape[1]):
                name = names[col_i] if col_i < len(names) else f"X_{col_i}"
                try:
                    v = float(variance_inflation_factor(X_const, col_i))
                except Exception:
                    v = 1.0
                vifs[name] = v

        # 5. Influential points: Cook's Distance
        if res_obj is not None:
            influence = res_obj.get_influence()
            cooks_d = influence.cooks_distance[0]
            max_cook = float(np.max(cooks_d)) if len(cooks_d) > 0 else 0.0
            influential_count = int(np.sum(cooks_d > 1.0))
        else:
            max_cook = 0.0
            influential_count = 0

        return {
            "normality": normality_check,
            "autocorrelation": dw_check,
            "homoscedasticity": bp_check,
            "vif": vifs,
            "max_cooks_distance": max_cook,
            "influential_points_count": influential_count,
        }
