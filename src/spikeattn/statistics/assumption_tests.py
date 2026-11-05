"""Statistical assumption testing: normality, homoscedasticity, outliers, and sphericity."""

from typing import Any

import numpy as np
from scipy import stats

from spikeattn.exceptions import InsufficientDataError


class AssumptionTester:
    """Verifies assumptions underlying parametric and non-parametric inferences."""

    @staticmethod
    def shapiro_wilk(x: np.ndarray | list[float], alpha: float = 0.05) -> dict[str, Any]:
        """Test the null hypothesis of composite normality via Shapiro-Wilk.

        Note: A non-significant result (p >= alpha) indicates 'no evidence against
        normality', NOT 'proof that the distribution is normal'.
        """
        arr = np.asarray(x, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        n = len(arr)
        if n < 3:
            raise InsufficientDataError("Shapiro-Wilk test requires at least 3 observations")

        stat, p_val = stats.shapiro(arr)
        reject = bool(p_val < alpha)

        interpretation = (
            f"Evidence against normality (p={p_val:.4e} < {alpha})"
            if reject
            else f"No evidence against normality at alpha={alpha} (p={p_val:.4e}). Note: does not prove normality."
        )

        return {
            "test": "Shapiro-Wilk",
            "statistic": float(stat),
            "p_value": float(p_val),
            "alpha": float(alpha),
            "reject_normality": reject,
            "passed": not reject,
            "n": n,
            "interpretation": interpretation,
        }

    @staticmethod
    def levene(*groups: np.ndarray | list[float], center: str = "median", alpha: float = 0.05) -> dict[str, Any]:
        """Test for equality of variances across groups (Brown-Forsythe Levene test)."""
        valid_groups = []
        for g in groups:
            arr = np.asarray(g, dtype=np.float64)
            arr = arr[np.isfinite(arr)]
            if len(arr) < 2:
                raise InsufficientDataError("Each group in Levene test must have at least 2 observations")
            valid_groups.append(arr)

        if len(valid_groups) < 2:
            raise InsufficientDataError("Levene test requires at least two groups")

        stat, p_val = stats.levene(*valid_groups, center=center)
        reject = bool(p_val < alpha)

        interpretation = (
            f"Heteroscedasticity detected (p={p_val:.4e} < {alpha})"
            if reject
            else f"No evidence against homoscedasticity at alpha={alpha} (p={p_val:.4e})"
        )

        return {
            "test": f"Levene ({center})",
            "statistic": float(stat),
            "p_value": float(p_val),
            "alpha": float(alpha),
            "reject_equal_variance": reject,
            "passed": not reject,
            "n_groups": len(valid_groups),
            "interpretation": interpretation,
        }

    @staticmethod
    def outliers(x: np.ndarray | list[float], method: str = "iqr", threshold: float = 1.5) -> dict[str, Any]:
        """Identify potential outliers via the Tukey IQR rule or robust z-scores."""
        arr = np.asarray(x, dtype=np.float64)
        n = len(arr)
        if n < 4:
            return {"has_outliers": False, "count": 0, "indices": [], "bounds": None, "method": method}

        if method == "iqr":
            q1, q3 = np.percentile(arr, [25, 75])
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            outlier_mask = (arr < lower) | (arr > upper)
            bounds = (float(lower), float(upper))
        elif method == "robust_z":
            med = np.median(arr)
            mad = np.median(np.abs(arr - med))
            scale = 1.4826 * mad if mad > 1e-12 else 1.0
            z_scores = np.abs(arr - med) / scale
            outlier_mask = z_scores > threshold
            bounds = (float(med - threshold * scale), float(med + threshold * scale))
        else:
            raise ValueError(f"Unknown outlier detection method: '{method}'")

        indices = np.where(outlier_mask)[0].tolist()
        return {
            "has_outliers": bool(len(indices) > 0),
            "count": int(len(indices)),
            "indices": indices,
            "bounds": bounds,
            "method": method,
            "passed": len(indices) == 0,
        }

    @staticmethod
    def qq_data(x: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
        """Compute theoretical and sample quantiles for Q-Q plotting."""
        arr = np.asarray(x, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 3:
            raise InsufficientDataError("Q-Q analysis requires at least 3 points")
        (osm, osr), _ = stats.probplot(arr, dist="norm")
        return osm, osr

    @staticmethod
    def sphericity(matrix: np.ndarray, alpha: float = 0.05) -> dict[str, Any]:
        """Assess sphericity across repeated measures (subjects x treatments).

        Computes criterion approximation and Greenhouse-Geisser epsilon.
        """
        mat = np.asarray(matrix, dtype=np.float64)
        n_subjects, k_treatments = mat.shape
        if k_treatments < 3:
            return {
                "test": "Sphericity W Test",
                "epsilon": 1.0,
                "passed": True,
                "note": "Sphericity is trivially satisfied for k < 3 treatments",
            }
        if n_subjects <= k_treatments:
            return {
                "test": "Sphericity W Test",
                "epsilon": 1.0,
                "passed": True,
                "note": "Insufficient subjects relative to treatments to formally test sphericity",
            }

        # Sample covariance matrix across treatments
        cov = np.cov(mat, rowvar=False)
        # Transformation matrix for orthonormal differences
        m = k_treatments - 1
        c_mat = np.zeros((k_treatments, m))
        for j in range(m):
            c_mat[:j + 1, j] = 1.0 / (j + 1)
            c_mat[j + 1, j] = -1.0
            c_mat[:, j] /= np.linalg.norm(c_mat[:, j])

        trans_cov = c_mat.T @ cov @ c_mat
        eigenvals = np.linalg.eigvalsh(trans_cov)
        eigenvals = eigenvals[eigenvals > 1e-12]
        if len(eigenvals) < m:
            epsilon = 1.0 / m
            return {
                "test": "Sphericity W Test",
                "epsilon": float(epsilon),
                "passed": False,
                "note": "Singular covariance across contrasts",
            }

        # Greenhouse-Geisser epsilon
        sum_e = np.sum(eigenvals)
        sum_e2 = np.sum(eigenvals ** 2)
        epsilon = float((sum_e ** 2) / (m * sum_e2))
        epsilon = max(1.0 / m, min(1.0, epsilon))

        # W statistic
        w_stat = float(np.prod(eigenvals) / ((sum_e / m) ** m))

        return {
            "test": "Sphericity W Test",
            "w_statistic": w_stat,
            "greenhouse_geisser_epsilon": epsilon,
            "passed": epsilon > 0.75,
            "alpha": alpha,
            "note": "Use Greenhouse-Geisser correction if epsilon < 0.75",
        }
