import numpy as np
import xarray as xr

from ds_greenland.utils.model import add_uncertainty_decomposition


class TestAddUncertaintyDecomposition:
    def test_derives_epistemic_and_aleatoric_std(self):
        std = xr.DataArray(np.array([2.0, 4.0]), dims="point")
        epistemic_frac = xr.DataArray(np.array([0.25, 0.5]), dims="point")
        pred = {"var": xr.Dataset({"std": std, "epistemic_frac": epistemic_frac})}

        add_uncertainty_decomposition(pred, "var")

        np.testing.assert_allclose(
            pred["var"]["epistemic_std"].values, std.values * np.sqrt([0.25, 0.5])
        )
        np.testing.assert_allclose(
            pred["var"]["aleatoric_std"].values, std.values * np.sqrt([0.75, 0.5])
        )

    def test_epistemic_and_aleatoric_variance_recombine_to_total(self):
        std = xr.DataArray(np.array([3.0]), dims="point")
        epistemic_frac = xr.DataArray(np.array([0.4]), dims="point")
        pred = {"var": xr.Dataset({"std": std, "epistemic_frac": epistemic_frac})}

        add_uncertainty_decomposition(pred, "var")

        total_var = (
            pred["var"]["epistemic_std"] ** 2 + pred["var"]["aleatoric_std"] ** 2
        )
        np.testing.assert_allclose(total_var.values, (std**2).values)
