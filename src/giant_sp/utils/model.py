import xarray as xr

from giant_sp.utils.training import batched

def compute_pred_ds(
    model, tasks, truth_ds, varname, pred_params, mask=None, res_factor=1
):
    pred = model.predict(
        tasks,
        X_t=truth_ds[varname],
        X_t_mask=mask,
        resolution_factor=res_factor,
        pred_params=pred_params,
    )
    mean_ds = pred[varname]["mean"].assign_coords(
        {"latitude": truth_ds.lat, "longitude": truth_ds.lon}
    )
    cutout = pred[varname].init_time if pred.forecasting_mode else pred[varname].time
    mean_ds, truth_ds = xr.align(
        mean_ds, truth_ds[varname].sel(time=cutout), join="inner"
    )
    err_ds = mean_ds - truth_ds
    return pred, err_ds


def get_pred_DAs_taskstream(
    model, task_stream, batch_size, truth_ds, varname, mask=None
):
    mean_arrays = []
    std_arrays = []
    for tasks in batched(task_stream, batch_size):
        pred = model.predict(list(tasks), X_t=truth_ds, X_t_mask=mask)
        batch_mean_da = pred[varname]["mean"]
        batch_std_da = pred[varname]["std"]
        mean_arrays.append(batch_mean_da)
        std_arrays.append(batch_std_da)
    dim = "init_time" if pred.forecasting_mode else "time"
    mean_da = xr.concat(mean_arrays, dim=dim)
    std_da = xr.concat(std_arrays, dim=dim)
    mean_da = mean_da.assign_coords(
        {"latitude": truth_ds.lat, "longitude": truth_ds.lon}
    )
    mean_da, truth_ds = xr.align(mean_da, truth_ds[varname], join="inner")
    err_da = mean_da - truth_ds

    return mean_da, std_da, err_da
