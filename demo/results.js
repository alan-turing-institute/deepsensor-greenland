// Shows the Pareto placement figures written by
// demo/results/placement_pareto_demo.py, one per result set listed in its
// manifest. Two pickers, "Cost" (the figure's x-axis) and "Informativeness"
// (its acquisition function, the y-axis), choose which figure is shown; each
// appears only when it has more than one option. If several runs share a
// cost and acquisition function, the first in the manifest is shown.
(async () => {
  const img = document.getElementById("pareto-figure");
  const controls = document.getElementById("pareto-controls");
  const costSelect = document.getElementById("pareto-cost");
  const acqSelect = document.getElementById("pareto-acq");
  const missing = document.getElementById("pareto-missing");
  let results;
  try {
    const r = await fetch("images/pareto/manifest.json");
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    results = await r.json();
  } catch (e) {
    console.error("Pareto manifest failed to load", e);
    return;
  }
  if (!results.length) return;

  // Fill a picker with the distinct values of `key`, in manifest order.
  const fill = (select, key, labelKey) => {
    const seen = new Set();
    for (const res of results) {
      if (seen.has(res[key])) continue;
      seen.add(res[key]);
      select.add(new Option(res[labelKey], res[key]));
    }
    select.closest("label").hidden = seen.size < 2;
    return seen.size;
  };

  const show = () => {
    const res = results.find(
      (r) => r.cost === costSelect.value && r.acq_fn === acqSelect.value,
    );
    img.hidden = !res;
    missing.hidden = Boolean(res);
    if (res) img.src = res.file;
  };

  const nCost = fill(costSelect, "cost", "cost_label");
  const nAcq = fill(acqSelect, "acq_fn", "acq_fn_label");
  costSelect.value = results[0].cost;
  acqSelect.value = results[0].acq_fn;
  costSelect.addEventListener("change", show);
  acqSelect.addEventListener("change", show);
  controls.hidden = nCost < 2 && nAcq < 2;
  show();
})();
