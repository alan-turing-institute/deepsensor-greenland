// Shows the Pareto placement figures written by
// demo/results/placement_pareto_demo.py, one per result set listed in its
// manifest. A picker appears only when there is more than one and the page
// has a #pareto-select (the Overview page shows just the first figure).
(async () => {
  const img = document.getElementById("pareto-figure");
  const select = document.getElementById("pareto-select");
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

  img.src = results[0].file;
  if (select && results.length > 1) {
    for (const [i, res] of results.entries()) {
      select.add(new Option(res.label, String(i)));
    }
    select.addEventListener("change", () => {
      img.src = results[Number(select.value)].file;
    });
    select.hidden = false;
  }
})();
