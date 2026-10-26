// Steps through the cards of every .card-grid and the .input-item inputs of
// every .input-grid that has ‹/› buttons: .step-next reveals the next one and
// .step-prev hides the last one shown. The first one is always shown.
document.querySelectorAll(".card-grid, .input-grid").forEach((grid) => {
  const controls = grid.querySelector(":scope > .overview-controls");
  if (!controls) return;
  const prev = controls.querySelector(".step-prev");
  const next = controls.querySelector(".step-next");
  const items = [...grid.querySelectorAll(":scope > .card, :scope > .input-item")];
  let shown = 1;
  const update = () => {
    items.forEach((item, n) => item.classList.toggle("shown", n < shown));
    prev.disabled = shown <= 1;
    next.disabled = shown === items.length;
  };
  prev.addEventListener("click", () => {
    shown -= 1;
    update();
  });
  next.addEventListener("click", () => {
    shown += 1;
    update();
  });
  update();
  controls.hidden = false;
});
