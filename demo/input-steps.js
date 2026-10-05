// Steps through the .input-item inputs of every .input-grid that has ‹/›
// buttons: .step-next reveals the next one and .step-prev hides the last one
// shown.
document.querySelectorAll(".input-grid").forEach((grid) => {
  const controls = grid.querySelector(":scope > .overview-controls");
  if (!controls) return;
  const prev = controls.querySelector(".step-prev");
  const next = controls.querySelector(".step-next");
  const items = [...grid.querySelectorAll(":scope > .input-item")];
  let shown = 0;
  const update = () => {
    items.forEach((item, n) => item.classList.toggle("shown", n < shown));
    prev.disabled = shown === 0;
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
