// Steps through the #challenge-cards cards: #challenge-next reveals the next
// one and #challenge-prev hides the last one shown.
(() => {
  const controls = document.getElementById("challenge-controls");
  const prev = document.getElementById("challenge-prev");
  const next = document.getElementById("challenge-next");
  const cards = [...document.querySelectorAll("#challenge-cards .card")];
  let shown = 0;
  const update = () => {
    cards.forEach((card, n) => card.classList.toggle("shown", n < shown));
    prev.disabled = shown === 0;
    next.disabled = shown === cards.length;
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
})();

// Reveals the #overview-figure images one after another once #overview-start
// is clicked: each GIF plays once and stops on its last frame before the next
// image appears, and #overview-pause pauses the sequence (Start then resumes
// it). So that they can be paused, the GIFs are drawn frame by frame on a
// canvas: each frame is split out of the file as a one-frame GIF, decoded, and
// drawn over the previous ones (these GIFs' frames only cover what changed and
// never clear what's underneath). Until Start is clicked, the first GIF shows
// its first frame. If a GIF can't be fetched or decoded (e.g. the page is
// opened from file://), it's shown as an ordinary looping image instead.
(async () => {
  const imgs = [...document.querySelectorAll("#overview-figure img[data-src]")];
  const figures = imgs.map((img) => img.closest("figure"));
  const controls = document.getElementById("overview-controls");
  const start = document.getElementById("overview-start");
  const pause = document.getElementById("overview-pause");

  // Splits a GIF into its frames, each decoded with its delay in ms.
  async function splitFrames(bytes) {
    let i = 13;
    if (bytes[10] & 0x80) i += 3 * 2 ** ((bytes[10] & 7) + 1);
    const head = bytes.subarray(0, i);
    const frames = [];
    let control = null;
    const skipSubBlocks = () => {
      while (bytes[i]) i += bytes[i] + 1;
      i += 1;
    };
    while (i < bytes.length && bytes[i] !== 0x3b) {
      const begin = i;
      if (bytes[i] === 0x21) {
        const label = bytes[i + 1];
        i += 2;
        skipSubBlocks();
        if (label === 0xf9) control = bytes.subarray(begin, i);
      } else if (bytes[i] === 0x2c) {
        const flags = bytes[i + 9];
        i += 10;
        if (flags & 0x80) i += 3 * 2 ** ((flags & 7) + 1);
        i += 1;
        skipSubBlocks();
        const parts = [head, ...(control ? [control] : []), bytes.subarray(begin, i), new Uint8Array([0x3b])];
        const delay = control ? control[4] | (control[5] << 8) : 0;
        // Browsers play delays under 20 ms at 100 ms.
        frames.push({ blob: new Blob(parts, { type: "image/gif" }), delay: (delay < 2 ? 10 : delay) * 10 });
        control = null;
      } else {
        throw new Error(`Unexpected GIF block 0x${bytes[i].toString(16)}`);
      }
    }
    return Promise.all(frames.map(async ({ blob, delay }) => ({ image: await createImageBitmap(blob), delay })));
  }

  // Returns a canvas to play a GIF on, or null if it isn't a GIF or failed.
  async function load(img) {
    const src = img.dataset.src;
    if (!src.endsWith(".gif")) return null;
    try {
      const r = await fetch(src);
      if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
      const frames = await splitFrames(new Uint8Array(await r.arrayBuffer()));
      const canvas = document.createElement("canvas");
      canvas.width = frames[0].image.width;
      canvas.height = frames[0].image.height;
      canvas.setAttribute("role", "img");
      canvas.setAttribute("aria-label", img.alt);
      img.hidden = true;
      img.after(canvas);
      return { ctx: canvas.getContext("2d"), frames };
    } catch (e) {
      console.error(`Couldn't split ${src} into frames`, e);
      return null;
    }
  }

  // A clock that stops while paused.
  let paused = false;
  let resume = null;
  async function wait(ms) {
    let left = ms;
    while (left > 0) {
      if (paused) await new Promise((resolve) => (resume = resolve));
      const t = performance.now();
      await new Promise((resolve) => setTimeout(resolve, Math.min(left, 50)));
      if (!paused) left -= performance.now() - t;
    }
  }

  // Fetch everything up front so there's no gap between steps.
  const gifs = await Promise.all(imgs.map(load));

  // Show the first figure on its first frame while waiting for Start.
  if (gifs[0]) {
    gifs[0].ctx.drawImage(gifs[0].frames[0].image, 0, 0);
    figures[0].classList.add("shown");
  }

  const running = () => {
    start.disabled = true;
    pause.disabled = false;
  };
  pause.disabled = true;
  controls.hidden = false;
  await new Promise((resolve) => start.addEventListener("click", resolve, { once: true }));
  running();
  start.addEventListener("click", () => {
    paused = false;
    resume?.();
    resume = null;
    running();
  });
  pause.addEventListener("click", () => {
    paused = true;
    pause.disabled = true;
    start.disabled = false;
    start.textContent = "▶ Resume";
  });

  for (const [n, img] of imgs.entries()) {
    const figure = figures[n];
    const gif = gifs[n];
    if (!gif) {
      await new Promise((resolve) => {
        img.onload = img.onerror = resolve;
        img.src = img.dataset.src;
      });
    }
    figure.classList.add("shown");
    // An operator sign between figures appears with the figure after it, or
    // once that figure's animation has finished if it has
    // data-show="after-next".
    const prev = figure.previousElementSibling;
    const op = prev?.classList.contains("overview-operator") ? prev : null;
    const afterNext = op?.dataset.show === "after-next";
    if (op && !afterNext) op.classList.add("shown");
    if (gif) {
      for (const [k, { image, delay }] of gif.frames.entries()) {
        // The first GIF's first frame is already showing.
        if (n > 0 || k > 0) gif.ctx.drawImage(image, 0, 0);
        await wait(delay);
      }
    }
    if (afterNext) op.classList.add("shown");
  }
  start.disabled = pause.disabled = true;
  start.textContent = "\u25B6 Start";
})();
