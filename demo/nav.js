const PAGES = [
  { id: "overview", href: "index.html", label: "Project overview", sub: "What GIANT-SP does" },
  { id: "data", href: "data.html", label: "Data", sub: "Explore the dataset" },
  { id: "model", href: "model.html", label: "Model", sub: "Architecture &amp; placement" },
];

class SiteShell extends HTMLElement {
  connectedCallback() {
    const active = this.getAttribute("active");
    const content = Array.from(this.childNodes);

    const navItems = PAGES.map(
      (page) => `
        <a class="nav-item${page.id === active ? " active" : ""}" href="${page.href}">
          <span class="label">${page.label}</span>
          <span class="sub">${page.sub}</span>
        </a>`
    ).join("");

    const currentIndex = PAGES.findIndex((page) => page.id === active);
    const next = PAGES[(currentIndex + 1) % PAGES.length];

    this.innerHTML = `
      <header>
        <img class="header-logo" src="logos/giant.png" alt="GIANT" />
        <div class="header-title">
          <h1>GIANT Sensor Placement — Demo Dashboard</h1>
          <p>Explore the dataset and view model outputs</p>
        </div>
        <a class="next-btn" href="${next.href}">Next: ${next.label} →</a>
        <div class="partner-logos">
          <img class="header-logo" src="logos/turing_white.png" alt="The Alan Turing Institute" />
          <img class="header-logo" src="logos/aria_white.png" alt="ARIA" />
        </div>
      </header>
      <div class="layout">
        <nav class="sidebar">${navItems}</nav>
        <main></main>
      </div>`;

    const main = this.querySelector("main");
    content.forEach((node) => main.appendChild(node));
  }
}

customElements.define("site-shell", SiteShell);
