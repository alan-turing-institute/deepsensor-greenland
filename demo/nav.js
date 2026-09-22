const PAGES = [
  { id: "overview", href: "index.html", label: "Project overview", sub: "What GIANT-SP does" },
  { id: "data", href: "data-overview.html", label: "Data overview", sub: "Explore the dataset" },
  { id: "models", href: "model-details.html", label: "Model details", sub: "Architecture &amp; placement" },
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

    this.innerHTML = `
      <header>
        <h1>GIANT Sensor Placement — Demo Dashboard</h1>
        <p>Explore the dataset and view model outputs</p>
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
