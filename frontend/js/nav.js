import { api } from "./api.js";
import { qs } from "./ui.js";

function initMobileNav() {
  const toggle = qs("[data-nav-toggle]");
  const nav = qs("[data-nav]");
  if (!toggle || !nav) return;

  const setOpen = (open) => {
    nav.classList.toggle("is-open", open);
    document.body.classList.toggle("nav-open", open);
    toggle.setAttribute("aria-expanded", String(open));
  };

  toggle.addEventListener("click", () => {
    setOpen(!nav.classList.contains("is-open"));
  });

  nav.addEventListener("click", (event) => {
    if (event.target.closest("a")) setOpen(false);
  });

  const backdrop = qs("[data-nav-backdrop]");
  if (backdrop) backdrop.addEventListener("click", () => setOpen(false));

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && nav.classList.contains("is-open")) {
      setOpen(false);
      toggle.focus();
    }
  });
}

function markActivePage() {
  const page = document.body.dataset.page;
  if (!page) return;
  document.querySelectorAll("[data-nav] a[data-nav-key]").forEach((link) => {
    if (link.dataset.navKey === page) link.setAttribute("aria-current", "page");
  });
}

function setFooterYear() {
  document.querySelectorAll("[data-year]").forEach((node) => {
    node.textContent = String(new Date().getFullYear());
  });
}

async function initModelBanner() {
  const banner = qs("[data-model-banner]");
  if (!banner) return;
  try {
    const info = await api.modelInfo();
    if (info.model_loaded && info.is_final) {
      banner.hidden = true;
      return;
    }
    const label = info.model_loaded
      ? `Development checkpoint in use (${info.model_stage}). Predictions are for testing the pipeline, not final model quality.`
      : "No model checkpoint is installed yet, so image analysis is disabled. See the deployment guide to install one.";
    banner.querySelector("[data-banner-text]").textContent = label;
    banner.querySelector("[data-banner-link]").hidden = false;
    banner.hidden = false;
    document.body.classList.add("has-model-banner");
  } catch {
    banner.hidden = true;
  }
}

function init() {
  initMobileNav();
  markActivePage();
  setFooterYear();
  initModelBanner();
}

init();
