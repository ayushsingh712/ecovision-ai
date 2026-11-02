import { CLASS_META } from "./config.js";

const SVG_NS = "http://www.w3.org/2000/svg";

export function qs(selector, root = document) {
  return root.querySelector(selector);
}

export function qsa(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

export function create(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key === "dataset") Object.assign(node.dataset, value);
    else if (key === "attrs") {
      for (const [name, attrValue] of Object.entries(value)) {
        node.setAttribute(name, attrValue);
      }
    } else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else {
      node.setAttribute(key, value);
    }
  }
  append(node, children);
  return node;
}

function append(node, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

export function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

export function icon(name, className = "") {
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("aria-hidden", "true");
  svg.setAttribute("focusable", "false");
  if (className) svg.setAttribute("class", className);
  const use = document.createElementNS(SVG_NS, "use");
  use.setAttribute("href", `assets/icons.svg#icon-${name}`);
  svg.append(use);
  return svg;
}

export function formatPercent(value, digits = 1) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatNumber(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "—";
  return new Intl.NumberFormat().format(value);
}

export function formatDate(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

export function classColor(label) {
  return (CLASS_META[label] && CLASS_META[label].color) || "#1f6f43";
}

export function classIcon(label) {
  return (CLASS_META[label] && CLASS_META[label].icon) || "trash";
}

export function classBadge(label, { withIcon = true } = {}) {
  const badge = create("span", { class: "chip" });
  if (withIcon) {
    badge.append(icon(classIcon(label)));
    const dot = create("span", { class: "dot" });
    dot.style.background = classColor(label);
    badge.prepend(dot);
  }
  badge.append(document.createTextNode(label));
  return badge;
}

export function ringDashoffset(confidence) {
  const circumference = 2 * Math.PI * 54;
  const clamped = Math.max(0, Math.min(1, Number(confidence) || 0));
  return circumference * (1 - clamped);
}

export function setRing(fillElement, valueElement, confidence) {
  if (!fillElement) return;
  fillElement.style.strokeDashoffset = String(ringDashoffset(confidence));
  if (valueElement) valueElement.textContent = formatPercent(confidence, 0);
}

export function toast(message, { error = false } = {}) {
  let element = document.querySelector(".toast");
  if (!element) {
    element = create("div", { class: "toast", role: "status", "aria-live": "polite" });
    document.body.append(element);
  }
  element.textContent = message;
  element.classList.toggle("toast-error", Boolean(error));
  element.classList.add("is-visible");
  window.clearTimeout(toast._timer);
  toast._timer = window.setTimeout(() => element.classList.remove("is-visible"), 4200);
}

export function errorMessage(error) {
  if (!error) return "Something went wrong.";
  if (error.message) return error.message;
  return String(error);
}

export function setButtonLoading(button, loading, loadingLabel = "Working…") {
  if (!button) return;
  if (loading) {
    if (!button.dataset.idleLabel) {
      button.dataset.idleLabel = button.textContent.trim();
    }
    button.disabled = true;
    clear(button);
    button.append(create("span", { class: "spinner", attrs: { "aria-hidden": "true" } }));
    button.append(document.createTextNode(loadingLabel));
  } else {
    button.disabled = false;
    button.textContent = button.dataset.idleLabel || button.textContent;
  }
}

export function emptyState({ title, message, iconName = "image" }) {
  return create(
    "div",
    { class: "empty-state" },
    create("div", { class: "empty-icon" }, icon(iconName)),
    create("strong", { text: title }),
    create("span", { class: "small", text: message }),
  );
}
