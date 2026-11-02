import { api } from "./api.js";
import {
  clear,
  create,
  formatDate,
  formatNumber,
  formatPercent,
  qs,
  qsa,
} from "./ui.js";

const SOURCE_TO_TARGET = {
  Plastic: ["Plastic"],
  Paper: ["Paper"],
  Cardboard: ["Cardboard"],
  Glass: ["Glass"],
  Metal: ["Metal"],
  Organic: ["Food Organics", "Vegetation"],
  General: ["Miscellaneous Trash", "Textile Trash"],
};

function setText(selector, value) {
  qsa(selector).forEach((node) => {
    node.textContent = value;
  });
}

function setStage(info) {
  const label = qs("[data-model-stage-label]");
  if (!info.model_loaded) {
    label.textContent = "Model status: no checkpoint installed";
    label.className = "badge badge-danger";
  } else if (info.is_final) {
    label.textContent = "Model status: final checkpoint";
    label.className = "badge";
  } else {
    label.textContent = `Model status: development (${info.model_stage})`;
    label.className = "badge badge-warn";
  }
  setText("[data-model-version]", info.model_version ? `Version ${info.model_version}` : "Version —");
  setText("[data-device]", info.device ? `Device ${info.device.toUpperCase()}` : "Device —");
}

function setArchitectureParams(info) {
  const catalog = info.architectures || [];
  const cnn = catalog.find((item) => item.name === "custom_cnn");
  const mobile = catalog.find((item) => item.name === "mobilenet_v2");
  setText(
    "[data-params-cnn]",
    cnn ? `${formatNumber(cnn.parameters)} parameters` : "Parameters unavailable",
  );
  setText(
    "[data-params-mobile]",
    mobile ? `${formatNumber(mobile.parameters)} parameters` : "Parameters unavailable",
  );
}

function setDatasetCounts(dataset) {
  const sourceCounts = dataset.source_classes || {};
  for (const [name, count] of Object.entries(sourceCounts)) {
    setText(`[data-source-count="${name}"]`, formatNumber(count));
  }
  for (const [target, sources] of Object.entries(SOURCE_TO_TARGET)) {
    const total = sources.reduce((sum, name) => sum + (sourceCounts[name] || 0), 0);
    setText(`[data-target-count="${target}"]`, formatNumber(total));
  }
}

function metricRow(entry) {
  const latency =
    typeof entry.cpu_latency_ms === "number" ? `${entry.cpu_latency_ms.toFixed(1)} ms` : "—";
  return create(
    "tr",
    {},
    create("td", { text: entry.model || entry.architecture || "—" }),
    create("td", { class: "num", text: formatPercent(entry.accuracy) }),
    create("td", { class: "num", text: formatPercent(entry.macro_f1) }),
    create("td", { class: "num", text: formatPercent(entry.weighted_f1) }),
    create("td", { class: "num", text: formatNumber(entry.parameters) }),
    create("td", { class: "num", text: latency }),
  );
}

function rowsFromMetrics(metrics) {
  if (Array.isArray(metrics.comparison) && metrics.comparison.length) {
    return metrics.comparison;
  }
  const models = metrics.models || {};
  return Object.entries(models).map(([arch, entry]) => ({
    architecture: arch,
    model: arch === "mobilenet_v2" ? "MobileNetV2" : "Custom CNN",
    accuracy: entry.metrics ? entry.metrics.accuracy : null,
    macro_f1: entry.metrics ? entry.metrics.macro_f1 : null,
    weighted_f1: entry.metrics ? entry.metrics.weighted_f1 : null,
    parameters: entry.parameters,
    cpu_latency_ms: entry.latency ? entry.latency.mean_ms : null,
  }));
}

function renderMetrics(info) {
  const status = qs("#metrics-status");
  const tableWrap = qs("#metrics-table-wrap");
  const body = qs("#metrics-body");
  const note = qs("#metrics-note");
  clear(status);
  clear(body);

  if (!info.metrics) {
    status.append(
      create("h3", { text: "Evaluation results are pending" }),
      create("p", {
        text: info.model_loaded
          ? "A checkpoint is installed, but no evaluation artifact was found for it. Run the evaluation step and install the resulting metrics before citing numbers."
          : "No checkpoint is installed yet, so there is nothing to evaluate. Train the models and install the artifacts to populate this section.",
      }),
    );
    tableWrap.hidden = true;
    return;
  }

  const metrics = info.metrics;
  const testImages = metrics.test_split ? metrics.test_split.images : null;
  const isFinal = Boolean(info.is_final);

  status.append(
    create("h3", { text: isFinal ? "Final evaluation installed" : "Development smoke evaluation" }),
    create("p", {
      text: isFinal
        ? `Held-out test split${testImages ? ` of ${formatNumber(testImages)} images` : ""}, evaluated ${formatDate(metrics.evaluated_at)}.`
        : "These numbers come from models trained on a small stratified subset to validate the pipeline. They are not final model results and should not be compared with published benchmarks.",
    }),
  );
  if (!isFinal) {
    status.append(create("span", { class: "badge badge-warn", text: "Not final" }));
  }

  const rows = rowsFromMetrics(metrics);
  if (!rows.length) {
    status.append(create("p", { text: "The metrics file contains no per-model results." }));
    tableWrap.hidden = true;
    return;
  }
  for (const entry of rows) body.append(metricRow(entry));

  const pieces = [];
  if (metrics.notes) pieces.push(metrics.notes);
  if (testImages) pieces.push(`Test split: ${formatNumber(testImages)} images.`);
  if (metrics.test_split && metrics.test_split.split_hash) {
    pieces.push(`Split hash ${String(metrics.test_split.split_hash).slice(0, 12)}…`);
  }
  note.textContent = pieces.join(" ");
  tableWrap.hidden = false;
}

function setLimitations(info) {
  const list = qs("#limitations-list");
  if (!info.limitations || !info.limitations.length) return;
  clear(list);
  for (const item of info.limitations) {
    list.append(create("li", { text: item }));
  }
}

function setAdviceVersion(info) {
  setText(
    "[data-advice-version]",
    info.advice_version ? `${info.advice_version} (reviewed ${info.advice_last_reviewed})` : "unknown",
  );
}

async function init() {
  try {
    const info = await api.modelInfo();
    setStage(info);
    setArchitectureParams(info);
    setDatasetCounts(info.dataset || {});
    renderMetrics(info);
    setLimitations(info);
    setAdviceVersion(info);
  } catch (error) {
    const label = qs("[data-model-stage-label]");
    label.textContent = "Model status: API unavailable";
    label.className = "badge badge-danger";
    const status = qs("#metrics-status");
    clear(status);
    status.append(
      create("h3", { text: "Could not load model information" }),
      create("p", { text: error.message }),
    );
  }
}

init();
