import { api, mediaUrl } from "./api.js";
import { CLASS_ORDER } from "./config.js";
import {
  classColor,
  clear,
  create,
  errorMessage,
  formatDate,
  formatNumber,
  formatPercent,
  icon,
  qs,
} from "./ui.js";

const LIMIT = 10;
const state = {
  stats: null,
  history: [],
  total: 0,
  offset: 0,
  charts: [],
  loading: false,
};

const els = {
  total: qs("#kpi-total"),
  confidence: qs("#kpi-confidence"),
  common: qs("#kpi-common"),
  commonNote: qs("#kpi-common-note"),
  feedback: qs("#kpi-feedback"),
  feedbackNote: qs("#kpi-feedback-note"),
  distributionCanvas: qs("#chart-distribution"),
  dailyCanvas: qs("#chart-daily"),
  legend: qs("#distribution-legend"),
  body: qs("#history-body"),
  summary: qs("#history-summary"),
  empty: qs("#history-empty"),
  tableWrap: qs(".table-wrap"),
  loadMore: qs("#load-more"),
  error: qs("#dashboard-error"),
  errorText: qs("#dashboard-error-text"),
  refresh: qs("#refresh-btn"),
};

function show(element, visible) {
  if (element) element.hidden = !visible;
}

function chartPlaceholder(canvas, message) {
  const box = canvas.parentElement;
  canvas.hidden = true;
  let placeholder = box.querySelector("[data-chart-empty]");
  if (!placeholder) {
    placeholder = create(
      "div",
      { class: "empty-state", dataset: { chartEmpty: "true" } },
      create("div", { class: "empty-icon" }, icon("chart")),
      create("strong", { text: "Nothing to chart yet" }),
      create("span", { class: "small", text: message }),
    );
    box.append(placeholder);
  }
  placeholder.hidden = false;
}

function clearChartPlaceholder(canvas) {
  const box = canvas.parentElement;
  const placeholder = box.querySelector("[data-chart-empty]");
  if (placeholder) placeholder.hidden = true;
  canvas.hidden = false;
}

function destroyCharts() {
  for (const chart of state.charts) chart.destroy();
  state.charts = [];
}

function renderStats(stats) {
  els.total.textContent = formatNumber(stats.total_predictions);
  els.confidence.textContent =
    stats.average_confidence === null ? "—" : formatPercent(stats.average_confidence);

  if (stats.most_common_class) {
    els.common.textContent = stats.most_common_class;
    const entry = stats.class_distribution.find(
      (item) => item.label === stats.most_common_class,
    );
    els.commonNote.textContent = entry
      ? `${formatNumber(entry.count)} predictions · ${entry.percentage.toFixed(1)}%`
      : "";
  } else {
    els.common.textContent = "—";
    els.commonNote.textContent = "no predictions yet";
  }

  const feedback = stats.feedback || {};
  if (feedback.total > 0) {
    els.feedback.textContent = formatPercent(feedback.user_reported_accuracy, 0);
    els.feedbackNote.textContent = `${feedback.correct} of ${feedback.total} marked correct`;
  } else {
    els.feedback.textContent = "—";
    els.feedbackNote.textContent = "no feedback submitted";
  }

  renderLegend(stats.class_distribution || []);
  renderCharts(stats);
}

function renderLegend(distribution) {
  clear(els.legend);
  for (const label of CLASS_ORDER) {
    const entry = distribution.find((item) => item.label === label) || {
      label,
      count: 0,
      percentage: 0,
    };
    const dot = create("span", { class: "dot" });
    dot.style.background = classColor(label);
    els.legend.append(
      create(
        "li",
        {},
        dot,
        document.createTextNode(label),
        create("strong", { text: formatNumber(entry.count) }),
      ),
    );
  }
}

function themeColors() {
  const styles = getComputedStyle(document.documentElement);
  const read = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
  return {
    text: read("--text-muted", "#5b6e64"),
    grid: read("--chart-grid", "rgba(18, 74, 53, 0.08)"),
    bar: read("--chart-bar", "#3f9e63"),
    barHover: read("--chart-bar-hover", "#1f6f43"),
  };
}

function renderCharts(stats) {
  if (typeof window.Chart === "undefined") {
    chartPlaceholder(els.distributionCanvas, "Chart.js failed to load.");
    chartPlaceholder(els.dailyCanvas, "Chart.js failed to load.");
    return;
  }

  destroyCharts();
  const colors = themeColors();
  window.Chart.defaults.font.family = '"Inter", system-ui, sans-serif';
  window.Chart.defaults.color = colors.text;

  const total = stats.total_predictions || 0;
  if (total > 0) {
    clearChartPlaceholder(els.distributionCanvas);
    const distribution = stats.class_distribution || [];
    state.charts.push(
      new window.Chart(els.distributionCanvas, {
        type: "doughnut",
        data: {
          labels: distribution.map((item) => item.label),
          datasets: [
            {
              data: distribution.map((item) => item.count),
              backgroundColor: distribution.map((item) => classColor(item.label)),
              borderWidth: 0,
              hoverOffset: 6,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          cutout: "62%",
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                label(context) {
                  const value = context.parsed;
                  const share = total ? ((value / total) * 100).toFixed(1) : "0.0";
                  return ` ${context.label}: ${value} (${share}%)`;
                },
              },
            },
          },
        },
      }),
    );
  } else {
    chartPlaceholder(els.distributionCanvas, "Analyze an image to see the class split.");
  }

  const daily = stats.daily || [];
  const dailyTotal = daily.reduce((sum, item) => sum + item.count, 0);
  if (dailyTotal > 0) {
    clearChartPlaceholder(els.dailyCanvas);
    state.charts.push(
      new window.Chart(els.dailyCanvas, {
        type: "bar",
        data: {
          labels: daily.map((item) => {
            const date = new Date(`${item.date}T00:00:00Z`);
            return new Intl.DateTimeFormat(undefined, {
              month: "short",
              day: "numeric",
              timeZone: "UTC",
            }).format(date);
          }),
          datasets: [
            {
              label: "Analyses",
              data: daily.map((item) => item.count),
              backgroundColor: colors.bar,
              hoverBackgroundColor: colors.barHover,
              borderRadius: 8,
              maxBarThickness: 38,
            },
          ],
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: {
              beginAtZero: true,
              ticks: { precision: 0 },
              grid: { color: colors.grid },
            },
            x: { grid: { display: false } },
          },
          plugins: { legend: { display: false } },
        },
      }),
    );
  } else {
    chartPlaceholder(els.dailyCanvas, "No analyses in the last seven days.");
  }
}

function feedbackCell(item) {
  if (!item.feedback) {
    return create("span", { class: "badge badge-neutral", text: "None" });
  }
  if (item.feedback.is_correct) {
    return create("span", { class: "badge", text: "Correct" });
  }
  return create(
    "span",
    { class: "badge badge-warn", text: `Wrong → ${item.feedback.corrected_class || "?"}` },
  );
}

function historyRow(item) {
  const thumb = create("img", {
    class: "thumb",
    src: mediaUrl(item.image_url),
    alt: `Thumbnail for ${item.filename}`,
    loading: "lazy",
    width: "46",
    height: "46",
  });

  const category = create("span", { class: "chip" });
  const dot = create("span", { class: "dot" });
  dot.style.background = classColor(item.predicted_class);
  category.append(dot, document.createTextNode(item.predicted_class));

  return create(
    "tr",
    {},
    create("td", {}, thumb),
    create("td", {}, category),
    create("td", { class: "num", text: formatPercent(item.confidence) }),
    create("td", {}, feedbackCell(item)),
    create("td", { class: "small muted", text: formatDate(item.created_at) }),
  );
}

function renderHistory() {
  clear(els.body);
  for (const item of state.history) {
    els.body.append(historyRow(item));
  }
  const hasItems = state.history.length > 0;
  show(els.empty, !hasItems);
  show(els.tableWrap, hasItems);
  els.summary.textContent = hasItems
    ? `Showing ${state.history.length} of ${formatNumber(state.total)} predictions`
    : "No predictions yet";
  show(els.loadMore, state.history.length < state.total);
}

async function load({ reset = true } = {}) {
  if (state.loading) return;
  state.loading = true;
  els.refresh.disabled = true;
  show(els.error, false);

  if (reset) {
    state.offset = 0;
    state.history = [];
  }

  try {
    const [stats, history] = await Promise.all([
      api.statistics(),
      api.history(LIMIT, state.offset),
    ]);
    state.stats = stats;
    state.total = history.total;
    state.history = reset ? history.items : state.history.concat(history.items);
    state.offset += history.items.length;
    renderStats(stats);
    renderHistory();
  } catch (error) {
    els.errorText.textContent = errorMessage(error);
    show(els.error, true);
  } finally {
    state.loading = false;
    els.refresh.disabled = false;
  }
}

els.refresh.addEventListener("click", () => load({ reset: true }));
els.loadMore.addEventListener("click", () => load({ reset: false }));

document.addEventListener("ecovision:themechange", () => {
  if (state.stats) renderCharts(state.stats);
});

load({ reset: true });
