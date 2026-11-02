import { api, mediaUrl } from "./api.js";
import { ALLOWED_TYPES, MAX_UPLOAD_MB } from "./config.js";
import {
  classColor,
  clear,
  create,
  errorMessage,
  formatPercent,
  icon,
  qs,
  setRing,
  toast,
} from "./ui.js";

const state = {
  file: null,
  previewUrl: null,
  result: null,
  submitting: false,
  feedbackSent: false,
  activeView: "original",
  explanation: "idle",
};

let explanationPromise = null;
let explanationFor = null;

const els = {
  dropzone: qs("#dropzone"),
  input: qs("#file-input"),
  previewWrap: qs("#preview-wrap"),
  previewImg: qs("#preview-img"),
  previewName: qs("#preview-name"),
  previewSize: qs("#preview-size"),
  removeBtn: qs("#remove-btn"),
  uploadError: qs("#upload-error"),
  analyzeBtn: qs("#analyze-btn"),
  analyzeLabel: qs("#analyze-label"),
  stageChip: qs("[data-model-stage]"),
  empty: qs("#result-empty"),
  loading: qs("#result-loading"),
  resultError: qs("#result-error"),
  resultErrorText: qs("#result-error-text"),
  content: qs("#result-content"),
  ringFill: qs("#ring-fill"),
  ringValue: qs("#ring-value"),
  resultClass: qs("#result-class"),
  resultStage: qs("#result-stage"),
  resultMeta: qs("#result-meta"),
  lowConfidence: qs("#low-confidence"),
  top3: qs("#top3-bars"),
  adviceBin: qs("#advice-bin"),
  adviceColor: qs("#advice-color"),
  adviceMethod: qs("#advice-method"),
  adviceSteps: qs("#advice-steps"),
  adviceExamples: qs("#advice-examples"),
  adviceImpact: qs("#advice-impact"),
  adviceCautions: qs("#advice-cautions"),
  adviceCautionsText: qs("#advice-cautions-text"),
  adviceDisclaimer: qs("#advice-disclaimer"),
  gradcamImg: qs("#gradcam-img"),
  gradcamCaption: qs("#gradcam-caption"),
  feedbackCard: qs("#feedback-card"),
  feedbackActions: qs("#feedback-actions"),
  feedbackCorrect: qs("#feedback-correct"),
  feedbackIncorrect: qs("#feedback-incorrect"),
  feedbackDetails: qs("#feedback-details"),
  feedbackClass: qs("#feedback-class"),
  feedbackComment: qs("#feedback-comment"),
  feedbackSubmit: qs("#feedback-submit"),
  feedbackError: qs("#feedback-error"),
  feedbackThanks: qs("#feedback-thanks"),
};

const VIEW_CAPTIONS = {
  original: "The normalized 512 px thumbnail that was sent through the model.",
  heatmap: "Grad-CAM heatmap: warmer colors mark regions with a stronger influence on the predicted class.",
  overlay: "Heatmap blended over the original image to show which areas drove the prediction.",
};

function humanSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function show(element, visible) {
  if (element) element.hidden = !visible;
}

function setUploadError(message) {
  if (!message) {
    show(els.uploadError, false);
    els.uploadError.textContent = "";
    return;
  }
  els.uploadError.textContent = message;
  show(els.uploadError, true);
}

function setResultError(message) {
  if (!message) {
    show(els.resultError, false);
    return;
  }
  els.resultErrorText.textContent = message;
  show(els.resultError, true);
}

function imageDimensions(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      URL.revokeObjectURL(url);
      resolve({ width: image.naturalWidth, height: image.naturalHeight });
    };
    image.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("This file could not be read as an image."));
    };
    image.src = url;
  });
}

async function acceptFile(file) {
  setUploadError("");
  if (!file) return;
  if (!ALLOWED_TYPES.includes(file.type)) {
    setUploadError("Please choose a JPEG, PNG, or WebP image.");
    return;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    setUploadError(`That image is ${humanSize(file.size)}. The limit is ${MAX_UPLOAD_MB} MB.`);
    return;
  }
  if (file.size === 0) {
    setUploadError("That file is empty.");
    return;
  }

  try {
    const { width, height } = await imageDimensions(file);
    if (width < 64 || height < 64) {
      setUploadError("Images must be at least 64×64 pixels.");
      return;
    }
  } catch (error) {
    setUploadError(error.message);
    return;
  }

  state.file = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  els.previewImg.src = state.previewUrl;
  els.previewImg.alt = `Preview of ${file.name}`;
  els.previewName.textContent = file.name;
  els.previewSize.textContent = humanSize(file.size);
  show(els.dropzone, false);
  show(els.previewWrap, true);
  els.analyzeBtn.disabled = false;
}

function clearSelection() {
  state.file = null;
  state.result = null;
  state.explanation = "idle";
  explanationPromise = null;
  explanationFor = null;
  setExplanationButtons(true);
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  els.previewImg.removeAttribute("src");
  els.input.value = "";
  show(els.previewWrap, false);
  show(els.dropzone, true);
  setUploadError("");
  els.analyzeBtn.disabled = true;
  els.analyzeLabel.textContent = "Analyze waste";
  show(els.empty, true);
  show(els.content, false);
  show(els.loading, false);
  setResultError("");
}

async function analyze() {
  if (!state.file || els.analyzeBtn.disabled) return;

  setResultError("");
  show(els.empty, false);
  show(els.content, false);
  show(els.loading, true);
  show(els.feedbackCard, false);
  els.analyzeBtn.disabled = true;
  els.analyzeLabel.textContent = "Analyzing…";

  try {
    const data = await api.predict(state.file);
    state.result = data;
    state.feedbackSent = false;
    state.activeView = "original";
    renderResult(data);
  } catch (error) {
    show(els.loading, false);
    show(els.empty, true);
    setResultError(errorMessage(error));
  } finally {
    els.analyzeBtn.disabled = false;
    els.analyzeLabel.textContent = "Analyze waste";
  }
}

function renderTop3(items) {
  clear(els.top3);
  const maxConfidence = Math.max(...items.map((item) => item.confidence), 0.01);
  for (const item of items) {
    const fill = create("div", { class: "bar-fill" });
    fill.style.background = classColor(item.label);
    const row = create(
      "div",
      { class: "bar-row" },
      create("span", { class: "bar-label", text: item.label }),
      create("div", { class: "bar-track" }, fill),
      create("span", { class: "bar-value", text: formatPercent(item.confidence) }),
    );
    els.top3.append(row);
    requestAnimationFrame(() => {
      fill.style.width = `${Math.max(6, (item.confidence / maxConfidence) * 100)}%`;
    });
  }
}

function renderAdvice(advice) {
  els.adviceBin.textContent = advice.bin_type || "—";
  els.adviceColor.textContent = advice.bin_color ? `${advice.bin_color} (typical)` : "—";
  els.adviceMethod.textContent = advice.disposal_method || "—";

  clear(els.adviceSteps);
  for (const step of advice.instructions || []) {
    els.adviceSteps.append(create("li", { text: step }));
  }

  if (advice.examples && advice.examples.length) {
    els.adviceExamples.textContent = `Examples: ${advice.examples.join(", ")}.`;
    show(els.adviceExamples, true);
  } else {
    show(els.adviceExamples, false);
  }

  els.adviceImpact.textContent = advice.environmental_impact || "";
  const cautions = (advice.cautions || []).filter(Boolean);
  if (cautions.length) {
    els.adviceCautionsText.textContent = cautions.join(" · ");
    show(els.adviceCautions, true);
  } else {
    show(els.adviceCautions, false);
  }
  els.adviceDisclaimer.textContent = advice.disclaimer || "";
}

function setExplanationButtons(enabled) {
  document.querySelectorAll("[data-view]").forEach((button) => {
    if (button.dataset.view !== "original") {
      button.disabled = !enabled;
    }
  });
}

async function ensureExplanation() {
  const result = state.result;
  if (!result) return false;
  if (state.explanation === "ready") return true;
  if (state.explanation === "unavailable") return false;
  if (explanationPromise && explanationFor === result) {
    return explanationPromise;
  }

  explanationFor = result;
  state.explanation = "pending";
  setExplanationButtons(false);
  els.gradcamCaption.textContent = "Generating Grad-CAM explanation…";

  explanationPromise = (async () => {
    try {
      const info = await api.explain(result.prediction_id);
      if (state.result !== result) return false;
      if (info.status === "ready") {
        state.explanation = "ready";
        setExplanationButtons(true);
        return true;
      }
      state.explanation = "unavailable";
      setExplanationButtons(false);
      els.gradcamCaption.textContent = info.status === "disabled"
        ? "Grad-CAM explanations are disabled on this deployment."
        : (info.message || "The explanation could not be generated. The classification result is still valid.");
      return false;
    } catch {
      if (state.result === result) {
        state.explanation = "unavailable";
        setExplanationButtons(false);
        els.gradcamCaption.textContent = "The explanation could not be generated. The classification result is still valid.";
      }
      return false;
    } finally {
      if (explanationFor === result) {
        explanationPromise = null;
        explanationFor = null;
      }
    }
  })();

  return explanationPromise;
}

async function setGradcamView(view) {
  if (!state.result) return;
  if (view !== "original") {
    const ready = await ensureExplanation();
    if (!ready || !state.result) return;
  }
  state.activeView = view;
  const urls = {
    original: state.result.gradcam.original_url,
    heatmap: state.result.gradcam.heatmap_url,
    overlay: state.result.gradcam.overlay_url,
  };
  const url = mediaUrl(urls[view]);
  els.gradcamImg.src = url;
  els.gradcamImg.alt = view === "original"
    ? `Original image predicted as ${state.result.predicted_class}`
    : `${view === "heatmap" ? "Grad-CAM heatmap" : "Grad-CAM overlay"} for ${state.result.predicted_class}`;
  els.gradcamCaption.textContent = VIEW_CAPTIONS[view];
  document.querySelectorAll("[data-view]").forEach((button) => {
    button.setAttribute("aria-selected", String(button.dataset.view === view));
  });
  const panel = qs("#gradcam-panel");
  const tab = document.getElementById(`tab-${view}`);
  if (panel && tab) panel.setAttribute("aria-labelledby", tab.id);
}

function resetFeedback(predictedClass) {
  state.feedbackSent = false;
  state.submitting = false;
  show(els.feedbackCard, true);
  show(els.feedbackActions, true);
  show(els.feedbackDetails, false);
  show(els.feedbackThanks, false);
  els.feedbackError.textContent = "";
  els.feedbackComment.value = "";
  const alternative = Array.from(els.feedbackClass.options).find(
    (option) => option.value !== predictedClass,
  );
  if (alternative) els.feedbackClass.value = alternative.value;
}

function renderResult(data) {
  show(els.loading, false);
  show(els.content, true);
  setResultError("");

  state.explanation = "idle";
  explanationPromise = null;
  explanationFor = null;
  setExplanationButtons(true);

  const isFinal = data.model_stage === "final";
  els.resultStage.textContent = isFinal ? "Final model" : "Development checkpoint";
  els.resultStage.className = isFinal ? "badge" : "badge badge-warn";
  els.resultClass.textContent = data.predicted_class;
  els.resultMeta.textContent = `${formatPercent(data.confidence)} confidence · ${data.filename}`;
  setRing(els.ringFill, els.ringValue, data.confidence);

  show(els.lowConfidence, Boolean(data.low_confidence));
  renderTop3(data.top3 || []);
  renderAdvice(data.advice || {});
  setGradcamView("original");
  resetFeedback(data.predicted_class);

  if (window.matchMedia("(max-width: 1020px)").matches) {
    qs(".result-card").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

async function sendFeedback({ isCorrect, correctedClass = null, comment = null }) {
  if (!state.result || state.feedbackSent || state.submitting) return;
  if (!isCorrect && correctedClass === state.result.predicted_class) {
    els.feedbackError.textContent = "Choose a different category than the predicted one.";
    return;
  }

  state.submitting = true;
  els.feedbackSubmit.disabled = true;
  els.feedbackError.textContent = "";

  try {
    await api.feedback({
      prediction_id: state.result.prediction_id,
      is_correct: isCorrect,
      corrected_class: isCorrect ? null : correctedClass,
      comment: comment || null,
    });
    state.feedbackSent = true;
    show(els.feedbackActions, false);
    show(els.feedbackDetails, false);
    show(els.feedbackThanks, true);
    toast("Feedback saved as unverified.");
  } catch (error) {
    els.feedbackError.textContent = errorMessage(error);
  } finally {
    state.submitting = false;
    els.feedbackSubmit.disabled = false;
  }
}

async function loadModelStage() {
  try {
    const info = await api.modelInfo();
    if (!info.model_loaded) {
      els.stageChip.textContent = "No model installed";
      els.stageChip.className = "badge badge-danger";
      return;
    }
    els.stageChip.textContent = info.is_final
      ? `Final model · ${info.model_version}`
      : `Development model · ${info.model_stage}`;
    els.stageChip.className = info.is_final ? "badge" : "badge badge-warn";
  } catch {
    els.stageChip.textContent = "Model status unknown";
    els.stageChip.className = "badge badge-neutral";
  }
}

function bindEvents() {
  els.input.addEventListener("change", (event) => {
    const [file] = event.target.files || [];
    acceptFile(file);
  });

  ["dragenter", "dragover"].forEach((type) => {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.add("is-active");
    });
  });

  ["dragleave", "drop"].forEach((type) => {
    els.dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      els.dropzone.classList.remove("is-active");
    });
  });

  els.dropzone.addEventListener("drop", (event) => {
    const [file] = event.dataTransfer.files || [];
    acceptFile(file);
  });

  els.dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      els.input.click();
    }
  });

  els.removeBtn.addEventListener("click", clearSelection);
  els.analyzeBtn.addEventListener("click", analyze);

  document.querySelectorAll("[data-view]").forEach((button) => {
    button.addEventListener("click", () => setGradcamView(button.dataset.view));
  });

  els.feedbackCorrect.addEventListener("click", () => {
    sendFeedback({ isCorrect: true });
  });

  els.feedbackIncorrect.addEventListener("click", () => {
    show(els.feedbackActions, false);
    show(els.feedbackDetails, true);
    els.feedbackClass.focus();
  });

  els.feedbackSubmit.addEventListener("click", () => {
    sendFeedback({
      isCorrect: false,
      correctedClass: els.feedbackClass.value,
      comment: els.feedbackComment.value.trim(),
    });
  });
}

els.gradcamImg.addEventListener("error", () => {
  if (state.explanation === "pending") return;
  els.gradcamCaption.textContent = "The explanation image could not be loaded.";
});

bindEvents();
loadModelStage();
