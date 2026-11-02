import { API_BASE, API_ROOT } from "./config.js";

export class ApiError extends Error {
  constructor(message, { code = "REQUEST_FAILED", status = 0, requestId = null, detail = null } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
    this.requestId = requestId;
    this.detail = detail;
  }
}

async function request(path, { method = "GET", body = null, headers = {}, timeout = 20000 } = {}) {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeout);
  try {
    const response = await fetch(`${API_ROOT}${path}`, {
      method,
      body,
      headers,
      signal: controller.signal,
      mode: "cors",
    });
    const text = await response.text();
    let data = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
    if (!response.ok) {
      const error = (data && data.error) || {};
      throw new ApiError(error.message || `Request failed with status ${response.status}.`, {
        code: error.code || "REQUEST_FAILED",
        status: response.status,
        requestId: error.request_id || response.headers.get("X-Request-ID"),
        detail: error.detail || null,
      });
    }
    return data;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error.name === "AbortError") {
      throw new ApiError("The request timed out. Please try again.", { code: "TIMEOUT" });
    }
    throw new ApiError(
      "Could not reach the EcoVision API. Make sure the backend is running and the API URL is correct.",
      { code: "NETWORK_ERROR" },
    );
  } finally {
    window.clearTimeout(timer);
  }
}

export const api = {
  health: () => request("/health", { timeout: 8000 }),
  modelInfo: () => request("/model-info"),
  statistics: () => request("/statistics"),
  history: (limit = 20, offset = 0) => request(`/history?limit=${limit}&offset=${offset}`),
  predict(file) {
    const form = new FormData();
    form.append("image", file, file.name || "waste-image.jpg");
    return request("/predict", { method: "POST", body: form, timeout: 45000 });
  },
  explain(predictionId) {
    return request(`/predictions/${predictionId}/explanation`, {
      method: "POST",
      timeout: 45000,
    });
  },
  feedback(payload) {
    return request("/feedback", {
      method: "POST",
      body: JSON.stringify(payload),
      headers: { "Content-Type": "application/json" },
      timeout: 12000,
    });
  },
};

export function mediaUrl(path) {
  if (!path || typeof path !== "string") return "";
  if (!path.startsWith("/media/")) return "";
  return `${API_BASE}${path}`;
}
