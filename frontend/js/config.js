// Runtime configuration. Set window.ECOVISION_CONFIG = { apiBase: "https://api.example.com" }
// before this module loads to point the static frontend at a separately deployed API.

const runtime = window.ECOVISION_CONFIG || {};

export const API_BASE = String(runtime.apiBase || "").replace(/\/+$/, "");
export const API_ROOT = `${API_BASE}/api`;

export const CLASS_ORDER = [
  "Plastic",
  "Paper",
  "Cardboard",
  "Glass",
  "Metal",
  "Organic",
  "General",
];

export const CLASS_META = {
  Plastic: { color: "#2f7fbf", icon: "bottle" },
  Paper: { color: "#c9a227", icon: "paper" },
  Cardboard: { color: "#a9713d", icon: "box" },
  Glass: { color: "#4fa3a5", icon: "glass" },
  Metal: { color: "#7b8794", icon: "can" },
  Organic: { color: "#4c9a4a", icon: "leaf" },
  General: { color: "#8a6f9e", icon: "trash" },
};

export const MAX_UPLOAD_MB = 8;
export const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
export const LOW_CONFIDENCE = 0.6;
