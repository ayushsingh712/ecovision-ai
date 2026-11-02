/* Light/dark theme: applies the preferred palette before first paint. */
(function () {
  "use strict";

  var STORAGE_KEY = "ecovision-theme";
  var root = document.documentElement;
  var current = "dark";
  var ready = false;

  function readStored() {
    try {
      var value = window.localStorage.getItem(STORAGE_KEY);
      return value === "dark" || value === "light" ? value : null;
    } catch (error) {
      return null;
    }
  }

  function writeStored(theme) {
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch (error) {
      /* Storage can be unavailable in private or embedded contexts. */
    }
  }

  function preferredTheme() {
    return readStored() || "dark";
  }

  function syncButtons() {
    var label = current === "dark" ? "Switch to light theme" : "Switch to dark theme";
    var buttons = document.querySelectorAll("[data-theme-toggle]");
    for (var index = 0; index < buttons.length; index += 1) {
      buttons[index].setAttribute("aria-label", label);
      buttons[index].setAttribute("title", label);
    }
  }

  function apply(theme, persist) {
    current = theme === "dark" ? "dark" : "light";
    root.setAttribute("data-theme", current);
    root.style.colorScheme = current;

    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) {
      meta.setAttribute("content", current === "dark" ? "#08080b" : "#f5f4fa");
    }

    syncButtons();
    if (persist) writeStored(current);
    if (ready) {
      document.dispatchEvent(
        new CustomEvent("ecovision:themechange", { detail: { theme: current } }),
      );
    }
  }

  apply(preferredTheme(), false);

  function start() {
    var buttons = document.querySelectorAll("[data-theme-toggle]");
    for (var index = 0; index < buttons.length; index += 1) {
      buttons[index].addEventListener("click", function () {
        apply(current === "dark" ? "light" : "dark", true);
      });
    }
    syncButtons();
    ready = true;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }

  window.addEventListener("storage", function (event) {
    if (event.key === STORAGE_KEY) {
      apply(event.newValue === "dark" ? "dark" : "light", false);
    }
  });

  window.EcoVisionTheme = {
    get: function () {
      return current;
    },
    toggle: function () {
      apply(current === "dark" ? "light" : "dark", true);
    },
  };
})();
