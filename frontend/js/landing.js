const targets = document.querySelectorAll("[data-reveal]");

if (targets.length && "IntersectionObserver" in window && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
  document.documentElement.classList.add("js-reveal");
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { rootMargin: "0px 0px -8% 0px", threshold: 0.08 },
  );
  targets.forEach((target) => observer.observe(target));
  window.setTimeout(() => {
    targets.forEach((target) => target.classList.add("is-visible"));
  }, 3000);
} else {
  targets.forEach((target) => target.classList.add("is-visible"));
}
