(() => {
  const root = document.documentElement;
  const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  root.classList.add('is-ready');
  const reveal = (element) => element.classList.add('is-revealed');
  const targets = [...document.querySelectorAll('.journey-method, .journey-workspace__frame')];
  if (reduced || !('IntersectionObserver' in window)) {
    targets.forEach(reveal);
  } else {
    const observer = new IntersectionObserver((entries, current) => {
      entries.forEach((entry) => { if (entry.isIntersecting) { reveal(entry.target); current.unobserve(entry.target); } });
    }, { threshold: .18 });
    targets.forEach((target) => observer.observe(target));
  }
  document.querySelectorAll('[data-scroll-workspace]').forEach((link) => link.addEventListener('click', (event) => {
    const target = document.querySelector(link.getAttribute('href'));
    if (!target) return;
    event.preventDefault();
    target.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
  }));
})();
