(function () {
  const root = document.documentElement;
  const params = new URLSearchParams(window.location.search);
  const feather = params.get('feather') || 'medium';
  root.dataset.feather = feather;
  const featherLabel = document.querySelector('#feather-label');
  if (featherLabel) featherLabel.textContent = feather === 'soft' ? '轻' : (feather === 'deep' ? '深' : '中');

  const slides = [
    'assets/background pics/D97F95BD-8267-4F40-ADB6-02410FEE299E.PNG',
    'assets/background pics/258428F2-9DAF-40F4-94C5-45F647045C46.PNG',
    'assets/background pics/IMG_1389.JPG'
  ];
  const art = document.querySelector('[data-hero-art]');
  const imageLayers = art ? Array.from(art.querySelectorAll('img')) : [];
  const dots = document.querySelectorAll('[data-slide]');
  let slideIndex = 0;
  let activeLayer = 0;
  let transitioning = false;
  let pendingIndex = null;
  let timer;

  function showSlide(index) {
    if (!art || imageLayers.length < 2) return;
    const nextIndex = (index + slides.length) % slides.length;
    if (nextIndex === slideIndex && !transitioning) return;
    if (transitioning) {
      pendingIndex = nextIndex;
      return;
    }

    const incoming = imageLayers[1 - activeLayer];
    const outgoing = imageLayers[activeLayer];
    const preload = new Image();
    transitioning = true;
    slideIndex = nextIndex;
    incoming.classList.add('is-preparing');

    preload.onload = function () {
      incoming.src = slides[slideIndex];
      incoming.alt = '敦煌莫高窟水彩示意图';
      dots.forEach(function (dot, i) { dot.classList.toggle('active', i === slideIndex); });
      window.requestAnimationFrame(function () {
        window.requestAnimationFrame(function () {
          incoming.classList.remove('is-preparing');
          incoming.classList.add('is-active');
          outgoing.classList.remove('is-active');
        });
      });
      window.setTimeout(function () {
        activeLayer = 1 - activeLayer;
        transitioning = false;
        if (pendingIndex !== null && pendingIndex !== slideIndex) {
          const queued = pendingIndex;
          pendingIndex = null;
          showSlide(queued);
        } else {
          pendingIndex = null;
        }
      }, 1750);
    };
    preload.onerror = function () {
      incoming.classList.remove('is-preparing');
      transitioning = false;
      pendingIndex = null;
    };
    preload.decoding = 'async';
    preload.src = slides[slideIndex];
  }

  function autoplay() {
    window.clearInterval(timer);
    timer = window.setInterval(function () { showSlide(slideIndex + 1); }, 5000);
  }

  dots.forEach(function (dot, i) {
    dot.addEventListener('click', function () { showSlide(i); autoplay(); });
  });
  if (art) autoplay();

  document.querySelectorAll('[data-flow]').forEach(function (card) {
    card.addEventListener('click', function () {
      document.querySelectorAll('[data-flow]').forEach(function (item) { item.classList.remove('active'); });
      card.classList.add('active');
    });
  });

  document.querySelectorAll('[data-tool-action]').forEach(function (button) {
    button.addEventListener('click', function () {
      const message = document.querySelector('[data-tool-message]');
      if (message) message.textContent = button.getAttribute('data-tool-action') + ' · 预览交互已触发';
    });
  });

  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const revealItems = document.querySelectorAll('[data-reveal], [data-line-draw]');
  if (revealItems.length) {
    root.classList.add('js-ready');
    const reveal = function (item) { item.classList.add('is-revealed'); };
    if (reduceMotion || !('IntersectionObserver' in window)) {
      revealItems.forEach(reveal);
    } else {
      const observer = new IntersectionObserver(function (entries, activeObserver) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          reveal(entry.target);
          activeObserver.unobserve(entry.target);
        });
      }, { threshold: .16, rootMargin: '0px 0px -8% 0px' });
      revealItems.forEach(function (item) { observer.observe(item); });
    }
  }

}());
