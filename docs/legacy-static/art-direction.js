/* Art-direction layer for the selected Cave Library Conservation Desk concept. */

document.addEventListener("DOMContentLoaded", () => {
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const stage = document.getElementById("conservationStage");
  const readout = document.getElementById("deskReadout");
  const controls = [...document.querySelectorAll(".desk-control")];
  const skip = document.getElementById("skipIntro");
  const timers = [];
  let introAnimations = [];

  const schedule = (fn, delay) => {
    const timer = window.setTimeout(fn, delay);
    timers.push(timer);
    return timer;
  };

  const setReadout = (text) => {
    if (readout) readout.textContent = text;
  };

  const setMode = (mode, initiatedByUser = false) => {
    if (!stage) return;
    controls.forEach((control) => {
      control.classList.toggle("active", control.dataset.deskMode === mode);
      control.setAttribute("aria-pressed", String(control.dataset.deskMode === mode));
    });
    stage.classList.remove("is-scanning", "is-evidence");

    if (mode === "scan") {
      // Restart the CSS scan even when the same control is selected twice.
      void stage.offsetWidth;
      stage.classList.add("is-scanning");
      setReadout("透光扫描中");
      schedule(() => {
        if (stage.classList.contains("is-scanning")) setReadout("扫描完成");
      }, reducedMotion ? 0 : 3250);
    } else if (mode === "evidence") {
      stage.classList.add("is-evidence");
      setReadout("连续性证据");
    } else {
      setReadout("稳定观察");
    }

    if (initiatedByUser) {
      timers.splice(0).forEach(window.clearTimeout);
    }
  };

  controls.forEach((control) => {
    control.setAttribute("aria-pressed", String(control.classList.contains("active")));
    control.addEventListener("click", () => setMode(control.dataset.deskMode, true));
  });

  if (stage && !reducedMotion) {
    const heroCopy = [...document.querySelectorAll(".hero-copy > *")];
    const fragments = [...stage.querySelectorAll(".hero-fragment")];

    introAnimations = heroCopy.map((element, index) => element.animate(
      [
        { opacity: 0, transform: "translateY(24px)" },
        { opacity: 1, transform: "translateY(0)" },
      ],
      {
        duration: 720,
        delay: 120 + index * 105,
        easing: "cubic-bezier(.18,.78,.2,1)",
        fill: "both",
      },
    ));

    introAnimations.push(stage.animate(
      [
        { opacity: 0, transform: "perspective(1400px) translateY(24px) rotateX(2.2deg) scale(.975)" },
        { opacity: 1, transform: "perspective(1400px) translateY(0) rotateX(0) scale(1)" },
      ],
      {
        duration: 1050,
        delay: 340,
        easing: "cubic-bezier(.16,.82,.2,1)",
        fill: "both",
      },
    ));

    fragments.forEach((fragment, index) => {
      const target = getComputedStyle(fragment).transform;
      introAnimations.push(fragment.animate(
        [
          {
            opacity: 0,
            transform: `translate3d(${index ? 105 : -105}px, 24px, 30px) rotate(${index ? 6 : -6}deg)`,
          },
          { opacity: 1, transform: target },
        ],
        {
          duration: 1250,
          delay: 760 + index * 210,
          easing: "cubic-bezier(.16,.82,.18,1)",
          fill: "both",
        },
      ));
    });

    // One guided conservation sequence on first view; afterward the controls are manual.
    schedule(() => setMode("scan"), 2550);
    schedule(() => setMode("evidence"), 6250);
    schedule(() => setMode("observe"), 10500);

    let frame = 0;
    stage.addEventListener("pointermove", (event) => {
      if (frame) cancelAnimationFrame(frame);
      frame = requestAnimationFrame(() => {
        const rect = stage.getBoundingClientRect();
        const x = (event.clientX - rect.left) / rect.width - 0.5;
        const y = (event.clientY - rect.top) / rect.height - 0.5;
        stage.style.setProperty("--desk-x", `${(x * 1.4).toFixed(2)}deg`);
        stage.style.setProperty("--desk-y", `${(-y * 1.1).toFixed(2)}deg`);
      });
    });
    stage.addEventListener("pointerleave", () => {
      stage.style.setProperty("--desk-x", "0deg");
      stage.style.setProperty("--desk-y", "0deg");
    });
  } else {
    setMode("observe");
  }

  skip?.addEventListener("click", () => {
    introAnimations.forEach((animation) => animation.finish());
    timers.splice(0).forEach(window.clearTimeout);
    setMode("observe", true);
    skip.hidden = true;
  });

  const revealTargets = [
    ...document.querySelectorAll(
      ".conservation-ledger, .module-heading, .steps, .evidence-thread, #uploadSection, .report-card, .server-bar",
    ),
  ];

  if (!reducedMotion && "IntersectionObserver" in window) {
    revealTargets.forEach((element) => element.classList.add("motion-reveal"));
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.08, rootMargin: "0px 0px -8% 0px" });
    revealTargets.forEach((element) => observer.observe(element));
  } else {
    revealTargets.forEach((element) => element.classList.add("is-visible"));
  }

  document.querySelectorAll(".upload-card").forEach((card) => {
    card.addEventListener("pointermove", (event) => {
      const rect = card.getBoundingClientRect();
      card.style.setProperty("--pointer-x", `${event.clientX - rect.left}px`);
      card.style.setProperty("--pointer-y", `${event.clientY - rect.top}px`);
    });
  });
});
