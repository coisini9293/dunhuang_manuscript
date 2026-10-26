(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const toast = (message) => {
    const node = $("#toast");
    if (!node) return;
    node.textContent = message;
    node.classList.add("show");
    clearTimeout(toast.timer);
    toast.timer = setTimeout(() => node.classList.remove("show"), 2600);
  };
  const downloadBlob = (name, data, type = "application/json") => {
    const href = URL.createObjectURL(new Blob([data], { type }));
    const link = Object.assign(document.createElement("a"), { href, download: name });
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(href), 1000);
  };
  const downloadUrl = (url, name) => {
    const link = Object.assign(document.createElement("a"), { href: url, download: name });
    document.body.append(link);
    link.click();
    link.remove();
  };

  const heroFilm = $("#heroFilm");
  const heroVideo = $("#heroMotion");
  const finishHero = () => {
    heroVideo?.pause();
    heroFilm?.classList.add("is-finished");
    sessionStorage.setItem("dunhuang-v3-hero-seen", "1");
  };
  $("#skipHeroFilm")?.addEventListener("click", finishHero);
  heroVideo?.addEventListener("ended", finishHero);
  if (heroFilm && heroVideo) {
    const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (reduced || sessionStorage.getItem("dunhuang-v3-hero-seen")) {
      heroFilm.classList.add("is-finished");
    } else {
      const play = heroVideo.play();
      play?.catch(() => {
        heroFilm.classList.remove("is-finished");
        toast("点击影像即可播放开场");
      });
      heroFilm.addEventListener("click", (event) => {
        if (event.target.closest("button")) return;
        heroVideo.paused ? heroVideo.play() : heroVideo.pause();
      });
    }
  }

  const previewA = $("#previewA");
  const previewB = $("#previewB");
  const cardA = $("#cardA");
  const cardB = $("#cardB");
  const replayPanel = $("#rejoinReplayPanel");
  const replayAssets = {
    a: "assets/rejoin/fragment_a.png",
    b: "assets/rejoin/fragment_b.png"
  };
  const setPreview = (image, card, metaId, src, label, dimensions) => {
    if (!image || !card) return;
    image.src = src;
    image.style.display = "block";
    card.classList.add("has-image");
    const placeholder = $(".upload-placeholder", card);
    if (placeholder) placeholder.style.display = "none";
    const meta = $(metaId);
    if (meta) {
      $("b", meta).textContent = label;
      $("span", meta).textContent = dimensions;
    }
  };
  const loadRejoinReplay = () => {
    setPreview(previewA, cardA, "#metaA", replayAssets.a, "残卷 A · 真实回放", "530 × 1143 px");
    setPreview(previewB, cardB, "#metaB", replayAssets.b, "残卷 B · 真实回放", "531 × 1143 px");
    replayPanel?.classList.add("is-visible");
    $("#btnRejoin")?.removeAttribute("disabled");
    replayPanel?.scrollIntoView({ behavior: "smooth", block: "start" });
    toast("已加载真实本地缀合回放；未调用服务器");
  };
  $("#loadRejoinReplay")?.addEventListener("click", loadRejoinReplay);
  $("#loadReplay")?.addEventListener("click", () => {
    $("#verifiedReplay")?.scrollIntoView({ behavior: "smooth" });
  });

  let swapped = false;
  let rotationA = 0;
  let rotationB = 0;
  $("#swapFragments")?.addEventListener("click", () => {
    swapped = !swapped;
    setPreview(previewA, cardA, "#metaA", swapped ? replayAssets.b : replayAssets.a, swapped ? "残卷 B · 交换位" : "残卷 A · 真实回放", swapped ? "531 × 1143 px" : "530 × 1143 px");
    setPreview(previewB, cardB, "#metaB", swapped ? replayAssets.a : replayAssets.b, swapped ? "残卷 A · 交换位" : "残卷 B · 真实回放", swapped ? "530 × 1143 px" : "531 × 1143 px");
    toast("残卷 A / B 已交换");
  });
  const rotate = (which) => {
    if (which === "a") {
      rotationA = (rotationA + 90) % 360;
      previewA?.style.setProperty("--fragment-rotation", `${rotationA}deg`);
    } else {
      rotationB = (rotationB + 90) % 360;
      previewB?.style.setProperty("--fragment-rotation", `${rotationB}deg`);
    }
  };
  $("#rotateFragmentA")?.addEventListener("click", () => rotate("a"));
  $("#rotateFragmentB")?.addEventListener("click", () => rotate("b"));
  $("#replaceFragmentA")?.addEventListener("click", () => $("#fileA")?.click());
  $("#replaceFragmentB")?.addEventListener("click", () => $("#fileB")?.click());
  const removePreview = (image, card, label) => {
    if (image) {
      image.removeAttribute("src");
      image.style.cssText = "";
    }
    card?.classList.remove("has-image");
    const placeholder = card && $(".upload-placeholder", card);
    if (placeholder) placeholder.style.display = "";
    toast(`${label} 已从托板移除`);
  };
  $("#removeFragmentA")?.addEventListener("click", () => removePreview(previewA, cardA, "残卷 A"));
  $("#removeFragmentB")?.addEventListener("click", () => removePreview(previewB, cardB, "残卷 B"));
  $("#resetWorkbench")?.addEventListener("click", () => {
    swapped = false; rotationA = 0; rotationB = 0;
    [previewA, previewB].forEach((image) => { if (image) image.style.cssText = ""; });
    [cardA, cardB].forEach((card) => {
      card?.classList.remove("has-image");
      const placeholder = card && $(".upload-placeholder", card);
      if (placeholder) placeholder.style.display = "";
    });
    replayPanel?.classList.remove("is-visible");
    toast("缀合案台已复位");
  });

  $$("[data-fullscreen]").forEach((button) => button.addEventListener("click", (event) => {
    event.stopPropagation();
    const image = document.getElementById(button.dataset.fullscreen);
    if (!image?.src) return toast("请先加载残卷");
    const lightbox = $("#artifactLightbox");
    const target = $("#lightboxImage");
    if (target) target.src = image.src;
    lightbox?.classList.add("show");
  }));

  const rejoinStage = $("#rejoinStage");
  $$("[data-rejoin-view]").forEach((button) => button.addEventListener("click", () => {
    $$("[data-rejoin-view]").forEach((item) => item.classList.toggle("active", item === button));
    if (rejoinStage) rejoinStage.dataset.view = button.dataset.rejoinView;
  }));
  $("#rejoinWipe")?.addEventListener("input", (event) => rejoinStage?.style.setProperty("--wipe", `${event.target.value}%`));
  $("#downloadRejoinImage")?.addEventListener("click", () => downloadUrl("assets/rejoin/panorama_reference.png", "EDH-DX01962-02052_panorama_reference.png"));
  $("#downloadRejoinReport")?.addEventListener("click", () => downloadBlob("local_rejoin_verified_replay.json", JSON.stringify({
    status: "LOCAL_REJOIN_HTTP_PASS",
    source: "saved local replay",
    source_dimensions: [855, 1143],
    fragment_a_dimensions: [530, 1143],
    fragment_b_dimensions: [531, 1143],
    verdict: "match",
    confidence: 0.999901533126831,
    pose: { dx: -0.6210638881, dy: -0.0556178097, rotation: 0.1172216162 },
    panorama_dimensions: [1062, 1134],
    note: "panorama_reference.png is a complete-image reference, not a newly generated model output"
  }, null, 2)));
  $$("[data-mobile-workbench]").forEach((button) => button.addEventListener("click", () => {
    $$("[data-mobile-workbench]").forEach((item) => item.classList.toggle("active", item === button));
    const target = button.dataset.mobileWorkbench;
    if (target === "a") cardA?.scrollIntoView({ behavior: "smooth", block: "center" });
    if (target === "b") cardB?.scrollIntoView({ behavior: "smooth", block: "center" });
    if (target === "compare") {
      if (!replayPanel?.classList.contains("is-visible")) loadRejoinReplay();
      else replayPanel.scrollIntoView({ behavior: "smooth", block: "start" });
    }
    if (target === "evidence") $("#offlineEvidenceDesk")?.scrollIntoView({ behavior: "smooth" });
  }));

  const evidenceLedger = $(".candidate-ledger");
  $("#offlineRetrievalMode")?.addEventListener("change", (event) => {
    const noise = event.target.value === "noise";
    evidenceLedger?.classList.toggle("show-noise", noise);
    const label = $("#candidateModeLabel");
    if (label) label.textContent = noise ? "噪声增强真实记录 · 4.704 秒" : "基线真实记录 · 486.573 毫秒";
    toast(noise ? "显示保存的噪声增强歧义候选" : "显示保存的基线结果");
  });
  $$("[data-copy-citation]").forEach((button) => button.addEventListener("click", async () => {
    const text = "G2075《御製釋迦牟尼佛讚》卷1 G083n2075_p0007a11–G083n2075_p0007a15（不詳，G）";
    try { await navigator.clipboard.writeText(text); } catch { downloadBlob("G2075_citation.txt", text, "text/plain"); }
    toast("完整佛典引用已复制");
  }));

  const regions = [
    { status: "已批准区域 · 保存记录", candidate: "候选文字：於", citation: "G2075《御製釋迦牟尼佛讚》卷1 · 完整引用保存在案例审计中。" },
    { status: "已拒绝区域 · 保存记录", candidate: "候选文字：未采纳", citation: "候选与视觉笔画不一致；区域不进入修复。" },
    { status: "已弃权区域 · 保存记录", candidate: "候选文字：证据不足", citation: "单侧上下文仅用于定位；保持原始像素。" }
  ];
  let activeRegion = 0;
  const renderRegion = () => {
    $$("[data-region]").forEach((button, index) => button.classList.toggle("active", index === activeRegion));
    $("#regionStatus").textContent = regions[activeRegion].status;
    $("#regionCandidate").textContent = regions[activeRegion].candidate;
    $("#regionCitation").textContent = regions[activeRegion].citation;
    $("#reviewProgressText").textContent = `${activeRegion + 1} / 3 示例区域`;
    $("#reviewProgressFill").style.width = `${((activeRegion + 1) / 3) * 100}%`;
  };
  $$("[data-region]").forEach((button, index) => button.addEventListener("click", () => { activeRegion = index; renderRegion(); }));
  $("#prevRegion")?.addEventListener("click", () => { activeRegion = (activeRegion + regions.length - 1) % regions.length; renderRegion(); });
  $("#nextRegion")?.addEventListener("click", () => { activeRegion = (activeRegion + 1) % regions.length; renderRegion(); });
  $$("[data-add-review]").forEach((button) => button.addEventListener("click", () => {
    activeRegion = 0; renderRegion();
    $("#offlineReviewDesk")?.scrollIntoView({ behavior: "smooth" });
    toast(`候选“${button.closest(".citation-slip")?.dataset.candidateText}”已放入审核台`);
  }));
  $$("[data-insufficient]").forEach((button) => button.addEventListener("click", () => {
    button.closest(".citation-slip")?.classList.remove("selected");
    toast("已标记证据不足；不会自动补字");
  }));
  $$("[data-decision]").forEach((button) => button.addEventListener("click", () => {
    $$("[data-decision]").forEach((item) => item.classList.toggle("is-selected", item === button));
    const labels = { accept: "接受", edit: "修改", reject: "拒绝", abstain: "弃权" };
    const item = document.createElement("li");
    item.innerHTML = `<b>区域${String(activeRegion).padStart(4, "0")}</b><span>${labels[button.dataset.decision]} · 今晚本地界面演示（未提交服务器）</span>`;
    $("#decisionHistory")?.prepend(item);
    toast(`已记录本地决策：${labels[button.dataset.decision]}`);
  }));
  $("#buildOfflinePlan")?.addEventListener("click", () => $("#offlineRepairPlan")?.scrollIntoView({ behavior: "smooth" }));

  const viewport = $("#imageViewport");
  const pan = $("#viewportPan");
  let viewerScale = 1;
  let viewerX = 0;
  let viewerY = 0;
  let dragging = false;
  let dragStart = { x: 0, y: 0 };
  const applyTransform = () => { if (pan) pan.style.transform = `translate(${viewerX}px, ${viewerY}px) scale(${viewerScale})`; };
  const setScale = (next) => { viewerScale = Math.min(4, Math.max(.5, next)); applyTransform(); };
  $("#viewerFit")?.addEventListener("click", () => { viewerScale = 1; viewerX = 0; viewerY = 0; applyTransform(); });
  $("#viewer100")?.addEventListener("click", () => { viewerScale = 1; applyTransform(); });
  $("#viewerZoomIn")?.addEventListener("click", () => setScale(viewerScale + .25));
  $("#viewerZoomOut")?.addEventListener("click", () => setScale(viewerScale - .25));
  viewport?.addEventListener("pointerdown", (event) => {
    if (event.target.matches("input")) return;
    dragging = true;
    dragStart = { x: event.clientX - viewerX, y: event.clientY - viewerY };
    viewport.setPointerCapture(event.pointerId);
  });
  viewport?.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    viewerX = event.clientX - dragStart.x;
    viewerY = event.clientY - dragStart.y;
    applyTransform();
  });
  viewport?.addEventListener("pointerup", () => { dragging = false; });

  $$("[data-layer]").forEach((button) => button.addEventListener("click", () => {
    $$("[data-layer]").forEach((item) => item.classList.toggle("active", item === button));
    if (viewport) viewport.dataset.layer = button.dataset.layer;
  }));
  $("#resultWipe")?.addEventListener("input", (event) => viewport?.style.setProperty("--wipe", `${event.target.value}%`));
  $("#layerOpacity")?.addEventListener("input", (event) => {
    const value = Number(event.target.value);
    viewport?.style.setProperty("--layer-opacity", value / 100);
    $("#opacityValue").textContent = `${value}%`;
  });
  $("#viewerFullscreen")?.addEventListener("click", () => $(".examiner-shell")?.requestFullscreen?.());

  let flickerTimer = null;
  $("#viewerFlicker")?.addEventListener("click", (event) => {
    if (flickerTimer) {
      clearInterval(flickerTimer);
      flickerTimer = null;
      event.currentTarget.textContent = "闪烁对比";
      return;
    }
    let result = false;
    flickerTimer = setInterval(() => {
      result = !result;
      if (viewport) viewport.dataset.layer = result ? "result" : "input";
    }, 520);
    event.currentTarget.textContent = "停止闪烁";
  });
  const lens = $("#viewerLens");
  $("#viewerMagnifier")?.addEventListener("click", () => {
    lens?.classList.toggle("is-active");
    toast(lens?.classList.contains("is-active") ? "移动指针观察局部笔画" : "局部放大镜已关闭");
  });
  viewport?.addEventListener("pointermove", (event) => {
    if (!lens?.classList.contains("is-active") || dragging) return;
    const rect = viewport.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    lens.style.left = `${x - 95}px`;
    lens.style.top = `${y - 95}px`;
    lens.style.backgroundPosition = `${(x / rect.width) * 100}% ${(y / rect.height) * 100}%`;
  });

  $("#downloadCurrentView")?.addEventListener("click", () => {
    const layer = viewport?.dataset.layer || "input";
    const sources = { input: "assets/replay/input.png", result: "assets/replay/result.png", difference: "assets/replay/comparison.png", mask: "assets/replay/approved_mask.png", approved: "assets/replay/approved_mask.png" };
    if (!sources[layer]) return toast("该审计状态没有独立像素图层");
    downloadUrl(sources[layer], `EDH-DX01962-02052_${layer}.png`);
  });
  $("#downloadAuditBundle")?.addEventListener("click", () => downloadBlob("EDH-DX01962-02052_audit.json", JSON.stringify({
    case_id: "8f45998b-6219-443e-a09d-af06243feecb",
    replay_status: "saved_verified_replay",
    official_autohdr_commit: "d92b223b3d9274b03ca6f1e4c62b09253f378c98",
    approved_regions: 1,
    rejected_regions: 1,
    abstained_regions: 151,
    changed_pixels_inside_mask: 1935,
    outside_mask_changed_pixels: 0,
    rejected_abstained_changed_pixels: 0,
    peak_gpu_gib: 4.64,
    human_review_status: "not_reviewed",
    note: "This is a saved audit record, not a new GPU run."
  }, null, 2)));

  const sections = ["#top", "#module1", "#offlineEvidenceDesk", "#offlineReviewDesk", "#professionalViewer"];
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        $$(".mobile-dock a").forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${entry.target.id}`));
      });
    }, { rootMargin: "-35% 0px -55%" });
    sections.forEach((selector) => { const section = $(selector); if (section) observer.observe(section); });
  }
})();
