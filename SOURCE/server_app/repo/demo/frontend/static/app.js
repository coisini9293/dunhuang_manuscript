/* Gurou Demo — Module 1 (rejoin + legacy DiffHDR) and Module 2 (AutoHDR + CBETA RAG jobs) */

let fileA = null;
let fileB = null;
let rejoinDone = false;
let maskEditor = null;
let rejoinPanoramaSrc = "";

const $ = (id) => document.getElementById(id);
const MIN_MASK_PIXELS = 50;
const escapeHtml = (value) => String(value ?? "").replace(
  /[&<>"']/g,
  (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char],
);
const cbetaText = (candidate) => candidate?.candidate_text || candidate?.missing_chars || candidate?.text || "";
const scorePct = (value) => Number.isFinite(Number(value)) ? Math.round(Number(value) * 100) + "%" : "—";

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.add("show");
  setTimeout(() => t.classList.remove("show"), 3200);
}

function setStep(n) {
  [1, 2, 3].forEach((i) => {
    $("step" + i).classList.toggle("active", i <= n);
  });
}

function bindUpload(inputId, previewId, cardId, slot) {
  const input = $(inputId);
  const preview = $(previewId);
  const card = $(cardId);

  input.addEventListener("change", () => {
    const file = input.files[0];
    if (!file) return;
    if (slot === "A") fileA = file;
    else fileB = file;
    rejoinDone = false;
    rejoinPanoramaSrc = "";
    maskEditor = null;
    $("maskAdvanced").classList.add("hidden");
    $("maskAdvanced").open = false;
    $("btnRestore").disabled = true;
    $("btnDownloadPanorama").disabled = true;
    $("btnSendToRestore").disabled = true;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      card.classList.add("has-image");
      $("btnRejoin").disabled = !(fileA && fileB);
    };
    reader.readAsDataURL(file);
  });

  card.addEventListener("dragover", (e) => { e.preventDefault(); card.style.borderColor = "#b8863a"; });
  card.addEventListener("dragleave", () => { card.style.borderColor = ""; });
  card.addEventListener("drop", (e) => {
    e.preventDefault();
    card.style.borderColor = "";
    const file = e.dataTransfer.files[0];
    if (!file || !file.type.startsWith("image/")) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event("change"));
  });
}

function setLoading(btn, loading) {
  btn.classList.toggle("loading", loading);
  btn.disabled = loading || (btn.id === "btnRestore" && !maskEditor?.hasDamage());
}

function animateConfidence(pct) {
  const arc = $("confArc");
  const circumference = 326.7;
  if (typeof gsap === "undefined") {
    $("confPct").textContent = Math.round(pct) + "%";
    arc.style.strokeDashoffset = circumference - (pct / 100) * circumference;
    return;
  }
  gsap.to({ val: 0 }, {
    val: pct,
    duration: 1.4,
    ease: "power2.out",
    onUpdate: function () {
      $("confPct").textContent = Math.round(this.targets()[0].val) + "%";
      arc.style.strokeDashoffset = circumference - (this.targets()[0].val / 100) * circumference;
    },
  });
}

function showSection(el) {
  el.classList.remove("hidden");
  if (typeof gsap !== "undefined") {
    gsap.fromTo(el, { opacity: 0, y: 32 }, { opacity: 1, y: 0, duration: 0.8, ease: "power3.out" });
  }
  el.classList.add("visible");
}

function createMaskEditor(panoramaSrc) {
  const display = $("maskDisplay");
  const binary = $("maskBinary");
  const displayCtx = display.getContext("2d");
  const binaryCtx = binary.getContext("2d");
  const pano = new Image();

  const state = {
    tool: "brush",
    brushSize: 28,
    drawing: false,
    damagePixels: 0,
  };

  function updateRestoreButton() {
    const ok = state.damagePixels >= MIN_MASK_PIXELS;
    $("btnRestore").disabled = !ok;
    $("restoreHint").textContent = ok
      ? "已标注缺损区域，可执行修复"
      : "如需 DiffHDR 修复，请展开高级选项并标注缺损区域";
    $("restoreHint").style.color = ok ? "var(--gold)" : "rgba(28,22,18,0.5)";
  }

  function countDamagePixels() {
    const data = binaryCtx.getImageData(0, 0, binary.width, binary.height).data;
    let count = 0;
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 127) count += 1;
    }
    state.damagePixels = count;
    const pct = binary.width * binary.height > 0
      ? (count / (binary.width * binary.height)) * 100
      : 0;
    $("maskAreaPct").textContent = pct.toFixed(2) + "%";
    updateRestoreButton();
  }

  function redraw() {
    displayCtx.clearRect(0, 0, display.width, display.height);
    displayCtx.drawImage(pano, 0, 0, display.width, display.height);

    const overlay = document.createElement("canvas");
    overlay.width = binary.width;
    overlay.height = binary.height;
    const octx = overlay.getContext("2d");
    octx.fillStyle = "rgba(184, 134, 58, 0.55)";
    octx.fillRect(0, 0, overlay.width, overlay.height);
    octx.globalCompositeOperation = "destination-in";
    octx.drawImage(binary, 0, 0);
    displayCtx.drawImage(overlay, 0, 0);
  }

  function canvasPoint(evt) {
    const rect = display.getBoundingClientRect();
    const clientX = evt.clientX ?? evt.touches?.[0]?.clientX;
    const clientY = evt.clientY ?? evt.touches?.[0]?.clientY;
    const x = ((clientX - rect.left) / rect.width) * display.width;
    const y = ((clientY - rect.top) / rect.height) * display.height;
    return { x, y };
  }

  function paint(evt) {
    const { x, y } = canvasPoint(evt);
    binaryCtx.lineCap = "round";
    binaryCtx.lineJoin = "round";
    binaryCtx.lineWidth = state.brushSize;
    if (state.tool === "brush") {
      binaryCtx.strokeStyle = "#ffffff";
      binaryCtx.globalCompositeOperation = "source-over";
    } else {
      binaryCtx.strokeStyle = "#000000";
      binaryCtx.globalCompositeOperation = "source-over";
    }
    binaryCtx.lineTo(x, y);
    binaryCtx.stroke();
    binaryCtx.beginPath();
    binaryCtx.moveTo(x, y);
    redraw();
    countDamagePixels();
  }

  function startDraw(evt) {
    evt.preventDefault();
    state.drawing = true;
    const { x, y } = canvasPoint(evt);
    binaryCtx.beginPath();
    binaryCtx.moveTo(x, y);
    paint(evt);
  }

  function endDraw() {
    state.drawing = false;
    binaryCtx.beginPath();
  }

  function clearMask() {
    binaryCtx.fillStyle = "#000000";
    binaryCtx.fillRect(0, 0, binary.width, binary.height);
    redraw();
    countDamagePixels();
  }

  function exportMaskBlob() {
    return new Promise((resolve) => {
      binary.toBlob((blob) => resolve(blob), "image/png");
    });
  }

  pano.onload = () => {
    display.width = pano.naturalWidth;
    display.height = pano.naturalHeight;
    binary.width = pano.naturalWidth;
    binary.height = pano.naturalHeight;
    clearMask();
    $("maskAdvanced").classList.remove("hidden");
  };
  pano.src = panoramaSrc;

  display.onmousedown = startDraw;
  display.onmousemove = (e) => { if (state.drawing) paint(e); };
  display.onmouseup = endDraw;
  display.onmouseleave = endDraw;
  display.ontouchstart = startDraw;
  display.ontouchmove = (e) => { if (state.drawing) paint(e); };
  display.ontouchend = endDraw;

  $("toolBrush").onclick = () => {
    state.tool = "brush";
    $("toolBrush").classList.add("active");
    $("toolEraser").classList.remove("active");
  };
  $("toolEraser").onclick = () => {
    state.tool = "eraser";
    $("toolEraser").classList.add("active");
    $("toolBrush").classList.remove("active");
  };
  $("toolClear").onclick = () => {
    clearMask();
    toast("已清除标注");
  };
  $("brushSize").oninput = (e) => {
    state.brushSize = Number(e.target.value);
  };

  return {
    hasDamage: () => state.damagePixels >= MIN_MASK_PIXELS,
    exportMaskBlob,
  };
}

async function runRejoin() {
  if (!fileA || !fileB) { toast("请先上传两片残卷"); return; }

  const btn = $("btnRejoin");
  setLoading(btn, true);

  const fd = new FormData();
  fd.append("img1", fileA);
  fd.append("img2", fileB);

  try {
    const resp = await fetch("/api/rejoin", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Rejoin failed");

    const r = data.rejoin;
    const pct = Math.round(r.confidence * 100);
    $("reportTime").textContent = "生成于 · " + new Date().toLocaleString("zh-CN");
    $("mDx").textContent = r.pose.dx.toFixed(1) + " px";
    $("mDy").textContent = r.pose.dy.toFixed(1) + " px";
    $("mRot").textContent = r.pose.rotation.toFixed(3) + " rad";
    $("mVerdict").textContent = r.verdict === "match" ? "可缀合" : "不可缀合";

    const badge = $("verdictBadge");
    badge.className = "verdict-badge " + r.verdict;
    badge.textContent = r.verdict === "match" ? "✦ 判定：可缀合" : "✦ 判定：非同源";

    animateConfidence(pct);
    $("rptFragA").src = data.images.fragment_a;
    $("rptFragB").src = data.images.fragment_b;
    $("rptPanorama").src = data.images.panorama;
    rejoinPanoramaSrc = data.images.panorama;
    $("btnDownloadPanorama").disabled = false;
    $("btnSendToRestore").disabled = false;

    maskEditor = createMaskEditor(data.images.panorama);
    rejoinDone = true;
    setStep(2);
    showSection($("reportSection"));
    $("reportSection").scrollIntoView({ behavior: "smooth", block: "start" });
    toast("缀合分析完成；如需 DiffHDR 修复，请展开高级选项标注缺损区域");
  } catch (err) {
    toast("缀合失败：" + err.message);
  } finally {
    setLoading(btn, false);
  }
}

async function downloadPanorama() {
  if (!rejoinPanoramaSrc) return;
  const blob = await fetch(rejoinPanoramaSrc).then((r) => r.blob());
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "rejoin_panorama.png";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function panoramaFile() {
  const [header, payload] = rejoinPanoramaSrc.split(",");
  const mime = (header.match(/data:([^;]+)/) || [])[1] || "image/png";
  const binary = atob(payload);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new File([bytes], "rejoin_panorama.png", { type: mime });
}

function sendPanoramaToRestore() {
  if (!rejoinPanoramaSrc) return;
  const input = $("m2FileInp");
  const transfer = new DataTransfer();
  transfer.items.add(panoramaFile());
  input.files = transfer.files;
  input.dataset.source = "智能缀合结果";
  input.dispatchEvent(new Event("change"));
  switchModule("module2");
  m2ShowStep(1);
  $("module2").scrollIntoView({ behavior: "smooth", block: "start" });
  toast("全景图已送入证据约束修复模块");
}

async function runRestore() {
  if (!rejoinDone || !fileA || !fileB) { toast("请先完成缀合分析"); return; }
  if (!maskEditor?.hasDamage()) { toast("请先在全景图上标注缺损区域"); return; }

  const btn = $("btnRestore");
  setLoading(btn, true);
  toast("DiffHDR 修复中，请稍候…");

  const maskBlob = await maskEditor.exportMaskBlob();
  const fd = new FormData();
  fd.append("img1", fileA);
  fd.append("img2", fileB);
  fd.append("mask", maskBlob, "mask.png");

  try {
    const resp = await fetch("/api/restore", { method: "POST", body: fd });
    const data = await resp.json();
    if (!resp.ok) throw new Error(data.error || "Restore failed");

    $("imgBefore").src = data.images.before;
    $("imgAfter").src = data.images.after;
    $("imgMask").src = data.images.mask;
    $("mDamage").textContent = (data.mask_area_ratio * 100).toFixed(2) + "%";

    setStep(3);
    showSection($("compareSection"));
    initCompareSlider();
    $("compareSection").scrollIntoView({ behavior: "smooth", block: "start" });
    toast("修复完成");
  } catch (err) {
    toast("修复失败：" + err.message);
  } finally {
    setLoading(btn, false);
    if (maskEditor?.hasDamage()) $("btnRestore").disabled = false;
  }
}

function initCompareSlider() {
  const wrap = $("compareWrap");
  const after = $("compareAfter");
  const handle = $("compareHandle");
  let dragging = false;

  function setPos(x) {
    const rect = wrap.getBoundingClientRect();
    let pct = ((x - rect.left) / rect.width) * 100;
    pct = Math.max(2, Math.min(98, pct));
    after.style.clipPath = `inset(0 ${100 - pct}% 0 0)`;
    handle.style.left = pct + "%";
  }

  handle.addEventListener("mousedown", () => { dragging = true; });
  window.addEventListener("mouseup", () => { dragging = false; });
  wrap.addEventListener("mousemove", (e) => { if (dragging) setPos(e.clientX); });
  wrap.addEventListener("touchmove", (e) => { if (e.touches[0]) setPos(e.touches[0].clientX); }, { passive: true });
  setPos(wrap.getBoundingClientRect().left + wrap.offsetWidth / 2);
}

/* ===================== Module switching ===================== */
function switchModule(mod) {
  document.querySelectorAll(".nav-tab").forEach((t) => t.classList.toggle("active", t.dataset.mod === mod));
  $("module1").classList.toggle("hidden", mod !== "module1");
  $("module2").classList.toggle("hidden", mod !== "module2");
  if (typeof gsap !== "undefined") {
    gsap.fromTo("#" + mod, { opacity: 0, y: 20 }, { opacity: 1, y: 0, duration: 0.6, ease: "power3.out" });
  }
}

/* ===================== MODULE 2: AutoHDR + CBETA RAG ===================== */
let m2RunId = null, m2Job = null;
let m2InputSource = "单张残卷上传";
let m2Sel = JSON.parse(localStorage.getItem("gurou_confirmed") || "{}");

async function m2Health() {
  try {
    const [r, rag] = await Promise.all([
      fetch("/api/system/health").then((response) => response.json()),
      fetch("/api/rag/health").then((response) => response.ok ? response.json() : ({ loaded: false })),
    ]);
    const modeNames = { mock: "Mock（演示）", replay: "Replay（回放）", official: "Official（未运行/待部署）" };
    const txt = modeNames[r.autohdr_backend] || r.autohdr_backend;
    const bs = $("m2BkStatus");
    if (bs) bs.textContent = txt;
    const mb = $("modeBadge");
    if (mb) {
      mb.style.display = "inline-block";
      mb.textContent = txt;
    }
    $("m2RagStatus").textContent = rag.loaded ? (rag.vector_enabled ? "RAG V2 混合检索" : "RAG V2 稀疏检索") : "不可用";
    $("m2CorpusScope").textContent = rag.corpus_scope === "full" ? "全量 CBETA" : "T0278 单经";
    $("m2SutraCount").textContent = rag.sutra_count ?? "-";
    $("m2CorpusVersion").textContent = rag.corpus_version || "-";
    $("m2IndexVersion").textContent = rag.index_version || "-";
    $("m2VectorStatus").textContent = rag.vector_enabled ? "已启用" : (rag.faiss_ready ? "已构建，默认关闭" : "未构建");
  } catch (e) { /* backend not reachable */ }
}

async function m2LoadRealCases() {
  const grid = $("m2CaseGrid");
  if (!grid) return;
  try {
    const response = await fetch("/api/real-cases?limit=12");
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "真实案例读取失败");
    if (!payload.cases?.length) {
      grid.innerHTML = '<div class="real-case-loading">尚无可用案例；仍可上传本地图像。</div>';
      return;
    }
    grid.innerHTML = payload.cases.map((item) => {
      const evidence = item.li_shengduo_evidence_level === "confirmed"
        ? "李盛铎关系：已获论文确认"
        : (item.case05_candidate ? "Case05 候选：仍阻塞" : "来源已登记 · 归属待复核");
      const disabled = item.upload_ready ? "" : " disabled";
      const action = item.image_status === "unavailable_placeholder"
        ? "馆藏原图待人工下载"
        : (item.upload_ready ? "选择并创建任务" : "超过 20MB，请先建工作副本");
      return `<article class="real-case-card${item.case05_candidate ? " case05" : ""}">
        <div class="real-case-image"><img src="${escapeHtml(item.image_url)}" alt="${escapeHtml(item.shelfmark || item.case_id)}" loading="lazy"></div>
        <div class="real-case-body">
          <div class="real-case-kicker">${escapeHtml(item.source_type)} · ${escapeHtml(item.case_id)}</div>
          <h3>${escapeHtml(item.shelfmark || item.group_id)}</h3>
          <p>${escapeHtml(item.institution)}</p>
          <div class="provenance-strip">${escapeHtml(evidence)}</div>
          <div class="case-state">释文：未开始 · Official：未运行</div>
          <button class="case-select" onclick="m2SelectRealCase('${escapeHtml(item.case_id)}', this)"${disabled}>${action}</button>
        </div>
      </article>`;
    }).join("");
  } catch (error) {
    grid.innerHTML = `<div class="real-case-loading">真实案例库暂不可用：${escapeHtml(error.message)}</div>`;
  }
}

async function m2SelectRealCase(caseId, button) {
  button.disabled = true;
  const oldText = button.textContent;
  button.textContent = "正在创建任务…";
  try {
    const response = await fetch(`/api/real-cases/${encodeURIComponent(caseId)}/jobs`, { method: "POST" });
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "案例任务创建失败");
    m2RunId = job.run_id;
    m2Job = job;
    m2Sel = {};
    const source = job.real_case || {};
    m2InputSource = `${source.institution || "真实案例库"} · ${source.shelfmark || caseId}`;
    $("m2Source").textContent = m2InputSource;
    $("m2CurRun").textContent = m2RunId;
    const preview = $("m2Preview");
    preview.src = source.image_url;
    preview.style.display = "block";
    $("m2Placeholder").style.display = "none";
    $("m2UpInfo").style.display = "block";
    $("m2UpInfo").innerHTML = `<p><b>${escapeHtml(source.shelfmark || caseId)}</b></p>
      <p>来源：${escapeHtml(source.institution || "真实案例库")}</p>
      <p>释文：未开始 · 专家复核：待处理 · Official AutoHDR：未运行</p>`;
    localStorage.removeItem("gurou_confirmed");
    localStorage.setItem("gurou_run", m2RunId);
    localStorage.setItem("gurou_source", m2InputSource);
    m2ShowStep(2);
    $("m2aRes").innerHTML = `<p>真实案例任务已创建，等待分析。</p>
      <div class="actions"><button class="btn btn-primary" id="m2BtnAnalyze">开始分析</button></div>`;
    $("m2BtnAnalyze").addEventListener("click", m2DoAnalyze);
    $("module2").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    toast("案例选择失败：" + error.message);
    button.disabled = false;
    button.textContent = oldText;
  }
}

function m2OnFile() {
  const input = $("m2FileInp");
  const f = input.files[0];
  if (!f) return;
  m2InputSource = input.dataset.source || "单张残卷上传";
  delete input.dataset.source;
  $("m2Source").textContent = m2InputSource;
  $("m2Placeholder").style.display = "none";
  const pv = $("m2Preview");
  pv.style.display = "block";
  pv.src = URL.createObjectURL(f);
  $("m2UpInfo").style.display = "block";
  $("m2UpInfo").innerHTML = `<p>${f.name} — ${(f.size / 1024).toFixed(1)} KB — ${f.type}</p><p>来源：${m2InputSource}</p>`;
  $("m2BtnUp").disabled = false;
}

async function m2DoUpload() {
  const f = $("m2FileInp").files[0];
  if (!f) return;
  const fd = new FormData();
  fd.append("image", f);
  const resp = await fetch("/api/jobs", { method: "POST", body: fd });
  const r = await resp.json();
  if (!resp.ok) throw new Error(r.detail || "上传失败");
  m2RunId = r.run_id;
  m2Job = r;
  m2Sel = {};
  localStorage.removeItem("gurou_confirmed");
  $("m2CurRun").textContent = m2RunId;
  localStorage.setItem("gurou_run", m2RunId);
  localStorage.setItem("gurou_source", m2InputSource);
  m2ShowStep(2);
  $("m2aRes").innerHTML = `<p>任务已创建，等待分析。</p><div class="actions"><button class="btn btn-primary" id="m2BtnAnalyze">开始分析</button></div>`;
  $("m2BtnAnalyze").addEventListener("click", m2DoAnalyze);
}

async function m2DoAnalyze() {
  const el = $("m2aRes");
  el.innerHTML = "<p>分析中…</p>";
  const resp = await fetch("/api/jobs/" + m2RunId + "/analyze", { method: "POST" });
  const r = await resp.json();
  if (!resp.ok) throw new Error(r.detail || "分析失败");
  m2Job = { ...m2Job, ...r };
  const dmg = r.damages || [];
  const chs = r.ocr_chars || [];
  let h = `<p>缺损：${dmg.length} 处 &nbsp;|&nbsp; OCR 识别：${chs.length} 字</p>`;
  if (r.overlay_url) h += `<div class="preview-frame" style="margin-top:12px"><img src="${r.overlay_url}" alt="缺损检测叠层"></div>`;
  const analysisMode = r.is_replay ? "Replay（回放）" : (r.backend === "official" ? "Official（未运行/待部署）" : "Mock（演示）");
  h += `<p style="color:rgba(28,22,18,0.5);font-size:0.85rem">分析模式：${analysisMode}</p>`;
  h += `<div class="actions"><button class="btn btn-gold" id="m2BtnEvidence">加载 CBETA 证据</button></div>`;
  el.innerHTML = h;
  $("m2BtnEvidence").addEventListener("click", m2DoEvidence);
}

async function m2DoEvidence() {
  m2ShowStep(3);
  const el = $("m2eRes");
  el.innerHTML = "<p>证据检索中…</p>";
  const resp = await fetch("/api/jobs/" + m2RunId + "/evidence", { method: "POST" });
  const r = await resp.json();
  if (!resp.ok) throw new Error(r.detail || "证据检索失败");
  const fs = r.fusions || [];
  const dmg = m2Job.damages || [];
  if (!fs.length) {
    el.innerHTML = "<p>无损伤区域</p>";
    $("m2BtnToC").style.display = "block";
    return;
  }
  let h = "";
  const fusionStatusNames = {
    evidence_agreement: "多源证据一致",
    evidence_conflict: "多源证据冲突",
    cbeta_ambiguous: "CBETA 存在歧义",
    cbeta_only: "仅 CBETA 原文候选",
    single_side_localization: "单侧上下文定位",
    autohdr_only: "仅 AutoHDR 候选",
    abstain: "暂不修复",
    manual_confirmed: "已人工确认",
  };
  for (let i = 0; i < fs.length; i++) {
    const f = fs[i];
    const d = dmg[i] || {};
    const rawStatus = f.fusion_status || f.status || "unknown";
    const displayStatus = fusionStatusNames[rawStatus] || "状态未知";
    h += `<div class="report-card" style="margin:10px 0;padding:14px"><b>${f.damage_id || d.damage_id}</b> — ${displayStatus}`;
    if (f.synthetic_fixture) h += ' <span class="mk">合成样例</span>';
    if (f.cbeta_status === "unavailable") h += '<br><small style="color:#9c3d2e">CBETA 暂不可用</small>';
    h += `<div style="font-size:0.82rem;color:rgba(28,22,18,0.6);margin:4px 0">${(d.left_context || "").slice(0, 30)} | ${(d.right_context || "").slice(0, 30)}</div>`;
    const ah = f.autohdr_candidates || [];
    const cb = f.cbeta_candidates || [];
    const qw = f.qwen_candidates || [];
    if (ah.length) h += `<div style="color:#2563eb;font-size:0.85rem">AutoHDR：${ah.map((c) => c.text).join("、")}</div>`;
    if (cb.length) {
      h += '<div class="cbeta-evidence-list">';
      for (const c of cb) {
        const scores = c.retrieval_components || {};
        const location = c.citation || `${c.sutra_id || ""} 卷${c.volume || "—"} p${c.page_start || "—"}${c.column_start || ""}${c.line_start || ""}`;
        const evidenceText = c.evidence_original_text || c.original_text || "";
        const evidenceHeads = c.evidence_line_heads || c.line_heads || [];
        const evidenceRange = evidenceHeads.length
          ? `${evidenceHeads[0]}${evidenceHeads.length > 1 ? ` – ${evidenceHeads[evidenceHeads.length - 1]}` : ""}`
          : "";
        h += `<article class="cbeta-evidence">
          <div class="cbeta-evidence-head">
            <span class="cbeta-rank">候选 ${escapeHtml(c.rank || 1)}</span>
            <strong>${escapeHtml(cbetaText(c)) || "仅定位，不补字"}</strong>
            <span class="cbeta-decision">${escapeHtml(c.decision_status || rawStatus)}</span>
          </div>
          <div class="cbeta-citation">${escapeHtml(location)}</div>
          <div class="cbeta-score-strip" aria-label="检索分数组成">
            <span>字符匹配 <b>${scorePct(scores.ngram)}</b></span>
            <span>BM25 <b>${scorePct(scores.bm25)}</b></span>
            <span>向量 <b>${scorePct(scores.vector)}</b></span>
            <span>上下文连续性 <b>${scorePct(scores.continuity)}</b></span>
          </div>
          <div class="cbeta-provenance">${escapeHtml(c.source_version || "")}${c.text_hash ? ` · 文本哈希 ${escapeHtml(c.text_hash.slice(0, 12))}…` : ""}</div>
          ${evidenceText ? `<details class="cbeta-source-window">
            <summary>查看 CBETA 原文证据窗口</summary>
            ${evidenceRange ? `<div class="cbeta-source-range">${escapeHtml(evidenceRange)}</div>` : ""}
            <div class="cbeta-source-text">${escapeHtml(evidenceText)}</div>
          </details>` : ""}
        </article>`;
      }
      h += "</div>";
    }
    if (qw.length) h += `<div style="color:#d97706;font-size:0.85rem">Qwen：${qw.map((c) => c.text).join("、")}</div>`;
    h += "</div>";
  }
  el.innerHTML = h;
  $("m2BtnToC").style.display = "block";
}

async function m2DoConfirm() {
  const j = await fetch("/api/jobs/" + m2RunId).then((r) => r.json());
  const fs = j.fusions || [];
  const dmg = j.damages || [];
  let h = "";
  for (let i = 0; i < dmg.length; i++) {
    const d = dmg[i];
    const f = fs[i] || {};
    const did = d.damage_id;
    const ah = (f.autohdr_candidates || []).map((c) => c.text);
    const cb = (f.cbeta_candidates || []).map(cbetaText).filter(Boolean);
    const qw = (f.qwen_candidates || []).map((c) => c.text);
    const cur = m2Sel[did] || {};
    h += `<div class="report-card" style="margin:10px 0;padding:14px"><b>${did}</b><div style="margin:8px 0">`;
    for (const t of ah) h += `<div class="cand${cur.text == t && cur.source == "autohdr" ? " sel" : ""}" onclick="m2Pick('${did}','${t}','autohdr',this)"><span class="label ah">AutoHDR</span>${t}</div>`;
    for (const t of cb) h += `<div class="cand${cur.text == t && cur.source == "cbeta" ? " sel" : ""}" onclick="m2Pick('${did}','${t}','cbeta',this)"><span class="label cb">CBETA</span>${t}</div>`;
    for (const t of qw) h += `<div class="cand${cur.text == t && cur.source == "qwen" ? " sel" : ""}" onclick="m2Pick('${did}','${t}','qwen',this)"><span class="label qw">Qwen</span>${t}</div>`;
    h += `<div class="cand" onclick="m2Pick('${did}','','abstain',this)">🚫 不修复</div>`;
    h += `<input class="field-input" id="m2mt_${did}" placeholder="手动输入" value="${cur.source == "manual" ? cur.text : ""}" oninput="m2Pick('${did}',this.value,'manual',null)" style="margin-top:6px">`;
    h += "</div></div>";
  }
  $("m2cList").innerHTML = h;
  m2ShowStep(4);
}

function m2Pick(did, text, source, el) {
  if (el) {
    el.parentElement.querySelectorAll(".sel").forEach((e) => e.classList.remove("sel"));
    el.classList.add("sel");
  }
  m2Sel[did] = {
    text: text,
    source: source,
    status: source === "abstain" ? "abstain" : text ? "manual_confirmed" : "abstain",
  };
  localStorage.setItem("gurou_confirmed", JSON.stringify(m2Sel));
}

async function m2SubmitConfirm() {
  const decs = [];
  const validDamageIds = new Set((m2Job?.damages || []).map((d) => d.damage_id));
  for (const [did, s] of Object.entries(m2Sel)) {
    if (!validDamageIds.has(did)) continue;
    decs.push({
      damage_id: did,
      selected_text: s.text || "",
      selected_source: s.source || "abstain",
      status: s.status || "abstain",
    });
  }
  const resp = await fetch("/api/jobs/" + m2RunId + "/confirm", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ decisions: decs }),
  });
  const result = await resp.json();
  if (!resp.ok) throw new Error(result.detail || "确认失败");
  m2ShowStep(5);
  await m2DoRestore();
}

async function m2DoRestore() {
  const el = $("m2rRes");
  el.innerHTML = "<p>修复中…</p>";
  const resp = await fetch("/api/jobs/" + m2RunId + "/restore", { method: "POST" });
  const r = await resp.json();
  if (!resp.ok) throw new Error(r.detail || "修复失败");
  m2Job = { ...m2Job, ...r };
  renderM2Restore(r);
}

function renderM2Restore(r) {
  const el = $("m2rRes");
  const originalUrl = r.original_url || m2Job?.original_url || ("/api/jobs/" + m2RunId + "/artifacts/input.png");
  const overlayUrl = r.overlay_url || m2Job?.overlay_url || "";
  const restoredUrl = r.restored_url || m2Job?.restored_url || "";
  const mode = r.is_mock ? "Mock（演示）" : (r.is_replay ? "Replay（回放）" : "Official（未运行/待部署）");
  let h = `<p><b>状态：</b>${r.status === "completed" || r.status === "COMPLETED" ? "已完成" : (r.status || "已完成")} &nbsp;|&nbsp; <b>运行模式：</b>${mode}</p>`;
  h += `<div class="result-triptych">
    <div class="result-panel"><div class="result-panel-media"><img src="${originalUrl}" alt="原图"></div><div class="result-panel-caption">原图</div></div>
    <div class="result-panel"><div class="result-panel-media">${overlayUrl ? `<img src="${overlayUrl}" alt="缺损检测叠层">` : '<div class="result-empty">本次分析未生成检测叠层</div>'}</div><div class="result-panel-caption">缺损检测叠层（Overlay）</div></div>
    <div class="result-panel"><div class="result-panel-media">${restoredUrl ? `<img src="${restoredUrl}" alt="修复结果">` : '<div class="result-empty">暂无修复结果</div>'}</div><div class="result-panel-caption">修复结果</div></div>
  </div>`;
  h += `<div class="actions">
    ${restoredUrl ? '<button class="btn btn-primary" onclick="m2DownloadRestored()">下载结果图</button>' : ''}
    <button class="btn btn-primary" onclick="m2Download()">下载任务 ZIP</button>
    <button class="btn btn-gold" onclick="m2Report()">下载 JSON 报告</button>
    <button class="btn" style="background:rgba(28,22,18,0.12);color:var(--ink)" onclick="m2Reset()">重新开始</button>
  </div>`;
  el.innerHTML = h;
}

function m2DownloadRestored() {
  const a = document.createElement("a");
  a.href = m2Job?.restored_url || ("/api/jobs/" + m2RunId + "/artifacts/restored_mock.png");
  a.download = m2RunId + "_restored.png";
  a.click();
}

function m2Reset() {
  localStorage.removeItem("gurou_run");
  localStorage.removeItem("gurou_source");
  localStorage.removeItem("gurou_confirmed");
  location.reload();
}

async function m2RestoreSavedJob() {
  const saved = localStorage.getItem("gurou_run");
  if (!saved) return;
  const resp = await fetch("/api/jobs/" + saved);
  if (!resp.ok) { localStorage.removeItem("gurou_run"); return; }
  const job = await resp.json();
  m2RunId = saved;
  m2Job = job;
  m2InputSource = localStorage.getItem("gurou_source") || "单张残卷上传";
  $("m2CurRun").textContent = saved;
  $("m2Source").textContent = m2InputSource;
  if (job.status === "COMPLETED" && job.restored_url) {
    m2ShowStep(5);
    renderM2Restore(job);
  }
}

function m2Download() {
  fetch("/api/jobs/" + m2RunId + "/download").then((r) => r.blob()).then((b) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = m2RunId + ".zip";
    a.click();
  });
}
function m2Report() {
  fetch("/api/jobs/" + m2RunId + "/report").then((r) => r.blob()).then((b) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    a.download = m2RunId + "_report.json";
    a.click();
  });
}

function m2ShowStep(n) {
  for (let i = 1; i <= 5; i++) {
    const v = $("m2v" + i);
    if (v) {
      v.classList.toggle("hidden", i !== n);
      v.classList.toggle("visible", i === n);
    }
    const s = $("m2s" + i);
    if (s) { s.classList.remove("active"); if (i === n) s.classList.add("active"); }
  }
}

/* ===================== Init ===================== */
document.addEventListener("DOMContentLoaded", () => {
  // Module 1 bindings
  bindUpload("fileA", "previewA", "cardA", "A");
  bindUpload("fileB", "previewB", "cardB", "B");
  $("btnRejoin").addEventListener("click", runRejoin);
  $("btnRestore").addEventListener("click", runRestore);
  $("btnDownloadPanorama").addEventListener("click", downloadPanorama);
  $("btnSendToRestore").addEventListener("click", sendPanoramaToRestore);

  // Module 2 bindings
  $("m2FileInp").addEventListener("change", m2OnFile);
  $("m2BtnUp").addEventListener("click", m2DoUpload);

  // Health check (mock badge)
  m2Health();
  m2LoadRealCases();
  m2RestoreSavedJob();
  m2ShowStep(1);

  if (typeof gsap !== "undefined") {
    gsap.from(".hero-eyebrow", { opacity: 0, y: 16, duration: 0.7, delay: 0.1 });
    gsap.from(".hero h1", { opacity: 0, y: 24, duration: 0.9, delay: 0.2 });
    gsap.from(".hero-sub", { opacity: 0, y: 16, duration: 0.7, delay: 0.35 });
    gsap.from(".upload-card", { opacity: 0, y: 20, duration: 0.7, stagger: 0.12, delay: 0.5 });
  }
});
