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
      : "如需图像修复，请展开高级选项并标注缺损区域";
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
    toast("缀合分析完成；如需图像修复，请展开高级选项标注缺损区域");
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
  toast("图像修复中，请稍候…");

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
    const backendLabel = data.is_replay
      ? "保存回放"
      : (data.is_mock ? "模拟结果" : "本地图像修复");
    toast(`${backendLabel} 修复完成`);
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

/* ===================== MODULE 2: formal server workflow ===================== */
const m2State = {
  apiBase: localStorage.getItem("gurou_server_api") || window.location.origin,
  caseId: localStorage.getItem("gurou_server_case") || null,
  fileId: localStorage.getItem("gurou_server_file") || null,
  localInputSha256: null,
  caseData: null,
  bundle: null,
  plan: null,
  restoreJobId: null,
  liveRestoreCompleted: false,
  system: null,
};
const m2Sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const m2Absolute = (path) => path ? (path.startsWith("http") ? path : m2State.apiBase + path) : "";

async function m2Api(path, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(m2State.apiBase + path, {
      ...options,
      signal: controller.signal,
      headers: { "X-Client-Version": "canonical-frontend-20260725", ...(options.headers || {}) },
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
    return payload;
  } catch (error) {
    if (error.name === "AbortError") throw new Error(`请求超时（${timeoutMs / 1000} 秒） / Request timeout`);
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function m2SetStatus(status) {
  $("m2CaseStatus").textContent = status || "—";
  $("m2CaseId").textContent = m2State.caseId || "—";
}

function m2SetService(name, label, state = "online") {
  const row = document.querySelector(`[data-live-service="${name}"]`);
  const value = $(`m2Svc${name.charAt(0).toUpperCase()}${name.slice(1)}`);
  if (row) row.dataset.state = state;
  if (value) value.textContent = label;
}

function m2FriendlyStatus(status) {
  const labels = {
    UPLOADED: "已上传，等待分析",
    ANALYSIS_RUNNING: "图像分析进行中",
    ANALYSIS_COMPLETE: "图像分析完成",
    HUMAN_REVIEW_PENDING: "等待人工审核",
    PLAN_READY: "修复方案已归档",
    RESTORE_RUNNING: "图像恢复进行中",
    RESTORE_COMPLETE: "已有保存的恢复结果",
  };
  return labels[status] || status || "—";
}

function m2RenderPlan(planResponse) {
  if (!planResponse) return;
  const plan = planResponse.plan || planResponse;
  const approved = plan.approved_regions || [];
  const rejected = plan.rejected_regions || [];
  const abstained = plan.abstained_regions || [];
  const edits = plan.manual_edits || [];
  const citations = plan.evidence_citations || [];
  const archive = $("m2PlanArchive");
  if (archive) {
    archive.innerHTML = `
      <div><span>案例编号</span><strong>${escapeHtml(plan.case_id || m2State.caseId || "—")}</strong></div>
      <div><span>方案编号</span><strong>${escapeHtml(planResponse.plan_id || "—")}</strong></div>
      <div><span>批准 / 修改</span><strong>${approved.length} / ${edits.length}</strong><small>仅这些区域可进入恢复</small></div>
      <div><span>拒绝 / 弃权</span><strong>${rejected.length} / ${abstained.length}</strong><small>保持原始像素</small></div>
      <div><span>确认文字</span><strong>${escapeHtml(approved.map((item) => item.selected_text).join("、") || "无")}</strong><small>人工提交，不由模型自动采纳</small></div>
      <div><span>佛典证据</span><strong>${citations.length} 条</strong><small>${escapeHtml(plan.rag_version || "—")}</small></div>
      <div class="plan-wide"><span>方案哈希</span><strong>${escapeHtml(plan.plan_sha256 || "—")}</strong><small>候选包 ${escapeHtml((plan.candidate_bundle_sha256 || "—").slice(0, 20))}…</small></div>
      <div class="plan-wide"><span>创建时间</span><strong>${escapeHtml(plan.created_at || planResponse.created_at || "—")}</strong><small>复核者：${escapeHtml(plan.actor || "—")}</small></div>`;
  }
  $("m2PlanSummary").textContent = JSON.stringify(planResponse, null, 2);
  m2SetService("plan", "可用 · 已归档");
}

function m2ShowStep(n) {
  for (let i = 1; i <= 5; i++) {
    const v = $("m2v" + i);
    if (v) {
      v.classList.toggle("hidden", i !== n);
      v.classList.toggle("visible", i === n);
    }
    const s = $("m2s" + i);
    if (s) s.classList.toggle("active", i <= n);
  }
}

async function m2Connect() {
  m2State.apiBase = $("m2ApiBase").value.trim().replace(/\/+$/, "");
  localStorage.setItem("gurou_server_api", m2State.apiBase);
  const lamp = $("m2ServerLamp");
  lamp.className = "server-lamp";
  lamp.textContent = "检查中…";
  try {
    const [health, rag, system, rejoin] = await Promise.all([
      m2Api("/api/health"),
      m2Api("/api/rag/health"),
      m2Api("/api/system/status"),
      m2Api("/api/system/health", {}, 4000)
        .then((response) => response.ok ? response.json() : null)
        .catch(() => null),
    ]);
    m2State.system = system;
    lamp.classList.add("pass");
    lamp.textContent = `${health.status || "ok"} · 实时服务器`;
    $("m2LiveStamp").textContent = "实时 · 已连接";
    $("m2LiveStamp").classList.remove("offline");
    $("m2RagStatus").textContent = `${rag.sutra_count?.toLocaleString?.() || rag.sutra_count || 0} 部 · ${rag.index_version || "—"} · 向量关闭`;
    $("m2CorpusScope").textContent = `${rag.corpus_version || "—"} / ${rag.record_count?.toLocaleString?.() || "—"} 条`;
    $("m2GpuStatus").textContent = system.gpu?.available ? "已开启" : "暂未开启（正常阶段）";
    m2SetService("rejoin", rejoin?.status === "ok" ? "在线" : "当前未连接", rejoin?.status === "ok" ? "online" : "offline");
    m2SetService("rag", "在线 · 真实索引");
    m2SetService("review", "在线");
    m2SetService("plan", "可用");
    m2SetService("gpu", system.gpu?.available ? "已开启" : "暂未开启", system.gpu?.available ? "online" : "waiting");
    m2SetService("replay", "已验证回放");
    const notice = $("m2GpuOffNotice");
    if (notice) {
      notice.classList.toggle("ready", Boolean(system.gpu?.available));
      notice.innerHTML = system.gpu?.available
        ? "<i></i><div><b>图像恢复服务已就绪</b><span>只有已归档且允许恢复的修复方案才能启动正式图像恢复。</span></div>"
        : "<i></i><div><b>图像恢复服务当前未开启</b><span>文献检索、人工审核与修复计划仍可使用。最终图像恢复将在服务可用后执行。</span></div>";
    }
    if (m2State.caseId) await m2RecoverCase();
  } catch (error) {
    lamp.classList.add("fail");
    lamp.textContent = "服务离线 · 可使用回放";
    $("m2LiveStamp").textContent = "实时 · 离线";
    $("m2LiveStamp").classList.add("offline");
    ["rag", "review", "plan"].forEach((name) => m2SetService(name, "暂时离线", "offline"));
    m2SetService("gpu", "暂未开启", "waiting");
    toast("全量文献服务尚未连接；当前仍可使用本地缀合与历史案例回放");
  }
}

async function m2OnFile() {
  const file = $("m2FileInp").files[0];
  if (!file) return;
  $("m2Placeholder").style.display = "none";
  $("m2Preview").style.display = "block";
  $("m2Preview").src = URL.createObjectURL(file);
  $("m2UpInfo").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB · 正在计算 SHA-256…`;
  const digest = await crypto.subtle.digest("SHA-256", await file.arrayBuffer());
  m2State.localInputSha256 = [...new Uint8Array(digest)]
    .map((value) => value.toString(16).padStart(2, "0"))
    .join("");
  $("m2UpInfo").textContent = `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB · SHA-256 ${m2State.localInputSha256}`;
  $("m2BtnUp").disabled = false;
}

async function m2DoUpload() {
  const file = $("m2FileInp").files[0];
  if (!file) return;
  $("m2BtnUp").disabled = true;
  try {
    const form = new FormData();
    form.append("image", file);
    const data = await m2Api("/api/cases", { method: "POST", body: form }, 120000);
    m2State.caseId = data.case_id;
    m2State.fileId = data.file_id || data.files?.[0]?.file_id || null;
    m2State.caseData = data;
    localStorage.setItem("gurou_server_case", m2State.caseId);
    if (m2State.fileId) localStorage.setItem("gurou_server_file", m2State.fileId);
    m2SetStatus(data.status);
    m2ShowStep(2);
    const hashPass = Boolean(
      m2State.localInputSha256
      && data.input_sha256
      && m2State.localInputSha256 === data.input_sha256
    );
    $("m2aRes").innerHTML = `<div class="operation-note hash-chain ${hashPass ? "pass" : "fail"}"><b>输入哈希链：${hashPass ? "通过" : "失败"}</b><span>缀合全景图：${escapeHtml(m2State.localInputSha256 || "—")}</span><span>上传文件：${escapeHtml(data.input_sha256 || "—")}</span>${data.deduplicated ? "<small>服务器识别到相同像素文件；本次仍建立了新的案例。</small>" : ""}</div>`;
    if (!hashPass) throw new Error("INPUT_HASH_CHAIN_MISMATCH");
  } catch (error) {
    toast(`上传失败：${error.message}`);
    $("m2BtnUp").disabled = false;
  }
}

async function m2RefreshCase() {
  if (!m2State.caseId) return null;
  const data = await m2Api(`/api/cases/${m2State.caseId}`);
  m2State.caseData = data;
  m2State.fileId = m2State.fileId || data.files?.[0]?.file_id || null;
  m2SetStatus(data.status);
  return data;
}

async function m2DoAnalyze() {
  if (!m2State.caseId) return;
  $("m2BtnAnalyze").disabled = true;
  try {
    const job = await m2Api(`/api/cases/${m2State.caseId}/analyze`, { method: "POST" });
    $("m2aRes").innerHTML = `<div class="operation-note">正式分析运行中 · 任务 ${escapeHtml(job.run_id)} · 进程 ${escapeHtml(job.worker_pid)}</div>`;
    for (let attempt = 0; attempt < 240; attempt++) {
      await m2Sleep(5000);
      const current = await m2RefreshCase();
      if (current.status === "ANALYSIS_COMPLETE") {
        await m2LoadBundle();
        m2ShowStep(3);
        toast("正式分析完成，进入佛典证据检索。");
        return;
      }
      if (current.status === "ANALYSIS_FAILED") throw new Error(current.error_detail || "正式分析失败");
      if (attempt % 6 === 5) $("m2aRes").innerHTML = `<div class="operation-note">正式分析运行中 · 已等待 ${Math.round((attempt + 1) * 5 / 60)} 分钟</div>`;
    }
    throw new Error("分析轮询达到 20 分钟上限，可稍后点击刷新状态");
  } catch (error) {
    toast(`分析未完成：${error.message}`);
    $("m2BtnAnalyze").disabled = false;
  }
}

async function m2LoadBundle() {
  m2State.bundle = await m2Api(`/api/cases/${m2State.caseId}/candidate-bundle`);
  m2RenderRegions();
  return m2State.bundle;
}

function m2RenderCandidate(candidate) {
  const scores = candidate.retrieval_components || {};
  const noise = candidate.noise_aware_rerank || {};
  return `<article class="evidence-card">
    <div class="live-candidate-head"><span class="live-stamp">实时</span><div><b>候选 ${escapeHtml(candidate.rank)}</b> · <strong>${escapeHtml(candidate.candidate_text || "仅定位，不补字")}</strong></div><span class="candidate-state">${escapeHtml(candidate.decision_status || "候选")}</span></div>
    <div class="candidate-meta">${escapeHtml(candidate.sutra_id || "经号未记录")} · ${escapeHtml(candidate.canonical_title || "经名未记录")} · ${escapeHtml(candidate.translator || "译者不详")} · 卷${escapeHtml(candidate.volume ?? "—")} · ${escapeHtml(candidate.edition || "—")}本</div>
    <div class="citation">${escapeHtml(candidate.citation || "引用不完整")}</div>
    <div class="candidate-context">${escapeHtml(candidate.matched_before || "")}<mark>${escapeHtml(candidate.inferred_gap || "")}</mark>${escapeHtml(candidate.matched_after || "")}</div>
    <div class="evidence-audit">综合 ${scorePct(scores.final)} · 字符 ${scorePct(scores.ngram)} · 词频匹配 ${scorePct(scores.bm25)} · 连续性 ${scorePct(scores.continuity)}${noise.final !== undefined ? ` · 噪声重排 ${scorePct(noise.final)} · 来源查询 ${candidate.noise_aware_origins?.length || 0}` : ""} · 文本哈希 ${escapeHtml(candidate.text_hash?.slice(0, 12) || "")}</div>
  </article>`;
}

function m2RenderRegions() {
  const regions = m2State.bundle?.regions || [];
  if (!regions.length) {
    $("m2Regions").innerHTML = '<div class="operation-note">未检测到可处理区域</div>';
    return;
  }
  $("m2Regions").innerHTML = regions.map((region, index) => {
    const latest = region.rag_queries?.[region.rag_queries.length - 1];
    const candidates = latest?.candidates || [];
    const before = region.ocr_before || region.ocr_contexts?.[0]?.context_before || "";
    const after = region.ocr_after || region.ocr_contexts?.[0]?.context_after || "";
    const mode = latest?.request?.retrieval_mode || "尚未检索";
    const latency = latest?.latency_ms;
    const decisionStatus = latest?.decision_status;
    return `<article class="region-workbench" data-region="${escapeHtml(region.region_id)}">
      <div class="region-head"><b>区域 ${index + 1}</b><span>${escapeHtml(region.region_id)} · ${escapeHtml(mode)}${latency !== undefined ? ` · ${Number(latency).toFixed(1)} 毫秒` : ""}${decisionStatus ? ` · ${escapeHtml(decisionStatus)}` : ""}</span></div>
      <div class="region-media">
        ${region.crop_url ? `<a href="${m2Absolute(region.crop_url)}" target="_blank"><img src="${m2Absolute(region.crop_url)}" alt="区域裁图"></a>` : "<div>无裁图</div>"}
        ${region.mask_url ? `<a href="${m2Absolute(region.mask_url)}" target="_blank"><img src="${m2Absolute(region.mask_url)}" alt="区域遮罩"></a>` : "<div>无遮罩</div>"}
        <div class="context-grid">
          <label>左侧可见文字<textarea class="field-input" data-field="before">${escapeHtml(before)}</textarea></label>
          <label>右侧可见文字<textarea class="field-input" data-field="after">${escapeHtml(after)}</textarea></label>
        </div>
      </div>
      <div data-evidence>${candidates.map(m2RenderCandidate).join("") || (latest ? '<div class="operation-note">证据不足：暂不裁决</div>' : '<div class="operation-note">尚未检索</div>')}</div>
    </article>`;
  }).join("");
}

async function m2SearchAll() {
  const cards = [...document.querySelectorAll("#m2Regions [data-region]")];
  const mode = document.querySelector('input[name="m2RetrievalMode"]:checked').value;
  $("m2BtnSearch").disabled = true;
  const started = performance.now();
  $("m2LiveQuerySummary").textContent = `${mode === "noise_aware" ? "噪声增强" : "基线"} 实时检索进行中；候选将保留完整佛典引用。`;
  try {
    let candidateCount = 0;
    let ambiguousCount = 0;
    let abstainCount = 0;
    for (let index = 0; index < cards.length; index++) {
      const card = cards[index];
      const contextBefore = card.querySelector('[data-field="before"]').value.trim();
      const contextAfter = card.querySelector('[data-field="after"]').value.trim();
      card.querySelector("[data-evidence]").innerHTML = `<div class="operation-note">检索中 ${index + 1}/${cards.length}…</div>`;
      if (!contextBefore && !contextAfter) continue;
      const result = await m2Api(`/api/cases/${m2State.caseId}/rag/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          region_id: card.dataset.region,
          context_before: contextBefore,
          context_after: contextAfter,
          expected_missing_length: 1,
          top_k: 5,
          retrieval_mode: mode,
        }),
      }, mode === "noise_aware" ? 180000 : 60000);
      candidateCount += result.candidates?.length || 0;
      if (String(result.decision_status).includes("ambiguous")) ambiguousCount++;
      if (result.decision_status === "abstain") abstainCount++;
      card.querySelector("[data-evidence]").innerHTML = result.candidates?.map(m2RenderCandidate).join("") || '<div class="operation-note">证据不足：暂不裁决</div>';
    }
    await m2LoadBundle();
    const elapsed = performance.now() - started;
    $("m2LiveQuerySummary").innerHTML = `<span class="live-stamp">实时</span> ${mode === "noise_aware" ? "噪声增强" : "基线"} · 前 ${candidateCount} 条 · 歧义 ${ambiguousCount} · 暂不裁决 ${abstainCount} · 请求耗时 ${elapsed.toFixed(1)} 毫秒`;
    toast(`检索完成：${mode}`);
  } catch (error) {
    $("m2LiveQuerySummary").textContent = `实时检索未完成：${error.message}。已保存回放不受影响。`;
    toast(`检索失败：${error.message}`);
  } finally {
    $("m2BtnSearch").disabled = false;
  }
}

function m2OpenReview() {
  const regions = m2State.bundle?.regions || [];
  $("m2ReviewCards").innerHTML = regions.map((region, index) => {
    const candidates = (region.rag_queries || []).flatMap((query) => query.candidates || []);
    const options = candidates.map((candidate) =>
      `<option value="${escapeHtml(candidate.candidate_id)}">${escapeHtml(candidate.candidate_text || "仅定位")} · ${escapeHtml(candidate.citation || "")}</option>`
    ).join("");
    return `<article class="review-card" data-review-region="${escapeHtml(region.region_id)}">
      <div class="region-head"><b>区域 ${index + 1}</b><span>${escapeHtml(region.region_id)}</span></div>
      <div class="region-review-media">
        ${region.crop_url ? `<a href="${m2Absolute(region.crop_url)}" target="_blank"><img src="${m2Absolute(region.crop_url)}" alt="审核区域裁图"></a>` : "<div>无裁图</div>"}
        ${region.mask_url ? `<a href="${m2Absolute(region.mask_url)}" target="_blank"><img src="${m2Absolute(region.mask_url)}" alt="审核区域遮罩"></a>` : "<div>无遮罩</div>"}
      </div>
      <div class="review-controls">
        <label>决定<select class="field-input" data-field="decision">
          <option value="">请选择</option><option value="accept">接受佛典候选</option>
          <option value="manual_modify">人工修改</option><option value="reject">拒绝</option><option value="abstain">弃权</option>
        </select></label>
        <label>证据候选<select class="field-input" data-field="candidate"><option value="">不选择</option>${options}</select></label>
        <label>单字修改<input class="field-input" data-field="manual" maxlength="1"></label>
        <label>说明<input class="field-input" data-field="note" placeholder="人工修改时必填"></label>
      </div>
    </article>`;
  }).join("");
  m2ShowStep(4);
}

function m2AbstainRemaining() {
  const pending = [...document.querySelectorAll('#m2ReviewCards [data-field="decision"]')]
    .filter((select) => !select.value);
  pending.forEach((select) => { select.value = "abstain"; });
  toast(`已将 ${pending.length} 个未决区域设为弃权；尚未提交。`);
}

async function m2SubmitDecisions() {
  const reviewer = $("m2Reviewer").value.trim();
  if (!reviewer) return toast("请填写复核者");
  const regions = m2State.bundle?.regions || [];
  const cards = [...document.querySelectorAll("[data-review-region]")];
  try {
    const decisions = cards.map((card) => {
      const region = regions.find((item) => item.region_id === card.dataset.reviewRegion);
      const decision = card.querySelector('[data-field="decision"]').value;
      if (!decision) throw new Error(`区域 ${card.dataset.reviewRegion} 尚未选择决定`);
      const candidateId = card.querySelector('[data-field="candidate"]').value;
      const candidate = (region?.rag_queries || []).flatMap((query) => query.candidates || []).find((item) => item.candidate_id === candidateId);
      const manual = card.querySelector('[data-field="manual"]').value.trim();
      return {
        region_id: card.dataset.reviewRegion,
        decision,
        selected_text: decision === "accept" ? (candidate?.candidate_text || "") : decision === "manual_modify" ? manual : "",
        selected_source: decision === "accept" ? "cbeta" : decision === "manual_modify" ? "manual_edit" : "",
        candidate_id: decision === "accept" ? candidateId : "",
        citation_hash: decision === "accept" ? (candidate?.citation_hash || "") : "",
        note: card.querySelector('[data-field="note"]').value.trim(),
      };
    });
    const response = await m2Api(`/api/cases/${m2State.caseId}/decisions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        decisions,
        reviewer,
        candidate_bundle_sha256: m2State.bundle.candidate_bundle_sha256,
        request_id: crypto.randomUUID(),
      }),
    });
    m2State.plan = await m2Api(`/api/cases/${m2State.caseId}/repair-plan`);
    m2RenderPlan(m2State.plan);
    $("m2DecisionFeedback").classList.add("confirmed");
    $("m2DecisionFeedback").textContent = response.idempotent
      ? "相同请求已安全识别为重复提交；既有决定与方案保持不变。"
      : `朱砂批注已归档。方案 ${response.plan_id} 已生成，刷新页面后仍可恢复。`;
    $("m2BtnRestore").disabled = !response.allow_restore;
    m2ShowStep(5);
    if (m2State.system?.gpu?.available) {
      await m2LoadResult();
    } else {
      $("m2rRes").innerHTML = '<div class="gpu-off-result" data-status="GPU_REQUIRED"><b>修复方案已经生成</b><span>图像恢复服务当前未开启；没有创建恢复任务，审核状态与修复方案均已保留。</span></div>';
    }
  } catch (error) {
    toast(`决定未提交：${error.message}`);
  }
}

async function m2StartRestore() {
  $("m2BtnRestore").disabled = true;
  try {
    const system = await m2Api("/api/system/status", {}, 8000);
    m2State.system = system;
    if (!system.gpu?.available) {
      $("m2BtnRestore").dataset.gpuOffHandled = "true";
      $("m2BtnRestore").textContent = "图像恢复服务开启后继续";
      $("m2rRes").innerHTML = `<div class="gpu-off-result" data-status="GPU_REQUIRED"><b>图像恢复服务当前未开启</b><span>文献检索、人工审核与修复计划仍可使用。方案 ${escapeHtml(m2State.plan?.plan_id || "—")} 已保留，没有创建恢复任务。</span></div>`;
      m2SetService("gpu", "暂未开启", "waiting");
      toast("修复方案已保留；图像恢复服务开启后可继续执行最终图像恢复。");
      return;
    }
    const job = await m2Api(`/api/cases/${m2State.caseId}/restore`, { method: "POST" });
    m2State.restoreJobId = job.job_id;
    m2State.liveRestoreCompleted = false;
    $("m2rRes").innerHTML = `<div class="operation-note"><span class="live-stamp">实时 · 已排队</span> 正式修复已入队 · 任务 ${escapeHtml(job.job_id)} · 进程 ${escapeHtml(job.worker_pid || "—")}</div>`;
    let lastPhase = "QUEUED";
    for (let attempt = 0; attempt < 240; attempt++) {
      await m2Sleep(5000);
      const current = await m2Api(`/api/jobs/${job.job_id}`);
      const phase = current.phase || current.status || "RUNNING";
      if (phase !== lastPhase) {
        lastPhase = phase;
        $("m2rRes").innerHTML = `<div class="operation-note"><span class="live-stamp">实时 · ${escapeHtml(phase)}</span> ${escapeHtml(current.phase_detail || "正式图像恢复正在执行")} · 任务 ${escapeHtml(job.job_id)}</div>`;
      }
      if (current.status === "COMPLETE") {
        m2State.liveRestoreCompleted = true;
        await m2LoadResult();
        toast("正式约束修复完成；结果仍待人工验收。");
        return;
      }
      if (current.status === "FAILED") throw new Error(current.detail_json || "正式修复失败");
    }
    throw new Error("修复轮询达到 20 分钟上限，可稍后刷新结果");
  } catch (error) {
    if (String(error.message).includes("GPU_REQUIRED")) {
      $("m2BtnRestore").dataset.gpuOffHandled = "true";
      $("m2BtnRestore").textContent = "图像恢复服务开启后继续";
      $("m2rRes").innerHTML = '<div class="gpu-off-result" data-status="GPU_REQUIRED"><b>图像恢复服务当前未开启</b><span>修复计划仍已安全保存；本次没有创建图像恢复任务。</span></div>';
      toast("图像恢复服务当前未开启；修复计划没有丢失。");
      return;
    }
    toast(`修复未完成：${error.message}`);
    $("m2BtnRestore").disabled = !(m2State.plan?.allow_restore);
  }
}

async function m2LoadResult() {
  if (!m2State.caseId) return;
  const [caseData, result] = await Promise.all([
    m2RefreshCase(),
    m2Api(`/api/cases/${m2State.caseId}/result?fresh=${Date.now()}`),
  ]);
  const fileId = m2State.fileId || caseData.files?.[0]?.file_id;
  const output = result.outputs?.find((item) => item.relative_path.endsWith("/constrained_restored.png"))
    || result.outputs?.[result.outputs.length - 1];
  const original = fileId ? m2Absolute(`/api/cases/${m2State.caseId}/files/${fileId}`) : "";
  const overlay = m2Absolute(`/api/cases/${m2State.caseId}/analysis-overlay`);
  const restored = output ? m2Absolute(output.url) : "";
  const hasSavedOutput = Boolean(restored);
  const resultBadge = hasSavedOutput && m2State.liveRestoreCompleted
    ? '<span class="live-stamp">实时 · 已完成</span>'
    : hasSavedOutput
      ? '<span class="replay-label">已验证回放 · 已保存服务器结果</span>'
      : '<span class="live-stamp">实时 · 等待图像恢复服务</span>';
  $("m2rRes").innerHTML = `<div class="operation-note">${resultBadge} 状态：${escapeHtml(m2FriendlyStatus(caseData.status))} · 人工视觉验收：${escapeHtml(result.human_review_status === "not_reviewed" ? "尚未完成" : result.human_review_status || "尚未完成")}</div>
    <div class="result-triptych">
      <div class="result-panel"><div class="result-panel-media">${original ? `<a href="${original}" target="_blank"><img src="${original}" alt="原图"></a>` : '<div class="result-empty">无原图</div>'}</div><div class="result-panel-caption">原图</div></div>
      <div class="result-panel"><div class="result-panel-media"><a href="${overlay}" target="_blank"><img src="${overlay}" alt="缺损检测叠层"></a></div><div class="result-panel-caption">缺损检测叠层</div></div>
      <div class="result-panel"><div class="result-panel-media">${restored ? `<a href="${restored}" target="_blank"><img src="${restored}" alt="约束修复结果"></a>` : '<div class="result-empty">图像恢复服务暂未开启<br><small>修复计划已保留，尚未生成新结果</small></div>'}</div><div class="result-panel-caption">${restored ? (m2State.liveRestoreCompleted ? "本次实时约束修复" : "已保存约束修复") : "等待图像恢复服务"}</div></div>
    </div>${restored ? `<a class="btn btn-primary" href="${restored}" download>下载修复图</a>` : ""}`;
}

async function m2DownloadAudit() {
  if (!m2State.caseId) return;
  const audit = await m2Api(`/api/cases/${m2State.caseId}/audit`);
  const blob = new Blob([JSON.stringify(audit, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = `${m2State.caseId}-audit.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

async function m2RecoverCase() {
  try {
    const data = await m2RefreshCase();
    if (data.status === "UPLOADED" || data.status === "ANALYSIS_RUNNING" || data.status === "ANALYSIS_FAILED") return m2ShowStep(2);
    await m2LoadBundle();
    if (data.status === "PLAN_READY" || data.status === "RESTORE_RUNNING" || data.status === "RESTORE_COMPLETE") {
      m2State.plan = data.repair_plan;
      m2RenderPlan(m2State.plan);
      $("m2BtnRestore").disabled = !m2State.plan?.allow_restore || data.status === "RESTORE_COMPLETE";
      m2ShowStep(5);
      if (data.status === "RESTORE_COMPLETE") {
        await m2LoadResult();
      } else {
        $("m2rRes").innerHTML = '<div class="gpu-off-result" data-status="GPU_REQUIRED"><b>修复方案已从服务器恢复</b><span>图像恢复服务当前未开启；可以继续检索、审核或下载审计，不会重复创建恢复任务。</span></div>';
      }
    } else {
      m2ShowStep(3);
    }
  } catch (error) {
    localStorage.removeItem("gurou_server_case");
    toast(`无法恢复服务器案例：${error.message}`);
  }
}

async function m2LoadCaseById() {
  const caseId = $("m2CaseLookup").value.trim();
  if (!/^[0-9a-f-]{36}$/i.test(caseId)) return toast("请输入有效的案例编号。");
  m2State.caseId = caseId;
  m2State.fileId = null;
  localStorage.setItem("gurou_server_case", caseId);
  localStorage.removeItem("gurou_server_file");
  try {
    const data = await m2RefreshCase();
    await m2LoadBundle();
    if (data.status === "PLAN_READY" || data.status === "RESTORE_RUNNING" || data.status === "RESTORE_COMPLETE") {
      m2State.plan = data.repair_plan || await m2Api(`/api/cases/${caseId}/repair-plan`);
      m2RenderPlan(m2State.plan);
    }
    m2ShowStep(3);
    toast(`实时案例已载入：${caseId.slice(0, 8)}…`);
  } catch (error) {
    toast(`案例载入失败：${error.message}`);
  }
}

function m2NewCase() {
  m2State.caseId = null;
  m2State.fileId = null;
  m2State.caseData = null;
  m2State.bundle = null;
  m2State.plan = null;
  m2State.restoreJobId = null;
  localStorage.removeItem("gurou_server_case");
  localStorage.removeItem("gurou_server_file");
  $("m2FileInp").value = "";
  $("m2Preview").removeAttribute("src");
  $("m2Preview").style.display = "none";
  $("m2Placeholder").style.display = "";
  $("m2UpInfo").textContent = "未选择图像";
  $("m2BtnUp").disabled = true;
  m2SetStatus("尚未创建 / Not created");
  m2ShowStep(1);
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
  $("m2Connect").addEventListener("click", m2Connect);
  $("m2LoadCase").addEventListener("click", m2LoadCaseById);
  $("m2NewCase").addEventListener("click", m2NewCase);
  $("m2BtnAnalyze").addEventListener("click", m2DoAnalyze);
  $("m2BtnRefreshCase").addEventListener("click", async () => {
    try {
      const current = await m2RefreshCase();
      if (current?.status === "ANALYSIS_COMPLETE") {
        await m2LoadBundle();
        m2ShowStep(3);
      }
    } catch (error) { toast(`刷新失败：${error.message}`); }
  });
  $("m2BtnSearch").addEventListener("click", m2SearchAll);
  $("m2BtnToReview").addEventListener("click", m2OpenReview);
  $("m2BtnAbstainRest").addEventListener("click", m2AbstainRemaining);
  $("m2BtnDecide").addEventListener("click", m2SubmitDecisions);
  $("m2BtnRestore").addEventListener("click", m2StartRestore);
  $("m2BtnResult").addEventListener("click", () => m2LoadResult().catch((error) => toast(error.message)));
  $("m2BtnAudit").addEventListener("click", () => m2DownloadAudit().catch((error) => toast(error.message)));
  $("m2ApiBase").value = m2State.apiBase;
  m2ShowStep(1);
  m2Connect();

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  $("loadReplay")?.addEventListener("click", () => {
    $("verifiedReplay").scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth" });
  });
  const lightbox = $("artifactLightbox");
  document.querySelectorAll(".replay-gallery img").forEach((image) => image.addEventListener("click", () => {
    $("lightboxImage").src = image.src;
    $("lightboxImage").alt = image.alt;
    lightbox.classList.add("open");
  }));
  const closeLightbox = () => lightbox.classList.remove("open");
  $("closeLightbox")?.addEventListener("click", closeLightbox);
  lightbox?.addEventListener("click", (event) => { if (event.target === lightbox) closeLightbox(); });
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeLightbox(); });
});
