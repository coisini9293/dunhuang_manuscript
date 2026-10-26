(() => {
  const root = document.documentElement;
  const reduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  root.classList.add('experience-motion');

  const revealTargets = [...document.querySelectorAll('.experience-reveal')];
  if (reduced || !('IntersectionObserver' in window)) {
    revealTargets.forEach((target) => target.classList.add('is-visible'));
  } else {
    const observer = new IntersectionObserver((entries, current) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add('is-visible');
        current.unobserve(entry.target);
      });
    }, { threshold: .1, rootMargin: '0px 0px -8% 0px' });
    revealTargets.forEach((target) => observer.observe(target));
  }

  const rejoinConsole = document.querySelector('[data-rejoin-console]');
  if (rejoinConsole) {
    const stage = rejoinConsole.querySelector('[data-rejoin-stage]');
    const viewButtons = [...rejoinConsole.querySelectorAll('[data-rejoin-view]')];
    const hint = rejoinConsole.querySelector('[data-rejoin-hint]');
    const status = rejoinConsole.querySelector('[data-rejoin-status]');
    const wipe = rejoinConsole.querySelector('[data-rejoin-wipe]');
    const reportToggle = rejoinConsole.querySelector('[data-report-toggle]');
    const report = document.querySelector('#rejoin-report');
    const scanButton = rejoinConsole.querySelector('[data-rejoin-scan]');
    const hints = {
      side: '先比较两片残卷的整体形制、栏线高度与墨色分布。',
      overlay: '叠层可以观察共同结构，但透明重合不能单独证明同源。',
      wipe: '拖动滑块，检查同一位置的断边和文字连续性。',
      edge: '校准线聚焦撕裂边缘的走向、缺口与纤维连续关系。',
      panorama: '完整图只作为保存案例的参考，不是本轮新模型输出。'
    };

    const setView = (view) => {
      stage.dataset.view = view;
      viewButtons.forEach((button) => {
        const active = button.dataset.rejoinView === view;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-selected', String(active));
      });
      if (hint) hint.textContent = hints[view] || hints.side;
    };

    viewButtons.forEach((button) => button.addEventListener('click', () => setView(button.dataset.rejoinView)));
    wipe?.addEventListener('input', () => stage.style.setProperty('--wipe-pos', `${wipe.value}%`));
    reportToggle?.addEventListener('click', () => {
      const open = report.hasAttribute('hidden');
      report.toggleAttribute('hidden', !open);
      reportToggle.setAttribute('aria-expanded', String(open));
      reportToggle.textContent = open ? '收起分析报告' : '展开分析报告';
    });
    scanButton?.addEventListener('click', () => {
      stage.classList.remove('is-scanning');
      window.requestAnimationFrame(() => stage.classList.add('is-scanning'));
    });

    window.addEventListener('rejoin-analysis-complete', () => {
      if (status) status.textContent = '本次机制演示已完成';
      rejoinConsole.classList.remove('is-highlighted');
      window.requestAnimationFrame(() => rejoinConsole.classList.add('is-highlighted'));
      stage.classList.remove('is-scanning');
      window.requestAnimationFrame(() => stage.classList.add('is-scanning'));
      document.querySelector('#rejoin-inspector')?.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
    });
  }

  const restorationFlow = document.querySelector('[data-restoration-flow]');
  if (restorationFlow) {
    const stepButtons = [...restorationFlow.querySelectorAll('[data-restoration-step]')];
    const panels = [...restorationFlow.querySelectorAll('[data-restoration-panel]')];
    const activateStep = (step, shouldScroll = false) => {
      stepButtons.forEach((button) => {
        const active = button.dataset.restorationStep === step;
        button.classList.toggle('is-active', active);
        if (active) button.setAttribute('aria-current', 'step');
        else button.removeAttribute('aria-current');
      });
      panels.forEach((panel) => {
        const active = panel.dataset.restorationPanel === step;
        panel.classList.toggle('is-active', active);
        panel.toggleAttribute('hidden', !active);
      });
      if (shouldScroll) restorationFlow.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
    };

    stepButtons.forEach((button) => button.addEventListener('click', () => activateStep(button.dataset.restorationStep)));

    const retrievalMode = restorationFlow.querySelector('[data-retrieval-mode]');
    const candidateStack = restorationFlow.querySelector('.candidate-stack');
    const candidateModeLabel = restorationFlow.querySelector('[data-candidate-mode-label]');
    const candidates = [...restorationFlow.querySelectorAll('[data-candidate]')];
    const selectedEvidence = restorationFlow.querySelector('[data-selected-evidence]');
    let selectedCandidate = candidates[0]?.dataset.candidate || '於';

    retrievalMode?.addEventListener('change', () => {
      const noise = retrievalMode.value === 'noise';
      candidateStack?.classList.toggle('show-noise', noise);
      if (candidateModeLabel) candidateModeLabel.textContent = noise ? 'Noise-aware 保存记录' : 'Baseline 保存记录';
    });

    candidates.forEach((candidate) => candidate.addEventListener('click', () => {
      candidates.forEach((item) => item.classList.toggle('is-selected', item === candidate));
      selectedCandidate = candidate.dataset.candidate;
      if (selectedEvidence) selectedEvidence.textContent = `当前选择：${selectedCandidate}`;
      const planText = restorationFlow.querySelector('[data-plan-text]');
      if (planText) planText.textContent = selectedCandidate;
    }));

    restorationFlow.querySelector('[data-send-review]')?.addEventListener('click', () => activateStep('review', true));

    const regionButtons = [...restorationFlow.querySelectorAll('[data-region]')];
    const regionLabel = restorationFlow.querySelector('[data-region-label]');
    const regionStatus = restorationFlow.querySelector('[data-region-status]');
    const regionCandidate = restorationFlow.querySelector('[data-region-candidate]');
    const regionCitation = restorationFlow.querySelector('[data-region-citation]');
    const reviewProgress = restorationFlow.querySelector('[data-review-progress]');
    const reviewProgressLabel = restorationFlow.querySelector('[data-review-progress-label]');
    const decisionButtons = [...restorationFlow.querySelectorAll('[data-decision]')];
    const decisionFeedback = restorationFlow.querySelector('[data-decision-feedback]');
    const reviewReason = restorationFlow.querySelector('[data-review-reason]');
    const reviewNote = restorationFlow.querySelector('[data-review-note]');
    let activeRegion = 0;
    const statuses = regionButtons.map((button) => button.dataset.status || 'abstained');
    const statusNames = { approved: '接受', edited: '人工修改', rejected: '拒绝', abstained: '弃权' };

    const updatePlan = () => {
      const approved = statuses.filter((statusValue) => statusValue === 'approved' || statusValue === 'edited').length;
      const rejected = statuses.filter((statusValue) => statusValue === 'rejected').length;
      const abstained = statuses.filter((statusValue) => statusValue === 'abstained').length;
      const approvedNode = restorationFlow.querySelector('[data-plan-approved]');
      const rejectedNode = restorationFlow.querySelector('[data-plan-rejected]');
      const abstainedNode = restorationFlow.querySelector('[data-plan-abstained]');
      if (approvedNode) approvedNode.textContent = approved;
      if (rejectedNode) rejectedNode.textContent = rejected;
      if (abstainedNode) abstainedNode.textContent = abstained;
    };

    const renderRegion = (index) => {
      activeRegion = index;
      regionButtons.forEach((button, buttonIndex) => {
        button.classList.toggle('is-active', buttonIndex === index);
        button.dataset.status = statuses[buttonIndex];
      });
      const current = regionButtons[index];
      const statusValue = statuses[index];
      if (!current) return;
      if (regionLabel) regionLabel.textContent = String(index).padStart(4, '0');
      if (regionStatus) regionStatus.textContent = `${statusNames[statusValue]} · ${statusValue === 'approved' ? '保存记录' : '人工裁决状态'}`;
      if (regionCandidate) regionCandidate.textContent = current.dataset.candidateText;
      if (regionCitation) regionCitation.textContent = current.dataset.citationText;
      if (reviewProgress) reviewProgress.style.width = `${((index + 1) / regionButtons.length) * 100}%`;
      if (reviewProgressLabel) reviewProgressLabel.textContent = `${index + 1} / ${regionButtons.length} 示例区域`;
      decisionButtons.forEach((button) => button.classList.toggle('is-selected', button.dataset.decision === statusValue));
    };

    regionButtons.forEach((button, index) => button.addEventListener('click', () => renderRegion(index)));
    decisionButtons.forEach((button) => button.addEventListener('click', () => {
      const decision = button.dataset.decision;
      statuses[activeRegion] = decision;
      regionButtons[activeRegion].dataset.status = decision;
      decisionButtons.forEach((item) => item.classList.toggle('is-selected', item === button));
      const reason = reviewReason?.value && reviewReason.value !== '不适用' ? `，理由：${reviewReason.value}` : '';
      const note = reviewNote?.value.trim() ? `；说明：${reviewNote.value.trim()}` : '';
      if (decisionFeedback) {
        decisionFeedback.textContent = `区域 ${String(activeRegion).padStart(4, '0')} 已标记为“${statusNames[decision]}”${reason}${note}`;
        decisionFeedback.classList.remove('is-stamped');
        window.requestAnimationFrame(() => decisionFeedback.classList.add('is-stamped'));
      }
      updatePlan();
      renderRegion(activeRegion);
    }));
    updatePlan();
    renderRegion(0);

    restorationFlow.querySelector('[data-build-plan]')?.addEventListener('click', () => activateStep('plan', true));
    restorationFlow.querySelector('[data-open-viewer]')?.addEventListener('click', () => activateStep('viewer', true));

    const viewport = restorationFlow.querySelector('[data-result-viewport]');
    const viewerPan = restorationFlow.querySelector('[data-viewer-pan]');
    const layerButtons = [...restorationFlow.querySelectorAll('[data-viewer-layer]')];
    const wipe = restorationFlow.querySelector('[data-result-wipe]');
    const opacity = restorationFlow.querySelector('[data-layer-opacity]');
    const opacityOutput = restorationFlow.querySelector('[data-opacity-output]');
    let zoom = 1;

    layerButtons.forEach((button) => button.addEventListener('click', () => {
      const layer = button.dataset.viewerLayer;
      viewport.dataset.layer = layer;
      layerButtons.forEach((item) => {
        const active = item === button;
        item.classList.toggle('is-active', active);
        item.setAttribute('aria-selected', String(active));
      });
    }));

    restorationFlow.querySelectorAll('[data-viewer-action]').forEach((button) => button.addEventListener('click', () => {
      const action = button.dataset.viewerAction;
      if (action === 'fit') zoom = 1;
      if (action === 'in') zoom = Math.min(2.4, zoom + .2);
      if (action === 'out') zoom = Math.max(.8, zoom - .2);
      if (action === 'flicker') {
        viewport.classList.toggle('is-flickering');
        button.classList.toggle('is-active', viewport.classList.contains('is-flickering'));
      }
      if (viewerPan) viewerPan.style.transform = `scale(${zoom})`;
    }));

    wipe?.addEventListener('input', () => viewport.style.setProperty('--result-wipe', `${wipe.value}%`));
    opacity?.addEventListener('input', () => {
      viewport.style.setProperty('--layer-opacity', String(Number(opacity.value) / 100));
      if (opacityOutput) opacityOutput.textContent = `${opacity.value}%`;
    });

    window.addEventListener('restoration-analysis-complete', () => activateStep('evidence', true));
  }

  const caseArchive = document.querySelector('[data-case-archive]');
  if (caseArchive) {
    const cards = [...caseArchive.querySelectorAll('[data-case-card]')];
    const focusType = caseArchive.querySelector('[data-case-focus-type]');
    const focusTitle = caseArchive.querySelector('[data-case-focus-title]');
    const focusDescription = caseArchive.querySelector('[data-case-focus-description]');
    const focusStatus = caseArchive.querySelector('[data-case-focus-status]');
    const selectCase = (card) => {
      cards.forEach((item) => item.classList.toggle('is-active', item === card));
      if (focusType) focusType.textContent = card.dataset.caseType;
      if (focusTitle) focusTitle.textContent = card.dataset.caseTitle;
      if (focusDescription) focusDescription.textContent = card.dataset.caseDescription;
      if (focusStatus) focusStatus.textContent = card.dataset.caseStatus;
    };
    cards.forEach((card) => card.addEventListener('click', () => selectCase(card)));
    if (cards[0]) selectCase(cards[0]);
  }
})();
