/**
 * TestPilot AI - Client Workspace Controller
 */

let currentResults = null;
let currentFilter = 'all';

document.addEventListener('DOMContentLoaded', function () {
  initSourceToggle();
  initBudgetToggle();
  initProjectBuildCascade();
  initStrategyToggle();
  initPrioritizeAction();
});

// 1. Source toggle: Benchmark vs Upload
function initSourceToggle() {
  const radioBenchmark = document.getElementById('source_benchmark');
  const radioUpload = document.getElementById('source_upload');
  const benchmarkInputs = document.getElementById('benchmarkInputs');
  const uploadInputs = document.getElementById('uploadInputs');

  radioBenchmark.addEventListener('change', function () {
    if (this.checked) {
      benchmarkInputs.style.display = 'block';
      uploadInputs.style.display = 'none';
    }
  });

  radioUpload.addEventListener('change', function () {
    if (this.checked) {
      benchmarkInputs.style.display = 'none';
      uploadInputs.style.display = 'block';
    }
  });

  const csvInput = document.getElementById('csvFileInput');
  if (csvInput) {
    csvInput.addEventListener('change', handleCsvUpload);
  }
}

// 2. Budget toggle & Alpha slider
function initBudgetToggle() {
  const budgetRadios = document.querySelectorAll('input[name="budget_preset"]');
  const customContainer = document.getElementById('customBudgetContainer');
  const customInput = document.getElementById('customBudgetSeconds');
  const budgetDisplay = document.getElementById('budgetTextDisplay');

  function updateBudgetLabel(val) {
    const mins = (val / 60).toFixed(1);
    budgetDisplay.textContent = `${val}s (${mins}m)`;
  }

  budgetRadios.forEach(radio => {
    radio.addEventListener('change', function () {
      if (this.value === 'custom') {
        customContainer.style.display = 'block';
        updateBudgetLabel(customInput.value);
      } else {
        customContainer.style.display = 'none';
        updateBudgetLabel(this.value);
      }
    });
  });

  if (customInput) {
    customInput.addEventListener('input', function () {
      updateBudgetLabel(this.value);
    });
  }

  const alphaSlider = document.getElementById('alphaSlider');
  const alphaBadge = document.getElementById('alphaValBadge');
  if (alphaSlider && alphaBadge) {
    alphaSlider.addEventListener('input', function () {
      alphaBadge.textContent = Number(this.value).toFixed(1);
    });
  }
}

// 3. Project & Build Cascade
function initProjectBuildCascade() {
  const projSelect = document.getElementById('projectSelect');
  if (!projSelect) return;

  projSelect.addEventListener('change', function () {
    loadBuildsForProject(this.value);
  });

  const buildSelect = document.getElementById('buildSelect');
  if (buildSelect) {
    buildSelect.addEventListener('change', function () {
      updateBuildMetadata(projSelect.value, this.value);
    });
  }

  // Initial load
  if (projSelect.value) {
    loadBuildsForProject(projSelect.value);
  }
}

function loadBuildsForProject(projectId) {
  const buildSelect = document.getElementById('buildSelect');
  buildSelect.innerHTML = '<option disabled>Loading builds...</option>';

  fetch(`/api/benchmark-builds/${encodeURIComponent(projectId)}`)
    .then(r => r.json())
    .then(data => {
      buildSelect.innerHTML = '';
      if (data.builds && data.builds.length > 0) {
        data.builds.forEach((b, idx) => {
          const opt = document.createElement('option');
          opt.value = b;
          opt.textContent = `Build #${b}`;
          if (idx === data.builds.length - 1) {
            opt.selected = true; // Select latest build by default
          }
          buildSelect.appendChild(opt);
        });
        updateBuildMetadata(projectId, buildSelect.value);
      } else {
        buildSelect.innerHTML = '<option disabled>No builds found</option>';
      }
    })
    .catch(err => {
      buildSelect.innerHTML = '<option disabled>Error loading builds</option>';
      console.error(err);
    });
}

function updateBuildMetadata(projectId, buildId) {
  const summaryBox = document.getElementById('buildSummaryText');
  summaryBox.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Fetching build info...';

  fetch(`/api/benchmark-build-info/${encodeURIComponent(projectId)}/${buildId}`)
    .then(r => r.json())
    .then(info => {
      const durMins = (info.total_duration / 60).toFixed(1);
      let verdictNote = '';
      if (info.failures_count !== undefined && info.failures_count !== null) {
        verdictNote = ` · <strong class="text-danger">${info.failures_count} failure(s)</strong>`;
      }
      summaryBox.innerHTML = `<strong>${info.test_count} tests</strong> · Unprioritized runtime: <strong>${info.total_duration.toFixed(1)}s</strong> (~${durMins}m)${verdictNote}`;
    })
    .catch(() => {
      summaryBox.textContent = 'Build details ready';
    });
}

// 4. Strategy & Model selection toggle
function initStrategyToggle() {
  const stratSelect = document.getElementById('strategySelect');
  const modelCont = document.getElementById('modelSelectContainer');
  const alphaCont = document.getElementById('alphaContainer');

  stratSelect.addEventListener('change', function () {
    const isMl = this.value.startsWith('ml_');
    const isValDensity = this.value === 'ml_value_density';
    const isLtr = this.value === 'ml_ltr';

    if (isMl && !isLtr) {
      modelCont.style.display = 'block';
    } else {
      modelCont.style.display = 'none';
    }

    if (isValDensity) {
      alphaCont.style.display = 'block';
    } else {
      alphaCont.style.display = 'none';
    }
  });
}

// 5. CSV Upload & Validation
function handleCsvUpload(e) {
  const file = e.target.files[0];
  const feedback = document.getElementById('uploadFeedback');
  if (!file) return;

  feedback.style.display = 'block';
  feedback.className = 'small mt-2 text-muted';
  feedback.innerHTML = '<span class="spinner-border spinner-border-sm me-1"></span> Validating schema...';

  const formData = new FormData();
  formData.append('csv_file', file);

  fetch('/api/upload-validate', {
    method: 'POST',
    body: formData
  })
    .then(r => r.json())
    .then(data => {
      if (data.status === 'valid') {
        feedback.className = 'small mt-2 text-success';
        feedback.innerHTML = `<i class="bi bi-check-circle-fill me-1"></i> Validated: <strong>${data.total_tests} test cases</strong>, total est. runtime ${data.total_duration.toFixed(1)}s. (${data.features_present} pre-execution features present).`;
      } else {
        feedback.className = 'small mt-2 text-danger';
        feedback.innerHTML = `<i class="bi bi-exclamation-triangle-fill me-1"></i> Incompatible CSV: ${data.message}`;
      }
    })
    .catch(err => {
      feedback.className = 'small mt-2 text-danger';
      feedback.innerHTML = `<i class="bi bi-exclamation-circle me-1"></i> Error reading file: ${err}`;
    });
}

// 6. Action: Prioritize Tests
function initPrioritizeAction() {
  const btn = document.getElementById('prioritizeBtn');
  btn.addEventListener('click', runPrioritization);
}

function runPrioritization() {
  const isUpload = document.getElementById('source_upload').checked;
  const btn = document.getElementById('prioritizeBtn');
  const spinner = document.getElementById('btnSpinner');
  const icon = document.getElementById('btnIcon');
  const label = document.getElementById('btnLabel');

  // Prepare parameters
  const strategy = document.getElementById('strategySelect').value;
  const modelName = document.getElementById('modelSelect').value;
  const alpha = parseFloat(document.getElementById('alphaSlider').value);

  let budgetSeconds = 300.0;
  const selectedPreset = document.querySelector('input[name="budget_preset"]:checked').value;
  if (selectedPreset === 'custom') {
    budgetSeconds = parseFloat(document.getElementById('customBudgetSeconds').value) || 300.0;
  } else {
    budgetSeconds = parseFloat(selectedPreset);
  }

  const payload = new FormData();
  payload.append('strategy', strategy);
  payload.append('model_name', modelName);
  payload.append('time_budget_seconds', budgetSeconds);
  payload.append('alpha', alpha);

  if (isUpload) {
    const fileInput = document.getElementById('csvFileInput');
    if (!fileInput.files || fileInput.files.length === 0) {
      showToast('Please select a CSV file to prioritize.', 'warning');
      return;
    }
    payload.append('source_type', 'upload');
    payload.append('csv_file', fileInput.files[0]);
  } else {
    const project = document.getElementById('projectSelect').value;
    const build = document.getElementById('buildSelect').value;
    if (!project || !build) {
      showToast('Please select a project and build.', 'warning');
      return;
    }
    payload.append('source_type', 'benchmark');
    payload.append('project_id', project);
    payload.append('build_id', build);
  }

  // UI loading state
  btn.disabled = true;
  spinner.style.display = 'inline-block';
  icon.style.display = 'none';
  label.textContent = 'Prioritizing...';

  const t0 = performance.now();

  fetch('/api/prioritize', {
    method: 'POST',
    body: payload
  })
    .then(r => {
      if (!r.ok) {
        return r.json().then(err => { throw new Error(err.message || 'Prioritization failed'); });
      }
      return r.json();
    })
    .then(data => {
      const elapsedMs = Math.round(performance.now() - t0);
      data.client_elapsed_ms = elapsedMs;
      currentResults = data;
      renderScheduleResults(data);
      showToast(`Prioritization completed in ${elapsedMs} ms`, 'success');
    })
    .catch(err => {
      console.error(err);
      showToast(err.message || 'Execution error', 'danger');
    })
    .finally(() => {
      btn.disabled = false;
      spinner.style.display = 'none';
      icon.style.display = 'inline-block';
      label.textContent = 'Prioritize Tests';
    });
}

// 7. Render Schedule Results
function renderScheduleResults(data) {
  document.getElementById('emptyStateCard').style.display = 'none';
  document.getElementById('resultsCard').style.display = 'block';

  // Strategy explanation & timing
  document.getElementById('resStrategyBadge').textContent = data.strategy_label;
  document.getElementById('resExplanation').textContent = data.explanation;
  document.getElementById('resExecTime').textContent = `${data.execution_time_ms.toFixed(1)} ms runtime`;

  // KPIs
  const schedCount = data.scheduled_test_count;
  const totalCount = data.total_tests_received;
  const schedPct = totalCount > 0 ? ((schedCount / totalCount) * 100).toFixed(1) : 0;
  const budget = data.time_budget_seconds;
  const scheduledTime = data.total_scheduled_time_seconds;
  const budgetLeft = data.remaining_budget_seconds;
  const utilPct = budget > 0 ? Math.min(100, ((scheduledTime / budget) * 100)).toFixed(1) : 0;

  document.getElementById('kpiScheduledCount').textContent = `${schedCount} / ${totalCount}`;
  document.getElementById('kpiScheduledPct').textContent = `${schedPct}% included in budget`;

  document.getElementById('kpiRuntime').textContent = `${scheduledTime.toFixed(1)}s`;
  document.getElementById('kpiBudgetLabel').textContent = `of ${budget.toFixed(0)}s budget`;

  document.getElementById('kpiBudgetLeft').textContent = `${budgetLeft.toFixed(1)}s`;
  document.getElementById('kpiUtilPct').textContent = `${utilPct}% utilized`;

  document.getElementById('kpiExcludedCount').textContent = `${data.excluded_test_count} tests`;

  // Defect recall KPI card (if verdicts available)
  const recallCard = document.getElementById('kpiRecallCard');
  const thVerdict = document.getElementById('thVerdict');
  if (data.total_failures_in_suite !== null && data.total_failures_in_suite !== undefined && data.total_failures_in_suite > 0) {
    recallCard.style.display = 'block';
    const recPct = ((data.failure_recall || 0) * 100).toFixed(1);
    document.getElementById('kpiRecallValue').textContent = `${data.failures_discovered} / ${data.total_failures_in_suite}`;
    document.getElementById('kpiRecallSub').textContent = `${recPct}% Retrospective Recall`;
    thVerdict.style.display = 'table-cell';
  } else {
    recallCard.style.display = 'none';
    thVerdict.style.display = 'none';
  }

  // Budget progress bar
  document.getElementById('budgetProgressLabel').textContent = `${scheduledTime.toFixed(1)}s / ${budget.toFixed(0)}s (${utilPct}%)`;
  const pBar = document.getElementById('budgetProgressBar');
  pBar.style.width = `${utilPct}%`;
  pBar.style.backgroundColor = utilPct > 98 ? 'var(--tp-amber)' : 'var(--tp-emerald)';

  // Update counts on filter buttons
  document.getElementById('countAll').textContent = totalCount;
  document.getElementById('countQueued').textContent = schedCount;
  document.getElementById('countExcluded').textContent = data.excluded_test_count;

  // Render Table Rows
  renderTableRows();
}

// 8. Render Table Rows with active filter and search
function renderTableRows() {
  if (!currentResults) return;

  const tbody = document.getElementById('scheduleTableBody');
  tbody.innerHTML = '';

  const searchVal = (document.getElementById('tableSearchInput').value || '').toLowerCase().trim();
  const allTests = currentResults.all_ranked_tests || [];

  let visibleCount = 0;

  allTests.forEach(test => {
    // Status filter
    if (currentFilter === 'queued' && !test.within_budget) return;
    if (currentFilter === 'excluded' && test.within_budget) return;

    // Search query filter
    if (searchVal && !test.test_id.toLowerCase().includes(searchVal)) return;

    visibleCount++;
    const tr = document.createElement('tr');
    if (!test.within_budget) {
      tr.className = 'row-excluded';
    }

    // Rank cell
    const rankTd = document.createElement('td');
    rankTd.className = 'font-mono text-muted small fw-semibold';
    rankTd.textContent = `#${String(test.rank).padStart(2, '0')}`;
    tr.appendChild(rankTd);

    // Test ID cell
    const idTd = document.createElement('td');
    idTd.className = 'test-id-cell';
    idTd.title = test.test_id;
    idTd.innerHTML = `<strong>${escapeHtml(test.test_id)}</strong>`;
    tr.appendChild(idTd);

    // Priority Score cell
    const scoreTd = document.createElement('td');
    scoreTd.className = 'font-mono small';
    scoreTd.textContent = Number(test.priority_score).toFixed(4);
    tr.appendChild(scoreTd);

    // Risk cell
    const riskTd = document.createElement('td');
    if (test.predicted_risk !== null && test.predicted_risk !== undefined) {
      const r = Number(test.predicted_risk);
      let badgeClass = 'badge-risk-low';
      if (r > 0.4) badgeClass = 'badge-risk-high';
      else if (r > 0.15) badgeClass = 'badge-risk-med';
      riskTd.innerHTML = `<span class="${badgeClass}">${(r * 100).toFixed(1)}%</span>`;
    } else {
      riskTd.innerHTML = '<span class="text-muted small">--</span>';
    }
    tr.appendChild(riskTd);

    // Duration cell
    const durTd = document.createElement('td');
    durTd.className = 'font-mono small';
    durTd.textContent = `${Number(test.estimated_duration_seconds).toFixed(1)}s`;
    tr.appendChild(durTd);

    // Cumul Duration cell
    const cumTd = document.createElement('td');
    cumTd.className = 'font-mono small text-muted';
    cumTd.textContent = `${Number(test.cumulative_duration_seconds).toFixed(1)}s`;
    tr.appendChild(cumTd);

    // Status Badge cell
    const statusTd = document.createElement('td');
    if (test.within_budget) {
      statusTd.innerHTML = '<span class="badge-queued"><i class="bi bi-check2"></i> Queued</span>';
    } else {
      statusTd.innerHTML = '<span class="badge-excluded"><i class="bi bi-x"></i> Excluded</span>';
    }
    tr.appendChild(statusTd);

    // Supporting Evidence cell
    const evTd = document.createElement('td');
    evTd.innerHTML = formatEvidencePills(test.supporting_evidence);
    tr.appendChild(evTd);

    // Verdict cell (if applicable)
    if (currentResults.total_failures_in_suite !== null && currentResults.total_failures_in_suite !== undefined && currentResults.total_failures_in_suite > 0) {
      const vTd = document.createElement('td');
      if (test.verdict === 1) {
        vTd.innerHTML = '<span class="badge bg-danger-subtle text-danger border border-danger-subtle fw-bold" style="font-size: 0.7rem;">FAIL</span>';
      } else {
        vTd.innerHTML = '<span class="badge bg-success-subtle text-success border border-success-subtle" style="font-size: 0.7rem;">PASS</span>';
      }
      tr.appendChild(vTd);
    }

    tbody.appendChild(tr);
  });

  if (visibleCount === 0) {
    const emptyTr = document.createElement('tr');
    emptyTr.innerHTML = '<td colspan="9" class="text-center py-4 text-muted small">No test cases match the current filter or search term.</td>';
    tbody.appendChild(emptyTr);
  }
}

function formatEvidencePills(ev) {
  if (!ev) return '<span class="text-muted small">Baseline</span>';
  const pills = [];

  if (ev.historical_fail_rate !== undefined && ev.historical_fail_rate > 0) {
    pills.push(`<span class="evidence-pill">FailRate: ${(ev.historical_fail_rate * 100).toFixed(0)}%</span>`);
  }
  if (ev.failures_last_5 !== undefined && ev.failures_last_5 > 0) {
    pills.push(`<span class="evidence-pill">Last5: ${ev.failures_last_5}</span>`);
  }
  if (ev.modified_test_file) {
    pills.push('<span class="evidence-pill text-primary fw-semibold">Modified: Yes</span>');
  }
  if (ev.changed_files_count !== undefined && ev.changed_files_count > 0) {
    pills.push(`<span class="evidence-pill">Files: ${ev.changed_files_count}</span>`);
  }

  return pills.length > 0 ? pills.join(' ') : '<span class="text-muted small">No prior failures</span>';
}

// 9. Table filter buttons (All, Queued, Excluded)
function filterTable(type) {
  currentFilter = type;
  document.getElementById('btnFilterAll').classList.toggle('active', type === 'all');
  document.getElementById('btnFilterQueued').classList.toggle('active', type === 'queued');
  document.getElementById('btnFilterExcluded').classList.toggle('active', type === 'excluded');
  renderTableRows();
}

function searchTable() {
  renderTableRows();
}

// 10. Export Schedule CSV
function exportScheduleCSV() {
  if (!currentResults || !currentResults.all_ranked_tests) {
    showToast('No active schedule to export.', 'warning');
    return;
  }

  const tests = currentResults.all_ranked_tests;
  const headers = ['rank', 'test_id', 'priority_score', 'predicted_risk', 'est_duration', 'cum_duration', 'status', 'verdict'];
  const rows = [headers.join(',')];

  tests.forEach(t => {
    const r = [
      t.rank,
      `"${t.test_id.replace(/"/g, '""')}"`,
      t.priority_score,
      t.predicted_risk !== null ? t.predicted_risk : '',
      t.estimated_duration_seconds,
      t.cumulative_duration_seconds,
      t.within_budget ? 'QUEUED' : 'EXCLUDED',
      t.verdict !== null ? t.verdict : ''
    ];
    rows.push(r.join(','));
  });

  const csvContent = 'data:text/csv;charset=utf-8,' + encodeURIComponent(rows.join('\n'));
  const link = document.createElement('a');
  link.setAttribute('href', csvContent);
  link.setAttribute('download', `testpilot_schedule_${currentResults.strategy}_${currentResults.time_budget_seconds}s.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);

  showToast('Test schedule CSV downloaded.', 'success');
}

// Helper: Toast alerts
function showToast(msg, variant = 'info') {
  const toastEl = document.getElementById('statusToast');
  const toastMsg = document.getElementById('toastMessage');
  if (!toastEl || !toastMsg) return;

  toastMsg.textContent = msg;
  toastEl.className = `toast align-items-center text-white bg-${variant === 'danger' ? 'danger' : variant === 'success' ? 'success' : 'dark'} border-0`;
  const toast = new bootstrap.Toast(toastEl, { delay: 3500 });
  toast.show();
}

function escapeHtml(string) {
  const entityMap = {
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;'
  };
  return String(string).replace(/[&<>"']/g, s => entityMap[s]);
}
