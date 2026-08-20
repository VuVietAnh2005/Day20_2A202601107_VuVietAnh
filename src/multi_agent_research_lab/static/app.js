document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('research-form');
  const queryInput = document.getElementById('query-input');
  const btnRun = document.getElementById('btn-run');
  const btnRunText = document.getElementById('btn-run-text');
  const btnSpinner = document.getElementById('btn-spinner');
  const btnBenchmark = document.getElementById('btn-benchmark');
  const pipelineStatus = document.getElementById('pipeline-status');

  const metricsBar = document.getElementById('metrics-bar');
  const resultsSection = document.getElementById('results-section');
  const benchmarkSection = document.getElementById('benchmark-section');
  const closeBenchmarkBtn = document.getElementById('close-benchmark-btn');

  const finalAnswerBody = document.getElementById('final-answer-body');
  const sourcesContainer = document.getElementById('sources-container');
  const researcherNotesBody = document.getElementById('researcher-notes-body');
  const analystNotesBody = document.getElementById('analyst-notes-body');
  const traceContainer = document.getElementById('trace-container');
  const sourceCountBadge = document.getElementById('source-count-badge');

  const metricLatency = document.getElementById('metric-latency');
  const metricIterations = document.getElementById('metric-iterations');
  const metricSources = document.getElementById('metric-sources');
  const metricRoutes = document.getElementById('metric-routes');

  const agentNodes = {
    supervisor: document.getElementById('node-supervisor'),
    researcher: document.getElementById('node-researcher'),
    analyst: document.getElementById('node-analyst'),
    writer: document.getElementById('node-writer'),
    critic: document.getElementById('node-critic'),
  };

  const agentStatuses = {
    supervisor: document.getElementById('status-supervisor'),
    researcher: document.getElementById('status-researcher'),
    analyst: document.getElementById('status-analyst'),
    writer: document.getElementById('status-writer'),
    critic: document.getElementById('status-critic'),
  };

  // 1. Quick Query Suggestions
  document.querySelectorAll('.chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      queryInput.value = chip.dataset.query;
      queryInput.focus();
    });
  });

  // 2. Tab Navigation
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach((p) => p.classList.remove('active'));

      btn.classList.add('active');
      const targetTab = document.getElementById(btn.dataset.tab);
      if (targetTab) targetTab.classList.add('active');
    });
  });

  // 3. Reset visual pipeline
  function resetPipeline() {
    Object.keys(agentNodes).forEach((key) => {
      agentNodes[key].classList.remove('active', 'completed');
      agentStatuses[key].textContent = 'Standby';
    });
  }

  function setAgentActive(name, status = 'Processing...') {
    if (agentNodes[name]) {
      agentNodes[name].classList.add('active');
      agentStatuses[name].textContent = status;
    }
  }

  function setAgentCompleted(name, status = 'Done') {
    if (agentNodes[name]) {
      agentNodes[name].classList.remove('active');
      agentNodes[name].classList.add('completed');
      agentStatuses[name].textContent = status;
    }
  }

  // 4. Submit Research Form
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = queryInput.value.trim();
    if (!query) return;

    const mode = document.querySelector('input[name="mode"]:checked').value;

    // UI Loading state
    btnRun.disabled = true;
    btnSpinner.classList.remove('hidden');
    btnRunText.textContent = 'Đang xử lý...';
    pipelineStatus.textContent = mode === 'multi-agent' ? 'Multi-Agent Orchestrating...' : 'Single-Agent Calling...';
    
    resetPipeline();
    metricsBar.classList.add('hidden');
    resultsSection.classList.add('hidden');
    benchmarkSection.classList.add('hidden');

    let animationInterval = null;
    if (mode === 'multi-agent') {
      let step = 0;
      const sequence = ['supervisor', 'researcher', 'analyst', 'writer', 'critic'];
      setAgentActive(sequence[0], 'Active');

      animationInterval = setInterval(() => {
        if (step < sequence.length - 1) {
          setAgentCompleted(sequence[step], 'Done');
          step++;
          setAgentActive(sequence[step], 'Active');
        }
      }, 5000);
    } else {
      setAgentActive('writer', 'Generating...');
    }

    try {
      const response = await fetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, mode }),
      });

      if (!response.ok) {
        throw new Error(`HTTP Error ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      if (animationInterval) clearInterval(animationInterval);

      // Mark all completed in pipeline
      if (mode === 'multi-agent') {
        ['supervisor', 'researcher', 'analyst', 'writer', 'critic'].forEach((ag) => {
          setAgentCompleted(ag, 'Finished');
        });
      } else {
        setAgentCompleted('writer', 'Single Call Complete');
      }

      pipelineStatus.textContent = 'Completed in ' + data.latency_seconds + 's';

      // Update Metrics Bar
      metricLatency.textContent = `${data.latency_seconds}s`;
      metricIterations.textContent = data.iteration || 1;
      metricSources.textContent = (data.sources || []).length;
      metricRoutes.textContent = (data.route_history || []).join(' → ') || (mode === 'baseline' ? 'Monolithic 1-Step' : 'N/A');
      metricsBar.classList.remove('hidden');

      // Update Results
      finalAnswerBody.innerHTML = marked.parse(data.final_answer || 'Không có câu trả lời.');

      // Update Sources Tab
      sourceCountBadge.textContent = (data.sources || []).length;
      sourcesContainer.innerHTML = '';
      if (data.sources && data.sources.length > 0) {
        data.sources.forEach((s, idx) => {
          const item = document.createElement('div');
          item.className = 'source-item';
          item.innerHTML = `
            <div class="source-header">
              <div class="source-title">[${idx + 1}] ${s.title}</div>
              <div class="source-badge">Source ${idx + 1}</div>
            </div>
            <div class="source-snippet">${s.snippet}</div>
            ${s.url ? `<a href="${s.url}" target="_blank" rel="noreferrer" class="source-url">🔗 ${s.url}</a>` : ''}
          `;
          sourcesContainer.appendChild(item);
        });
      } else {
        sourcesContainer.innerHTML = '<p class="text-dim">Không có nguồn tài liệu ngoài được trích xuất (Baseline Mode).</p>';
      }

      // Update Workspace Tab
      researcherNotesBody.innerHTML = marked.parse(data.research_notes || '*Không có ghi chú Researcher.*');
      analystNotesBody.innerHTML = marked.parse(data.analysis_notes || '*Không có phân tích Analyst.*');

      // Update Trace Tab
      traceContainer.innerHTML = '';
      if (data.trace && data.trace.length > 0) {
        data.trace.forEach((t) => {
          const ev = document.createElement('div');
          ev.className = 'timeline-event';
          ev.innerHTML = `
            <div class="timeline-name">${t.name}</div>
            <div class="timeline-payload">${JSON.stringify(t.payload)}</div>
          `;
          traceContainer.appendChild(ev);
        });
      } else {
        traceContainer.innerHTML = '<p class="text-dim">Chưa có trace event.</p>';
      }

      resultsSection.classList.remove('hidden');
      resultsSection.scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
      if (animationInterval) clearInterval(animationInterval);
      pipelineStatus.textContent = 'Failed: ' + err.message;
      alert('Đã xảy ra lỗi khi thực thi: ' + err.message);
    } finally {
      btnRun.disabled = false;
      btnSpinner.classList.add('hidden');
      btnRunText.textContent = '🚀 Bắt đầu Nghiên cứu';
    }
  });

  // 5. Benchmark Button
  btnBenchmark.addEventListener('click', async () => {
    const query = queryInput.value.trim() || 'Research GraphRAG state-of-the-art and write a 500-word summary';
    btnBenchmark.disabled = true;
    btnBenchmark.textContent = '⏳ Đang benchmark...';

    try {
      const response = await fetch('/api/benchmark', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();
      const tbody = document.getElementById('benchmark-table-body');
      tbody.innerHTML = '';

      (data.results || []).forEach((res) => {
        const tr = document.createElement('tr');
        const costStr = res.estimated_cost_usd !== null ? `$${res.estimated_cost_usd.toFixed(5)}` : 'N/A';
        const qualityStr = res.quality_score !== null ? `${res.quality_score.toFixed(1)}/10` : 'N/A';
        const citationStr = res.citation_coverage !== null ? `${Math.round(res.citation_coverage * 100)}%` : 'N/A';
        const failureStr = res.failure_rate !== null ? `${Math.round(res.failure_rate * 100)}%` : '0%';

        tr.innerHTML = `
          <td><strong>${res.run_name}</strong></td>
          <td>${res.latency_seconds}s</td>
          <td>${costStr}</td>
          <td><span class="badge ${res.quality_score >= 6 ? 'badge-success' : 'badge-pulse'}">${qualityStr}</span></td>
          <td>${citationStr}</td>
          <td>${failureStr}</td>
        `;
        tbody.appendChild(tr);
      });

      benchmarkSection.classList.remove('hidden');
      benchmarkSection.scrollIntoView({ behavior: 'smooth' });
    } catch (err) {
      alert('Lỗi khi chạy benchmark: ' + err.message);
    } finally {
      btnBenchmark.disabled = false;
      btnBenchmark.textContent = '📊 Chạy Benchmark';
    }
  });

  closeBenchmarkBtn.addEventListener('click', () => {
    benchmarkSection.classList.add('hidden');
  });
});
