function el(id) { return document.getElementById(id); }

// ── Source controls ────────────────────────────────────────────

async function switchToWebcam() {
  setUploadStatus('loading', 'Starting webcam…');
  try {
    await fetch('/api/source/webcam', { method: 'POST' });
    reloadFeed();
    setUploadStatus('ok', 'Webcam active');
    setTimeout(() => setUploadStatus('', ''), 3000);
  } catch {
    setUploadStatus('err', 'Failed');
  }
}

async function uploadFile(file) {
  if (!file) return;
  setUploadStatus('loading', `Uploading ${file.name}…`);
  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!r.ok) throw new Error();
    reloadFeed();
    setUploadStatus('ok', `Loaded: ${file.name}`);
    setTimeout(() => setUploadStatus('', ''), 4000);
  } catch {
    setUploadStatus('err', 'Upload failed');
  }
  el('file-input').value = '';
}

function reloadFeed() {
  const img = el('video-img');
  img.style.display = 'none';
  el('video-idle').style.display = 'flex';
  setTimeout(() => {
    img.src = '/video_feed?' + Date.now();
    img.style.display = 'block';
    el('video-idle').style.display = 'none';
    el('overlay-badge').style.display = 'block';
  }, 800);
}

function setUploadStatus(type, msg) {
  const s = el('upload-status');
  s.className = 'upload-status' + (type ? ` ${type}` : '');
  s.textContent = msg;
}

// ── Drag-and-drop ──────────────────────────────────────────────

const videoCol = el('video-col');
let dragCounter = 0;

videoCol.addEventListener('dragenter', e => {
  e.preventDefault();
  dragCounter++;
  el('drag-overlay').classList.add('active');
});

videoCol.addEventListener('dragleave', () => {
  dragCounter--;
  if (dragCounter === 0) el('drag-overlay').classList.remove('active');
});

videoCol.addEventListener('dragover', e => e.preventDefault());

videoCol.addEventListener('drop', e => {
  e.preventDefault();
  dragCounter = 0;
  el('drag-overlay').classList.remove('active');
  const file = e.dataTransfer.files[0];
  if (file) uploadFile(file);
});

// ── Status polling (every 2 s) ─────────────────────────────────

let _prevSource = null;

async function pollStatus() {
  try {
    const d = await fetch('/api/status').then(r => r.json());

    el('dot').className = d.running ? 'dot' : 'dot offline';
    el('live-text').textContent = d.running
      ? `Live · Frame ${d.frame_count}`
      : (d.source === 'none' ? 'No source — upload or use webcam' : 'Detection stopped');

    const risk = d.risk_level || 'UNKNOWN';

    el('overlay-badge').textContent = `● ${risk}`;
    el('overlay-badge').className   = `video-overlay-badge rb-${risk}`;
    if (d.running) el('overlay-badge').style.display = 'block';

    el('risk-val').textContent = risk;
    el('risk-val').className   = `risk-value color-${risk}`;

    const classes = d.det_classes || [];
    el('risk-sub').textContent = classes.length
      ? `Detected: ${classes.join(', ')}`
      : d.dry_veg_detected ? 'Dry vegetation present' : 'No active threat';

    el('frame-ct').textContent = `Frame #${d.frame_count}`;

    const src = d.source || 'none';
    el('source-name').textContent = src;
    if (src !== _prevSource && src !== 'none') _prevSource = src;

    if (d.weather) {
      const w = d.weather;
      el('w-temp').textContent  = w.temp      ?? '—';
      el('w-hum').textContent   = w.humidity  ?? '—';
      el('w-wind').textContent  = w.wind      ?? '—';
      el('w-rain').textContent  = w.rain      ?? '—';
      const fwi = w.fwi || 0;
      el('fwi-label').textContent = `${fwi} · ${w.label}`;
      el('fwi-bar').style.width   = Math.min((fwi / 50) * 100, 100) + '%';
    }

    const ratio = d.dry_veg_ratio || 0;
    const pct   = (ratio * 100).toFixed(1);
    el('dveg-bar').style.width    = pct + '%';
    el('dveg-pct').textContent    = pct + '%';
    el('dveg-status').textContent = d.dry_veg_detected
      ? `⚠ Dry vegetation — ${pct}% of frame`
      : 'No dry vegetation detected';
    el('dveg-status').style.color = d.dry_veg_detected ? '#ca8a04' : 'var(--muted)';

  } catch { /* server not ready */ }
}

// ── Stats + history polling (every 10 s) ───────────────────────

async function pollStats() {
  try {
    const s = await fetch('/api/stats').then(r => r.json());
    el('s-total').textContent    = s.total       || 0;
    el('s-critical').textContent = s.critical    || 0;
    el('s-high').textContent     = s.high        || 0;
    el('s-alerts').textContent   = s.alerts_sent || 0;
  } catch {}
}

async function pollDetections() {
  try {
    const rows = await fetch('/api/detections?limit=25').then(r => r.json());
    const tbody = el('det-tbody');
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="7" class="empty">No events logged yet</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map(r => {
      const ts      = new Date(r.timestamp).toLocaleString();
      const classes = (r.det_classes || '').split(',').filter(Boolean);
      const chips   = classes.map(c => `<span class="chip">${c}</span>`).join('')
                      || '<span style="color:var(--muted)">—</span>';
      const dryPct  = r.dry_veg_ratio ? (r.dry_veg_ratio * 100).toFixed(1) + '%' : '—';
      const fwi     = r.fwi_score != null ? `${r.fwi_score} (${r.fwi_label})` : '—';
      const temp    = r.temp      != null ? `${r.temp}°C` : '—';
      const alert   = r.alert_sent
        ? '<span class="alert-yes">✓ Sent</span>'
        : '<span class="alert-no">—</span>';
      return `
        <tr>
          <td style="color:var(--muted);white-space:nowrap">${ts}</td>
          <td><span class="pill pill-${r.risk_level}">${r.risk_level}</span></td>
          <td>${chips}</td>
          <td>${dryPct}</td>
          <td style="color:var(--muted)">${fwi}</td>
          <td style="color:var(--muted)">${temp}</td>
          <td>${alert}</td>
        </tr>`;
    }).join('');
  } catch {}
}

// ── Boot ───────────────────────────────────────────────────────

pollStatus();
pollStats();
pollDetections();
setInterval(pollStatus, 2000);
setInterval(() => { pollStats(); pollDetections(); }, 10000);
