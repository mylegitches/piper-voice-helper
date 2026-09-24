'use strict';

const $ = (sel) => document.querySelector(sel);
const show = (el, visible) => el.classList.toggle('hidden', !visible);

const TEST_SENTENCES = {
  en: 'Hello! This is my new voice. The front door is locked and the lights are off.',
};

let info = null;          // /api/info
let voice = null;         // selected voice (from /api/voices/{name})
let status = null;        // latest training status
let events = null;        // EventSource
let logCount = 0;
let selectedPreset = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    throw new Error(await response.text() || response.statusText);
  }
  const type = response.headers.get('content-type') || '';
  return type.includes('json') ? response.json() : response.blob();
}

function postJson(path, body) {
  return api(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  });
}

function voiceUrl(suffix = '') {
  return `/api/voices/${encodeURIComponent(voice.name)}${suffix}`;
}

// ---------------------------------------------------------------------------
// 1. Voice

async function loadVoices(selectName) {
  const { voices } = await api('/api/voices');
  const select = $('#voice-select');
  select.innerHTML = '<option value="">— choose a voice —</option>';
  voices.forEach((v) => {
    const option = new Option(`${v.name} · ${v.languageName} · ${v.recorded} recordings`, v.name);
    select.add(option);
  });

  let name = selectName;
  if (!name) {
    try { name = localStorage.getItem('voice'); } catch (e) { name = null; }
  }
  if (name && voices.some((v) => v.name === name)) {
    select.value = name;
  } else if (voices.length > 0) {
    select.value = voices[0].name;
  }

  show($('#new-voice-form'), voices.length === 0);
  await selectVoice(select.value);
}

async function selectVoice(name) {
  if (events) {
    events.close();
    events = null;
  }
  voice = name ? await api(`/api/voices/${encodeURIComponent(name)}`) : null;
  try { localStorage.setItem('voice', name || ''); } catch (e) { /* ignore */ }

  [$('#record-card'), $('#train-card'), $('#test-card')].forEach((card) => show(card, !!voice));
  if (!voice) {
    $('#voice-summary').textContent = '';
    return;
  }

  $('#voice-summary').textContent =
    `${voice.languageName} · ${voice.gender} · phonemes: ${voice.espeak_voice} · model: ${voice.modelName}.onnx`;
  $('#howto-name').textContent = voice.modelName;
  $('#speak-input').value = TEST_SENTENCES[voice.language.split('-')[0]] || '';

  skip = 0;
  $('#record-btn').innerHTML = '● Record <kbd>R</kbd>';
  $('#record-status').textContent = 'Read each sentence naturally in a quiet room. Press R to record, R again to stop.';
  updateRecorded(voice.recorded);
  await loadPrompt();
  fillTrainForm(voice.defaults);
  logCount = 0;
  $('#log-panel').textContent = '';
  renderStatus(voice.training);
  connectEvents();
}

$('#voice-select').addEventListener('change', (e) => selectVoice(e.target.value));
$('#new-voice-btn').addEventListener('click', () => {
  show($('#new-voice-form'), true);
  $('#new-name').focus();
});
$('#cancel-new-voice').addEventListener('click', () => show($('#new-voice-form'), false));

$('#new-voice-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  try {
    const created = await postJson('/api/voices', {
      name: $('#new-name').value,
      language: $('#new-language').value,
      gender: $('#new-gender').value,
    });
    $('#new-name').value = '';
    await loadVoices(created.name);
  } catch (err) {
    alert(err.message);
  }
});

// ---------------------------------------------------------------------------
// 2. Record

let prompt = null;
let skip = 0;
let mediaStream = null;
let recorder = null;
let chunks = [];
let recordedBlob = null;
let analyser = null;

function updateRecorded(count) {
  voice.recorded = count;
  $('#recorded-count').textContent = count;
  const fill = $('#readiness-fill');
  fill.style.width = `${Math.min(100, count / 10)}%`;
  let text;
  if (count < 50) {
    text = `record at least 50 to train (${50 - count} to go)`;
    fill.style.background = 'var(--danger)';
  } else if (count < 300) {
    text = 'enough to train · 300+ sounds much better';
    fill.style.background = 'var(--warning)';
  } else {
    text = 'great · more recordings still help';
    fill.style.background = 'var(--success)';
  }
  $('#readiness-text').textContent = text;
}

async function loadPrompt() {
  const result = await api(voiceUrl(`/prompt?skip=${skip}`));
  prompt = result.prompt;
  updateRecorded(result.recorded);
  resetTake();
  if (prompt) {
    $('#prompt-text').textContent = prompt.text;
  } else {
    $('#prompt-text').textContent = skip > 0
      ? 'No more sentences after the skipped ones. Reload to see them again.'
      : 'All sentences recorded — nice work!';
  }
  $('#record-btn').disabled = !prompt;
  $('#skip-btn').disabled = !prompt;
}

function resetTake() {
  recordedBlob = null;
  $('#play-btn').disabled = true;
  $('#save-btn').disabled = true;
  $('#prompt-box').classList.remove('recording', 'recorded');
}

async function listMicrophones() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const select = $('#mic-select');
    const current = select.value;
    select.innerHTML = '<option value="">Default microphone</option>';
    devices.filter((d) => d.kind === 'audioinput' && d.deviceId).forEach((d, i) => {
      select.add(new Option(d.label || `Microphone ${i + 1}`, d.deviceId));
    });
    select.value = current;
  } catch (e) { /* no devices API */ }
}

async function openMicrophone() {
  if (mediaStream) {
    return mediaStream;
  }
  const deviceId = $('#mic-select').value;
  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      deviceId: deviceId ? { exact: deviceId } : undefined,
      // Raw audio is best for training
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
      channelCount: 1,
    },
  });
  const context = new AudioContext();
  analyser = context.createAnalyser();
  analyser.fftSize = 1024;
  context.createMediaStreamSource(mediaStream).connect(analyser);
  const samples = new Float32Array(analyser.fftSize);
  const tick = () => {
    if (!analyser) {
      return;
    }
    analyser.getFloatTimeDomainData(samples);
    let peak = 0;
    for (const s of samples) {
      peak = Math.max(peak, Math.abs(s));
    }
    $('#vu-bar').style.width = `${Math.min(100, peak * 100)}%`;
    $('#vu-bar').style.background = peak > 0.95 ? 'var(--danger)' : 'var(--success)';
    requestAnimationFrame(tick);
  };
  tick();
  listMicrophones();
  return mediaStream;
}

$('#mic-select').addEventListener('change', () => {
  if (mediaStream) {
    mediaStream.getTracks().forEach((t) => t.stop());
    mediaStream = null;
    analyser = null;
  }
});

function preferredMimeType() {
  const types = ['audio/webm;codecs=opus', 'audio/ogg;codecs=opus', 'audio/webm', 'audio/mp4'];
  return types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || '';
}

async function toggleRecording() {
  if (!prompt) {
    return;
  }
  if (recorder && recorder.state === 'recording') {
    // Keep the tail of the last word
    $('#record-btn').disabled = true;
    setTimeout(() => recorder.stop(), 400);
    return;
  }

  try {
    await openMicrophone();
  } catch (err) {
    $('#record-status').textContent = `Microphone not available: ${err.message}`;
    return;
  }

  chunks = [];
  const mimeType = preferredMimeType();
  recorder = new MediaRecorder(mediaStream, mimeType ? { mimeType } : undefined);
  recorder.ondataavailable = (e) => chunks.push(e.data);
  recorder.onstop = () => {
    recordedBlob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' });
    $('#playback').src = URL.createObjectURL(recordedBlob);
    $('#record-btn').classList.remove('recording');
    $('#record-btn').innerHTML = '● Re-record <kbd>R</kbd>';
    $('#record-btn').disabled = false;
    $('#play-btn').disabled = false;
    $('#save-btn').disabled = false;
    $('#prompt-box').classList.replace('recording', 'recorded');
    $('#record-status').textContent = 'Listen back with P, then save with S (or re-record with R).';
  };
  recorder.start(250);
  resetTake();
  $('#prompt-box').classList.add('recording');
  $('#record-btn').classList.add('recording');
  $('#record-btn').innerHTML = '■ Stop <kbd>R</kbd>';
  $('#record-status').textContent = 'Recording… read the sentence, then press R.';
}

async function saveRecording() {
  if (!recordedBlob || !prompt) {
    return;
  }
  const form = new FormData();
  form.set('group', prompt.group);
  form.set('id', prompt.id);
  form.set('text', prompt.text);
  form.set('audio', recordedBlob, 'audio');
  $('#save-btn').disabled = true;
  try {
    const result = await api(voiceUrl('/recordings'), { method: 'POST', body: form });
    updateRecorded(result.recorded);
    $('#record-btn').innerHTML = '● Record <kbd>R</kbd>';
    $('#record-status').textContent = 'Saved. Next sentence:';
    await loadPrompt();
  } catch (err) {
    $('#save-btn').disabled = false;
    $('#record-status').textContent = `Save failed: ${err.message}`;
  }
}

$('#record-btn').addEventListener('click', toggleRecording);
$('#play-btn').addEventListener('click', () => $('#playback').play());
$('#save-btn').addEventListener('click', saveRecording);
$('#skip-btn').addEventListener('click', async () => {
  skip += 1;
  $('#record-btn').innerHTML = '● Record <kbd>R</kbd>';
  await loadPrompt();
});

document.addEventListener('keydown', (e) => {
  if (!voice || e.ctrlKey || e.metaKey || e.altKey || e.repeat) {
    return;
  }
  if (['INPUT', 'SELECT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
    return;
  }
  const key = e.key.toLowerCase();
  const actions = { r: '#record-btn', p: '#play-btn', s: '#save-btn', k: '#skip-btn' };
  if (actions[key] && !$(actions[key]).disabled) {
    e.preventDefault();
    $(actions[key]).click();
  }
});

$('#upload-btn').addEventListener('click', async () => {
  const file = $('#upload-input').files[0];
  if (!file) {
    return;
  }
  const form = new FormData();
  form.set('dataset', file);
  $('#upload-btn').disabled = true;
  $('#upload-btn').textContent = 'Uploading…';
  try {
    const result = await api(voiceUrl('/upload'), { method: 'POST', body: form });
    updateRecorded(result.recorded);
    alert(`Imported ${result.imported} recordings.`);
  } catch (err) {
    alert(`Upload failed: ${err.message}`);
  } finally {
    $('#upload-btn').disabled = false;
    $('#upload-btn').textContent = 'Upload';
  }
});

// ---------------------------------------------------------------------------
// 3. Train

function renderPresets() {
  const box = $('#presets');
  box.innerHTML = '';
  Object.entries(info.presets).forEach(([key, preset]) => {
    const button = document.createElement('button');
    button.className = 'preset';
    button.dataset.preset = key;
    button.innerHTML = '<strong></strong><span></span>';
    button.querySelector('strong').textContent = preset.label;
    button.querySelector('span').textContent = preset.hint;
    button.addEventListener('click', () => selectPreset(key));
    box.appendChild(button);
  });
}

function selectPreset(key) {
  selectedPreset = key;
  document.querySelectorAll('.preset').forEach((b) => b.classList.toggle('selected', b.dataset.preset === key));
  if (key !== 'custom' && info.presets[key]) {
    $('#hours-input').value = info.presets[key].hours;
  }
}

$('#hours-input').addEventListener('input', () => {
  const hours = Number($('#hours-input').value);
  const match = Object.entries(info.presets).find(([, p]) => p.hours === hours);
  selectedPreset = match ? match[0] : 'custom';
  document.querySelectorAll('.preset').forEach((b) => b.classList.toggle('selected', b.dataset.preset === selectedPreset));
});

function fillCheckpoints(defaults) {
  const select = $('#checkpoint-select');
  select.innerHTML = '';
  if (voice.training.hasCheckpoint) {
    select.add(new Option("Continue this voice's training", 'latest'));
  }
  const groups = Object.entries(info.checkpoints).sort(([a], [b]) => {
    const rank = (g) => (g === voice.espeak_voice || g === voice.espeak_voice.split('-')[0] ? 0 : g === 'generic' ? 1 : 2);
    return rank(a) - rank(b) || a.localeCompare(b);
  });
  groups.forEach(([group, entries]) => {
    const optgroup = document.createElement('optgroup');
    optgroup.label = group === 'generic' ? 'Any language' : group;
    entries.forEach((entry) => optgroup.appendChild(new Option(`${entry.name} · ${entry.gender}`, entry.url)));
    select.appendChild(optgroup);
  });
  select.add(new Option('Custom path or URL…', '__custom__'));
  select.add(new Option('Nothing (train from scratch — very slow)', ''));

  const known = Array.from(select.options).some((o) => o.value === defaults.checkpoint);
  select.value = known ? defaults.checkpoint : '__custom__';
  $('#checkpoint-input').value = known ? '' : defaults.checkpoint;
  show($('#checkpoint-input'), select.value === '__custom__');
}

$('#checkpoint-select').addEventListener('change', () => {
  show($('#checkpoint-input'), $('#checkpoint-select').value === '__custom__');
});

function fillTrainForm(defaults) {
  fillCheckpoints(defaults);
  $('#hours-input').value = defaults.hours;
  $('#epochs-input').value = defaults.epochs;
  $('#batch-input').value = defaults.batch_size;
  $('#device-select').value = defaults.accelerator;
  $('#rate-select').value = String(defaults.sample_rate);
  selectPreset(defaults.preset);
}

function trainSettings() {
  const choice = $('#checkpoint-select').value;
  return {
    preset: selectedPreset,
    hours: Number($('#hours-input').value) || 0,
    epochs: Number($('#epochs-input').value) || 0,
    checkpoint: choice === '__custom__' ? $('#checkpoint-input').value.trim() : choice,
    batch_size: Number($('#batch-input').value) || 0,
    accelerator: $('#device-select').value,
    sample_rate: Number($('#rate-select').value),
  };
}

$('#train-btn').addEventListener('click', async () => {
  if (voice.recorded < 50 && !confirm(`Only ${voice.recorded} recordings. The voice will sound rough. Train anyway?`)) {
    return;
  }
  try {
    renderStatus(await postJson(voiceUrl('/train'), trainSettings()));
  } catch (err) {
    showError(err.message);
  }
});

$('#stop-btn').addEventListener('click', async () => {
  if (!confirm('Stop training? The latest version will be exported so you can still use it.')) {
    return;
  }
  renderStatus(await postJson(voiceUrl('/stop')));
});

function showError(text) {
  $('#train-error').textContent = text;
  show($('#train-error'), !!text);
}

function formatDuration(seconds) {
  const s = Math.max(0, Math.round(seconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  return h > 0 ? `${h}h ${String(m).padStart(2, '0')}m` : `${m}m ${String(s % 60).padStart(2, '0')}s`;
}

const STAGES = ['prepare', 'download', 'train', 'export'];
const STAGE_LABELS = {
  prepare: 'Preparing recordings…',
  download: 'Downloading base voice…',
  train: 'Training',
  export: 'Exporting voice…',
};

function renderStatus(s) {
  status = s;
  const running = s.state === 'running';
  const busyElsewhere = info.busy && info.busy !== voice.name;

  show($('#train-btn'), !running);
  show($('#stop-btn'), running);
  $('#train-btn').disabled = s.exporting || !!busyElsewhere || !info.device.ok;
  $('#train-btn').textContent = s.exports.length > 0 || s.hasCheckpoint ? 'Train more' : 'Train';
  document.querySelectorAll('#train-card input, #train-card select, .preset').forEach((el) => {
    el.disabled = running;
  });
  show($('#progress'), running || s.exporting || s.state !== 'idle');

  // Stage chips
  const current = STAGES.indexOf(s.stage);
  document.querySelectorAll('.stage').forEach((chip) => {
    const index = STAGES.indexOf(chip.dataset.stage);
    chip.classList.toggle('active', index === current);
    chip.classList.toggle('done', current >= 0 ? index < current : s.state === 'succeeded');
  });

  // Label + time bar
  const fill = $('#timebar-fill');
  const limit = (s.settings && s.settings.hours) ? s.settings.hours * 3600 : 0;
  const elapsed = s.started ? ((s.finished || s.now) - s.started) : 0;
  let label;
  if (running) {
    label = STAGE_LABELS[s.stage] || 'Working…';
    if (s.stage === 'train') {
      label += s.epoch !== null ? ` · epoch ${s.epoch}` : '';
      label += ` · ${formatDuration(elapsed)}${limit ? ` of ${formatDuration(limit)}` : ''}`;
    }
  } else if (s.exporting) {
    label = STAGE_LABELS.export;
  } else {
    label = {
      succeeded: `Finished after ${formatDuration(elapsed)}`,
      stopped: `Stopped after ${formatDuration(elapsed)}`,
      failed: 'Training failed',
      idle: '',
    }[s.state];
  }
  $('#progress-label').textContent = label;
  const timed = running && s.stage === 'train' && limit;
  fill.classList.toggle('indeterminate', (running || s.exporting) && !timed);
  fill.style.width = timed ? `${Math.min(100, (100 * elapsed) / limit)}%` : (s.state === 'idle' ? '0' : '100%');

  showError(s.state === 'failed' || s.error ? s.error || 'See the log for details.' : '');
  if (busyElsewhere && !running) {
    showError(`Another voice (${info.busy}) is training. Wait for it to finish or stop it first.`);
  }

  renderExports(s);
}

function renderExports(s) {
  const exports = s.exports || [];
  show($('#export-area'), exports.length > 0);
  show($('#no-exports'), exports.length === 0);
  show($('#export-btn'), s.hasCheckpoint && (s.state === 'running' || exports.length === 0 || s.exporting));
  $('#export-btn').disabled = s.exporting;
  $('#export-btn').textContent = s.exporting ? 'Exporting…' : 'Export latest version now';

  const select = $('#export-select');
  const key = exports.map((e) => e.dir).join(',');
  if (select.dataset.key !== key) {
    const previous = select.value;
    const newest = exports.length > 0 ? exports[0].dir : '';
    select.innerHTML = '';
    exports.forEach((e, i) => {
      const when = new Date(e.created * 1000).toLocaleString();
      select.add(new Option(`Epoch ${e.epoch}${i === 0 ? ' (latest)' : ''} · ${when}`, e.dir));
    });
    // Jump to a newly exported version
    select.value = select.dataset.key && select.dataset.newest !== newest ? newest : (previous || newest);
    if (!select.value) {
      select.value = newest;
    }
    select.dataset.key = key;
    select.dataset.newest = newest;
    updateDownloads();
  }
}

function updateDownloads() {
  const dir = $('#export-select').value;
  const entry = (status.exports || []).find((e) => e.dir === dir);
  if (!entry) {
    return;
  }
  const base = voiceUrl(`/exports/${dir}`);
  $('#dl-zip').href = `${base}/home-assistant.zip`;
  $('#dl-onnx').href = `${base}/${entry.model}`;
  $('#dl-json').href = `${base}/${entry.model}.json`;
}

$('#export-select').addEventListener('change', updateDownloads);
$('#export-btn').addEventListener('click', async () => {
  try {
    renderStatus(await postJson(voiceUrl('/export')));
  } catch (err) {
    showError(err.message);
  }
});

$('#speak-btn').addEventListener('click', async () => {
  const button = $('#speak-btn');
  button.disabled = true;
  button.textContent = 'Speaking…';
  try {
    const wav = await postJson(voiceUrl('/speak'), {
      text: $('#speak-input').value,
      export: $('#export-select').value,
    });
    const audio = $('#speak-audio');
    audio.src = URL.createObjectURL(wav);
    show(audio, true);
    audio.play();
  } catch (err) {
    alert(err.message);
  } finally {
    button.disabled = false;
    button.textContent = '🔊 Speak';
  }
});
$('#speak-input').addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    $('#speak-btn').click();
  }
});

// ---------------------------------------------------------------------------
// Live status

function connectEvents() {
  const name = voice.name;
  events = new EventSource(`/api/voices/${encodeURIComponent(name)}/events?since=${logCount}`);
  events.onmessage = (e) => {
    if (!voice || voice.name !== name) {
      return;
    }
    const data = JSON.parse(e.data);
    const wasRunning = status && status.state === 'running';
    logCount = data.logCount;
    if (data.lines.length > 0) {
      const panel = $('#log-panel');
      const atBottom = panel.scrollHeight - panel.scrollTop - panel.clientHeight < 30;
      panel.textContent = (panel.textContent + '\n' + data.lines.join('\n')).split('\n').slice(-1500).join('\n').trim();
      if (atBottom) {
        panel.scrollTop = panel.scrollHeight;
      }
    }
    if (wasRunning !== (data.state === 'running')) {
      refreshInfo();
    }
    renderStatus(data);
  };
}

async function refreshInfo() {
  info = await api('/api/info');
}

// ---------------------------------------------------------------------------

async function main() {
  info = await api('/api/info');
  const languageSelect = $('#new-language');
  info.languages.forEach((l) => languageSelect.add(new Option(l.name, l.code)));
  const browserLanguage = (navigator.language || 'en-US').toLowerCase();
  const match = info.languages.find((l) => l.code.toLowerCase() === browserLanguage)
    || info.languages.find((l) => l.code.toLowerCase().startsWith(browserLanguage.split('-')[0]));
  if (match) {
    languageSelect.value = match.code;
  }
  info.accelerators.forEach((a) => $('#device-select').add(new Option(a, a)));
  renderPresets();

  const banner = $('#device-banner');
  if (!info.device.ok) {
    banner.textContent = 'Piper training is not installed, so you can record but not train. '
      + 'Use the Docker image or run script/setup (see README).';
    show(banner, true);
  } else if (!info.device.gpu) {
    banner.textContent = 'No NVIDIA GPU found. Training will run on the CPU and be very slow '
      + '(expect days, not hours, for a good voice).';
    show(banner, true);
  }

  await loadVoices();
}

main().catch((err) => {
  document.body.insertAdjacentHTML('afterbegin', '<div class="banner banner--error"></div>');
  document.querySelector('.banner--error').textContent = `Failed to load: ${err.message}`;
});
