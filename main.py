"""
AI Influencer Content Factory — FastAPI dashboard
Replicates the KVK AUTOMATES n8n workflow in pure Python.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from pipeline import PipelineConfig, run_pipeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="AI Influencer Content Factory")

# ---------------------------------------------------------------------------
# In-memory state (fine for single-instance; use Redis for multi-instance)
# ---------------------------------------------------------------------------
jobs: dict = {}       # job_id → job metadata + posts
job_logs: dict = {}   # job_id → list of log entries

_public_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "")
_app_base_url = f"https://{_public_domain}" if _public_domain else f"http://localhost:{os.getenv('PORT', 8000)}"

config = PipelineConfig(
    openai_api_key=os.getenv("OPENAI_API_KEY", ""),
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
    groq_api_key=os.getenv("GROQ_API_KEY", ""),
    kie_ai_api_key=os.getenv("KIE_AI_API_KEY", ""),
    telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
    telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
    openai_model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
    groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    app_base_url=_app_base_url,
)


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.post("/api/generate")
async def generate(request: Request, background_tasks: BackgroundTasks):
    body = await request.json()

    # Validate required fields
    required = ["character_url", "num_images", "num_videos", "aspect_ratio", "creative_direction"]
    for field in required:
        if not body.get(field) and body.get(field) != 0:
            raise HTTPException(400, f"Missing required field: {field}")

    job_id = uuid.uuid4().hex[:8]
    jobs[job_id] = {
        "id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "form_data": body,
        "posts": [],
        "error": None,
        "completed_at": None,
    }
    job_logs[job_id] = []

    background_tasks.add_task(run_pipeline, job_id, body, config, jobs, job_logs)
    return {"job_id": job_id}


@app.get("/api/jobs")
async def list_jobs():
    return list(jobs.values())


@app.get("/api/job/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    return jobs[job_id]


@app.get("/api/job/{job_id}/logs")
async def stream_logs(job_id: str):
    """Server-Sent Events stream for real-time log tailing."""
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")

    async def event_gen():
        sent = 0
        while True:
            logs = job_logs.get(job_id, [])
            for entry in logs[sent:]:
                yield f"data: {json.dumps(entry)}\n\n"
            sent = len(logs)

            status = jobs.get(job_id, {}).get("status", "")
            if status in ("done", "failed"):
                yield "data: {\"msg\":\"__STREAM_END__\",\"level\":\"info\"}\n\n"
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/img/{filename}")
async def image_proxy(filename: str, url: str):
    """Fetch any image URL, convert to JPEG, serve with .jpg path.
    kie.ai requires a URL ending in .jpg that returns image/jpeg content."""
    import io
    import httpx
    from PIL import Image

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ImageProxy/1.0)",
        "Accept": "image/*,*/*",
    }
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url, headers=headers)
        if not resp.is_success:
            raise HTTPException(502, f"Source image returned HTTP {resp.status_code}")
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        return Response(
            content=buf.getvalue(),
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        logging.error("image_proxy error for %s: %s", url, exc)
        raise HTTPException(502, f"Image conversion failed: {exc}")


@app.get("/api/proxy-check")
async def proxy_check(url: str):
    """Check if our server can fetch a given URL — useful for debugging."""
    import httpx
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ImageProxy/1.0)", "Accept": "image/*,*/*"}
    async with httpx.AsyncClient(follow_redirects=True, timeout=15) as client:
        resp = await client.get(url, headers=headers)
    return {
        "status_code": resp.status_code,
        "content_type": resp.headers.get("content-type", "unknown"),
        "content_length": len(resp.content),
        "reachable": resp.is_success,
    }


@app.get("/api/test-kieai")
async def test_kieai():
    """Probe kie.ai with a minimal payload to see the raw response."""
    import httpx
    payload = {
        "model": "nano-banana-pro",
        "input": {
            "prompt": "test",
            "image_input": ["https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"],
            "aspect_ratio": "1:1",
            "resolution": "2K",
            "output_format": "jpg",
        },
    }
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            "https://api.kie.ai/api/v1/jobs/createTask",
            headers={"Authorization": f"Bearer {config.kie_ai_api_key}"},
            json=payload,
        )
    return {"status_code": resp.status_code, "body": resp.json()}


@app.get("/api/health")
async def health():
    model_in_use = (
        config.anthropic_model if config.anthropic_api_key
        else config.groq_model if config.groq_api_key
        else config.openai_model
    )
    return {
        "status": "ok",
        "ai_provider": config.ai_provider,
        "anthropic": bool(config.anthropic_api_key),
        "groq": bool(config.groq_api_key),
        "openai": bool(config.openai_api_key),
        "kie_ai": bool(config.kie_ai_api_key),
        "telegram": bool(config.telegram_bot_token and config.telegram_chat_id),
        "model": model_in_use,
    }


# ---------------------------------------------------------------------------
# Dashboard HTML (single-file SPA)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AI Influencer Content Factory</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    :root { color-scheme: dark; }

    body { background: #0a0f1e; }

    .card  { background: #111827; border: 1px solid #1f2937; border-radius: 12px; overflow: hidden; }
    .input { width:100%; background:#1f2937; border:1px solid #374151; border-radius:8px;
             padding:8px 12px; font-size:14px; color:#f9fafb; outline:none; }
    .input:focus { border-color:#6366f1; box-shadow:0 0 0 2px #6366f120; }
    .input::placeholder { color:#4b5563; }
    textarea.input { resize:vertical; }

    .btn-primary {
      background: linear-gradient(135deg,#6366f1,#8b5cf6);
      color:#fff; font-weight:600; padding:12px 20px; border-radius:8px;
      border:none; cursor:pointer; width:100%; transition:opacity .2s;
    }
    .btn-primary:hover:not(:disabled) { opacity:.9; }
    .btn-primary:disabled { opacity:.5; cursor:not-allowed; }

    .badge { display:inline-flex; align-items:center; gap:4px;
             padding:2px 10px; border-radius:999px; font-size:11px; font-weight:600; }
    .badge-queued   { background:#1f2937; color:#9ca3af; }
    .badge-running  { background:#1e3a5f; color:#60a5fa; }
    .badge-done     { background:#14532d; color:#4ade80; }
    .badge-failed   { background:#7f1d1d; color:#fca5a5; }
    .badge-image_done        { background:#1e3a5f; color:#93c5fd; }
    .badge-generating_image  { background:#1a2e4a; color:#67e8f9; }
    .badge-generating_video  { background:#2d1a4a; color:#c084fc; }
    .badge-image_failed { background:#7f1d1d; color:#fca5a5; }
    .badge-video_failed { background:#7f1d1d; color:#fca5a5; }

    .log-info    { color:#64748b; }
    .log-error   { color:#f87171; }
    .log-success { color:#4ade80; }
    .log-warning { color:#fb923c; }
    #log-box { font-family:'JetBrains Mono','Courier New',monospace; font-size:11px;
               height:220px; overflow-y:auto; padding:12px; background:#050a12;
               border-top:1px solid #1f2937; }
    #log-box div { line-height:1.6; }

    .job-row { padding:14px 20px; cursor:pointer; transition:background .15s; border-bottom:1px solid #1f2937; }
    .job-row:hover { background:#1a2236; }
    .job-row.active { background:#1e2740; }

    .post-card { background:#1a2236; border:1px solid #1f2937; border-radius:10px; overflow:hidden; }
    .post-thumb { width:100%; height:140px; object-fit:cover; display:block; }
    .post-thumb-placeholder { width:100%; height:140px; background:#1f2937;
                               display:flex; align-items:center; justify-content:center; }

    .label { font-size:12px; font-weight:500; color:#9ca3af; margin-bottom:4px; }
    .section-head { padding:14px 20px; font-weight:600; color:#f9fafb; font-size:14px;
                    border-bottom:1px solid #1f2937; background:#0d1117; display:flex;
                    align-items:center; justify-content:space-between; }

    .dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
    .dot-ok  { background:#4ade80; }
    .dot-bad { background:#f87171; }

    ::-webkit-scrollbar { width:6px; }
    ::-webkit-scrollbar-track { background:transparent; }
    ::-webkit-scrollbar-thumb { background:#374151; border-radius:3px; }
  </style>
</head>
<body class="text-slate-100 min-h-screen">

  <!-- ── HEADER ────────────────────────────────────────────────────── -->
  <header style="background:#0d1117;border-bottom:1px solid #1f2937;" class="px-6 py-4">
    <div class="max-w-7xl mx-auto flex items-center gap-3">
      <div style="font-size:26px;">🤖</div>
      <div>
        <h1 class="text-lg font-bold text-white">AI Influencer Content Factory</h1>
        <p class="text-xs text-slate-500">KVK AUTOMATES — NanoBanana Pro + Kling 2.6 + Telegram</p>
      </div>
      <div class="ml-auto flex items-center gap-4 text-xs text-slate-500" id="api-indicators">
        Loading…
      </div>
    </div>
  </header>

  <!-- ── MAIN ──────────────────────────────────────────────────────── -->
  <main class="max-w-7xl mx-auto px-4 py-6 grid grid-cols-1 lg:grid-cols-5 gap-5">

    <!-- LEFT: Form -->
    <section class="lg:col-span-2">
      <div class="card">
        <div class="section-head">
          <span>✨ New Content Job</span>
        </div>
        <form id="gen-form" class="p-5 space-y-4">

          <div>
            <div class="label">Character Image URL *</div>
            <input id="character_url" type="url" required class="input"
                   placeholder="https://… character reference image" />
          </div>

          <div>
            <div class="label">Setting Image URL</div>
            <input id="setting_url" type="url" class="input"
                   placeholder="Optional: background / location URL" />
          </div>

          <div>
            <div class="label">Item / Product Image URL</div>
            <input id="item_url" type="url" class="input"
                   placeholder="Optional: product the character holds" />
          </div>

          <div class="grid grid-cols-3 gap-3">
            <div>
              <div class="label">Images</div>
              <select id="num_images" class="input">
                <option>1</option><option selected>2</option>
                <option>3</option><option>4</option><option>5</option>
              </select>
            </div>
            <div>
              <div class="label">Videos</div>
              <select id="num_videos" class="input">
                <option selected>0</option>
                <option>1</option><option>2</option><option>3</option>
              </select>
            </div>
            <div>
              <div class="label">Ratio</div>
              <select id="aspect_ratio" class="input">
                <option>9:16</option><option>16:9</option><option>1:1</option>
              </select>
            </div>
          </div>

          <div>
            <div class="label">Creative Direction *</div>
            <textarea id="creative_direction" rows="4" required class="input"
                      placeholder="Describe outfit, mood, setting, actions, brand mentions…"></textarea>
          </div>

          <div>
            <div class="label">Character Brief</div>
            <textarea id="character_brief" rows="3" class="input"
                      placeholder="Name, age, personality, voice style, target audience…"></textarea>
          </div>

          <button type="submit" id="submit-btn" class="btn-primary">
            <span id="submit-label">🚀 Generate Content</span>
          </button>

        </form>
      </div>
    </section>

    <!-- RIGHT: Jobs + Details -->
    <section class="lg:col-span-3 flex flex-col gap-5">

      <!-- Jobs list -->
      <div class="card">
        <div class="section-head">
          <span>📋 Jobs</span>
          <button onclick="loadJobs()" class="text-xs text-slate-500 hover:text-slate-300">↻ Refresh</button>
        </div>
        <div id="jobs-list">
          <div class="text-center text-slate-600 text-sm py-10">
            No jobs yet — submit the form to start.
          </div>
        </div>
      </div>

      <!-- Job detail panel (hidden until a job is selected) -->
      <div id="detail-panel" class="card hidden">
        <div class="section-head">
          <span>📊 Job <span id="detail-id" class="text-slate-400 font-mono font-normal"></span></span>
          <span id="detail-status" class="badge badge-queued">queued</span>
        </div>

        <!-- Posts grid -->
        <div id="posts-grid" class="p-4 grid grid-cols-1 sm:grid-cols-2 gap-3"></div>

        <!-- Log viewer -->
        <div>
          <div class="section-head" style="font-size:12px; padding:10px 20px;">
            <span>📄 Live Logs</span>
            <button onclick="document.getElementById('log-box').innerHTML=''"
                    class="text-xs text-slate-600 hover:text-slate-400">Clear</button>
          </div>
          <div id="log-box"></div>
        </div>
      </div>

    </section>
  </main>

  <!-- ── JS ─────────────────────────────────────────────────────────── -->
  <script>
    let currentJobId = null;
    let logEs = null;
    let pollTimer = null;

    // ── Health check ────────────────────────────────────────────────
    async function checkHealth() {
      try {
        const h = await fetch('/api/health').then(r => r.json());
        const aiLabel = h.anthropic ? `Claude (${h.model})` : h.groq ? `Groq (${h.model})` : h.openai ? `OpenAI (${h.model})` : 'No AI key';
        const aiOk    = h.anthropic || h.groq || h.openai;
        document.getElementById('api-indicators').innerHTML = [
          dot(aiOk,       aiLabel),
          dot(h.kie_ai,   'kie.ai'),
          dot(h.telegram, 'Telegram'),
        ].join('');
      } catch {}
    }

    function dot(ok, label) {
      return `<span class="flex items-center gap-1">
                <span class="dot ${ok ? 'dot-ok' : 'dot-bad'}"></span>${label}
              </span>`;
    }

    // ── Form submit ──────────────────────────────────────────────────
    document.getElementById('gen-form').addEventListener('submit', async e => {
      e.preventDefault();
      const btn  = document.getElementById('submit-btn');
      const lbl  = document.getElementById('submit-label');
      btn.disabled = true;
      lbl.textContent = '⏳ Starting…';

      const body = {
        character_url:      v('character_url'),
        setting_url:        v('setting_url'),
        item_url:           v('item_url'),
        num_images:         parseInt(v('num_images')),
        num_videos:         parseInt(v('num_videos')),
        aspect_ratio:       v('aspect_ratio'),
        creative_direction: v('creative_direction'),
        character_brief:    v('character_brief'),
      };

      try {
        const res  = await fetch('/api/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Request failed');
        selectJob(data.job_id);
        loadJobs();
      } catch (err) {
        alert('Error: ' + err.message);
      } finally {
        btn.disabled = false;
        lbl.textContent = '🚀 Generate Content';
      }
    });

    function v(id) { return document.getElementById(id).value.trim(); }

    // ── Jobs list ────────────────────────────────────────────────────
    async function loadJobs() {
      const jobs = await fetch('/api/jobs').then(r => r.json()).catch(() => []);
      const el   = document.getElementById('jobs-list');

      if (!jobs.length) {
        el.innerHTML = '<div class="text-center text-slate-600 text-sm py-10">No jobs yet.</div>';
        return;
      }

      jobs.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));

      el.innerHTML = jobs.map(job => {
        const dir = (job.form_data?.creative_direction || '').substring(0, 60);
        const cnt = (job.posts || []).length;
        const active = job.id === currentJobId ? 'active' : '';
        return `
          <div class="job-row ${active}" onclick="selectJob('${job.id}')">
            <div class="flex items-center gap-2">
              <span class="text-sm font-mono text-slate-300">#${job.id}</span>
              <span class="badge badge-${job.status}">${job.status}</span>
              <span class="ml-auto text-xs text-slate-600">${cnt} post${cnt !== 1 ? 's' : ''}</span>
            </div>
            <div class="text-xs text-slate-600 mt-1 truncate">${dir}…</div>
            <div class="text-xs text-slate-700 mt-0.5">${fmtDate(job.created_at)}</div>
          </div>`;
      }).join('');
    }

    function fmtDate(iso) {
      try { return new Date(iso).toLocaleString(); } catch { return iso; }
    }

    // ── Select + stream a job ────────────────────────────────────────
    async function selectJob(jobId) {
      currentJobId = jobId;

      // Reset panel
      document.getElementById('detail-panel').classList.remove('hidden');
      document.getElementById('detail-id').textContent = jobId;
      document.getElementById('log-box').innerHTML = '';
      document.getElementById('posts-grid').innerHTML = '';

      // Stop old streams
      if (logEs)    { logEs.close(); logEs = null; }
      if (pollTimer){ clearInterval(pollTimer); pollTimer = null; }

      await refreshDetail(jobId);

      // SSE log stream
      logEs = new EventSource(`/api/job/${jobId}/logs`);
      const logBox = document.getElementById('log-box');
      logEs.onmessage = e => {
        try {
          const entry = JSON.parse(e.data);
          if (entry.msg === '__STREAM_END__') { logEs.close(); return; }
          const t = new Date(entry.time).toLocaleTimeString();
          const div = document.createElement('div');
          div.className = `log-${entry.level}`;
          div.textContent = `[${t}] ${entry.msg}`;
          logBox.appendChild(div);
          logBox.scrollTop = logBox.scrollHeight;
        } catch {}
      };
      logEs.onerror = () => logEs.close();

      // Poll job state every 3s for post updates
      pollTimer = setInterval(() => refreshDetail(jobId), 3000);
    }

    async function refreshDetail(jobId) {
      try {
        const job = await fetch(`/api/job/${jobId}`).then(r => r.json());

        // Status badge
        const sb = document.getElementById('detail-status');
        sb.className = `badge badge-${job.status}`;
        sb.textContent = job.status;

        renderPosts(job.posts || []);

        if (['done','failed'].includes(job.status)) {
          clearInterval(pollTimer);
          pollTimer = null;
          loadJobs();
        }
      } catch {}
    }

    // ── Post cards ───────────────────────────────────────────────────
    function renderPosts(posts) {
      const grid = document.getElementById('posts-grid');
      if (!posts.length) {
        grid.innerHTML = '<div class="col-span-2 text-center text-slate-600 text-sm py-6">Generating…</div>';
        return;
      }
      grid.innerHTML = posts.map((p, i) => `
        <div class="post-card">
          ${p.image_url
            ? `<div class="relative">
                 <img src="${p.image_url}" class="post-thumb" alt="${esc(p.title)}" />
                 ${p.video_url ? '<div style="position:absolute;top:6px;right:6px;background:rgba(0,0,0,.7);border-radius:4px;padding:2px 6px;font-size:10px;">🎬</div>' : ''}
               </div>`
            : `<div class="post-thumb-placeholder">
                 <span style="font-size:28px;">${statusEmoji(p.status)}</span>
               </div>`}
          <div class="p-3">
            <div class="flex items-center justify-between gap-2 mb-1">
              <span class="text-sm font-semibold text-white truncate">${esc(p.title || 'Untitled')}</span>
              <span class="badge badge-${p.post_type || 'image'}" style="flex-shrink:0">${p.post_type || 'image'}</span>
            </div>
            <div class="text-xs text-slate-500 line-clamp-2">${esc(p.caption || '')}</div>
            ${p.status ? `<div class="mt-2"><span class="badge badge-${p.status}">${p.status}</span></div>` : ''}
            <div class="mt-2 flex flex-col gap-1">
              ${p.image_url ? `<a href="${p.image_url}" target="_blank" class="text-xs text-blue-400 hover:text-blue-300">🖼 View Image →</a>` : ''}
              ${p.video_url ? `<a href="${p.video_url}" target="_blank" class="text-xs text-purple-400 hover:text-purple-300">🎬 View Video →</a>` : ''}
            </div>
          </div>
        </div>
      `).join('');
    }

    function statusEmoji(s) {
      const m = { done:'✅', image_done:'🖼️', generating_image:'⏳',
                  generating_video:'🎬', image_failed:'❌', video_failed:'❌' };
      return m[s] || '⏳';
    }

    function esc(s) {
      return String(s || '')
        .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
        .replace(/"/g,'&quot;');
    }

    // ── Boot ─────────────────────────────────────────────────────────
    checkHealth();
    loadJobs();
    setInterval(loadJobs, 8000);
    setInterval(checkHealth, 30000);
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
