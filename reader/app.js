/* taichi-academy — static multi-project reader. Vanilla JS, no build step.
   Series landing (card grid from manifest.js) -> per-project reader
   (sticky TOC + one calm step) -> deterministic check-my-code.
   Project data.js/fulls.js are injected on demand; progress is per-project. */

(function () {
  const PROJECTS = window.ACADEMY_PROJECTS || [];
  const app = document.getElementById("app");
  const topRight = document.getElementById("topRight");
  const progressFill = document.getElementById("progressFill");

  let SOT = null;   // current project's window.ACADEMY_SOT[id]
  let FULLS = null; // current project's window.ACADEMY_FULLS[id]
  let state = { view: "menu" };

  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const playable = (ch) => Array.isArray(ch.steps) && ch.steps.length > 0;
  const byId = (id) => SOT.chapters.find((c) => c.id === id);
  const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

  const KEY = () => "academy:" + SOT.project;
  const load = () => { try { return JSON.parse(localStorage.getItem(KEY())) || {}; } catch { return {}; } };
  const save = () => { if (SOT) localStorage.setItem(KEY(), JSON.stringify(state)); };

  /* ---------- dynamic project loading (script injection works from file://) ---------- */
  const loaded = {};
  function inject(src) {
    return new Promise((res, rej) => {
      const s = document.createElement("script");
      s.src = src; s.onload = res; s.onerror = () => rej(new Error("failed to load " + src));
      document.body.appendChild(s);
    });
  }
  async function openProject(id) {
    const meta = PROJECTS.find((p) => p.id === id);
    if (!meta || meta.status !== "available") return;
    if (!loaded[id]) {
      app.innerHTML = `<div class="coming"><h3>Loading ${esc(meta.title)}…</h3></div>`;
      await inject(`projects/${id}/data.js`);
      await inject(`projects/${id}/fulls.js`);
      loaded[id] = true;
    }
    SOT = window.ACADEMY_SOT[id];
    FULLS = (window.ACADEMY_FULLS || {})[id] || null;
    state = Object.assign({ view: "reader", chapter: 1, step: 0, unlocked: 1, doneChapters: [] }, load(), { view: "reader" });
    localStorage.setItem("academy:last", id);
    location.hash = "#/" + id;
    render();
  }
  function closeProject() {
    save();
    SOT = null; FULLS = null;
    state = { view: "menu" };
    location.hash = "";
    render();
  }

  function setProgress() {
    const pct = SOT ? (state.doneChapters.length / SOT.chapters.length) * 100 : 0;
    progressFill.style.width = pct + "%";
  }

  function render() {
    save(); setProgress();
    if (state.view === "menu") { topRight.innerHTML = "Learn the GPU by typing it"; return renderMenu(); }
    if (state.view === "setup") { topRight.innerHTML = "Getting set up"; return renderSetup(); }
    topRight.innerHTML = `<b>${esc(SOT.title)}</b> · ${esc(cap(SOT.tier))}`;
    renderReader();
  }

  /* ---------- series landing ---------- */
  function renderMenu() {
    const ready = PROJECTS.filter((p) => p.status === "available").length;
    const cards = PROJECTS.map((p) => {
      const avail = p.status === "available";
      return `<div class="card ${avail ? "" : "is-coming"}" data-proj="${p.id}" ${avail ? "" : 'aria-disabled="true"'}>
        <span class="badge">✦ ${esc(cap(p.tier))} · ${esc(p.language)}${avail ? "" : " · coming"}</span>
        <h2>${esc(p.title)}</h2>
        <p class="desc">${esc(p.pitch)}</p>
        <div class="meta">${avail ? `<span class="go">Start building →</span>` : `<span>In the roadmap</span>`}</div>
      </div>`;
    }).join("");
    app.innerHTML = `
      <section class="hero">
        <div class="eyebrow">MoreSalamander StudioLabs</div>
        <h1>Teach your GPU<br/>to grow worlds.</h1>
        <p>${PROJECTS.length} simulation projects, one calm step at a time — reaction-diffusion to universe sandbox. You type every line; a deterministic checker has your back. ${ready} ready today.</p>
      </section>
      <div class="projects">${cards}</div>
      <p class="setup-cta">First time here? <a id="setupLink">Set up the repo first →</a></p>`;
    document.querySelectorAll(".card[data-proj]").forEach((el) => {
      const id = el.dataset.proj;
      const meta = PROJECTS.find((p) => p.id === id);
      if (meta && meta.status === "available") el.onclick = () => openProject(id);
    });
    document.getElementById("setupLink").onclick = () => { state = { view: "setup" }; render(); };
  }

  /* ---------- setup page (static, terse — this repo, not generic) ---------- */
  function renderSetup() {
    app.innerHTML = `<div class="setup">
      <div class="crumbs"><a id="home">← All projects</a></div>
      <div class="chat">
        <div class="bubble bot">One-time setup — five commands in a terminal, from wherever you keep your projects:</div>
        <div class="bubble bot"><code>git clone https://github.com/MoreSalamander/taichi-academy.git<br>
cd taichi-academy<br>
python3.11 -m venv .venv<br>
source .venv/bin/activate<br>
pip install -e ".[dev]"</code></div>
        <div class="bubble bot">Sanity check — this should open a window of living coral (Esc closes it):<br><code>python projects/01-reaction-diffusion/reference/gray_scott.py</code></div>
        <div class="bubble bot">When you build along with a project, type into its <b>my_build/</b> folder — e.g. create <code>projects/01-reaction-diffusion/my_build/gray_scott.py</code> and run it from there with <code>python gray_scott.py</code>. The reference stays pristine; my_build is yours.</div>
      </div>
      <div class="setup-controls"><button class="btn primary" id="build">I'm set up — show me the projects</button></div>
    </div>`;
    document.getElementById("home").onclick = () => { state = { view: "menu" }; render(); };
    document.getElementById("build").onclick = () => { state = { view: "menu" }; render(); };
  }

  /* ---------- reader ---------- */
  function renderReader() {
    const ch = byId(state.chapter);

    const toc = SOT.chapters.map((c) => {
      const done = state.doneChapters.includes(c.id);
      const cur = c.id === state.chapter;
      const locked = c.id > state.unlocked;
      const cls = ["", locked ? "locked" : "is-unlocked", cur ? "current" : "", done ? "done" : ""].join(" ");
      const ic = done ? "✓" : locked ? "🔒" : c.id;
      return `<li class="${cls}" data-ch="${c.id}"><span class="ic">${ic}</span><span>${esc(c.title)}</span></li>`;
    }).join("");

    const sidebar = `
      <aside class="sidebar">
        <div class="proj-title">${esc(SOT.title)}</div>
        <div class="proj-sub">${esc(cap(SOT.tier))} · ${esc(SOT.language)} · ${SOT.chapters.length} chapters</div>
        <ul class="toc">${toc}</ul>
      </aside>`;

    let main;
    if (!playable(ch)) {
      main = `<div>
        <div class="crumbs"><a id="home">← All projects</a></div>
        <div class="coming"><h3>Coming soon</h3><p>“${esc(ch.title)}” isn't written yet.</p></div>
      </div>`;
    } else {
      const total = ch.steps.length;
      const idx = Math.min(state.step, total - 1);
      const s = ch.steps[idx];
      const last = idx === total - 1;
      const fullFiles = (FULLS && FULLS[ch.id - 1] && FULLS[ch.id - 1][idx]) || null;
      const codeBox = (fname, code) => `<div class="codewrap"><div class="chrome"><span class="d"></span><span class="d"></span><span class="d"></span><span class="fname">${esc(fname)}</span></div><pre class="language-python"><code class="language-python">${esc(code)}</code></pre></div>`;
      const fullHTML = fullFiles
        ? `<details class="fullcode"><summary>📄 Your whole file so far</summary>${Object.entries(fullFiles).map(([f, c]) => codeBox(f, c)).join("")}</details>`
        : "";
      main = `<div>
        <div class="crumbs"><a id="home">← All projects</a></div>
        <div class="chapter-head">
          <div class="ch-num">Chapter ${ch.id}</div>
          <h2>${esc(ch.title)}</h2>
          <p class="build">You'll build ${esc(ch.build)}</p>
        </div>
        <div class="step-meter"><span>Step ${idx + 1} of ${total}</span><span class="bar"><i style="width:${((idx + 1) / total) * 100}%"></i></span></div>

        <article class="step">
          <span class="kicker">Step ${idx + 1}</span>
          <h3>${esc(s.title)}</h3>

          <div class="label">What you're adding</div>
          <p class="adding">${esc(s.adding)}</p>

          <div class="label">The code</div>
          ${codeBox(SOT.file || "main.py", s.code)}
          ${fullHTML}

          <div class="label">What it does</div>
          <p>${esc(s.does)}</p>

          <div class="label">Why it matters</div>
          <p>${esc(s.why)}</p>

          <div class="label">Run it — what you'll see</div>
          <div class="callout see"><span class="ci">👁</span><span>${esc(s.see)}</span></div>

          <div class="label">Checkpoint</div>
          <div class="callout check"><span class="ci">✅</span><span>${esc(s.checkpoint)}</span></div>

          ${(s.recovery && s.recovery.length) ? `<details class="recovery"><summary>It's not right?</summary><ul>${s.recovery.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></details>` : ""}
        </article>

        ${helperHTML()}

        <div class="nav">
          <button class="btn" id="prev" ${idx === 0 ? "disabled" : ""}>← Previous</button>
          <button class="btn primary" id="next">${last ? "Finish chapter ✓" : "Next step →"}</button>
        </div>
        ${last ? `<div class="beat"><h3>🧪 ${esc(ch.beat)}</h3><p>You can stop here — it runs.</p></div>` : ""}
      </div>`;
    }

    app.innerHTML = `<div class="reader">${sidebar}${main}</div>`;

    if (window.Prism) Prism.highlightAllUnder(app);
    wireTOC();
    const home = document.getElementById("home"); if (home) home.onclick = closeProject;
    if (playable(ch)) wireStep(ch);
  }

  function wireStep(ch) {
    const total = ch.steps.length;
    const idx = Math.min(state.step, total - 1);
    const s = ch.steps[idx];
    const last = idx === total - 1;
    document.getElementById("prev").onclick = () => { if (idx > 0) { state.step = idx - 1; render(); } };
    document.getElementById("next").onclick = () => {
      if (!last) { state.step = idx + 1; render(); return; }
      if (!state.doneChapters.includes(ch.id)) state.doneChapters.push(ch.id);
      const nx = byId(ch.id + 1);
      if (nx) { state.unlocked = Math.max(state.unlocked, ch.id + 1); state.chapter = ch.id + 1; state.step = 0; }
      render();
    };
    wireHelper(s, ch);
  }

  function helperHTML() {
    return `<details class="helper">
      <summary>🛟 Helper — want a check, or stuck?</summary>
      <div class="body">
        <p class="hint">Paste your code (or the red error you got), then choose:</p>
        <textarea id="paste" placeholder="Paste your ${SOT && SOT.file ? esc(SOT.file) : "file"} — or an error message…"></textarea>
        <div class="hbtns">
          <button class="btn" id="check">Check my code</button>
          <button class="btn primary" id="help">Something's not right</button>
        </div>
        <div class="result" id="result"></div>
        <p class="note">Your code is checked instantly in your browser. When the live helper is on, it adds a plain-language hand.</p>
      </div>
    </details>`;
  }

  function wireHelper(s, ch) {
    const checkBtn = document.getElementById("check");
    const helpBtn = document.getElementById("help");
    const result = document.getElementById("result");
    if (!checkBtn) return;

    const norm = (t) => t.replace(/\s+/g, " ").trim();
    const br = (t) => esc(t).replace(/\n/g, "<br>");
    const ctx = { project: SOT.title, chapter: `${ch.id} — ${ch.title}`, stepTitle: s.title, code: s.code, see: s.see, recovery: s.recovery || [] };
    const set = (cls, html) => { result.className = "result " + cls; result.innerHTML = html; };

    function deterministic(pasted) {
      const lines = s.code.split("\n").map(norm).filter(Boolean);
      const hay = norm(pasted);
      const missing = lines.filter((ln) => !hay.includes(ln));
      return { ok: missing.length === 0, missing };
    }

    async function ask(payload) {
      const r = await fetch("/api/chat", { method: "POST", headers: { "content-type": "application/json" }, body: JSON.stringify(payload) });
      if (!r.ok) throw new Error("http " + r.status);
      const d = await r.json();
      if (!d.text) throw new Error(d.error || "no text");
      return d.text;
    }

    checkBtn.onclick = async () => {
      const pasted = document.getElementById("paste").value || "";
      if (!pasted.trim()) { set("no", "Paste your code first and I'll check it."); return; }
      const f = deterministic(pasted);
      const verdict = f.ok
        ? "✓ This step's code is in there."
        : "I don't see this step's line yet — it should include:<br><code>" + esc(f.missing[0]) + "</code>";
      set(f.ok ? "ok" : "no", verdict + `<div class="voice" id="voice"><span class="spin"></span> checking with the helper…</div>`);
      const findings = f.ok ? "All of this step's lines are present." : ("Missing line(s): " + f.missing.join(" | "));
      try {
        const text = await ask({ mode: "check", ctx, pasted, findings });
        const v = document.getElementById("voice"); if (v) v.innerHTML = br(text);
      } catch {
        const v = document.getElementById("voice"); if (v) v.remove();
      }
    };

    helpBtn.onclick = async () => {
      const pasted = document.getElementById("paste").value || "";
      set("", `<span class="spin"></span> looking at it…`);
      try {
        const text = await ask({ mode: "help", ctx, pasted });
        set("ok", br(text));
      } catch {
        const fixes = (s.recovery && s.recovery.length)
          ? "<ul>" + s.recovery.map((r) => "<li>" + esc(r) + "</li>").join("") + "</ul>"
          : "Re-check this step's code against the snippet above.";
        set("no", "<b>The live helper is offline</b> — but here are the usual fixes for this step:" + fixes);
      }
    };
  }

  function wireTOC() {
    document.querySelectorAll(".toc li").forEach((el) => {
      const id = +el.dataset.ch;
      if (id > state.unlocked) return;
      el.onclick = () => { state.chapter = id; state.step = 0; render(); };
    });
  }

  /* ---------- ambient particle field (reaction-diffusion vibes) ---------- */
  (function sky() {
    const c = document.getElementById("sky");
    const ctx = c.getContext("2d");
    let w, h, stars, bursts = [];
    function resize() {
      w = c.width = innerWidth; h = c.height = innerHeight;
      stars = Array.from({ length: 150 }, () => ({
        x: Math.random() * w, y: Math.random() * h,
        r: Math.random() * 1.3 + 0.3,
        a: Math.random() * 0.5 + 0.15,
        ph: Math.random() * 6.28, sp: Math.random() * 0.02 + 0.005
      }));
    }
    function spawnBurst() {
      const bx = w * (0.15 + Math.random() * 0.7), by = h * (0.1 + Math.random() * 0.3);
      const hue = Math.random() < 0.5 ? "245,189,84" : "142,123,255";
      const n = 26, parts = [];
      for (let i = 0; i < n; i++) {
        const ang = (i / n) * 6.28, sp = Math.random() * 1.6 + 0.6;
        parts.push({ x: bx, y: by, vx: Math.cos(ang) * sp, vy: Math.sin(ang) * sp });
      }
      bursts.push({ parts, life: 1, hue });
    }
    let t = 0;
    function frame() {
      ctx.clearRect(0, 0, w, h);
      t += 1;
      for (const s of stars) {
        s.ph += s.sp;
        const a = s.a * (0.6 + 0.4 * Math.sin(s.ph));
        ctx.globalAlpha = a; ctx.fillStyle = "#cdd2ff";
        ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, 6.28); ctx.fill();
      }
      for (const b of bursts) {
        b.life -= 0.012;
        for (const p of b.parts) { p.x += p.vx; p.y += p.vy; p.vy += 0.01; }
        ctx.globalAlpha = Math.max(0, b.life) * 0.5;
        ctx.fillStyle = "rgba(" + b.hue + ",1)";
        for (const p of b.parts) { ctx.beginPath(); ctx.arc(p.x, p.y, 1.4, 0, 6.28); ctx.fill(); }
      }
      bursts = bursts.filter((b) => b.life > 0);
      if (t % 360 === 0 && Math.random() < 0.8) spawnBurst();
      ctx.globalAlpha = 1;
      requestAnimationFrame(frame);
    }
    addEventListener("resize", resize);
    resize(); setTimeout(spawnBurst, 1200); frame();
  })();

  /* ---------- boot: honor #/<id>, else land on the series menu ---------- */
  const hashId = (location.hash.match(/^#\/(.+)$/) || [])[1];
  const boot = hashId && PROJECTS.some((p) => p.id === hashId && p.status === "available") ? hashId : null;
  if (boot) { openProject(boot); } else { render(); }
})();
