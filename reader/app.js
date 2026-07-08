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
    document.getElementById("setupLink").onclick = () => startSetup();
  }

  /* ---------- setup interview: deterministic readiness machine, chat UI ----------
     One shared .venv serves every project (see CLAUDE.md) — this walkthrough runs
     once, from a genuinely empty machine, not per-project. */
  const sBot = (t) => state.setup.log.push({ who: "bot", text: t });
  const sYou = (t) => state.setup.log.push({ who: "you", text: t });
  const OS = {
    mac: {
      where: "your Mac", term: "Press ⌘ + Space, type Terminal, and press Enter.",
      pycmd: "python3.11 --version", pyinstall: "https://www.python.org/downloads/release/python-3119/ (or `brew install python@3.11`)",
      venvcmd: "python3.11 -m venv .venv", activatecmd: "source .venv/bin/activate", prompt: "(.venv)",
    },
    other: {
      where: "your PC", term: "Click Start, type PowerShell, and press Enter.",
      pycmd: "py -3.11 --version", pyinstall: "https://www.python.org/downloads/release/python-3119/ — tick \"Add Python to PATH\" during install",
      venvcmd: "py -3.11 -m venv .venv", activatecmd: ".venv\\Scripts\\activate", prompt: "(.venv)",
    },
  };
  const linkify = (s) => s.replace(/(https?:\/\/[^\s]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');

  function startSetup() {
    state = { view: "setup", setup: { os: null, stage: "os", log: [] } };
    sBot("Let's set this up from a clean machine — about five minutes, one step at a time. First: are you on a Mac, or Windows/Linux?");
    render();
  }

  function renderSetup() {
    const st = state.setup;
    const fmtBot = (t) => linkify(esc(t)).replace(/\n/g, "<br>");
    const fmtYou = (t) => esc(t).replace(/\n/g, "<br>");
    const bubbles = st.log.map((m) => `<div class="bubble ${m.who}">${m.who === "bot" ? fmtBot(m.text) : fmtYou(m.text)}</div>`).join("");
    app.innerHTML = `<div class="setup">
      <div class="crumbs"><a id="home">← All projects</a></div>
      <div class="chat">${bubbles}</div>
      <div class="setup-controls" id="ctrls"></div>
    </div>`;
    document.getElementById("home").onclick = () => { state = { view: "menu" }; render(); };
    const c = document.getElementById("ctrls");
    const btn = (id, label, primary) => `<button class="btn ${primary ? "primary" : ""}" id="${id}">${label}</button>`;
    const on = (id, fn) => { const e = document.getElementById(id); if (e) e.onclick = fn; };

    if (st.stage === "os") {
      c.innerHTML = btn("os-mac", "🍎 Mac", true) + btn("os-other", "🪟 Windows / 🐧 Linux");
      on("os-mac", () => chooseOS("mac")); on("os-other", () => chooseOS("other"));
    } else if (st.stage === "python") {
      c.innerHTML = `<textarea id="pyout" placeholder="Paste what the command printed…"></textarea><div class="hbtns">${btn("py-go", "Check it", true)}</div>`;
      on("py-go", submitPython);
    } else if (st.stage === "python_unclear") {
      c.innerHTML = btn("py-have", "It showed Python 3.11.x", true) + btn("py-wrong", "It showed a different version") + btn("py-none", "It showed an error");
      on("py-have", () => pythonResult("ok")); on("py-wrong", () => pythonResult("wrong")); on("py-none", () => pythonResult("missing"));
    } else if (st.stage === "python_install") {
      c.innerHTML = btn("py-again", "I've installed 3.11 — check again", true);
      on("py-again", recheckPython);
    } else if (st.stage === "clone") {
      c.innerHTML = btn("cloned", "Done — I'm inside the taichi-academy folder", true);
      on("cloned", () => { sYou("Cloned and cd'd in"); goVenvCreate(); render(); });
    } else if (st.stage === "venv_create") {
      c.innerHTML = btn("venv-made", "Ran it, no red text", true);
      on("venv-made", () => { sYou("Created the venv"); goVenvActivate(); render(); });
    } else if (st.stage === "venv_activate") {
      c.innerHTML = `<textarea id="promptout" placeholder="Paste what your prompt line looks like now…"></textarea><div class="hbtns">${btn("prompt-go", "Check it", true)}</div>`;
      on("prompt-go", submitActivate);
    } else if (st.stage === "activate_unclear") {
      c.innerHTML = btn("prompt-yes", `Yes, it shows ${OS[st.os].prompt}`, true) + btn("prompt-no", "No, it looks the same as before");
      on("prompt-yes", () => activateResult(true)); on("prompt-no", () => activateResult(false));
    } else if (st.stage === "pip_install") {
      c.innerHTML = btn("pip-done", "It finished, no red text", true);
      on("pip-done", () => { sYou("Installed dependencies"); goVerify(); render(); });
    } else if (st.stage === "verify") {
      c.innerHTML = `<textarea id="verout" placeholder="Paste what printed…"></textarea><div class="hbtns">${btn("ver-go", "Check it", true)}</div>`;
      on("ver-go", submitVerify);
    } else if (st.stage === "verify_unclear") {
      c.innerHTML = btn("ver-yes", "Yes, something like 1.7.4", true) + btn("ver-no", "No, an error");
      on("ver-yes", () => verifyResult(true)); on("ver-no", () => verifyResult(false));
    } else if (st.stage === "ready") {
      c.innerHTML = btn("build", "Start building 🧪", true);
      on("build", () => { state = { view: "menu" }; render(); });
    }
    const chat = document.querySelector(".chat"); if (chat) chat.scrollTop = chat.scrollHeight;
  }

  function chooseOS(os) {
    state.setup.os = os; sYou(os === "mac" ? "Mac" : "Windows / Linux");
    const o = OS[os];
    sBot(`Good. First, does this machine already have Python 3.11 — the exact version this repo needs (not 3.12, not 3.9).\n1) ${o.term}\n2) Type this and press Enter:   ${o.pycmd}\n3) Paste whatever it says back to me below.`);
    state.setup.stage = "python"; render();
  }
  function submitPython() {
    const out = (document.getElementById("pyout").value || "").trim();
    if (!out) return;
    sYou(out);
    if (/python\s*3\.11(\.|$| )/i.test(out)) return pythonResult("ok");
    if (/python\s*3\.\d+/i.test(out)) return pythonResult("wrong");
    if (/command not found|not recognized|no such|not found/i.test(out)) return pythonResult("missing");
    sBot("I couldn't read that for sure. Did it show Python 3.11.something, a different version, or an error?");
    state.setup.stage = "python_unclear"; render();
  }
  function pythonResult(kind) {
    if (kind === "ok") {
      sBot("Python 3.11 confirmed. ✓");
      goClone();
    } else if (kind === "wrong") {
      const o = OS[state.setup.os];
      sBot(`That's a different Python version — this repo is pinned to 3.11 exactly (taichi's wheels and the .python-version file both expect it). Install 3.11 alongside whatever you have — it won't conflict:\n1) Go to ${o.pyinstall}\n2) Download and run the installer for 3.11.\nThen come back and we'll check again.`);
      state.setup.stage = "python_install";
    } else {
      const o = OS[state.setup.os];
      sBot(`No problem — let's install it:\n1) Go to ${o.pyinstall}\n2) Download and run the installer.` +
        (state.setup.os === "other" ? "\n3) IMPORTANT: tick \"Add Python to PATH\" at the bottom before clicking Install." : "") +
        "\nThen come back and we'll check again.");
      state.setup.stage = "python_install";
    }
    render();
  }
  function recheckPython() {
    sYou("Installed Python 3.11");
    sBot(`Let's confirm. Type this, press Enter, and paste the result:   ${OS[state.setup.os].pycmd}`);
    state.setup.stage = "python"; render();
  }
  function goClone() {
    sBot('Next, get the code (skip this if you already have the folder open):\n1) In the same terminal, run:\ngit clone https://github.com/MoreSalamander/taichi-academy.git\n2) Then:\ncd taichi-academy');
    state.setup.stage = "clone";
  }
  function goVenvCreate() {
    const o = OS[state.setup.os];
    sBot(`Now the venv — a private sandbox of dependencies just for this repo, so taichi here never clashes with anything else on your machine. Every project in the series shares this ONE venv. Create it:\n${o.venvcmd}\nYou'll see no output at all if it worked — that's success.`);
    state.setup.stage = "venv_create";
  }
  function goVenvActivate() {
    const o = OS[state.setup.os];
    sBot(`Now switch your terminal INTO that sandbox — you have to do this every time you open a new terminal:\n${o.activatecmd}\nLook at your prompt line afterward and paste what it looks like now.`);
    state.setup.stage = "venv_activate";
  }
  function submitActivate() {
    const out = (document.getElementById("promptout").value || "").trim();
    if (!out) return;
    sYou(out);
    if (out.includes(".venv") || /^\(\.venv\)/.test(out)) return activateResult(true);
    state.setup.stage = "activate_unclear"; render();
  }
  function activateResult(ok) {
    if (ok) {
      sBot("The (.venv) tag means you're in the sandbox now — every python and pip command from here goes into it. ✓");
      goPipInstall();
    } else {
      const o = OS[state.setup.os];
      sBot(`It should start with (.venv) — that's how you'll always know you're in the right place. Run the activate command again and check your prompt:\n${o.activatecmd}`);
      state.setup.stage = "venv_activate";
    }
    render();
  }
  function goPipInstall() {
    sBot('Last install step — this pulls in taichi and numpy (taichi\'s wheel is sizeable, give it a minute):\npip install -e ".[dev]"\nWait for it to finish and land back at your prompt with no red text.');
    state.setup.stage = "pip_install";
  }
  function goVerify() {
    sBot('One check that it all actually works — paste back what this prints:\npython -c "import taichi; print(taichi.__version__)"');
    state.setup.stage = "verify";
  }
  function submitVerify() {
    const out = (document.getElementById("verout").value || "").trim();
    if (!out) return;
    sYou(out);
    if (/^\(?\d+,?\s*\d+,?\s*\d+\)?$/.test(out) || /\d+\.\d+\.\d+/.test(out)) return verifyResult(true);
    if (/no module named|traceback|error/i.test(out)) return verifyResult(false);
    state.setup.stage = "verify_unclear"; render();
  }
  function verifyResult(ok) {
    if (ok) {
      sBot("That's a real Taichi version string — you're fully set up. ✓ Python 3.11, the venv, taichi: all working.");
      goReady();
    } else {
      sBot('That means the venv isn\'t active in this terminal, or the install step didn\'t finish. Re-activate and reinstall:\nsource .venv/bin/activate   (or .venv\\Scripts\\activate on Windows)\npip install -e ".[dev]"\nThen try the import check again.');
      state.setup.stage = "pip_install";
    }
    render();
  }
  function goReady() {
    sBot('You\'re ready. One habit for every project: type your own code into its my_build/ folder (e.g. projects/01-reaction-diffusion/my_build/gray_scott.py) and run it from there — the reference/ files stay untouched as your answer key. Let\'s build something.');
    state.setup.stage = "ready";
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
