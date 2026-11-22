/* Epistemic — frontend for the LLM uncertainty pipeline.
 * Talks to webapp/server.py:  GET /api/health, POST /api/analyze.
 * Everything rendered in the results view comes from the backend record. */
(() => {
  "use strict";

  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finePointer = matchMedia("(hover: hover) and (pointer: fine)").matches;

  /** Tiny DOM builder. Text is always set via textContent (never innerHTML). */
  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v == null || v === false) continue;
        if (k === "class") el.className = v;
        else if (k === "style") el.style.cssText = v;
        else if (k === "text") el.textContent = v;
        else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else el.setAttribute(k, v === true ? "" : v);
      }
    }
    for (const kid of kids.flat()) {
      if (kid == null || kid === false) continue;
      el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return el;
  }

  /* =====================================================================
   * 1. Dithered landscape background (procedural + ordered Bayer dither)
   * ===================================================================== */
  const Scene = (() => {
    const canvas = $("#scene");
    const ctx = canvas.getContext("2d", { alpha: false });
    const PALETTE = [
      [3, 5, 34], [6, 11, 69], [12, 20, 150], [22, 38, 230], [52, 71, 255], [90, 51, 224],
      [124, 72, 214], [11, 52, 52], [24, 104, 92], [52, 170, 146], [126, 232, 200], [214, 246, 220],
      [255, 225, 74], [242, 150, 58], [236, 240, 255], [1, 2, 18],
    ];
    // RGB → palette lookup cube (32 levels per channel)
    const LUT = new Uint8Array(32 * 32 * 32);
    for (let r = 0; r < 32; r++) for (let g = 0; g < 32; g++) for (let b = 0; b < 32; b++) {
      const R = r * 8.2, G = g * 8.2, B = b * 8.2;
      let best = 0, bd = 1e9;
      for (let i = 0; i < PALETTE.length; i++) {
        const p = PALETTE[i];
        const d = (R - p[0]) ** 2 * 0.3 + (G - p[1]) ** 2 * 0.59 + (B - p[2]) ** 2 * 0.11;
        if (d < bd) { bd = d; best = i; }
      }
      LUT[(r << 10) | (g << 5) | b] = best;
    }
    const BAYER = [0, 32, 8, 40, 2, 34, 10, 42, 48, 16, 56, 24, 50, 18, 58, 26, 12, 44, 4, 36, 14, 46, 6, 38, 60, 28, 52, 20, 62, 30, 54, 22, 3, 35, 11, 43, 1, 33, 9, 41, 51, 19, 59, 27, 49, 17, 57, 25, 15, 47, 7, 39, 13, 45, 5, 37, 63, 31, 55, 23, 61, 29, 53, 21].map(v => (v / 64 - 0.5) * 46);

    const hash = (x, y) => { const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453; return s - Math.floor(s); };
    const noise = (x, y) => {
      const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
      const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
      const a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
      return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
    };
    const fbm = (x, y) => noise(x, y) * 0.55 + noise(x * 2.1, y * 2.1) * 0.3 + noise(x * 4.3, y * 4.3) * 0.15;
    const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

    let W = 0, H = 0, img = null, scroll = 0, mx = 0, my = 0, tmx = 0, tmy = 0, last = 0, running = true;

    function resize() {
      const px = innerWidth < 700 ? 3 : 4;
      W = Math.max(120, Math.ceil(innerWidth / px));
      H = Math.max(120, Math.ceil(innerHeight / px));
      canvas.width = W; canvas.height = H;
      img = ctx.createImageData(W, H);
      draw(performance.now());
    }

    function draw(t) {
      const time = reduceMotion ? 0 : t / 1000;
      const d = img.data;
      const aspect = W / H;
      const lift = Math.min(scroll / innerHeight, 1.5) * 0.22; // hill sinks as you scroll
      mx += (tmx - mx) * 0.08; my += (tmy - my) * 0.08;
      const par = mx * 0.012;
      const M = Math.min(W, H * 1.2);
      const treeR = W < H ? 0.16 : 0.3;
      for (let y = 0; y < H; y++) {
        const v = y / H;
        for (let x = 0; x < W; x++) {
          const u = x / W;
          const hu = u + par;
          // horizon / hill silhouette
          const crest = 0.76 + lift - 0.12 * Math.exp(-((hu - 0.44) ** 2) / 0.2) + (fbm(hu * 6, 3.3) - 0.5) * 0.02;
          let c;
          if (v < crest) {
            // sky: navy → electric blue → violet glow near horizon
            const s = v / crest;
            c = s < 0.55 ? mix([5, 9, 70], [20, 34, 225], s / 0.55) : mix([20, 34, 225], [104, 60, 220], (s - 0.55) / 0.45);
            const cloud = fbm(u * 3 * aspect * 0.5 + time * 0.012, v * 7);
            if (cloud > 0.6) c = mix(c, [60, 80, 255], (cloud - 0.6) * 1.2);
            // distant bushes on the horizon
            const bush = crest - v < 0.05 + (fbm(hu * 22, 9) - 0.6) * 0.18 && (hu < 0.2 || (hu > 0.52 && hu < 0.6));
            if (bush) c = mix([10, 46, 50], [30, 110, 96], fbm(u * 80, v * 80));
            // big tree, top-right, gently swaying
            const sway = Math.sin(time * 0.6) * 0.006 * M;
            const tx = (x - (W - 0.02 * M + sway)) / M, ty = (y - 0.1 * H) / M;
            const r = Math.hypot(tx * 0.9, ty * 1.05);
            const leaf = fbm(x / M * 16 + time * 0.05, y / M * 16);
            if (r < treeR + (leaf - 0.5) * 0.22 && v < crest - 0.02) {
              c = leaf > 0.55 ? mix([60, 190, 160], [180, 245, 210], (leaf - 0.55) * 2.4) : mix([8, 40, 46], [24, 104, 92], leaf * 1.6);
            }
          } else {
            // meadow with perspective-scaled grass and flowers
            const depth = (v - crest) / (1 - crest + 1e-3);
            const f = 6 + depth * 34;
            const wind = Math.sin(time * 0.9 + u * 6) * 0.15;
            const g = fbm(u * f * aspect * 1.1 + wind, v * f * 1.3);
            c = mix([12, 60, 62], [64, 180, 156], Math.max(0, Math.min(1, g * 1.1 - depth * 0.3 + 0.1)));
            if (depth < 0.05) c = mix(c, [190, 248, 220], 0.45 - depth * 8); // rim light
            const fl = noise(u * (70 + depth * 180) * aspect * 0.6 + wind * 2, v * (110 + depth * 240));
            if (fl > 0.84 - depth * 0.1 && depth > 0.1) c = fl > 0.9 ? [255, 222, 80] : [240, 150, 64];
            c = mix(c, [3, 5, 34], Math.max(0, depth - 0.55) * 0.9);
          }
          const o = BAYER[((y & 7) << 3) | (x & 7)];
          const r = Math.max(0, Math.min(255, c[0] + o)) >> 3;
          const gg = Math.max(0, Math.min(255, c[1] + o)) >> 3;
          const b = Math.max(0, Math.min(255, c[2] + o)) >> 3;
          const p = PALETTE[LUT[(r << 10) | (gg << 5) | b]];
          const i = (y * W + x) * 4;
          d[i] = p[0]; d[i + 1] = p[1]; d[i + 2] = p[2]; d[i + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
    }

    function loop(t) {
      if (running && t - last > 110) { last = t; draw(t); }
      requestAnimationFrame(loop);
    }

    addEventListener("resize", () => { clearTimeout(resize._t); resize._t = setTimeout(resize, 120); });
    addEventListener("scroll", () => { scroll = scrollY; if (reduceMotion) draw(0); }, { passive: true });
    if (finePointer) addEventListener("mousemove", e => { tmx = (e.clientX / innerWidth - 0.5) * 2; tmy = (e.clientY / innerHeight - 0.5) * 2; }, { passive: true });
    document.addEventListener("visibilitychange", () => { running = !document.hidden; });
    resize();
    if (!reduceMotion) requestAnimationFrame(loop);
    return { draw };
  })();

  /* =====================================================================
   * 2. Entrance + reveal animations, nav state
   * ===================================================================== */
  $$(".headline .word").forEach((w, i) => w.style.setProperty("--i", i));
  requestAnimationFrame(() => document.body.classList.add("ready"));
  const io = new IntersectionObserver(entries => {
    entries.forEach(e => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } });
  }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
  $$("[data-reveal]").forEach((el, i) => {
    el.style.transitionDelay = el.closest(".hero, .nav") ? `${500 + i * 90}ms` : "0ms";
    io.observe(el);
  });
  const nav = $(".nav");
  addEventListener("scroll", () => nav.classList.toggle("scrolled", scrollY > 20), { passive: true });

  /* =====================================================================
   * 3. Custom cursor + magnetic buttons (fine pointers only)
   * ===================================================================== */
  if (finePointer) {
    document.documentElement.classList.add("has-cursor");
    const cur = $(".cursor"), ring = $(".cursor-ring"), dot = $(".cursor-dot");
    let x = -100, y = -100, rx = -100, ry = -100;
    addEventListener("mousemove", e => { x = e.clientX; y = e.clientY; cur.classList.remove("is-hidden"); }, { passive: true });
    document.addEventListener("mouseleave", () => cur.classList.add("is-hidden"));
    addEventListener("mousedown", () => cur.classList.add("is-down"));
    addEventListener("mouseup", () => cur.classList.remove("is-down"));
    document.addEventListener("mouseover", e => {
      const t = e.target;
      const text = t.closest("textarea, input[type=text], pre, .r-text, .answer p");
      const link = !text && t.closest("a, button, label.toggle, .chip, [role=button]");
      const card = !text && !link && t.closest(".card");
      cur.classList.toggle("is-text", !!text);
      cur.classList.toggle("is-link", !!link);
      cur.classList.toggle("is-card", !!card);
    });
    const k = reduceMotion ? 1 : 0.18;
    (function tick() {
      rx += (x - rx) * k; ry += (y - ry) * k;
      dot.style.transform = `translate3d(${x}px, ${y}px, 0)`;
      ring.style.transform = `translate3d(${rx}px, ${ry}px, 0)`;
      requestAnimationFrame(tick);
    })();

    if (!reduceMotion) {
      document.addEventListener("mousemove", e => {
        $$(".magnetic").forEach(btn => {
          const r = btn.getBoundingClientRect();
          const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
          const dx = e.clientX - cx, dy = e.clientY - cy;
          const near = Math.abs(dx) < r.width / 2 + 24 && Math.abs(dy) < r.height / 2 + 24;
          btn.style.setProperty("--mx", near && !btn.disabled ? `${dx * 0.18}px` : "0px");
          btn.style.setProperty("--my", near && !btn.disabled ? `${dy * 0.28}px` : "0px");
        });
      }, { passive: true });
    }
  }

  /* =====================================================================
   * 4. Backend health
   * ===================================================================== */
  const statusEl = $("#status");
  async function checkHealth() {
    const text = $(".status-text", statusEl);
    try {
      const res = await fetch("/api/health", { cache: "no-store" });
      const hj = await res.json();
      const gm = hj.generation_model?.name || "model";
      statusEl.classList.remove("ok", "bad");
      if (!hj.ollama?.reachable) {
        statusEl.classList.add("bad");
        text.textContent = "Ollama offline";
        statusEl.title = `Cannot reach Ollama at ${hj.ollama?.host}. Start it with \`ollama serve\`.`;
      } else if (hj.generation_model?.installed === false || hj.embedding_model?.installed === false) {
        statusEl.classList.add("bad");
        text.textContent = "Model missing";
        statusEl.title = `Pull the missing model(s): ${gm}, ${hj.embedding_model?.name}`;
      } else if (!hj.classifier?.exists) {
        statusEl.classList.add("bad");
        text.textContent = "Classifier missing";
        statusEl.title = `No trained classifier at ${hj.classifier?.path}`;
      } else {
        statusEl.classList.add("ok");
        text.textContent = `${gm} · ready`;
        statusEl.title = `Ollama ${hj.ollama.host} · embeddings ${hj.embedding_model?.name} · ${hj.classifier?.type} classifier`;
      }
      $("#footer-model").textContent = `${gm} · ${hj.embedding_model?.name || ""} · ${hj.classifier?.type || ""}`;
      return hj;
    } catch (err) {
      statusEl.classList.remove("ok");
      statusEl.classList.add("bad");
      text.textContent = "Backend offline";
      statusEl.title = "The adapter server (webapp/server.py) is not reachable.";
      return null;
    }
  }
  checkHealth();

  /* =====================================================================
   * 5. Ask form + analysis flow
   * ===================================================================== */
  const form = $("#ask-form"), qEl = $("#question"), submitBtn = $("#submit"), forceEl = $("#force-multi");
  const stage = $("#stage"), loading = $("#loading"), errorBox = $("#error"), results = $("#results");
  let busy = false, lastQuestion = "", stepTimer = null, clock = null;

  if (innerWidth < 560) qEl.placeholder = "Ask any question…";
  const autosize = () => { qEl.style.height = "auto"; qEl.style.height = Math.min(qEl.scrollHeight, 200) + "px"; };
  qEl.addEventListener("input", autosize);
  qEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
  });
  $$(".chip").forEach(c => c.addEventListener("click", () => { qEl.value = c.dataset.q; autosize(); qEl.focus(); }));
  form.addEventListener("submit", e => { e.preventDefault(); run(qEl.value, forceEl.checked); });
  $("#error-retry").addEventListener("click", () => run(lastQuestion, forceEl.checked));

  function show(which) {
    stage.hidden = false;
    loading.hidden = which !== "loading";
    errorBox.hidden = which !== "error";
    results.hidden = which !== "results";
  }

  function startLoadingUI(question) {
    $("#loading-q").textContent = question;
    const steps = $$("#steps li");
    steps.forEach(s => s.classList.remove("active", "seen"));
    let i = 0;
    steps[0].classList.add("active");
    // Illustrative pacing only — not tied to backend events.
    stepTimer = setInterval(() => {
      if (i < steps.length - 1) { steps[i].classList.replace("active", "seen"); i++; steps[i].classList.add("active"); }
    }, 2600);
    const t0 = performance.now();
    const timer = $("#timer");
    clock = setInterval(() => { timer.textContent = ((performance.now() - t0) / 1000).toFixed(1) + "s"; }, 100);
  }
  function stopLoadingUI() { clearInterval(stepTimer); clearInterval(clock); }

  async function run(question, forceMulti) {
    question = (question || "").trim();
    if (!question) { qEl.focus(); form.querySelector(".ask-field").animate([{ transform: "translateX(0)" }, { transform: "translateX(-6px)" }, { transform: "translateX(6px)" }, { transform: "translateX(0)" }], { duration: 300 }); return; }
    if (busy) return;
    busy = true; lastQuestion = question;
    form.classList.add("busy");
    submitBtn.disabled = true; submitBtn.classList.add("loading");
    $(".btn-label", submitBtn).textContent = "Analyzing";
    $(".hero").classList.add("compact");
    show("loading");
    startLoadingUI(question);
    // Let the hero finish collapsing before measuring the scroll target.
    setTimeout(() => stage.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" }), reduceMotion ? 0 : 820);

    try {
      let res;
      try {
        res = await fetch("/api/analyze", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question, force_multi: !!forceMulti }),
        });
      } catch (netErr) {
        throw Object.assign(new Error("Could not reach the adapter server at " + location.origin + ". Is `python webapp/server.py` running?"), { kind: "network" });
      }
      let body;
      try { body = await res.json(); } catch { throw new Error(`Backend returned HTTP ${res.status} with a non-JSON body.`); }
      if (!res.ok || !body.record) throw Object.assign(new Error(body.error || `HTTP ${res.status}`), { kind: body.error_type || "pipeline" });
      stopLoadingUI();
      renderResults(body.record);
      show("results");
    } catch (err) {
      stopLoadingUI();
      renderError(err);
      show("error");
      checkHealth();
    } finally {
      busy = false;
      form.classList.remove("busy");
      submitBtn.disabled = false; submitBtn.classList.remove("loading");
      $(".btn-label", submitBtn).textContent = "Analyze";
    }
  }

  function renderError(err) {
    const msg = err.message || String(err);
    $("#error-msg").textContent = msg;
    let title = "The pipeline returned an error", help = "No results are shown because the backend did not produce any.";
    if (err.kind === "network") { title = "Backend not reachable"; help = "Start the adapter from the project root: .venv/bin/python webapp/server.py"; }
    else if (/ollama/i.test(msg) || err.kind === "OllamaClientError") { title = "Ollama is not available"; help = "Start Ollama (`ollama serve`) and make sure the configured models are pulled, then try again."; }
    $("#error-title").textContent = title;
    $("#error-help").textContent = help;
  }

  /* =====================================================================
   * 6. Results rendering (real backend values only)
   * ===================================================================== */
  const CLUSTER_COLORS = ["#fff31a", "#5b6dff", "#2fe3a1", "#f2963a", "#b07bff", "#ff7aa8", "#7ee8ff", "#ffffff"];
  const LABEL_COPY = {
    KNOWN: "The question appears well-specified and within what the model can answer.",
    AMBIGUOUS: "The question admits more than one materially different interpretation.",
    UNKNOWN: "The answer appears inaccessible to the model — it can’t reliably know this.",
  };
  const FEATURE_INFO = {
    semantic_entropy: ["Entropy", "nats", false],
    normalized_entropy: ["Entropy", "", true],
    dominant_agreement: ["Agreement", "", true],
    second_agreement: ["Agreement", "", true],
    agreement_margin: ["Agreement", "", true],
    cluster_count: ["Clustering", "", false],
    cluster_ratio: ["Clustering", "", true],
    mean_pairwise_similarity: ["Similarity", "", true],
    min_pairwise_similarity: ["Similarity", "", true],
    response_len_mean: ["Length", "chars", false],
    response_len_std: ["Length", "chars", false],
    distinct_1gram_ratio: ["Lexical diversity", "", true],
    distinct_2gram_ratio: ["Lexical diversity", "", true],
    question_overlap_mean: ["Question overlap", "", true],
    hedging_score: ["Hedging & negation", "", true],
    negation_count: ["Hedging & negation", "per response", false],
  };

  const isNum = v => typeof v === "number" && Number.isFinite(v);
  const fmt = (v, d = 3) => (isNum(v) ? v.toFixed(d) : "—");
  const pct = v => (isNum(v) ? (v * 100).toFixed(0) + "%" : "—");
  const humanize = s => (s ? String(s).replace(/_/g, " ").toLowerCase().replace(/^\w/, c => c.toUpperCase()) : "—");

  function countUp(el, value, decimals, suffix = "") {
    if (!isNum(value)) { el.textContent = "—"; return; }
    const final = value.toFixed(decimals) + suffix;
    if (reduceMotion) { el.textContent = final; return; }
    const t0 = performance.now(), dur = 1100;
    (function step(t) {
      const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = (value * e).toFixed(decimals) + suffix;
      if (p < 1) requestAnimationFrame(step); else el.textContent = final;
    })(t0);
  }
  const growLater = (el, w) => requestAnimationFrame(() => requestAnimationFrame(() => { el.style.width = w; }));

  function renderResults(r) {
    results.replaceChildren();
    let d = 0;
    const add = node => { node.style.setProperty("--d", d++); results.append(node); return node; };
    const label = r.predicted_label;
    const responses = Array.isArray(r.responses) ? r.responses : [];
    const labels = r.clusters?.labels || [];
    const counts = r.clusters?.cluster_counts || [];
    const domIdx = r.clusters?.dominant_rep_idx, secIdx = r.clusters?.second_rep_idx;
    const isFast = r.path === "FAST_PATH";

    // ---- Verdict ----
    // classifier_status looks like "PREDICTED" or "ASSESSED_..._RULE_HEURISTIC (evidence)".
    const statusRaw = String(r.classifier_status || "");
    const sm = /^([A-Z0-9_ ]+?)\s*\(([\s\S]*)\)\s*$/.exec(statusRaw);
    const statusCode = sm ? sm[1].trim() : statusRaw;
    const evidence = sm ? sm[2] : null;
    const SOURCE = {
      PREDICTED: "Trained classifier",
      ASSESSED_INACCESSIBLE_RULE_HEURISTIC: "Inaccessibility safeguard",
      ASSESSED_AMBIGUOUS_RULE_HEURISTIC: "Ambiguity safeguard",
    };
    const source = SOURCE[statusCode] || (statusCode.startsWith("ASSESSED_FAST_PATH") ? "Fast-path assessor · " + humanize(statusCode.replace("ASSESSED_FAST_PATH_", "")) : humanize(statusCode));
    const confEl = h("span", { class: "conf" });
    const verdict = add(h("section", { class: `card verdict ${label || ""}`, "aria-label": "Classification" },
      h("div", { class: "verdict-main" },
        h("p", { class: "eyebrow", text: "Classification" }),
        h("p", { class: "verdict-q", text: r.question }),
        h("div", { class: "verdict-label" },
          h("span", { class: "lbl", text: label || "Unclassified" }),
          isNum(r.confidence) ? h("span", { class: "conf-wrap mono small" }, "confidence ", confEl) : null),
        h("p", { class: "verdict-sub", text: label ? LABEL_COPY[label] || "" : `Classifier status: ${r.classifier_status}` })),
      h("div", { class: "verdict-side" },
        h("dl", { class: "kv" },
          h("dt", { text: "Path" }), h("dd", { text: isFast ? "Fast path" : "Multi-response" }),
          h("dt", { text: "Decided by" }), h("dd", { text: source, title: statusRaw }),
          h("dt", { text: "HITL decision" }), h("dd", { text: humanize(r.clarification_decision) }),
          h("dt", { text: "Question clarity" }), h("dd", { text: r.clarity_status ? `${humanize(r.clarity_status)}${isNum(r.clarity_confidence) ? ` · ${r.clarity_confidence.toFixed(2)}` : ""}` : "—" }),
          r.ambiguity_type ? [h("dt", { text: "Ambiguity type" }), h("dd", { text: humanize(r.ambiguity_type) })] : null,
          h("dt", { text: "Elapsed" }), h("dd", { text: isNum(r.elapsed_seconds) ? r.elapsed_seconds.toFixed(2) + " s" : "—" })),
        evidence ? h("details", { class: "evidence" }, h("summary", { class: "mono small", text: "Safeguard evidence" }), h("code", { class: "mono small", text: evidence })) : null)));
    if (isNum(r.confidence)) countUp(confEl, r.confidence, 2);

    // ---- Answer ----
    add(h("section", { class: "card answer" },
      h("h4", { text: "Generated answer" }),
      h("p", { text: r.final_answer }),
      r.clarification_reason ? h("p", { class: "reason", text: r.clarification_reason }) : null));

    // ---- Fast path notice ----
    if (isFast) {
      add(h("section", { class: "card notice" },
        h("p", { text: "This question was answered on the adaptive fast path: a single greedy response, so the consistency metrics below are the pipeline’s single-response defaults rather than a measured spread across samples." }),
        h("button", { class: "btn btn-ghost btn-sm", type: "button", onclick: () => { forceEl.checked = true; run(r.question, true); } }, "Run full multi-response analysis")));
    }

    // ---- Metric grid ----
    const metrics = [
      ["Semantic entropy", r.semantic_entropy, 3, " nats", null, "Spread of meaning across samples"],
      ["Normalized entropy", r.normalized_entropy, 3, "", r.normalized_entropy, "0 = consensus · 1 = full dispersion"],
      ["Dominant agreement", isNum(r.dominant_cluster_agreement) ? r.dominant_cluster_agreement * 100 : null, 0, "%", r.dominant_cluster_agreement, "Share of the largest cluster"],
      ["Second agreement", isNum(r.second_cluster_agreement) ? r.second_cluster_agreement * 100 : null, 0, "%", r.second_cluster_agreement, "Share of the runner-up cluster"],
      ["Agreement margin", r.agreement_margin, 2, "", r.agreement_margin, "Dominant minus second"],
      ["Mean similarity", r.mean_pairwise_similarity, 3, "", r.mean_pairwise_similarity, "Mean pairwise cosine similarity"],
      ["Clusters", r.cluster_count, 0, "", null, "Semantic equivalence classes"],
      ["Responses", r.response_count, 0, "", null, isFast ? "Fast path: one greedy sample" : "Sampled from the local LLM"],
    ];
    const grid = h("section", { class: "metrics", "aria-label": "Consistency metrics" });
    metrics.forEach(([name, val, dec, suf, bar, desc]) => {
      const valEl = h("div", { class: "m-value" });
      const barI = h("i");
      grid.append(h("div", { class: "card metric" },
        h("div", { class: "m-label", text: name }), valEl,
        h("p", { class: "m-desc", text: desc }),
        isNum(bar) ? h("div", { class: "m-bar" }, barI) : null));
      if (isNum(val)) {
        const num = h("span"); valEl.append(num);
        if (suf.trim()) valEl.append(h("small", { text: suf.trim() }));
        countUp(num, val, dec, suf.trim() ? "" : suf);
      } else valEl.textContent = "—";
      if (isNum(bar)) growLater(barI, Math.max(0, Math.min(1, bar)) * 100 + "%");
    });
    add(grid);

    // ---- Clusters + run details ----
    const bar = h("div", { class: "clusters-bar", role: "img", "aria-label": `Cluster sizes: ${counts.join(", ")}` });
    const legend = h("ul", { class: "cluster-legend" });
    const total = counts.reduce((a, b) => a + b, 0) || 1;
    counts.forEach((n, i) => {
      const col = CLUSTER_COLORS[i % CLUSTER_COLORS.length];
      const seg = h("span", { style: `background:${col}`, title: `Cluster ${i + 1}: ${n} response${n > 1 ? "s" : ""}`, text: n / total >= 0.12 ? String(n) : "" });
      bar.append(seg);
      requestAnimationFrame(() => requestAnimationFrame(() => { seg.style.flexGrow = n; }));
      const rep = i === 0 ? domIdx : i === 1 ? secIdx : null;
      legend.append(h("li", null, h("i", { class: "sw", style: `background:${col}` }),
        h("span", { text: `Cluster ${i + 1}${i === 0 ? " · dominant" : ""}${rep != null && rep >= 0 ? ` · medoid #${String(rep + 1).padStart(2, "0")}` : ""}` }),
        h("span", { class: "n", text: `${n} · ${((n / total) * 100).toFixed(0)}%` })));
    });

    const temps = [...new Set(responses.map(x => x.temperature).filter(isNum))];
    const toks = responses.map(x => x.eval_count).filter(isNum).reduce((a, b) => a + b, 0);
    const genTime = responses.map(x => x.elapsed_seconds).filter(isNum).reduce((a, b) => a + b, 0);
    add(h("div", { class: "two-col" },
      h("section", { class: "card" }, h("h4", null, "Semantic clusters", h("span", { text: `${counts.length} found` })), bar, legend),
      h("section", { class: "card" }, h("h4", { text: "Run details" }),
        h("dl", { class: "kv" },
          h("dt", { text: "Generation model" }), h("dd", { text: r.model }),
          h("dt", { text: "Embedding model" }), h("dd", { text: r.embedding_model }),
          h("dt", { text: "Temperatures" }), h("dd", { text: temps.length ? temps.join(", ") : "—" }),
          h("dt", { text: "Tokens generated" }), h("dd", { text: toks ? String(toks) : "—" }),
          h("dt", { text: "Generation time" }), h("dd", { text: genTime ? genTime.toFixed(2) + " s" : "—" }),
          h("dt", { text: "Total time" }), h("dd", { text: isNum(r.elapsed_seconds) ? r.elapsed_seconds.toFixed(2) + " s" : "—" }),
          h("dt", { text: "Timestamp" }), h("dd", { text: r.timestamp ? new Date(r.timestamp).toLocaleString() : "—" })))));

    // ---- Feature vector ----
    const feats = r.features || {};
    const names = Object.keys(feats);
    if (names.length) {
      const tbody = h("tbody");
      names.forEach(n => {
        const [group, unit, bounded] = FEATURE_INFO[n] || ["Other", "", false];
        const v = feats[n];
        const fi = h("i");
        tbody.append(h("tr", null,
          h("td", null, h("span", { class: "fname", text: n }), h("span", { class: "fgroup", text: group + (unit ? ` · ${unit}` : "") })),
          h("td", { class: "b" }, bounded && isNum(v) ? h("div", { class: "fbar" }, fi) : null),
          h("td", { class: "v", text: fmt(v, Math.abs(v) >= 100 ? 1 : 3) })));
        if (bounded && isNum(v)) growLater(fi, Math.max(0, Math.min(1, v)) * 100 + "%");
      });
      add(h("section", { class: "card" },
        h("h4", null, "Extracted feature vector", h("span", { text: `${names.length} features → ${r.path === "FAST_PATH" ? "fast-path assessor" : "classifier"}` })),
        h("table", { class: "features" },
          h("thead", null, h("tr", null, h("th", { text: "Feature" }), h("th", { class: "b" }), h("th", { class: "v", style: "text-align:right", text: "Value" }))),
          tbody)));
    }

    // ---- Responses ----
    if (responses.length) {
      const list = h("div", { class: "responses" });
      const filter = h("div", { class: "response-filter", role: "group", "aria-label": "Filter responses by cluster" });
      const setFilter = c => {
        $$("button", filter).forEach(b => b.setAttribute("aria-pressed", String(b.dataset.c === String(c))));
        $$(".response", list).forEach((el, i) => {
          const show = c === "all" || el.dataset.c === String(c);
          el.classList.toggle("hidden", !show);
          if (show) { el.classList.remove("enter"); void el.offsetWidth; el.style.setProperty("--i", i % 10); el.classList.add("enter"); }
        });
      };
      filter.append(h("button", { type: "button", "data-c": "all", "aria-pressed": "true", onclick: () => setFilter("all") }, "All"));
      if (counts.length > 1) counts.forEach((_, i) => filter.append(h("button", { type: "button", "data-c": String(i), "aria-pressed": "false", onclick: () => setFilter(i) }, `Cluster ${i + 1}`)));

      const tpl = $("#tpl-response");
      responses.forEach((resp, i) => {
        const node = tpl.content.firstElementChild.cloneNode(true);
        const cl = labels[i];
        const col = isNum(cl) ? CLUSTER_COLORS[cl % CLUSTER_COLORS.length] : "#fff";
        node.style.setProperty("--cc", col);
        node.style.setProperty("--i", i);
        node.dataset.c = String(cl);
        node.classList.add("enter");
        $(".r-idx", node).textContent = "#" + String(resp.index != null ? resp.index + 1 : i + 1).padStart(2, "0");
        const tags = $(".r-tags", node);
        if (isNum(cl)) tags.append(h("span", { class: "tag", style: `color:${col}`, text: `Cluster ${cl + 1}` }), " ");
        if (i === domIdx) tags.append(h("span", { class: "tag y", text: "Dominant medoid" }), " ");
        if (i === secIdx) tags.append(h("span", { class: "tag", text: "Second medoid" }), " ");
        if (resp.temperature === 0) tags.append(h("span", { class: "tag", text: "Greedy" }));
        const meta = [];
        if (isNum(resp.temperature)) meta.push(`T=${resp.temperature}`);
        if (isNum(resp.elapsed_seconds)) meta.push(`${resp.elapsed_seconds.toFixed(2)}s`);
        if (isNum(resp.eval_count)) meta.push(`${resp.eval_count} tok`);
        $(".r-meta", node).textContent = meta.join(" · ");
        $(".r-text", node).textContent = resp.text ?? "";
        list.append(node);
      });
      add(h("div", { class: "responses-head" },
        h("h3", { text: `${responses.length} generated response${responses.length > 1 ? "s" : ""}` }), filter));
      add(list);
    }

    // ---- Actions ----
    const copyBtn = h("button", { class: "btn btn-ghost", type: "button" }, "Copy raw JSON");
    copyBtn.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(JSON.stringify(r, null, 2)); copyBtn.textContent = "Copied"; }
      catch { copyBtn.textContent = "Copy failed"; }
      setTimeout(() => (copyBtn.textContent = "Copy raw JSON"), 1600);
    });
    add(h("div", { class: "actions" },
      h("button", { class: "btn btn-yellow magnetic", type: "button", onclick: () => { scrollTo({ top: 0, behavior: reduceMotion ? "auto" : "smooth" }); setTimeout(() => { qEl.select(); qEl.focus({ preventScroll: true }); }, reduceMotion ? 0 : 500); } }, "Ask another question"),
      copyBtn));

    void verdict;
  }
})();
