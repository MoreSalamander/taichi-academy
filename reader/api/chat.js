// Vercel serverless function — the page-aware helper's brain.
// The API key lives ONLY here (server-side), never in the browser.
// The deterministic verdict is computed client-side; this endpoint only
// *voices* it (and diagnoses pasted errors), grounded in the step's facts.
//
// POST /api/chat  { mode: "check"|"help", ctx, pasted, findings }  ->  { text }
// Requires env var: ANTHROPIC_API_KEY (set in the Vercel dashboard).

const SYSTEM =
  "You are the calm helper inside 'Build It', a coding guide for absolute beginners. " +
  "Reply in 2-5 short sentences, warm and plain — never a jargon dump, never condescending. " +
  "You are helping with ONE specific step; stay on it. " +
  "Ground every claim in the provided step facts, the deterministic check result, and the learner's pasted text. " +
  "NEVER invent an error that isn't in the deterministic result or the pasted text. " +
  "If everything looks right, say so plainly and tell them to run it and look for the expected result.";

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "POST only" });

  const key = process.env.ANTHROPIC_API_KEY || process.env.Anthropic_api_key;
  if (!key) return res.status(503).json({ error: "helper-not-configured" });

  try {
    const body = typeof req.body === "string" ? JSON.parse(req.body || "{}") : (req.body || {});
    const { mode, ctx = {}, pasted = "", findings = "" } = body;

    const L = [];
    L.push(`Chapter: ${ctx.chapter || "?"}`);
    L.push(`Step: ${ctx.stepTitle || "?"}`);
    if (ctx.code) L.push(`The code this step adds:\n${ctx.code}`);
    if (ctx.see) L.push(`What they should see when it works: ${ctx.see}`);
    if (Array.isArray(ctx.recovery) && ctx.recovery.length)
      L.push(`Known things that go wrong on this step:\n- ${ctx.recovery.join("\n- ")}`);

    if (mode === "check") {
      L.push(`The learner pasted this code:\n${pasted || "(nothing)"}`);
      L.push(`Deterministic check result: ${findings}`);
      L.push("Task: In 2-4 sentences, kindly tell them whether this step's code is correctly in place, using ONLY the deterministic result. If a line is missing, say which and where it goes. Do not check anything beyond the deterministic result.");
    } else {
      L.push(`The learner says something isn't right and pasted this:\n${pasted || "(nothing)"}`);
      L.push("Task: Diagnose calmly using the step facts and the known-things-that-go-wrong. If their paste matches a known issue, give that fix. If it's an error message, say in plain words what it means and the likely one-line fix. Keep it short.");
    }

    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({
        model: "claude-haiku-4-5",
        max_tokens: 400,
        system: SYSTEM,
        messages: [{ role: "user", content: L.join("\n\n") }],
      }),
    });

    const data = await r.json();
    const text = data?.content?.[0]?.text || "";
    if (!text) return res.status(502).json({ error: "no-text", detail: data?.error?.message || "" });
    return res.status(200).json({ text });
  } catch (e) {
    return res.status(500).json({ error: String(e) });
  }
}
