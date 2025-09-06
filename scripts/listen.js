#!/usr/bin/env node
const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");
const { spawn, spawnSync } = require("child_process");

const WHISPER_BIN = process.env.WHISPER_BIN || "whisper";
const WHISPER_TIMEOUT_MS = Number(process.env.WHISPER_TIMEOUT_MS || 15 * 60 * 1000); // 15 min
const WHISPER_VENV = process.env.WHISPER_VENV || process.env.VIRTUAL_ENV || "";

function ts() {
  const d = new Date();
  return d.toISOString().replace('T',' ').replace('Z','');
}

function which(cmd, envPath = null) {
  try {
    const env = envPath ? { ...process.env, PATH: envPath } : process.env;
    const r = spawnSync("which", [cmd], { encoding: "utf8", env });
    return r.status === 0 ? r.stdout.trim() : "";
  } catch { return ""; }
}

/**
 * Detect virtual environment and prepare environment for Whisper
 */
function prepareWhisperEnvironment() {
  const venvPath = WHISPER_VENV;
  let whisperPath = "";
  let envPath = process.env.PATH;
  
  console.log(`[${ts()}][whisper-env] === ENVIRONMENT SETUP ===`);
  console.log(`[${ts()}][whisper-env] WHISPER_VENV: ${venvPath || '(not set)'}`);
  console.log(`[${ts()}][whisper-env] VIRTUAL_ENV: ${process.env.VIRTUAL_ENV || '(not set)'}`);
  console.log(`[${ts()}][whisper-env] Current PATH: ${process.env.PATH}`);
  
  if (venvPath && fs.existsSync(venvPath)) {
    const venvBinPath = path.join(venvPath, 'bin');
    if (fs.existsSync(venvBinPath)) {
      envPath = `${venvBinPath}:${process.env.PATH}`;
      console.log(`[${ts()}][whisper-env] Added venv bin to PATH: ${venvBinPath}`);
    } else {
      console.warn(`[${ts()}][whisper-env] Virtual env bin directory not found: ${venvBinPath}`);
    }
  }
  
  // Try to find whisper with the modified PATH
  whisperPath = which(WHISPER_BIN, envPath);
  
  if (!whisperPath) {
    // Try common locations
    const commonPaths = [
      '/usr/local/bin/whisper',
      '/opt/homebrew/bin/whisper',
      path.join(process.env.HOME || '', '.local/bin/whisper')
    ];
    
    for (const tryPath of commonPaths) {
      if (fs.existsSync(tryPath)) {
        whisperPath = tryPath;
        console.log(`[${ts()}][whisper-env] Found whisper at: ${whisperPath}`);
        break;
      }
    }
  }
  
  console.log(`[${ts()}][whisper-env] Final whisper path: ${whisperPath || 'NOT FOUND'}`);
  console.log(`[${ts()}][whisper-env] Final PATH: ${envPath}`);
  console.log(`[${ts()}][whisper-env] === SETUP COMPLETE ===`);
  
  return { whisperPath, envPath };
}

async function transcribeAudioWithWhisper(filePath) {
  const transcriptPath = filePath.replace(/\.\w+$/, ".txt");
  const cwd = path.dirname(filePath);
  const base = path.basename(filePath);

  console.log(`[${ts()}][whisper] === WHISPER TRANSCRIPTION START ===`);
  console.log(`[${ts()}][whisper] Audio file: ${filePath}`);
  console.log(`[${ts()}][whisper] File exists: ${fs.existsSync(filePath)}`);
  console.log(`[${ts()}][whisper] File size: ${fs.existsSync(filePath) ? fs.statSync(filePath).size : 'N/A'} bytes`);
  console.log(`[${ts()}][whisper] Working directory: ${cwd}`);
  console.log(`[${ts()}][whisper] Base filename: ${base}`);
  console.log(`[${ts()}][whisper] Timeout: ${WHISPER_TIMEOUT_MS}ms`);
  
  const { whisperPath, envPath } = prepareWhisperEnvironment();
  
  if (!whisperPath) {
    console.error(`[${ts()}][whisper] ❌ Whisper executable not found!`);
    console.error(`[${ts()}][whisper] Searched for: ${WHISPER_BIN}`);
    console.error(`[${ts()}][whisper] PATH: ${envPath}`);
    console.error(`[${ts()}][whisper] Try setting WHISPER_BIN or WHISPER_VENV environment variables`);
    return "";
  }

  const args = [base, "--model", "base", "--language", "en", "--output_format", "txt"];
  console.log(`[${ts()}][whisper] Command: ${whisperPath} ${args.join(" ")}`);
  console.log(`[${ts()}][whisper] Environment PATH: ${envPath}`);

  try {
    console.log(`[${ts()}][whisper] Spawning Whisper process...`);
    const p = spawn(whisperPath, args, { 
      cwd,
      env: { ...process.env, PATH: envPath }
    });

    let so = "", se = "";
    const killTimer = setTimeout(() => {
      console.warn(`[${ts()}][whisper] ⏰ Timeout after ${WHISPER_TIMEOUT_MS}ms → sending SIGKILL`);
      try { p.kill("SIGKILL"); } catch {}
    }, WHISPER_TIMEOUT_MS).unref();

    p.stdout.on("data", d => {
      const s = d.toString();
      so += s;
      console.log(`[${ts()}][whisper][stdout] ${s.trim().slice(0, 400)}`);
    });

    p.stderr.on("data", d => {
      const s = d.toString();
      se += s;
      console.log(`[${ts()}][whisper][stderr] ${s.trim().slice(0, 400)}`);
    });

    p.on('error', (err) => {
      console.error(`[${ts()}][whisper] ❌ Spawn error: ${err.message}`);
    });

    console.log(`[${ts()}][whisper] Waiting for process to complete...`);
    const code = await new Promise(resolve => p.on("close", resolve));
    clearTimeout(killTimer);
    console.log(`[${ts()}][whisper] Process exited with code: ${code}`);

    // Check for transcript file
    const outPath = path.join(cwd, path.basename(transcriptPath));
    const exists = fs.existsSync(outPath);
    console.log(`[${ts()}][whisper] Expected transcript: ${outPath}`);
    console.log(`[${ts()}][whisper] Transcript exists: ${exists}`);
    
    if (exists) {
      const stats = fs.statSync(outPath);
      console.log(`[${ts()}][whisper] Transcript size: ${stats.size} bytes`);
    }

    if (code === 0 && exists) {
      const txt = fs.readFileSync(outPath, "utf8").trim();
      console.log(`[${ts()}][whisper] ✅ Success! Transcript length: ${txt.length} characters`);
      console.log(`[${ts()}][whisper] Transcript preview: "${txt.slice(0, 200)}${txt.length > 200 ? '...' : ''}"`);
      console.log(`[${ts()}][whisper] === WHISPER TRANSCRIPTION SUCCESS ===`);
      return txt;
    }

    console.warn(`[${ts()}][whisper] ❌ Transcription failed (exit code: ${code})`);
    if (se) console.warn(`[${ts()}][whisper] Stderr: "${se.trim().slice(0,400)}"`);
    if (so) console.warn(`[${ts()}][whisper] Stdout: "${so.trim().slice(0,400)}"`);
    console.log(`[${ts()}][whisper] === WHISPER TRANSCRIPTION FAILED ===`);
    return "";
  } catch (e) {
    console.error(`[${ts()}][whisper] ❌ Exception: ${e.message}`);
    console.log(`[${ts()}][whisper] === WHISPER TRANSCRIPTION ERROR ===`);
    return "";
  }
}

const port = Number(process.env.PORT || 8765);

/* -------------------- config & helpers -------------------- */

// Slack cosmetics (used when posting back via response_url)
const SLACK_BOT_USERNAME = process.env.SLACK_BOT_USERNAME || "Webhook Assistant";
const SLACK_BOT_ICON_URL = process.env.SLACK_BOT_ICON_URL || "";
const SLACK_BOT_ICON_EMOJI = process.env.SLACK_BOT_ICON_EMOJI || ":robot_face:";

// Logging
const MAX_LOG_BYTES = Number(process.env.MAX_LOG_BYTES || 4096);
const LOG_HEADERS   = process.env.LOG_HEADERS === "1";
const LOG_BODY      = process.env.LOG_BODY === "1";

// Attachment guardrails
const ATTACH_MAX_FILES = Number(process.env.ATTACH_MAX_FILES || 6);
const ATTACH_MAX_BYTES = Number(process.env.ATTACH_MAX_BYTES || 50 * 1024 * 1024); // 50 MB per file

// Gemini knobs
const GEMINI_TIMEOUT_MS = Number(process.env.GEMINI_TIMEOUT_MS || 120000);
const GEMINI_SYSTEM_PATH = process.env.GEMINI_SYSTEM || "/home/johncapobianco/.gemini/GEMINI.md";

// Block Kit mode feature flag (optional): if "1", default JSON-blocks prompts for QA mode
const SLACK_BLOCKS_DEFAULT = process.env.SLACK_BLOCKS === "1";

// Load system prompt once (optional)
let SYSTEM_TEXT = "";
try {
  if (GEMINI_SYSTEM_PATH && fs.existsSync(GEMINI_SYSTEM_PATH)) {
    SYSTEM_TEXT = fs.readFileSync(GEMINI_SYSTEM_PATH, "utf8").trim();
    console.log(`[init] Loaded system prompt from ${GEMINI_SYSTEM_PATH} (${SYSTEM_TEXT.length} chars)`);
  }
} catch (e) {
  console.warn(`[init] Could not read system file: ${e.message}`);
}

const color = {
  dim:    (s) => `\x1b[2m${s}\x1b[0m`,
  mag:    (s) => `\x1b[35m${s}\x1b[0m`,
  red:    (s) => `\x1b[31m${s}\x1b[0m`,
  yellow: (s) => `\x1b[33m${s}\x1b[0m`,
};

function truncate(str, max = MAX_LOG_BYTES) {
  if (!str) return "";
  if (str.length <= max) return str;
  return str.slice(0, max) + `\n… (${str.length - max} more bytes truncated)`;
}
function safeJson(obj) { try { return JSON.stringify(obj, null, 2); } catch { return String(obj); } }
function redactHeaders(h) {
  const clone = { ...h };
  for (const k of Object.keys(clone)) {
    if (/authorization|token|cookie|set-cookie|api[-_]key/i.test(k)) clone[k] = "[redacted]";
  }
  return clone;
}

const isPcap = (fname = "") => /\.pcap(?:ng)?$/i.test(fname);

/** Spawn Gemini CLI with timeout; return { stdout, stderr, code }. */
function runGeminiProcess(args) {
  return new Promise((resolve) => {
    console.log(`[${ts()}][gemini] Spawning: gemini ${args.join(' ')}`);
    const p = spawn("gemini", args, { stdio: ["ignore", "pipe", "pipe"] });
    let out = "", err = "";

    const to = setTimeout(() => {
      console.warn(`[${ts()}][gemini] ⏰ Timeout after ${GEMINI_TIMEOUT_MS}ms`);
      try { p.kill("SIGKILL"); } catch {}
      resolve({ stdout: out.trim(), stderr: (err.trim() + " [timeout]"), code: 124 });
    }, GEMINI_TIMEOUT_MS).unref();

    p.stdout.on("data", d => {
      const chunk = d.toString();
      out += chunk;
      // Log first few lines of output
      const lines = chunk.split('\n').slice(0, 3);
      for (const line of lines) {
        if (line.trim()) console.log(`[${ts()}][gemini][stdout] ${line.trim().slice(0, 200)}`);
      }
    });
    
    p.stderr.on("data", d => {
      const chunk = d.toString();
      err += chunk;
      console.log(`[${ts()}][gemini][stderr] ${chunk.trim().slice(0, 400)}`);
    });
    
    p.on("close", code => {
      clearTimeout(to);
      console.log(`[${ts()}][gemini] Process completed with code: ${code}`);
      console.log(`[${ts()}][gemini] Output length: ${out.trim().length} characters`);
      resolve({ stdout: out.trim(), stderr: err.trim(), code });
    });

    p.on("error", (err) => {
      console.error(`[${ts()}][gemini] ❌ Spawn error: ${err.message}`);
      clearTimeout(to);
      resolve({ stdout: out.trim(), stderr: err.message, code: -1 });
    });
  });
}

/** Compose final prompt (prepend system text if present). */
function composePrompt(taskPrompt) {
  if (SYSTEM_TEXT) {
    return ["### System", SYSTEM_TEXT, "", "### Task", taskPrompt].join("\n");
  }
  return taskPrompt;
}

/** Summarize local files without inlining bytes. */
function listLocalFiles(saved) {
  if (!saved.length) return "";
  const lines = ["### Local files (available in current working directory)", ""];
  for (const f of saved) {
    lines.push(`- ${f.filename} (${f.bytes} bytes, ${f.mime || "application/octet-stream"})`);
  }
  lines.push(
    "",
    "You can open/read these files directly from the working directory to answer.",
    ""
  );
  return lines.join("\n");
}

/**
 * Build a Q&A prompt (Markdown answer expected).
 * No base64 embedding. Just pass user's text + optional context + a list of local files.
 */
function buildQAPrompt(userText, context, saved) {
  const lines = ["You are a helpful assistant. Be clear and concise."];
  if (context) lines.push("", "### Context", context);
  lines.push("", "### Question", userText || "(no text)");
  if (saved.length) lines.push("", listLocalFiles(saved));
  return lines.join("\n");
}

/**
 * Build a Q&A prompt that instructs the model to return Slack Block Kit JSON ONLY.
 * The model must return a JSON object like: { "blocks": [ ... ] }
 */
function buildSlackBlocksPrompt(userText, context, saved) {
  const lines = [
    "You are replying in Slack. Return ONLY valid JSON for Slack Block Kit. Do not include backticks or any text outside JSON.",
    "Respond as a single JSON object with this shape: {\"blocks\": [ ... ]}.",
    "Rules:",
    "- Use {\"type\":\"section\",\"text\":{\"type\":\"mrkdwn\",\"text\":\"...\"}} for most content.",
    "- You may use header, divider, context, fields, and actions (buttons).",
    "- Keep total blocks ≤ 50; each section text ≤ 3000 chars.",
    "- Escape characters per JSON.",
  ];
  if (context) lines.push("", "Context:\n" + context);
  lines.push("", "User message:\n" + (userText || "(no text)"));
  if (saved.length) lines.push("", listLocalFiles(saved));
  return lines.join("\n");
}

/** Build an ops-style prompt (no embedding). */
function buildOpsPrompt(obj, headers, saved, userTextForOps = "") {
  const lines = [
    "You are an assistant receiving a webhook. Summarize the event and suggest next steps.",
    "",
    "### Headers",
    "```json",
    safeJson(LOG_HEADERS ? headers : redactHeaders(headers)),
    "```",
    "",
    "### Body",
    "```json",
    safeJson(LOG_BODY ? obj : { ...obj, raw: undefined }),
    "```",
  ];
  if (saved.length) lines.push("", listLocalFiles(saved));
  if (userTextForOps) {
    lines.push("### Request", userTextForOps, "");
  }
  return lines.join("\n");
}

/** Just pass the composed prompt to gemini; do NOT inline file bytes; gemini reads local files. */
async function runGemini(taskPrompt) {
  const finalPrompt = composePrompt(taskPrompt);

  console.log(color.mag("→ Prompt to model:"));
  console.log(color.dim(truncate(finalPrompt)));

  const args = ["--yolo", "--prompt", finalPrompt];
  const res = await runGeminiProcess(args);

  if (res.code === 0) return res.stdout;
  throw new Error(res.stderr || `gemini exit ${res.code}`);
}

/** Validate a minimal Block Kit object: { blocks: Array } */
function extractBlocks(jsonText) {
  try {
    const obj = JSON.parse(jsonText);
    if (obj && Array.isArray(obj.blocks)) return obj.blocks;
  } catch {}
  return null;
}

function sanitizeFilename(name) {
  return (name || "upload.bin").replace(/[^a-zA-Z0-9._-]/g, "_").slice(0, 128);
}
function uniquify(baseDir, fname) {
  let p = path.join(baseDir, fname);
  if (!fs.existsSync(p)) return p;
  const ext = path.extname(fname);
  const stem = path.basename(fname, ext);
  let i = 1;
  while (true) {
    const trial = path.join(baseDir, `${stem}-${i}${ext}`);
    if (!fs.existsSync(trial)) return trial;
    i++;
  }
}

/**
 * Decode attachments from JSON and write them into CWD.
 * Expects: parsed.attachments: [{ filename, mime, data_base64 }]
 * Returns: [{ absPath, bytes, mime, filename }]
 */
function materializeAttachmentsInCWD(parsed) {
  const saved = [];
  const baseDir = process.cwd();
  const atts = Array.isArray(parsed?.attachments) ? parsed.attachments.slice(0, ATTACH_MAX_FILES) : [];
  
  console.log(`[${ts()}][attach] Processing ${atts.length} attachments`);
  
  for (const att of atts) {
    try {
      const fname = sanitizeFilename(att.filename || "upload.bin");
      const b64   = att.data_base64 || "";
      console.log(`[${ts()}][attach] Processing: ${fname} (${b64.length} base64 chars)`);
      
      if (!b64) {
        console.warn(`[${ts()}][attach] Skipping ${fname}: no base64 data`);
        continue;
      }
      
      const buf   = Buffer.from(b64, "base64");
      if (buf.length > ATTACH_MAX_BYTES) {
        console.warn(`[${ts()}][attach] Skipping ${fname}: ${buf.length} bytes > ${ATTACH_MAX_BYTES} limit`);
        continue;
      }
      
      const dest = uniquify(baseDir, fname);
      fs.writeFileSync(dest, buf);
      const rec = {
        absPath: path.resolve(dest),
        bytes: buf.length,
        mime: att.mime || "application/octet-stream",
        filename: path.basename(dest),
      };
      saved.push(rec);
      console.log(`[${ts()}][attach] ✅ Wrote ${rec.absPath} (${rec.bytes} bytes, ${rec.mime})`);
    } catch (e) {
      console.error(`[${ts()}][attach] ❌ Failed to write attachment: ${e.message}`);
    }
  }
  return saved;
}

// Simple POST JSON
function postJSON(url, payload) {
  const data = JSON.stringify(payload);
  const u = new URL(url);
  const opts = {
    method: "POST",
    hostname: u.hostname,
    path: u.pathname + u.search,
    headers: { "Content-Type": "application/json", "Content-Length": Buffer.byteLength(data) }
  };
  const client = u.protocol === "http:" ? http : https;
  const req = client.request(opts, () => {});
  req.on("error", e => console.error("postJSON error:", e.message));
  req.write(data);
  req.end();
}

/** Post to Slack response_url with name/icon + visibility. Supports blocks. */
function postSlack(responseUrl, text, visibility = "in_channel", options = {}) {
  const payload = { response_type: visibility, text, username: SLACK_BOT_USERNAME };
  if (SLACK_BOT_ICON_URL) payload.icon_url = SLACK_BOT_ICON_URL;
  else if (SLACK_BOT_ICON_EMOJI) payload.icon_emoji = SLACK_BOT_ICON_EMOJI;
  for (const k of ["thread_ts","blocks","attachments","replace_original","delete_original","unfurl_links","unfurl_media"]) {
    if (options[k] !== undefined) payload[k] = options[k];
  }
  postJSON(responseUrl, payload);
}

/* -------------------- server -------------------- */

const server = http.createServer((req, res) => {
  if (req.method === "GET" && req.url === "/health") {
    res.writeHead(200, { "Content-Type": "text/plain" });
    return res.end("ok");
  }

  if (req.method === "POST" && req.url.startsWith("/event")) {
    let raw = "";
    req.on("data", c => (raw += c));
    req.on("end", async () => {
      const ct = (req.headers["content-type"] || "").toLowerCase();

      // Slack slash command (form-encoded)
      if (ct.includes("application/x-www-form-urlencoded")) {
        const params = new URLSearchParams(raw);
        const text = params.get("text") || "";
        const user = params.get("user_name") || "";
        const channel = params.get("channel_name") || "";
        const responseUrl = params.get("response_url");
        const threadTs = params.get("thread_ts");
        // Optional: allow "/cmd blocks ..." to force Block Kit mode
        const wantsBlocks = /\bblocks\b/.test(text) || SLACK_BLOCKS_DEFAULT;

        res.writeHead(200, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ response_type: "ephemeral", text: "Working on it…" }));

        try {
          const prompt = wantsBlocks
            ? buildSlackBlocksPrompt(`Slack user @${user} in #${channel} asked:\n\n${text}`, "", [])
            : `Slack user @${user} in #${channel} asked:\n\n${text}`;

          const reply = await runGemini(prompt);
          const blocks = wantsBlocks ? extractBlocks(reply) : null;

          if (blocks) {
            postSlack(responseUrl, "", "in_channel", {
              thread_ts: threadTs || undefined,
              blocks,
              unfurl_links: false,
              unfurl_media: false
            });
          } else {
            postSlack(responseUrl, reply, "in_channel", {
              thread_ts: threadTs || undefined,
              unfurl_links: false,
              unfurl_media: false
            });
          }
        } catch (e) {
          postSlack(responseUrl, `❌ Error: ${e.message}`, "ephemeral", { thread_ts: threadTs || undefined });
        }
        return;
      }

      // Generic JSON webhook (from Bolt or others)
      let parsed;
      try {
        parsed = ct.includes("application/json") ? JSON.parse(raw || "{}") : { raw };
      } catch (e) {
        console.error(color.red("[json] parse error:"), e);
        res.writeHead(200, { "Content-Type": "application/json" });
        return res.end(JSON.stringify({ ok: false, reply: "❌ Invalid JSON payload." }));
      }

      console.log(`[${ts()}][request] Processing webhook request`);

      // Materialize attachments to CWD
      const saved = materializeAttachmentsInCWD(parsed);

      // Async semantics (response_url or Prefer: respond-async or ?async=1)
      const responseUrl =
        (parsed && typeof parsed === "object" && parsed.response_url) ||
        (parsed && parsed.slack && parsed.slack.response_url) ||
        null;
      const url = new URL(req.url, `http://${req.headers.host}`);
      const forceAsync = url.searchParams.get("async") === "1" ||
        (req.headers["prefer"] || "").toLowerCase().includes("respond-async");
      const mode = (parsed && parsed.mode) || url.searchParams.get("mode") || "";

      console.log(`[${ts()}][request] Mode: ${mode || 'default'}, Async: ${!!(responseUrl || forceAsync)}`);

      // Build prompt AFTER attachments are known (no inlining)
      let prompt;
      let requestText = parsed?.text || "";
      
      console.log(`[${ts()}][audio] === AUDIO DETECTION START ===`);
      console.log(`[${ts()}][audio] Checking ${saved.length} files for audio content`);
      
      // Enhanced audio file detection and processing
      for (const file of saved) {
        const isAudio = /\.(m4a|mp3|wav|ogg|flac|aiff|wma|aac)$/i.test(file.filename);
        console.log(`[${ts()}][audio] File: ${file.filename}, Is audio: ${isAudio}`);
        
        if (isAudio) {
          console.log(`[${ts()}][audio] 🎵 AUDIO FILE DETECTED: ${file.filename}`);
          console.log(`[${ts()}][audio] Starting transcription process...`);
          
          const transcript = await transcribeAudioWithWhisper(file.absPath);
          
          if (transcript) {
            console.log(`[${ts()}][audio] ✅ Transcription successful for ${file.filename}`);
            console.log(`[${ts()}][audio] Transcript length: ${transcript.length} characters`);
            requestText += `\n\n### Audio Transcript from ${file.filename}:\n${transcript}`;
          } else {
            console.warn(`[${ts()}][audio] ❌ Transcription failed for ${file.filename}`);
            requestText += `\n\n⚠️ Unable to transcribe audio file ${file.filename}.`;
          }
        }
      }
      
      console.log(`[${ts()}][audio] === AUDIO DETECTION COMPLETE ===`);
      console.log(`[${ts()}][audio] Final request text length: ${requestText.length}`);

      // Build the appropriate prompt based on mode
      if (mode === "qa_blocks" || (mode === "qa" && SLACK_BLOCKS_DEFAULT)) {
        const context = parsed?.context || "";
        prompt = buildSlackBlocksPrompt(requestText, context, saved);
      } else if (mode === "qa") {
        const context = parsed?.context || "";
        prompt = buildQAPrompt(requestText, context, saved);
      } else {
        prompt = buildOpsPrompt(parsed, req.headers, saved, requestText);
      }

      const doWork = async () => {
        try {
          console.log(`[${ts()}][processing] Starting Gemini processing...`);
          const reply = await runGemini(prompt);
          const blocks = (mode === "qa_blocks" || (mode === "qa" && SLACK_BLOCKS_DEFAULT)) ? extractBlocks(reply) : null;
          console.log(`[${ts()}][processing] ✅ Processing complete`);
          if (blocks) return { ok: true, blocks };
          return { ok: true, reply };
        } catch (e) {
          const msg = String(e.message || e);
          console.error(color.red(`[${ts()}][processing] ❌ Processing failed: ${msg}`));
          const reply =
            `❌ Processing failed.\n\nDetails:\n${truncate(msg, 2000)}\n\n` +
            `Tips:\n• Ensure the local files exist in the working directory (names listed above)\n` +
            `• Ensure your Gemini CLI/tools can read local files when asked\n`;
          return { ok: false, reply };
        }
      };

      if (responseUrl || forceAsync) {
        res.writeHead(202, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ ok: true, status: "accepted", note: "Processing asynchronously" }));

        const result = await doWork();
        if (responseUrl) {
          try {
            const u = new URL(responseUrl);
            if (u.hostname.endsWith("slack.com")) {
              postSlack(
                responseUrl,
                result.blocks ? "" : (result.reply || ""),
                result.ok ? "in_channel" : "ephemeral",
                {
                  blocks: result.blocks || undefined,
                  unfurl_links: false,
                  unfurl_media: false
                }
              );
            } else {
              postJSON(responseUrl, { ok: result.ok, reply: result.reply || null, blocks: result.blocks || null });
            }
          } catch {
            postJSON(responseUrl, { ok: result.ok, reply: result.reply || null, blocks: result.blocks || null });
          }
        }
        return;
      }

      // Sync response (always 200)
      const result = await doWork();
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: result.ok, reply: result.reply || null, blocks: result.blocks || null }));
    });
    return;
  }

  res.writeHead(404, { "Content-Type": "text/plain" });
  res.end("Not found");
});

server.listen(port, () => {
  console.log(`[${ts()}][init] === SERVER STARTUP ===`);
  console.log(`[${ts()}][init] Webhook listener on http://127.0.0.1:${port}/event`);
  console.log(`[${ts()}][init] Health check: http://127.0.0.1:${port}/health`);
});

// Environment diagnostics on startup
console.log(`[${ts()}][init] === ENVIRONMENT DIAGNOSTICS ===`);
console.log(`[${ts()}][init] Node.js version: ${process.version}`);
console.log(`[${ts()}][init] Working directory: ${process.cwd()}`);
console.log(`[${ts()}][init] PATH: ${process.env.PATH}`);
console.log(`[${ts()}][init] VIRTUAL_ENV: ${process.env.VIRTUAL_ENV || '(not set)'}`);
console.log(`[${ts()}][init] WHISPER_BIN: ${WHISPER_BIN}`);
console.log(`[${ts()}][init] WHISPER_VENV: ${WHISPER_VENV || '(not set)'}`);

const { whisperPath } = prepareWhisperEnvironment();
console.log(`[${ts()}][init] Whisper executable: ${whisperPath || 'NOT FOUND'}`);

// Test Whisper availability
if (whisperPath) {
  console.log(`[${ts()}][init] Testing Whisper availability...`);
  try {
    const testResult = spawnSync(whisperPath, ['--help'], { 
      timeout: 5000, 
      encoding: 'utf8',
      env: { ...process.env, PATH: prepareWhisperEnvironment().envPath }
    });
    if (testResult.status === 0) {
      console.log(`[${ts()}][init] ✅ Whisper is working`);
    } else {
      console.warn(`[${ts()}][init] ⚠️ Whisper test failed (status: ${testResult.status})`);
      if (testResult.stderr) console.warn(`[${ts()}][init] Whisper stderr: ${testResult.stderr.slice(0, 200)}`);
    }
  } catch (e) {
    console.warn(`[${ts()}][init] ⚠️ Whisper test error: ${e.message}`);
  }
} else {
  console.error(`[${ts()}][init] ❌ Whisper not found! Audio transcription will not work.`);
  console.error(`[${ts()}][init] To fix this:`);
  console.error(`[${ts()}][init] 1. Install whisper: pip install openai-whisper`);
  console.error(`[${ts()}][init] 2. Set WHISPER_VENV=/path/to/your/venv`);
  console.error(`[${ts()}][init] 3. Or set WHISPER_BIN=/full/path/to/whisper`);
}

console.log(`[${ts()}][init] === STARTUP COMPLETE ===`);

process.on("SIGINT", () => {
  console.log(`[${ts()}][shutdown] Received SIGINT, shutting down…`);
  server.close(() => {
    console.log(`[${ts()}][shutdown] Server closed.`);
    process.exit(0);
  });
  setTimeout(() => {
    console.log(`[${ts()}][shutdown] Force exit after timeout`);
    process.exit(0);
  }, 1500).unref();
});