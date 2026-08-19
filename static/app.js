"use strict";

let TARGETS = ["rk3588"];
let AVAILABILITY = {};
let currentSource = null;

// Field schema per pipeline. Rendered into #params on selection.
const OPSET = { key: "opset", label: "opset", type: "number", min: 12, max: 19, step: 1, default: 17 };
const IMGSZ = { key: "imgsz", label: "imgsz", type: "number", min: 32, max: 4096, step: 32, default: 640 };
const HALF = { key: "half", label: "half (FP16)", type: "checkbox", default: false };

const FIELDS = {
  pt_to_onnx: [
    OPSET,
    { key: "simplify", label: "simplify", type: "checkbox", default: true },
    { key: "dynamic", label: "dynamic shapes", type: "checkbox", default: false },
    HALF,
    { key: "batch", label: "batch", type: "number", min: 1, max: 64, step: 1, default: 1 },
    IMGSZ,
  ],
  pt_to_rknn: [
    { key: "target_platform", label: "target platform", type: "select", dynamic: () => TARGETS, default: "rk3588" },
    {
      key: "quantize",
      label: "quantize",
      type: "select",
      options: [["0", "FP32 (none)"], ["8", "INT8 (+calibration)"], ["16", "FP16 / w16a16"]],
      default: "0",
    },
    OPSET,
    IMGSZ,
    { ...HALF, label: "half (FP16 ONNX; conflicts with INT8)" },
  ],
  onnx_to_rknn: [
    { key: "target_platform", label: "target platform", type: "select", dynamic: () => TARGETS, default: "rk3588" },
    { key: "do_quantization", label: "INT8 quantization", type: "checkbox", default: false },
    { key: "quantized_dtype", label: "quantized dtype", type: "select", options: ["w8a8", "w16a16", "w4a16", "w4a8", "bf16"], default: "w8a8" },
    { key: "quantized_method", label: "quantized method", type: "select", options: ["channel", "layer"], default: "channel" },
    { key: "quantized_algorithm", label: "quantized algorithm", type: "select", options: ["mmse", "normal", "kl"], default: "mmse" },
    { key: "mean_values", label: "mean (RGB)", type: "csv3", default: "0,0,0" },
    { key: "std_values", label: "std (RGB)", type: "csv3", default: "255,255,255" },
    OPSET,
    IMGSZ,
  ],
};

const $ = (id) => document.getElementById(id);

function currentPipeline() {
  const el = document.querySelector('input[name="pipeline"]:checked');
  return el ? el.value : "pt_to_onnx";
}

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
 return node;
}

function renderParams() {
  const pipeline = currentPipeline();
  const container = $("params");
  container.innerHTML = "";
  for (const f of FIELDS[pipeline]) {
    const label = el("label", { class: "pfield" }, el("span", { class: "pname" }, f.label));
    let input;
    if (f.type === "select") {
      input = el("select", { id: "p-" + f.key, "data-key": f.key, "data-type": f.type });
      const opts = f.options || (f.dynamic ? f.dynamic().map((t) => [t, t]) : []);
      for (const [val, text] of opts) {
        const o = el("option", { value: String(val) }, String(text));
        if (String(val) === String(f.default)) o.selected = true;
        input.appendChild(o);
      }
    } else if (f.type === "checkbox") {
      input = el("input", { id: "p-" + f.key, "data-key": f.key, "data-type": f.type, type: "checkbox" });
      if (f.default) input.checked = true;
    } else {
      input = el("input", {
        id: "p-" + f.key, "data-key": f.key, "data-type": f.type,
        type: f.type === "csv3" ? "text" : f.type, value: String(f.default),
      });
      if (f.min != null) input.min = f.min;
      if (f.max != null) input.max = f.max;
      if (f.step != null) input.step = f.step;
    }
    input.addEventListener("change", refreshConditional);
    label.appendChild(input);
    container.appendChild(label);
  }
  refreshConditional();
}

function refreshConditional() {
  const pipeline = currentPipeline();
  // Pipeline-specific Detect-head notes.
  $("rknn-warning").classList.toggle("hidden", pipeline !== "onnx_to_rknn");
  $("pt-rknn-note").classList.toggle("hidden", pipeline !== "pt_to_rknn");
  // Calibration field: visible only when INT8 is active.
  let int8 = false;
  if (pipeline === "pt_to_rknn") {
    int8 = val("quantize") === 8;
  } else if (pipeline === "onnx_to_rknn") {
    int8 = !!val("do_quantization") && ["w8a8", "w4a8"].includes(String(val("quantized_dtype")));
  }
  $("calib-field").classList.toggle("hidden", !int8);
  $("calib").required = int8;
}

function val(key) {
  const node = document.querySelector(`[data-key="${key}"]`);
  if (!node) return undefined;
  if (node.type === "checkbox") return node.checked;
  if (node.dataset.type === "number") return parseInt(node.value, 10);
  if (node.dataset.type === "csv3") {
    return node.value.split(",").map((s) => parseInt(s.trim(), 10));
  }
  // select with numeric option (quantize)
  if (key === "quantize") {
    const n = parseInt(node.value, 10);
    return isNaN(n) || n === 0 ? null : n;
  }
  return node.value;
}

function collectParams() {
  const pipeline = currentPipeline();
  const params = { pipeline };
  for (const f of FIELDS[pipeline]) {
    params[f.key] = val(f.key);
  }
  return params;
}

function setMsg(text, cls) {
  const m = $("submit-msg");
  m.textContent = text || "";
  m.className = cls || "muted";
}

async function loadTargets() {
  try {
    const r = await fetch("/api/targets");
    const d = await r.json();
    TARGETS = d.targets || TARGETS;
    AVAILABILITY = d.availability || {};
    renderAvailability();
  } catch (e) {
    /* keep defaults */
  }
  renderParams();
}

function renderAvailability() {
  const box = $("availability");
  box.innerHTML = "";
  const lines = {
    pt_to_onnx: "PT → ONNX",
    pt_to_rknn: "PT → RKNN",
    onnx_to_rknn: "ONNX → RKNN",
  };
  for (const [k, label] of Object.entries(lines)) {
    const ok = AVAILABILITY[k];
    const tag = el("span", { class: "tag " + (ok ? "tag-ok" : "tag-off") }, ok ? "ready" : "unavailable");
    box.appendChild(el("div", { class: "avail-line" }, el("span", {}, label), tag));
  }
  if (AVAILABILITY.mock) {
    box.appendChild(el("div", { class: "avail-line muted" }, "⚠ mock mode (no real conversion)"));
  }
}

function appendLog(line) {
  const log = $("log");
  log.textContent += line + "\n";
  log.scrollTop = log.scrollHeight;
}

function setProgress(pct, msg) {
  $("progress-bar").style.width = (pct || 0) + "%";
  if (msg) $("status-line").textContent = msg;
}

function openSSE(jobId) {
  if (currentSource) currentSource.close();
  $("output").classList.remove("hidden");
  $("job-id").textContent = jobId;
  $("log").textContent = "";
  setProgress(0, "queued…");
  $("download").classList.add("hidden");

  const src = new EventSource(`/api/jobs/${jobId}/events`);
  currentSource = src;
  src.onmessage = (ev) => {
    let evt;
    try { evt = JSON.parse(ev.data); } catch (e) { return; }
    if (evt.type === "snapshot") {
      (evt.progress || []).forEach(appendLog);
      if (evt.status === "success" || evt.status === "failed") {
        finalize(evt.status, null);
      }
    } else if (evt.type === "log") {
      appendLog(evt.msg);
    } else if (evt.type === "progress") {
      setProgress(evt.pct, evt.msg);
      if (evt.msg) appendLog(evt.msg);
    } else if (evt.type === "end") {
      finalize(evt.status, evt);
    }
  };
  src.onerror = () => { /* keepalive comment lines or network blips; EventSource auto-reconnects */ };
}

function finalize(status, evt) {
  if (currentSource) { currentSource.close(); currentSource = null; }
  if (status === "success") {
    const jobId = $("job-id").textContent;
    setProgress(100, "done ✓");
    const dl = $("download");
    dl.href = `/api/jobs/${jobId}/download`;
    dl.textContent = "Download " + (evt && evt.result ? evt.result : "result");
    dl.classList.remove("hidden");
    setMsg("Conversion succeeded.", "ok");
  } else {
    setProgress($("progress-bar").style.width ? parseInt($("progress-bar").style.width) : 0, "failed ✗");
    if (evt && evt.error) appendLog("ERROR: " + evt.error);
    setMsg("Conversion failed.", "err");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadTargets();
  document.querySelectorAll('input[name="pipeline"]').forEach((r) => r.addEventListener("change", renderParams));

  $("job-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const model = $("model").files[0];
    if (!model) { setMsg("Choose a model file.", "err"); return; }
    const params = collectParams();
    const fd = new FormData();
    fd.append("params", JSON.stringify(params));
    fd.append("model", model);
    if (!$("calib-field").classList.contains("hidden")) {
      const c = $("calib").files[0];
      if (!c) { setMsg("INT8 requires the calibration zip.", "err"); return; }
      fd.append("calib", c);
    }
    $("submit").disabled = true;
    setMsg("Submitting…", "muted");
    try {
      const r = await fetch("/api/jobs", { method: "POST", body: fd });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || `HTTP ${r.status}`);
      }
      const d = await r.json();
      setMsg("Job started.", "ok");
      openSSE(d.id);
    } catch (err) {
      setMsg(err.message, "err");
    } finally {
      $("submit").disabled = false;
    }
  });
});
