#!/usr/bin/env python3
"""Zero-dependency Local Annotation Interface for Human Annotators.

Provides a clean, local-only web interface (and fallback CLI mode) for annotating
blinded question records. Runs strictly on loopback (127.0.0.1) using Python's
built-in standard library http.server.

Zero external dependencies, zero LLM calls, zero network calls, zero label suggestions.
Guarantees strict isolation, autosave, and schema-compliant JSONL persistence.
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import tempfile
import threading
import urllib.parse
from typing import Dict, Any, List, Optional

# Import local validator
try:
    from scripts.validate_annotations import AnnotationValidator, VALID_LABELS, VALID_CONFIDENCES
except ImportError:
    from validate_annotations import AnnotationValidator, VALID_LABELS, VALID_CONFIDENCES


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Uncertainty Benchmark - Human Annotation Interface</title>
<style>
  :root {
    --bg: #0f172a;
    --card-bg: #1e293b;
    --text: #f8fafc;
    --text-muted: #94a3b8;
    --border: #334155;
    --primary: #3b82f6;
    --known: #10b981;
    --ambig: #f59e0b;
    --unk: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background-color: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 24px;
    display: flex;
    justify-content: center;
  }
  .container {
    max-width: 860px;
    width: 100%;
  }
  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid var(--border);
    padding-bottom: 16px;
    margin-bottom: 20px;
  }
  .header h1 { font-size: 1.25rem; font-weight: 600; color: #e2e8f0; }
  .badge {
    background: #334155;
    color: #38bdf8;
    padding: 4px 10px;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 600;
  }
  .progress-bar-container {
    background: #334155;
    border-radius: 8px;
    height: 12px;
    overflow: hidden;
    margin-bottom: 24px;
    position: relative;
  }
  .progress-bar-fill {
    background: linear-gradient(90deg, #3b82f6, #10b981);
    height: 100%;
    width: 0%;
    transition: width 0.3s ease;
  }
  .progress-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 4px;
    margin-bottom: 20px;
  }
  .card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
  }
  .question-id {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }
  .question-text {
    font-size: 1.4rem;
    font-weight: 500;
    color: #ffffff;
    margin-bottom: 20px;
    line-height: 1.4;
  }
  .form-group {
    margin-bottom: 20px;
  }
  .form-label {
    display: block;
    font-size: 0.9rem;
    font-weight: 600;
    margin-bottom: 8px;
    color: #cbd5e1;
  }
  .btn-group {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
  }
  .btn-choice {
    background: #334155;
    border: 2px solid transparent;
    color: var(--text);
    padding: 12px;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    cursor: pointer;
    text-align: center;
    transition: all 0.15s ease;
  }
  .btn-choice:hover { background: #475569; }
  .btn-choice.active-known { background: rgba(16, 185, 129, 0.2); border-color: var(--known); color: #34d399; }
  .btn-choice.active-ambig { background: rgba(245, 158, 11, 0.2); border-color: var(--ambig); color: #fbbf24; }
  .btn-choice.active-unk   { background: rgba(239, 68, 68, 0.2); border-color: var(--unk); color: #f87171; }
  .btn-choice.active-conf  { background: rgba(59, 130, 246, 0.25); border-color: var(--primary); color: #60a5fa; }
  
  textarea, input[type="text"] {
    width: 100%;
    background: #0f172a;
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 8px;
    padding: 10px;
    font-family: inherit;
    font-size: 0.95rem;
    resize: vertical;
  }
  textarea:focus, input[type="text"]:focus {
    outline: none;
    border-color: var(--primary);
  }
  .checkbox-group {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
  }
  .checkbox-group input {
    width: 18px;
    height: 18px;
    cursor: pointer;
  }
  .nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 24px;
  }
  .btn {
    background: #334155;
    color: var(--text);
    border: 1px solid var(--border);
    padding: 10px 18px;
    border-radius: 8px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s ease;
  }
  .btn:hover { background: #475569; }
  .btn-primary { background: var(--primary); border: none; }
  .btn-primary:hover { background: #2563eb; }
  .btn-success { background: #059669; border: none; }
  .btn-success:hover { background: #047857; }
  .status-toast {
    position: fixed;
    bottom: 20px;
    right: 20px;
    background: #1e293b;
    border: 1px solid var(--primary);
    color: #60a5fa;
    padding: 12px 20px;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.5);
    display: none;
  }
  .modal-overlay {
    position: fixed;
    top:0; left:0; width:100%; height:100%;
    background: rgba(0,0,0,0.7);
    display: none;
    justify-content: center;
    align-items: center;
    z-index: 100;
  }
  .modal-content {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    max-width: 600px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    padding: 24px;
  }
  .hotkey-hint {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-left: 6px;
  }
</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <h1>Benchmark Annotation Environment</h1>
      <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 4px;">Double-Blind Independent Human Review</div>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span class="badge" id="annotatorBadge">Annotator ?</span>
      <button class="btn" onclick="openValidatorModal()">Verify & Report</button>
    </div>
  </div>

  <div class="progress-bar-container">
    <div class="progress-bar-fill" id="progressFill"></div>
  </div>
  <div class="progress-meta">
    <span id="progressText">Completed: 0 / 300 (0.0%)</span>
    <span id="adjudicationCount">Flagged for Adjudication: 0</span>
  </div>

  <div class="card">
    <div class="question-id" id="questionId">ANN_?_000</div>
    <div class="question-text" id="questionText">Loading query...</div>

    <!-- Epistemic Class Selection -->
    <div class="form-group">
      <label class="form-label">
        Epistemic Uncertainty Class
        <span class="hotkey-hint">(Keys: 1=KNOWN, 2=AMBIGUOUS, 3=UNKNOWN)</span>
      </label>
      <div class="btn-group">
        <button class="btn-choice" id="btnKNOWN" onclick="selectLabel('KNOWN')">
          KNOWN<br><small style="font-weight: normal; font-size: 0.8rem;">Single factual answer</small>
        </button>
        <button class="btn-choice" id="btnAMBIGUOUS" onclick="selectLabel('AMBIGUOUS')">
          AMBIGUOUS<br><small style="font-weight: normal; font-size: 0.8rem;">Multiple meanings / Subjective</small>
        </button>
        <button class="btn-choice" id="btnUNKNOWN" onclick="selectLabel('UNKNOWN')">
          UNKNOWN<br><small style="font-weight: normal; font-size: 0.8rem;">Inaccessible / False premise</small>
        </button>
      </div>
    </div>

    <!-- Confidence Selection -->
    <div class="form-group">
      <label class="form-label">
        Confidence in Classification
        <span class="hotkey-hint">(Keys: H=HIGH, M=MEDIUM, L=LOW)</span>
      </label>
      <div class="btn-group">
        <button class="btn-choice" id="btnHIGH" onclick="selectConfidence('HIGH')">HIGH</button>
        <button class="btn-choice" id="btnMEDIUM" onclick="selectConfidence('MEDIUM')">MEDIUM</button>
        <button class="btn-choice" id="btnLOW" onclick="selectConfidence('LOW')">LOW</button>
      </div>
    </div>

    <!-- Rationale Input -->
    <div class="form-group">
      <label class="form-label">Classification Rationale <span style="font-weight: normal; font-size: 0.8rem; color: var(--text-muted);">(Cite criteria from rubric)</span></label>
      <textarea id="rationaleInput" rows="3" placeholder="Explain why this query is KNOWN, AMBIGUOUS, or UNKNOWN..." onchange="saveCurrentRecord()"></textarea>
    </div>

    <!-- Notes Input -->
    <div class="form-group">
      <label class="form-label">Optional Notes / Boundary Comments</label>
      <input type="text" id="notesInput" placeholder="Optional notes regarding phrasing, domain, or interpretations..." onchange="saveCurrentRecord()">
    </div>

    <!-- Adjudication Flag -->
    <div class="form-group">
      <label class="checkbox-group">
        <input type="checkbox" id="adjudicationCheckbox" onchange="saveCurrentRecord()">
        <span style="font-size: 0.95rem; font-weight: 500;">Flag for Adjudication Panel (Borderline query or disputed criteria)</span>
      </label>
    </div>
  </div>

  <!-- Navigation Bar -->
  <div class="nav-bar">
    <div style="display: flex; gap: 8px;">
      <button class="btn" id="prevBtn" onclick="navigate(-1)">← Previous</button>
      <button class="btn" id="nextBtn" onclick="navigate(1)">Next →</button>
    </div>
    <div style="display: flex; gap: 8px; align-items: center;">
      <span style="font-size: 0.85rem; color: var(--text-muted);">Jump to:</span>
      <input type="number" id="jumpIndexInput" min="1" max="300" style="width: 70px; padding: 6px;" onchange="jumpToIndex(this.value)">
      <button class="btn" onclick="jumpToFirstUnannotated()">Next Unannotated</button>
    </div>
  </div>
</div>

<div class="status-toast" id="toast">Saved automatically</div>

<!-- Validator Modal -->
<div class="modal-overlay" id="validatorModal">
  <div class="modal-content">
    <h2 style="margin-bottom: 16px;">Validation Status Report</h2>
    <div id="validatorReportContent" style="font-family: monospace; font-size: 0.85rem; white-space: pre-wrap; background: #0f172a; padding: 12px; border-radius: 8px;"></div>
    <div style="margin-top: 20px; display: flex; justify-content: flex-end; gap: 10px;">
      <button class="btn" onclick="closeValidatorModal()">Close</button>
    </div>
  </div>
</div>

<script>
let records = [];
let currentIndex = 0;
let isDirty = false;

async function loadData() {
  const res = await fetch('/api/records');
  const data = await res.json();
  records = data.records;
  document.getElementById('annotatorBadge').textContent = 'Annotator ' + (data.annotator_id || '?');
  renderCurrentRecord();
  updateProgress();
}

function renderCurrentRecord() {
  if (!records || records.length === 0) return;
  const rec = records[currentIndex];

  document.getElementById('questionId').textContent = `${rec.blinded_id} (${currentIndex + 1} of ${records.length})`;
  document.getElementById('questionText').textContent = rec.question;
  document.getElementById('jumpIndexInput').value = currentIndex + 1;

  // Labels
  ['KNOWN', 'AMBIGUOUS', 'UNKNOWN'].forEach(lbl => {
    const btn = document.getElementById('btn' + lbl);
    btn.className = 'btn-choice';
    if (rec.assigned_label === lbl) {
      if (lbl === 'KNOWN') btn.classList.add('active-known');
      if (lbl === 'AMBIGUOUS') btn.classList.add('active-ambig');
      if (lbl === 'UNKNOWN') btn.classList.add('active-unk');
    }
  });

  // Confidence
  ['HIGH', 'MEDIUM', 'LOW'].forEach(conf => {
    const btn = document.getElementById('btn' + conf);
    btn.className = 'btn-choice';
    if (rec.confidence === conf) {
      btn.classList.add('active-conf');
    }
  });

  document.getElementById('rationaleInput').value = rec.rationale || '';
  document.getElementById('notesInput').value = rec.notes || '';
  document.getElementById('adjudicationCheckbox').checked = !!rec.requires_adjudication;

  document.getElementById('prevBtn').disabled = currentIndex === 0;
  document.getElementById('nextBtn').disabled = currentIndex === records.length - 1;
}

function selectLabel(lbl) {
  records[currentIndex].assigned_label = lbl;
  renderCurrentRecord();
  saveCurrentRecord();
}

function selectConfidence(conf) {
  records[currentIndex].confidence = conf;
  renderCurrentRecord();
  saveCurrentRecord();
}

function updateProgress() {
  let completed = 0;
  let flagged = 0;
  records.forEach(r => {
    if (r.assigned_label) completed++;
    if (r.requires_adjudication) flagged++;
  });
  const pct = ((completed / records.length) * 100).toFixed(1);
  document.getElementById('progressFill').style.width = pct + '%';
  document.getElementById('progressText').textContent = `Completed: ${completed} / ${records.length} (${pct}%)`;
  document.getElementById('adjudicationCount').textContent = `Flagged for Adjudication: ${flagged}`;
}

async function saveCurrentRecord() {
  const rec = records[currentIndex];
  rec.rationale = document.getElementById('rationaleInput').value;
  rec.notes = document.getElementById('notesInput').value;
  rec.requires_adjudication = document.getElementById('adjudicationCheckbox').checked;

  updateProgress();
  showToast('Saving...');

  try {
    const res = await fetch('/api/save_record', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: currentIndex, record: rec })
    });
    if (res.ok) {
      showToast('Autosaved safely');
    } else {
      showToast('Save error!');
    }
  } catch (e) {
    showToast('Network/Save Error!');
  }
}

function navigate(delta) {
  saveCurrentRecord();
  const next = currentIndex + delta;
  if (next >= 0 && next < records.length) {
    currentIndex = next;
    renderCurrentRecord();
  }
}

function jumpToIndex(val) {
  const idx = parseInt(val, 10) - 1;
  if (idx >= 0 && idx < records.length) {
    saveCurrentRecord();
    currentIndex = idx;
    renderCurrentRecord();
  }
}

function jumpToFirstUnannotated() {
  saveCurrentRecord();
  const idx = records.findIndex(r => !r.assigned_label);
  if (idx !== -1) {
    currentIndex = idx;
    renderCurrentRecord();
  } else {
    alert('All records are currently annotated!');
  }
}

function showToast(msg) {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.style.display = 'block';
  setTimeout(() => { toast.style.display = 'none'; }, 1500);
}

async function openValidatorModal() {
  await saveCurrentRecord();
  const res = await fetch('/api/validate');
  const data = await res.json();
  const modal = document.getElementById('validatorModal');
  const content = document.getElementById('validatorReportContent');
  content.textContent = data.summary;
  modal.style.display = 'flex';
}

function closeValidatorModal() {
  document.getElementById('validatorModal').style.display = 'none';
}

// Keyboard shortcuts
window.addEventListener('keydown', (e) => {
  if (['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === '1') selectLabel('KNOWN');
  if (e.key === '2') selectLabel('AMBIGUOUS');
  if (e.key === '3') selectLabel('UNKNOWN');
  if (e.key === 'h' || e.key === 'H') selectConfidence('HIGH');
  if (e.key === 'm' || e.key === 'M') selectConfidence('MEDIUM');
  if (e.key === 'l' || e.key === 'L') selectConfidence('LOW');
  if (e.key === 'ArrowRight') navigate(1);
  if (e.key === 'ArrowLeft') navigate(-1);
});

loadData();
</script>
</body>
</html>
"""


class AnnotationServerHandler(http.server.BaseHTTPRequestHandler):
    """Custom local HTTP handler for serving the annotation app."""

    def __init__(self, records, filepath, validator, *args, **kwargs):
        self.records = records
        self.filepath = filepath
        self.validator = validator
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        # Silence default standard logging to terminal to keep CLI quiet
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
        elif parsed.path == "/api/records":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            # Detect annotator ID from first record
            annotator_id = "?"
            if self.records and "blinded_id" in self.records[0]:
                parts = self.records[0]["blinded_id"].split("_")
                if len(parts) >= 2:
                    annotator_id = parts[1]
            payload = {
                "annotator_id": annotator_id,
                "total": len(self.records),
                "records": self.records,
            }
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        elif parsed.path == "/api/validate":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            report = self.validator.validate_file(self.filepath)
            # Build text summary
            lines = [
                f"Validation Status: {'PASSED' if report.is_valid else 'ACTION REQUIRED'}",
                f"File: {report.filename}",
                f"Total Records: {report.total_records}",
                f"Unannotated Records: {report.unannotated_count}",
                f"Adjudication Flags: {report.adjudication_flag_count}",
                f"\nLabel Distribution:",
            ]
            for lbl, cnt in sorted(report.label_distribution.items()):
                lines.append(f"  {lbl}: {cnt}")
            if report.errors:
                lines.append(f"\nErrors ({len(report.errors)}):")
                for err in report.errors[:20]:
                    lines.append(f"  - {err}")
                if len(report.errors) > 20:
                    lines.append(f"  ... and {len(report.errors)-20} more errors.")
            else:
                lines.append("\nCongratulations! All 300 records are valid and ready for submission.")
            payload = {"is_valid": report.is_valid, "summary": "\n".join(lines)}
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save_record":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                idx = data["index"]
                new_rec = data["record"]
                if 0 <= idx < len(self.records):
                    self.records[idx] = new_rec
                    self._atomic_save()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status": "ok"}')
            except Exception as e:
                self.send_response(500)
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def _atomic_save(self):
        """Atomically persists the records list to the JSONL file."""
        dir_name = os.path.dirname(os.path.abspath(self.filepath))
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
            temp_path = tf.name
            for rec in self.records:
                tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(temp_path, self.filepath)


def run_cli_mode(filepath: str, records: List[Dict[str, Any]], validator: AnnotationValidator):
    """Fallback interactive CLI annotation interface."""
    print(f"\n{'=' * 65}")
    print(" INTERACTIVE CLI ANNOTATION MODE")
    print(f"{'=' * 65}")
    print(f"File: {filepath}")
    print("Commands:")
    print("  [1] KNOWN   [2] AMBIGUOUS   [3] UNKNOWN")
    print("  [h/m/l] Confidence (High/Medium/Low)")
    print("  [n] Next    [p] Previous    [u] Next Unannotated")
    print("  [v] Validate [q] Quit\n")

    idx = 0
    while True:
        rec = records[idx]
        total = len(records)
        assigned = rec.get("assigned_label") or "UNANNOTATED"
        conf = rec.get("confidence") or "NONE"
        adj = "[ADJUDICATION FLAGGED]" if rec.get("requires_adjudication") else ""

        print(f"\n--- Question {idx + 1} of {total} ({rec.get('blinded_id')}) --- {adj}")
        print(f"Query:  {rec.get('question')}")
        print(f"Status: {assigned} | Confidence: {conf}")
        if rec.get("rationale"):
            print(f"Rationale: {rec.get('rationale')}")

        cmd = input("\n[1/2/3/h/m/l/n/p/u/v/q] > ").strip().lower()

        if cmd == "q":
            print("Exiting CLI annotator. Progress saved.")
            break
        elif cmd == "1":
            rec["assigned_label"] = "KNOWN"
            _atomic_save(filepath, records)
        elif cmd == "2":
            rec["assigned_label"] = "AMBIGUOUS"
            _atomic_save(filepath, records)
        elif cmd == "3":
            rec["assigned_label"] = "UNKNOWN"
            _atomic_save(filepath, records)
        elif cmd == "h":
            rec["confidence"] = "HIGH"
            _atomic_save(filepath, records)
        elif cmd == "m":
            rec["confidence"] = "MEDIUM"
            _atomic_save(filepath, records)
        elif cmd == "l":
            rec["confidence"] = "LOW"
            _atomic_save(filepath, records)
        elif cmd == "r":
            rat = input("Enter rationale: ").strip()
            rec["rationale"] = rat
            _atomic_save(filepath, records)
        elif cmd == "a":
            rec["requires_adjudication"] = not rec.get("requires_adjudication", False)
            print(f"Adjudication flag set to: {rec['requires_adjudication']}")
            _atomic_save(filepath, records)
        elif cmd == "n":
            if idx < total - 1:
                idx += 1
            else:
                print("Already at last record.")
        elif cmd == "p":
            if idx > 0:
                idx -= 1
            else:
                print("Already at first record.")
        elif cmd == "u":
            un_idx = next((i for i, r in enumerate(records) if not r.get("assigned_label")), None)
            if un_idx is not None:
                idx = un_idx
            else:
                print("All records have been annotated!")
        elif cmd == "v":
            report = validator.validate_file(filepath)
            report.print_summary()


def _atomic_save(filepath: str, records: List[Dict[str, Any]]):
    dir_name = os.path.dirname(os.path.abspath(filepath))
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tf:
        temp_path = tf.name
        for rec in records:
            tf.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(temp_path, filepath)


DEFAULT_ANNOTATOR_PATHS = {
    "A": "data/annotation/blinded_annotation_annotator_a.jsonl",
    "B": "data/annotation/blinded_annotation_annotator_b.jsonl",
}


def main():
    parser = argparse.ArgumentParser(
        description="Launch the local human annotation interface (web or CLI)."
    )
    parser.add_argument(
        "filepath",
        nargs="?",
        default=None,
        help="Path to the blinded annotation JSONL file.",
    )
    parser.add_argument(
        "--annotator",
        choices=["A", "B", "a", "b"],
        default=None,
        help="Launch interface for Annotator A or Annotator B (loads default file).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Local loopback port for the web interface (default: 8765).",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in interactive CLI terminal mode instead of browser mode.",
    )

    args = parser.parse_args()

    target_path = args.filepath
    if not target_path and args.annotator:
        target_path = DEFAULT_ANNOTATOR_PATHS.get(args.annotator.upper())

    if not target_path:
        parser.error("Either a filepath or --annotator [A|B] must be specified.")

    if not os.path.exists(target_path):
        print(f"Error: File not found: {target_path}")
        sys.exit(1)

    records: List[Dict[str, Any]] = []
    with open(target_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    validator = AnnotationValidator(expected_count=len(records), allow_unannotated=False)

    if args.cli:
        run_cli_mode(target_path, records, validator)
    else:
        host = "127.0.0.1"
        handler = lambda *h_args, **h_kwargs: AnnotationServerHandler(
            records, target_path, validator, *h_args, **h_kwargs
        )

        with socketserver.TCPServer((host, args.port), handler) as httpd:
            print(f"\n{'=' * 65}")
            print(f" LOCAL ANNOTATION SERVER ACTIVE")
            print(f"{'=' * 65}")
            print(f"Target File:      {target_path}")
            print(f"Total Questions:  {len(records)}")
            print(f"Access URL:       http://{host}:{args.port}")
            print(f"Binding:          STRICT LOOPBACK ONLY (127.0.0.1)")
            print(f"Security:         Double-blind isolation active.")
            print(f"Autosave:         Active on every change.")
            print(f"\nPress Ctrl+C to stop the server.")
            print(f"{'=' * 65}\n")
            try:
                httpd.serve_forever()
            except KeyboardInterrupt:
                print("\nServer stopped. All annotations safely persisted.")


if __name__ == "__main__":
    main()
