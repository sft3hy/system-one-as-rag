"""Tier B encoder sidecar: local HTTP server speaking the SystemOne subset.

    HF sidecar for laya:   .venv-encoder/bin/python scripts/encoder_sidecar.py laya 8101
    HF sidecar for von:    .venv-encoder/bin/python scripts/encoder_sidecar.py von 8102

Protocol (the subset our EncoderSidecarStrategy consumes):
    POST /system_one {"state": str, "questions": {...typed...}}
      -> {"answers": {qid: {"probabilities": {...}, "label": str}}, "truncated": bool}
    GET /health -> {"ok": true, "model": ...}

Kept deliberately stdlib-http: the only heavyweight deps are the model SDKs
themselves, installed in .venv-encoder so the harness venv stays lean and
encoders never touch the GPU decision slot.
"""
import json
import os
import sys
import pathlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = (sys.argv[1] if len(sys.argv) > 1 else "laya").lower()
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8101
MODELS_DIR = pathlib.Path.home() / "models"


def build_backend(model: str):
    if model in ("laya", "laya-typed-decisions"):
        from laya import Router  # pip install laya
        subdir = "typed-decisions" if model == "laya-typed-decisions" else "multilingual"
        if model == "laya-typed-decisions":
            src = str(MODELS_DIR / "convaiinnovations" / "laya-typed-decisions")
        else:
            src = str(MODELS_DIR / "convaiinnovations" / "laya" / subdir)
        # local path first so we never re-fetch 3 GB; "default" is what predict
        # routes to when the request carries no explicit model name
        router = Router(models={"default": src}, default="default", preload=True)

        def predict(state, questions):
            # laya's native question format IS our wire dict:
            # {"type": "choice"|"noul"|"score", "instructions", "criteria"}
            res = router.predict(state, dict(questions))
            answers = res.get("answers", {}) if isinstance(res, dict) else {}
            out = {}
            for qid, a in answers.items():
                d = a if isinstance(a, dict) else getattr(a, "model_dump", lambda: dict(a))()
                out[qid] = {"probabilities": d.get("probabilities") or {},
                            "label": str(d.get("choice", d.get("value", "")))}
            return out, bool(res.get("truncated", False)) \
                if isinstance(res, dict) else False
        return f"laya ({model})", predict

    if model == "von":
        import os
        os.environ.setdefault("VON_CHECKPOINT_DIR", str(MODELS_DIR / "wfzyx" / "von"))
        import von  # pip install von-sdk (module name: von)
        client = von.VonClient(local=True, timeout=120.0)

        def predict(state, questions):
            res = client.system_one(state, questions)
            dump = res.model_dump()
            return dump.get("answers", {}), bool(dump.get("truncation"))
        return "von-1.3", predict

    raise SystemExit(f"unknown encoder model {model!r}; supported: laya, laya-typed-decisions, von")


BACKEND_NAME, PREDICT = build_backend(MODEL)


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"ok": True, "model": BACKEND_NAME})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/systemone":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            answers, truncated = PREDICT(req["state"], req["questions"])
            self._send(200, {"answers": answers, "truncated": truncated})
        except Exception as e:  # surface, never kill the server
            self._send(500, {"error": f"{type(e).__name__}: {e}"})

    def log_message(self, fmt, *args):  # silence per-request spam
        return


if __name__ == "__main__":
    print(f"[sidecar] {BACKEND_NAME} on :{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
