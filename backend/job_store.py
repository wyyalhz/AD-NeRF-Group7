# 所有 job 存取/追加日志都放这里
import json
from pathlib import Path

def jobs_dir() -> Path:
    d = Path("./static/text/jobs")
    d.mkdir(parents=True, exist_ok=True)
    return d

def job_json_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.json"

def job_log_path(job_id: str) -> Path:
    return jobs_dir() / f"{job_id}.log"

def write_job(job_id: str, payload: dict):
    """原子写，避免读到半截 JSON"""
    p = job_json_path(job_id)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)

def read_job(job_id: str) -> dict:
    p = job_json_path(job_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def append_job_log(job_id: str, line: str):
    lp = job_log_path(job_id)
    with lp.open("a", encoding="utf-8") as f:
        f.write(line.rstrip("\n") + "\n")
