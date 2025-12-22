# 封装“启动 AD-NeRF 推理任务（后台线程）+ 写 log + 写 job.json”
import os
import time
import threading
import subprocess
import shutil

from .job_store import write_job, read_job, append_job_log, job_log_path

def _stem_from_audio_path(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    return stem

def start_adnerf_job(project_root: str, static_videos_dir: str,
                     person_id: str, in_audio: str, test_size: int,
                     job_id: str | None = None) -> str:
    """
    启动后台任务，返回 job_id
    产物：
      - static/text/jobs/<job_id>.json
      - static/text/jobs/<job_id>.log
    """
    if job_id is None:
        job_id = str(int(time.time() * 1000))

    # init job + clear log
    write_job(job_id, {
        "ready": False,
        "error": "",
        "video_path": "",
        "started_at": time.time(),
        "finished_at": 0,
    })
    job_log_path(job_id).write_text("", encoding="utf-8")

    def worker():
        try:
            run_adnerf = os.path.join(project_root, "AD-NeRF", "run_adnerf.sh")
            cmd = [
                "bash", run_adnerf, "infer",
                "--id", person_id,
                "--aud_file", in_audio,
                "--test_size", str(test_size),
            ]
            append_job_log(job_id, f"[backend] CMD: {' '.join(cmd)}")

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"

            p = subprocess.Popen(
                cmd,
                cwd=os.path.join(project_root, "AD-NeRF"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
            )

            assert p.stdout is not None
            for line in p.stdout:
                append_job_log(job_id, line.rstrip("\n"))

            rc = p.wait()
            append_job_log(job_id, f"[backend] exit={rc}")
            if rc != 0:
                raise RuntimeError(f"AD-NeRF infer failed, exit={rc}")

            name = _stem_from_audio_path(in_audio)
            source_path = os.path.join(
                project_root, "AD-NeRF", "dataset", person_id, "render_out", f"{name}.mp4"
            )
            if not os.path.exists(source_path):
                raise FileNotFoundError(f"未找到输出视频: {source_path}")

            os.makedirs(static_videos_dir, exist_ok=True)
            destination_name = f"adnerf_{person_id}_{name}.mp4"
            destination_path = os.path.join(static_videos_dir, destination_name)
            shutil.copy(source_path, destination_path)

            rel_path = os.path.join("static", "videos", destination_name).replace("\\", "/")
            append_job_log(job_id, f"[backend] DONE video: /{rel_path}")

            write_job(job_id, {
                "ready": True,
                "error": "",
                "video_path": rel_path,
                "started_at": read_job(job_id).get("started_at", time.time()),
                "finished_at": time.time(),
            })

        except Exception as e:
            append_job_log(job_id, f"[backend] ERROR: {e}")
            write_job(job_id, {
                "ready": True,
                "error": str(e),
                "video_path": "",
                "started_at": read_job(job_id).get("started_at", time.time()),
                "finished_at": time.time(),
            })

    threading.Thread(target=worker, daemon=True).start()
    return job_id
