import os
import subprocess
import shutil


def _safe_int(value, default=300):
    """把 test_size 转成 int，非法就用默认值。"""
    try:
        return int(value)
    except Exception:
        return default


def _stem_from_audio_path(path: str) -> str:
    """从输入音频路径推导基名：other.wav/other.npy -> other"""
    base = os.path.basename(path)
    stem, _ext = os.path.splitext(base)
    return stem


def _tfg_root() -> str:
    """
    返回 TFG_ui 项目根目录（.../TFG_ui）。
    当前位置：TFG_ui/backend/video_generator.py
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_video(data):
    """
    视频生成逻辑：接收来自前端的参数，并返回一个视频路径（相对 static/ 的路径）。
    """
    print("[backend.video_generator] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    project_root = _tfg_root()  # /root/TFG_ui
    static_videos_dir = os.path.join(project_root, "static", "videos")
    os.makedirs(static_videos_dir, exist_ok=True)

    # ============================
    # SyncTalk 分支（基本保持原逻辑，改成绝对路径更稳）
    # ============================
    if data.get("model_name") == "SyncTalk":
        try:
            cmd = [
                os.path.join(project_root, "SyncTalk", "run_synctalk.sh"), "infer",
                "--model_dir", data.get("model_param", ""),
                "--audio_path", data.get("ref_audio", ""),
                "--gpu", data.get("gpu_choice", "GPU0"),
            ]

            print(f"[backend.video_generator] 执行命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=project_root,   # 确保工作目录在 TFG_ui
            )

            print("命令标准输出:", result.stdout)
            if result.stderr:
                print("命令标准错误:", result.stderr)

            model_dir_name = os.path.basename(data.get("model_param", ""))
            source_path = os.path.join(
                project_root, "SyncTalk", "model", model_dir_name, "results", "test_audio.mp4"
            )

            audio_name = os.path.splitext(os.path.basename(data.get("ref_audio", "")))[0]
            video_filename = f"{model_dir_name}_{audio_name}.mp4"
            destination_path = os.path.join(static_videos_dir, video_filename)

            if os.path.exists(source_path):
                shutil.copy(source_path, destination_path)
                rel_path = os.path.join("static", "videos", video_filename)
                print(f"[backend.video_generator] 视频生成完成，路径：{rel_path}")
                return rel_path

            print(f"[backend.video_generator] 视频文件不存在: {source_path}")

            # 兜底：找 results 下最新 mp4
            results_dir = os.path.join(project_root, "SyncTalk", "model", model_dir_name, "results")
            if os.path.exists(results_dir):
                mp4_files = [f for f in os.listdir(results_dir) if f.endswith(".mp4")]
                if mp4_files:
                    latest_file = max(
                        mp4_files,
                        key=lambda f: os.path.getctime(os.path.join(results_dir, f)),
                    )
                    source_path = os.path.join(results_dir, latest_file)
                    shutil.copy(source_path, destination_path)
                    rel_path = os.path.join("static", "videos", video_filename)
                    print(f"[backend.video_generator] 找到最新视频文件: {rel_path}")
                    return rel_path

            return os.path.join("static", "videos", "out.mp4")

        except Exception as e:
            print(f"[backend.video_generator] [SyncTalk] 错误: {e}")
            return os.path.join("static", "videos", "out.mp4")

    # ============================
    # AD-NeRF 分支（支持可变 ID）
    # ============================
    elif data.get("model_name") == "AD-NeRF":
        try:
            person_id = (data.get("id") or "Obama").strip()
            in_audio = (data.get("ref_audio") or "").strip()
            if not in_audio:
                print("[backend.video_generator] [AD-NeRF] ref_audio 为空")
                return os.path.join("static", "videos", "out.mp4")

            test_size = _safe_int(data.get("test_size", 300), default=300)

            # 运行推理脚本（传 --id）
            run_adnerf = os.path.join(project_root, "AD-NeRF", "run_adnerf.sh")
            cmd = [
                run_adnerf, "infer",
                "--id", person_id,
                "--aud_file", in_audio,
                "--test_size", str(test_size),
            ]

            print(f"[backend.video_generator] [AD-NeRF] 执行命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.join(project_root, "AD-NeRF"),  # 确保脚本在 AD-NeRF 目录里跑
            )

            print("[backend.video_generator] [AD-NeRF] 标准输出:", result.stdout)
            if result.stderr:
                print("[backend.video_generator] [AD-NeRF] 标准错误:", result.stderr)

            # 按你的 run_infer.sh 规则：输出 = dataset/<ID>/render_out/<name>.mp4
            name = _stem_from_audio_path(in_audio)
            render_dir = os.path.join(project_root, "AD-NeRF", "dataset", person_id, "render_out")
            source_path = os.path.join(render_dir, f"{name}.mp4")

            if not os.path.exists(source_path):
                print(f"[backend.video_generator] [AD-NeRF] 未找到输出视频: {source_path}")
                return os.path.join("static", "videos", "out.mp4")

            # 复制到 static/videos 供前端访问
            destination_name = f"adnerf_{person_id}_{name}.mp4"
            destination_path = os.path.join(static_videos_dir, destination_name)
            shutil.copy(source_path, destination_path)

            rel_path = os.path.join("static", "videos", destination_name)
            print(f"[backend.video_generator] [AD-NeRF] 视频生成完成，路径：{rel_path}")
            return rel_path

        except Exception as e:
            print(f"[backend.video_generator] [AD-NeRF] 推理失败: {e}")
            return os.path.join("static", "videos", "out.mp4")

    # ============================
    # 兜底
    # ============================
    return os.path.join("static", "videos", "out.mp4")
