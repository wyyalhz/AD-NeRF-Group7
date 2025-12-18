import subprocess
import os
import time

def train_model(data):
    """
    模拟模型训练逻辑。
    """
    print("[backend.model_trainer] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")
    
    print("[backend.model_trainer] 模型训练中...")
        
    if data['model_choice'] == "AD-NeRF":
        try:
            person_id = (data.get("id") or "Obama").strip()
            preprocess = (data.get("preprocess") or "off").strip().lower()
            if preprocess not in ("on", "off"):
                preprocess = "off"

            cmd = [
                "./AD-NeRF/run_adnerf.sh", "train",
                "--id", person_id,
                "--preprocess", preprocess,
            ]

            print(f"[backend.model_trainer] [AD-NeRF] 执行命令: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # TFG_ui 根目录
            )

            print("[backend.model_trainer] [AD-NeRF] 标准输出:", result.stdout)
            if result.stderr:
                print("[backend.model_trainer] [AD-NeRF] 标准错误:", result.stderr)

        except Exception as e:
            print(f"[backend.model_trainer] [AD-NeRF] 训练失败: {e}")
            
    video_path = f"dataset/vids/{person_id}.mp4"


    print("[backend.model_trainer]训练完成")
    return video_path
