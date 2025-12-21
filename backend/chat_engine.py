import os
import random
from pathlib import Path
import time
import requests

# 可选：语音转文字（你现在先忽略也行）
try:
    import speech_recognition as sr
except Exception:
    sr = None


GENIE_TTS_URL = "http://122.9.44.244:8000/tts"
GENIE_TOKEN = ""  # 如果你云端设置了 GENIE_TOKEN，就填同一个

# ====== 路径配置（跟你们现在代码保持一致） ======
INPUT_AUDIO = "./static/audio/aud.wav"
INPUT_TEXT_FILE = "./static/text/input.txt"
OUTPUT_TEXT_FILE = "./static/text/output.txt"

def tts_via_cloud(text: str, save_wav_path: str) -> str:
    headers = {}
    if GENIE_TOKEN:
        headers["X-Token"] = GENIE_TOKEN

    r = requests.post(
        GENIE_TTS_URL,
        json={"text": text},
        headers=headers,
        timeout=180,
    )
    r.raise_for_status()

    with open(save_wav_path, "wb") as f:
        f.write(r.content)

    return save_wav_path


def chat_response(data: dict):
    """
    实时对话系统（当前阶段：生成 AI 回复文本，写到 output.txt）
    之后你们可以把 output 接到 Genie TTS -> AD-NeRF -> ffmpeg
    """
    print("[backend.chat_engine] 收到数据：")
    for k, v in data.items():
        print(f"  {k}: {v}")

    api_choice = (data.get("api_choice") or "dummy").strip().lower()

    # 1) 获取用户输入文本（优先：音频->文字；否则：input.txt；否则：data['text']）
    user_text = _get_user_text(data)

    # 2) 调大模型 / dummy 生成回复
    reply = get_ai_response(
        user_text=user_text,
        api_choice=api_choice,
        output_text_path=OUTPUT_TEXT_FILE,
        data=data,
    )

    # 你们目前 app.py 期待返回 video_path（哪怕现在还没生成视频）
    video_path = os.path.join("static", "videos", "chat_response.mp4")
    print(f"[backend.chat_engine] 回复文本：{reply}")
    print(f"[backend.chat_engine] 生成视频路径（占位）：{video_path}")
    
    # 写出 output.txt（保持你们前端逻辑不变）
    Path(OUTPUT_TEXT_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_TEXT_FILE).write_text(reply, encoding="utf-8")
    print(f"[backend.chat_engine] 答复已保存到: {OUTPUT_TEXT_FILE}")
    
    # 立刻做 TTS，输出到 static/audios，返回给前端播放
    ts = int(time.time() * 1000)
    wav_filename = f"reply_{ts}.wav"
    wav_disk_path = f"./static/audios/{wav_filename}"
    audio_url = f"/static/audios/{wav_filename}"

    try:
        tts_via_cloud(reply, wav_disk_path)
    except Exception as e:
        # 不要让它 500：验收时至少文字还能出
        print(f"[backend.chat_engine] TTS 失败，跳过音频。原因：{e}")
        audio_url = ""
    
    print("TTS wav saved to:", wav_filename)
    
    return {
        "video_path": "",      # 你们以后再填真实生成的视频
        "audio_path": audio_url,
        # "reply_text": reply,   # 可选：以后前端也能显示文字
    }


# -----------------------------
# 输入获取：音频优先，其次本地 input.txt，再其次 data['text']
# -----------------------------
def _get_user_text(data: dict) -> str:
    # A) 有音频且你想用语音输入
    if os.path.exists(INPUT_AUDIO) and sr is not None:
        try:
            text = audio_to_text(INPUT_AUDIO, INPUT_TEXT_FILE)
            if text:
                return text.strip()
        except Exception as e:
            print(f"[backend.chat_engine] 语音识别失败，将尝试文字输入。原因：{e}")

    # B) 直接读 input.txt（你说准备改成只输入文本，这条就是主路）
    if os.path.exists(INPUT_TEXT_FILE):
        txt = Path(INPUT_TEXT_FILE).read_text(encoding="utf-8").strip()
        if txt:
            return txt

    # C) 如果前端/调用方愿意直接传 text
    direct = (data.get("text") or "").strip()
    if direct:
        Path(INPUT_TEXT_FILE).parent.mkdir(parents=True, exist_ok=True)
        Path(INPUT_TEXT_FILE).write_text(direct, encoding="utf-8")
        return direct

    # 都没有就报错（但别让 Flask 500：给个可读提示）
    raise FileNotFoundError(
        f"找不到输入：既没有 {INPUT_AUDIO}，也没有 {INPUT_TEXT_FILE}，也没在请求里传 text。"
    )


def audio_to_text(input_audio: str, input_text: str) -> str:
    if sr is None:
        raise RuntimeError("speech_recognition 未安装或导入失败。")

    recognizer = sr.Recognizer()
    with sr.AudioFile(input_audio) as source:
        recognizer.adjust_for_ambient_noise(source)
        audio_data = recognizer.record(source)

    print("[backend.chat_engine] 正在识别语音...")
    text = recognizer.recognize_google(audio_data, language="zh-CN")

    Path(input_text).parent.mkdir(parents=True, exist_ok=True)
    Path(input_text).write_text(text, encoding="utf-8")

    print(f"[backend.chat_engine] 语音识别完成：{text}")
    return text


# -----------------------------
# 生成 AI 回复：dummy / groq / openrouter / openai / zhipu
# -----------------------------
def get_ai_response(
    user_text: str,
    api_choice: str,
    output_text_path: str,
    data: dict | None = None,
) -> str:
    data = data or {}

    try:
        if api_choice == "dummy":
            reply = _dummy_reply(user_text)

        elif api_choice == "groq":
            # 需要：export GROQ_API_KEY=...
            # 可选：export GROQ_MODEL=...
            reply = _openai_compatible_chat(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.getenv("GROQ_API_KEY", "").strip(),
                model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"),
                user_text=user_text,
            )

        elif api_choice == "openrouter":
            # 需要：export OPENROUTER_API_KEY=...
            # 可选：export OPENROUTER_MODEL=...
            reply = _openai_compatible_chat(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
                model=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
                user_text=user_text,
                extra_headers={
                    # OpenRouter 建议带（可留空/可删）
                    "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
                    "X-Title": os.getenv("OPENROUTER_X_TITLE", "AD-NeRF-Group7"),
                },
            )

        elif api_choice == "openai":
            # 需要：export OPENAI_API_KEY=...
            # 可选：export OPENAI_MODEL=...
            reply = _openai_compatible_chat(
                base_url=os.getenv("OPENAI_BASE_URL", ""),  # 为空则走默认 OpenAI
                api_key=os.getenv("OPENAI_API_KEY", "").strip(),
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                user_text=user_text,
            )

        elif api_choice == "zhipu":
            # 需要：export ZHIPU_API_KEY=d957e120df71402f8e56eb9d521df3b4.9TxLF1EsGHACDl32
            # 可选：export ZHIPU_MODEL=glm-4-flash-250414
            reply = _zhipu_chat(
                api_key=os.getenv("ZHIPU_API_KEY", "").strip(),
                model=os.getenv("ZHIPU_MODEL", "glm-4-flash-250414"),
                user_text=user_text,
            )
        
        elif api_choice == "deepseek":
            # export DEEPSEEK_API_KEY=sk-292bd1e00ec841ae98a9e36790014bfe
            # export DEEPSEEK_MODEL="deepseek-chat"   # 可选
            reply = _openai_compatible_chat(
                base_url="https://api.deepseek.com",
                api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                user_text=user_text,
            )

        else:
            # 未知选择：降级 dummy
            reply = _dummy_reply(user_text)

    except Exception as e:
        print(f"[backend.chat_engine] LLM 调用失败（{api_choice}），降级 dummy。原因：{e}")
        reply = _dummy_reply(user_text)

    # # 写出 output.txt（保持你们前端逻辑不变）
    # Path(output_text_path).parent.mkdir(parents=True, exist_ok=True)
    # Path(output_text_path).write_text(reply, encoding="utf-8")
    # print(f"[backend.chat_engine] 答复已保存到: {output_text_path}")
    
    # 保存音频到本地
    # tts_audio_path = "./static/audios/tts.wav"
    # tts_via_cloud(reply, tts_audio_path)
    # print("TTS wav saved to:", tts_audio_path)

    return reply


def _dummy_reply(user_text: str) -> str:
    # 作业保命：短、稳定、不会胡写大作文
    templates = [
        "我收到了：{x}。我会简短回答：好的。",
        "你说“{x}”。我理解了：可以。",
        "收到：{x}。我的回应是：没问题。",
        "已读取输入：{x}。简短回复：明白。",
    ]
    x = user_text.strip().replace("\n", " ")
    x = x[:60] + ("…" if len(x) > 60 else "")
    return random.choice(templates).format(x=x)


def _openai_compatible_chat(
    base_url: str,
    api_key: str,
    model: str,
    user_text: str,
    extra_headers: dict | None = None,
) -> str:
    if not api_key:
        raise RuntimeError("缺少 API Key（环境变量未设置）。")

    # 延迟导入：避免你只用 dummy 时还要装 openai
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if extra_headers:
        kwargs["default_headers"] = extra_headers

    client = OpenAI(**kwargs)

    messages = [
        {"role": "system", "content": "你是一个简短回答的助手，每次回答不超过30字。"},
        {"role": "user", "content": user_text},
    ]

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=80,
        temperature=0.7,
    )
    return (resp.choices[0].message.content or "").strip()


def _zhipu_chat(api_key: str, model: str, user_text: str) -> str:
    if not api_key:
        raise RuntimeError("缺少 ZHIPU_API_KEY（环境变量未设置）。")

    # 你们原本就装了 zhipuai
    from zhipuai import ZhipuAI

    client = ZhipuAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个简短回答的助手，每次回答不超过30字。"},
            {"role": "user", "content": user_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()
