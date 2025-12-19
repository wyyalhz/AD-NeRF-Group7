from flask import Flask, render_template, request, jsonify
import os
from backend.video_generator import generate_video
from backend.model_trainer import train_model
from backend.chat_engine import chat_response

app = Flask(__name__)

# 首页
@app.route('/')
def index():
    return render_template('index.html')

# 视频生成界面
@app.route('/video_generation', methods=['GET', 'POST'])
def video_generation():
    if request.method == 'POST':
        data = {
            "model_name": request.form.get('model_name'),
            "id": request.form.get('id'),
            "ref_audio": request.form.get('ref_audio'),
            "gpu_choice": request.form.get('gpu_choice'),
            "target_text": request.form.get('target_text'),
            "test_size": request.form.get('test_size'),
        }

        video_path = generate_video(data)
        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('video_generation.html')


# 模型训练界面
@app.route('/model_training', methods=['GET', 'POST'])
def model_training():
    if request.method == 'POST':
        data = {
            "model_choice": request.form.get('model_choice'),
            "id": request.form.get('id'),
            "gpu_choice": request.form.get('gpu_choice'),
            "epoch": request.form.get('epoch'),
            "custom_params": request.form.get('custom_params'),
            "preprocess": request.form.get('preprocess'),
        }

        video_path = train_model(data)
        video_path = "/" + video_path.replace("\\", "/")

        return jsonify({'status': 'success', 'video_path': video_path})

    return render_template('model_training.html')


# 实时对话系统界面
@app.route('/chat_system', methods=['GET', 'POST'])
def chat_system():
    if request.method == 'POST':
        
        text = (request.form.get('text') or '').strip()
        # 把前端文本写入 input.txt
        if text:
            os.makedirs('./static/text', exist_ok=True)
            with open('./static/text/input.txt', 'w', encoding='utf-8') as f:
                f.write(text)

        data = {
            "model_name": request.form.get('model_name'),
            "model_param": request.form.get('model_param'),
            "voice_clone": request.form.get('voice_clone'),
            "api_choice": request.form.get('api_choice'),
            "text": text,  # 也直接传给 chat_engine（更稳）
        }

        # video_path = chat_response(data)
        try:
            result = chat_response(data)  # 我们下面会让它返回 dict
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500

        # result 现在建议是 dict: {video_path, audio_path}
        video_path = result.get("video_path", "")
        audio_path = result.get("audio_path", "")

        if video_path:
            video_path = video_path.replace("\\", "/")
            if not audio_path.startswith("/"):
                video_path = "/" + video_path
        
        if audio_path:
            audio_path = audio_path.replace("\\", "/")
            # 避免变成 //static/... （浏览器会当成 http(s)://static/...）
            if not audio_path.startswith("/"):
                audio_path = "/" + audio_path

        return jsonify({
            'status': 'success',
            'video_path': video_path,
            'audio_path': audio_path,
        })

    return render_template('chat_system.html')

@app.route('/save_audio', methods=['POST'])
def save_audio():
    if 'audio' not in request.files:
        return jsonify({'status': 'error', 'message': '没有音频文件'})
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return jsonify({'status': 'error', 'message': '没有选择文件'})
    
    # 确保目录存在
    os.makedirs('./static/audios', exist_ok=True)
    
    # 保存文件
    audio_file.save('./static/audios/input.wav')
    
    return jsonify({'status': 'success', 'message': '音频保存成功'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True)
