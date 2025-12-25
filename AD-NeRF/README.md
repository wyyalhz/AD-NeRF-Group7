# AD-NeRF: Audio Driven Neural Radiance Fields for Talking Head Synthesis

![](paper_data/pipeline.png)

模型在 [AD-NeRF](https://github.com/YudongGuo/AD-NeRF) 的基础上做出复现并进行了少许修改。

仓库结构：

```text
AD-NeRF/
├── add_audio/            # 合成音视频
├── data_util/             # 人脸跟踪与数据预处理
├── dataset/               # 输入与输出数据
│   ├── vids/              # 输入视频
│   └── <id>/              # 单人物的预处理/训练/渲染产物
├── evaluation/            # 评估脚本
├── example_outputs/       # 示例输出
├── NeRFs/                 # Head/Torso NeRF 训练与推理
├── paper_data/            # README 展示用图片
├── pretrained_models/     # 预训练模型
└── private_docs/          # 个人复现记录
```

## Prerequisites
- 环境搭建

    由于复现时间距项目初次发布时间较远，原项目中的依赖版本不能很好地适配新的硬件，而部分代码又无法支持最新版本的依赖，因此需要创建3个conda环境分别处理数据预处理和训练。
    
    针对数据预处理，我们需要两个环境conda_ds和conda_pre。原项目中将数据预处理的8个step合并为1条指令执行，但是由于step 0用到基于tensorflow 1.15的deepspeech，而其余步骤要用到pytorch3d，所以只能分开执行

    ```python
    # 用于数据预处理step 0
    conda env create -n adnerf_ds python=3.7 -y
    conda activate adnerf
    pip install "tensorflow==1.15.0" "numpy==1.19.5"
    pip install "protobuf==3.20.3" --force-reinstall #降级protobuf
    pip install opencv-python face_alignment scikit-learn resampy pandas python_speech_features tensorflow natsort configargparse

    # 用于数据预处理step 1-8和结果评估
    conda create -n adnerf_pre python=3.10 -y
    conda activate adnerf_pre
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124
    pip install iopath fvcore pybind11
    
    pip install scipy opencv-python face_alignment scikit-learn resampy pandas python_speech_features tensorflow natsort configargparse

    # 如果你不是50系显卡，可以选择与上面一致的cu124版本，否则选cu128版本
    conda create -n adnerf python=3.10 -y
    conda activate adnerf
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install numpy scipy tqdm matplotlib opencv-python imageio imageio-ffmpeg configargparse tensorboardX natsort face-alignment
    ```
- [PyTorch3D](https://github.com/facebookresearch/pytorch3d)

    推荐通过本地克隆安装
    ```
    conda activate adnerf_pre
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d && pip install -e . --no-build-isolation #关闭隔离
    ```
- [Basel Face Model 2009](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-1-0&id=details) 

    将 "01_MorphableModel.mat" 放到 data_util/face_tracking/3DMM/; cd data_util/face_tracking; run

    ```
    python convert_BFM.py
    ```

## Train AD-NeRF
- Data Preprocess ($id Obama for example)
    ```
    bash process_data.sh Obama
    ```
    - Input: A portrait video at 25fps containing voice audio. (dataset/vids/$id.mp4)
    - Output: folder dataset/$id that contains all files for training

- Train Two NeRFs (Head-NeRF and Torso-NeRF)
    - Train Head-NeRF with command 
        ```
        python NeRFs/HeadNeRF/run_nerf.py --config dataset/$id/HeadNeRF_config.txt
        ```
    - Copy latest trainied model from dataset/$id/logs/$id_head to dataset/$id/logs/$id_com
    - Train Torso-NeRF with command 
        ```
        python NeRFs/TorsoNeRF/run_nerf.py --config dataset/$id/TorsoNeRF_config.txt
        ```
    - 如果需要使用预训练模型，可以将pretrained_models中的.txt和.json放入dataset/$id文件夹下，然后将.tar放入dataset/$id/logs/&id_com（如果要单独训练头部就把*head.tar放入dataset/$id/logs/&id_head）
## Run AD-NeRF for rendering
- Reconstruct original video with audio input
    ```
    python NeRFs/TorsoNeRF/run_nerf.py --config dataset/$id/TorsoNeRFTest_config.txt --aud_file=dataset/$id/aud.npy --test_size=300
    ```
- Drive the target person with another audio input
    ```
    python NeRFs/TorsoNeRF/run_nerf.py --config dataset/$id/TorsoNeRFTest_config.txt --aud_file=${deepspeechfile.npy} --test_size=-1
    ```

## 用ffmpeg合成视频和音频

将渲染所得视频（在dataset/$id/logs/$id_com/test_aud_rst中）和驱动音频拷贝到add_audio，然后执行

```bash
cd add_audio
python add_audio.py --video "adnerf渲染出的视频".avi --audio "音频".wav --output "你自己取个名字".mp4 --vcodec libx264 --crf 18
```

## 对训练结果进行评估

- 环境准备

```bash
conda activate adnerf_pre
pip install pytorch-fid face-alignment matplotlib
```

- 在AD-NeRF目录下运行评估
```bash
# 推荐：评估所有指标
python evaluation/evaluate.py --subject Obama --metrics all

# 仅评估图像质量
python evaluation/evaluate.py --subject Obama --metrics psnr ssim fid
```

- 查看结果

结果保存在 `evaluation/results/Obama/evaluation_report.txt`

可视化图表在同目录下 `metrics_plot.png`

## Acknowledgments
We use [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) for parsing head and torso maps, and [DeepSpeech](https://github.com/mozilla/DeepSpeech) for audio feature extraction. The NeRF model is implemented based on [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch).

## 致谢

感谢7组同学的付出与协作，感谢助教老师提供资源与指导❤
