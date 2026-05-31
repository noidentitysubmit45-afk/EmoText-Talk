# EmoText-Talk

Text-driven emotion and intensity control for talking head generation.

## Results
- 91.67% Acc_emo on MEAD (EAT's Emotion-FAN checkpoint)
- +8.1 points over EmoCAST (previous text-driven SOTA)

## Setup
1. Clone EDTalk: `git clone https://github.com/tanshuai0219/EDTalk`
2. Install deps: `pip install -r requirements.txt`
3. Download checkpoints: https://drive.google.com/file/d/1yx7boF8uCuQgaa4BEX1CuTu7lGCfVMt_/view?usp=sharing
4. Download MEAD: https://drive.google.com/drive/u/0/folders/1GwXP-KpWOxOenOxITTsURJZQ_1pkd4-j

## Inference
python demo/demo.py \
    --source_path source.jpg \
    --audio_driving_path audio.wav \
    --pose_driving_path pose.mp4 \
    --prompt "The person is expressing happiness." \
    --save_path output.mp4

## Training
python train/train.py
