"""
Inference with trained CLIP Emotion Expression Model.
Follows EDTalk's demo_EDTalk_A_using_predefined_exp_weights.py EXACTLY,
but replaces predefined exp weights with CLIP-driven emotion model output.
"""

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from moviepy.editor import *
import cv2
import clip

sys.path.insert(0, '.')
from networks.generator import Generator
from networks.audio_encoder import Audio2Lip
from train.networks_exp.clip_emotion_model_v2_intensity_v3 import CLIPEmotionExpModelV2Intensity as CLIPEmotionExpModelV2
import audio as audio_module


def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps
    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)
    return audio_length, num_frames


def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav


def get_mel(audio_path):
    """Returns mel features in EDTalk format: (bs*T, 1, 80, 16), bs, T"""
    wav = audio_module.load_wav(audio_path, 16000)
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio_module.melspectrogram(wav).T
    
    indiv_mels = []
    for i in range(num_frames):
        start_frame_num = i - 2
        start_idx = int(80. * (start_frame_num / float(25)))
        end_idx = start_idx + 16
        seq = list(range(start_idx, end_idx))
        seq = [min(max(item, 0), orig_mel.shape[0] - 1) for item in seq]
        m = orig_mel[seq, :]
        indiv_mels.append(m.T)
    
    indiv_mels = np.asarray(indiv_mels)  # (T, 80, 16)
    # EDTalk format: (1, T, 1, 80, 16) -> view as (T, 1, 80, 16)
    source_audio_feature = torch.FloatTensor(indiv_mels).unsqueeze(0).unsqueeze(2)  # (1, T, 1, 80, 16)
    mel_input = source_audio_feature
    bs = mel_input.shape[0]
    T = mel_input.shape[1]
    audiox = mel_input.view(-1, 1, 80, 16)  # (bs*T, 1, 80, 16)
    return audiox, bs, T


def vid_preprocessing(vid_path):
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    vid = torch.FloatTensor(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)
    vid_norm = (vid / 255.0 - 0.5) * 2.0
    transform = transforms.Compose([transforms.Resize((256, 256))])
    resized = torch.stack([transform(f) for f in vid_norm[0]], dim=0).unsqueeze(0)
    return resized, fps


def conv_feat(features, k_size=3, sigma=1):
    """Smooth lip features — same as EDTalk demo"""
    import scipy.ndimage
    features_np = features.cpu().numpy()
    smoothed = scipy.ndimage.gaussian_filter1d(features_np, sigma=sigma, axis=0)
    return torch.from_numpy(smoothed).to(features.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_path', required=True)
    parser.add_argument('--audio_driving_path', required=True)
    parser.add_argument('--pose_driving_path', required=True)
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--emotion_ckpt', default='ckpts/emotion_train_v2_intensity_v3/emotion_v2_epoch_029.pt')
    parser.add_argument('--edtalk_ckpt', default='ckpts/EDTalk.pt')
    parser.add_argument('--audio2lip_ckpt', default='ckpts/Audio2Lip.pt')
    parser.add_argument('--save_path', default='res/v2_output.mp4')
    args = parser.parse_args()
    
    device = torch.device('cuda')
    
    # === Load models — EXACTLY as EDTalk demo ===
    print("Loading models...")
    
    # Audio2Lip
    audio2lip = Audio2Lip().to(device)
    ckpt_a2l = torch.load(args.audio2lip_ckpt, map_location=device, weights_only=False)
    audio2lip.load_state_dict(ckpt_a2l['audio2lip'])
    audio2lip.eval()
    
    # Generator
    gen = Generator(256, style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10, channel_multiplier=1).to(device)
    ckpt = torch.load(args.edtalk_ckpt, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt['gen'])
    gen.eval()
    
    # Our emotion model
    emotion_model = CLIPEmotionExpModelV2(clip_dim=512, exp_dim=10).to(device)
    if os.path.exists(args.emotion_ckpt):
        emo_ckpt = torch.load(args.emotion_ckpt, map_location=device, weights_only=False)
        emotion_model.load_state_dict(emo_ckpt['emotion_model'], strict=False)
        print(f"Loaded emotion checkpoint: {args.emotion_ckpt}")
    emotion_model.eval()
    
    # CLIP
    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()
    
    # === Get emotion exp weights from text prompt ===
    tokens = clip.tokenize([args.prompt]).to(device)
    with torch.no_grad():
        text_feat = clip_model.encode_text(tokens).float()
        alpha_D_exp, details = emotion_model(text_feat, return_details=True)
        # Intensity-aware scaling
        raw_intensity = details['intensity'].item()
        # Remap learned range [0.72, 0.91] → amplified range [0.3, 1.0]
        amplified = 0.6 + (raw_intensity - 0.69) / (0.93 - 0.69) * (1.2 - 0.6)
        amplified = max(0.4, min(1.4, amplified))  # clamp for safety
        # Undo model's internal scaling, apply amplified scale
        alpha_D_exp = alpha_D_exp * amplified
        print(f"  Intensity: raw={raw_intensity:.3f} amplified={amplified:.3f}")  # scale to match EDTalk predefined weight magnitude
        attn_weights = details['attn_weights']
        intensity = 1.0  # V2 handles intensity implicitly via dynamic graph
    
    emotions = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    print(f"Prompt: '{args.prompt}'")
    print(f"Exp weights: {alpha_D_exp.cpu().numpy().round(3)}")
    print(f"Emotion attention: {dict(zip(emotions, attn_weights[0].cpu().numpy().round(3)))}")
    print(f"Intensity: {intensity:.3f}")
    
    # === Load source image — same as EDTalk demo ===
    img_source = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])(Image.open(args.source_path).convert('RGB')).unsqueeze(0).to(device)
    
    # === Load audio — EXACTLY as EDTalk demo ===
    audio_data, bs, T = get_mel(args.audio_driving_path)
    audio_data = audio_data.to(device)
    
    # === Load pose video — same as EDTalk demo ===
    pose_vid, fps = vid_preprocessing(args.pose_driving_path)
    pose_vid = pose_vid.to(device)
    
    # === Generate — following EDTalk demo exactly ===
    print(f"Generating {T} frames...")
    with torch.no_grad():
        # Get lip weights from audio — same as EDTalk
        lip_vid_target = audio2lip(audio_data, bs, T)[0]  # [T, 20]
        lip_vid_target = conv_feat(lip_vid_target, k_size=3, sigma=1)
        
        vid_recon = []
        h_start = None
        
        for i in tqdm(range(lip_vid_target.size(0))):
            img_target_lip = lip_vid_target[i:i+1]  # [1, 20]
            
            # Pose from video (cycle if needed)
            pose_idx = i % pose_vid.shape[1]
            img_target_pose = pose_vid[:, pose_idx]  # [1, 3, 256, 256]
            
            # Generate with our emotion exp weights
            img_recon = gen.test_EDTalk_A_use_exp_weight(
                img_source, img_target_lip, img_target_pose, alpha_D_exp, h_start
            )
            
            vid_recon.append(img_recon.unsqueeze(2))
            
            if i == 0:
                h_start = img_recon
        
        vid_recon = torch.cat(vid_recon, dim=2)  # [1, 3, T, 256, 256]
    
    # === Save video — same as EDTalk demo ===
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    temp_path = args.save_path.replace('.mp4', '_temp.mp4')
    
    vid_np = vid_recon[0].permute(1, 2, 3, 0).cpu().numpy()  # [T, 256, 256, 3]
    vid_np = ((vid_np + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    
    writer = cv2.VideoWriter(temp_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (256, 256))
    for frame in vid_np:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    
    # Add audio
    cmd = f'ffmpeg -y -i "{temp_path}" -i "{args.audio_driving_path}" -vcodec copy "{args.save_path}" -loglevel quiet'
    os.system(cmd)
    if os.path.exists(temp_path):
        os.remove(temp_path)
    
    print(f"Saved: {args.save_path}")


if __name__ == '__main__':
    main()
