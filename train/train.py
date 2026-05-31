

import os, sys, json, glob, argparse, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from io import BytesIO
import lmdb
import clip

sys.path.insert(0, '.')
from networks.generator import Generator
from train.networks_exp.clip_emotion_model_v2_intensity_v3 import (
    CLIPEmotionExpModelV2Intensity as CLIPEmotionExpModelV2, CLIPVerificationLoss, EMOTION_TO_IDX
)


def format_for_lmdb(*args):
    key_parts = []
    for arg in args:
        if isinstance(arg, int):
            arg = str(arg).zfill(7)
        key_parts.append(arg)
    return '-'.join(key_parts).encode('utf-8')


class MEADDataset(Dataset):
    def __init__(self, lmdb_path, list_file, prompt_dir, resolution=256):
        self.env = lmdb.open(lmdb_path, max_readers=32, readonly=True,
                             lock=False, readahead=False, meminit=False)
        
        with open(list_file) as f:
            videos = json.load(f)
        
        self.mead_videos = [v for v in videos if len(v.split('#')) == 4]
        self.prompt_dir = prompt_dir
        self.resolution = resolution
        
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        # Cache video lengths
        self.video_lengths = {}
        with self.env.begin(write=False) as txn:
            for v in tqdm(self.mead_videos, desc='Caching lengths', leave=False):
                key = format_for_lmdb(v, 'length')
                val = txn.get(key)
                if val is not None:
                    length = int(val.decode('utf-8'))
                    if length > 0:
                        self.video_lengths[v] = length
        
        self.mead_videos = [v for v in self.mead_videos if v in self.video_lengths]
        
        # Group by actor for source sampling
        self.actor_neutral = {}
        for v in self.mead_videos:
            actor, emotion = v.split('#')[0], v.split('#')[1]
            if emotion == 'neutral':
                if actor not in self.actor_neutral:
                    self.actor_neutral[actor] = []
                self.actor_neutral[actor].append(v)
        
        print(f"V2 Dataset: {len(self.mead_videos)} videos")
    
    def __len__(self):
        return len(self.mead_videos)
    
    def __getitem__(self, idx):
        video = self.mead_videos[idx]
        actor, emotion, level, clip_id = video.split('#')
        num_frames = self.video_lengths[video]
        
        # Random target frame
        tgt_frame_idx = random.randint(0, num_frames - 1)
        
        # Source: random neutral frame of same actor
        neutral_vids = self.actor_neutral.get(actor, [])
        if neutral_vids:
            src_video = random.choice(neutral_vids)
            src_len = self.video_lengths.get(src_video, 1)
            src_frame_idx = random.randint(0, src_len - 1)
        else:
            src_video = video
            src_frame_idx = 0
        
        with self.env.begin(write=False) as txn:
            src_key = format_for_lmdb(src_video, src_frame_idx)
            tgt_key = format_for_lmdb(video, tgt_frame_idx)
            src_bytes = txn.get(src_key)
            tgt_bytes = txn.get(tgt_key)
        
        src_img = self.transform(Image.open(BytesIO(src_bytes)))
        tgt_img = self.transform(Image.open(BytesIO(tgt_bytes)))
        
        # Prompt
        prompt_path = os.path.join(self.prompt_dir, video + '.txt')
        if os.path.exists(prompt_path):
            with open(prompt_path) as f:
                prompt = f.read().strip()
        else:
            intensity_map = {'level_1': 'low', 'level_2': 'moderate', 'level_3': 'high'}
            intensity = intensity_map.get(level, 'normal')
            if emotion == 'neutral':
                prompt = "The person is speaking with a neutral expression."
            else:
                prompt = f"The person is expressing {emotion} with {intensity} intensity."
        
        emotion_idx = EMOTION_TO_IDX.get(emotion, 5)
        intensity_target = {'level_1': 0.6, 'level_2': 0.8, 'level_3': 1.0}.get(level, 0.8)
        
        return {
            'source_image': src_img,
            'target_image': tgt_img,
            'prompt': prompt,
            'emotion_idx': emotion_idx,
            'video_name': video,
            'intensity_target': intensity_target,
        }


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision import models
        vgg = models.vgg19(pretrained=True).features[:16]
        self.vgg = vgg.eval()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def forward(self, pred, target):
        pred = (pred + 1) / 2
        target = (target + 1) / 2
        pred = (pred - self.mean.to(pred.device)) / self.std.to(pred.device)
        target = (target - self.mean.to(target.device)) / self.std.to(target.device)
        return F.l1_loss(self.vgg(pred), self.vgg(target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmdb_path', default='EDTalk_lmdb')
    parser.add_argument('--train_list', default='lists/MEAD_HDTF_train_heldout.json')
    parser.add_argument('--prompt_dir', default='MEAD_front/prompts')
    parser.add_argument('--edtalk_ckpt', default='ckpts/EDTalk.pt')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--save_dir', default='ckpts/emotion_train_v2_intensity_v3')
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5)
    # Loss weights
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_percep', type=float, default=0.5)
    parser.add_argument('--lambda_distib', type=float, default=0.1,
                        help='Weight for DisTIB (compression + prediction)')
    parser.add_argument('--lambda_clip_verify', type=float, default=0.1,
                        help='Weight for CLIP verification loss')
    parser.add_argument('--distib_beta', type=float, default=0.01,
                        help='Beta for DisTIB compression-prediction tradeoff')
    parser.add_argument('--lambda_regression', type=float, default=0.5,
                        help='Weight for regression to optimized predefined weights')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device('cuda')
    
    # --- Load frozen EDTalk generator ---
    print("Loading EDTalk generator...")
    gen = Generator(256, style_dim=512, lip_dim=20, pose_dim=6, exp_dim=10,
                    channel_multiplier=1).to(device)
    ckpt = torch.load(args.edtalk_ckpt, map_location=device, weights_only=False)
    gen.load_state_dict(ckpt['gen'])
    gen.eval()
    for p in gen.parameters():
        p.requires_grad = False
    
    # --- Load frozen CLIP ---
    print("Loading CLIP...")
    clip_model, _ = clip.load('ViT-B/32', device=device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False
    
    # --- Our trainable V2 model ---
    print("Initializing V2 emotion model...")
    emotion_model = CLIPEmotionExpModelV2(
        clip_dim=512, exp_dim=10, num_emotions=8, num_heads=4
    ).to(device)
    emotion_model.distib_loss.beta = args.distib_beta
    
    # Load optimized predefined weights as regression targets
    EMOTIONS_LIST = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'neutral', 'sad', 'surprised']
    optimized_weights = {}
    for emo in EMOTIONS_LIST:
        for suffix in ['_allactor18', '_optimized', '']:
            path = f'ckpts/predefined_exp_weights/{emo}{suffix}.npy'
            if os.path.exists(path):
                w = np.load(path).flatten()
                break
        optimized_weights[emo] = torch.tensor(w, dtype=torch.float32, device=device)
    opt_weight_tensor = torch.stack([optimized_weights[e] for e in EMOTIONS_LIST]).to(device)  # [8, 10]
    print(f"Loaded optimized target weights for {len(optimized_weights)} emotions")
    
    # CLIP Verification Loss
    clip_verify = CLIPVerificationLoss(clip_model).to(device)
    
    # VGG Perceptual Loss
    vgg_loss = VGGPerceptualLoss().to(device)
    
    total_params = sum(p.numel() for p in emotion_model.parameters())
    print(f"V2 trainable params: {total_params:,}")
    
    # --- Optimizer ---
    optimizer = torch.optim.AdamW(emotion_model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # --- Dataset ---
    print("Loading dataset...")
    dataset = MEADDataset(
        lmdb_path=args.lmdb_path,
        list_file=args.train_list,
        prompt_dir=args.prompt_dir,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)
    
    print(f"Training: {len(dataset)} samples, {len(loader)} batches/epoch")
    print(f"Losses: recon({args.lambda_recon}) + percep({args.lambda_percep}) + "
          f"distib({args.lambda_distib}, beta={args.distib_beta}) + "
          f"clip_verify({args.lambda_clip_verify})")
    print(f"Starting V2+RegOpt training for {args.epochs} epochs...")
    
    # --- Training ---
    for epoch in range(args.epochs):
        emotion_model.train()
        
        totals = {'loss': 0, 'recon': 0, 'percep': 0, 'distib': 0,
                  'clip_v': 0, 'compress': 0, 'predict': 0, 'attn': 0, 'regr': 0, 'intens': 0}
        
        for batch_idx, batch in enumerate(tqdm(loader, desc=f'Epoch {epoch}', leave=False)):
            src_img = batch['source_image'].to(device)
            tgt_img = batch['target_image'].to(device)
            prompts = batch['prompt']
            emotion_idx = batch['emotion_idx'].to(device)
            intensity_target = batch['intensity_target'].float().to(device).unsqueeze(1)  # [B, 1]
            
            # --- CLIP encode text ---
            tokens = clip.tokenize(prompts, truncate=True).to(device)
            with torch.no_grad():
                text_feat = clip_model.encode_text(tokens).float()
            
            # --- V2 forward (with details for DisTIB) ---
            exp_weights, details = emotion_model(text_feat, return_details=True)
            
            # --- EDTalk generation (frozen) ---
            with torch.no_grad():
                wa, wa_t, feats, _ = gen.enc(src_img, tgt_img)
                shared_fc = gen.fc(wa_t)
                alpha_lip = gen.lip_fc(shared_fc)
                alpha_pose = gen.pose_fc(shared_fc)
                
                # Get lip latent for DisTIB (this is the Z — sample-exclusive)
                lip_latent = shared_fc  # [B, 512]
            
            # Combine with our V2 exp weights
            alpha_D = torch.cat([alpha_lip, alpha_pose, exp_weights], dim=-1)
            
            a = gen.direction_exp.get_shared_out(alpha_D, gen.direction_lipnonlip.weight)
            exp_latent = gen.direction_exp.get_exp_latent(a)
            directions = gen.direction_exp(alpha_D, gen.direction_lipnonlip.weight)
            latent = wa + directions
            img_recon = gen.dec(latent, feats, exp_latent)
            
            # --- Losses ---
            # 1. Reconstruction
            loss_recon = F.l1_loss(img_recon, tgt_img)
            
            # 2. Perceptual
            loss_percep = vgg_loss(img_recon, tgt_img)
            
            # 3. DisTIB (replaces MI loss)
            # Need lip encoder stats — approximate from lip_latent
            lip_mean = lip_latent.detach()
            lip_logvar = torch.zeros_like(lip_mean)  # frozen encoder, treat as deterministic
            
            loss_distib, distib_dict = emotion_model.compute_distib_loss(
                details['exp_mean'], details['exp_logvar'],
                lip_mean, lip_logvar,
                details['cls_logits'], emotion_idx,
            )
            
            # 4. CLIP Verification Loss
            loss_clip_v = clip_verify(img_recon, text_feat)
            
            # 5. Attention supervision — force correct emotion node
            loss_attn = emotion_model.compute_attn_loss(
                details["attn_weights"], emotion_idx)
            
            # 6. Regression to optimized predefined weights
            target_w = opt_weight_tensor[emotion_idx]  # [B, 10]
            loss_regression = F.mse_loss(exp_weights, target_w)

            # 7. Intensity loss — teach IntensityHead to predict level
            pred_intensity = details['intensity']  # [B, 1]
            loss_intensity = F.mse_loss(pred_intensity, intensity_target)
            
            # --- Total ---
            loss = (args.lambda_recon * loss_recon +
                    args.lambda_percep * loss_percep +
                    args.lambda_distib * loss_distib +
                    args.lambda_clip_verify * loss_clip_v +
                    1.0 * loss_attn +
                    args.lambda_regression * loss_regression +
                    0.5 * loss_intensity)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(emotion_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track
            totals['loss'] += loss.item()
            totals['recon'] += loss_recon.item()
            totals['percep'] += loss_percep.item()
            totals['distib'] += loss_distib.item()
            totals['clip_v'] += loss_clip_v.item()
            totals['attn'] += loss_attn.item()
            totals['compress'] += distib_dict['compress_exp']
            totals['predict'] += distib_dict['prediction']
            totals['regr'] += loss_regression.item()
            totals['intens'] = totals.get('intens', 0) + loss_intensity.item()
            
            if (batch_idx + 1) % args.log_interval == 0:
                n = batch_idx + 1
                print(f"  [{n}/{len(loader)}] loss={totals['loss']/n:.4f} "
                      f"recon={totals['recon']/n:.4f} percep={totals['percep']/n:.4f} "
                      f"distib={totals['distib']/n:.4f} clip_v={totals['clip_v']/n:.4f} "
                      f"compress={totals['compress']/n:.4f} predict={totals['predict']/n:.4f} "
              f"attn={totals['attn']/n:.4f} "
                      f"attn={totals['attn']/n:.4f}")
        
        scheduler.step()
        n = max(len(loader), 1)
        print(f"Epoch {epoch:03d} | loss={totals['loss']/n:.4f} "
              f"recon={totals['recon']/n:.4f} percep={totals['percep']/n:.4f} "
              f"distib={totals['distib']/n:.4f} clip_v={totals['clip_v']/n:.4f} "
              f"compress={totals['compress']/n:.4f} predict={totals['predict']/n:.4f} "
              f"attn={totals['attn']/n:.4f} "
              f"regr={totals['regr']/n:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.6f}")
        
        # Save
        if (epoch + 1) % args.save_interval == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.save_dir, f'emotion_v2_epoch_{epoch:03d}.pt')
            torch.save({
                'epoch': epoch,
                'emotion_model': emotion_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': vars(args),
            }, save_path)
            print(f"Saved: {save_path}")
    
    print("V2 training complete!")


if __name__ == '__main__':
    main()
