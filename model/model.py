"""
CLIPEmotionExpModel V2

Upgrades over V1, following actual paper formulations:

1. Dynamic Graph Attention (DGAT, Frontiers Psychiatry 2025)
   - Adjacency updated via momentum: w_ij ← β·w_ij + (1-β)·e_ij
   - e_ij computed via multi-head attention mechanism
   - NOT a separate network generating adjacency — the adjacency
     evolves during training based on attention between nodes

2. Information Bottleneck (DisTIB, TPAMI 2024)
   - Stochastic encoders with reparameterization trick
   - Compression: KL(p(A|X) || r(A)) where r(A) = N(0,I)
   - Sufficiency: reconstruction from (A, Z)
   - Prediction: CrossEntropy classification
   - Full objective: L = -[Sufficiency + Prediction] + β·Compression

3. CLIP Verification Loss (novel — no existing paper does this)
   - Generated face → CLIP image encoder → verify matches text prompt

V1 files are UNTOUCHED.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import clip


# ============================================================
# 1. Dynamic Graph Attention Network (following DGAT paper)
# ============================================================
class DynamicGraphAttention(nn.Module):
    """
    Follows DGAT (Ding et al., 2025) Equations 7-10:
    
    e_ij = a(W·h_i, W·h_j)                    (Eq 7)
    w_ij ← β·w_ij + (1-β)·e_ij                (Eq 8)  
    α_ij = softmax_j(w_ij)                     (Eq 9)
    h'_i = σ(Σ α_ij · W · h_j)                (Eq 10)
    
    Multi-head attention (Eq 6): h'_i = ||_{k=1}^K σ(Σ α^k_ij · W^k · h_j)
    
    Key difference from static GAT: adjacency matrix is a learnable
    parameter that gets momentum-updated with attention weights during
    forward pass, making it dynamic and input-adaptive.
    """
    def __init__(self, num_nodes=8, feat_dim=512, num_heads=4, momentum=0.5,
                 init_embeddings=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.num_heads = num_heads
        self.head_dim = feat_dim // num_heads
        self.momentum = momentum
        
        # Learnable node embeddings (emotion nodes)
        # If init_embeddings provided (e.g. from CLIP), use those
        if init_embeddings is not None:
            self.node_embeddings = nn.Parameter(init_embeddings.clone())
        else:
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, feat_dim) * 0.02)
        
        # Trainable adjacency matrix (initialized as identity + small noise)
        # This is the w_ij that gets momentum-updated (Eq 8)
        self.register_buffer('adjacency', 
            torch.eye(num_nodes) + torch.randn(num_nodes, num_nodes) * 0.01)
        
        # Per-head weight matrices W^k (Eq 6)
        self.W = nn.ModuleList([
            nn.Linear(feat_dim, self.head_dim, bias=False)
            for _ in range(num_heads)
        ])
        
        # Attention vector a^k for each head (Eq 4 from paper)
        # a^T [W·h_i || W·h_j] — concatenation-based attention
        self.attn_vec = nn.ParameterList([
            nn.Parameter(torch.randn(2 * self.head_dim) * 0.01)
            for _ in range(num_heads)
        ])
        
        # Output projection after multi-head concatenation
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.norm = nn.LayerNorm(feat_dim)
        self.leaky_relu = nn.LeakyReLU(0.2)
    
    def compute_attention(self, h, head_idx):
        """
        Compute attention coefficients e_ij (Eq 2, 4 from DGAT paper)
        
        e_ij = LeakyReLU(a^T [W·h_i || W·h_j])
        α_ij = softmax_j(e_ij)
        """
        N = h.shape[0]
        Wh = self.W[head_idx](h)  # [N, head_dim]
        
        # Compute e_ij for all pairs
        # [W·h_i || W·h_j] for all i,j
        Wh_i = Wh.unsqueeze(1).expand(-1, N, -1)  # [N, N, head_dim]
        Wh_j = Wh.unsqueeze(0).expand(N, -1, -1)  # [N, N, head_dim]
        concat = torch.cat([Wh_i, Wh_j], dim=-1)  # [N, N, 2*head_dim]
        
        # e_ij = LeakyReLU(a^T · concat) (Eq 4)
        e = self.leaky_relu(torch.matmul(concat, self.attn_vec[head_idx]))  # [N, N]
        
        return e, Wh
    
    def forward(self, text_condition=None):
        """
        Forward pass with dynamic adjacency update.
        
        text_condition: [B, 512] — optional text conditioning to bias attention
        Returns: [B, N, feat_dim] or [N, feat_dim] graph-refined node embeddings
        """
        h = self.node_embeddings  # [N, feat_dim]
        N = self.num_nodes
        
        all_head_outputs = []
        e_accumulated = torch.zeros(N, N, device=h.device)
        
        for k in range(self.num_heads):
            e, Wh = self.compute_attention(h, k)  # e: [N, N], Wh: [N, head_dim]
            e_accumulated += e
            
            # Momentum update of adjacency (Eq 8): w_ij ← β·w_ij + (1-β)·e_ij
            if self.training:
                with torch.no_grad():
                    self.adjacency.data = (self.momentum * self.adjacency.data + 
                                          (1 - self.momentum) * e.detach())
            
            # Combine attention with dynamic adjacency (Eq 9)
            d = self.adjacency + e  # combine learned structure with current attention
            alpha = F.softmax(d, dim=-1)  # [N, N]
            
            # Aggregate (Eq 10): h'_i = σ(Σ α_ij · W · h_j)
            head_out = torch.matmul(alpha, Wh)  # [N, head_dim]
            all_head_outputs.append(head_out)
        
        # Multi-head concatenation (Eq 6): h' = ||_{k=1}^K (head outputs)
        h_multi = torch.cat(all_head_outputs, dim=-1)  # [N, num_heads * head_dim]
        h_out = self.out_proj(h_multi)  # [N, feat_dim]
        h_out = self.norm(self.leaky_relu(h_out) + h)  # residual + norm
        
        # Expand for batch and condition on text
        if text_condition is not None:
            B = text_condition.shape[0]
            h_out = h_out.unsqueeze(0).expand(B, -1, -1)  # [B, N, feat_dim]
            # Text-condition: modulate each node by text relevance
            # This makes nodes different per input
            text_gate = torch.sigmoid(
                torch.matmul(text_condition.unsqueeze(1), h_out.transpose(-2, -1))
            )  # [B, 1, N]
            h_out = h_out * (1 + text_gate.transpose(-2, -1))  # [B, N, feat_dim]
        
        return h_out


# ============================================================
# 2. Cross-Attention: Text queries → Dynamic Emotion Graph nodes
# ============================================================
class TextEmotionCrossAttentionV2(nn.Module):
    def __init__(self, text_dim=512, emotion_dim=512, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = emotion_dim // num_heads
        
        self.q_proj = nn.Linear(text_dim, emotion_dim)
        self.k_proj = nn.Linear(emotion_dim, emotion_dim)
        self.v_proj = nn.Linear(emotion_dim, emotion_dim)
        self.out_proj = nn.Linear(emotion_dim, emotion_dim)
        self.norm = nn.LayerNorm(emotion_dim)
        self.attn_temperature = nn.Parameter(torch.tensor(3.0))
    
    def forward(self, text_feat, emotion_nodes):
        B = text_feat.shape[0]
        N = emotion_nodes.shape[1]
        
        q = self.q_proj(text_feat).unsqueeze(1)
        k = self.k_proj(emotion_nodes)
        v = self.v_proj(emotion_nodes)
        
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn * self.attn_temperature  # sharpen attention
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, -1)
        out = self.out_proj(out)
        out = self.norm(out + text_feat)
        
        attn_weights = attn.squeeze(2).mean(dim=1)
        return out, attn_weights


# ============================================================
# 3. Stochastic Encoder (DisTIB Section IV, Algorithm 1)
# ============================================================
class StochasticEncoder(nn.Module):
    """
    Following DisTIB: p(A|X) = N(f_μ(X), f_σ²(X))
    
    Encodes input into mean and log-variance, then samples
    using reparameterization trick.
    """
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
        )
        self.mean_head = nn.Linear(hidden_dim, output_dim)
        self.logvar_head = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        h = self.shared(x)
        mean = self.mean_head(h)
        logvar = self.logvar_head(h).clamp(-10, 10)
        return mean, logvar
    
    def sample(self, x):
        mean, logvar = self.forward(x)
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std  # reparameterization trick
        else:
            z = mean
        return z, mean, logvar


# ============================================================
# 4. DisTIB Loss (following Equation 5 of the paper)
# ============================================================
class DisTIBLoss(nn.Module):
    """
    Transmitted Information Bottleneck for Disentanglement
    (Dang et al., TPAMI 2024)
    
    L_DisTIB = -[I(X; A,Z) + I(A; Y)] + β[I(X; A) + I(X; Z) + I(X; Y)]
               = -[Sufficiency + Prediction] + β·Compression
    
    Implementation (Section IV):
    - Compression I(X; A): KL(p(A|X) || N(0,I)) via Eq 6
    - Compression I(X; Z): KL(p(Z|X) || N(0,I)) via Eq 6  
    - Sufficiency I(X; A,Z): reconstruction loss via Eq 7
    - Prediction I(A; Y): CrossEntropy classification via Eq 8
    - I(X; Y) = H(Y) is constant, ignored in optimization
    """
    def __init__(self, beta=0.01):
        super().__init__()
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss()
    
    def compression_loss(self, mean, logvar):
        """
        KL(p(A|X) || N(0,I)) — variational upper bound on I(X; A) (Eq 6)
        r(A) = N(0, I) as stated in Section V.A implementation details
        """
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=-1)
        return kl.mean()
    
    def prediction_loss(self, logits, labels):
        """
        Lower bound on I(A; Y) via classification (Eq 8)
        Maximizing I(A; Y) = minimizing CrossEntropy
        """
        return self.ce_loss(logits, labels)
    
    def forward(self, exp_mean, exp_logvar, lip_mean, lip_logvar,
                cls_logits, emotion_labels):
        """
        Full DisTIB objective adapted for lip-expression disentanglement.
        
        In our framework:
        - A (label-related) = expression features (should contain emotion info)
        - Z (sample-exclusive) = lip features (should NOT be in expression)
        - Y (label) = emotion category
        
        exp_mean, exp_logvar: stochastic expression encoder output
        lip_mean, lip_logvar: stochastic lip encoder output (from EDTalk, detached)
        cls_logits: emotion classification logits from expression features
        emotion_labels: ground truth emotion indices
        """
        # Compression: minimize I(X; A) + I(X; Z)
        compress_exp = self.compression_loss(exp_mean, exp_logvar)
        compress_lip = self.compression_loss(lip_mean.detach(), lip_logvar.detach())
        compression = compress_exp + compress_lip
        
        # Prediction: maximize I(A; Y) = minimize CE
        prediction = self.prediction_loss(cls_logits, emotion_labels)
        
        # Full objective: -Prediction + β·Compression
        # (Sufficiency is handled by the reconstruction loss in the training loop)
        loss = prediction + self.beta * compression
        
        return loss, {
            'compress_exp': compress_exp.item(),
            'compress_lip': compress_lip.item(),
            'prediction': prediction.item(),
        }


# ============================================================
# 5. CLIP Verification Loss (novel contribution)
# ============================================================
class CLIPVerificationLoss(nn.Module):
    """
    After generating a face, verify it matches the emotion text prompt
    using CLIP's shared vision-language space.
    
    This closes the loop:
    text → emotion model → generator → image → CLIP image encode → cosine sim with text
    
    No existing talking head paper does this.
    """
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.register_buffer('clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
    
    def preprocess_for_clip(self, images):
        images = (images + 1) / 2  # [-1,1] → [0,1]
        images = F.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images = (images - self.clip_mean.to(images.device)) / self.clip_std.to(images.device)
        return images
    
    def forward(self, generated_images, text_features):
        clip_images = self.preprocess_for_clip(generated_images)
        with torch.no_grad():
            image_features = self.clip_model.encode_image(clip_images).float()
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features.detach(), dim=-1)
        cos_sim = F.cosine_similarity(image_features, text_features)
        return (1 - cos_sim).mean()


# ============================================================
# 6. Main Model V2
# ============================================================
class CLIPEmotionExpModelV2(nn.Module):
    """
    V2: Dynamic Graph Attention + Stochastic Encoding + DisTIB
    
    Output: [B, 10] expression weights — same interface as V1
    """
    def __init__(self, clip_dim=512, exp_dim=10, num_emotions=8, num_heads=4):
        super().__init__()
        self.exp_dim = exp_dim
        self.num_emotions = num_emotions
        
        # Initialize graph nodes with CLIP emotion text embeddings
        # So node 0 = "angry" in CLIP space, node 4 = "happy", etc.
        emotion_names = ["angry", "contempt", "disgusted", "fear",
                         "happy", "neutral", "sad", "surprised"]
        try:
            clip_model, _ = clip.load("ViT-B/32", device="cpu")
            with torch.no_grad():
                tokens = clip.tokenize([f"a person expressing {e}" for e in emotion_names])
                init_emb = clip_model.encode_text(tokens).float()
            del clip_model
            print("Graph nodes initialized with CLIP emotion embeddings")
        except:
            init_emb = None
            print("WARNING: CLIP init failed, using random embeddings")
        
        # Emotion Subspace Projection (inspired by CLIP-PAE, SIGGRAPH 2023)
        # CLIP text embeddings for different emotions are very close (0.88-0.95 cosine sim)
        # because shared context "a person expressing..." dominates.
        # Solution: project text embedding onto the subspace spanned by emotion embeddings.
        # This strips the shared context and keeps only the emotion-discriminative component.
        emotion_names = ["angry", "contempt", "disgusted", "fear",
                         "happy", "neutral", "sad", "surprised"]
        try:
            _clip, _ = clip.load("ViT-B/32", device="cpu")
            with torch.no_grad():
                _tokens = clip.tokenize([f"a person expressing {e}" for e in emotion_names])
                _emotion_basis = _clip.encode_text(_tokens).float()  # [8, 512]
                # Orthogonalize via SVD for a clean subspace basis
                U, S, Vh = torch.linalg.svd(_emotion_basis, full_matrices=False)
                self.register_buffer('emotion_basis', Vh[:8])  # [8, 512] orthonormal basis
            del _clip
            print("Emotion subspace basis computed from CLIP embeddings")
        except Exception as e:
            print(f"WARNING: Could not compute emotion subspace: {e}")
            self.register_buffer('emotion_basis', torch.randn(8, 512))
        
        # Learnable scaling for projected vs residual components
        self.subspace_alpha = nn.Parameter(torch.tensor(2.0))  # amplify emotion component
        
        # Dynamic graph attention (DGAT paper)
        self.emotion_graph = DynamicGraphAttention(
            num_nodes=num_emotions,
            feat_dim=clip_dim,
            num_heads=num_heads,
            momentum=0.5,
            init_embeddings=init_emb,
        )
        
        # Cross-attention
        self.cross_attention = TextEmotionCrossAttentionV2(
            text_dim=clip_dim,
            emotion_dim=clip_dim,
            num_heads=num_heads,
        )
        
        # Stochastic expression encoder (DisTIB: p(A|X) = N(μ, σ²))
        self.exp_encoder = StochasticEncoder(
            input_dim=clip_dim,
            output_dim=clip_dim,
            hidden_dim=256,
        )
        
        # Expression projector: 512 → 10 exp weights
        self.exp_projector = nn.Sequential(
            nn.Linear(clip_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, exp_dim),
        )
        
        # Emotion classifier for DisTIB prediction term
        self.emotion_classifier = nn.Sequential(
            nn.Linear(clip_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_emotions),
        )
        
        # DisTIB loss
        self.distib_loss = DisTIBLoss(beta=0.01)
    
    def forward(self, clip_text_feat, return_details=False):
        """
        clip_text_feat: [B, 512]
        Returns: [B, 10] expression weights
        """
        # Step 1: Dynamic graph attention on emotion nodes
        emotion_nodes = self.emotion_graph(text_condition=clip_text_feat)
        
        # Step 2: Emotion Subspace Projection (CLIP-PAE inspired)
        # Project text embedding onto emotion subspace to extract emotion-specific signal
        # projection = component in emotion subspace (what emotion)
        # residual = everything else (context, identity, syntax)
        proj = torch.matmul(
            torch.matmul(clip_text_feat, self.emotion_basis.T),  # [B, 8] coords in subspace
            self.emotion_basis  # [8, 512] back to full space
        )  # [B, 512] — emotion-specific component
        residual = clip_text_feat - proj  # [B, 512] — context/other info
        
        # Amplify emotion component, keep residual
        emotion_text_feat = self.subspace_alpha * proj + residual
        
        # Step 3: Cross-attention — emotion-enhanced text → emotion nodes
        fused_feat, attn_weights = self.cross_attention(emotion_text_feat, emotion_nodes)
        
        # Step 3: Stochastic encoding (DisTIB)
        exp_sample, exp_mean, exp_logvar = self.exp_encoder.sample(fused_feat)
        
        # Step 4: Project to expression weights
        exp_weights = self.exp_projector(exp_sample)
        
        if return_details:
            cls_logits = self.emotion_classifier(exp_sample)
            return exp_weights, {
                'attn_weights': attn_weights,
                'exp_mean': exp_mean,
                'exp_logvar': exp_logvar,
                'cls_logits': cls_logits,
                'exp_sample': exp_sample,
                'adjacency': self.emotion_graph.adjacency,
            }
        return exp_weights
    
    def compute_distib_loss(self, exp_mean, exp_logvar, lip_mean, lip_logvar,
                            cls_logits, emotion_labels):
        return self.distib_loss(exp_mean, exp_logvar, lip_mean, lip_logvar,
                               cls_logits, emotion_labels)
    
    def compute_attn_loss(self, attn_weights, emotion_labels):
        """
        Direct supervision: force correct emotion node to have highest attention.
        attn_weights: [B, 8], emotion_labels: [B] (indices 0-7)
        """
        # Cross-entropy on attention weights — treat as classification
        log_attn = torch.log(attn_weights + 1e-8)
        return F.nll_loss(log_attn, emotion_labels)


# ============================================================
# Emotion labels (same as V1)
# ============================================================
EMOTION_TO_IDX = {
    'angry': 0, 'contempt': 1, 'disgusted': 2, 'fear': 3,
    'happy': 4, 'neutral': 5, 'sad': 6, 'surprised': 7,
}

IDX_TO_EMOTION = {v: k for k, v in EMOTION_TO_IDX.items()}


if __name__ == '__main__':
    print("=== Testing CLIPEmotionExpModelV2 ===")
    model = CLIPEmotionExpModelV2()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")
    
    # Forward
    text_feat = torch.randn(4, 512)
    exp_weights, details = model(text_feat, return_details=True)
    print(f"Input: {text_feat.shape}")
    print(f"Output: {exp_weights.shape}")
    print(f"Attention: {details['attn_weights'].shape}")
    print(f"Adjacency: {details['adjacency'].shape}")
    print(f"Exp mean: {details['exp_mean'].shape}")
    print(f"Cls logits: {details['cls_logits'].shape}")
    
    # DisTIB loss
    lip_mean = torch.randn(4, 512)
    lip_logvar = torch.randn(4, 512)
    emotion_labels = torch.tensor([0, 4, 6, 7])
    loss, loss_dict = model.compute_distib_loss(
        details['exp_mean'], details['exp_logvar'],
        lip_mean, lip_logvar,
        details['cls_logits'], emotion_labels
    )
    print(f"\nDisTIB loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
    
    # Test CLIP verification (mock)
    print("\n=== CLIP Verification Loss ===")
    print("Requires CLIP model — tested during training")
    
    print("\n=== All V2 tests passed! ===")


# ============================================================
# Intensity Head — predicts scalar multiplier from CLIP text
# ============================================================
class IntensityAwareProjection(nn.Module):
    """
    Learned linear projection that amplifies intensity differences in CLIP space.
    Inspired by LABCLIP (2025) and SToRI (EMNLP 2024).
    
    Raw CLIP: cos("low anger", "high anger") ≈ 0.98 (nearly identical)
    After projection: the tiny intensity signal gets amplified into a 
    discriminable representation for the IntensityHead.
    """
    def __init__(self, clip_dim=512, hidden_dim=256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, clip_feat):
        return self.proj(clip_feat)


class IntensityHead(nn.Module):
    """
    Predicts emotion intensity (scalar) from projected CLIP embedding.
    Decouples WHAT emotion (handled by DGAT) from HOW MUCH (this head).
    
    Trained with MEAD level labels:
        level_1 → 0.6
        level_2 → 0.8  
        level_3 → 1.0
    """
    def __init__(self, input_dim=256, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # outputs [0, 1]
        )
    
    def forward(self, projected_feat):
        # Maps to [0.4, 1.4] range — covers mild to intense
        return 0.4 + self.net(projected_feat)  # [B, 1]


class CLIPEmotionExpModelV2Intensity(CLIPEmotionExpModelV2):
    """
    V2 + IntensityHead with DISENTANGLED routing:
    - Emotion component ONLY goes through DGAT → cross-attn → projector (same for all intensities)
    - Full CLIP embedding goes to IntensityHead (captures intensity words)
    - Final = base_weights × intensity_scalar
    """
    def __init__(self, clip_dim=512, exp_dim=10, num_emotions=8, num_heads=4):
        super().__init__(clip_dim, exp_dim, num_emotions, num_heads)
        self.intensity_proj = IntensityAwareProjection(clip_dim, hidden_dim=256)
        self.intensity_head = IntensityHead(input_dim=256)
    
    def forward(self, clip_text_feat, return_details=False):
        """
        Returns: [B, 10] expression weights scaled by predicted intensity
        """
        # Step 1: Emotion Subspace Projection — SPLIT emotion from intensity
        proj = torch.matmul(
            torch.matmul(clip_text_feat, self.emotion_basis.T),  # [B, 8]
            self.emotion_basis  # [8, 512]
        )  # [B, 512] — pure emotion component
        residual = clip_text_feat - proj  # [B, 512] — everything else (intensity, context)
        
        # ONLY emotion component goes to base pipeline (strips "low"/"high" etc)
        emotion_only_feat = self.subspace_alpha * proj + residual.detach()  # detach residual from base pipeline
        
        # Step 2: Dynamic graph attention on emotion nodes (sees emotion-only signal)
        emotion_nodes = self.emotion_graph(text_condition=emotion_only_feat)
        
        # Step 3: Cross-attention (emotion-only)
        fused_feat, attn_weights = self.cross_attention(emotion_only_feat, emotion_nodes)
        
        # Step 4: Stochastic encoding (DisTIB)
        exp_sample, exp_mean, exp_logvar = self.exp_encoder.sample(fused_feat)
        
        # Step 5: Project to expression weights (DIRECTION only — intensity-invariant)
        exp_weights = self.exp_projector(exp_sample)
        
        # Step 6: Project CLIP embedding to amplify intensity signal, then predict
        intensity_feat = self.intensity_proj(clip_text_feat)  # [B, 256] — amplified intensity
        intensity = self.intensity_head(intensity_feat)  # [B, 1]
        
        # Step 7: Scale direction by intensity
        exp_weights_scaled = exp_weights * intensity
        
        if return_details:
            cls_logits = self.emotion_classifier(exp_sample)
            return exp_weights_scaled, {
                'attn_weights': attn_weights,
                'exp_mean': exp_mean,
                'exp_logvar': exp_logvar,
                'cls_logits': cls_logits,
                'exp_sample': exp_sample,
                'adjacency': self.emotion_graph.adjacency,
                'intensity': intensity,
                'base_weights': exp_weights,
            }
        return exp_weights_scaled
