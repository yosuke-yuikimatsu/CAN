import torch
import torch.nn as nn
import torch.nn.functional as F

from model import CliffordNet


class CliffordImageEncoder(nn.Module):
    """Image encoder wrapper that turns a CliffordNet backbone into CLIP-style embeddings."""

    def __init__(
        self,
        embed_dim: int = 512,
        patch_size: int = 2,
        backbone_dim: int = 128,
        depth: int = 12,
        cli_mode: str = "full",
        ctx_mode: str = "diff",
        shifts=None,
        drop_path_rate: float = 0.1,
        enable_cuda: bool = False,
    ):
        super().__init__()
        if shifts is None:
            shifts = [1, 2]

        self.backbone = CliffordNet(
            num_classes=embed_dim,
            patch_size=patch_size,
            embed_dim=backbone_dim,
            cli_mode=cli_mode,
            ctx_mode=ctx_mode,
            shifts=shifts,
            depth=depth,
            drop_path_rate=drop_path_rate,
            enable_cuda=enable_cuda,
        )
        self.proj = nn.Linear(backbone_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.backbone.forward_features(images)
        x = x.mean(dim=[-2, -1])
        x = self.backbone.norm(x)
        x = self.proj(x)
        return F.normalize(x, dim=-1)


class CliffordCLIP(nn.Module):
    """
    Standard CLIP pipeline:
      - text embeddings from a pretrained text model
      - image embeddings from CliffordNet
      - symmetric image/text contrastive logits
    """

    def __init__(
        self,
        text_model_name: str = "openai/clip-vit-base-patch32",
        embed_dim: int = 512,
        image_encoder_kwargs=None,
        freeze_text_model: bool = True,
    ):
        super().__init__()

        if image_encoder_kwargs is None:
            image_encoder_kwargs = {}

        try:
            from transformers import AutoTokenizer, CLIPTextModelWithProjection
        except ImportError as exc:
            raise ImportError(
                "transformers is required for the CLIP text encoder. "
                "Install with: pip install transformers"
            ) from exc

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(text_model_name)
        self.text_projection = nn.Identity()

        text_embed_dim = self.text_encoder.config.projection_dim
        if text_embed_dim != embed_dim:
            self.text_projection = nn.Linear(text_embed_dim, embed_dim)

        self.image_encoder = CliffordImageEncoder(embed_dim=embed_dim, **image_encoder_kwargs)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))

        if freeze_text_model:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

    def tokenize(self, texts, max_length: int = 77, device=None):
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {k: v.to(device) for k, v in encoded.items()}
        return encoded

    def encode_text(self, text_inputs) -> torch.Tensor:
        outputs = self.text_encoder(**text_inputs)
        text_embeds = self.text_projection(outputs.text_embeds)
        return F.normalize(text_embeds, dim=-1)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def forward(self, images: torch.Tensor, text_inputs):
        image_features = self.encode_image(images)
        text_features = self.encode_text(text_inputs)

        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return {
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "image_embeds": image_features,
            "text_embeds": text_features,
        }


def clip_contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    targets = torch.arange(logits_per_image.size(0), device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)
    loss_t = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (loss_i + loss_t)
