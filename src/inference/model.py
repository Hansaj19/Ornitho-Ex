"""
Ornitho-Ex | Model Architectures
Exact replicas of the architectures defined in bird_sound_training.ipynb.
"""

import torch
import torch.nn as nn
from typing import Tuple

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

NUM_CLASSES  = 182
NUM_CONCEPTS = 6


class ConceptBottleneckModel(nn.Module):
    """
    Concept Bottleneck Model (CBM) — λ=0.5 checkpoint.

    Architecture:
        EfficientNet-B0 (1-channel, 1280 features)
            → Concept Head: Linear(1280→256) → BN → ReLU → Dropout → Linear(256→6)
            → Species Head: Dropout → Linear(6→182)

    The species classifier operates *strictly* on the 6 predicted acoustic
    concepts — there is no skip connection back to the 1280-dim embedding.
    """

    def __init__(
        self,
        num_classes:  int = NUM_CLASSES,
        num_concepts: int = NUM_CONCEPTS,
        pretrained:   bool = False,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm")

        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,          # return pooled 1280-dim feature vector
        )
        in_features = self.encoder.num_features  # 1280

        # Concept Bottleneck Head
        self.concept_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_concepts),
        )

        # Species Classifier (inputs ONLY the concept predictions)
        self.species_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_concepts, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)           # (B, 1280)
        c_pred   = self.concept_head(features)  # (B, 6)
        y_pred   = self.species_head(c_pred)    # (B, 182)
        return y_pred, c_pred


def load_cbm_checkpoint(ckpt_path: str, device: str = "cpu") -> ConceptBottleneckModel:
    """
    Load a PyTorch-Lightning CBM checkpoint, stripping the Lightning
    'model.' prefix from state-dict keys before loading into the raw
    ConceptBottleneckModel.

    Parameters
    ----------
    ckpt_path : str
        Path to the .ckpt / .zip checkpoint file.
    device : str
        Target device ('cpu' or 'cuda').

    Returns
    -------
    ConceptBottleneckModel
        Model in eval() mode on the target device.
    """
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw_sd = checkpoint.get("state_dict", checkpoint)

    # Strip Lightning wrapper prefix "model."
    cleaned = {}
    for k, v in raw_sd.items():
        new_key = k[len("model."):] if k.startswith("model.") else k
        cleaned[new_key] = v

    model = ConceptBottleneckModel(pretrained=False)
    model.load_state_dict(cleaned, strict=True)
    model.to(device)
    model.eval()
    return model
