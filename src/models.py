from __future__ import annotations
from typing import Any
import torch
from torch import nn
from src.config import IMG_CHANNELS, IMG_SIZE, NUM_CLASSES

def convBlock(inChannels: int, outChannels: int, useBatchnorm: bool = True) -> nn.Sequential:
    """Crée un bloc de convolution avec une couche Conv2d, une activation ReLU et un MaxPool2d."""
    layers: list[nn.Module] = [
        nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
    ]
    if useBatchnorm:
        layers.append(nn.BatchNorm2d(outChannels))
    layers.append(nn.ReLU(inplace=True))
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class BaselineCNN(nn.Module):
    """Un CNN simple pour la classification d'images."""
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IMG_CHANNELS,
        img_size: tuple[int, int] = IMG_SIZE,
        n_blocks: int = 4,
        n_filters: int = 32,
        fc_units: int = 256,
        dropout: float = 0.5,
        use_batchnorm: bool = True,
    ) -> None:
        super().__init__()
        if n_blocks < 1:
            raise ValueError(f"n_blocks doit valoir au moins 1 (reçu : {n_blocks}).")
        if n_filters < 1 or fc_units < 2:
            raise ValueError(
                f"n_filters ({n_filters})  doit valoir >= 1 et fc_units ({fc_units}) >=2."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout doit appartenir à [0, 1[ (reçu : {dropout}).")
        height, width = img_size
        reduction = 2 ** n_blocks
        if height % reduction or width % reduction:
            raise ValueError(
                f"L'image {img_size} n'est pas divisible par {reduction} : "
                f"les {n_blocks} MaxPool successifs produiraient une carte vide ou tronquée."
            )
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = tuple(img_size)
        self.n_blocks = n_blocks
        self.n_filters = n_filters
        self.fc_units = fc_units
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm

        blocks: list[nn.Module] = []
        channels = in_channels
        for block in range(n_blocks):
            outChannels = n_filters * (2 ** block)
            blocks.append(convBlock(channels, outChannels, useBatchnorm=use_batchnorm))
            channels = outChannels
        self.features = nn.Sequential(*blocks)


        self.feature_shape: tuple[int, int, int] = (
            channels,
            height // reduction,
            width // reduction,
        )
        flattened = channels * (height // reduction) * (width // reduction)

        hidden = max(2, fc_units // 2)
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(flattened, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_units, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.6),
        )

        self.head = nn.Linear(hidden, num_classes)
        self.initWeights()

    def initWeights(self) -> None:
       
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    @property
    def lastConvLayer(self) -> nn.Conv2d:
        convolutions = [m for m in self.features.modules() if isinstance(m, nn.Conv2d)]
        return convolutions[-1]

    def hyperparameters(self) -> dict: 
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "img_size": self.img_size,
            "n_blocks": self.n_blocks,
            "n_filters": self.n_filters,
            "fc_units": self.fc_units,
            "dropout": self.dropout,
            "use_batchnorm": self.use_batchnorm,
        }
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return self.head(x)
    
def buildBaseline(**hparams: Any) -> BaselineCNN:
    """Factory utilisée par loadModel() et par skorch (module=buildBaseline)."""
    return BaselineCNN(**hparams)


def countParameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    counts = {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }
    for sectionName, section in model.named_children():
        counts[sectionName] = sum(p.numel() for p in section.parameters())
    return counts


def summarizeModel(model: BaselineCNN) -> str:
    counts = countParameters(model)

    lines = [f"Modèle          : {model.__class__.__name__}"]
    for key, value in model.hyperparameters().items():
        lines.append(f"  {key:<14}: {value}")

    lines.append(f"Entrée          : {(model.in_channels, *model.img_size)}")
    lines.append(f"Carte finale    : {model.feature_shape}")
    lines.append(f"Sortie          : {model.num_classes} logits")
    lines.append("")
    lines.append(f"Paramètres totaux      : {counts['total']:,}".replace(",", " "))
    lines.append(f"Paramètres entraînables: {counts['trainable']:,}".replace(",", " "))

    for sectionName in ("features", "classifier", "head"):
        share = 100 * counts[sectionName] / counts["total"]
        value = f"{counts[sectionName]:,}".replace(",", " ")
        lines.append(f"  - {sectionName:<12}: {value} ({share:.1f} %)")

    return "\n".join(lines)

