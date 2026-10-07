from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Sequence

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import models as tvmodels

from src.config import (IMAGENET_MEAN,IMAGENET_STD,IMG_CHANNELS,IMG_SIZE,NUM_CLASSES,)
from src.models import countParameters

from src.config import CLASSES, FIGURES_DIR, MODELS_DIR, setSeed
from src.data import load_datasets, trainAugmentation
from src.evaluate import (
    evaluateModel, loadMetrics, metricsSummary, plotHistory,
    pltConfusionMatrix, saveMetrics,
)
from src.model_io import checkRoundTrip, saveModel
from src.train import fit_model, makeLoaders

BACKBONES: dict[str, tuple[Callable[..., nn.Module], Any]] = {
    "resnet18": (tvmodels.resnet18, tvmodels.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (tvmodels.resnet34, tvmodels.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (tvmodels.resnet50, tvmodels.ResNet50_Weights.IMAGENET1K_V2),
}

STAGES: tuple[str, ...] = ("conv1", "bn1", "layer1", "layer2", "layer3", "layer4")

def _truncatedBackbone(name: str, pretrained: bool) -> tuple[nn.Sequential, int]:
    """ResNet sans avgpool ni fc, en gardant les noms d'etages pour le degel selectif."""
    if name not in BACKBONES:
        raise ValueError(f"Backbone '{name}' inconnu (disponibles : {sorted(BACKBONES)}).")

    constructor, weights = BACKBONES[name]
    resnet = constructor(weights=weights if pretrained else None)

    backbone = nn.Sequential(OrderedDict([
        ("conv1", resnet.conv1),
        ("bn1", resnet.bn1),
        ("relu", resnet.relu),
        ("maxpool", resnet.maxpool),
        ("layer1", resnet.layer1),
        ("layer2", resnet.layer2),
        ("layer3", resnet.layer3),
        ("layer4", resnet.layer4),
    ]))
    return backbone, resnet.fc.in_features

class TransferResNet(nn.Module):

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        in_channels: int = IMG_CHANNELS,
        img_size: tuple[int, int] = IMG_SIZE,
        backbone: str = "resnet18",
        pretrained: bool = True,
        conv_filters: int = 256,
        conv_blocks: int = 2,
        fc_units: int = 256,
        dropout: float = 0.4,
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if in_channels != 3:
            raise ValueError("Les poids ImageNet attendent 3 canaux d'entree.")
        if conv_blocks < 1:
            raise ValueError(f"conv_blocks doit valoir au moins 1 (recu : {conv_blocks}).")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout doit appartenir a [0, 1[ (recu : {dropout}).")

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.img_size = tuple(img_size)
        self.backbone_name = backbone
        self.conv_filters = conv_filters
        self.conv_blocks = conv_blocks
        self.fc_units = fc_units
        self.dropout = dropout
        self.normalize = normalize
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean if normalize else torch.zeros_like(mean))
        self.register_buffer("pixel_std", std if normalize else torch.ones_like(std))

        self.backbone, backbone_dim = _truncatedBackbone(backbone, pretrained)
        self.backbone_dim = backbone_dim

        layers: list[nn.Module] = []
        channels = backbone_dim
        for _ in range(conv_blocks):
            layers += [
                nn.Conv2d(channels, conv_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(conv_filters),
                nn.ReLU(inplace=True),
            ]
            channels = conv_filters
        self.convHead = nn.Sequential(*layers)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()

        hidden = max(2, fc_units // 2)
        self.classifier = nn.Sequential(
            nn.Linear(conv_filters, fc_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_units, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout * 0.6),
        )
        self.head = nn.Linear(hidden, num_classes)

        self.feature_shape = self._inferFeatureShape()
        self.initHeadWeights()

    def initHeadWeights(self) -> None:
        for module in list(self.convHead.modules()) + list(self.classifier.modules()) + [self.head]:
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

    @torch.no_grad()
    def _inferFeatureShape(self) -> tuple[int, int, int]:
        was_training = self.training
        self.eval()
        dummy = torch.zeros(1, self.in_channels, *self.img_size)
        shape = tuple(self.convHead(self.backbone(dummy)).shape[1:])
        self.train(was_training)
        return shape 

    def setTrainableStages(self, stages: Sequence[str] | str = ()) -> None:

        wanted = STAGES if stages == "all" else tuple(stages)
        unknown = set(wanted) - set(STAGES)
        if unknown:
            raise ValueError(f"Etages inconnus : {sorted(unknown)} (attendus : {STAGES}).")

        for name, param in self.backbone.named_parameters():
            param.requires_grad = any(name.startswith(stage) for stage in wanted)
        self.trainable_stages = wanted
        self.train(self.training)  # reapplique le gel des BatchNorm

    def train(self, mode: bool = True):

        super().train(mode)
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d) and not module.weight.requires_grad:
                module.eval()
        return self

    @property
    def lastConvLayer(self) -> nn.Conv2d:
        convolutions = [m for m in self.convHead.modules() if isinstance(m, nn.Conv2d)]
        return convolutions[-1]

    def hyperparameters(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "img_size": self.img_size,
            "backbone": self.backbone_name,
            # False volontairement : au rechargement seule l'architecture compte
            "pretrained": False,
            "conv_filters": self.conv_filters,
            "conv_blocks": self.conv_blocks,
            "fc_units": self.fc_units,
            "dropout": self.dropout,
            "normalize": self.normalize,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        x = self.backbone(x)
        x = self.convHead(x)
        x = self.flatten(self.pool(x))
        x = self.classifier(x)
        return self.head(x)


def buildTransferResNet(**hparams: Any) -> TransferResNet:
    """Factory utilisee par loadModel() et checkRoundTrip(), comme buildBaseline()."""
    return TransferResNet(**hparams)


def summarizeTransfer(model: TransferResNet) -> str:
    counts = countParameters(model)
    lines = [f"Modele : {model.__class__.__name__} ({model.backbone_name})"]
    for key, value in model.hyperparameters().items():
        lines.append(f"  {key:<14}: {value}")

    lines.append(f"Entree : {(model.in_channels, *model.img_size)}")
    lines.append(f"Sortie backbone : {(model.backbone_dim, *model.feature_shape[1:])}")
    lines.append(f"Carte finale : {model.feature_shape}")
    lines.append(f"Sortie : {model.num_classes} logits")
    lines.append("")
    for label in ("total", "trainable", "frozen"):
        lines.append(f"Parametres {label:<12}: {counts[label]:,}".replace(",", " "))
    for section in ("backbone", "convHead", "classifier", "head"):
        share = 100 * counts[section] / counts["total"]
        lines.append(f"  - {section:<12}: {counts[section]:,} ({share:.1f} %)".replace(",", " "))
    return "\n".join(lines)


def mergeHistories(*histories: dict) -> dict:
    merged: dict = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    offset = 0
    for history in histories:
        for key in merged:
            merged[key].extend(history[key])
        offset += len(history["train_loss"])
    merged["best_epoch"] = offset - len(histories[-1]["train_loss"]) + histories[-1]["best_epoch"]
    merged["best_val_loss"] = histories[-1]["best_val_loss"]
    merged["phase_boundaries"] = [len(h["train_loss"]) for h in histories]
    return merged

COMPARED_METRICS: tuple[str, ...] = ("accuracy", "precision", "recall", "f1")


def compareMetrics(named_metrics: dict[str, dict]) -> str:
    names = list(named_metrics)
    width = max(len(n) for n in names) + 2

    lines = ["Comparaison sur le jeu de test", ""]
    lines.append(f"{'modele':<{width}}" + "".join(f"{m:>12}" for m in COMPARED_METRICS) + f"{'loss':>12}")
    lines.append("-" * (width + 12 * (len(COMPARED_METRICS) + 1)))
    for name, metrics in named_metrics.items():
        row = f"{name:<{width}}" + "".join(f"{metrics[m]:>12.4f}" for m in COMPARED_METRICS)
        lines.append(row + f"{metrics['loss']:>12.4f}")

    if len(names) == 2:
        reference, candidate = named_metrics[names[0]], named_metrics[names[1]]
        lines.append("")
        lines.append(f"Ecart ({names[1]} - {names[0]}) :")
        for metric in COMPARED_METRICS:
            delta = candidate[metric] - reference[metric]
            lines.append(f"  {metric:<10}: {delta:+.4f} ({100 * delta:+.2f} pts)")
    return "\n".join(lines)


def plotComparison(named_metrics: dict[str, dict], save_path: Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 5))
    n = len(named_metrics)
    width = 0.8 / n

    for i, (name, metrics) in enumerate(named_metrics.items()):
        positions = [x + i * width - 0.4 + width / 2 for x in range(len(COMPARED_METRICS))]
        values = [metrics[m] for m in COMPARED_METRICS]
        bars = ax.bar(positions, values, width=width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

    ax.set_xticks(range(len(COMPARED_METRICS)), COMPARED_METRICS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Baseline CNN vs transfer learning (test)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":

    setSeed()
    name = "resnet18_transfer"

    train_ds, val_ds, test_ds = load_datasets(train_transform=trainAugmentation())
    train_loader, val_loader, test_loader = makeLoaders(train_ds, val_ds, test_ds, batch_size=64)

    model = TransferResNet(backbone="resnet18", pretrained=True, conv_filters=256, dropout=0.4)

    model.setTrainableStages()
    print(summarizeTransfer(model))
    print("\n=== Phase 1 : extraction de caracteristiques (backbone gele) ===")
    model, history_head = fit_model(model, train_loader, val_loader, epochs=10, lr=1e-3, patience=3)

    model.setTrainableStages(["layer4"])
    print(f"\nParametres entrainables : {countParameters(model)['trainable']:,}".replace(",", " "))
    print("=== Phase 2 : fine-tuning de layer4 ===")
    model, history_ft = fit_model(model, train_loader, val_loader, epochs=10, lr=1e-3,
                                  weight_decay=1e-4, patience=3)

    history = mergeHistories(history_head, history_ft)

    path = saveModel(model, name)
    print(f"model saved to {path}")

    metrics = evaluateModel(model, test_loader)
    print(metricsSummary(name, metrics))
    saveMetrics(metrics, MODELS_DIR / f"{name}_metrics.json")

    plotHistory(history, title=name, save_path=FIGURES_DIR / f"{name}_history.png")
    pltConfusionMatrix(metrics["confusion_matrix"], CLASSES, title=name,
                       save_path=FIGURES_DIR / f"{name}_confusion_matrix.png")

    sample, _ = test_ds[0]
    checkRoundTrip(model, buildTransferResNet, name, sample.unsqueeze(0))

    baseline_path = MODELS_DIR / "baseline_cnn_metrics.json"
    if baseline_path.exists():
        comparison = {"baseline_cnn": loadMetrics(baseline_path), name: metrics}
        print("\n" + compareMetrics(comparison))
        plotComparison(comparison, save_path=FIGURES_DIR / "comparison_baseline_vs_transfer.png")
        (MODELS_DIR / "comparison.json").write_text(
            json.dumps(
                {k: {m: float(v[m]) for m in (*COMPARED_METRICS, "loss")}
                 for k, v in comparison.items()},
                indent=4,
            ),
            encoding="utf-8",
        )
    else:
        print(f"\n{baseline_path} absent : lance d'abord `python -m src.train`.")