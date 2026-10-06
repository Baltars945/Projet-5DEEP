from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from torch import nn
import torch 
from PIL import Image, ImageOps
from torchvision import transforms
from src.config import DEVICE, MODELS_DIR, CLASSES, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE

def modelPath(name: str | Path, models_dir: Path = MODELS_DIR) -> Path:
    """Return the path to the model file."""
    if isinstance(name, str):
        name = Path(name)
    if not name.suffix:
        name = name.with_suffix(".pt")
    return models_dir / name

def saveModel(model: nn.Module, name: str | Path, models_dir: Path = MODELS_DIR, hyperparameters: dict | None = None) -> Path:
    """Save the model to a file."""
    path = modelPath(name, models_dir)
    if hyperparameters is None and hasattr(model, "hyperparameters"):
        hyperparameters = model.hyperparameters()

    stateDict = { key: value.cpu() for key, value in model.state_dict().items() }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": stateDict,
            "hyperparameters": hyperparameters,
        },
        path,
    )

    metaPath = path.with_suffix(".json")
    metaPath.write_text(
        json.dumps(
            {
                "model_class": model.__class__.__name__,
                "hyperparameters": hyperparameters,
                "classes": CLASSES,
            },
            indent=4,
            ensure_ascii=False,
            default=list,
        ),
        encoding="utf-8",
    )   
    return path

def loadHparameters(path: str | Path, models_dir: Path = MODELS_DIR) -> dict:
    """Load the hyperparameters from a model file."""
    path = modelPath(path, models_dir).with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"Le fichier de modèle '{path}' n'existe pas.")

    metadata = json.loads(path.read_text(encoding="utf-8"))
    hparameters = metadata.get("hyperparameters", {})

    if isinstance(hparameters.get("img_size"), list):
        hparameters["img_size"] = tuple(hparameters["img_size"])

    return hparameters

def loadModel(builder: Callable[..., nn.Module],  name: str | Path,  device: torch.device = DEVICE,  models_dir: Path = MODELS_DIR, strict: bool = True, **hparams: Any) -> nn.Module:
    """Load a model from a file."""
    path = modelPath(name, models_dir)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier de modèle '{path}' n'existe pas.")

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    if not hparams:
        hparams = checkpoint.get("hyperparameters") or {}
        if isinstance(hparams.get("img_size"), list):
            hparams["img_size"] = tuple(hparams["img_size"])

    model = builder(**hparams).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    model.eval()

    return model

def defaultTransform(img_size: tuple[int, int] = IMG_SIZE) -> Callable:
    """Return a default transform for images."""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def loadImage(path: str | Path, transform: Callable | None = None, device: torch.device = DEVICE) -> torch.Tensor:
    """Load an image from a file and return it as a tensor."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Le fichier d'image '{path}' n'existe pas.")

    with Image.open(path) as raw:
        image = ImageOps.exif_transpose(raw.convert("RGB"))
    transform = transform or defaultTransform()
    tensor = transform(image).unsqueeze(0).to(device)
    return tensor



@torch.no_grad()
def predictImage(
    model: nn.Module,
    path: str | Path,
    transform: Callable | None = None,
    device: torch.device = DEVICE,
    class_names: list[str] | None = None,
    topk: int = 3,
) -> dict:
    """PIL -> transform -> logits -> softmax. Renvoie la classe prédite et les probabilités."""
    class_names = class_names or CLASSES

    tensor = loadImage(path, transform=transform, device=device)

    model.to(device)
    model.eval()
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    k = max(1, min(topk, probabilities.numel()))
    values, indices = probabilities.topk(k)
    index = int(indices[0])

    return {
        "path": str(path),
        "index": index,
        "label": class_names[index],
        "confidence": float(values[0]),
        "probabilities": {
            name: float(p) for name, p in zip(class_names, probabilities)
        },
        "topk": [
            (class_names[int(i)], float(v)) for v, i in zip(values, indices)
        ],
    }

def checkRoundTrip(
    model: nn.Module,
    builder: Callable[..., nn.Module],
    name: str | Path,
    sample: torch.Tensor,
    models_dir: Path = MODELS_DIR,
    device: torch.device = DEVICE,
    atol: float = 1e-6,
) -> bool:
    """Vérifie que save -> load redonne exactement les mêmes logits (critère A-05)."""
    hparams = model.hyperparameters() if hasattr(model, "hyperparameters") else {}
    saveModel(model, name, models_dir=models_dir, hyperparameters=hparams)

    model.eval()
    reloaded = loadModel(builder, name, device=device, models_dir=models_dir)

    sample = sample.to(device)
    with torch.no_grad():
        before = model(sample)
        after = reloaded(sample)

    identical = torch.allclose(before, after, atol=atol)
    ecart = float((before - after).abs().max())
    print(f"Round-trip '{name}' : {'OK' if identical else 'ECHEC'} (écart max {ecart:.2e})")
    return identical