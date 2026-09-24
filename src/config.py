import os
import random
from pathlib import Path
import numpy as np
import torch

# seedPath

SEED: int = 42


def setSeed(seed: int = SEED, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seedWorker() -> None:
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def getGenerator(seed: int = SEED) -> torch.Generator:
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


# appareils

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# données

SOURCE_IMG_SIZE: tuple[int, int] = (150, 150)
IMG_SIZE: tuple[int, int] = (128, 128)
IMG_CHANNELS: int = 3
INPUT_SHAPE: tuple[int, int, int] = (IMG_CHANNELS, *IMG_SIZE)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.486, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0, 224, 0.225)

CLASSES: list[str] = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street"
]
NUM_CLASSES: int = len(CLASSES)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}
assert CLASSES == sorted(CLASSES)

VAL_SPLIT: float = 0.15
STRATIFY_SPLIT: bool = True

# paths

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
MODELS_DIR: Path = ROOT_DIR / "models"
PHOTOS_DIR: Path = ROOT_DIR / "photos"
SOURCE_DIR: Path = ROOT_DIR / "src"

if __name__ == "__main__":
    setSeed()
    print(f"SEED            : {SEED}")
    print(f"DEVICE          : {DEVICE}")
    print(f"IMG_SIZE        : {IMG_SIZE} (source {SOURCE_IMG_SIZE})")
    print(f"CLASSES         : {CLASSES}")
    print(f"ROOT_DIR        : {ROOT_DIR}")
    print(f"DATA_DIR        : {DATA_DIR}")
    print(f"NOTEBOOKS_DIR   : {NOTEBOOKS_DIR}")
    print(f"MODELS_DIR      : {MODELS_DIR}")
    print(f"PHOTOS_DIR      : {PHOTOS_DIR}")
    print(f"SOURCE_DIR      : {SOURCE_DIR}")