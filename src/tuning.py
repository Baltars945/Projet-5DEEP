"""Recherche d'hyperparametres par GridSearchCV (sklearn) sur le CNN PyTorch.

Le pont entre les deux mondes est assure par skorch :
  - NeuralNetClassifier enveloppe BaselineCNN en estimateur sklearn ;
  - SliceDataset fait passer un Dataset PyTorch pour un tableau numpy aux yeux
    de sklearn, sans charger les images en RAM (decodage paresseux par batch).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from skorch import NeuralNetClassifier
from skorch.helper import SliceDataset
from torch import nn

from src.config import DEVICE, MODELS_DIR, SEED, setSeed
from src.data import inteldataset, load_datasets
from src.models import buildBaseline

# Grille par defaut : volontairement petite, cf. le calcul de cout dans le README
# de la fonction gridSearch. Les cles "module__*" sont transmises a BaselineCNN,
# les autres (lr, batch_size, max_epochs) a skorch lui-meme.
DEFAULT_GRID: dict[str, list] = {
    "lr": [1e-3, 3e-4],
    "module__n_filters": [16, 32],
    "module__dropout": [0.3, 0.5],
}


def datasetLabels(ds: inteldataset) -> np.ndarray:
    """Labels d'un inteldataset sans decoder une seule image.

    L'acces par colonne sur un split HuggingFace ne touche pas la colonne image,
    contrairement a ds[i] qui decoderait le JPEG.
    """
    return np.asarray(ds.data["label"], dtype=np.int64)[ds.indices]


def subsetDataset(ds: inteldataset, size: int, seed: int = SEED) -> tuple[inteldataset, np.ndarray]:
    """Sous-echantillon stratifie, pour garder la recherche dans un budget tenable."""
    labels = datasetLabels(ds)
    if size >= len(ds):
        return ds, labels
    keep, _ = train_test_split(
        np.arange(len(ds)), train_size=size, stratify=labels, random_state=seed
    )
    indices = [ds.indices[i] for i in keep]
    return inteldataset(ds.data, indices, transform=ds.transform), labels[keep]


def makeNet(
    max_epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device = DEVICE,
) -> NeuralNetClassifier:
    """Estimateur sklearn-compatible enveloppant BaselineCNN."""
    return NeuralNetClassifier(
        module=buildBaseline,
        # BaselineCNN sort des logits bruts : CrossEntropyLoss est obligatoire,
        # le defaut de skorch (NLLLoss) attendrait des log-probabilites.
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=lr,
        max_epochs=max_epochs,
        batch_size=batch_size,
        # None : c'est GridSearchCV qui decoupe train/val via la CV. Laisser le
        # split interne de skorch amputerait encore les donnees d'entrainement.
        train_split=None,
        iterator_train__shuffle=True,
        iterator_train__num_workers=num_workers,
        iterator_valid__num_workers=num_workers,
        device=device,
        verbose=0,
    )


def gridSearch(
    train_ds: inteldataset,
    grid: dict[str, list] | None = None,
    cv: int = 3,
    max_epochs: int = 10,
    subset: int | None = None,
    batch_size: int = 64,
    num_workers: int = 0,
    scoring: str = "accuracy",
    seed: int = SEED,
    verbose: int = 2,
) -> GridSearchCV:
    """Lance la recherche en grille avec validation croisee stratifiee.

    Cout = len(grille) * cv * max_epochs entrainements d'epoque. Sur CPU, garder
    subset et max_epochs bas : la combinaison par defaut (8 combinaisons, cv=3)
    represente deja 24 entrainements complets.

    n_jobs reste a 1 : paralleliser dupliquerait le modele sur le meme device.
    """
    grid = DEFAULT_GRID if grid is None else grid
    setSeed(seed)

    ds, y = (subsetDataset(train_ds, subset, seed) if subset else (train_ds, datasetLabels(train_ds)))

    # SliceDataset(ds, idx=0) -> les images ; les labels sont passes a part en
    # numpy pour que StratifiedKFold et le scoring sklearn fonctionnent.
    X = SliceDataset(ds, idx=0)

    net = makeNet(max_epochs=max_epochs, batch_size=batch_size, num_workers=num_workers)

    search = GridSearchCV(
        estimator=net,
        param_grid=grid,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed),
        scoring=scoring,
        n_jobs=1,
        refit=False,
        verbose=verbose,
        error_score="raise",
    )
    search.fit(X, y)
    return search


def searchResults(search: GridSearchCV) -> list[dict]:
    """cv_results_ aplati et trie du meilleur au pire, pret pour JSON ou DataFrame."""
    results = search.cv_results_
    rows = [
        {
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
            "mean_test_score": float(results["mean_test_score"][i]),
            "std_test_score": float(results["std_test_score"][i]),
            "mean_fit_time": float(results["mean_fit_time"][i]),
            "rank": int(results["rank_test_score"][i]),
        }
        for i, params in enumerate(results["params"])
    ]
    return sorted(rows, key=lambda r: r["rank"])


def saveSearch(search: GridSearchCV, save_path: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "scoring": search.scoring,
        "cv": search.cv.get_n_splits(),
        "results": searchResults(search),
    }
    save_path.write_text(json.dumps(payload, indent=4, default=str), encoding="utf-8")
    return save_path


def splitBestParams(best_params: dict) -> tuple[dict, dict]:
    """Separe les hyperparametres du module (-> BaselineCNN) de ceux de skorch (-> fit_model)."""
    module_params = {k.removeprefix("module__"): v for k, v in best_params.items() if k.startswith("module__")}
    fit_params = {k: v for k, v in best_params.items() if not k.startswith("module__")}
    return module_params, fit_params


if __name__ == "__main__":
    setSeed()
    train_ds, val_ds, test_ds = load_datasets()

    # Budget volontairement reduit : 8 combinaisons x 3 folds sur 2000 images.
    search = gridSearch(train_ds, cv=3, max_epochs=5, subset=2000)

    print(f"\nmeilleur score CV ({search.scoring}) : {search.best_score_:.4f}")
    print(f"meilleurs parametres : {search.best_params_}\n")
    for row in searchResults(search):
        print(f"  rank {row['rank']:>2} | {row['mean_test_score']:.4f} "
              f"+/- {row['std_test_score']:.4f} | {row['mean_fit_time']:6.1f}s")

    path = saveSearch(search, MODELS_DIR / "gridsearch_results.json")
    print(f"\nresultats sauvegardes : {path}")
