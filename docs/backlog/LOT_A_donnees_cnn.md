# Lot A — Données, exploration & CNN de base

| | |
|---|---|
| **Responsable** | Dev A |
| **Questions couvertes** | Q1, Q2, Q3, Q4 + socle commun `src/` |
| **Notebook de travail** | `notebooks/A_donnees_cnn.ipynb` |
| **Branches** | `feature/lot-a-socle`, `feature/lot-a-exploration`, `feature/lot-a-baseline` |
| **Sprints** | S0 (socle), S1 (Q1–Q4), S2 (Q8 support + relecture), S3 (aide assemblage) |

## Objectif du lot

Fournir aux deux autres lots un **pipeline de données fiable** et un **modèle de référence** (`baseline`) contre lequel toutes les améliorations seront mesurées. Ce lot est sur le chemin critique : le socle doit être livré dès le sprint S0.

## Attendus (livrables)

| # | Livrable | Forme | Critère d'acceptation |
|---|---|---|---|
| A-L1 | Socle commun `src/` (PyTorch) | `config.py`, `data.py`, `train.py`, `evaluate.py`, `io.py` | Les trois membres exécutent `load_datasets()`, `make_loaders()` et `evaluate_model()` sans erreur ; PR relue par B et C. |
| A-L2 | Section Q1 — Exploration des données | Cellules Markdown + code | Description du dataset (source, licence, taille, splits, résolution, format), grille d'exemples par classe, histogramme des classes, statistiques (dimensions, canaux), **interprétation rédigée**. |
| A-L3 | Section Q2 — Équilibre des classes | Markdown + code | Tableau des effectifs et proportions train/test ; réponse argumentée « équilibré ou non » ; décision de rééquilibrage justifiée (et appliquée si nécessaire : `class_weight`, sur/sous-échantillonnage). |
| A-L4 | Section Q3 — Architecture du CNN | Markdown + code + `print(model)` / `torchinfo.summary()` / schéma | Contient au minimum : convolutions, pooling, dropout, couches denses cachées. Chaque bloc est justifié (rôle, nombre de filtres, taille de noyau, activation). Nombre de paramètres commenté. |
| A-L5 | Section Q4 — Entraînement et performance | Markdown + code + figures | Courbes loss/accuracy train vs validation, métriques test via `evaluate_model()`, matrice de confusion, analyse du sur/sous-apprentissage et des confusions entre classes (ex. glacier/mountain, street/building). |
| A-L6 | Modèle `baseline` sauvegardé | `models/baseline.pt` (`state_dict`) | Rechargeable via `io.load_model(BaselineCNN, "baseline")` ; métriques test reproduites. |
| A-L7 | Script de prédiction sur photo | `io.predict_image()` | Prend un chemin de fichier, renvoie classe + probabilités ; utilisé pour Q8 par le lot C. |

## Backlog

Priorité : **P0** bloquant pour les autres lots · **P1** requis pour le rendu · **P2** amélioration.
Estimation en demi-journées (½ j).

| ID | Tâche | Priorité | Estim. | Sprint | Critère d'acceptation |
|---|---|---|---|---|---|
| A-01 | Créer le squelette du dépôt : `src/`, `notebooks/`, `models/`, `photos/`, `data/`, `.gitignore` (data, models, checkpoints, zip, `.ipynb_checkpoints`), `requirements.txt` | P0 | ½ | S0 | Structure conforme à `GESTION_DE_PROJET.md` §5 ; `pip install -r requirements.txt` fonctionne. |
| A-02 | `src/config.py` : `SEED`, `set_seed()` (random, numpy, torch, cuda), `DEVICE`, `IMG_SIZE`, `CLASSES`, chemins, `FAST_RUN` | P0 | ¼ | S0 | Valeurs conformes aux décisions D2–D6 ; `DEVICE` détecté automatiquement. |
| A-03 | `src/data.py` : téléchargement HF (`datasets.load_dataset`), classe `Dataset` PyTorch (redimensionnement 128×128, `ToTensor`, transform optionnelle), split train/val stratifié (`train_test_split` sur les indices), `make_loaders()` ; cache local dans `data/` | P0 | 1 | S0 | `load_datasets()` renvoie 3 `Dataset` ; un batch a la forme `(B, 3, 128, 128)` ; temps de chargement < 5 min après cache ; testé sur Colab et en local. |
| A-04 | `src/evaluate.py` : `evaluate_model(model, loader)` (`model.eval()`, `torch.no_grad()`, accuracy, precision/recall/F1 macro, rapport par classe), `plot_history()`, `plot_confusion_matrix()` | P0 | ½ | S0 | Sortie = `dict` sérialisable + figures légendées ; validé par B et C. |
| A-05 | `src/io.py` : `save_model()` (`state_dict`), `load_model(builder, name)`, `predict_image(model, path, transform)` (PIL → transform → `softmax` sur les logits) | P0 | ½ | S0–S1 | Sauvegarde/rechargement identiques (mêmes prédictions) ; `predict_image()` gère JPEG/PNG de taille quelconque et l'orientation EXIF. |
| A-06 | Q1 — Exploration : description du dataset, exemples visuels par classe, dimensions, format, distribution | P1 | 1 | S1 | Livrable A-L2. |
| A-07 | Q2 — Analyse de l'équilibre et décision de rééquilibrage (argumentée ; appliquer `class_weight` ou équivalent si l'écart le justifie) | P1 | ½ | S1 | Livrable A-L3 ; la décision est justifiée chiffres à l'appui. |
| A-08 | Q3 — Conception du CNN de base `src/models.py::BaselineCNN(nn.Module)` (constructeur paramétrable : nb de filtres, taux de dropout, taille des couches denses ; `forward` renvoie les logits) | P0 | 1 | S1 | Livrable A-L4 ; les noms des paramètres du constructeur sont **validés avec Dev B** car exposés à `GridSearchCV` via skorch (`module__n_filters`, `module__dropout`…). |
| A-09 | Q4 — Entraînement (`src/train.py::fit_model()` : boucle train/val, `Adam`, `CrossEntropyLoss`, early stopping sur la loss de validation, sauvegarde du meilleur epoch), évaluation, courbes, matrice de confusion, interprétation | P1 | 1 | S1 | Livrable A-L5 ; `baseline` sauvegardé (A-L6). |
| A-10 | Rédaction des sections Q1–Q4 : justification de chaque choix, interprétation de chaque résultat, relecture orthographique | P1 | 1 | S1–S2 | DoD §8 respectée ; relu par Dev C. |
| A-11 | Support Q8 : collecte des photos personnelles (chaque membre ≥ 2 photos par classe), organisation dans `photos/<classe>/`, nettoyage (orientation EXIF, format) | P1 | ½ | S2 | ≥ 36 photos, prêtes pour `predict_image()`. |
| A-12 | Relecture des sections du lot B (code + rédaction) | P1 | ½ | S2 | Commentaires de PR déposés. |
| A-13 | Aide à l'assemblage du notebook final avec Dev C (fusion sections A, exécution complète, vérification des sorties) | P1 | 1 | S3 | Notebook final exécuté de bout en bout sans erreur. |
| A-14 | Ajouter `nn.BatchNorm2d` et comparer avec/sans dans la baseline | P2 | ½ | S2 | Tableau comparatif interprété. |
| A-15 | Tests unitaires légers sur `src/data.py` (formes, plage de valeurs, ordre des classes) | P2 | ½ | S1 | `pytest` vert. |

## Dépendances

- **Entrantes** : aucune (framework fixé : PyTorch).
- **Sortantes** : A-03/A-04/A-05 (socle) → lots B et C ; A-08 (constructeur `BaselineCNN`) → B-02 (`GridSearchCV`) ; A-L6 (`baseline`) → B-05 (comparaison), C-04 (comparaison), B-09 (Grad-CAM de secours).

## Points de vigilance

- Le dataset HF fournit `train` et `test` ; **ne jamais** utiliser `test` pour l'early stopping ou le choix d'hyperparamètres (D5).
- Vérifier le nom exact des colonnes / labels du dataset HF (`image`, `label`) et l'ordre des classes → source unique de vérité : `config.CLASSES`.
- Garder la baseline **simple et rapide** (quelques minutes sur GPU) : elle sera ré-entraînée par `GridSearchCV`.
- Toujours appeler `model.train()` / `model.eval()` au bon moment (dropout !) et envelopper l'évaluation dans `torch.no_grad()`.
- `DataLoader` : `num_workers=0` sous Windows en cas de problème de multiprocessing ; `pin_memory=True` sur GPU.
- Documenter le temps d'entraînement et le matériel utilisé (CPU/GPU) : utile pour l'interprétation et la reproductibilité.
