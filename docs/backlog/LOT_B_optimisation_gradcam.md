# Lot B — Optimisation, augmentation, meilleur modèle & Grad-CAM

| | |
|---|---|
| **Responsable** | Dev B |
| **Questions couvertes** | Q5, Q6, Q7, Bonus 2 |
| **Notebook de travail** | `notebooks/B_optimisation_gradcam.ipynb` |
| **Branches** | `feature/lot-b-gridsearch`, `feature/lot-b-augmentation`, `feature/lot-b-gradcam` |
| **Sprints** | S1 (outillage), S2 (Q5–Q7), S3 (Bonus 2) |

## Objectif du lot

Partir du CNN de base du lot A et **l'améliorer de manière méthodique** (recherche d'hyperparamètres, augmentation de données), **désigner et sauvegarder le meilleur modèle** « from scratch », puis **expliquer ses prédictions** avec Grad-CAM. Ce lot produit le modèle central du rendu ; les lots A et C s'y comparent.

## Attendus (livrables)

| # | Livrable | Forme | Critère d'acceptation |
|---|---|---|---|
| B-L1 | Section Q5 — Recherche d'hyperparamètres avec `GridSearchCV` | Markdown + code + tableau `cv_results_` | Utilise **explicitement** `sklearn.model_selection.GridSearchCV` sur un `skorch.NeuralNetClassifier` enveloppant `BaselineCNN` ; grille justifiée ; compromis temps/exhaustivité expliqué ; meilleurs paramètres commentés. |
| B-L2 | Section Q6 — Augmentation d'images | Markdown + code + figures | Visualisation d'images augmentées ; entraînement avec/sans augmentation à hyperparamètres égaux ; tableau comparatif via `evaluate_model()` ; réponse argumentée aux deux questions du sujet (*améliore-t-il ? était-ce prévisible ?*). |
| B-L3 | Section Q7 — Sélection et sauvegarde du meilleur modèle | Markdown + code | Tableau récapitulatif baseline / GridSearch / augmentation ; critère de choix explicite (F1 macro test, sur-apprentissage) ; sauvegarde dans `models/best_model.*` + preuve de rechargement. |
| B-L4 | Fichier `models/best_model.pt` | `state_dict` + hyperparamètres du constructeur (`models/best_model.json`) | Rechargeable par `io.load_model("best_model")` ; utilisé par Q8, Q9 (comparaison) et Bonus 2. |
| B-L5 | Section Bonus 2 — Grad-CAM | Markdown (état de l'art sourcé) + code + figures | Présentation du but et du principe (gradients de la classe cible par rapport aux cartes d'activation, pondération, ReLU) avec **citation précise** de Selvaraju et al. (2017) ; formule expliquée avec ses propres mots ; application au meilleur modèle sur ≥ 6 images dont des cas ambigus street/building et glacier/mountain ; interprétation de chaque carte. |
| B-L6 | `src/explain.py::compute_gradcam()` + `overlay_heatmap()` | Code réutilisable | Fonctionne sur n'importe quel modèle convolutif (pas de contrainte d'architecture) ; utilisable aussi par Dev C sur le modèle pré-entraîné. |

## Backlog

| ID | Tâche | Priorité | Estim. | Sprint | Critère d'acceptation |
|---|---|---|---|---|---|
| B-01 | Installer et prendre en main **skorch** (`NeuralNetClassifier`, `device=DEVICE`, `iterator_train__shuffle=True`, callback `EarlyStopping`) sur un mini-modèle jouet ; valider `GridSearchCV(cv=3)` de bout en bout | P0 | ½ | S1 | Exemple minimal fonctionnel commité ; versions `torch`/`skorch`/`scikit-learn` compatibles notées dans `requirements.txt`. |
| B-02 | Définir avec Dev A les paramètres du constructeur de `BaselineCNN` (filtres, dropout, unités denses) et la grille skorch (`module__*`, `lr`, `optimizer`, `batch_size`) | P0 | ¼ | S1 | Signature validée dans `src/models.py` ; utilisée par A-08. |
| B-03 | Pipeline d'augmentation `data.augment_transform()` (`torchvision.transforms` : `RandomHorizontalFlip`, `RandomRotation`, `RandomResizedCrop`, `ColorJitter`) + visualisation de 8 images augmentées | P1 | ½ | S1 | Augmentations **plausibles** pour des paysages (pas de flip vertical) ; justification écrite. |
| B-04 | Q5 — Concevoir la grille (2–3 hyperparamètres × 2–3 valeurs), lancer `GridSearchCV` sur un sous-échantillon stratifié du train (tenseurs `X` (N, 3, 128, 128) et `y` en mémoire, skorch les exige) avec peu d'epochs | P1 | 1½ | S2 | Livrable B-L1 ; durée totale < 2 h GPU ; `cv_results_` exporté en tableau lisible et **interprété**. |
| B-05 | Ré-entraîner le CNN avec les meilleurs hyperparamètres sur le train complet ; comparer à `baseline` | P1 | ½ | S2 | Métriques test via `evaluate_model()` ; gain (ou absence de gain) commenté. |
| B-06 | Q6 — Entraîner le même modèle avec augmentation ; comparer courbes et métriques ; répondre aux questions du sujet | P1 | 1 | S2 | Livrable B-L2 ; discussion du lien augmentation ↔ sur-apprentissage ↔ écart train/val. |
| B-07 | Q7 — Tableau récapitulatif, choix argumenté du meilleur modèle, sauvegarde `best_model`, test de rechargement, annonce aux lots A et C | P0 | ½ | S2 | Livrables B-L3, B-L4 ; issue « best_model disponible » fermée. |
| B-08 | Bibliographie Grad-CAM : lecture de Selvaraju et al. (ICCV 2017, arXiv:1610.02391), synthèse rédigée en français avec équations et schéma personnel, citations précises | P1 | 1 | S1–S2 | Section Markdown relue par Dev C (cohérence avec la partie CAM). |
| B-09 | `src/explain.py::compute_gradcam(model, image, class_index, target_layer)` via hooks (`register_forward_hook` / `register_full_backward_hook`) sur la dernière couche conv + `overlay_heatmap()` ; test sur `baseline` en attendant `best_model` | P1 | 1 | S2 | Fonctionne sur `baseline` et `best_model` sans modification ; carte superposée lisible. |
| B-10 | Bonus 2 — Sélection d'images ambiguës (street/building, glacier/mountain, sea/glacier), génération des Grad-CAM pour la classe prédite **et** pour la classe concurrente, interprétation | P1 | 1 | S3 | Livrable B-L5 ; ≥ 6 images, figures légendées, texte d'analyse par image. |
| B-11 | Rédaction complète des sections Q5–Q7 et Bonus 2, relecture orthographique | P1 | 1 | S2–S3 | DoD §8 ; relu par Dev A. |
| B-12 | Relecture des sections du lot C (code + rédaction) | P1 | ½ | S3 | Commentaires de PR déposés. |
| B-13 | `RandomizedSearchCV` ou affinage local autour des meilleurs paramètres | P2 | ½ | S2 | Comparaison commentée avec la grille. |
| B-14 | Grad-CAM sur le modèle pré-entraîné (lot C) pour comparer les zones d'attention des deux modèles | P2 | ½ | S3 | Figure comparative + interprétation. |

## Dépendances

- **Entrantes** : socle `src/` (A-03 à A-05) ; `build_baseline_cnn` paramétrable (A-08) ; `baseline` (A-L6) pour les comparaisons et les tests Grad-CAM.
- **Sortantes** : `best_model` (B-L4) → Q8 (lot C), comparaison Q9 (lot C), Bonus 2 ; `compute_gradcam` (B-L6) → B-14, éventuellement lot C.

## Points de vigilance

- `GridSearchCV` **ré-entraîne** le modèle `cv × |grille|` fois : anticiper le coût et **justifier** dans le notebook chaque réduction (sous-échantillon, epochs, `cv=3`). Le sujet impose la fonction, pas l'exhaustivité.
- Comparer les modèles **à conditions égales** (mêmes données, même seed, même nombre d'epochs max, même early stopping) — sinon la conclusion sur l'augmentation n'est pas valide.
- L'augmentation s'applique **uniquement au train**, jamais à la validation ni au test.
- Pour Grad-CAM, cibler la **dernière couche convolutive** (`model.features[-1]` ou équivalent) ; documenter la couche utilisée. Calculer la carte sur les **logits** (avant softmax) comme dans l'article ; penser à `model.eval()` et à `zero_grad()` avant chaque `backward()`.
- skorch attend des tenseurs en mémoire, pas un `DataLoader` : prévoir une fonction `dataset_to_tensors(ds)` dans `src/data.py` (ou un `Dataset` compatible skorch).
- Livrer `best_model` **tôt** dans S2, même si perfectible : les lots A et C en ont besoin ; une mise à jour ultérieure reste possible tant que la comparaison Q9 est relancée.
