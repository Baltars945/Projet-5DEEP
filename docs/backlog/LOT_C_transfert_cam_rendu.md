# Lot C — Transfert d'apprentissage, CAM, photos personnelles & rendu final

| | |
|---|---|
| **Responsable** | Dev C |
| **Questions couvertes** | Q8, Q9, Bonus 1 + assemblage du notebook final et export |
| **Notebook de travail** | `notebooks/C_transfert_cam.ipynb` puis `notebooks/Projet_5DEEP.ipynb` (final) |
| **Branches** | `feature/lot-c-pretrained`, `feature/lot-c-cam`, `feature/lot-c-photos`, `feature/lot-c-final` |
| **Sprints** | S1 (pré-entraîné + biblio), S2 (Q9, Bonus 1), S2–S3 (Q8, assemblage, export) |

## Objectif du lot

Confronter le meilleur modèle « from scratch » à un **modèle pré-entraîné personnalisé**, construire un réseau à **architecture CAM** (GAP + dense) et visualiser ses activations, tester le meilleur modèle sur **des photos prises pour l'occasion**, puis **assembler, exécuter et exporter** le notebook final. Dev C est garant de la cohérence et de la qualité rédactionnelle du rendu.

## Attendus (livrables)

| # | Livrable | Forme | Critère d'acceptation |
|---|---|---|---|
| C-L1 | Section Q9 — Modèle pré-entraîné personnalisé | Markdown + code + figures | Choix du modèle justifié (`torchvision.models` : ResNet50 / MobileNetV3 / EfficientNet…) ; base gelée puis (option) fine-tuning des derniers blocs ; **ajout de couches de convolution et de couches denses** comme exigé ; entraînement, courbes, métriques test via `evaluate_model()` ; **comparaison argumentée** avec `best_model` (performance, nombre de paramètres, temps d'entraînement, données nécessaires). |
| C-L2 | Section Bonus 1 — CAM | Markdown (état de l'art sourcé) + code + figures | But et principe expliqués avec **citation précise** de Zhou et al. (CVPR 2016) ; contrainte architecturale (conv → GAP → dense softmax, pas de couches denses intermédiaires) expliquée ; réseau `CamCNN` construit, entraîné, évalué ; cartes CAM sur ≥ 6 images dont des cas ambigus street/building ; interprétation par image. |
| C-L3 | Section Q8 — Photos personnelles | Markdown + code + figures | ≥ 6 photos prises par les membres (idéalement ≥ 2 par classe), affichées avec prédiction, probabilités et vérité terrain ; analyse des erreurs (différence de distribution avec le dataset, cadrage, saison, résolution). |
| C-L4 | Notebook final `Projet_5DEEP.ipynb` | `.ipynb` exécuté, sorties présentes | Page de garde, sommaire, sections Q1 → Q9, Bonus 1, Bonus 2, conclusion, bibliographie ; exécution *Restart & Run All* sans erreur ; numérotation continue. |
| C-L5 | Export `Projet_5DEEP.html` (ou `.pdf`) | Fichier | Généré après la dernière exécution ; toutes les figures visibles ; mise en page propre. |
| C-L6 | Archive `Projet_5DEEP.zip` | Fichier | Contenu conforme à la checklist §9 de `GESTION_DE_PROJET.md` ; testée en la décompressant dans un dossier vierge. |

## Backlog

| ID | Tâche | Priorité | Estim. | Sprint | Critère d'acceptation |
|---|---|---|---|---|---|
| C-01 | Choisir le modèle pré-entraîné dans `torchvision.models` (critères : taille, temps d'inférence, compatibilité avec `IMG_SIZE`) et rédiger la justification | P1 | ¼ | S1 | Choix consigné dans le notebook avec 3 critères comparés. |
| C-02 | `src/models.py::build_pretrained(name, trainable_layers)` : backbone `torchvision` avec `weights=DEFAULT`, `requires_grad=False` sur la base, bloc `nn.Conv2d` personnalisé + `AdaptiveAvgPool2d` + couches denses + dropout (logits en sortie) ; transform `weights.transforms()` / `Normalize` ImageNet (D3) | P1 | 1 | S1 | Modèle instanciable ; `print(model)` montre la base gelée et les couches ajoutées ; test sur un mini-batch. |
| C-03 | Bibliographie CAM et Grad-CAM : lecture de Zhou et al. (2016, arXiv:1512.04150) ; coordination avec Dev B (B-08) pour une section « Explicabilité » cohérente (CAM → Grad-CAM comme généralisation) | P1 | 1 | S1 | Section Markdown avec équations, schéma personnel, citations complètes (auteurs, titre, conférence, année, URL). |
| C-04 | Q9 — Entraîner le modèle pré-entraîné (phase 1 : base gelée ; phase 2 optionnelle : fine-tuning des derniers blocs à faible learning rate), évaluer, comparer à `best_model` | P1 | 1½ | S2 | Livrable C-L1 ; tableau comparatif à métriques égales ; discussion coût/bénéfice du transfert. |
| C-05 | `src/models.py::CamCNN(nn.Module)` : blocs conv → `AdaptiveAvgPool2d(1)` → `nn.Linear` unique (aucune couche dense cachée) ; entraînement ; évaluation | P1 | 1 | S2 | Modèle entraîné, métriques test, comparaison rapide avec `baseline` (la contrainte GAP coûte-t-elle en performance ?). |
| C-06 | `src/explain.py::compute_cam(model, image, class_index)` + réutilisation de `overlay_heatmap()` (lot B) ; sélection d'images ambiguës ; génération des cartes pour la classe prédite et la classe concurrente ; interprétation | P1 | 1 | S2 | Livrable C-L2 ; ≥ 6 images, figures légendées, texte par image. |
| C-07 | Q8 — Organiser la collecte des photos (tous les membres, cf. A-11) ; prédire avec `best_model` via `io.predict_image()` ; figures + analyse des erreurs | P1 | ¾ | S2 | Livrable C-L3 ; note sur l'interprétation du terme « signes » du sujet. |
| C-08 | Préparer le gabarit du notebook final : page de garde, sommaire, titres de sections normalisés, style des cellules Markdown, bibliographie, conclusion | P1 | ½ | S2 | Gabarit validé par A et B. |
| C-09 | Fusion des notebooks A, B, C dans `Projet_5DEEP.ipynb` (ordre du sujet), harmonisation des imports (`from src import ...`), suppression des doublons, ajout des transitions entre sections | P0 | 1 | S3 | Une seule cellule d'imports/config en tête ; aucune redéfinition de fonction ; transitions rédigées. |
| C-10 | Exécution complète *Restart & Run All* (GPU), avec rechargement des modèles lourds depuis `models/` si `FAST_RUN` ; correction des erreurs | P0 | 1 | S3 | Aucune erreur ; toutes les sorties présentes ; durée d'exécution notée. |
| C-11 | Relecture croisée finale (orthographe, grammaire, style, cohérence des chiffres entre sections et tableau récapitulatif) — avec Dev A | P0 | ½ | S3 | Zéro faute détectée par un correcteur ; chiffres du récapitulatif identiques aux sections. |
| C-12 | Export HTML (`jupyter nbconvert --to html`) et/ou PDF ; vérification visuelle page par page | P0 | ¼ | S3 | Livrable C-L5. |
| C-13 | Constitution de l'archive `.zip`, test de décompression, dépôt sur la plateforme de rendu, tag `v1.0.0-rendu` sur `main` | P0 | ¼ | S3 | Livrable C-L6 ; rendu ≥ 24 h avant la date limite. |
| C-14 | Relecture des sections du lot A (code + rédaction) | P1 | ½ | S2 | Commentaires de PR déposés. |
| C-15 | Comparer deux modèles pré-entraînés (ex. ResNet50 vs MobileNetV2) | P2 | 1 | S2 | Tableau comparatif interprété. |
| C-16 | Fine-tuning progressif (dégel par blocs) et courbe de gain | P2 | 1 | S2 | Figure + interprétation. |

## Dépendances

- **Entrantes** : socle `src/` (A-03 à A-05) ; `overlay_heatmap()` (B-09) — en attendant, une version locale suffit ; `best_model` (B-L4) pour C-04 (comparaison) et C-07 (Q8) ; photos (A-11) ; toutes les sections A et B terminées pour C-09.
- **Sortantes** : `build_pretrained` → B-14 (optionnel) ; gabarit C-08 → structure des notebooks de travail A et B ; rendu final.

## Points de vigilance

- Le sujet impose de **personnaliser** le modèle pré-entraîné avec des couches de **convolution** *et* complètement connectées : ne pas se contenter d'une tête dense.
- Le pré-traitement du modèle pré-entraîné (`Normalize` avec les moyennes/écarts-types ImageNet) diffère du simple `ToTensor()` des modèles from scratch : bien l'appliquer aussi au **jeu de test et aux photos Q8**.
- CAM en PyTorch : les poids de la classe sont `model.classifier.weight[class_index]` ; les cartes d'activation s'obtiennent par un `forward_hook` sur le dernier bloc conv (même mécanisme que Grad-CAM, sans `backward`).
- CAM : l'article original impose GAP directement suivi de la couche de sortie ; la carte s'obtient en pondérant les dernières cartes d'activation par les poids de la classe. Expliquer **pourquoi** cette contrainte existe et en quoi Grad-CAM la lève (cohérence avec le lot B).
- Photos Q8 : prévoir des cas difficiles (rue avec bâtiments, montagne enneigée) — ce sont aussi de bons candidats pour CAM/Grad-CAM.
- Assemblage : bloquer S3 pour cette tâche. Ne **jamais** modifier les notebooks de travail pendant la fusion ; toute correction se fait dans `Projet_5DEEP.ipynb` puis, si utile, rétro-portée.
- Conserver **une journée de marge** : l'exécution complète du notebook final peut révéler des incompatibilités entre sections (noms de variables, versions).
