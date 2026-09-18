# Gestion de projet — Projet 5DEEP (Intel Image Classification)

| | |
|---|---|
| **Sujet** | [docs/SUJET.md](SUJET.md) |
| **Membres** | 3 étudiants (Dev A, Dev B, Dev C) |
| **Livrable final** | `Projet_5DEEP.zip` = notebook `.ipynb` + export PDF/HTML |
| **Dépôt** | `git@github.com:Baltars945/Projet-5DEEP.git` |
| **Backlogs détaillés** | [Lot A](backlog/LOT_A_donnees_cnn.md) · [Lot B](backlog/LOT_B_optimisation_gradcam.md) · [Lot C](backlog/LOT_C_transfert_cam_rendu.md) |

| Rôle | Nom | Branches |
|---|---|---|
| Dev A — Données & CNN de base | _à compléter_ | `feature/lot-a-*` |
| Dev B — Optimisation & Grad-CAM | _à compléter_ | `feature/lot-b-*` |
| Dev C — Transfert, CAM & rendu final | _à compléter_ | `feature/lot-c-*` |

---

## 1. Objectif et principe du découpage

Le sujet est **séquentiel** (les questions 1 à 9 s'enchaînent et le bonus 2 dépend du « meilleur modèle ») et produit **un seul notebook**. Le travail à trois en parallèle repose sur deux règles :

1. **Un socle commun figé dès le jour 1** (`src/`) : chargement des données, prétraitement, fonction d'évaluation, fonction de sauvegarde. Tout le monde s'appuie dessus ; personne ne réinvente le pipeline dans son coin.
2. **Un notebook de travail par lot** (`notebooks/A_*.ipynb`, `B_*.ipynb`, `C_*.ipynb`), puis **une fusion en un notebook final unique** en fin de projet, prise en charge par Dev C. Cela évite les conflits Git sur un fichier `.ipynb` partagé.

Les dépendances entre lots sont gérées par des **stubs** : chaque lot peut démarrer sur le modèle de base (`baseline`) et rebrancher sur le « meilleur modèle » quand il sera disponible.

## 2. Décisions à figer au démarrage (jour 1, réunion de lancement)

Ces décisions conditionnent tout le code commun. Elles sont consignées ici et **ne changent plus** sans accord des trois membres.

| ID | Décision | Valeur retenue | Justification |
|---|---|---|---|
| D1 | Framework imposé par le formateur | **PyTorch** (`torch`, `torchvision`) — **décision confirmée** | Non négociable (cf. sujet). Le wrapper scikit-learn pour `GridSearchCV` est donc **skorch** (`NeuralNetClassifier`). |
| D2 | Taille des images en entrée | `128 × 128 × 3` (source : 150 × 150) | Compromis temps d'entraînement / qualité ; les modèles pré-entraînés acceptent cette taille. |
| D3 | Normalisation | tenseurs `float32` (C, H, W) ∈ [0, 1] via `transforms.ToTensor()` pour les modèles from scratch ; `transforms.Normalize(mean, std)` ImageNet (`weights.transforms()`) pour le modèle pré-entraîné du lot C | Standard `torchvision`. |
| D4 | Ordre des classes | `["buildings", "forest", "glacier", "mountain", "sea", "street"]` (ordre alphabétique = ordre HF) | Un seul mapping `label ↔ index` dans `src/config.py`. |
| D5 | Jeu de validation | 15 % du train, stratifié, `seed = 42` | Le jeu de test HF **ne sert qu'à l'évaluation finale**. |
| D6 | Graine aléatoire | `SEED = 42` partout (numpy, framework, split) | Reproductibilité des résultats présentés. |
| D7 | Métriques rapportées | accuracy, precision/recall/F1 macro, matrice de confusion, courbes loss/accuracy | Une seule fonction `evaluate_model()` utilisée par tous → comparaisons équitables. |
| D8 | Format de sauvegarde | `torch.save(model.state_dict(), "models/<nom>.pt")` + fonction de construction correspondante dans `src/models.py` (jamais `torch.save(model)` entier : fragile entre versions) ; `models/` ignoré par Git sauf le meilleur modèle si < 100 Mo | Bonnes pratiques PyTorch. |
| D9 | Environnement | Python 3.11, `requirements.txt` versionné (`torch`, `torchvision`, `skorch`, `scikit-learn`, `datasets`, `matplotlib`, `seaborn`, `pandas`, `jupyter`) ; `DEVICE = cuda si disponible, sinon cpu` dans `config.py` ; exécution GPU (Colab / Kaggle) acceptée pour les entraînements lourds | Le notebook final doit être ré-exécutable par chaque membre. |
| D10 | Langue du rendu | Français, style rédigé (pas de listes de commentaires bruts), orthographe relue | Critère de notation explicite. |

## 3. Répartition en trois lots

| Lot | Responsable | Questions du sujet | Résumé |
|---|---|---|---|
| **A — Données & CNN de base** | Dev A | Q1, Q2, Q3, Q4 + socle `src/` | Exploration, équilibrage, CNN from scratch, entraînement et évaluation de référence (`baseline`). |
| **B — Optimisation & Grad-CAM** | Dev B | Q5, Q6, Q7, Bonus 2 | `GridSearchCV`, augmentation d'images, sélection et sauvegarde du meilleur modèle, Grad-CAM. |
| **C — Transfert, CAM & rendu final** | Dev C | Q8, Q9, Bonus 1 + assemblage | Modèle pré-entraîné personnalisé, CAM (architecture GAP dédiée), photos personnelles, fusion du notebook, export PDF, relecture. |

Charge estimée : équilibrée (~ 1/3 par lot). Le lot A est le plus **critique en temps** (les autres en dépendent) ; le lot C est le plus **chargé en fin de projet** (assemblage). Dev A vient donc renforcer Dev C sur l'assemblage une fois son lot livré.

### 3.1 Graphe de dépendances

```
S0 : socle src/ (Dev A, revu par B et C)
        |
        +--> Lot A : baseline (Q1-Q4) --> Lot B : GridSearch + augmentation (Q5-Q6) --> meilleur modele (Q7)
        |                                                                                      |
        +--> Lot C : pre-entraine (Q9) + CAM (Bonus 1)  <------------- comparaison <-----------+
                                                                                               |
             Q8 photos perso (tous, script Dev A)  +  Bonus 2 Grad-CAM (Dev B)  <--------------+
                                    |
             Assemblage notebook final + export PDF (Dev C, aide Dev A)
```

Tant que le meilleur modèle n'existe pas, les lots B et C travaillent sur `baseline` (livré par A en fin de sprint 1) ou sur un mini-modèle jouet ; le rebranchement final est un simple changement de chemin de fichier.

## 4. Contrat d'interface du socle commun (`src/`)

Ce contrat est **la seule chose à connaître** pour travailler sur son lot sans attendre les autres. Signatures indicatives, en PyTorch (D1).

```
src/
├── config.py       # SEED, DEVICE, IMG_SIZE, CLASSES, chemins (DATA_DIR, MODELS_DIR, PHOTOS_DIR), FAST_RUN, set_seed()
├── data.py         # load_datasets(transform_train=None) -> (train_ds, val_ds, test_ds)  [torch.utils.data.Dataset]
│                   # make_loaders(datasets, batch_size) -> DataLoader x3 ; class_distribution(ds) -> DataFrame
│                   # base_transform() ; augment_transform(**params)  [torchvision.transforms]
├── models.py       # BaselineCNN(nn.Module, **hparams) ; CamCNN(nn.Module) ; build_pretrained(name, trainable_layers)
├── train.py        # fit_model(model, train_loader, val_loader, epochs, lr, patience) -> history  (early stopping + checkpoint)
├── evaluate.py     # evaluate_model(model, loader) -> dict(metrics) ; plot_history() ; plot_confusion_matrix()
├── explain.py      # compute_cam(...) ; compute_gradcam(...) (hooks forward/backward) ; overlay_heatmap(...)
└── io.py           # save_model(model, name) ; load_model(builder, name) ; predict_image(model, path, transform)
```

Règles :
- `load_datasets()` renvoie **toujours** des `Dataset` produisant `(tenseur (3, 128, 128) ∈ [0, 1], label int)` dans l'ordre et le format définis par D2–D5 ; l'augmentation ne s'applique qu'au `train_ds`.
- Tout modèle est un `nn.Module` dont le `forward` renvoie des **logits** (pas de softmax dans le modèle : `nn.CrossEntropyLoss` l'inclut, et CAM/Grad-CAM travaillent sur les logits).
- Le `DEVICE` vient de `config.py` ; aucun `.cuda()` codé en dur.
- `evaluate_model()` est **l'unique** fonction de mesure de performance : tout tableau comparatif du notebook final est construit avec elle.
- Toute modification du socle passe par une PR relue par les deux autres membres.

## 5. Organisation du dépôt

```
Projet-5DEEP/
├── .github/workflows/semantic.yml   # CI existante (semantic-release sur main)
├── docs/
│   ├── SUJET.md
│   ├── GESTION_DE_PROJET.md         # ce document
│   └── backlog/LOT_A_*.md, LOT_B_*.md, LOT_C_*.md
├── src/                             # socle commun (section 4)
├── notebooks/
│   ├── A_donnees_cnn.ipynb          # notebook de travail Dev A
│   ├── B_optimisation_gradcam.ipynb # notebook de travail Dev B
│   ├── C_transfert_cam.ipynb        # notebook de travail Dev C
│   └── Projet_5DEEP.ipynb           # notebook FINAL (fusion, seul rendu)
├── models/                          # poids sauvegardés (.gitignore sauf best_model)
├── photos/                          # photos personnelles Q8 (6 classes × >= 2 photos par membre)
├── data/                            # cache HF local (.gitignore)
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Workflow Git

Le dépôt utilise déjà `semantic-release` : les messages de commit suivent la convention **Conventional Commits**.

- **Branches** : `main` (releases, protégée) ← `develop` (intégration) ← `feature/lot-<a|b|c>-<sujet>` (travail).
- **Commits** : `feat(lot-b): grid search sur lr et dropout`, `fix(src): normalisation des images de test`, `docs: interprétation de la matrice de confusion`. Types autorisés : `feat`, `fix`, `docs`, `refactor`, `chore`, `test`.
- **Pull requests** : une PR par tâche de backlog terminée, vers `develop`, **relue par au moins un autre membre** (checklist en section 8). Pas de push direct sur `develop` ni `main`.
- **Notebooks** :
  - notebooks de travail (`A_`, `B_`, `C_`) : commiter **sans sorties** (`jupyter nbconvert --clear-output --inplace` ou `nbstripout`) pour garder des diffs lisibles ;
  - notebook final `Projet_5DEEP.ipynb` : commiter **avec sorties** (c'est le rendu) ;
  - jamais deux personnes sur le même `.ipynb` en même temps.
- **Fichiers lourds** : `data/`, `models/*` (sauf le meilleur modèle), `*.zip` ignorés. Si le meilleur modèle dépasse 100 Mo, le partager via un lien Drive noté dans le README.
- **Synchronisation** : point de coordination de 15 min tous les deux jours (avancement, blocages, décisions) ; suivi des tâches dans l'onglet *Projects* / *Issues* GitHub, une issue par ligne de backlog (`A-01`, `B-03`…).

## 7. Planning

Durée cible : **4 sprints**. Les dates sont à renseigner à la réunion de lancement.

| Sprint | Période | Objectif | Jalon de fin de sprint |
|---|---|---|---|
| **S0 — Lancement** (½ journée) | ___ | Décisions D1–D10, création du squelette de dépôt, `requirements.txt`, `src/config.py`, `src/data.py` (Dev A), issues créées | `load_datasets()` s'exécute avec succès sur le poste de chaque membre / Colab |
| **S1 — Fondations** | ___ → ___ | A : Q1–Q4 terminées, `baseline` sauvegardé. B : wrapper GridSearch + pipeline d'augmentation testés sur un mini-modèle. C : modèle pré-entraîné chargé et personnalisé, bibliographie CAM/Grad-CAM rédigée | `models/baseline` disponible ; `evaluate_model()` validé |
| **S2 — Cœur du projet** | ___ → ___ | B : GridSearch + augmentation + **meilleur modèle** choisi et sauvegardé (Q5–Q7). C : Q9 entraîné et comparé, CAM entraîné et visualisé (Bonus 1). A : photos Q8 collectées par tous, script `predict_image`, relecture des sections A | `models/best_model` disponible ; toutes les sections rédigées dans les notebooks de travail |
| **S3 — Consolidation & rendu** | ___ → ___ | B : Grad-CAM sur le meilleur modèle (Bonus 2). C + A : fusion dans `Projet_5DEEP.ipynb`, exécution complète de bout en bout (*Restart & Run All*), export PDF/HTML, relecture croisée orthographe/style, zip | **Archive `.zip` rendue** au moins 24 h avant la date limite |

Marge : conserver **une journée de sécurité** avant la date limite pour l'exécution complète du notebook final (un entraînement complet peut prendre plusieurs heures).

## 8. Définition de « terminé » (Definition of Done)

Une tâche est terminée lorsque **tous** les critères ci-dessous sont remplis :

- [ ] Le code s'exécute de bout en bout sans erreur à partir d'un noyau vierge.
- [ ] Il utilise le socle `src/` (pas de pipeline de données ou d'évaluation dupliqué).
- [ ] Chaque étape est **commentée et justifiée** dans une cellule Markdown : *pourquoi* ce choix, pas seulement *quoi*.
- [ ] Chaque résultat (courbe, tableau, matrice) est **interprété** en français rédigé : observation, signification, conclusion.
- [ ] Les graphiques ont un titre, des axes légendés et une légende.
- [ ] Les sources externes (articles CAM/Grad-CAM, documentation) sont **citées précisément** (auteurs, titre, année, URL/DOI).
- [ ] Le texte a été relu pour l'orthographe et la grammaire par **un autre membre**.
- [ ] La PR a été approuvée par au moins un autre membre et fusionnée dans `develop`.

## 9. Checklist du rendu final

- [ ] `Projet_5DEEP.ipynb` exécuté intégralement, sorties présentes, numérotation des cellules continue (1, 2, 3…).
- [ ] Export PDF **ou** HTML du même notebook, généré après la dernière exécution (`jupyter nbconvert --to html` ; PDF via navigateur si LaTeX indisponible).
- [ ] Page de garde : titre, noms des trois membres, date, framework utilisé.
- [ ] Sommaire suivant l'ordre des questions du sujet (Q1 → Q9, Bonus 1, Bonus 2), chaque section clairement titrée.
- [ ] Tableau récapitulatif final comparant tous les modèles (baseline, GridSearch, augmentation, pré-entraîné, CAM) avec les **mêmes métriques** sur le **même jeu de test**.
- [ ] Photos personnelles (Q8) incluses dans l'archive avec leurs prédictions.
- [ ] Meilleur modèle sauvegardé (fichier ou lien) + code de rechargement fonctionnel.
- [ ] Bibliographie complète.
- [ ] Archive `Projet_5DEEP.zip` contenant : notebook, export, `src/`, `photos/`, `requirements.txt`, `README.md` (et modèle si taille raisonnable).
- [ ] Vérification finale : le projet utilise **PyTorch** exclusivement (D1), aucun import `keras`/`tensorflow` — condition d'ajournement.

## 10. Risques et parades

| Risque | Impact | Parade |
|---|---|---|
| Utilisation d'un autre framework que PyTorch (D1) | Ajournement | Aucun import `keras` / `tensorflow` dans le projet ; vérification finale (section 9). |
| `GridSearchCV` trop long (CNN × combinaisons × folds) | Sprint S2 en retard | Grille réduite (2–3 hyperparamètres, 2–3 valeurs), `cv=3`, sous-échantillon du train, peu d'epochs ; `skorch` avec `device=DEVICE` ; justifier ce compromis dans le notebook. |
| Conflits Git sur les notebooks | Perte de travail | Un notebook par personne + fusion unique par Dev C ; sorties nettoyées avant commit. |
| Lot A en retard → B et C bloqués | Effet domino | Socle `src/` livré dès S0 ; B et C démarrent sur un mini-modèle jouet. |
| Notebook final impossible à ré-exécuter en entier (temps GPU) | Rendu incohérent | Entraînements lourds sauvegardés dans `models/` et rechargés ; garder une exécution « rapide » possible via un flag `FAST_RUN` dans `config.py`. |
| Rédaction bâclée en fin de projet | Perte de points | Rédaction faite **au fil de l'eau** dans les notebooks de travail (DoD), relecture croisée en S3. |
| Sources CAM/Grad-CAM non citées ou paraphrase d'IA | Pénalité explicite au sujet | Lire et citer les articles originaux (Zhou et al., 2016 ; Selvaraju et al., 2017) ; reformuler avec ses propres schémas et équations. |
