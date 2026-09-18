# Projet-5DEEP — Classification d'images (Intel Image Classification)

Projet d'examen 5DEEP : classification de scènes en six classes (Building, Forest, Glacier, Mountain, Sea, Street) à partir du dataset [sfarrukhm/intel-image-classification](https://huggingface.co/datasets/sfarrukhm/intel-image-classification), avec des réseaux convolutifs développés *from scratch* en **PyTorch** (framework imposé par le formateur), une recherche d'hyperparamètres, de l'augmentation d'images, un modèle pré-entraîné personnalisé et des visualisations CAM / Grad-CAM.

## Documentation

| Document | Contenu |
|---|---|
| [docs/SUJET.md](docs/SUJET.md) | Consignes officielles de l'examen (référence). |
| [docs/GESTION_DE_PROJET.md](docs/GESTION_DE_PROJET.md) | Découpage en 3 lots parallèles, décisions techniques, contrat d'interface `src/`, workflow Git, planning par sprints, Definition of Done, checklist de rendu, risques. |
| [docs/backlog/LOT_A_donnees_cnn.md](docs/backlog/LOT_A_donnees_cnn.md) | Lot A — Données, exploration, CNN de base (Q1–Q4, socle commun). |
| [docs/backlog/LOT_B_optimisation_gradcam.md](docs/backlog/LOT_B_optimisation_gradcam.md) | Lot B — GridSearchCV, augmentation, meilleur modèle, Grad-CAM (Q5–Q7, Bonus 2). |
| [docs/backlog/LOT_C_transfert_cam_rendu.md](docs/backlog/LOT_C_transfert_cam_rendu.md) | Lot C — Modèle pré-entraîné, CAM, photos personnelles, assemblage et export (Q8, Q9, Bonus 1). |

## Suivi des tâches sur GitHub (Issues + Projects)

Les backlogs peuvent être importés automatiquement en issues GitHub (labels par lot et priorité, jalons par sprint) et ajoutés à un tableau *Projects* :

```bash
winget install GitHub.cli                 # une seule fois
gh auth login --scopes repo,project       # une seule fois
python scripts/create_github_issues.py --dry-run   # vérifier
python scripts/create_github_issues.py             # créer (à lancer une seule fois)
```

## Démarrage rapide (après le sprint S0)

```bash
git clone git@github.com:Baltars945/Projet-5DEEP.git
cd Projet-5DEEP
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

## Workflow Git (résumé)

`main` (releases, protégée) ← `develop` (intégration) ← `feature/lot-<a|b|c>-<sujet>`.
Commits au format [Conventional Commits](https://www.conventionalcommits.org/fr/) (`feat`, `fix`, `docs`, `refactor`, `chore`, `test`), une PR par tâche, relue par un autre membre. Détails dans `docs/GESTION_DE_PROJET.md` §6.

## Rendu

Archive `Projet_5DEEP.zip` contenant `notebooks/Projet_5DEEP.ipynb` (exécuté) et son export HTML/PDF. Checklist complète : `docs/GESTION_DE_PROJET.md` §9.
