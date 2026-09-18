"""Crée les labels, jalons (sprints), issues et le tableau GitHub Projects
à partir des backlogs Markdown de docs/backlog/.

Prérequis :
    winget install GitHub.cli          (ou https://cli.github.com)
    gh auth login --scopes repo,project

Usage (depuis la racine du dépôt) :
    python scripts/create_github_issues.py            # crée tout
    python scripts/create_github_issues.py --dry-run  # affiche sans rien créer
    python scripts/create_github_issues.py --no-project   # issues seulement

Le script est idempotent sur les labels et jalons (ignore les doublons) mais
PAS sur les issues : ne le lancer qu'une fois, ou supprimer les issues avant.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "Baltars945/Projet-5DEEP"
OWNER = REPO.split("/")[0]
PROJECT_OWNER = None  # compte authentifie, resolu au lancement (gh api user)
PROJECT_TITLE = "Projet 5DEEP — Backlog"
BACKLOG_DIR = Path(__file__).resolve().parent.parent / "docs" / "backlog"

LOTS = {
    "A": ("lot-a", "Données & CNN de base", "LOT_A_donnees_cnn.md"),
    "B": ("lot-b", "Optimisation & Grad-CAM", "LOT_B_optimisation_gradcam.md"),
    "C": ("lot-c", "Transfert, CAM & rendu", "LOT_C_transfert_cam_rendu.md"),
}
LABELS = {
    "lot-a": ("1d76db", "Lot A — Données & CNN de base"),
    "lot-b": ("0e8a16", "Lot B — Optimisation & Grad-CAM"),
    "lot-c": ("d93f0b", "Lot C — Transfert, CAM & rendu"),
    "P0": ("b60205", "Bloquant pour les autres lots"),
    "P1": ("fbca04", "Requis pour le rendu"),
    "P2": ("c5def5", "Amélioration / bonus"),
}
MILESTONES = {
    "S0": "S0 — Lancement : décisions D1–D10, socle src/",
    "S1": "S1 — Fondations : baseline, outillage, bibliographie",
    "S2": "S2 — Cœur du projet : meilleur modèle, Q9, CAM, photos",
    "S3": "S3 — Consolidation & rendu : Grad-CAM, assemblage, export, zip",
}

ROW_RE = re.compile(r"^\|\s*([ABC]-\d{2})\s*\|")


def gh(*args: str, dry_run: bool = False, capture: bool = True) -> str:
    cmd = ["gh", *args]
    if dry_run:
        print("  $", " ".join(cmd))
        return ""
    res = subprocess.run(cmd, capture_output=capture, text=True, encoding="utf-8")
    if res.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{res.stderr}")
    return res.stdout.strip()


def parse_backlog(path: Path) -> list[dict]:
    """Extrait les lignes `| X-NN | tâche | prio | estim | sprint | critère |`."""
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not ROW_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        task_id, title, prio, estim, sprint, criteria = cells[:6]
        tasks.append(
            dict(
                id=task_id,
                title=title,
                priority=prio,
                estimate=estim,
                sprint=sprint.split("–")[0].split("-")[0].strip(),  # "S1–S2" -> "S1"
                sprint_raw=sprint,
                criteria=criteria,
            )
        )
    return tasks


def ensure_labels(dry_run: bool) -> None:
    print("Labels…")
    for name, (color, desc) in LABELS.items():
        try:
            gh("label", "create", name, "--repo", REPO, "--color", color,
               "--description", desc, "--force", dry_run=dry_run)
        except RuntimeError as e:
            print("  ! label", name, ":", e.splitlines()[-1])


def ensure_milestones(dry_run: bool) -> dict[str, str]:
    print("Jalons (sprints)…")
    existing = {}
    if not dry_run:
        out = gh("api", f"repos/{REPO}/milestones?state=all&per_page=100")
        existing = {m["title"]: m["title"] for m in json.loads(out)}
    for title in MILESTONES.values():
        if title in existing:
            continue
        gh("api", f"repos/{REPO}/milestones", "-f", f"title={title}", dry_run=dry_run)
    return MILESTONES


def ensure_project(dry_run: bool) -> str | None:
    """Cree le tableau sous le compte authentifie et le lie au depot."""
    print("Tableau Projects…")
    if dry_run:
        print(f"  $ gh project create --owner @me --title \"{PROJECT_TITLE}\"")
        return None
    out = gh("project", "list", "--owner", PROJECT_OWNER, "--format", "json")
    number = None
    for p in json.loads(out).get("projects", []):
        if p["title"] == PROJECT_TITLE:
            number = str(p["number"])
    if number is None:
        out = gh("project", "create", "--owner", PROJECT_OWNER, "--title", PROJECT_TITLE, "--format", "json")
        number = str(json.loads(out)["number"])
    try:
        gh("project", "link", number, "--owner", PROJECT_OWNER, "--repo", REPO)
    except RuntimeError as e:
        print("  ! liaison au depot impossible (le tableau doit appartenir au proprietaire du depot) :", str(e).strip().splitlines()[-1])
    return number


def issue_body(t: dict, lot_label: str, lot_name: str, source: str) -> str:
    return (
        f"**Lot** : {lot_name} (`{lot_label}`)  \n"
        f"**Priorité** : {t['priority']} · **Estimation** : {t['estimate']} ½ j · "
        f"**Sprint** : {t['sprint_raw']}\n\n"
        f"## Critère d'acceptation\n\n{t['criteria']}\n\n"
        f"## Definition of Done\n\n"
        f"- [ ] Code exécutable de bout en bout, basé sur `src/`\n"
        f"- [ ] Étapes commentées et justifiées, résultats interprétés\n"
        f"- [ ] Relu par un autre membre, PR fusionnée dans `develop`\n\n"
        f"_Source : `docs/backlog/{source}` — voir aussi `docs/GESTION_DE_PROJET.md`._"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-project", action="store_true")
    args = ap.parse_args()

    global PROJECT_OWNER
    if not args.dry_run:
        try:
            gh("auth", "status")
            PROJECT_OWNER = gh("api", "user", "--jq", ".login")
        except (RuntimeError, FileNotFoundError):
            print("gh introuvable ou non authentifié : `gh auth login --scopes repo,project`")
            return 1

    ensure_labels(args.dry_run)
    milestones = ensure_milestones(args.dry_run)
    project = None if args.no_project else ensure_project(args.dry_run)

    total = 0
    for lot, (label, lot_name, filename) in LOTS.items():
        tasks = parse_backlog(BACKLOG_DIR / filename)
        print(f"Lot {lot} : {len(tasks)} tâches")
        for t in tasks:
            title = f"[{t['id']}] {t['title']}"
            if len(title) > 200:
                title = title[:197] + "…"
            cmd = [
                "issue", "create", "--repo", REPO,
                "--title", title,
                "--body", issue_body(t, label, lot_name, filename),
                "--label", f"{label},{t['priority']}",
            ]
            ms = milestones.get(t["sprint"])
            if ms:
                cmd += ["--milestone", ms]
            url = gh(*cmd, dry_run=args.dry_run)
            if project and url:
                gh("project", "item-add", project, "--owner", PROJECT_OWNER, "--url", url)
            print("  +", title[:90], "->", url or "(dry-run)")
            total += 1

    print(f"\n{total} issues {'simulées' if args.dry_run else 'créées'}.")
    if project:
        print(f"Tableau : https://github.com/users/{PROJECT_OWNER}/projects/{project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
