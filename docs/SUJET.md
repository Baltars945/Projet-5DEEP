# Sujet de l'examen — Classification d'images (Intel Image Classification)

> Document de référence : consignes officielles du formateur, reproduites intégralement.
> Toute décision de projet doit pouvoir être justifiée au regard de ce sujet.

## Modalités

Pour rappel, cet examen est à réaliser par **groupes d'au plus trois étudiants**.

Selon les consignes de votre formateur, que vous respecterez scrupuleusement, vous utiliserez la librairie **Keras** ou la librairie **PyTorch**. Le choix n'est absolument pas du ressort des étudiants et tout projet qui contreviendrait à cette directive sera ajourné sans contestation possible.

Vous devez utiliser les données disponibles à cette adresse : <https://huggingface.co/datasets/sfarrukhm/intel-image-classification>, le dataset étant constitué d'un **jeu d'entraînement** et d'un **jeu de test**.

Il s'agit d'un problème de **classification d'images avec six classes** :

1. Building
2. Forest
3. Glacier
4. Mountain
5. Sea
6. Street

Vous devrez développer **"from scratch"** vos propres réseaux de neurones pour résoudre ce problème.

## Rendu

Votre rendu se fera sous la forme d'une **archive au format `.zip`** contenant :

- un **notebook Jupyter** (code source Python) ;
- son **export au format PDF (ou HTML)**.

Tout projet ne comportant pas ces deux éléments sera ajourné sans contestation possible.

Toutes les étapes de votre projet devront être **commentées et justifiées**. Tous les résultats devront être **interprétés**. La qualité de la rédaction (style, grammaire et orthographe) sera prise en compte. **Un rendu professionnel est attendu.**

## Questions

1. Charger et explorer l'ensemble des données.
2. Ce dataset est-il équilibré ? Est-il nécessaire de rééquilibrer les données ? Le faire si besoin est.
3. Construire un réseau de neurones convolutif pour résoudre ce problème de classification. Il devra contenir au minimum les éléments suivants : couches de convolution, couche de "pooling", "dropout", couches cachées complètement connectées. Vous êtes libres d'ajouter d'autres éléments.
4. Entraîner le modèle construit à la question précédente et mesurer sa performance.
5. Faire une recherche de meilleurs hyperparamètres avec la fonction `GridSearchCV`.
6. Utiliser une technique d'augmentation d'images. Les résultats de vos modèles s'en trouvent-ils améliorés ? Était-ce prévisible ?
7. Sauvegarder votre meilleur modèle.
8. Utiliser votre meilleur modèle avec des photos prises pour l'occasion où vous représenterez différents signes.
9. Choisir un des modèles pré-entraînés (par exemple ResNet). Le charger et le personnaliser avec des couches de convolution et complètement connectées. L'entraîner et mesurer sa performance. La comparer avec celles de votre meilleur modèle.

## Bonus

**Bonus 1 — CAM (Class Activation Mapping).** Se documenter sur la technique du CAM, "Class Activation Mapping". En présenter le but et le principe de fonctionnement en citant précisément ses sources. Ne pas se contenter de régurgiter les informations fournies par une IAG. Cette méthode nécessite une structure particulière de réseau. En construire un y répondant, l'entraîner, évaluer sa performance, puis appliquer cette technique afin de visualiser sur quelques images les zones ayant contribué à leur classification. On utilisera en particulier des images pouvant potentiellement appartenir à deux classes, par exemple Street et Building, afin de mettre en évidence les zones ayant déterminé la prédiction.

**Bonus 2 — Grad-CAM (Gradient-weighted Class Activation Mapping).** Se documenter sur la technique du Grad-CAM, "Gradient-weighted Class Activation Mapping". En présenter le but et le principe de fonctionnement en citant précisément ses sources. Ne pas se contenter de régurgiter les informations fournies par une IAG. Cette méthode diffère en particulier de la précédente sur le fait qu'elle n'impose pas une structure de réseau. L'appliquer à votre meilleur modèle afin de visualiser sur quelques images les zones ayant contribué à leur classification. On utilisera en particulier des images pouvant potentiellement appartenir à deux classes, par exemple Street et Building, afin de mettre en évidence les zones ayant déterminé la prédiction.

---

### Notes d'interprétation

- La question 8 mentionne « différents signes » : formulation vraisemblablement héritée d'un autre sujet. Interprétation retenue : **photos prises pour l'occasion, représentant les six catégories de scènes** (bâtiment, forêt, glacier/neige, montagne, mer, rue). À confirmer auprès du formateur.
- Framework imposé par le formateur : **PyTorch** (décision D1 dans [GESTION_DE_PROJET.md](GESTION_DE_PROJET.md)).
