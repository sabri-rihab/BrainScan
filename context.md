# BrainScan AI — Détection des tumeurs cérébrales par CNN

---

## Contexte du projet

**BrainScan AI** est une startup marocaine innovante spécialisée dans les technologies d'imagerie médicale assistée par l'intelligence artificielle. Elle souhaite améliorer le diagnostic précoce des tumeurs cérébrales grâce à une approche automatisée et précise.

L'objectif du projet est de développer une application intelligente capable de :

- Analyser et classer les images IRM du cerveau pour détecter la présence de cancer.
- Assister les médecins dans l'interprétation rapide et fiable des résultats.
- Optimiser le temps de diagnostic tout en réduisant les erreurs humaines.

Cette initiative s'inscrit dans une démarche visant à allier **santé, innovation et IA** pour renforcer la qualité et la rapidité des soins médicaux au Maroc.

---

## 1. Chargement et Prétraitement du Dataset

- Importer les bibliothèques nécessaires.
- Charger les images du dataset et vérifier leurs extensions (`.jpeg`, `.jpg`, `.bmp`, `.png`). Supprimer celles qui ne correspondent pas.
- Utiliser un bloc `try-except` pour gérer les erreurs lors du chargement.
- Explorer les classes du dataset (les noms des dossiers représentent les classes).
- Mélanger les images de toutes les classes dans deux listes :
  - une pour les images
  - une autre pour les étiquettes
  - Chaque image et son label doivent avoir le **même indice** dans les deux listes.
- Redimensionner les images à une taille fixe (ex. `224×224`) à l'aide de la bibliothèque OpenCV.
- Une fois toutes les images et leurs labels chargés, convertir les listes images et étiquettes en tableaux **NumPy** pour les rendre exploitables par le modèle CNN.
- Afficher graphiquement le nombre d'images dans chaque classe.
- Montrer un échantillon d'images pour chaque classe.
- Vérifier l'équilibre entre les classes et appliquer un rééquilibrage si nécessaire.
- Transformer les étiquettes textuelles en format numérique (ex. avec `LabelEncoder` ou `to_categorical`).
- Diviser les données en ensembles d'entraînement et de test.
- Normaliser les valeurs des pixels dans la plage `[0, 1]`.

---

## 2. Conception du Modèle CNN

- Définir l'architecture CNN (`Conv2D` + `MaxPooling` + `Dropout` + `Dense`) avec des paramètres optimaux.
- Choisir les meilleures fonctions d'activation pour les couches cachées et la couche de sortie.
- Compiler le modèle (ex. : Optimiseur `Adam`, fonction de perte `categorical_crossentropy`).
- Utiliser les fonctions `model.summary()` et `plot_model()` pour vérifier l'architecture du modèle.
- Déterminer les hyperparamètres (taux d'apprentissage, nombre d'époques, taille de batch).
- Utiliser la bibliothèque `time` pour mesurer la durée de l'entraînement.

---

## 3. Entraînement et Évaluation du Modèle

- Lancer l'entraînement avec `model.fit()`.
- Sauvegarder le meilleur modèle à l'aide du callback `ModelCheckpoint`.
- Évaluer les performances sur l'ensemble de test.
- Visualiser les courbes d'apprentissage (accuracy / loss).
- Générer une matrice de confusion et un rapport de classification.
- Afficher des exemples de prédictions correctes et incorrectes.

---

## 4. Déploiement / Utilisation

- Créer une fonction `predict_image(path)` pour tester une image.
- Sauvegarder le modèle entraîné (`.h5` ou `.pt`).
- Développer une interface avec **Streamlit** pour tester les prédictions.
