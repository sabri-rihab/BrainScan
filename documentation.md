# BrainScan AI — Documentation des Concepts

---

## 1. Les Bibliothèques (Libraries)

**Ce que c'est :** Des outils externes que tu importes pour ne pas tout coder from scratch. Chaque bibliothèque a un rôle précis dans le pipeline.

**Ce que tu dois comprendre :**

- `os` — naviguer dans les dossiers, lister les fichiers, construire des chemins. Fonctions clés : `os.listdir()`, `os.path.join()`, `os.path.splitext()`
- `numpy` — toute image en mémoire est un tableau NumPy (ndarray). Comprendre les shapes : `(hauteur, largeur, canaux)` ex. `(224, 224, 3)`. Fonctions clés : `np.array()`, `np.zeros()`, `np.argmax()`
- `cv2` (OpenCV) — lit les images depuis le disque, les redimensionne, convertit les espaces de couleur. Important : OpenCV lit en **BGR** pas RGB. Fonctions clés : `cv2.imread()`, `cv2.resize()`, `cv2.cvtColor()`
- `matplotlib` — affiche des images et des graphiques dans le notebook. Fonctions clés : `plt.imshow()`, `plt.bar()`, `plt.show()`
- `sklearn` — encodage des labels, split du dataset, métriques d'évaluation. Fonctions clés : `LabelEncoder`, `train_test_split`, `confusion_matrix`, `classification_report`
- `tensorflow / keras` — construit, compile, entraîne et évalue le modèle CNN. Keras est l'API haut niveau de TensorFlow. Fonctions clés : `Sequential`, `Conv2D`, `Dense`, `model.fit()`, `to_categorical`
- `time` — mesure la durée d'entraînement avec `time.time()` avant et après `model.fit()`
- `streamlit` — crée une interface web interactive en Python pur pour tester le modèle

---

## 2. Chargement des Images

**Ce que c'est :** Lire les fichiers images depuis les dossiers du dataset et les charger en mémoire sous forme de tableaux NumPy.

**Ce que tu dois comprendre :**

- La structure du dataset est `dataset/nom_classe/image.jpg` — le nom du dossier = le label
- `cv2.imread(chemin)` retourne un tableau NumPy ou `None` si le fichier est illisible — toujours vérifier que le résultat n'est pas `None`
- Pourquoi filtrer les extensions : les datasets contiennent souvent des fichiers cachés, `.txt`, ou corrompus qui font planter `imread`
- Toujours appeler `.lower()` sur l'extension avant de vérifier — `.JPG` et `.jpg` sont le même format mais Python les voit différemment
- `os.path.splitext('image.jpg')` retourne `('image', '.jpg')` — tu prends l'index `[1]` pour l'extension
- Le bloc `try-except` : `imread` retourne `None` silencieusement sur les fichiers corrompus, mais d'autres erreurs (permissions, chemins invalides) lèvent des exceptions — le try-except les attrape sans arrêter tout le chargement

---

## 3. Exploration des Classes (EDA)

**Ce que c'est :** Comprendre la composition du dataset avant de toucher aux données. Détecter les problèmes tôt.

**Ce que tu dois comprendre :**

- Chaque dossier = une classe. `os.listdir(dataset_path)` te donne directement les noms des classes
- Compter les images par classe pour détecter le **déséquilibre de classes** — si une classe a 3000 images et une autre 300, le modèle sera biaisé vers la classe majoritaire
- Afficher un graphique en barres (`plt.bar`) pour visualiser la distribution — une distribution équilibrée est idéale
- Afficher des échantillons d'images par classe pour vérifier visuellement la qualité des données (images floues, mal orientées, etc.)
- Convertir BGR → RGB avec `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` avant `plt.imshow()` — sinon les couleurs sont fausses

---

## 4. Rééquilibrage des Classes

**Ce que c'est :** Corriger un déséquilibre entre les classes pour que le modèle n'apprenne pas à favoriser une classe.

**Ce que tu dois comprendre :**

- **Oversampling** : dupliquer ou générer des images dans les classes minoritaires (ex. avec `ImageDataGenerator`)
- **Undersampling** : réduire les images dans les classes majoritaires — risque de perdre de l'information
- **Class weights** : donner plus de poids aux classes minoritaires lors de l'entraînement avec le paramètre `class_weight` dans `model.fit()`
- Pourquoi c'est critique en médical : si 90% des images sont "pas de tumeur", le modèle peut atteindre 90% d'accuracy en prédisant toujours "pas de tumeur" — ce qui est inutile et dangereux

---

## 5. Listes Images & Labels Synchronisées

**Ce que c'est :** Stocker toutes les images dans une liste et leurs labels correspondants dans une autre, en garantissant que `images[i]` et `labels[i]` correspondent toujours au même échantillon.

**Ce que tu dois comprendre :**

- C'est le contrat fondamental du dataset — briser cette synchronisation corrompt tout
- On utilise `enumerate(class_names)` pour obtenir simultanément l'index (label numérique) et le nom de la classe
- `images.append(img)` et `labels.append(label)` doivent être dans la **même itération** de boucle
- Garder un dictionnaire de correspondance `{0: 'glioma', 1: 'meningioma', ...}` pour décoder les prédictions plus tard
- Vérifier à la fin : `len(images) == len(labels)` doit être `True`

---

## 6. Redimensionnement des Images

**Ce que c'est :** Forcer toutes les images à la même taille fixe pour pouvoir les empiler dans un seul tableau NumPy.

**Ce que tu dois comprendre :**

- Les CNN exigent une taille d'entrée fixe — impossible de mélanger des images de tailles différentes dans un même batch
- `cv2.resize(img, (largeur, hauteur))` — attention : OpenCV prend `(largeur, hauteur)` mais NumPy stocke en `(hauteur, largeur, canaux)` — l'ordre est inversé
- `224×224` est la taille standard (utilisée par VGG16, ResNet) — plus grand = plus de détails mais plus de mémoire et de calcul
- Le redimensionnement change le ratio d'aspect si l'image n'est pas carrée — acceptable pour les images médicales à cette taille

---

## 7. Conversion en Tableaux NumPy

**Ce que c'est :** Transformer les listes Python en tableaux NumPy multidimensionnels exploitables par le modèle.

**Ce que tu dois comprendre :**

- `X = np.array(images)` produit un tableau 4D de shape `(N, 224, 224, 3)` — N images, 224×224 pixels, 3 canaux (RGB)
- `y = np.array(labels)` produit un tableau 1D de shape `(N,)`
- Cette conversion est obligatoire — Keras et sklearn ne travaillent pas avec des listes Python brutes
- Vérifier avec `X.shape` et `y.shape` après conversion

---

## 8. Encodage des Labels

**Ce que c'est :** Convertir les noms de classes (strings) en nombres que le modèle peut calculer.

**Ce que tu dois comprendre :**

- **LabelEncoder** : `'glioma'` → `0`, `'meningioma'` → `1`, etc. Retourne des entiers.
- **to_categorical** : convertit les entiers en vecteurs one-hot. `0` → `[1,0,0,0]`, `1` → `[0,1,0,0]`
- Pourquoi le one-hot : avec `categorical_crossentropy`, le modèle compare un vecteur de probabilités à un vecteur one-hot — les entiers bruts ne fonctionnent pas avec cette loss
- Alternative : utiliser `sparse_categorical_crossentropy` qui accepte directement les entiers (pas besoin de one-hot)
- `y_encoded.shape` doit être `(N, num_classes)` après one-hot

---

## 9. Split Train / Test

**Ce que c'est :** Diviser le dataset en deux parties — une pour entraîner le modèle, une pour l'évaluer honnêtement.

**Ce que tu dois comprendre :**

- **Train set** (~80%) : le modèle apprend dessus
- **Test set** (~20%) : utilisé uniquement à la fin pour mesurer les vraies performances — jamais pendant l'entraînement
- `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`
- `stratify=y` : garantit que chaque split a les mêmes proportions de classes que le dataset original — essentiel avec des classes déséquilibrées
- `random_state=42` : rend le split reproductible — le même split à chaque exécution
- Optionnel : créer un 3ème split **validation** pour monitorer l'overfitting pendant l'entraînement

---

## 10. Normalisation des Pixels

**Ce que c'est :** Mettre les valeurs des pixels à l'échelle [0, 1] au lieu de [0, 255].

**Ce que tu dois comprendre :**

- `X = X / 255.0` — division simple et efficace
- Pourquoi c'est nécessaire : des valeurs d'entrée grandes (0–255) causent de grands gradients → entraînement instable. Des valeurs petites (0–1) stabilisent les calculs.
- Appliquer **exactement la même normalisation** sur train, validation et test — ne jamais recalculer une nouvelle échelle sur le test set
- Vérifier : `X.min()` doit être `0.0` et `X.max()` doit être `1.0` après normalisation

---

## 11. Architecture CNN

**Ce que c'est :** La structure du réseau de neurones — la séquence de couches qui transforme une image en prédiction de classe.

**Ce que tu dois comprendre :**

- **Conv2D** : applique des filtres (kernels) qui glissent sur l'image pour détecter des features (bords, textures, formes). Chaque filtre produit une feature map. Paramètres : `filters`, `kernel_size`, `activation`, `input_shape`
- **MaxPooling2D** : réduit la taille spatiale des feature maps (ex. 224→112) en gardant la valeur max dans chaque fenêtre. Réduit le calcul et rend le modèle moins sensible à la position exacte des features.
- **Dropout** : désactive aléatoirement un pourcentage de neurones pendant l'entraînement. Empêche l'overfitting en forçant le réseau à ne pas dépendre d'un seul neurone. Automatiquement désactivé en prédiction.
- **Flatten** : convertit les feature maps 3D en vecteur 1D pour les passer aux couches Dense
- **Dense** : couche fully connected — chaque neurone est connecté à tous les neurones précédents. C'est là que la classification finale se fait.
- **Fonctions d'activation couches cachées** : `ReLU` — met les valeurs négatives à 0, introduit la non-linéarité nécessaire pour apprendre des patterns complexes
- **Fonction d'activation couche de sortie** : `Softmax` — convertit les scores bruts en probabilités qui somment à 1. La classe avec la probabilité la plus haute est la prédiction.
- `model.summary()` — affiche toutes les couches, leurs shapes de sortie et le nombre de paramètres
- `plot_model()` — génère un schéma visuel de l'architecture

---

## 12. Compilation du Modèle

**Ce que c'est :** Configurer comment le modèle va apprendre — quel algorithme met à jour les poids, comment mesurer l'erreur, quoi surveiller.

**Ce que tu dois comprendre :**

- **Optimiseur Adam** : algorithme qui ajuste les poids pour minimiser la loss. Adam adapte automatiquement le taux d'apprentissage par paramètre — c'est le choix standard.
- **Taux d'apprentissage (learning rate)** : contrôle la taille de chaque mise à jour des poids. Trop grand = entraînement instable. Trop petit = convergence très lente. Valeur par défaut Adam : `0.001`
- **Loss `categorical_crossentropy`** : mesure l'écart entre les probabilités prédites et les labels one-hot. À utiliser quand les labels sont one-hot encodés.
- **Loss `sparse_categorical_crossentropy`** : même chose mais pour les labels entiers (pas de one-hot)
- **Metrics `accuracy`** : pourcentage de prédictions correctes — affiché après chaque epoch pour train et validation

---

## 13. Hyperparamètres

**Ce que c'est :** Les paramètres que tu choisis avant l'entraînement et qui contrôlent comment le modèle apprend.

**Ce que tu dois comprendre :**

- **Epochs** : nombre de fois que le modèle voit tout le dataset d'entraînement. Plus d'epochs = plus d'apprentissage, mais risque d'overfitting au-delà d'un certain point.
- **Batch size** : nombre d'images traitées avant chaque mise à jour des poids. Valeurs courantes : 16, 32, 64. Petit batch = mises à jour plus fréquentes mais plus bruitées. Grand batch = plus stable mais plus de mémoire.
- **Nombre de filtres Conv2D** : plus de filtres = plus de features détectées mais plus de calcul
- **Taux de Dropout** : entre 0.2 et 0.5 typiquement. Trop élevé = underfitting.
- Ces paramètres se trouvent par expérimentation — il n'y a pas de valeur universelle parfaite

---

## 14. Entraînement avec model.fit()

**Ce que c'est :** L'appel qui lance l'entraînement — le modèle voit les données, calcule l'erreur, et met à jour ses poids répétitivement.

**Ce que tu dois comprendre :**

- `history = model.fit(X_train, y_train, epochs=..., batch_size=..., validation_data=(X_val, y_val))`
- L'objet `history` contient les valeurs de loss et accuracy par epoch — indispensable pour tracer les courbes
- Toujours passer `validation_data` — sans ça, impossible de détecter l'overfitting pendant l'entraînement
- **Overfitting** : la loss d'entraînement baisse mais la loss de validation remonte → le modèle mémorise au lieu d'apprendre. Solutions : Dropout, plus de données, early stopping.
- **Underfitting** : les deux losses restent élevées → le modèle est trop simple ou l'entraînement trop court

---

## 15. ModelCheckpoint (Callback)

**Ce que c'est :** Un mécanisme qui sauvegarde automatiquement le modèle à chaque fois qu'une métrique s'améliore pendant l'entraînement.

**Ce que tu dois comprendre :**

- `ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)`
- `monitor='val_accuracy'` : surveille la précision sur la validation — sauvegarde uniquement quand elle s'améliore
- `save_best_only=True` : ne garde que le meilleur modèle, pas le dernier
- Pourquoi c'est important : l'entraînement peut dégrader le modèle après un certain nombre d'epochs (overfitting) — le checkpoint garde la meilleure version
- Se passe en paramètre `callbacks=[checkpoint]` dans `model.fit()`

---

## 16. Courbes d'Apprentissage

**Ce que c'est :** Des graphiques qui montrent l'évolution de la loss et de l'accuracy sur les epochs, pour diagnostiquer comment le modèle apprend.

**Ce que tu dois comprendre :**

- Tracer `history.history['accuracy']` vs `history.history['val_accuracy']`
- Tracer `history.history['loss']` vs `history.history['val_loss']`
- **Bon entraînement** : les deux courbes (train et val) descendent ensemble et se stabilisent
- **Overfitting** : val_loss remonte pendant que train_loss continue de baisser — les courbes divergent
- **Underfitting** : les deux courbes restent hautes et ne convergent pas

---

## 17. Matrice de Confusion & Rapport de Classification

**Ce que c'est :** Des outils pour analyser en détail où le modèle se trompe, classe par classe.

**Ce que tu dois comprendre :**

- **Matrice de confusion** : tableau N×N (N = nombre de classes). Lignes = classes réelles, colonnes = classes prédites. La diagonale = prédictions correctes. Hors diagonale = erreurs.
- **Precision** : parmi toutes les images prédites comme classe X, combien le sont vraiment
- **Recall (Sensibilité)** : parmi toutes les images réellement de classe X, combien le modèle en a détecté — critique en médical : rater une tumeur est dangereux
- **F1-score** : moyenne harmonique de precision et recall — bon indicateur unique pour les classes déséquilibrées
- **Support** : nombre d'images réelles par classe dans le test set
- `classification_report(y_true, y_pred, target_names=class_names)` — affiche tout ça en une fois
- `y_pred_classes = np.argmax(model.predict(X_test), axis=1)` — convertit les probabilités en indices de classes

---

## 18. Sauvegarde & Chargement du Modèle

**Ce que c'est :** Persister le modèle entraîné sur le disque pour le réutiliser sans réentraîner.

**Ce que tu dois comprendre :**

- `model.save('model.h5')` — sauvegarde architecture + poids + état de l'optimiseur dans un seul fichier
- `tf.keras.models.load_model('model.h5')` — recharge le modèle complet
- Format `.h5` vs SavedModel : `.h5` = un fichier, SavedModel = un dossier — les deux fonctionnent avec Keras
- Sauvegarder aussi le dictionnaire de labels `{0: 'glioma', ...}` (avec `json` ou `pickle`) — sans ça, les prédictions sont des chiffres illisibles

---

## 19. Fonction predict_image(path)

**Ce que c'est :** Une fonction réutilisable qui prend le chemin d'une image, la prétraite exactement comme les données d'entraînement, et retourne la classe prédite.

**Ce que tu dois comprendre :**

- Le prétraitement dans cette fonction doit être **identique** à celui du training : même resize, même conversion BGR→RGB, même normalisation `/255.0`
- `model.predict()` retourne un tableau de probabilités — utiliser `np.argmax()` pour obtenir l'index de la classe prédite
- Ajouter une dimension batch avec `np.expand_dims(img, axis=0)` — le modèle attend un input de shape `(1, 224, 224, 3)` pas `(224, 224, 3)`
- Retourner le nom de la classe (pas juste l'index) en utilisant le dictionnaire de labels

---

## 20. Interface Streamlit

**Ce que c'est :** Un framework Python qui permet de créer une application web interactive sans HTML/CSS/JS — idéal pour déployer un modèle ML rapidement.

**Ce que tu dois comprendre :**

- `st.file_uploader()` — widget pour uploader une image depuis le navigateur
- `st.image()` — affiche l'image uploadée
- `st.write()` / `st.success()` — affiche le résultat de la prédiction
- Le flux : l'utilisateur uploade une image → tu la lis → tu appelles `predict_image()` → tu affiches la classe prédite
- Lancer avec `streamlit run app.py` dans le terminal
- Le modèle doit être chargé **une seule fois** au démarrage de l'app (pas à chaque prédiction) — utiliser `@st.cache_resource` pour ça
