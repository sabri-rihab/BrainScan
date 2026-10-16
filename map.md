# BrainScan AI — Project Task Map

---

## BIG STEPS (Project Overview)

1. Import all libraries
2. Load images from folders (filter bad extensions, handle errors)
3. Explore classes (count images per class, visualize distribution)
4. Build synchronized image + label lists
5. Resize all images to fixed size (224×224)
6. Convert lists to NumPy arrays
7. Visualize class distribution + sample images
8. Check class balance → rebalance if needed
9. Encode labels (text → numbers → one-hot)
10. Split data into Train / Test sets
11. Normalize pixel values to [0, 1]
12. Build CNN architecture (Conv2D + Pooling + Dropout + Dense)
13. Compile model (optimizer + loss + metrics)
14. Train model with model.fit() + save best with ModelCheckpoint
15. Evaluate model (accuracy, confusion matrix, classification report)
16. Visualize learning curves + correct/incorrect predictions
17. Create predict_image() function
18. Save trained model to disk (.h5)
19. Build Streamlit interface

---

---

## STEP 1 — Import Libraries

**Purpose:** Load every tool needed before writing any logic. If an import fails you know immediately, before running anything else.

### Tasks:

- [ ] **Import `os`**
  - Why: navigate folders, list files, build file paths
  - Know: `os.listdir()` returns file/folder names in a directory. `os.path.join()` builds a full path safely. `os.path.splitext('file.jpg')` returns `('file', '.jpg')`.

- [ ] **Import `numpy as np`**
  - Why: every image in memory is a NumPy array. All math on images uses NumPy.
  - Know: what an ndarray is. Shape `(224, 224, 3)` means 224 rows, 224 columns, 3 color channels. `np.array()` converts a list to an array. `np.argmax()` returns the index of the highest value.

- [ ] **Import `cv2`**
  - Why: reads images from disk, resizes them, converts color spaces
  - Know: `cv2.imread()` returns a NumPy array or `None` on failure. OpenCV reads images in **BGR** not RGB. `cv2.resize(img, (w, h))` resizes. `cv2.cvtColor()` converts color spaces.

- [ ] **Import `matplotlib.pyplot as plt`**
  - Why: display images and plots inside the notebook
  - Know: `plt.imshow()` shows an image. `plt.bar()` draws a bar chart. `plt.show()` renders the plot. `plt.subplot()` shows multiple images in a grid.

- [ ] **Import from `sklearn`**
  - `LabelEncoder` — converts string labels to integers
  - `train_test_split` — splits X and y into train/test subsets
  - `confusion_matrix` — builds the N×N error table
  - `classification_report` — prints precision, recall, F1 per class
  - Know: what each one returns and what parameters it takes.

- [ ] **Import from `tensorflow.keras`**
  - `Sequential` — model container (layers stacked in a line)
  - `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout` — layer types
  - `to_categorical` — one-hot encodes integer labels
  - `ModelCheckpoint` — callback to save best model during training
  - Know: Keras is the high-level API built into TensorFlow. `from tensorflow import keras` or `from tensorflow.keras import ...`

- [ ] **Import `time`**
  - Why: measure how long training takes
  - Know: `start = time.time()` before fit, `end = time.time()` after. Duration = `end - start` in seconds.

---

## STEP 2 — Load Images from Disk

**Purpose:** Read every image file from the dataset folders into Python memory as NumPy arrays. Filter out anything that is not a valid image so it does not crash the pipeline.

### Tasks:

- [ ] **Define the dataset root path**
  - Set a variable: `dataset_path = 'path/to/dataset'`
  - Know: the folder structure is `dataset_root/class_name/image_file.jpg`. The class name is the folder name.

- [ ] **Define valid extensions as a set**
  - `VALID_EXT = {'.jpeg', '.jpg', '.bmp', '.png'}`
  - Know: a set is used instead of a list because checking `ext in set` is O(1) — instant lookup regardless of size.
  - Know: always call `.lower()` on the extension before checking — `.JPG` and `.jpg` are the same format but Python sees them as different strings.

- [ ] **Loop through each class folder**
  - `for class_name in os.listdir(dataset_path):`
  - Build the full folder path: `class_path = os.path.join(dataset_path, class_name)`
  - Know: `os.listdir()` returns names only, not full paths. You must join with `os.path.join()` to get a usable path.

- [ ] **Loop through each file inside the class folder**
  - `for filename in os.listdir(class_path):`
  - Extract extension: `ext = os.path.splitext(filename)[1].lower()`
  - Skip invalid files: `if ext not in VALID_EXT: continue`
  - Know: `continue` skips the rest of the current loop iteration and moves to the next file.

- [ ] **Wrap image reading in try-except**
  - Inside `try`: build full path, call `cv2.imread(full_path)`, check if result is `None`
  - If `None`: `continue` — imread fails silently on corrupted files
  - Inside `except`: print the filename and error, then `continue`
  - Know: `cv2.imread()` does NOT raise an exception on corrupted files — it just returns `None`. The try-except catches other unexpected errors (permissions, invalid paths).


---

## STEP 3 — Explore Classes (EDA)

**Purpose:** Understand what you are working with before touching the data. Catch problems like class imbalance early so you can fix them before training.

### Tasks:

- [ ] **List all class names**
  - `class_names = os.listdir(dataset_path)`
  - Print them — in a brain tumor dataset, typical classes: `glioma`, `meningioma`, `no_tumor`, `pituitary`
  - Know: the number of class names = number of output neurons in the final Dense layer.

- [ ] **Count images per class**
  - For each class folder, count how many valid image files it contains
  - Store in a dict: `{'glioma': 826, 'meningioma': 822, 'no_tumor': 395, 'pituitary': 827}`
  - Know: **class imbalance** = when one class has significantly more images than others. The model will be biased toward the majority class and ignore the minority class.

- [ ] **Print total image count**
  - Sum all class counts
  - Know: total dataset size directly affects training time and how well the model generalizes.

---

## STEP 4 — Build Synchronized Image & Label Lists

**Purpose:** Merge all images from all classes into two parallel lists — one for images, one for labels — keeping them perfectly synchronized by index.

### Tasks:

- [ ] **Initialize two empty lists**
  - `images = []` and `labels = []`
  - Know: these will be converted to NumPy arrays later. Python lists are used first because appending to a list is efficient.

- [ ] **Loop through classes with enumerate**
  - `for label_index, class_name in enumerate(class_names):`
  - Know: `enumerate` gives you both the integer index `(0, 1, 2...)` and the class name simultaneously. The integer becomes the numeric label.

- [ ] **Append image and label in the same iteration**
  - `images.append(img)` and `labels.append(label_index)` inside the same loop body
  - Know: this is the fundamental contract — `images[i]` and `labels[i]` must ALWAYS refer to the same sample. Appending them together in the same iteration guarantees this.

- [ ] **Store a label map dictionary**
  - `label_map = {0: 'glioma', 1: 'meningioma', ...}`
  - Know: you need this later to decode model predictions (integers) back to human-readable class names.

- [ ] **Verify synchronization**
  - `assert len(images) == len(labels)` — must be True
  - Print `images[0].shape` to confirm images loaded correctly
  - Know: if lengths differ, something in the loop logic is broken and the dataset is corrupted.

---

## STEP 5 — Resize All Images

**Purpose:** Force every image to the same fixed dimensions so they can be stacked into one NumPy array and fed into the CNN.

### Tasks:

- [ ] **Choose a target size**
  - Standard: `IMG_SIZE = (224, 224)` — used by VGG16, ResNet
  - Know: CNNs require a fixed input shape. You cannot mix images of different sizes in the same batch.

- [ ] **Apply resize inside the loading loop (Step 2)**
  - `img = cv2.resize(img, IMG_SIZE)` right after imread
  - Know: `cv2.resize()` takes `(width, height)` but NumPy shape is `(height, width, channels)` — the order is reversed. `(224, 224)` works here because the image is square.

- [ ] **Convert BGR to RGB**
  - `img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`
  - Know: OpenCV reads images in BGR order. TensorFlow/Keras expects RGB. Skipping this means the model learns from wrong color data.

- [ ] **Verify the shape of one image**
  - Print `images[0].shape` — should be `(224, 224, 3)`
  - Know: 3 channels = RGB. If you loaded a grayscale image it would be `(224, 224)` with no channel dimension — this would break the model.

---

## STEP 6 — Convert Lists to NumPy Arrays

**Purpose:** Transform Python lists into NumPy arrays so sklearn and Keras can work with them.

### Tasks:

- [ ] **Convert images list**
  - `X = np.array(images)`
  - Know: produces a 4D array of shape `(N, 224, 224, 3)` — N images, each 224×224 with 3 channels. This is the exact input shape the CNN expects.

- [ ] **Convert labels list**
  - `y = np.array(labels)`
  - Know: produces a 1D array of shape `(N,)` containing integers like `[0, 0, 1, 2, 3, 1, ...]`

- [ ] **Verify shapes**
  - Print `X.shape` → `(N, 224, 224, 3)`
  - Print `y.shape` → `(N,)`
  - Print `X.dtype` → should be `uint8` at this stage (0–255 integers)
  - Know: if shapes are wrong here, everything downstream will fail.


---

## STEP 7 — Visualize Class Distribution & Sample Images

**Purpose:** Confirm visually that the data loaded correctly and spot any obvious problems before training.

### Tasks:

- [ ] **Plot a bar chart of class counts**
  - `plt.bar(class_names, counts)`
  - Know: a balanced chart = all bars roughly equal height. A very uneven chart = class imbalance problem that needs fixing.

- [ ] **Display sample images per class**
  - Use `plt.subplot()` to show a grid — one row per class, a few images per row
  - Know: always convert BGR→RGB before `plt.imshow()` or colors will look wrong (red and blue channels swapped).
  - Know: this lets you visually verify the images loaded correctly and the labels match what you see.

---

## STEP 8 — Check & Fix Class Imbalance

**Purpose:** Ensure the model does not learn to favor one class over another due to unequal representation in the dataset.

### Tasks:

- [ ] **Identify if imbalance exists**
  - Compare counts — if the ratio between largest and smallest class is > 2:1, consider rebalancing
  - Know: a model trained on imbalanced data can reach high accuracy by always predicting the majority class — useless and dangerous in medical diagnosis.

- [ ] **Option A — Class weights**
  - Compute weights: give more importance to minority classes during training
  - Pass `class_weight={0: w0, 1: w1, ...}` to `model.fit()`
  - Know: this does not change the data — it tells the model to penalize mistakes on minority classes more heavily.

- [ ] **Option B — Oversampling**
  - Duplicate images from minority classes OR use `ImageDataGenerator` to generate augmented versions
  - Know: augmentation = applying random flips, rotations, zooms to create new training samples from existing ones.

- [ ] **Option C — Undersampling**
  - Randomly remove images from majority classes to match the minority class count
  - Know: this loses data — only use if the dataset is large enough that losing samples is acceptable.

---

## STEP 9 — Encode Labels

**Purpose:** Convert string class names into a numeric format the model can compute with.

### Tasks:

- [ ] **Understand why encoding is needed**
  - Neural networks do math — they cannot compute on strings like `'glioma'`
  - Know: two-step process: string → integer → one-hot vector

- [ ] **Step 1 — Integer encoding with LabelEncoder**
  - `le = LabelEncoder()`
  - `y_int = le.fit_transform(y_strings)` → `'glioma'`→`0`, `'meningioma'`→`1`, etc.
  - Know: `fit_transform` learns the mapping AND applies it in one call.

- [ ] **Step 2 — One-hot encoding with to_categorical**
  - `y_onehot = to_categorical(y_int, num_classes=4)`
  - `0` → `[1,0,0,0]`, `1` → `[0,1,0,0]`, etc.
  - Know: required when using `categorical_crossentropy` as the loss function. The model outputs a probability vector — one-hot gives it the same format to compare against.
  - Know: alternative — skip one-hot and use `sparse_categorical_crossentropy` which works directly on integers.

- [ ] **Verify encoded labels**
  - Print `y_onehot[0]` → should be a vector with one `1.0` and the rest `0.0`
  - Print `y_onehot.shape` → `(N, num_classes)`

---

## STEP 10 — Split Train / Test

**Purpose:** Separate data into a training set (model learns from this) and a test set (honest final evaluation on unseen data).

### Tasks:

- [ ] **Understand the purpose of each split**
  - Train set (~80%): the model sees this data and updates its weights
  - Test set (~20%): used ONLY at the very end — never during training or tuning
  - Know: using test data during training gives falsely optimistic results — the model has "seen" the answers.

- [ ] **Call train_test_split**
  - `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`
  - Know: `stratify=y` ensures each split has the same class proportions as the full dataset — critical with imbalanced classes.
  - Know: `random_state=42` makes the split reproducible — same result every run.

- [ ] **Verify split sizes**
  - Print shapes of all 4 arrays
  - `X_train.shape[0] + X_test.shape[0]` must equal total N

- [ ] **Optional — create a validation split**
  - Split X_train further into train + validation (~15% of total)
  - Know: validation set is used during training to monitor overfitting — it is NOT the same as the test set.

---

## STEP 11 — Normalize Pixel Values

**Purpose:** Scale pixel values from [0, 255] to [0, 1] so the model trains stably.

### Tasks:

- [ ] **Apply normalization**
  - `X_train = X_train / 255.0`
  - `X_test = X_test / 255.0`
  - Know: pixel values are integers 0–255. Dividing by 255 converts them to floats 0.0–1.0.

- [ ] **Understand why it matters**
  - Know: large input values (0–255) produce large gradients during backpropagation → training becomes unstable or diverges. Small values (0–1) keep the math stable.
  - Know: apply the SAME scale to train and test. Never fit a new scaler on the test set.

- [ ] **Verify**
  - `X_train.max()` → `1.0`, `X_train.min()` → `0.0`
  - `X_train.dtype` → `float64` or `float32`
  - Know: if max is still 255, you forgot to normalize — the model will likely fail to converge.


---

## STEP 12 — Build CNN Architecture

**Purpose:** Define the sequence of layers that transforms an input image into a class prediction.

### Tasks:

- [ ] **Create a Sequential model**
  - `model = Sequential()`
  - Know: Sequential = layers stacked one after another in a straight line. Each layer's output is the next layer's input.

- [ ] **Add Conv2D layers**
  - `Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3))`
  - Know: `32` = number of filters (feature detectors). Each filter slides over the image and produces one feature map.
  - Know: `(3,3)` = kernel size — the filter looks at 3×3 pixel patches at a time.
  - Know: `input_shape` is only needed on the FIRST layer — Keras infers the rest automatically.
  - Know: ReLU activation = sets all negative values to 0. Adds non-linearity so the model can learn complex patterns, not just straight lines.

- [ ] **Add MaxPooling2D after each Conv2D**
  - `MaxPooling2D(pool_size=(2,2))`
  - Know: reduces spatial size by half (224→112→56...). Takes the maximum value in each 2×2 window.
  - Know: reduces computation and makes the model less sensitive to the exact position of a feature.

- [ ] **Add Dropout layers**
  - `Dropout(0.5)`
  - Know: during training, randomly sets 50% of neurons to 0 each batch. Forces the network not to rely on any single neuron → reduces overfitting.
  - Know: Dropout is automatically disabled during prediction/evaluation — only active during training.

- [ ] **Add Flatten**
  - `Flatten()`
  - Know: converts the 3D output of the last pooling layer `(H, W, C)` into a 1D vector. Required before Dense layers.

- [ ] **Add Dense hidden layer**
  - `Dense(128, activation='relu')`
  - Know: fully connected — every neuron connects to every input from the previous layer. This is where high-level reasoning happens.

- [ ] **Add output Dense layer**
  - `Dense(num_classes, activation='softmax')`
  - Know: output size MUST equal the number of classes.
  - Know: Softmax converts raw scores into probabilities that sum to 1.0. The class with the highest probability is the prediction.

- [ ] **Verify architecture**
  - `model.summary()` — prints all layers, output shapes, parameter counts
  - `plot_model(model, show_shapes=True)` — generates a visual diagram
  - Know: check that output shapes decrease as expected through Conv+Pool layers, and that the final layer has `num_classes` units.

---

## STEP 13 — Compile the Model

**Purpose:** Configure how the model will learn — what algorithm updates weights, what measures the error, what to track.

### Tasks:

- [ ] **Choose optimizer: Adam**
  - `optimizer='adam'`
  - Know: Adam automatically adapts the learning rate per parameter. It is the standard default for CNNs.
  - Know: learning rate (default `0.001`) controls step size of each weight update. Too high = unstable. Too low = very slow.

- [ ] **Choose loss function**
  - `loss='categorical_crossentropy'` if labels are one-hot encoded
  - `loss='sparse_categorical_crossentropy'` if labels are plain integers
  - Know: loss measures how wrong the predictions are. The optimizer minimizes this value over training.

- [ ] **Set metrics**
  - `metrics=['accuracy']`
  - Know: accuracy = % of correct predictions. Logged after each epoch for both train and validation.

- [ ] **Call compile**
  - `model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])`

- [ ] **Determine hyperparameters**
  - `EPOCHS = 20` (start here, adjust based on curves)
  - `BATCH_SIZE = 32` (common default)
  - Know: epoch = one full pass through the training set. Batch = chunk of data processed before each weight update. With 1000 images and batch 32 → ~31 updates per epoch.

---

## STEP 14 — Train the Model

**Purpose:** Feed training data through the model repeatedly so it adjusts its weights to minimize the loss.

### Tasks:

- [ ] **Set up ModelCheckpoint callback**
  - `checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)`
  - Know: saves the model automatically whenever `val_accuracy` improves. `save_best_only=True` keeps only the best version, not the last.
  - Know: important because training can degrade the model after peak performance (overfitting) — the checkpoint preserves the best state.

- [ ] **Measure training time with time**
  - `start = time.time()` before fit
  - `end = time.time()` after fit
  - `print(f'Training time: {end - start:.2f}s')`

- [ ] **Call model.fit()**
  - `history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val), callbacks=[checkpoint])`
  - Know: always pass `validation_data` — without it you cannot detect overfitting during training.
  - Know: `history` stores loss and accuracy per epoch — you need it to plot learning curves.

- [ ] **Understand what happens each epoch**
  - Model processes all training batches → computes loss → updates weights via backpropagation
  - Then evaluates on validation set (no weight updates during validation)
  - Prints: `loss`, `accuracy`, `val_loss`, `val_accuracy`
  - Know: **overfitting** = val_loss increases while train_loss keeps dropping. **Underfitting** = both losses stay high.

---

## STEP 15 — Evaluate the Model

**Purpose:** Measure real performance on the test set — data the model has never seen during training or tuning.

### Tasks:

- [ ] **Run model.evaluate() on test set**
  - `test_loss, test_accuracy = model.evaluate(X_test, y_test)`
  - Know: this is the only honest performance measure. Train/val accuracy can be inflated.

- [ ] **Generate predictions**
  - `y_pred_probs = model.predict(X_test)` → array of probability vectors
  - `y_pred = np.argmax(y_pred_probs, axis=1)` → class index with highest probability
  - Know: `argmax` picks the index of the max value in each row — that is the predicted class.

- [ ] **Decode true labels for comparison**
  - If y_test is one-hot: `y_true = np.argmax(y_test, axis=1)`
  - If y_test is integers: use directly

- [ ] **Print classification report**
  - `print(classification_report(y_true, y_pred, target_names=class_names))`
  - Know — what each metric means:
    - **Precision**: of all predicted as class X, how many actually are X
    - **Recall**: of all actual class X, how many did the model catch — CRITICAL in medical: missing a tumor is dangerous
    - **F1-score**: balance between precision and recall — best single metric for imbalanced datasets
    - **Support**: number of real samples per class in the test set

- [ ] **Plot confusion matrix**
  - `cm = confusion_matrix(y_true, y_pred)`
  - Visualize with `plt.imshow(cm)` or seaborn heatmap
  - Know: rows = actual class, columns = predicted class. Diagonal = correct. Off-diagonal = mistakes.
  - Know: identify the most dangerous misclassification — predicting `no_tumor` when there IS a tumor.

- [ ] **Display correct and incorrect predictions**
  - Find indices where `y_pred == y_true` (correct) and where they differ (incorrect)
  - Show the image, the true label, and the predicted label side by side
  - Know: this gives intuition about what kinds of images fool the model.


---

## STEP 16 — Visualize Learning Curves

**Purpose:** Diagnose how the model trained — detect overfitting, underfitting, or a good fit by reading the accuracy and loss curves.

### Tasks:

- [ ] **Plot accuracy curves**
  - `plt.plot(history.history['accuracy'], label='Train')`
  - `plt.plot(history.history['val_accuracy'], label='Validation')`
  - Know: both curves should rise and converge. A large gap between them = overfitting.

- [ ] **Plot loss curves**
  - `plt.plot(history.history['loss'], label='Train')`
  - `plt.plot(history.history['val_loss'], label='Validation')`
  - Know: both should decrease and stabilize. Val_loss rising while train_loss drops = overfitting.

- [ ] **Read the curves and diagnose**
  - Good fit: train and val curves close together, both improving then stabilizing
  - Overfitting: val curve diverges upward after a certain epoch → reduce epochs, add Dropout, add more data
  - Underfitting: both curves flat and high → add more layers, train longer, reduce Dropout rate

---

## STEP 17 — Create predict_image() Function

**Purpose:** A reusable function that takes any image path, preprocesses it exactly like training data, and returns the predicted class name.

### Tasks:

- [ ] **Define the function signature**
  - `def predict_image(path):`

- [ ] **Load and preprocess the image inside the function**
  - Read with `cv2.imread(path)`
  - Resize to `(224, 224)`
  - Convert BGR→RGB
  - Normalize: `/ 255.0`
  - Know: the preprocessing MUST be identical to what was done during training. Any difference (different size, no normalization) will produce wrong predictions.

- [ ] **Add the batch dimension**
  - `img = np.expand_dims(img, axis=0)` → shape becomes `(1, 224, 224, 3)`
  - Know: the model always expects a batch as input — shape `(batch_size, H, W, C)`. A single image has no batch dimension, so you add it manually.

- [ ] **Run prediction**
  - `pred = model.predict(img)` → shape `(1, num_classes)`
  - `class_index = np.argmax(pred, axis=1)[0]`
  - `class_name = label_map[class_index]`
  - Know: `label_map` is the dictionary you saved in Step 4 — `{0: 'glioma', 1: 'meningioma', ...}`

- [ ] **Return the result**
  - Return both the class name and the confidence: `return class_name, pred[0][class_index]`
  - Know: confidence = the probability value for the predicted class (0.0 to 1.0). High confidence = model is certain.

---

## STEP 18 — Save the Trained Model

**Purpose:** Persist the trained model to disk so it can be reloaded and used without retraining.

### Tasks:

- [ ] **Save the model**
  - `model.save('brainscan_model.h5')`
  - Know: `.h5` format stores architecture + weights + optimizer state in one file.
  - Know: alternative SavedModel format: `model.save('brainscan_model/')` — saves as a folder, preferred for deployment.

- [ ] **Save the label map**
  - Save `{0: 'glioma', 1: 'meningioma', ...}` using `json.dump()` to a `.json` file
  - Know: without this, predictions are just numbers — you cannot tell the user what class was detected.

- [ ] **Test reloading**
  - `loaded_model = tf.keras.models.load_model('brainscan_model.h5')`
  - Run `predict_image()` with the loaded model on one test image
  - Verify the prediction matches what the original model gave
  - Know: always verify a saved model works correctly before considering the project done.

---

## STEP 19 — Build Streamlit Interface

**Purpose:** Create an interactive web app where a user can upload a brain MRI image and get a prediction — no code required from the user.

### Tasks:

- [ ] **Install and import Streamlit**
  - `pip install streamlit`
  - `import streamlit as st`
  - Know: Streamlit turns a Python script into a web app. Run with `streamlit run app.py` in the terminal.

- [ ] **Load model once at startup**
  - Use `@st.cache_resource` decorator on the loading function
  - Know: without caching, the model reloads from disk on every user interaction — very slow. `@st.cache_resource` loads it once and keeps it in memory.

- [ ] **Add a title and description**
  - `st.title('BrainScan AI')` and `st.write('...')`

- [ ] **Add image uploader**
  - `uploaded_file = st.file_uploader('Upload an MRI image', type=['jpg','jpeg','png','bmp'])`
  - Know: `file_uploader` returns a file-like object or `None` if nothing is uploaded yet.

- [ ] **Display the uploaded image**
  - `st.image(uploaded_file, caption='Uploaded MRI')`

- [ ] **Run prediction and display result**
  - Read the uploaded file → preprocess → call `predict_image()`
  - `st.success(f'Prediction: {class_name} ({confidence:.2%} confidence)')`
  - Know: wrap this in `if uploaded_file is not None:` — only run when the user has actually uploaded something.

- [ ] **Test the full flow**
  - Run `streamlit run app.py`
  - Upload a test image
  - Verify the prediction matches what `model.predict()` returns directly
  - Know: the Streamlit app is the final deliverable — it is what a doctor would actually use.
