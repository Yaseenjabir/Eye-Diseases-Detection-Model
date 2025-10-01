# 🧠 Eye Disease Classification (OCT2017 Dataset)

This project trains a deep learning model to classify **Retinal OCT (Optical Coherence Tomography) scans** into 4 categories of eye conditions:

* **CNV** – Choroidal Neovascularization
* **DME** – Diabetic Macular Edema
* **DRUSEN** – Drusen deposits
* **NORMAL** – Normal retina

The dataset comes from [Kermany et al., 2018 (OCT2017)](https://www.kaggle.com/datasets/paultimothymooney/kermany2018).
The model is built with **TensorFlow/Keras** and achieves high accuracy on test data.

---

## 🚀 Features

* Trains a **Convolutional Neural Network (CNN)** on OCT images
* Achieves **>95% test accuracy**
* Supports **saving/loading models** in `.keras` format
* Allows predictions on **new unseen images**

---

## 📂 Project Structure

```
.
├── eye_disease_model.keras   # Trained model (saved weights & architecture)
├── predict.py                # Script to predict on new images
├── train.py                  # Training script
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation
```

---

## 🔧 Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/Yaseenjabir/Eye-Diseases-Detection-Model
cd Eye-Diseases-Detection-Model
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate    # On Linux/Mac
venv\Scripts\activate       # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📊 Dataset

Download the **OCT2017 dataset** from Kaggle:
👉 [https://www.kaggle.com/datasets/paultimothymooney/kermany2018](https://www.kaggle.com/datasets/paultimothymooney/kermany2018)

Expected folder structure after unzipping:

```
OCT2017/
   ├── train/
   │     ├── CNV/
   │     ├── DME/
   │     ├── DRUSEN/
   │     └── NORMAL/
   ├── test/
   └── val/
```

Place this `OCT2017` folder inside the project root.

---

## 🏋️ Training the Model

Run:

```bash
python train.py
```

This will:

* Load the dataset from `OCT2017/train` and `OCT2017/val`
* Train a CNN for 10 epochs
* Save the trained model as `eye_disease_model.keras`

---

## 🔮 Running Predictions

To classify a new image:

```bash
python predict.py
```

### Example output:

```
CNV: 1.23%
DME: 0.56%
DRUSEN: 97.80%
NORMAL: 0.41%

Predicted class: DRUSEN
```

---

## ⚠️ Notes

* The model expects input images resized to **224x224** and normalized (`0-1`).
* Predictions work best on images similar to the dataset (OCT retinal scans). Random Google images may not be recognized properly.
* The order of classes is determined by TensorFlow’s directory reading:

  ```
  ['CNV', 'DME', 'DRUSEN', 'NORMAL']
  ```

  (Alphabetical order of subfolders inside `train/`).

---

## 📈 Results

* **Training accuracy:** ~99%
* **Validation accuracy:** ~96–97%
* **Test accuracy:** ~96%

Confusion matrix and more results can be added later.

---

## 📜 Requirements

* Python 3.9+
* TensorFlow 2.15+
* NumPy

(Install all with `pip install -r requirements.txt`)

