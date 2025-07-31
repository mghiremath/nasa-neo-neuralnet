# Hazardous Asteroid Classification

Predicts if an asteroid is "potentially hazardous" using real NASA data.

---

## 1. Traditional Machine Learning (scikit-learn)

**Models:**

- Random Forest
- Logistic Regression

**Highlights:**

- Good baseline accuracy.
- Balanced recall and precision for hazardous asteroids.

**Sample Results:**

| Metric    | Precision | Recall | F1     | Accuracy |
| --------- | --------- | ------ | ------ | -------- |
| Hazardous | 0.3080    | 0.6144 | 0.4103 | 0.8356   |

---

## 2. Custom Neural Network (micrograd MLP)

**How:**

- Implemented MLP from scratch, using Karpathy-style autodiff and training loops.

**Highlights:**

- Very high precision, but much lower recall for hazardous asteroids.
- Great for learning core mechanics.

**Sample Results:**

| Metric    | Precision | Recall | F1     | Accuracy |
| --------- | --------- | ------ | ------ | -------- |
| Hazardous | 0.7580    | 0.1612 | 0.2659 | 0.9134   |

---

## 3. Deep Learning with PyTorch

**How:**

- MLP in PyTorch with class weighting and threshold tuning.
- Trained with Adam optimizer, multiple layers.

**Highlights:**

- Best at finding most hazardous asteroids (high recall).
- Tuned threshold for optimal F1-score.

**Sample Results (best threshold):**

| Metric    | Precision | Recall | F1     | Accuracy |
| --------- | --------- | ------ | ------ | -------- |
| Hazardous | 0.2890    | 0.8275 | 0.4284 | 0.7851   |

---

## Model Tradeoff Summary

| Model         | Precision (haz) | Recall (haz) | F1 (haz) | Accuracy |
| ------------- | --------------- | ------------ | -------- | -------- |
| Random Forest | 0.3080          | 0.6144       | 0.4103   | 0.8356   |
| micrograd MLP | 0.7580          | 0.1612       | 0.2659   | 0.9134   |
| PyTorch MLP   | 0.2890          | 0.8275       | 0.4284   | 0.7851   |

---

## Key Learnings

- Class imbalance is a core challenge—handled via weighting, resampling, and threshold tuning.
- Traditional ML is a strong baseline; PyTorch MLP gives best recall with threshold tuning.
- Writing a neural net from scratch helps deeply understand training and backpropagation.

---

**Run the notebook [`nasa-neo.ipynb`](nasa-neo.ipynb) to see all code and comparisons!**

---
