# ============================================================
#  train_ann.py  –  ANN-Based Student Performance Evaluator
#  Tasks 1 → 8  (Dataset exploration → Save model)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression   # Bonus: comparison

# ──────────────────────────────────────────────
# TASK 1 – Understand the Dataset
# ──────────────────────────────────────────────
print("=" * 60)
print("  TASK 1 – Dataset Overview")
print("=" * 60)

df = pd.read_excel("dataset.xlsx")

print("\n📋 First 5 rows:")
print(df.head())

print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n🏷️  Column names: {df.columns.tolist()}")

print("\n📊 Data types:")
print(df.dtypes)

print("\n📈 Basic statistics:")
print(df.describe())

print("\n❓ Missing values:", df.isnull().sum().sum())

print("""
🔍 Column Explanations:
  attendance   – % of classes attended by student (0–100)
  assignment   – marks scored in assignments
  quiz         – marks scored in quizzes
  mid          – mid-term exam marks
  study_hours  – average daily study hours
  result       – target: 1 = Pass, 0 = Fail

🧠 Problem Type: CLASSIFICATION
   Reason: The output (result) is a discrete label (0 or 1),
   not a continuous value. We predict a category, not a number.
""")

# Input features & target
X = df[["attendance", "assignment", "quiz", "mid", "study_hours"]]
y = df["result"]

print(f"✅ X shape (features): {X.shape}")
print(f"✅ y shape (target)  : {y.shape}")
print(f"   Class distribution : Pass={sum(y==1)}, Fail={sum(y==0)}")

# ──────────────────────────────────────────────
# TASK 3 – Data Preprocessing
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 3 – Data Preprocessing")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTraining set  : {X_train.shape[0]} samples (80%)")
print(f"Testing  set  : {X_test.shape[0]}  samples (20%)")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("""
🔑 Why scaling is required in ANN:
   Neural networks use gradient-based optimization. If features
   have very different ranges (e.g. attendance 0–100 vs
   study_hours 0–12), larger-valued features dominate gradient
   updates and slow convergence. StandardScaler makes every
   feature have mean=0 and std=1, so all features contribute
   equally and training is faster and more stable.
""")

# ──────────────────────────────────────────────
# TASK 4 + 5 – Build & Train ANN Model
# ──────────────────────────────────────────────
print("=" * 60)
print("  TASK 4 & 5 – Build & Train ANN")
print("=" * 60)

print("""
🧠 ANN Concepts:
  Neurons          – basic computing units; each neuron receives
                     inputs, multiplies by weights, adds bias,
                     applies activation function, and outputs.
  Activation func  – introduces non-linearity (relu, tanh, sigmoid).
                     Without it, the network is just linear algebra.
  Hidden layers    – layers between input and output; they learn
                     intermediate representations of the data.
  Our architecture :
      Input  layer  → 5 neurons  (one per feature)
      Hidden layer1 → 64 neurons (relu)
      Hidden layer2 → 32 neurons (relu)
      Output layer  → 1 neuron   (sigmoid for binary classification)
""")

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),   # 2 hidden layers
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
    verbose=False,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

print("⏳ Training ANN model ...")
model.fit(X_train_scaled, y_train)

print(f"✅ Training complete!")
print(f"   Iterations ran  : {model.n_iter_}")
print(f"   Best val score  : {model.best_validation_score_:.4f}")
print(f"   Loss at final   : {model.loss_:.4f}")

# ──────────────────────────────────────────────
# TASK 6 – Evaluate Model
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 6 – Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test_scaled)
acc    = accuracy_score(y_test, y_pred)
cm     = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Fail","Pass"])

print(f"\n🎯 Accuracy : {acc * 100:.2f}%")
print("\n📊 Confusion Matrix:")
print(cm)
print("\n📋 Classification Report:")
print(report)

print(f"""
💡 What accuracy means:
   {acc*100:.1f}% of all predictions in the test set were correct.
   It tells us the overall fraction of right answers.

🔍 Mistakes the model may make:
   False Positives (FP) – predicted Pass but actually Failed → {cm[0][1]} cases
   False Negatives (FN) – predicted Fail but actually Passed → {cm[1][0]} cases
   FN is more costly in education (missing a struggling student).
""")

# ── Bonus: Logistic Regression comparison ──
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_scaled))
print(f"📊 Bonus – Logistic Regression Accuracy : {lr_acc*100:.2f}%  (vs ANN {acc*100:.2f}%)")

# ── Bonus: Confusion matrix heatmap ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Fail","Pass"])
disp.plot(ax=axes[0], colorbar=False, cmap="Blues")
axes[0].set_title("ANN – Confusion Matrix")

# Loss curve
axes[1].plot(model.loss_curve_, label="Train Loss", color="royalblue")
if hasattr(model, "validation_scores_"):
    axes[1].plot(
        [1 - s for s in model.validation_scores_],
        label="Val Loss (1-score)", linestyle="--", color="tomato"
    )
axes[1].set_title("Training Loss Curve")
axes[1].set_xlabel("Iteration")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_report.png", dpi=150)
print("\n📷 Saved training_report.png")

# ──────────────────────────────────────────────
# TASK 7 – Evaluation Function
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 7 – evaluate_student() function")
print("=" * 60)

def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Predict pass/fail for a new student using the trained ANN.

    Parameters
    ----------
    attendance   : int/float  – attendance percentage (0–100)
    assignment   : int/float  – assignment score
    quiz         : int/float  – quiz score
    mid          : int/float  – mid-term score
    study_hours  : int/float  – average daily study hours

    Returns
    -------
    dict with 'result' (0/1), 'label' (Fail/Pass), 'probability'
    """
    features = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled   = scaler.transform(features)
    pred     = model.predict(scaled)[0]
    proba    = model.predict_proba(scaled)[0]
    return {
        "result"     : int(pred),
        "label"      : "Pass ✅" if pred == 1 else "Fail ❌",
        "probability": {"Fail": round(proba[0]*100, 1),
                        "Pass": round(proba[1]*100, 1)}
    }

# Demo
demo = evaluate_student(85, 90, 78, 70, 8)
print(f"\n🧪 Demo prediction  : {demo}")
demo2 = evaluate_student(40, 30, 25, 20, 1)
print(f"🧪 Demo prediction 2: {demo2}")

# ──────────────────────────────────────────────
# TASK 8 – Save Model & Scaler
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("  TASK 8 – Save Model & Scaler")
print("=" * 60)

joblib.dump(model,  "model.joblib")
joblib.dump(scaler, "scaler.joblib")

print("""
✅ model.joblib  – saved
✅ scaler.joblib – saved

💡 Why BOTH must be saved:
   The model was trained on SCALED data. If we load only the model
   and feed raw (unscaled) inputs, the predictions will be completely
   wrong. The scaler must be saved so we apply the EXACT same
   transformation at prediction time that was applied at training time.
""")

print("=" * 60)
print("  All Tasks Complete! ✅")
print("=" * 60)
