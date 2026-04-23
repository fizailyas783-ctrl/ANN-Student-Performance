# ============================================================
#  predict.py  –  Load saved model and evaluate new students
#  Task 7 (reusable evaluation function)
# ============================================================

import numpy as np
import joblib
import os

# ── Load model and scaler once ──────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
model  = joblib.load(os.path.join(_BASE, "model.joblib"))
scaler = joblib.load(os.path.join(_BASE, "scaler.joblib"))


def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    """
    Predict Pass / Fail for a student using the trained ANN.

    Parameters
    ----------
    attendance   : float  Attendance percentage (0–100)
    assignment   : float  Assignment marks
    quiz         : float  Quiz marks
    mid          : float  Mid-term marks
    study_hours  : float  Average daily study hours

    Returns
    -------
    dict
        result      – 0 (Fail) or 1 (Pass)
        label       – human-readable string
        probability – dict with 'Pass' and 'Fail' percentages
        performance – Low / Medium / High (bonus tier)
    """
    features = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled   = scaler.transform(features)

    pred  = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]  # [P(Fail), P(Pass)]
    pass_prob = proba[1] * 100

    # Bonus: 3-tier performance
    if pass_prob >= 75:
        perf = "🟢 High Performance"
    elif pass_prob >= 45:
        perf = "🟡 Medium Performance"
    else:
        perf = "🔴 Low Performance"

    return {
        "result"     : int(pred),
        "label"      : "Pass ✅" if pred == 1 else "Fail ❌",
        "probability": {
            "Pass": round(pass_prob, 1),
            "Fail": round(proba[0] * 100, 1)
        },
        "performance": perf
    }


# ── Command-line interface (Option A) ───────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("  Student Performance Evaluator (CLI)")
    print("  Powered by ANN  |  Type 'q' to quit")
    print("=" * 50)

    while True:
        print()
        try:
            inp = input("Attendance   (0-100): ").strip()
            if inp.lower() == "q":
                break
            attendance   = float(inp)
            assignment   = float(input("Assignment marks   : "))
            quiz         = float(input("Quiz marks         : "))
            mid          = float(input("Mid-term marks     : "))
            study_hours  = float(input("Daily study hours  : "))
        except ValueError:
            print("⚠️  Please enter valid numbers.")
            continue

        result = evaluate_student(attendance, assignment, quiz, mid, study_hours)

        print("\n" + "-" * 40)
        print(f"  Prediction   : {result['label']}")
        print(f"  Performance  : {result['performance']}")
        print(f"  Pass chance  : {result['probability']['Pass']}%")
        print(f"  Fail chance  : {result['probability']['Fail']}%")
        print("-" * 40)
