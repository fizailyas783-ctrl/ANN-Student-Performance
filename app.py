# ============================================================
#  app.py  –  Streamlit UI for Student Performance Evaluator
#  Task 9 – User Interface
#  Run: streamlit run app.py
# ============================================================

import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Evaluator",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Load model & scaler ─────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    mdl = joblib.load(os.path.join(base, "model.joblib"))
    scl = joblib.load(os.path.join(base, "scaler.joblib"))
    return mdl, scl

model, scaler = load_artifacts()

# ── Core prediction function ────────────────────────────────
def evaluate_student(attendance, assignment, quiz, mid, study_hours):
    features = np.array([[attendance, assignment, quiz, mid, study_hours]])
    scaled   = scaler.transform(features)
    pred     = model.predict(scaled)[0]
    proba    = model.predict_proba(scaled)[0]
    pass_pct = round(proba[1] * 100, 1)
    fail_pct = round(proba[0] * 100, 1)

    if pass_pct >= 75:
        tier, tier_color = "High Performance 🟢", "#2ecc71"
    elif pass_pct >= 45:
        tier, tier_color = "Medium Performance 🟡", "#f39c12"
    else:
        tier, tier_color = "Low Performance 🔴", "#e74c3c"

    return {
        "result"     : int(pred),
        "label"      : "Pass ✅" if pred == 1 else "Fail ❌",
        "pass_pct"   : pass_pct,
        "fail_pct"   : fail_pct,
        "tier"       : tier,
        "tier_color" : tier_color,
    }

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/graduation-cap.png",
        width=80
    )
    st.title("ANN Evaluator")
    st.markdown("---")
    st.markdown("""
**About this App**

This app uses an **Artificial Neural Network (ANN)**
trained on 600 student records to predict whether a
student will **Pass or Fail** based on academic inputs.

**Model:** MLPClassifier (scikit-learn)
**Architecture:** 5 → 64 → 32 → 1
**Activation:** ReLU
    """)
    st.markdown("---")
    st.markdown("**Task 11 – Key Concepts**")
    with st.expander("What is ANN?"):
        st.write("""
An Artificial Neural Network mimics the human brain.
It consists of layers of connected *neurons* that
learn patterns from data by adjusting weights during
training. Our ANN learned: given 5 academic features,
predict Pass or Fail.
        """)
    with st.expander("Why scaling?"):
        st.write("""
Attendance ranges 0–100 while study_hours ranges 0–12.
Without scaling, larger features dominate the gradients
and training becomes unstable. StandardScaler normalizes
all features to mean=0, std=1.
        """)
    with st.expander("Model Limitations"):
        st.write("""
- Only 600 training samples (small)
- Doesn't capture personal/social factors
- Binary output only (no nuance between close cases)
- May not generalize to different curricula
        """)

# ══════════════════════════════════════════════════════════════
#  MAIN PAGE
# ══════════════════════════════════════════════════════════════
st.title("🎓 Student Performance Evaluator")
st.markdown("##### Powered by Artificial Neural Network (ANN)")
st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📋 Batch Prediction", "📊 Model Info"])

# ══════════════════════════════════════
#  TAB 1 – Single Prediction
# ══════════════════════════════════════
with tab1:
    st.subheader("Enter Student Details")
    st.markdown("Adjust the sliders to match the student's academic profile:")

    col1, col2 = st.columns(2)
    with col1:
        attendance  = st.slider("📅 Attendance (%)",    0, 100, 75,
                                help="Percentage of classes attended")
        assignment  = st.slider("📝 Assignment Marks",  0, 100, 70,
                                help="Total marks in assignments")
        quiz        = st.slider("❓ Quiz Marks",         0, 100, 65,
                                help="Total marks in quizzes")
    with col2:
        mid         = st.slider("📖 Mid-Term Marks",    0, 100, 60,
                                help="Mid-term examination marks")
        study_hours = st.slider("⏱️ Study Hours/Day",   0,  16,  5,
                                help="Average daily study hours")
        st.markdown("<br>", unsafe_allow_html=True)

    predict_btn = st.button("🚀 Predict Performance", type="primary", use_container_width=True)

    if predict_btn:
        res = evaluate_student(attendance, assignment, quiz, mid, study_hours)

        st.markdown("---")
        # Big result banner
        banner_bg = "#d5f5e3" if res["result"] == 1 else "#fadbd8"
        banner_border = "#2ecc71" if res["result"] == 1 else "#e74c3c"
        st.markdown(
            f"""
            <div style="background:{banner_bg}; border-left:6px solid {banner_border};
                        border-radius:8px; padding:20px; text-align:center;">
                <h2 style="margin:0; color:{banner_border};">{res['label']}</h2>
                <p style="font-size:18px; margin:6px 0; color:#555;">{res['tier']}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability columns
        c1, c2 = st.columns(2)
        c1.metric("✅ Pass Probability", f"{res['pass_pct']}%")
        c2.metric("❌ Fail Probability", f"{res['fail_pct']}%")

        # Probability bar chart
        fig, ax = plt.subplots(figsize=(5, 1.8))
        ax.barh(["Fail", "Pass"], [res["fail_pct"], res["pass_pct"]],
                color=["#e74c3c", "#2ecc71"], height=0.5)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        ax.spines[["top","right","left"]].set_visible(False)
        for i, v in enumerate([res["fail_pct"], res["pass_pct"]]):
            ax.text(v + 1, i, f"{v}%", va="center", fontsize=10)
        st.pyplot(fig, use_container_width=False)

        # Input summary table
        st.markdown("**📋 Input Summary:**")
        input_df = pd.DataFrame({
            "Feature"    : ["Attendance", "Assignment", "Quiz", "Mid-Term", "Study Hours"],
            "Your Input" : [attendance, assignment, quiz, mid, study_hours],
            "Typical Range": ["60–95%", "50–100", "40–100", "40–100", "2–10 hrs"]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════
#  TAB 2 – Batch Prediction
# ══════════════════════════════════════
with tab2:
    st.subheader("Batch Predict from CSV / Manual Table")
    st.markdown("""
Upload a CSV file with columns:
`attendance, assignment, quiz, mid, study_hours`
    """)

    sample_csv = pd.DataFrame({
        "attendance" : [85, 42, 70, 95, 30],
        "assignment" : [90, 35, 60, 88, 20],
        "quiz"       : [78, 25, 55, 92, 15],
        "mid"        : [70, 30, 58, 80, 18],
        "study_hours": [8,   1,  5,  10,  0],
    })

    st.download_button(
        "⬇️ Download Sample CSV",
        sample_csv.to_csv(index=False),
        file_name="sample_students.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Upload your CSV", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        required = {"attendance","assignment","quiz","mid","study_hours"}
        if not required.issubset(batch_df.columns):
            st.error(f"CSV must have columns: {required}")
        else:
            results = []
            for _, row in batch_df.iterrows():
                r = evaluate_student(
                    row["attendance"], row["assignment"],
                    row["quiz"],       row["mid"],
                    row["study_hours"]
                )
                results.append({
                    "Result"     : r["label"],
                    "Pass %"     : r["pass_pct"],
                    "Tier"       : r["tier"],
                })
            out_df = pd.concat([batch_df, pd.DataFrame(results)], axis=1)
            st.success(f"✅ Predicted {len(out_df)} students!")
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "⬇️ Download Results",
                out_df.to_csv(index=False),
                file_name="predictions.csv"
            )

# ══════════════════════════════════════
#  TAB 3 – Model Info
# ══════════════════════════════════════
with tab3:
    st.subheader("🧠 ANN Architecture")

    arch_data = {
        "Layer"   : ["Input Layer", "Hidden Layer 1", "Hidden Layer 2", "Output Layer"],
        "Neurons" : [5, 64, 32, 1],
        "Activation": ["—", "ReLU", "ReLU", "Softmax"],
        "Purpose" : [
            "Receives 5 student features",
            "Learns complex patterns",
            "Refines representations",
            "Outputs Pass/Fail probability"
        ]
    }
    st.table(pd.DataFrame(arch_data))

    st.markdown("---")
    st.subheader("📖 Task 11 – Conceptual Explanation")

    st.markdown("""
**1. What is ANN in your own words?**
An ANN is a mathematical system inspired by the brain. It has layers of
"neurons" connected by weights. During training, it adjusts these weights
using gradient descent so that it can map input features (attendance, marks, etc.)
to a correct output (Pass/Fail). It essentially *learns a function* from data.

**2. What function did your model learn?**
The model learned the function:
```
f(attendance, assignment, quiz, mid, study_hours) → Pass or Fail
```
Internally this is a composition of matrix multiplications and ReLU activations
across two hidden layers.

**3. How does the system evaluate a new student?**
1. Collect 5 features from the student
2. Scale them using the saved StandardScaler
3. Pass scaled values through the ANN (forward pass)
4. Read the output neuron's probability
5. If P(Pass) ≥ 0.5 → Predict Pass, else Fail

**4. Why is scaling important?**
Features have different ranges. Attendance is 0–100 but study_hours is 0–16.
Without scaling, large-valued features dominate the gradient updates, causing
slow or incorrect training. Scaling makes all features contribute equally.

**5. Limitations of this model?**
- Small dataset (600 records) — may not generalize perfectly
- Only 5 features — ignores many real-world factors (mental health, family, etc.)
- Binary output — doesn't capture borderline cases
- Assumes features are the same across all institutions
    """)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:12px;'>"
    "ANN Student Evaluator • Built with Streamlit & scikit-learn"
    "</p>",
    unsafe_allow_html=True
)
