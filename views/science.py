"""The science behind ClearMind: how the survey score is calculated, and the EEG research."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from model_core import RiskModel, feature_label
from ui import brand_header

ROOT = Path(__file__).parent.parent


@st.cache_resource
def load_model():
    return RiskModel()


model = load_model()
eeg = json.loads((ROOT / "models" / "eeg_model.json").read_text())
h, d = eeg["heldout_metrics"], eeg["data_summary"]

with st.sidebar:
    st.page_link("views/home.py", label="Home", icon=":material/home:")
    st.page_link("views/survey.py", label="Take the survey", icon=":material/edit_note:")

brand_header("the science behind")
tab_score, tab_eeg = st.tabs(["How your score works", "The EEG research"])


# ---------------------------------------------------------------- survey score
with tab_score:
    meta = model.meta
    st.subheader("How the score is calculated")
    st.markdown(r"""
**1. From your answers to model inputs.** Each 1-10 answer $a$ is mapped linearly onto the scale
the feature uses in the training data, $x = x_{\min} + \frac{a-1}{9}(x_{\max}-x_{\min})$.
Anxiety uses the GAD-7 range (0-21); the others use 0-5. *"Prefer not to say"* is passed as a
missing value, which the model handles natively by following each tree's learned default branch.

**2. Gradient-boosted trees.** The model is XGBoost, an ensemble of $K$ decision trees $f_k$.
Each tree adds a small correction, and the total is a log-odds score:
$$z(\mathbf{x}) = \sum_{k=1}^{K} f_k(\mathbf{x})$$

**3. Log-odds to probability.** The logistic (sigmoid) function turns that into a risk between 0 and 1:
$$\text{risk} = \sigma(z) = \frac{1}{1+e^{-z}}$$
Bands: **low** < 40%, **medium** 40-70%, **high** ≥ 70%.

**4. Explaining the score with SHAP.** SHAP (SHapley Additive exPlanations) splits $z$ into one
contribution per input, using Shapley values from cooperative game theory -- each feature's
average marginal effect over every possible ordering of features:
$$\phi_i = \sum_{S \subseteq F\setminus\{i\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!}\,\big[v(S\cup\{i\}) - v(S)\big]$$
For tree models these are computed *exactly* with the TreeSHAP algorithm, and they add up
precisely to the model output: $z = \phi_0 + \sum_i \phi_i$, where $\phi_0$ is the
average prediction. That's what the "What's driving your score" chart shows.
""")

    r = st.session_state.get("last_result")
    if r:
        st.markdown("**Your latest check-in, worked through:**")
        rows = [{"Term": "Baseline φ₀", "Value": r["base_logit"]}] + [
            {"Term": f"φ ({feature_label(f)})", "Value": v} for f, v in r["shap"].items()]
        rows.append({"Term": "Total z", "Value": r["logit"]})
        st.dataframe(pd.DataFrame(rows).style.format({"Value": "{:+.3f}"}), hide_index=True)
        st.latex(rf"\sigma({r['logit']:.3f}) = \frac{{1}}{{1+e^{{{-r['logit']:.3f}}}}} = {r['risk']:.3f}")
    else:
        st.caption("Take the survey to see your own numbers worked through here.")

    with st.expander("About this model"):
        st.json({"source": meta.get("source"), "target": meta.get("target"),
                 "features": meta["feature_names"], "metrics": meta.get("metrics")})
    st.markdown("""
**Limitations.** The model was trained on a public Kaggle survey dataset of student stress
factors, not on real exam outcomes, so it estimates *stress risk*, not your grade. Mapping a
1-10 rating onto scales like GAD-7 is an approximation. This tool is educational and is not
a medical or psychological assessment.
""")


# ---------------------------------------------------------------- EEG research
with tab_eeg:
    st.subheader("Reading workload from brain activity")
    st.write(f"Alongside the student tool, ClearMind studies cognitive workload directly from brain "
             f"activity. Using the public STEW dataset, {d['n_subjects']} adults wore a 14-sensor EEG "
             f"headset twice: once resting, and once doing a demanding multitasking test. Our model "
             f"learned to tell the two apart from brain-wave patterns.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Held-out AUC", f"{h['auc']:.3f}", help=f"95% CI {h['auc_ci_low']:.3f}–{h['auc_ci_high']:.3f}")
    m2.metric("Accuracy", f"{h['accuracy']:.1%}")
    m3.metric("EEG epochs analysed", f"{d['n_epochs_total']:,}")

    st.markdown("""
**Tested on people it had never seen.** Everyone's brain waves have a personal fingerprint, so
we kept each person entirely in training or entirely in testing. That makes the score harder to
earn, but it's the honest measure of how the model would work for someone new.

**What the model relies on.** Frontal theta (4–8 Hz brain waves at the front of the head), which
decades of research link to working memory, was among the strongest signals, and stayed strong
after removing eye-movement artifacts. The single strongest feature, theta at the back of the
head, was a surprise we're still investigating.

**What it doesn't show yet.** The "low workload" condition was rest rather than an easy task,
the final test group was small (9–10 people), and everyone used the same headset. The survey on
this site does not use brain data.
""")
    c1, c2 = st.columns(2)
    c1.image(str(ROOT / "assets" / "science" / "fig1_roc.png"),
             caption="ROC curves on held-out participants")
    c2.image(str(ROOT / "assets" / "science" / "fig2_shap_summary.png"),
             caption="Which EEG features drove predictions (SHAP)")
