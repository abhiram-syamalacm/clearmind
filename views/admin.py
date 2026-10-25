"""ClearMind admin backend: student data access + the EEG workload model.

Locked behind ADMIN_PASSWORD (Streamlit secrets or environment). If it isn't set,
the page refuses to open.
"""
import hmac
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

import storage
from eeg_core import EEGModel
from model_core import QUESTIONS, feature_label
from ui import brand_header

with st.sidebar:
    st.page_link("views/home.py", label="Home", icon=":material/home:")
brand_header("admin")


def admin_password():
    try:
        pw = st.secrets.get("ADMIN_PASSWORD")
    except Exception:  # no secrets.toml present
        pw = None
    return pw or os.environ.get("ADMIN_PASSWORD")


# ---------------------------------------------------------------- login
expected = admin_password()
if not expected:
    st.error("Admin is disabled: set `ADMIN_PASSWORD` in `.streamlit/secrets.toml` "
             "(or the Streamlit Cloud Secrets box) and restart.")
    st.stop()

if not st.session_state.get("admin"):
    fails = st.session_state.setdefault("admin_fails", 0)
    with st.form("admin_login"):
        pw = st.text_input("Admin password", type="password")
        if st.form_submit_button("Log in", type="primary"):
            if fails >= 5:
                st.error("Too many attempts. Refresh the page and try again later.")
            elif hmac.compare_digest(pw.encode(), expected.encode()):
                st.session_state.admin = True
                st.session_state.admin_fails = 0
                st.rerun()
            else:
                st.session_state.admin_fails = fails + 1
                time.sleep(1)  # slow down guessing
                st.error("Wrong password.")
    st.stop()

with st.sidebar:
    if st.button("Log out of admin"):
        st.session_state.admin = False
        st.rerun()

tab_students, tab_eeg, tab_model = st.tabs(["Student data", "EEG model", "Model report"])


# ---------------------------------------------------------------- student data
with tab_students:
    users = storage.list_users()
    checks = storage.all_checkins()

    n_guest = sum(u["type"] == "guest" for u in users)
    c1, c2, c3 = st.columns(3)
    c1.metric("Students surveyed", sum(u["checkins"] > 0 for u in users),
              help=f"{len(users) - n_guest} accounts + {n_guest} guests ({len(users)} total sign-ups)")
    c2.metric("Check-ins", len(checks), help=f"{storage.reminder_count()} students get reminder emails")
    c3.metric("Avg latest risk",
              f"{pd.Series([u['latest_risk'] for u in users if u['latest_risk'] is not None]).mean():.0%}"
              if any(u["latest_risk"] is not None for u in users) else "–")

    if checks:
        df = pd.DataFrame([{"time": pd.to_datetime(c["created_at"]).tz_convert(None),
                            "risk": c["risk"], "band": c["band"]} for c in checks])
        st.markdown("**Band mix across all check-ins**")
        st.altair_chart(alt.Chart(df).mark_bar(cornerRadius=4).encode(
            x=alt.X("band:N", sort=["low", "medium", "high"], title=None),
            y=alt.Y("count():Q", title="Check-ins"),
            color=alt.Color("band:N", legend=None, scale=alt.Scale(
                domain=["low", "medium", "high"], range=["#1557F4", "#FFC718", "#E5484D"])),
        ).properties(height=220), width="stretch")

        export = pd.DataFrame([{"user_id": c["user_id"], "created_at": c["created_at"],
                                "risk": c["risk"], "band": c["band"],
                                **{f"answer_{k}": v for k, v in c["answers"].items()}}
                               for c in checks])
        st.download_button("Export all check-ins (de-identified CSV)", export.to_csv(index=False).encode(),
                           "clearmind_all_checkins.csv", "text/csv")

    st.divider()
    st.markdown("**Find a student** (for data access or deletion requests)")
    if not users:
        st.info("No student accounts yet.")
    else:
        st.dataframe(pd.DataFrame(users).drop(columns="id").style.format(
            {"latest_risk": lambda v: "–" if v is None else f"{v:.0%}"}), hide_index=True)
        name = st.selectbox("Student", [u["username"] for u in users], index=None,
                            placeholder="Choose a username")
        if name:
            u = next(u for u in users if u["username"] == name)
            hist = storage.get_checkins(u["id"])
            if hist:
                sdf = pd.DataFrame([{"created_at": h["created_at"], "risk": round(h["risk"], 3),
                                     "band": h["band"], **{feature_label(k) if k in QUESTIONS else k: v
                                                          for k, v in h["answers"].items()}}
                                    for h in hist])
                st.dataframe(sdf, hide_index=True)
                st.download_button(f"Download {name}'s data (CSV)", sdf.to_csv(index=False).encode(),
                                   f"clearmind_{name}.csv", "text/csv")
            else:
                st.caption("No check-ins yet.")
            with st.expander(f"Delete {name}'s account"):
                st.write("Permanently deletes the account and every check-in. Use this for deletion requests.")
                confirm = st.text_input("Type the username to confirm")
                if st.button("Delete permanently", type="primary", disabled=confirm != name):
                    storage.delete_user(u["id"])
                    st.success(f"Deleted {name}.")
                    st.rerun()


# ---------------------------------------------------------------- EEG model
@st.cache_resource
def load_eeg():
    return EEGModel()


eeg = load_eeg()

with tab_eeg:
    st.subheader("Score an EEG recording")
    st.caption("STEW format: a text file of 14 Emotiv channels (AF3, F7, F3, FC5, T7, P7, O1, O2, "
               "P8, T8, FC6, F4, F8, AF4) sampled at 128 Hz, one sample per row. Processing uses "
               "the paper's pipeline exactly: 1-40 Hz filter, average reference, 15 s trim, "
               "2 s epochs with 50% overlap, 100 µV rejection.")
    up = st.file_uploader("Recording", type=["txt", "csv", "dat"])
    if up:
        with st.spinner("Filtering, epoching and scoring..."):
            try:
                r = eeg.predict_file(up.getvalue(), suffix="." + up.name.rsplit(".", 1)[-1])
            except Exception as e:
                st.error(f"Couldn't score this file: {e}")
                st.stop()
        c1, c2, c3 = st.columns(3)
        c1.metric("High-workload probability", f"{r['mean_prob']:.0%}")
        c2.metric("Epochs scored as high", f"{r['share_high']:.0%}")
        c3.metric("Epochs kept", f"{r['n_epochs'] - r['n_rejected']} / {r['n_epochs']}")
        st.caption(f"{r['duration_s']:.0f} s recording. Probability is averaged over clean 2 s epochs.")

        tl = pd.DataFrame({"Epoch": range(1, len(r["epoch_probs"]) + 1), "P(high load)": r["epoch_probs"]})
        st.markdown("**Workload over the recording**")
        st.altair_chart(alt.Chart(tl).mark_line(color="#1557F4", point=True).encode(
            x="Epoch:Q", y=alt.Y("P(high load):Q", scale=alt.Scale(domain=[0, 1]))
        ).properties(height=220), width="stretch")

        st.markdown("**Features driving this prediction** (average log-odds contribution)")
        cdf = pd.DataFrame(r["top_contributions"], columns=["Feature", "Contribution"])
        cdf["Direction"] = cdf["Contribution"].map(lambda v: "Toward high load" if v > 0 else "Toward low load")
        st.altair_chart(alt.Chart(cdf).mark_bar(cornerRadius=4).encode(
            x=alt.X("Contribution:Q", title="Log-odds"),
            y=alt.Y("Feature:N", sort=cdf["Feature"].tolist(), title=None, axis=alt.Axis(labelLimit=260)),
            color=alt.Color("Direction:N", legend=alt.Legend(orient="bottom", title=None),
                            scale=alt.Scale(domain=["Toward high load", "Toward low load"],
                                            range=["#FFC718", "#1557F4"])),
        ).properties(height=alt.Step(26)), width="stretch")
        if r["n_rejected"] / max(r["n_epochs"], 1) > 0.5:
            st.warning("Over half the epochs were rejected as artifact -- treat this prediction with caution.")


# ---------------------------------------------------------------- model report
with tab_model:
    m = eeg.meta
    h, cv, d = m["heldout_metrics"], m["cv_metrics"], m["data_summary"]
    st.subheader("EEG workload model")
    st.markdown(f"Standardised **logistic regression** on {len(m['feature_names'])} spectral features "
                f"(band powers, relative powers, theta/alpha ratios, alpha asymmetry). Deployed model "
                f"is fit on all {m['n_training_epochs']:,} clean epochs from the STEW dataset.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Held-out AUC", f"{h['auc']:.3f}", help=f"95% CI {h['auc_ci_low']:.3f}–{h['auc_ci_high']:.3f}, "
              "subject-level bootstrap")
    c2.metric("Held-out accuracy", f"{h['accuracy']:.1%}")
    c3.metric("Subject-wise CV AUC", f"{cv['cv_auc_mean']:.3f} ± {cv['cv_auc_std']:.3f}")
    st.caption(f"Held-out = {len(d['test_subjects'])} participants never seen in training or CV "
               f"(GroupKFold by subject). 95% CI {h['auc_ci_low']:.3f}–{h['auc_ci_high']:.3f}.")
    st.markdown(f"""
| | |
|---|---|
| Subjects / recordings | {d['n_subjects']} / {d['n_recordings']} |
| Clean epochs (low / high) | {d['n_epochs_low']:,} / {d['n_epochs_high']:,} |
| Epochs rejected (low / high) | {d['rejection_rate_low']:.1%} / {d['rejection_rate_high']:.1%} |
| Sensitivity / specificity | {h['sensitivity']:.3f} / {h['specificity']:.3f} |
| F1 | {h['f1']:.3f} |
""")
    st.info("Research model: it was trained on 48 adults doing one lab task (STEW/SIMKAP) with a "
            "14-channel Emotiv headset. Predictions on other headsets, tasks or people are unvalidated.")
