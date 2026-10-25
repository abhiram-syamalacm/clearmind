#!/usr/bin/env python3
"""
ClearMind — STEW cognitive workload classification pipeline
===========================================================

Runs the complete analysis for the paper and writes every number it needs
to results.json, plus publication figures.

USAGE
-----
    python clearmind_pipeline.py --data /path/to/STEW/folder

    # optional: also run ICA-based ocular artifact removal (slower, ~3x)
    python clearmind_pipeline.py --data /path/to/STEW --ica

OUTPUTS
-------
    results.json             every value the paper's tables need
    fig1_roc.png             ROC curves, all models
    fig2_shap_summary.png    SHAP beeswarm
    fig3_waterfall_*.png     two single-epoch attributions
    fig4_calibration.png     reliability diagram

METHODOLOGICAL NOTE
-------------------
Cross-validation is SUBJECT-WISE (GroupKFold). Epochs from one participant
never appear in both train and test. Random epoch-level splitting would let
the model memorise each participant's spectral fingerprint and report an
accuracy that does not generalise to a new user. This is the single most
important line in the file; do not change it to KFold.
"""

import argparse
import json
import os
import re
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
SFREQ = 128.0
CHANNELS = ["AF3", "F7", "F3", "FC5", "T7", "P7",
            "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}

FRONTAL = ["AF3", "AF4", "F3", "F4", "F7", "F8"]
PARIETAL = ["P7", "P8"]
ASYM_PAIRS = [("AF4", "AF3"), ("F4", "F3"), ("F8", "F7")]  # (right, left)

TRIM_SECONDS = 15.0     # paper: first and last 15 s excluded
EPOCH_SECONDS = 2.0
EPOCH_OVERLAP = 0.5     # 50%
REJECT_UV = 100.0       # amplitude threshold AFTER filtering
BANDPASS = (1.0, 40.0)

N_FOLDS = 5
N_TEST_SUBJECTS = 10    # held out entirely, never seen in CV
N_BOOT = 1000
RANDOM_STATE = 42

rng = np.random.default_rng(RANDOM_STATE)


def log(msg):
    print(msg, flush=True)


def section(msg):
    print(f"\n{'='*70}\n{msg}\n{'='*70}", flush=True)


# ----------------------------------------------------------------------------
# 1. Loading
# ----------------------------------------------------------------------------
def discover_files(data_dir):
    """Find STEW recordings and infer (subject, condition) from the filename."""
    found = []
    for root, _dirs, files in os.walk(data_dir):
        for fn in files:
            if not fn.lower().endswith((".txt", ".csv", ".dat")):
                continue
            low = fn.lower()
            if "rating" in low:          # ratings file, not EEG
                continue
            m = re.search(r"(\d+)", fn)
            if not m:
                continue
            subject = int(m.group(1))
            if re.search(r"(_|\b)(lo|low)(_|\.|\b)", low):
                cond = 0
            elif re.search(r"(_|\b)(hi|high)(_|\.|\b)", low):
                cond = 1
            else:
                continue
            found.append((subject, cond, os.path.join(root, fn)))
    return sorted(found)


def load_recording(path):
    """Load one recording as (n_channels, n_samples), auto-detecting layout."""
    arr = None
    for delim in [None, ",", "\t", ";"]:
        try:
            arr = np.loadtxt(path, delimiter=delim)
            break
        except Exception:
            continue
    if arr is None:
        raise IOError(f"could not parse {path}")
    if arr.ndim != 2:
        raise ValueError(f"{path}: expected 2-D, got shape {arr.shape}")

    n_ch = len(CHANNELS)
    if arr.shape[1] == n_ch:
        data = arr.T                 # rows were samples
    elif arr.shape[0] == n_ch:
        data = arr                   # rows were channels
    else:
        raise ValueError(
            f"{path}: neither axis is {n_ch} channels (shape {arr.shape}). "
            "Check the file format before continuing.")
    return np.ascontiguousarray(data, dtype=np.float64)


# ----------------------------------------------------------------------------
# 2. Preprocessing + epoching
# ----------------------------------------------------------------------------
def preprocess(data, use_ica=False):
    """Bandpass filter, average-reference. Returns (n_channels, n_samples)."""
    import mne
    info = mne.create_info(ch_names=CHANNELS, sfreq=SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)
    raw.set_montage("standard_1020", on_missing="ignore", verbose=False)
    raw.filter(BANDPASS[0], BANDPASS[1], fir_design="firwin",
               phase="zero", verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    if use_ica:
        ica = mne.preprocessing.ICA(n_components=min(12, len(CHANNELS) - 1),
                                    random_state=RANDOM_STATE,
                                    max_iter="auto", verbose=False)
        ica.fit(raw, verbose=False)
        # frontal channels sit closest to the eyes on the EPOC
        bad, _scores = ica.find_bads_eog(raw, ch_name=["AF3", "AF4"],
                                         verbose=False)
        ica.exclude = bad
        raw = ica.apply(raw, verbose=False)

    return raw.get_data()


def epoch(data):
    """Trim, then split into overlapping windows -> (n_epochs, n_ch, n_times)."""
    trim = int(TRIM_SECONDS * SFREQ)
    data = data[:, trim:-trim] if data.shape[1] > 2 * trim else data

    win = int(EPOCH_SECONDS * SFREQ)
    step = int(win * (1 - EPOCH_OVERLAP))
    n = (data.shape[1] - win) // step + 1
    if n <= 0:
        return np.empty((0, data.shape[0], win))
    return np.stack([data[:, i * step:i * step + win] for i in range(n)])


def reject(epochs):
    """Amplitude-threshold rejection. Returns (kept_epochs, n_rejected)."""
    if len(epochs) == 0:
        return epochs, 0
    peak = np.abs(epochs).max(axis=(1, 2))
    keep = peak < REJECT_UV
    return epochs[keep], int((~keep).sum())


# ----------------------------------------------------------------------------
# 3. Features
# ----------------------------------------------------------------------------
def build_feature_names():
    names = []
    for ch in CHANNELS:
        for b in BANDS:
            names.append(f"abs_{b}_{ch}")
    for ch in CHANNELS:
        for b in BANDS:
            names.append(f"rel_{b}_{ch}")
    for ch in FRONTAL:
        names.append(f"theta_alpha_ratio_{ch}")
    for r, l in ASYM_PAIRS:
        names.append(f"alpha_asym_{r}_{l}")
    names.append("frontal_theta_over_parietal_alpha")
    names.append("frontal_theta_mean")
    names.append("frontal_alpha_mean")
    return names


def extract_features(epochs):
    """(n_epochs, n_ch, n_times) -> (n_epochs, n_features)."""
    from scipy.signal import welch

    nperseg = int(SFREQ)            # 1 s window -> 1 Hz resolution
    freqs, psd = welch(epochs, fs=SFREQ, nperseg=nperseg,
                       noverlap=nperseg // 2, axis=-1)
    # psd: (n_epochs, n_ch, n_freqs)

    band_power = {}
    for b, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        try:
            integ = np.trapezoid
        except AttributeError:
            integ = np.trapz
        band_power[b] = integ(psd[:, :, mask], freqs[mask], axis=-1)

    total = sum(band_power.values()) + 1e-12
    ci = {ch: i for i, ch in enumerate(CHANNELS)}

    feats = []
    for ch in CHANNELS:                              # absolute (log)
        for b in BANDS:
            feats.append(np.log(band_power[b][:, ci[ch]] + 1e-12))
    for ch in CHANNELS:                              # relative
        for b in BANDS:
            feats.append(band_power[b][:, ci[ch]] / total[:, ci[ch]])
    for ch in FRONTAL:                               # theta/alpha
        feats.append(band_power["theta"][:, ci[ch]] /
                     (band_power["alpha"][:, ci[ch]] + 1e-12))
    for r, l in ASYM_PAIRS:                          # alpha asymmetry
        feats.append(np.log(band_power["alpha"][:, ci[r]] + 1e-12) -
                     np.log(band_power["alpha"][:, ci[l]] + 1e-12))

    f_theta = band_power["theta"][:, [ci[c] for c in FRONTAL]].mean(axis=1)
    p_alpha = band_power["alpha"][:, [ci[c] for c in PARIETAL]].mean(axis=1)
    f_alpha = band_power["alpha"][:, [ci[c] for c in FRONTAL]].mean(axis=1)
    feats.append(f_theta / (p_alpha + 1e-12))
    feats.append(np.log(f_theta + 1e-12))
    feats.append(np.log(f_alpha + 1e-12))

    return np.column_stack(feats)


# ----------------------------------------------------------------------------
# 4. Metrics helpers
# ----------------------------------------------------------------------------
def classification_metrics(y, prob, thresh=0.5):
    from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
    pred = (prob >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, prob)),
        "accuracy": float((tp + tn) / max(len(y), 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "precision": float(tp / max(tp + fp, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_auc_by_subject(y, prob, groups, n_boot=N_BOOT):
    """Resample SUBJECTS, not epochs — propagates between-subject variance."""
    from sklearn.metrics import roc_auc_score
    subs = np.unique(groups)
    idx_by_sub = {s: np.where(groups == s)[0] for s in subs}
    aucs = []
    for _ in range(n_boot):
        draw = rng.choice(subs, size=len(subs), replace=True)
        idx = np.concatenate([idx_by_sub[s] for s in draw])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], prob[idx]))
    if not aucs:
        return (float("nan"), float("nan"))
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder containing STEW files")
    ap.add_argument("--out", default=".", help="output folder")
    ap.add_argument("--ica", action="store_true", help="run ICA ocular removal")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import roc_curve, brier_score_loss, roc_auc_score
    from xgboost import XGBClassifier
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    results = {"config": {
        "sfreq": SFREQ, "bandpass": BANDPASS, "epoch_seconds": EPOCH_SECONDS,
        "epoch_overlap": EPOCH_OVERLAP, "trim_seconds": TRIM_SECONDS,
        "reject_uv": REJECT_UV, "n_folds": N_FOLDS,
        "n_test_subjects": N_TEST_SUBJECTS, "n_boot": N_BOOT,
        "welch_nperseg_s": 1.0, "welch_overlap_pct": 50,
        "ica_used": bool(args.ica), "random_state": RANDOM_STATE,
        "bands": {k: list(v) for k, v in BANDS.items()},
    }}

    # ---- load -------------------------------------------------------------
    section("1. LOADING")
    files = discover_files(args.data)
    if not files:
        log(f"ERROR: no STEW files found under {args.data}")
        log("Expected filenames containing a subject number and 'lo'/'hi'.")
        log("Run:  ls " + args.data + "   and send me the output.")
        sys.exit(1)
    subs = sorted({s for s, _, _ in files})
    log(f"found {len(files)} recordings across {len(subs)} subjects")

    X_list, y_list, g_list = [], [], []
    rejected = {0: 0, 1: 0}
    total_epochs_pre = {0: 0, 1: 0}

    for i, (sub, cond, path) in enumerate(files, 1):
        try:
            raw = load_recording(path)
            clean = preprocess(raw, use_ica=args.ica)
            eps = epoch(clean)
            total_epochs_pre[cond] += len(eps)
            eps, n_rej = reject(eps)
            rejected[cond] += n_rej
            if len(eps) == 0:
                log(f"  [{i}/{len(files)}] sub{sub:02d} cond{cond}: ALL REJECTED")
                continue
            X_list.append(extract_features(eps))
            y_list.append(np.full(len(eps), cond))
            g_list.append(np.full(len(eps), sub))
            if i % 12 == 0 or i == len(files):
                log(f"  [{i}/{len(files)}] processed")
        except Exception as e:
            log(f"  WARNING sub{sub:02d} cond{cond}: {e}")

    if not X_list:
        log("ERROR: every recording failed to load. Send me the warnings above.")
        sys.exit(1)

    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(g_list)
    feat_names = build_feature_names()

    assert X.shape[1] == len(feat_names), \
        f"feature count mismatch: {X.shape[1]} vs {len(feat_names)}"

    n_low, n_high = int((y == 0).sum()), int((y == 1).sum())
    rej_rate = {c: rejected[c] / max(total_epochs_pre[c], 1) for c in (0, 1)}

    log(f"\nepochs: {len(y)} total | {n_low} low-load | {n_high} high-load")
    log(f"features: {X.shape[1]}")
    log(f"rejected: low {rejected[0]} ({rej_rate[0]:.1%}) | "
        f"high {rejected[1]} ({rej_rate[1]:.1%})")
    if abs(rej_rate[0] - rej_rate[1]) > 0.05:
        log("  NOTE: rejection rates differ by >5 points between conditions.")
        log("  This is reportable — see Methods 2.2 in the draft.")

    results["data"] = {
        "n_subjects": len(subs), "n_recordings": len(files),
        "n_epochs_total": int(len(y)), "n_epochs_low": n_low,
        "n_epochs_high": n_high, "n_features": int(X.shape[1]),
        "n_rejected_low": rejected[0], "n_rejected_high": rejected[1],
        "rejection_rate_low": round(rej_rate[0], 4),
        "rejection_rate_high": round(rej_rate[1], 4),
        "class_balance_ratio": round(n_high / max(n_low, 1), 3),
    }

    # ---- split ------------------------------------------------------------
    section("2. SUBJECT-WISE SPLIT")
    n_test = min(N_TEST_SUBJECTS, max(2, len(subs) // 4))
    shuffled = rng.permutation(subs)
    test_subs = set(shuffled[:n_test].tolist())
    dev_mask = ~np.isin(groups, list(test_subs))
    test_mask = ~dev_mask

    Xd, yd, gd = X[dev_mask], y[dev_mask], groups[dev_mask]
    Xt, yt, gt = X[test_mask], y[test_mask], groups[test_mask]
    log(f"development: {len(yd)} epochs / {len(np.unique(gd))} subjects")
    log(f"held-out:    {len(yt)} epochs / {len(np.unique(gt))} subjects")
    results["data"]["test_subjects"] = sorted(int(s) for s in test_subs)
    results["config"]["n_test_subjects"] = int(n_test)

    models = {
        "logistic_regression": lambda: LogisticRegression(
            max_iter=2000, C=1.0, random_state=RANDOM_STATE),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1),
        "xgboost": lambda: XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, tree_method="hist"),
    }

    # ---- cross-validation -------------------------------------------------
    section("3. SUBJECT-WISE CROSS-VALIDATION")
    n_folds = min(N_FOLDS, len(np.unique(gd)))
    gkf = GroupKFold(n_splits=n_folds)
    results["config"]["n_folds"] = int(n_folds)
    results["models"] = {}

    for name, make in models.items():
        fold_aucs = []
        for tr, va in gkf.split(Xd, yd, groups=gd):
            sc = StandardScaler().fit(Xd[tr])
            m = make().fit(sc.transform(Xd[tr]), yd[tr])
            p = m.predict_proba(sc.transform(Xd[va]))[:, 1]
            fold_aucs.append(roc_auc_score(yd[va], p))
        results["models"][name] = {
            "cv_auc_mean": round(float(np.mean(fold_aucs)), 4),
            "cv_auc_std": round(float(np.std(fold_aucs)), 4),
            "cv_auc_folds": [round(float(a), 4) for a in fold_aucs],
            "cv_auc_min": round(float(np.min(fold_aucs)), 4),
            "cv_auc_max": round(float(np.max(fold_aucs)), 4),
        }
        log(f"  {name:22s} CV AUC {np.mean(fold_aucs):.3f} "
            f"± {np.std(fold_aucs):.3f}  "
            f"(range {np.min(fold_aucs):.3f}–{np.max(fold_aucs):.3f})")

    # ---- held-out ---------------------------------------------------------
    section("4. HELD-OUT EVALUATION")
    scaler = StandardScaler().fit(Xd)
    Xd_s, Xt_s = scaler.transform(Xd), scaler.transform(Xt)
    fitted, probs = {}, {}

    for name, make in models.items():
        m = make().fit(Xd_s, yd)
        p = m.predict_proba(Xt_s)[:, 1]
        fitted[name], probs[name] = m, p
        met = classification_metrics(yt, p)
        lo, hi = bootstrap_auc_by_subject(yt, p, gt)
        met["auc_ci_low"], met["auc_ci_high"] = round(lo, 4), round(hi, 4)
        met = {k: (round(v, 4) if isinstance(v, float) else v)
               for k, v in met.items()}
        results["models"][name]["heldout"] = met
        log(f"  {name:22s} AUC {met['auc']:.3f} "
            f"[{lo:.3f}–{hi:.3f}]  F1 {met['f1']:.3f}  "
            f"acc {met['accuracy']:.3f}")

    best = max(results["models"],
               key=lambda k: results["models"][k]["heldout"]["auc"])
    results["best_model"] = best
    log(f"\n  best by held-out AUC: {best}")

    xgb, p_xgb = fitted["xgboost"], probs["xgboost"]

    # ---- attribution ------------------------------------------------------
    section("5. FEATURE ATTRIBUTION")
    log("  computing SHAP values...")
    explainer = shap.TreeExplainer(xgb)
    sv = explainer.shap_values(Xt_s)
    if isinstance(sv, list):
        sv = sv[1]
    mean_abs = np.abs(sv).mean(axis=0)

    log("  computing permutation importance...")
    perm = permutation_importance(xgb, Xt_s, yt, n_repeats=10,
                                  random_state=RANDOM_STATE,
                                  scoring="roc_auc", n_jobs=-1)
    perm_mean = perm.importances_mean

    from scipy.stats import spearmanr
    rho, p_rho = spearmanr(mean_abs, perm_mean)
    results["attribution"] = {
        "spearman_rho_shap_vs_permutation": round(float(rho), 4),
        "spearman_p": float(p_rho),
    }
    log(f"  SHAP vs permutation agreement: Spearman rho = {rho:.3f} "
        f"(p = {p_rho:.2e})")

    shap_order = np.argsort(mean_abs)[::-1]
    perm_rank = {i: r + 1 for r, i in enumerate(np.argsort(perm_mean)[::-1])}

    top = []
    log("\n  top 10 features by mean |SHAP|:")
    for rank, i in enumerate(shap_order[:10], 1):
        corr = np.corrcoef(Xt_s[:, i], sv[:, i])[0, 1]
        direction = ("higher -> high load" if corr > 0
                     else "higher -> low load" if corr < 0 else "mixed")
        top.append({
            "rank": rank, "feature": feat_names[i],
            "mean_abs_shap": round(float(mean_abs[i]), 5),
            "permutation_importance": round(float(perm_mean[i]), 5),
            "permutation_rank": int(perm_rank[i]),
            "direction": direction,
        })
        log(f"    {rank:2d}. {feat_names[i]:38s} "
            f"|SHAP| {mean_abs[i]:.4f}  perm#{perm_rank[i]:<4d} {direction}")
    results["attribution"]["top_features"] = top

    # a-priori hypothesis check (Section 1.3 of the draft)
    top10 = [t["feature"] for t in top]
    theta_hits = [f for f in top10
                  if "theta" in f and ("ratio" in f or any(
                      c in f for c in FRONTAL) or "frontal" in f)]
    results["attribution"]["hypothesis_frontal_theta_in_top10"] = bool(theta_hits)
    results["attribution"]["frontal_theta_features_in_top10"] = theta_hits
    log(f"\n  a-priori hypothesis (frontal theta / theta-alpha in top 10): "
        f"{'SUPPORTED' if theta_hits else 'NOT SUPPORTED'}")
    if theta_hits:
        log(f"    {theta_hits}")
    else:
        log("    -> the Discussion must report this honestly. Check whether")
        log("       the top features look like ocular artifact (low-frequency")
        log("       power at AF3/AF4/F7/F8).")

    # ---- calibration ------------------------------------------------------
    section("6. CALIBRATION")
    brier = brier_score_loss(yt, p_xgb)
    deciles = []
    edges = np.linspace(0, 1, 11)
    for k in range(10):
        m = ((p_xgb >= edges[k]) & (p_xgb < edges[k + 1])) if k < 9 else \
            ((p_xgb >= edges[k]) & (p_xgb <= 1.0))
        deciles.append({
            "decile": k + 1,
            "range": [round(edges[k], 1), round(edges[k + 1], 1)],
            "mean_predicted": round(float(p_xgb[m].mean()), 4) if m.sum() else None,
            "observed_frequency": round(float(yt[m].mean()), 4) if m.sum() else None,
            "n": int(m.sum()),
        })
    results["calibration"] = {"brier_score": round(float(brier), 4),
                             "deciles": deciles}
    log(f"  Brier score: {brier:.4f}")
    for d in deciles:
        if d["n"]:
            log(f"    decile {d['decile']:2d}  pred {d['mean_predicted']:.3f}  "
                f"obs {d['observed_frequency']:.3f}  n={d['n']}")

    # ---- thresholds -------------------------------------------------------
    section("7. DECISION THRESHOLD")
    rows = []
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        m = classification_metrics(yt, p_xgb, thresh=t)
        rows.append({"threshold": t, "precision": round(m["precision"], 4),
                     "recall": round(m["sensitivity"], 4),
                     "f1": round(m["f1"], 4), "fp": m["fp"], "fn": m["fn"]})
        log(f"  t={t:.2f}  P {m['precision']:.3f}  R {m['sensitivity']:.3f}  "
            f"F1 {m['f1']:.3f}  FP {m['fp']:<5d} FN {m['fn']}")
    best_t = max(rows, key=lambda r: r["f1"])
    results["thresholds"] = {"table": rows, "f1_optimal": best_t["threshold"]}
    log(f"  F1 maximised at threshold {best_t['threshold']}")

    # ---- figures ----------------------------------------------------------
    section("8. FIGURES")

    plt.figure(figsize=(6, 5.5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(yt, p)
        a = results["models"][name]["heldout"]["auc"]
        plt.plot(fpr, tpr, lw=1.8,
                 label=f"{name.replace('_',' ')} (AUC {a:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — held-out participants"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(f"{args.out}/fig1_roc.png", dpi=300)
    plt.close(); log("  fig1_roc.png")

    plt.figure()
    shap.summary_plot(sv, Xt_s, feature_names=feat_names,
                      max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig2_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close(); log("  fig2_shap_summary.png")

    hi_i = int(np.argmax(np.where(yt == 1, p_xgb, -1)))
    lo_i = int(np.argmin(np.where(yt == 0, p_xgb, 2)))
    base = (explainer.expected_value if np.isscalar(explainer.expected_value)
            else explainer.expected_value[0])
    for tag, idx in [("high", hi_i), ("low", lo_i)]:
        try:
            plt.figure()
            shap.plots._waterfall.waterfall_legacy(
                base, sv[idx], feature_names=feat_names,
                max_display=12, show=False)
            plt.tight_layout()
            plt.savefig(f"{args.out}/fig3_waterfall_{tag}.png", dpi=300,
                        bbox_inches="tight")
            plt.close()
        except Exception as e:
            log(f"  (waterfall {tag} skipped: {e})")
    log("  fig3_waterfall_high.png / fig3_waterfall_low.png")
    results["attribution"]["waterfall_examples"] = {
        "high_load_epoch_prob": round(float(p_xgb[hi_i]), 4),
        "low_load_epoch_prob": round(float(p_xgb[lo_i]), 4),
    }

    obs = [d["observed_frequency"] for d in deciles if d["n"]]
    pred = [d["mean_predicted"] for d in deciles if d["n"]]
    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="perfect")
    plt.plot(pred, obs, "o-", lw=1.8, label=f"XGBoost (Brier {brier:.3f})")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed frequency")
    plt.title("Calibration — held-out participants"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{args.out}/fig4_calibration.png", dpi=300)
    plt.close(); log("  fig4_calibration.png")

    # ---- save -------------------------------------------------------------
    out_json = os.path.join(args.out, "results.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    section("DONE")
    log(f"results -> {out_json}")
    log(f"figures -> {args.out}/fig*.png")
    log("\nSend results.json back and the paper's tables can be filled in.")
    hb = results["models"][best]["heldout"]
    log(f"\nHeadline: {best} — held-out AUC {hb['auc']:.3f} "
        f"[{hb['auc_ci_low']:.3f}–{hb['auc_ci_high']:.3f}]")


if __name__ == "__main__":
    main()
