import csv
import wave
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ttest_1samp, friedmanchisquare
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

from spafe.features.gfcc import gfcc
from statsmodels.stats.anova import AnovaRM

# pip install numpy matplotlib scipy scikit-learn statsmodels pandas spafe

# ============================================================
# Configuration
# ============================================================

DATASET_DIR = Path("i2s_sonification_dataset")
MANIFEST_PATH = DATASET_DIR / "manifest.csv"
RESULTS_DIR = Path("i2s_analysis_results_gfcc_left_right_joint")

EPS = 1e-12
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 12345

FRAME_SIZE = 2048
HOP_SIZE = 512
GFCC_NUM_CEPS = 13

# Keep GFCC coefficient means in the features CSV, but exclude them from
# distance/PCA/clustering/statistical modelling by default.
INCLUDE_GFCC_MEANS_IN_ANALYSIS = False

REPRESENTATIONS = ["left", "right", "joint"]


# ============================================================
# WAV loading
# ============================================================

def read_wav_stereo_int16(path):
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth != 2:
        raise ValueError("Expected 16-bit WAV: {}".format(path))
    if n_channels != 2:
        raise ValueError("Expected stereo WAV: {}".format(path))

    data = np.frombuffer(raw, dtype=np.int16).reshape(-1, n_channels)
    data = data.astype(np.float32) / 32767.0
    return data, framerate


# ============================================================
# Feature extraction
# ============================================================

def frame_signal(x, frame_size, hop_size):
    if len(x) < frame_size:
        pad = np.zeros(frame_size - len(x), dtype=x.dtype)
        x = np.concatenate([x, pad])

    num_frames = 1 + int(np.floor((len(x) - frame_size) / hop_size))
    frames = np.zeros((num_frames, frame_size), dtype=np.float32)

    for i in range(num_frames):
        start = i * hop_size
        frames[i, :] = x[start:start + frame_size]

    return frames


def spectral_centroid_feature(x, fs, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    frames = frame_signal(x, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / fs)

    centroids = []
    for frame in frames:
        mag = np.abs(np.fft.rfft(frame * window))
        mag_sum = np.sum(mag) + EPS
        centroid = np.sum(freqs * mag) / mag_sum
        centroids.append(centroid)

    centroids = np.array(centroids, dtype=np.float32)
    return float(np.mean(centroids))


def spectral_flux_feature(x, frame_size=FRAME_SIZE, hop_size=HOP_SIZE):
    frames = frame_signal(x, frame_size, hop_size)
    window = np.hanning(frame_size).astype(np.float32)

    prev_mag = None
    flux_values = []

    for frame in frames:
        mag = np.abs(np.fft.rfft(frame * window)).astype(np.float32)
        mag = mag / (np.sum(mag) + EPS)

        if prev_mag is not None:
            diff = mag - prev_mag
            flux = np.sqrt(np.sum(diff * diff))
            flux_values.append(flux)

        prev_mag = mag

    if not flux_values:
        return 0.0

    return float(np.mean(flux_values))


def autocorrelation_features(x, fs):
    x = x.astype(np.float32)
    x = x - np.mean(x)

    if np.max(np.abs(x)) < EPS:
        return 0.0, 0.0

    acf_full = np.correlate(x, x, mode="full")
    acf = acf_full[len(acf_full) // 2:]
    acf = acf / (acf[0] + EPS)

    min_lag = 1
    max_lag = min(len(acf), int(fs * 0.1))
    if max_lag <= min_lag:
        return 0.0, 0.0

    search = acf[min_lag:max_lag]
    idx = int(np.argmax(search))
    peak_height = float(search[idx])
    peak_lag = idx + min_lag
    peak_lag_seconds = float(peak_lag / float(fs))

    return peak_height, peak_lag_seconds


def gfcc_features(x, fs, num_ceps=GFCC_NUM_CEPS):
    """
    Compute GFCCs and return summary statistics per coefficient.
    """
    try:
        coeffs = gfcc(
            sig=x,
            fs=fs,
            num_ceps=num_ceps,
            nfilts=26,
            nfft=FRAME_SIZE
        )
    except Exception as e:
        print("GFCC failed: {}".format(e))
        return {}

    means = np.mean(coeffs, axis=0)

    features = {}
    for i in range(num_ceps):
        features["gfcc_mean_{}".format(i)] = float(means[i])

    gfcc_energy = float(np.mean(np.abs(coeffs)))
    gfcc_variability = float(np.mean(np.std(coeffs, axis=0)))

    features["gfcc_energy"] = gfcc_energy
    features["gfcc_variability"] = gfcc_variability

    return features


def extract_channel_features(x, fs, prefix):
    """
    Extract compact symmetric features for one channel.

    Returns:
      spectral_centroid_<prefix>
      spectral_flux_<prefix>
      acf_peak_height_<prefix>
      acf_peak_lag_sec_<prefix>
      gfcc_energy_<prefix>
      gfcc_variability_<prefix>
      and optionally gfcc_mean_*_<prefix> for CSV transparency
    """
    centroid = spectral_centroid_feature(x, fs)
    flux = spectral_flux_feature(x, fs)
    acf_peak_height, acf_peak_lag_sec = autocorrelation_features(x, fs)
    gfcc_feats = gfcc_features(x, fs)

    features = {
        "spectral_centroid_{}".format(prefix): centroid,
        "spectral_flux_{}".format(prefix): flux,
        "acf_peak_height_{}".format(prefix): acf_peak_height,
        "acf_peak_lag_sec_{}".format(prefix): acf_peak_lag_sec,
        "gfcc_energy_{}".format(prefix): gfcc_feats.get("gfcc_energy", 0.0),
        "gfcc_variability_{}".format(prefix): gfcc_feats.get("gfcc_variability", 0.0),
    }

    for k, v in gfcc_feats.items():
        if k.startswith("gfcc_mean_"):
            features["{}_{}".format(k, prefix)] = v

    return features


def extract_features_from_wav(path):
    stereo, fs = read_wav_stereo_int16(path)
    left = stereo[:, 0]
    right = stereo[:, 1]

    features = {}
    features.update(extract_channel_features(left, fs, "left"))
    features.update(extract_channel_features(right, fs, "right"))
    return features


# ============================================================
# Dataset loading
# ============================================================

def load_manifest(manifest_path):
    rows = []
    with open(str(manifest_path), "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["repetition"] = int(row["repetition"])
            row["oversample"] = int(row["oversample"])
            row["output_seconds"] = float(row["output_seconds"])
            row["fs_out"] = int(row["fs_out"])
            rows.append(row)
    return rows


def build_feature_table(dataset_dir, manifest_rows):
    feature_rows = []

    for i, row in enumerate(manifest_rows, start=1):
        wav_path = dataset_dir / row["filename"]
        features = extract_features_from_wav(wav_path)

        out_row = dict(row)
        out_row.update(features)
        feature_rows.append(out_row)

        print("[{0:03d}/{1}] Extracted features from {2}".format(
            i, len(manifest_rows), row["filename"]
        ))

    return feature_rows


# ============================================================
# Analysis helpers
# ============================================================

def rows_for_oversample(feature_rows, oversample):
    return [r for r in feature_rows if r["oversample"] == oversample]


def get_representation_feature_names(representation, rows):
    base_left = [
        "spectral_centroid_left",
        "spectral_flux_left",
        "acf_peak_height_left",
        "acf_peak_lag_sec_left",
        "gfcc_energy_left",
        "gfcc_variability_left",
    ]

    base_right = [
        "spectral_centroid_right",
        "spectral_flux_right",
        "acf_peak_height_right",
        "acf_peak_lag_sec_right",
        "gfcc_energy_right",
        "gfcc_variability_right",
    ]

    if representation == "left":
        feature_names = base_left[:]
    elif representation == "right":
        feature_names = base_right[:]
    elif representation == "joint":
        feature_names = base_left + base_right
    else:
        raise ValueError("Unknown representation: {}".format(representation))

    feature_names = [f for f in feature_names if any(f in r for r in rows)]

    if INCLUDE_GFCC_MEANS_IN_ANALYSIS:
        extra = []
        for row in rows:
            for k in row.keys():
                if representation == "left" and k.startswith("gfcc_mean_") and k.endswith("_left"):
                    extra.append(k)
                elif representation == "right" and k.startswith("gfcc_mean_") and k.endswith("_right"):
                    extra.append(k)
                elif representation == "joint" and k.startswith("gfcc_mean_") and (
                    k.endswith("_left") or k.endswith("_right")
                ):
                    extra.append(k)
        feature_names.extend(sorted(set(extra)))

    return feature_names


def make_feature_matrix(rows, feature_names):
    X = np.array([
        [r.get(name, 0.0) for name in feature_names]
        for r in rows
    ], dtype=np.float64)

    labels = [r["condition"] for r in rows]
    names = [r["filename"] for r in rows]

    return X, labels, names, feature_names


def zscore_features(X):
    mu = np.mean(X, axis=0, keepdims=True)
    sigma = np.std(X, axis=0, keepdims=True)
    sigma[sigma < EPS] = 1.0
    Z = (X - mu) / sigma
    return Z, mu, sigma


def compute_distance_matrix(Xz):
    return squareform(pdist(Xz, metric="euclidean"))


def compute_class_distance_metrics(distance_matrix, labels):
    n = len(labels)
    within = []
    between = []

    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within.append(distance_matrix[i, j])
            else:
                between.append(distance_matrix[i, j])

    mean_within = float(np.mean(within)) if within else 0.0
    mean_between = float(np.mean(between)) if between else 0.0
    separation_ratio = float(mean_between / (mean_within + EPS))

    return {
        "mean_within_class_distance": mean_within,
        "mean_between_class_distance": mean_between,
        "separation_ratio": separation_ratio,
    }


def compute_silhouette(Xz, labels):
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return float("nan")
    return float(silhouette_score(Xz, labels))


def compute_silhouette_samples_safe(Xz, labels):
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return np.full(len(labels), np.nan, dtype=np.float64)

    counts = {lbl: labels.count(lbl) for lbl in unique_labels}
    if min(counts.values()) < 2:
        return np.full(len(labels), np.nan, dtype=np.float64)

    return silhouette_samples(Xz, labels)


def save_rows_csv(rows, output_path):
    if not rows:
        return

    # Collect the union of all keys across all rows, preserving first-seen order.
    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(str(output_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ============================================================
# Effect sizes and bootstrap helpers
# ============================================================

def partial_eta_squared_from_f(f_stat, df_effect, df_error):
    denom = (f_stat * df_effect) + df_error
    if denom <= EPS:
        return float("nan")
    return float((f_stat * df_effect) / denom)


def kendalls_w_from_friedman(chi_square, n_cases, k_conditions):
    denom = n_cases * (k_conditions - 1.0)
    if denom <= EPS:
        return float("nan")
    return float(chi_square / denom)


def bootstrap_mean_ci(values, n_boot=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(values[idx])

    mean_val = float(np.mean(values))
    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    return mean_val, ci_low, ci_high


# ============================================================
# Repeated-measures inferential statistics
# ============================================================

def make_case_id(row):
    return "{}__{}__rep-{}".format(row["condition"], row["payload"], row["repetition"])


def run_trend_analysis(df, variables, oversamples, output_dir):
    slope_rows = []
    summary_rows = []

    x = np.array(oversamples, dtype=np.float64)

    for var in variables:
        wide = df.pivot(index="case_id", columns="oversample", values=var)

        missing_cols = [os for os in oversamples if os not in wide.columns]
        if missing_cols:
            summary_rows.append({
                "variable": var,
                "n_cases": 0,
                "mean_slope": "",
                "slope_t_statistic": "",
                "slope_p_value": "",
                "slope_ci_low": "",
                "slope_ci_high": "",
                "mean_r_squared": "",
                "status": "missing_oversample_columns",
            })
            continue

        wide = wide[oversamples].dropna()

        if len(wide) == 0:
            summary_rows.append({
                "variable": var,
                "n_cases": 0,
                "mean_slope": "",
                "slope_t_statistic": "",
                "slope_p_value": "",
                "slope_ci_low": "",
                "slope_ci_high": "",
                "mean_r_squared": "",
                "status": "no_complete_cases",
            })
            continue

        case_slopes = []
        case_r2 = []

        for case_id, row in wide.iterrows():
            y = row.values.astype(np.float64)

            if np.allclose(y, y[0], atol=1e-15, rtol=0.0):
                slope = 0.0
                intercept = float(y[0])
                r2 = 1.0
            else:
                slope, intercept = np.polyfit(x, y, 1)
                y_hat = slope * x + intercept
                ss_res = np.sum((y - y_hat) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot < EPS:
                    r2 = 1.0
                else:
                    r2 = float(1.0 - (ss_res / ss_tot))

            case_slopes.append(float(slope))
            case_r2.append(float(r2))

            slope_rows.append({
                "variable": var,
                "case_id": case_id,
                "slope_per_oversample_unit": float(slope),
                "r_squared": float(r2),
            })

        case_slopes = np.array(case_slopes, dtype=np.float64)
        case_r2 = np.array(case_r2, dtype=np.float64)

        if np.allclose(case_slopes, 0.0, atol=1e-15, rtol=0.0):
            t_stat = 0.0
            p_value = 1.0
            mean_slope, ci_low, ci_high = 0.0, 0.0, 0.0
            status = "constant_or_degenerate"
        else:
            t_stat, p_value = ttest_1samp(case_slopes, 0.0)
            mean_slope, ci_low, ci_high = bootstrap_mean_ci(
                case_slopes,
                n_boot=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED + 999
            )
            status = "ok"

        summary_rows.append({
            "variable": var,
            "n_cases": int(len(case_slopes)),
            "mean_slope": float(mean_slope),
            "slope_t_statistic": float(t_stat),
            "slope_p_value": float(p_value),
            "slope_ci_low": float(ci_low),
            "slope_ci_high": float(ci_high),
            "mean_r_squared": float(np.mean(case_r2)),
            "status": status,
        })

    save_rows_csv(slope_rows, output_dir / "trend_linear_slopes_by_case.csv")
    save_rows_csv(summary_rows, output_dir / "trend_linear_summary.csv")

    return slope_rows, summary_rows


def get_inferential_variables_for_representation(representation, df_columns):
    if representation == "left":
        variables = [
            "spectral_centroid_left",
            "spectral_flux_left",
            "acf_peak_height_left",
            "acf_peak_lag_sec_left",
            "gfcc_energy_left",
            "gfcc_variability_left",
            "sample_silhouette",
        ]
    elif representation == "right":
        variables = [
            "spectral_centroid_right",
            "spectral_flux_right",
            "acf_peak_height_right",
            "acf_peak_lag_sec_right",
            "gfcc_energy_right",
            "gfcc_variability_right",
            "sample_silhouette",
        ]
    elif representation == "joint":
        variables = [
            "spectral_centroid_left",
            "spectral_flux_left",
            "acf_peak_height_left",
            "acf_peak_lag_sec_left",
            "gfcc_energy_left",
            "gfcc_variability_left",
            "spectral_centroid_right",
            "spectral_flux_right",
            "acf_peak_height_right",
            "acf_peak_lag_sec_right",
            "gfcc_energy_right",
            "gfcc_variability_right",
            "sample_silhouette",
        ]
    else:
        raise ValueError("Unknown representation: {}".format(representation))

    if INCLUDE_GFCC_MEANS_IN_ANALYSIS:
        for c in sorted(df_columns):
            if representation == "left" and c.startswith("gfcc_mean_") and c.endswith("_left"):
                variables.append(c)
            elif representation == "right" and c.startswith("gfcc_mean_") and c.endswith("_right"):
                variables.append(c)
            elif representation == "joint" and c.startswith("gfcc_mean_") and (
                c.endswith("_left") or c.endswith("_right")
            ):
                variables.append(c)

    return [v for v in variables if v in df_columns]


def run_repeated_measures_stats(inferential_rows, output_dir, representation):
    """
    Repeated-measures ANOVA, Friedman test and trend analysis for a
    single representation (left, right, or joint).

    Subject/block = base synthetic case:
      condition + payload + repetition
    """
    df = pd.DataFrame(inferential_rows)
    if len(df) == 0:
        return [], [], []

    df["case_id"] = df.apply(make_case_id, axis=1)

    variables = get_inferential_variables_for_representation(representation, df.columns)

    key_cols = ["case_id", "oversample"]
    if df.duplicated(subset=key_cols).any():
        raise ValueError(
            "Duplicate case_id/oversample rows found in inferential data for representation={}".format(
                representation
            )
        )

    oversamples = sorted(df["oversample"].unique())

    # ----------------------------
    # Repeated-measures ANOVA
    # ----------------------------
    anova_rows = []

    for var in variables:
        sub = df[["case_id", "oversample", var]].copy()
        sub = sub[np.isfinite(sub[var].values)]

        if len(sub) == 0:
            anova_rows.append({
                "representation": representation,
                "variable": var,
                "groups_compared": ",".join(str(x) for x in oversamples),
                "f_statistic": "",
                "p_value": "",
                "num_df": "",
                "den_df": "",
                "partial_eta_squared": "",
                "status": "no_finite_values",
            })
            continue

        if np.allclose(sub[var].values, sub[var].values[0], atol=1e-15, rtol=0.0):
            anova_rows.append({
                "representation": representation,
                "variable": var,
                "groups_compared": ",".join(str(x) for x in oversamples),
                "f_statistic": "",
                "p_value": "",
                "num_df": "",
                "den_df": "",
                "partial_eta_squared": "",
                "status": "constant_or_degenerate",
            })
            continue

        try:
            aov = AnovaRM(
                data=sub,
                depvar=var,
                subject="case_id",
                within=["oversample"]
            ).fit()

            table = aov.anova_table
            f_stat = float(table.loc["oversample", "F Value"])
            p_value = float(table.loc["oversample", "Pr > F"])
            num_df = float(table.loc["oversample", "Num DF"])
            den_df = float(table.loc["oversample", "Den DF"])
            eta_p2 = partial_eta_squared_from_f(f_stat, num_df, den_df)

            anova_rows.append({
                "representation": representation,
                "variable": var,
                "groups_compared": ",".join(str(x) for x in oversamples),
                "f_statistic": f_stat,
                "p_value": p_value,
                "num_df": num_df,
                "den_df": den_df,
                "partial_eta_squared": eta_p2,
                "status": "ok",
            })

        except Exception:
            anova_rows.append({
                "representation": representation,
                "variable": var,
                "groups_compared": ",".join(str(x) for x in oversamples),
                "f_statistic": "",
                "p_value": "",
                "num_df": "",
                "den_df": "",
                "partial_eta_squared": "",
                "status": "failed_or_degenerate",
            })

    save_rows_csv(anova_rows, output_dir / "anova_repeated_measures.csv")

    # ----------------------------
    # Friedman test
    # ----------------------------
    friedman_rows = []

    for var in variables:
        wide = df.pivot(index="case_id", columns="oversample", values=var)

        missing_cols = [os for os in oversamples if os not in wide.columns]
        if missing_cols:
            friedman_rows.append({
                "representation": representation,
                "variable": var,
                "groups_compared": ",".join(str(x) for x in oversamples),
                "n_cases": 0,
                "friedman_chi_square": "",
                "p_value": "",
                "kendalls_w": "",
                "status": "missing_oversample_columns",
            })
            continue

        wide = wide[oversamples].replace([np.inf, -np.inf], np.nan).dropna()

        if len(wide) == 0:
            friedman_rows.append({
                "representation": representation,
                "variable": var,
                "groups_compared": ",".join(str(x) for x in oversamples),
                "n_cases": 0,
                "friedman_chi_square": "",
                "p_value": "",
                "kendalls_w": "",
                "status": "no_complete_cases",
            })
            continue

        arr = wide.values.astype(np.float64)
        if np.allclose(arr, arr[:, [0]], atol=1e-15, rtol=0.0):
            friedman_rows.append({
                "representation": representation,
                "variable": var,
                "groups_compared": ",".join(str(x) for x in oversamples),
                "n_cases": int(len(wide)),
                "friedman_chi_square": "",
                "p_value": "",
                "kendalls_w": "",
                "status": "constant_or_degenerate",
            })
            continue

        try:
            stat, p_value = friedmanchisquare(*[wide[os].values for os in oversamples])
            kendalls_w = kendalls_w_from_friedman(stat, len(wide), len(oversamples))
            friedman_rows.append({
                "representation": representation,
                "variable": var,
                "groups_compared": ",".join(str(x) for x in oversamples),
                "n_cases": int(len(wide)),
                "friedman_chi_square": float(stat),
                "p_value": float(p_value),
                "kendalls_w": float(kendalls_w),
                "status": "ok",
            })
        except Exception:
            friedman_rows.append({
                "representation": representation,
                "variable": var,
                "groups_compared": ",".join(str(x) for x in oversamples),
                "n_cases": int(len(wide)),
                "friedman_chi_square": "",
                "p_value": "",
                "kendalls_w": "",
                "status": "failed_or_degenerate",
            })

    save_rows_csv(friedman_rows, output_dir / "friedman_tests.csv")

    # ----------------------------
    # Linear trend analysis
    # ----------------------------
    trend_slope_rows, trend_summary_rows = run_trend_analysis(
        df=df,
        variables=variables,
        oversamples=oversamples,
        output_dir=output_dir,
    )

    return anova_rows, friedman_rows, trend_summary_rows


# ============================================================
# Plotting
# ============================================================

def plot_distance_matrix(distance_matrix, labels, names, title, output_path):
    order = np.argsort(labels)
    d_ord = distance_matrix[order][:, order]
    labels_ord = [labels[i] for i in order]

    plt.figure(figsize=(8, 7))
    im = plt.imshow(d_ord, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Samples")

    tick_positions = np.arange(len(labels_ord))
    tick_labels = labels_ord
    plt.xticks(tick_positions, tick_labels, rotation=90, fontsize=6)
    plt.yticks(tick_positions, tick_labels, fontsize=6)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_dendrogram(Xz, labels, title, output_path):
    Z = linkage(Xz, method="ward")

    plt.figure(figsize=(10, 5))
    dendrogram(Z, labels=labels, leaf_rotation=90, leaf_font_size=7)
    plt.title(title)
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_pca(Xz, labels, title, output_path):
    pca = PCA(n_components=2)
    Y = pca.fit_transform(Xz)

    plt.figure(figsize=(7, 6))
    for lbl in sorted(set(labels)):
        idx = [i for i, x in enumerate(labels) if x == lbl]
        plt.scatter(Y[idx, 0], Y[idx, 1], label=lbl, s=50, alpha=0.8)

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_oversampling_comparison(metrics_rows, representation, output_path):
    rows_rep = [r for r in metrics_rows if r["representation"] == representation]
    rows_rep = sorted(rows_rep, key=lambda r: r["oversample"])

    oversamples = [r["oversample"] for r in rows_rep]
    separation = [r["separation_ratio"] for r in rows_rep]
    silhouette = [r["silhouette_score"] for r in rows_rep]

    x = np.arange(len(oversamples), dtype=np.float64)

    plt.figure(figsize=(8, 5))
    plt.plot(x, separation, marker="o", label="Separation ratio")
    plt.plot(x, silhouette, marker="o", label="Silhouette score")

    plt.xticks(x, [str(o) for o in oversamples])
    plt.xlabel("Oversampling factor")
    plt.ylabel("Metric value")
    plt.title("Oversampling comparison ({})".format(representation))
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


def plot_representation_comparison(metrics_rows, metric_name, output_path):
    plt.figure(figsize=(8, 5))
    last_xticks = []

    for representation in REPRESENTATIONS:
        rows_rep = [r for r in metrics_rows if r["representation"] == representation]
        rows_rep = sorted(rows_rep, key=lambda r: r["oversample"])
        x = np.arange(len(rows_rep), dtype=np.float64)
        y = [r[metric_name] for r in rows_rep]
        last_xticks = [str(r["oversample"]) for r in rows_rep]
        plt.plot(x, y, marker="o", label=representation)

    if last_xticks:
        plt.xticks(np.arange(len(last_xticks)), last_xticks)

    plt.xlabel("Oversampling factor")
    plt.ylabel(metric_name)
    plt.title("{} across representations".format(metric_name))
    plt.legend()
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=200)
    plt.close()


# ============================================================
# Main analysis
# ============================================================

def analyze_dataset(dataset_dir=DATASET_DIR, manifest_path=MANIFEST_PATH, results_dir=RESULTS_DIR):
    results_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_manifest(manifest_path)
    feature_rows = build_feature_table(dataset_dir, manifest_rows)

    save_rows_csv(feature_rows, results_dir / "features_all.csv")

    metrics_rows = []
    inferential_rows = []

    oversamples = sorted(set(r["oversample"] for r in feature_rows))

    for oversample in oversamples:
        print("\nAnalyzing oversample = {}".format(oversample))

        rows_os = rows_for_oversample(feature_rows, oversample)

        for representation in REPRESENTATIONS:
            print("  Representation = {}".format(representation))
            rep_dir = results_dir / representation
            rep_dir.mkdir(parents=True, exist_ok=True)

            feature_names = get_representation_feature_names(representation, rows_os)
            X_raw, labels, names, feature_names = make_feature_matrix(rows_os, feature_names)

            # -------------------------------------------------
            # Balance GFCC summary vs base features, separately
            # for each representation.
            # -------------------------------------------------
            non_gfcc_idx = [
                i for i, n in enumerate(feature_names)
                if not n.startswith("gfcc_")
            ]
            gfcc_idx = [
                i for i, n in enumerate(feature_names)
                if n.startswith("gfcc_energy") or n.startswith("gfcc_variability") or
                   (INCLUDE_GFCC_MEANS_IN_ANALYSIS and n.startswith("gfcc_mean_"))
            ]

            if len(non_gfcc_idx) > 0 and len(gfcc_idx) > 0:
                X_raw[:, gfcc_idx] *= np.sqrt(float(len(non_gfcc_idx)) / float(len(gfcc_idx)))

            # -------------------------------------------------
            # Standardise
            # -------------------------------------------------
            Xz, mu, sigma = zscore_features(X_raw)

            distance_matrix = compute_distance_matrix(Xz)
            class_metrics = compute_class_distance_metrics(distance_matrix, labels)
            silhouette = compute_silhouette(Xz, labels)
            sample_sil = compute_silhouette_samples_safe(Xz, labels)

            for i, row in enumerate(rows_os):
                out = {
                    "filename": row["filename"],
                    "condition": row["condition"],
                    "payload": row["payload"],
                    "repetition": row["repetition"],
                    "oversample": row["oversample"],
                    "representation": representation,
                    "sample_silhouette": float(sample_sil[i]) if np.isfinite(sample_sil[i]) else np.nan,
                }

                for fname in feature_names:
                    out[fname] = row.get(fname, 0.0)

                inferential_rows.append(out)

            metrics_rows.append({
                "representation": representation,
                "oversample": oversample,
                "num_samples": len(rows_os),
                "num_features": len(feature_names),
                "mean_within_class_distance": class_metrics["mean_within_class_distance"],
                "mean_between_class_distance": class_metrics["mean_between_class_distance"],
                "separation_ratio": class_metrics["separation_ratio"],
                "silhouette_score": silhouette,
                "mean_sample_silhouette": float(np.nanmean(sample_sil)),
            })

            dist_csv = rep_dir / "distance_matrix_os{}.csv".format(oversample)
            with open(str(dist_csv), "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["filename"] + names)
                for i, row_name in enumerate(names):
                    writer.writerow([row_name] + list(distance_matrix[i]))

            plot_distance_matrix(
                distance_matrix=distance_matrix,
                labels=labels,
                names=names,
                title="Distance matrix ({}, oversample = {})".format(representation, oversample),
                output_path=rep_dir / "distance_matrix_os{}.png".format(oversample),
            )

            plot_dendrogram(
                Xz=Xz,
                labels=labels,
                title="Hierarchical clustering ({}, oversample = {})".format(representation, oversample),
                output_path=rep_dir / "dendrogram_os{}.png".format(oversample),
            )

            plot_pca(
                Xz=Xz,
                labels=labels,
                title="PCA feature space ({}, oversample = {})".format(representation, oversample),
                output_path=rep_dir / "pca_os{}.png".format(oversample),
            )

    metrics_rows = sorted(metrics_rows, key=lambda r: (r["representation"], r["oversample"]))

    save_rows_csv(metrics_rows, results_dir / "oversampling_metrics.csv")
    save_rows_csv(inferential_rows, results_dir / "inferential_samples.csv")

    all_anova_rows = []
    all_friedman_rows = []
    all_trend_rows = []

    for representation in REPRESENTATIONS:
        print("\n=== Running inferential statistics for {} ===".format(representation))
        rep_dir = results_dir / representation
        subset = [r for r in inferential_rows if r["representation"] == representation]

        anova_rows, friedman_rows, trend_rows = run_repeated_measures_stats(
            subset,
            rep_dir,
            representation
        )

        all_anova_rows.extend(anova_rows)
        all_friedman_rows.extend(friedman_rows)
        all_trend_rows.extend(trend_rows)

        plot_oversampling_comparison(
            metrics_rows=metrics_rows,
            representation=representation,
            output_path=rep_dir / "oversampling_comparison.png",
        )

    save_rows_csv(all_anova_rows, results_dir / "anova_repeated_measures_all.csv")
    save_rows_csv(all_friedman_rows, results_dir / "friedman_tests_all.csv")
    save_rows_csv(all_trend_rows, results_dir / "trend_linear_summary_all.csv")

    plot_representation_comparison(
        metrics_rows=metrics_rows,
        metric_name="separation_ratio",
        output_path=results_dir / "representation_comparison_separation_ratio.png",
    )

    plot_representation_comparison(
        metrics_rows=metrics_rows,
        metric_name="silhouette_score",
        output_path=results_dir / "representation_comparison_silhouette_score.png",
    )

    print("\n=== Oversampling comparison ===")
    for representation in REPRESENTATIONS:
        print("\n[Representation: {}]".format(representation))
        rows_rep = [r for r in metrics_rows if r["representation"] == representation]
        rows_rep = sorted(rows_rep, key=lambda r: r["oversample"])
        for row in rows_rep:
            print(
                "OS={0}: within={1:.4f}, between={2:.4f}, ratio={3:.4f}, silhouette={4:.4f}".format(
                    row["oversample"],
                    row["mean_within_class_distance"],
                    row["mean_between_class_distance"],
                    row["separation_ratio"],
                    row["silhouette_score"],
                )
            )

    print("\n=== Repeated-measures ANOVA ===")
    for representation in REPRESENTATIONS:
        print("\n[Representation: {}]".format(representation))
        rep_rows = [r for r in all_anova_rows if r["representation"] == representation]
        for row in rep_rows:
            if row["status"] == "ok":
                print(
                    "{0}: F={1:.6f}, p={2:.6g}, eta_p2={3:.6f}".format(
                        row["variable"],
                        row["f_statistic"],
                        row["p_value"],
                        row["partial_eta_squared"],
                    )
                )
            else:
                print("{0}: {1}".format(row["variable"], row["status"]))

    print("\n=== Friedman tests ===")
    for representation in REPRESENTATIONS:
        print("\n[Representation: {}]".format(representation))
        rep_rows = [r for r in all_friedman_rows if r["representation"] == representation]
        for row in rep_rows:
            if row["status"] == "ok":
                print(
                    "{0}: chi2={1:.6f}, p={2:.6g}, Kendall_W={3:.6f}".format(
                        row["variable"],
                        row["friedman_chi_square"],
                        row["p_value"],
                        row["kendalls_w"],
                    )
                )
            else:
                print("{0}: {1}".format(row["variable"], row["status"]))

    print("\n=== Linear trend analysis ===")
    for representation in REPRESENTATIONS:
        print("\n[Representation: {}]".format(representation))
        rep_dir = results_dir / representation
        trend_path = rep_dir / "trend_linear_summary.csv"
        if trend_path.exists():
            trend_df = pd.read_csv(str(trend_path))
            for _, row in trend_df.iterrows():
                if row["status"] == "ok":
                    print(
                        "{0}: mean_slope={1:.6f}, t={2:.6f}, p={3:.6g}, "
                        "95% CI=[{4:.6f}, {5:.6f}], mean_R2={6:.6f}".format(
                            row["variable"],
                            row["mean_slope"],
                            row["slope_t_statistic"],
                            row["slope_p_value"],
                            row["slope_ci_low"],
                            row["slope_ci_high"],
                            row["mean_r_squared"],
                        )
                    )
                else:
                    print("{0}: {1}".format(row["variable"], row["status"]))

    print("\nResults written to: {}".format(results_dir))


if __name__ == "__main__":
    analyze_dataset()