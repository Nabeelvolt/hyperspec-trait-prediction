import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# -------------------------------
# CONFIG
# -------------------------------
DATASET_VERSION = "v1"
RAW_ROOT = None  # e.g. "/data/raw" if you don't have dataset.constants
SAVE_DIR = "/app/practice"
os.makedirs(SAVE_DIR, exist_ok=True)

# Fixed input grid 400–2400 nm (inclusive)
WVL = np.arange(400, 2401)

# -------------------------------
# Helpers
# -------------------------------
def resolve_raw_dir():
    if RAW_ROOT is not None:
        return os.path.join(RAW_ROOT, DATASET_VERSION)
    try:
        import constants
        return os.path.join(constants.RAW_DIR, DATASET_VERSION)
    except Exception:
        raise RuntimeError("Set RAW_ROOT or provide dataset.constants.RAW_DIR")

def list_wave_csvs(raw_dir):
    candidates = [
        os.path.join(raw_dir, f)
        for f in os.listdir(raw_dir)
        if f.lower().endswith(".csv") and os.path.isfile(os.path.join(raw_dir, f))
    ]
    valid = []
    for p in candidates:
        try:
            # read only the header
            cols = pd.read_csv(p, nrows=0).columns
            if any(c.startswith("Wave_") for c in cols):
                valid.append(p)
        except Exception:
            pass
    return valid

def parse_wave_columns(df):
    wave_cols = [c for c in df.columns if c.startswith("Wave_")]
    if not wave_cols:
        raise ValueError("No Wave_* columns")
    # Extract numeric wavelengths from names like Wave_400, Wave_400.5, Wave_0400
    def to_num(c):
        m = re.search(r"Wave_(\d+(?:\.\d+)?)", c)
        return float(m.group(1)) if m else None
    pairs = [(to_num(c), c) for c in wave_cols]
    pairs = [(w, c) for w, c in pairs if w is not None]
    pairs.sort(key=lambda x: x[0])
    w_src = np.array([w for w, _ in pairs], dtype=float)
    cols_sorted = [c for _, c in pairs]
    return w_src, cols_sorted

def resample_rows_to_grid(w_src, R_rows, w_target):
    # R_rows: shape (n_samples, n_waves)
    out = np.empty((R_rows.shape[0], w_target.size), dtype=float)
    for i in range(R_rows.shape[0]):
        out[i, :] = np.interp(w_target, w_src, R_rows[i, :], left=np.nan, right=np.nan)
    return np.nan_to_num(out, nan=0.0)

# Truncated normal samplers
def make_truncnorm(mu, sigma, lo, hi):
    a = (lo - mu) / sigma
    b = (hi - mu) / sigma
    return truncnorm(a=a, b=b, loc=mu, scale=sigma)

LOW_DIST  = make_truncnorm(400.0, 100.0, 400.0, 700.0)    # L
HIGH_DIST = make_truncnorm(2400.0, 500.0, 1000.0, 2400.0) # H

def sample_bounds(min_width=350.0, max_tries=10000):
    for _ in range(max_tries):
        L = float(LOW_DIST.rvs())
        H = float(HIGH_DIST.rvs())
        if H - L >= min_width:
            return L, H
    raise RuntimeError("Could not sample L,H satisfying min width")

def spectral_trim_rows_keep_length(R_rows, wvl=WVL, min_width=350.0):
    """
    For each row (sample), sample L,H and zero values outside [L,H].
    Returns trimmed array and arrays of L,H.
    """
    n = R_rows.shape[0]
    trimmed = R_rows.copy()
    Ls = np.empty(n); Hs = np.empty(n); zero_counts = np.empty(n, dtype=int)
    for i in range(n):
        L, H = sample_bounds(min_width=min_width)
        mask = (wvl < L) | (wvl > H)
        trimmed[i, mask] = 0.0
        Ls[i], Hs[i], zero_counts[i] = L, H, int(mask.sum())
    return trimmed, Ls, Hs, zero_counts

# Plots (saved for headless use)
"""
def plot_cutoff_pdfs(path):
    lower_dom = np.linspace(400.0, 700.0, 1000)
    upper_dom = np.linspace(1000.0, 2400.0, 1000)
    plt.figure(figsize=(7, 4))
    plt.plot(lower_dom, LOW_DIST.pdf(lower_dom), label="Lower cutoff PDF (L)")
    plt.fill_between(lower_dom, LOW_DIST.pdf(lower_dom), alpha=0.3)
    plt.plot(upper_dom, HIGH_DIST.pdf(upper_dom), label="Upper cutoff PDF (H)")
    plt.fill_between(upper_dom, HIGH_DIST.pdf(upper_dom), alpha=0.3)
    plt.legend(loc="upper right"); plt.xlabel("Wavelength (nm)"); plt.ylabel("Probability density")
    plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
"""
def plot_sample_overlay(wvl, orig, trimmed, L, H, path, title):
    plt.figure(figsize=(8, 4))
    plt.plot(wvl, orig, label="Original (400–2400)")
    plt.plot(wvl, trimmed, label=f"Trimmed (zeros outside {int(L)}–{int(H)} nm)")
    plt.axvline(L, linestyle="--"); plt.axvline(H, linestyle="--")
    plt.xlabel("Wavelength (nm)"); plt.ylabel("Reflectance"); plt.title(title)
    plt.legend(loc="best"); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

# -------------------------------
# MAIN
# -------------------------------
def main():
    raw_dir = resolve_raw_dir()
    wave_csvs = list_wave_csvs(raw_dir)
    if not wave_csvs:
        raise RuntimeError(f"No CSVs with Wave_* columns found in {raw_dir}")
    print(f"[Step 0] Found {len(wave_csvs)} Wave_* CSV(s) in: {raw_dir}")

    # Process each file; plot the first sample only (per file)
    for csv_path in wave_csvs:
        print(f"[Step 1] Loading: {os.path.basename(csv_path)}")
        df = pd.read_csv(csv_path)
        w_src, cols_sorted = parse_wave_columns(df)

        # Extract reflectance matrix (rows = samples, cols = sorted wavelengths)
        R_rows = df[cols_sorted].astype(float).to_numpy()  # shape (n_samples, n_src_waves)

        # Step 2: resample each row to fixed 400–2400 nm grid
        R_fixed = resample_rows_to_grid(w_src, R_rows, WVL)

        # Steps 3–5: per-row random L,H and zero outside [L,H]
        R_trim, Ls, Hs, zero_counts = spectral_trim_rows_keep_length(R_fixed, WVL, min_width=350.0)
        print(f"[Step 3-5] Trimmed {R_trim.shape[0]} sample(s). Example width: {Hs[0]-Ls[0]:.1f} nm; zeroed points: {zero_counts[0]}")

        # Save plots for the first sample
        base = os.path.splitext(os.path.basename(csv_path))[0]
        plot_cutoff_pdfs(os.path.join(SAVE_DIR, f"{base}_step1_cutoff_pdfs.png"))
        plot_sample_overlay(
            WVL,
            R_fixed[0, :],
            R_trim[0, :],
            Ls[0], Hs[0],
            os.path.join(SAVE_DIR, f"{base}_step3_5_trimmed_overlay.png"),
            title=f"{base} - sample 0"
        )

        # Save trimmed CSV with Wave_400..Wave_2400 columns (plus original non-Wave_* meta, if any)
        meta_cols = [c for c in df.columns if not c.startswith("Wave_")]
        out_df = df[meta_cols].copy()
        for j, wl in enumerate(WVL):
            out_df[f"Wave_{int(wl)}"] = R_trim[:, j]
        out_csv = os.path.join(SAVE_DIR, f"{base}_trimmed.csv")
        out_df.to_csv(out_csv, index=False)
        print(f"[Done] Wrote trimmed CSV: {out_csv}")

if __name__ == "__main__":
    # avoid Unicode dash issues on ASCII stdout
    try:
        import sys
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    main()

