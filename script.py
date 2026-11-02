# -*- coding: utf-8 -*-
"""
Plant cane (2025) and ratoon cane (2026): multivariate Fuzzy performance score
(PCA + zero-order Takagi-Sugeno FIS + PSO)  --  VERSION 3 (revised, English)

Usage:
    python sugarcane_fuzzy_v3.py [path/to/dataset.xlsx]
Optional environment variables:
    FUZZY_DATA    dataset path                    (default: dataset.xlsx)
    FUZZY_OUT     output folder                   (default: fuzzy_results)
    FUZZY_QUICK   1 = fast smoke test (few PSO runs / permutations)
    FUZZY_SHOW    0 = do not open plot windows (figures are only saved)

Dependencies: numpy, pandas, scipy, matplotlib, openpyxl (no scikit-learn).

Expected dataset (one sheet per season, named "2025" and "2026"). Column names:
    Acronym, Treatment, Cultivar,
    Fi, Pu, PC, TRS, TCH, TPH, E, A, Ci, gsw, FS, RWC, PFM, PDM, SFM, SDM,
    LFM, LDM, SL, SD, NT, Clfa, Clfb, CAR
  Acronym      = plot ID ending in the block number (e.g., T3B2 -> block 2)
  Treatment = "Controle - nSi", "Estresse - nSi", "Controle + nSi", "Estresse + nSi"
               (English labels "Control/Stress +/- nSi" are also accepted)

Variables: fiber content (Fi), juice purity (Pu), pol in cane (PC), total
recoverable sugars (TRS), tons of cane per hectare (TCH), tons of pol per hectare
(TPH), transpiration rate (E), net photosynthetic rate (A), intercellular CO2
concentration (Ci), stomatal conductance (gsw), foliar succulence (FS), relative
water content (RWC), plant fresh mass (PFM), plant dry mass (PDM), stalk fresh
mass (SFM), stalk dry mass (SDM), leaf fresh mass (LFM), leaf dry mass (LDM),
stalk length (SL), stalk diameter (SD), number of tillers (NT), chlorophyll a
(Clfa), chlorophyll b (Clfb) and carotenoids (CAR).
NOTE: "PC" = pol in cane; principal components are called PCA1, PCA2, PCA3.

---------------------------------------------------------------------------
WHAT CHANGED RELATIVE TO THE ORIGINAL SCRIPT
---------------------------------------------------------------------------
 [1] Circularity (J_Fisher uses group labels, then the same scores are tested):
     the whole pipeline (z-score, PCA, polarity, normalisation, PSO) is refitted
     in leave-one-block-out cross-validation (LOBO-CV; the same block is held out
     in both seasons). All inference uses ONLY out-of-fold (OOF) scores.
     A label-permutation test reruns the PSO with shuffled labels to measure how
     much separation the optimiser can "manufacture" from noise.
 [2] FIS redundancy: compared against reference indices (PCA1, mean z-score of
     vigour traits, eigenvalue-weighted PCA index) by adjusted R2 of the 16
     groups, in-sample and out-of-sample.
 [3] K=3 explains little variance: Kaiser, Horn's parallel analysis and
     cumulative variance are reported; sensitivity to K (K=2 -> 9 rules);
     polarity s_k reported with |r| and a warning when weak; Z_vigor defined.
 [4] Factorial design: factorial PERMANOVA (Cultivar x Water x nSi) with blocks
     as strata (Freedman-Lane residual permutation restricted within blocks)
     + PERMDISP. Also applied to the univariate S_Fuzzy.
 [5] IEF without ratio scale: original IEF kept (flagged), plus IEF on TCH/TPH
     (ratio scale) and difference-in-differences in score points. Delta_nSi is
     paired by block with 95% CI (t, df=3) and an exact sign-flip test.
 [6] Independence/pooling: inference per season; repeated plots across seasons
     -> plot-wise Spearman correlation; covariance-structure comparison (Tucker
     congruence) and loading stability by block bootstrap; raw variables compared
     between seasons (paired Wilcoxon by plot).
 [7] Tests: Holm adjustment for Conover-Iman, Mann-Whitney and PERMANOVA terms;
     prints the smallest attainable exact p with n=4 vs 4.
 [8] Formulation: centre ordering enforced by parametrisation (hard constraint,
     J_Order removed); sigma bounded consistently; overlap/coverage evaluated on
     a grid over [0,1]; J_Fisher replaced by pooled SS_within/SS_total; ridge
     regularisation of consequents; rule-support report; percentile
     normalisation fitted on training data only (transferable score); all PSO
     hyper-parameters explicit; several independent runs + stability; velocity
     clamping.
 [9] Data: automatic QA (identities TPH=TCH*PC/100, PFM=SFM+LFM, PDM=SDM+LDM;
     duplicated mass rows; zero within-group variance; scale between seasons;
     robust outliers). Deterministically derived variables (TPH, PFM, PDM) are
     excluded from the PCA by default; sensitivity with all 24 variables and
     without mass variables. RWC fraction -> % standardised.
 [10] v3: English outputs; no column renaming (dataset columns already use the
     final variable names); principal components named PCA1-PCA3.
---------------------------------------------------------------------------
"""
import os
import sys
import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("FUZZY_SHOW", "1") != "1":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["font.size"] = 12

# =============================================================================
# 0. CONFIGURATION
# =============================================================================
QUICK = os.environ.get("FUZZY_QUICK", "0") == "1"
SHOW = os.environ.get("FUZZY_SHOW", "1") == "0"

CFG = dict(
    data_path=sys.argv[1] if len(sys.argv) > 1 else os.environ.get("FUZZY_DATA", "dataset.xlsx"),
    out_dir=os.environ.get("FUZZY_OUT", "fuzzy_results"),
    seed=42,
    K=3,                      # number of PCs / FIS inputs (M=3 terms -> 3**K rules)
    delta=0.25,               # minimum separation between adjacent centres (hard constraint)
    gamma=1.15,               # target overlap factor of the memberships
    cov_min=0.50,             # minimum of the maximum membership at any point of the universe
    sig_lo=0.12, sig_hi=0.45,  # sigma bounds (sigma_min, sigma_max)
    beta_ov=0.15,             # weight of the overlap / coverage terms
    beta_disc=0.10,           # weight of the discrimination term SS_within / SS_total
    beta_reg=0.05,            # weight of the ridge term on consequents
    norm_pct=(2.5, 97.5),     # robust percentile normalisation (fitted on training data only)
    pso=dict(n_particles=60, n_iter=150, w_max=0.9, w_min=0.4, c1=1.6, c2=1.6, vmax_frac=0.25),
    n_runs=6,                 # independent PSO runs
    n_runs_sens=2,            # PSO runs in sensitivity analyses
    n_runs_null=3,            # PSO runs per permutation in the label-permutation test
    n_null=20,                # label permutations (PSO is refitted each time)
    n_perm=9999,              # PERMANOVA / PERMDISP permutations
    n_pa=300,                 # parallel-analysis iterations
    n_boot=200,               # block-bootstrap replicates (loading stability)
    alpha=0.05,
    exclude_derived=["TPH", "PFM", "PDM"],   # deterministic derived variables (see QA)
    qa_tol=0.02,
)
if QUICK:
    CFG["pso"].update(n_particles=20, n_iter=40)
    CFG.update(n_runs=2, n_runs_sens=1, n_runs_null=1, n_null=4, n_perm=499, n_pa=30, n_boot=30)

SEASONS = [2025, 2026]
VARS = ["Fi", "Pu", "PC", "TRS", "TCH", "TPH", "E", "A", "Ci", "gsw", "FS", "RWC",
        "PFM", "PDM", "SFM", "SDM", "LFM", "LDM", "SL", "SD", "NT", "Clfa", "Clfb", "CAR"]
VIGOR = ["TCH", "TPH", "TRS", "A", "gsw", "PFM", "PDM"]   # Z_vigor = mean z-score of these variables
MASS_VARS = ["PFM", "PDM", "SFM", "SDM", "LFM", "LDM"]
# Treatment labels found in the dataset -> English labels used in all outputs
TRT_MAP = {
    "Controle - nSi": "Control - nSi", "Estresse - nSi": "Stress - nSi",
    "Controle + nSi": "Control + nSi", "Estresse + nSi": "Stress + nSi",
    "Control - nSi": "Control - nSi", "Stress - nSi": "Stress - nSi",
    "Control + nSi": "Control + nSi", "Stress + nSi": "Stress + nSi",
}
TRT_ORDER = ["Control - nSi", "Stress - nSi", "Control + nSi", "Stress + nSi"]
CULTIVARS = ["RB027040", "RB867515"]
EPS = 1e-8

REPORT = []          # lines of the final text report
TABLES = {}          # tables exported to Excel


def say(msg=""):
    print(msg)
    REPORT.append(str(msg))


def finish_fig(fig, name, pdf=False):
    path = os.path.join(CFG["out_dir"], name)
    fig.savefig(path + ".png", dpi=300, bbox_inches="tight")
    if pdf:
        fig.savefig(path + ".pdf", bbox_inches="tight")
    if SHOW:
        plt.show()
    plt.close(fig)


def fmt_p(v):
    return "<0.001" if v < 0.001 else f"={v:.3f}"


# =============================================================================
# 1. DATA LOADING AND QA
# =============================================================================
def load_data(path):
    frames = []
    for sh in ["2025", "2026"]:
        d = pd.read_excel(path, sheet_name=sh)
        d["Season"] = int(sh)
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    missing = [v for v in VARS + ["Acronym", "Treatment", "Cultivar"] if v not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the dataset: {missing}")
    unknown = sorted(set(df["Treatment"]) - set(TRT_MAP))
    if unknown:
        raise ValueError(f"Unrecognised treatment labels: {unknown}")
    df["Treatment"] = df["Treatment"].map(TRT_MAP)
    df["Block"] = df["Acronym"].astype(str).str.extract(r"B(\d+)$")[0].astype(int)
    df["Water"] = np.where(df["Treatment"].str.contains("Stress"), "Stress", "Control")
    df["nSi"] = np.where(df["Treatment"].str.contains(r"\+\s*nSi"), "+nSi", "-nSi")
    df["Group"] = df.groupby(["Season", "Cultivar", "Treatment"]).ngroup()
    # RWC: standardise fraction -> % (the z-score is invariant, but the unit becomes explicit)
    for s in SEASONS:
        m = df["Season"] == s
        if df.loc[m, "RWC"].mean() < 1.5:
            df.loc[m, "RWC"] *= 100.0
            say(f"[QA] RWC in {s} was stored as a fraction; converted to % (x100).")
    return df


def data_qa(df):
    tol = CFG["qa_tol"]
    ident, dup, zerov, scale, outl = [], [], [], [], []
    for s_ in SEASONS:
        s = df[df.Season == s_]
        checks = {
            "PFM = SFM + LFM": s["PFM"] - s["SFM"] - s["LFM"],
            "PDM = SDM + LDM": s["PDM"] - s["SDM"] - s["LDM"],
            "TPH = TCH*PC/100": s["TPH"] - s["TCH"] * s["PC"] / 100.0,
        }
        for name, diff in checks.items():
            bad = s.loc[diff.abs() > tol, "Acronym"].tolist()
            ident.append(dict(Season=s_, Identity=name, n_violations=len(bad),
                              max_abs_deviation=round(float(diff.abs().max()), 4), plots=",".join(bad)))
        d = s[s.duplicated(subset=MASS_VARS, keep=False)]
        dup.append(dict(Season=s_, n_rows_identical_masses=len(d), plots=",".join(d["Acronym"].tolist())))
        for (cult, trt), g in s.groupby(["Cultivar", "Treatment"]):
            for v in VARS:
                if g[v].nunique() == 1 and len(g) > 1:
                    zerov.append(dict(Season=s_, Cultivar=cult, Treatment=trt, Variable=v, value=float(g[v].iloc[0])))
        for v in VARS:
            x = s[v].values.astype(float)
            mad = np.median(np.abs(x - np.median(x))) * 1.4826
            if mad > 0:
                n_out = int((np.abs(x - np.median(x)) / mad > 4).sum())
                if n_out:
                    outl.append(dict(Season=s_, Variable=v, n_robust_outliers=n_out))
    m25 = df[df.Season == 2025][VARS].mean()
    m26 = df[df.Season == 2026][VARS].mean()
    for v in VARS:
        r = abs(m26[v]) / (abs(m25[v]) + 1e-12)
        scale.append(dict(Variable=v, mean_2025=round(m25[v], 4), mean_2026=round(m26[v], 4),
                          ratio_2026_2025=round(r, 3), scale_warning=bool(r > 10 or r < 0.1)))
    qa = dict(identities=pd.DataFrame(ident), duplicates=pd.DataFrame(dup),
              zero_variance=pd.DataFrame(zerov), scale_between_seasons=pd.DataFrame(scale),
              outliers=pd.DataFrame(outl))
    say("\n=== DATA QA ===")
    say(qa["identities"].drop(columns="plots").to_string(index=False))
    for _, r in qa["identities"].iterrows():
        if r.n_violations:
            say(f"  [WARNING] {r.Season} {r.Identity}: violations in {r.plots}")
    for _, r in qa["duplicates"].iterrows():
        if r.n_rows_identical_masses:
            say(f"  [WARNING] {r.Season}: plots with identical mass blocks: {r.plots}")
    if len(qa["zero_variance"]):
        say(f"  [WARNING] {len(qa['zero_variance'])} group x variable combinations have ZERO variance "
            f"(possible rounding/imputation) -> see sheet QA_zero_variance.")
    al = qa["scale_between_seasons"]
    al = al[al.scale_warning]
    if len(al):
        say("  [WARNING] ratio of means between seasons outside [0.1; 10]: " + ", ".join(al.Variable.tolist())
            + " (within-season z-scoring neutralises this, but confirm unit/method.)")
    TABLES.update({f"QA_{k}": v for k, v in qa.items()})
    return qa


# =============================================================================
# 2. ZERO-ORDER TAKAGI-SUGENO FIS (pure functions)
# =============================================================================
def rule_index(K, M=3):
    return np.array(list(product(range(M), repeat=K)))


def decode_centers(u, delta):
    """u (K,3) in [0,1] -> ordered centres with minimum separation delta, inside [0,1]
    (hard constraint: replaces the J_Order penalty)."""
    c1 = u[:, 0] * (1 - 2 * delta)
    c2 = c1 + delta + u[:, 1] * (1 - c1 - 2 * delta)
    c3 = c2 + delta + u[:, 2] * (1 - c2 - delta)
    return np.stack([c1, c2, c3], axis=1)


def decode_params(p, K, delta):
    R = 3 ** K
    u = p[:3 * K].reshape(K, 3)
    sig = p[3 * K:6 * K].reshape(K, 3)
    theta = p[6 * K:6 * K + R]
    return decode_centers(u, delta), sig, theta


def memberships(U, c, s):
    return np.exp(-0.5 * ((U[:, :, None] - c[None, :, :]) / s[None, :, :]) ** 2)


def fis_output(U, c, s, theta, RIDX):
    mu = memberships(U, c, s)
    W = np.prod(mu[:, np.arange(RIDX.shape[1]), RIDX], axis=2)   # (N, R) firing strengths
    S = (W @ theta) / (W.sum(1) + EPS)
    return S, W


def r2_groups(x, groups):
    """R2 and adjusted R2 of a one-way ANOVA (groups) on index x."""
    x = np.asarray(x, float)
    gid = np.unique(groups, return_inverse=True)[1]
    G, n = gid.max() + 1, len(x)
    gm = np.bincount(gid, weights=x) / np.bincount(gid)
    ssw = np.sum((x - gm[gid]) ** 2)
    sst = np.sum((x - x.mean()) ** 2) + 1e-12
    return 1 - ssw / sst, 1 - (ssw / (n - G)) / (sst / (n - 1))


# =============================================================================
# 3. PSO
# =============================================================================
def pso_minimize(obj, K, theta0, rng, pso):
    R = 3 ** K
    dim = 6 * K + R
    lb = np.r_[np.zeros(3 * K), np.full(3 * K, CFG["sig_lo"]), np.zeros(R)]
    ub = np.r_[np.ones(3 * K), np.full(3 * K, CFG["sig_hi"]), np.full(R, 100.0)]
    span = ub - lb
    P = pso["n_particles"]
    u0 = np.tile([0.30, 0.2857, 0.40], K)          # nominal centres ~ (0.15, 0.50, 0.85)
    pos = np.empty((P, dim))
    pos[:, :3 * K] = np.clip(u0 + rng.uniform(-0.2, 0.2, (P, 3 * K)), 0, 1)
    pos[:, 3 * K:6 * K] = rng.uniform(0.15, 0.40, (P, 3 * K))
    pos[:, 6 * K:] = np.clip(theta0 + rng.uniform(-15, 15, (P, R)), 0, 100)
    pos[0] = np.r_[u0, np.full(3 * K, 0.25), theta0]
    vmax = pso["vmax_frac"] * span
    vel = rng.uniform(-1, 1, (P, dim)) * 0.1 * span
    pbest = pos.copy()
    pval = np.array([obj(x) for x in pos])
    g = int(np.argmin(pval))
    gpos, gval = pbest[g].copy(), pval[g]
    hist_g, hist_m = [], []
    for it in range(pso["n_iter"]):
        w = pso["w_max"] - (pso["w_max"] - pso["w_min"]) * it / max(pso["n_iter"] - 1, 1)  # linear inertia decay
        r1, r2 = rng.random((P, dim)), rng.random((P, dim))
        vel = w * vel + pso["c1"] * r1 * (pbest - pos) + pso["c2"] * r2 * (gpos - pos)
        vel = np.clip(vel, -vmax, vmax)
        pos = pos + vel
        out = (pos < lb) | (pos > ub)
        pos = np.clip(pos, lb, ub)
        vel[out] = 0.0
        vals = np.array([obj(x) for x in pos])
        imp = vals < pval
        pbest[imp], pval[imp] = pos[imp], vals[imp]
        g = int(np.argmin(pval))
        if pval[g] < gval:
            gval, gpos = pval[g], pbest[g].copy()
        hist_g.append(gval)
        hist_m.append(float(pval.mean()))
    return gpos, gval, np.array(hist_g), np.array(hist_m)


# =============================================================================
# 4. FULL PIPELINE (z-score -> PCA -> polarity -> normalisation -> FIS/PSO)
#    Everything is fitted on the training set only and is applicable to new data.
# =============================================================================
class FuzzyPipeline:
    def __init__(self, pca_vars, K=None):
        self.pca_vars = list(pca_vars)
        self.K = K or CFG["K"]
        self.RIDX = rule_index(self.K)

    def _fit_season_stats(self, df):
        self.mu, self.sd = {}, {}
        for s in SEASONS:
            x = df.loc[df.Season == s, VARS].values.astype(float)
            self.mu[s], self.sd[s] = x.mean(0), x.std(0, ddof=1)

    def zscore(self, df):
        Z = np.empty((len(df), len(VARS)))
        for s in SEASONS:
            m = df["Season"].values == s
            if m.any():
                Z[m] = (df.loc[m, VARS].values.astype(float) - self.mu[s]) / self.sd[s]
        return Z

    def project(self, df):
        Z = self.zscore(df)
        idx = [VARS.index(v) for v in self.pca_vars]
        sc = (Z[:, idx] - self.pc_mean) @ self.V
        return np.clip((sc - self.lo) / (self.hi - self.lo), 0, 1)

    def score(self, df):
        U = self.project(df)
        c, s, theta = decode_params(self.params, self.K, CFG["delta"])
        return fis_output(U, c, s, theta, self.RIDX)[0]

    def baselines(self, df):
        U = self.project(df)
        Z = self.zscore(df)
        vig = Z[:, [VARS.index(v) for v in VIGOR]].mean(1)
        return {"PCA1": U[:, 0], "Vigour (mean z)": vig, "Weighted PCA index": U @ self.alpha}

    def fit(self, df, seed=0, n_runs=None, groups=None):
        K, cfg = self.K, CFG
        self._fit_season_stats(df)
        Z = self.zscore(df)
        idx = [VARS.index(v) for v in self.pca_vars]
        Zp = Z[:, idx]
        self.pc_mean = Zp.mean(0)
        Zc = Zp - self.pc_mean
        n = len(Zc)
        _, sv, Vt = np.linalg.svd(Zc, full_matrices=False)
        self.eig = sv ** 2 / (n - 1)
        self.evr = self.eig / self.eig.sum()
        V = Vt.T[:, :K].copy()
        sc = Zc @ V
        vig = Z[:, [VARS.index(v) for v in VIGOR]].mean(1)       # Z_vigor
        self.pol_corr = np.array([np.corrcoef(sc[:, k], vig)[0, 1] for k in range(K)])
        self.sign = np.where(self.pol_corr >= 0, 1.0, -1.0)       # semantic polarity s_k
        self.V, sc = V * self.sign, sc * self.sign
        self.scores = sc
        self.lo, self.hi = np.percentile(sc, cfg["norm_pct"], axis=0)
        U = np.clip((sc - self.lo) / (self.hi - self.lo), 0, 1)
        self.U = U
        self.alpha = self.eig[:K] / self.eig[:K].sum()
        icp = U @ self.alpha
        self.y_ref = 10 + 80 * (icp - icp.min()) / (icp.max() - icp.min())
        base = np.array([15.0, 50.0, 85.0])
        self.theta0 = np.array([sum(self.alpha[k] * base[m[k]] for k in range(K)) for m in self.RIDX])
        grp = df["Group"].values if groups is None else np.asarray(groups)
        parts = self._make_parts(U, self.y_ref, grp)
        w = (1.0, cfg["beta_ov"], cfg["beta_ov"], cfg["beta_disc"], cfg["beta_reg"])
        obj = lambda p: float(np.dot(w, parts(p)))
        self.parts, self.obj = parts, obj

        self.runs = []
        for r in range(n_runs or cfg["n_runs"]):
            rng = np.random.default_rng(cfg["seed"] + 1000 * seed + r)
            gp, gv, hg, hm = pso_minimize(obj, K, self.theta0, rng, cfg["pso"])
            c, s, th = decode_params(gp, K, cfg["delta"])
            S = fis_output(U, c, s, th, self.RIDX)[0]
            pr = parts(gp)
            self.runs.append(dict(params=gp, J=gv, hist_g=hg, hist_m=hm, S=S,
                                  j_pca=pr[0], j_disc=pr[3]))
        best = min(self.runs, key=lambda d: d["J"])
        self.params, self.J, self.S_in = best["params"], best["J"], best["S"]
        self.centers, self.sigmas, self.theta = decode_params(self.params, K, cfg["delta"])
        _, W = fis_output(U, self.centers, self.sigmas, self.theta, self.RIDX)
        self.rule_support = (W / (W.sum(1, keepdims=True) + EPS)).sum(0)   # effective n per rule
        return self

    def _make_parts(self, U, yref, groups):
        K, delta, RIDX, theta0 = self.K, CFG["delta"], self.RIDX, self.theta0
        gid = np.unique(groups, return_inverse=True)[1]
        counts = np.bincount(gid)
        grid = np.linspace(0, 1, 101)[:, None] * np.ones((1, K))
        gamma, cov_min = CFG["gamma"], CFG["cov_min"]

        def parts(p):
            c, s, th = decode_params(p, K, delta)
            S, _ = fis_output(U, c, s, th, RIDX)
            j_pca = np.mean(((S - yref) / 80.0) ** 2)                    # fidelity to the PCA reference
            mg = memberships(grid, c, s)
            j_ov = np.mean((mg.sum(2) - gamma) ** 2)                      # overlap
            j_cov = np.mean(np.maximum(0.0, cov_min - mg.max(2)) ** 2)    # coverage gaps
            gm = np.bincount(gid, weights=S) / counts
            j_disc = np.sum((S - gm[gid]) ** 2) / (np.sum((S - S.mean()) ** 2) + 1e-9)   # SS_within / SS_total
            j_reg = np.mean(((th - theta0) / 100.0) ** 2)                 # ridge on consequents
            return np.array([j_pca, j_ov, j_cov, j_disc, j_reg])
        return parts


def best_of(runs, m):
    return min(runs[:m], key=lambda d: d["J"])


def run_stability(runs):
    S = np.array([r["S"] for r in runs])
    if len(S) < 2:
        return np.nan, np.nan
    rho = np.corrcoef(stats.rankdata(S, axis=1))
    off = rho[np.triu_indices(len(S), 1)]
    return float(off.mean()), float(off.min())


# =============================================================================
# 5. LEAVE-ONE-BLOCK-OUT CROSS-VALIDATION -- out-of-fold scores
# =============================================================================
def lobo_cv(df, pca_vars, K, n_runs):
    n = len(df)
    S_oof = np.full(n, np.nan)
    base_oof = {k: np.full(n, np.nan) for k in ["PCA1", "Vigour (mean z)", "Weighted PCA index"]}
    for b in sorted(df.Block.unique()):
        tr, te = (df.Block != b).values, (df.Block == b).values
        pipe = FuzzyPipeline(pca_vars, K).fit(df[tr].reset_index(drop=True), seed=int(b), n_runs=n_runs)
        dte = df[te].reset_index(drop=True)
        S_oof[te] = pipe.score(dte)
        for k, v in pipe.baselines(dte).items():
            base_oof[k][te] = v
    return S_oof, base_oof


# =============================================================================
# 6. STATISTICS: factorial PERMANOVA (blocks as strata), PERMDISP, CLD
# =============================================================================
def holm(p):
    p = np.asarray(p, float)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def factorial_permanova(Y, meta, n_perm, seed):
    """Factorial PERMANOVA (Euclidean distance) Cultivar x Water x nSi in an RCBD.
    Sums of squares by projection (balanced design -> orthogonal contrasts).
    Term p-values: residual permutation (Freedman-Lane) restricted within blocks.
    'Treatment (global)': permutation of rows within blocks."""
    Y = np.asarray(Y, float)
    if Y.ndim == 1:
        Y = Y[:, None]
    n = len(Y)
    eff = lambda v: np.where(np.asarray(v) == sorted(set(v))[0], -1.0, 1.0)
    c, w, s = eff(meta.Cultivar), eff(meta.Water), eff(meta.nSi)
    terms = {"Cultivar": c, "Water": w, "nSi": s, "Cultivar x Water": c * w,
             "Cultivar x nSi": c * s, "Water x nSi": w * s, "Cultivar x Water x nSi": c * w * s}
    blk = np.asarray(meta.Block)
    levels = sorted(set(blk))
    Bcols = [(blk == b).astype(float) - (blk == levels[-1]).astype(float) for b in levels[:-1]]
    X = np.column_stack([np.ones(n)] + Bcols + list(terms.values()))
    Qf = np.linalg.qr(X)[0]
    T = len(terms)
    df_res = n - X.shape[1]
    Qt = np.column_stack([v / np.linalg.norm(v) for v in terms.values()])      # (n, T)
    ss_t = ((Qt.T @ Y) ** 2).sum(1)
    ss_res = (Y ** 2).sum() - ((Qf.T @ Y) ** 2).sum()
    sst_c = ((Y - Y.mean(0)) ** 2).sum()
    F_obs = ss_t / (ss_res / df_res)
    F_glob = (ss_t.sum() / T) / (ss_res / df_res)

    fitted = np.stack([Qf @ (Qf.T @ Y) - np.outer(Qt[:, t], Qt[:, t] @ Y) for t in range(T)])   # (T, n, p)
    resid = Y[None] - fitted
    groups_idx = [np.where(blk == b)[0] for b in levels]
    rng = np.random.default_rng(seed)
    cnt = np.zeros(T)
    cnt_g = 0
    for _ in range(n_perm):
        idx = np.arange(n)
        for gi in groups_idx:
            idx[gi] = rng.permutation(gi)
        Ys = fitted + resid[:, idx, :]                                         # Freedman-Lane
        num = (np.einsum("nt,tnp->tp", Qt, Ys) ** 2).sum(1)
        den = (Ys ** 2).sum((1, 2)) - (np.einsum("nr,tnp->trp", Qf, Ys) ** 2).sum((1, 2))
        cnt += (num / (den / df_res) >= F_obs - 1e-12)
        Yg = Y[idx]                                                            # global H0
        ssg = ((Qt.T @ Yg) ** 2).sum()
        rg = (Yg ** 2).sum() - ((Qf.T @ Yg) ** 2).sum()
        cnt_g += ((ssg / T) / (rg / df_res) >= F_glob - 1e-12)
    p = (1 + cnt) / (n_perm + 1)
    out = pd.DataFrame({"Term": list(terms), "df": 1, "SS": ss_t, "R2_%": 100 * ss_t / sst_c,
                        "Pseudo-F": F_obs, "p": p, "p_Holm": holm(p)})
    glob_row = pd.DataFrame([{"Term": "Treatment (global, 7 df)", "df": T, "SS": ss_t.sum(),
                              "R2_%": 100 * ss_t.sum() / sst_c, "Pseudo-F": F_glob,
                              "p": (1 + cnt_g) / (n_perm + 1), "p_Holm": np.nan}])
    return pd.concat([out, glob_row], ignore_index=True)


def permdisp(Y, groups, n_perm, seed):
    """Homogeneity of dispersion (distance to group centroid; permutation F)."""
    Y = np.asarray(Y, float)
    groups = np.asarray(groups)
    d = np.empty(len(Y))
    for g in np.unique(groups):
        m = groups == g
        d[m] = np.sqrt(((Y[m] - Y[m].mean(0)) ** 2).sum(1))

    def fstat(x):
        gs = np.unique(groups)
        gm = np.array([x[groups == g].mean() for g in gs])
        ng = np.array([(groups == g).sum() for g in gs])
        ssb = (ng * (gm - x.mean()) ** 2).sum()
        ssw = sum(((x[groups == g] - x[groups == g].mean()) ** 2).sum() for g in gs)
        return (ssb / (len(gs) - 1)) / (ssw / (len(x) - len(gs)) + 1e-12)
    f0 = fstat(d)
    rng = np.random.default_rng(seed)
    cnt = sum(fstat(rng.permutation(d)) >= f0 - 1e-12 for _ in range(n_perm))
    return f0, (1 + cnt) / (n_perm + 1)


def conover_pvalues(groups):
    """Conover-Iman post-hoc test after Kruskal-Wallis; raw two-sided p (k x k matrix)."""
    k = len(groups)
    allv = np.concatenate(groups)
    N = len(allv)
    ranks = stats.rankdata(allv)
    rl, i0 = [], 0
    for g in groups:
        rl.append(ranks[i0:i0 + len(g)])
        i0 += len(g)
    H = stats.kruskal(*groups)[0]
    S2 = (np.sum(ranks ** 2) - N * (N + 1.0) ** 2 / 4.0) / (N - 1.0)
    df_err = N - k
    denom = S2 * ((N - 1.0 - H) / df_err)
    P = np.ones((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            t = abs(rl[i].mean() - rl[j].mean()) / (np.sqrt(max(denom, 1e-12) * (1 / len(rl[i]) + 1 / len(rl[j]))) + 1e-12)
            P[i, j] = P[j, i] = 2 * (1 - stats.t.cdf(t, df_err))
    return P


def cld_letters(names, p_adj, alpha):
    """Compact letter display from an adjusted p-value matrix."""
    k = len(names)
    sets = [set(range(k))]
    for i in range(k):
        for j in range(i + 1, k):
            if p_adj[i, j] < alpha:
                new = []
                for s in sets:
                    new.extend([s - {j}, s - {i}] if (i in s and j in s) else [s])
                sets = [a for ia, a in enumerate(new) if not any(a < b for ib, b in enumerate(new) if ia != ib)]
                uniq = []
                for s in sets:
                    if s not in uniq:
                        uniq.append(s)
                sets = uniq
    sets.sort(key=lambda s: min(s))
    res = {n: "" for n in names}
    for li, s in enumerate(sets):
        for e in s:
            res[names[e]] += "abcdefghijklmnopqrstuvwxyz"[li]
    return res


def tucker_congruence(A, B):
    A = A / np.linalg.norm(A, axis=0)
    B = B / np.linalg.norm(B, axis=0)
    return np.abs(A.T @ B)


# =============================================================================
# 7. REVISED IEF
# =============================================================================
def signflip_p(d):
    d = np.asarray(d, float)
    signs = np.array(list(product([-1, 1], repeat=len(d))))
    obs = abs(d.sum())
    return float(np.mean(np.abs(signs @ d) >= obs - 1e-12))


def ief_tables(df, s_col, extra=("TCH", "TPH")):
    rows = []
    for s_ in SEASONS:
        for cult in CULTIVARS:
            sub = df[(df.Season == s_) & (df.Cultivar == cult)]
            for var in [s_col] + list(extra):
                vec = lambda t: sub[sub.Treatment == t].sort_values("Block")[var].values.astype(float)
                Cn, En, Cs, Es = vec("Control - nSi"), vec("Stress - nSi"), vec("Control + nSi"), vec("Stress + nSi")
                ief_n, ief_s = En / Cn.mean() * 100, Es / Cs.mean() * 100
                d_ief = ief_s - ief_n                                   # paired by block
                d_abs = (Es - Cs.mean()) - (En - Cn.mean())             # difference-in-differences (variable units)
                nb = len(d_ief)
                tcrit = stats.t.ppf(0.975, nb - 1)
                se = d_ief.std(ddof=1) / np.sqrt(nb)
                rows.append(dict(Season=s_, Cultivar=cult, Variable=var,
                                 ratio_scale="no (score)" if var == s_col else "yes",
                                 IEF_minus_nSi=ief_n.mean(), IEF_plus_nSi=ief_s.mean(),
                                 Delta_nSi_pp=d_ief.mean(), SD=d_ief.std(ddof=1),
                                 CI95_low=d_ief.mean() - tcrit * se, CI95_high=d_ief.mean() + tcrit * se,
                                 p_signflip=signflip_p(d_ief),
                                 Delta_diff_in_diff=d_abs.mean()))
    return pd.DataFrame(rows)


# =============================================================================
# 8. PCA / BETWEEN-SEASON STRUCTURE DIAGNOSTICS
# =============================================================================
def parallel_analysis(Z, n_iter, rng, pct=95):
    n, p = Z.shape
    e = np.empty((n_iter, p))
    for i in range(n_iter):
        Zp = np.column_stack([rng.permutation(Z[:, j]) for j in range(p)])
        e[i] = np.sort(np.linalg.eigvalsh(np.corrcoef(Zp.T)))[::-1]
    return np.percentile(e, pct, axis=0)


def season_structure(df, pca_vars, K, n_boot, rng):
    idx = [VARS.index(v) for v in pca_vars]
    V, C = {}, {}
    for s_ in SEASONS:
        x = df.loc[df.Season == s_, VARS].values.astype(float)[:, idx]
        z = (x - x.mean(0)) / x.std(0, ddof=1)
        C[s_] = np.corrcoef(z.T)
        _, v = np.linalg.eigh(C[s_])
        V[s_] = v[:, ::-1][:, :K]
    tu = tucker_congruence(V[2025], V[2026])
    iu = np.triu_indices(len(idx), 1)
    r_corr = np.corrcoef(C[2025][iu], C[2026][iu])[0, 1]
    # loading stability by block bootstrap (within-season z-score)
    Zfull = np.vstack([(lambda x: (x - x.mean(0)) / x.std(0, ddof=1))(
        df.loc[df.Season == s_, VARS].values.astype(float)[:, idx]) for s_ in SEASONS])
    blocks = df.Block.values
    ref = np.linalg.eigh(np.cov(Zfull.T))[1][:, ::-1][:, :K]
    stab = []
    ub = sorted(set(blocks))
    for _ in range(n_boot):
        pick = rng.choice(ub, len(ub), replace=True)
        rows = np.concatenate([np.where(blocks == b)[0] for b in pick])
        Vb = np.linalg.eigh(np.cov(Zfull[rows].T))[1][:, ::-1][:, :K]
        stab.append(tucker_congruence(ref, Vb).max(axis=0))
    return tu, r_corr, np.mean(stab, axis=0)


def season_raw_comparison(df, vars_=("TCH", "TPH", "PC", "TRS", "Fi", "Pu", "NT", "SL", "SD")):
    w = df.pivot(index="Acronym", columns="Season", values=list(vars_))
    rows = []
    for v in vars_:
        d = (w[(v, 2026)] - w[(v, 2025)]).dropna()
        try:
            p = stats.wilcoxon(d).pvalue
        except ValueError:
            p = np.nan
        rows.append(dict(Variable=v, median_2025=w[(v, 2025)].median(), median_2026=w[(v, 2026)].median(),
                         median_diff_2026_minus_2025=d.median(), p_paired_Wilcoxon=p))
    out = pd.DataFrame(rows)
    out["p_Holm"] = holm(out.p_paired_Wilcoxon.fillna(1.0).values)
    return out


# =============================================================================
# 9. MAIN PROGRAM
# =============================================================================
def main():
    t0 = time.time()
    os.makedirs(CFG["out_dir"], exist_ok=True)
    rng = np.random.default_rng(CFG["seed"])
    say(f"Mode: {'QUICK (smoke test)' if QUICK else 'FULL'} | dataset: {CFG['data_path']}")
    df = load_data(CFG["data_path"])
    data_qa(df)

    pca_main = [v for v in VARS if v not in CFG["exclude_derived"]]
    pca_all = list(VARS)
    pca_nomass = [v for v in pca_main if v not in MASS_VARS]
    K = CFG["K"]
    n_per_season = int((df.Season == 2025).sum())
    say(f"\nVariables in the PCA (main analysis): {len(pca_main)} (excluded as derived: {CFG['exclude_derived']})")
    say(f"n/p ratio per season = {n_per_season}/{len(pca_main)} = {n_per_season / len(pca_main):.2f}  "
        f"(low: unstable covariance estimates; see loading stability)")

    # ---------------- 9.1 PCA diagnostics ----------------
    full = FuzzyPipeline(pca_main, K).fit(df, seed=0)
    Zall = full.zscore(df)[:, [VARS.index(v) for v in pca_main]]
    eig_corr = np.sort(np.linalg.eigvalsh(np.corrcoef(Zall.T)))[::-1]
    pa = parallel_analysis(Zall, CFG["n_pa"], rng)
    evr = eig_corr / eig_corr.sum()
    n_kaiser, n_pa = int((eig_corr > 1).sum()), int((eig_corr > pa).sum())
    say("\n=== PCA: CHOICE OF K ===")
    say(f"Variance explained by PCA1-PCA{K}: " + ", ".join(f"{100 * v:.1f}%" for v in evr[:K])
        + f" | cumulative = {100 * evr[:K].sum():.1f}%  ({100 * (1 - evr[:K].sum()):.1f}% discarded)")
    say(f"Kaiser criterion: {n_kaiser} PCs | Horn's parallel analysis (p95): {n_pa} PCs  -> K={K} is a parsimony "
        f"choice for the FIS (3^K rules), not a statistical criterion.")
    diag = []
    for k in range(K):
        sc = full.scores[:, k]
        r2t = r2_groups(sc, df.Treatment.values)[0]
        r2c = r2_groups(sc, df.Cultivar.values)[0]
        diag.append(dict(Component=f"PCA{k + 1}", var_explained_pct=100 * full.evr[k], eigenvalue=full.eig[k],
                         parallel_analysis_threshold=pa[k], corr_with_Z_vigor=full.pol_corr[k],
                         weak_polarity=abs(full.pol_corr[k]) < 0.30,
                         R2_Treatment_pct=100 * r2t, R2_Cultivar_pct=100 * r2c))
        if abs(full.pol_corr[k]) < 0.30:
            say(f"  [WARNING] PCA{k + 1}: |r| with Z_vigor = {abs(full.pol_corr[k]):.2f} < 0.30 -> "
                f"'higher = better' polarity has no semantic basis.")
    TABLES["PCA_diagnostics"] = pd.DataFrame(diag)
    say(TABLES["PCA_diagnostics"].round(3).to_string(index=False))

    tu, r_cc, stab = season_structure(df, pca_main, K, CFG["n_boot"], rng)
    say("\n=== COVARIANCE STRUCTURE BETWEEN SEASONS (supports or undermines pooling) ===")
    say(f"Correlation between the 2025 and 2026 correlation matrices (upper triangle): r = {r_cc:.3f}")
    say("Tucker congruence |phi| between 2025 PCs (rows) and 2026 PCs (columns):")
    tu_df = pd.DataFrame(tu, index=[f"PCA{i + 1}-2025" for i in range(K)], columns=[f"PCA{i + 1}-2026" for i in range(K)])
    say(tu_df.round(2).to_string())
    say("Loading stability (block bootstrap, mean congruence with the full-sample loadings): "
        + ", ".join(f"PCA{i + 1}={v:.2f}" for i, v in enumerate(stab)))
    TABLES["season_structure"] = tu_df

    # ---------------- 9.2 Full FIS: PSO stability ----------------
    say("\n=== PSO: STABILITY AND RULE SUPPORT (fit on all data) ===")
    mean_rho, min_rho = run_stability(full.runs)
    Js = np.array([r["J"] for r in full.runs])
    say(f"{len(full.runs)} independent runs | J: min={Js.min():.4f}, median={np.median(Js):.4f}, max={Js.max():.4f} "
        f"| mean/min Spearman between run scores = {mean_rho:.3f}/{min_rho:.3f}")
    n_unsup = int((full.rule_support < 1.0).sum())
    say(f"Rules with effective n < 1 observation (no support; consequent ~ ridge prior): {n_unsup} of {3 ** K}")
    TABLES["rule_support"] = pd.DataFrame(
        [dict(rule=i + 1, combination="-".join("LMH"[j] for j in m), effective_n=full.rule_support[i], theta=full.theta[i])
         for i, m in enumerate(full.RIDX)])
    df["S_Fuzzy_insample"] = full.S_in

    # ---------------- 9.3 CROSS-VALIDATION (OOF) ----------------
    say("\n=== LEAVE-ONE-BLOCK-OUT CROSS-VALIDATION (inference uses OOF scores only) ===")
    S_oof, base_oof = lobo_cv(df, pca_main, K, CFG["n_runs"])
    df["S_Fuzzy"] = S_oof
    base_in = full.baselines(df)
    cmp_rows = []
    for name, ins, oof in [("FIS (PSO)", full.S_in, S_oof)] + [(k, base_in[k], base_oof[k]) for k in base_in]:
        _, r_in_adj = r2_groups(ins, df.Group.values)
        _, r_o_adj = r2_groups(oof, df.Group.values)
        cmp_rows.append(dict(Index=name, adjR2_in_sample=r_in_adj, adjR2_OOF=r_o_adj,
                             Spearman_OOF_with_FIS=stats.spearmanr(oof, S_oof)[0]))
    cmp = pd.DataFrame(cmp_rows)
    TABLES["FIS_vs_baselines"] = cmp
    say("Adjusted R2 of the 16 groups (Season x Cultivar x Treatment): in-sample vs out-of-sample")
    say(cmp.round(3).to_string(index=False))
    gap = cmp.loc[0, "adjR2_in_sample"] - cmp.loc[0, "adjR2_OOF"]
    say(f"-> FIS optimism (in-sample - OOF) = {gap:.3f}. "
        + ("The FIS does NOT outperform the best simple index out of sample."
           if cmp.loc[0, "adjR2_OOF"] <= cmp.iloc[1:]["adjR2_OOF"].max()
           else "The FIS outperforms the simple indices out of sample."))

    # carry-over between seasons (repeated plots)
    w = df.pivot(index="Acronym", columns="Season", values="S_Fuzzy")
    rho_co = stats.spearmanr(w[2025], w[2026])
    say(f"Dependence between seasons (same plots): Spearman OOF S 2025 x 2026 = {rho_co[0]:.3f} (p={rho_co[1]:.3f}) "
        f"-> seasons are NOT independent; inference is done per season.")

    # label-permutation test (how much separation PSO can fabricate from noise)
    m_null = CFG["n_runs_null"]
    obs_disc = best_of(full.runs, m_null)["j_disc"]
    null = []
    for b in range(CFG["n_null"]):
        perm = np.random.default_rng(10_000 + b).permutation(df.Group.values)
        pp = FuzzyPipeline(pca_main, K).fit(df, seed=50 + b, n_runs=m_null, groups=perm)
        null.append(best_of(pp.runs, m_null)["j_disc"])
    null = np.array(null)
    p_null = (1 + (null <= obs_disc).sum()) / (len(null) + 1)
    sd_null = null.std(ddof=1) if len(null) > 1 else float("nan")
    say(f"\nLabel-permutation test: optimised SS_within/SS_total = {obs_disc:.3f} (true labels) "
        f"vs {null.mean():.3f} +- {sd_null:.3f} (shuffled labels, n={len(null)}); p = {p_null:.3f}")
    TABLES["label_permutation"] = pd.DataFrame({"j_disc_null": null})

    # ---------------- 9.4 SENSITIVITY ----------------
    say("\n=== SENSITIVITY (OOF adjusted R2 of the 16 groups) ===")
    sens = []
    combos = [("Main (K=3, derived variables excluded)", pca_main, K, None),
              ("K=2 (9 rules)", pca_main, 2, CFG["n_runs_sens"]),
              ("All 24 variables", pca_all, K, CFG["n_runs_sens"]),
              ("Without mass variables", pca_nomass, K, CFG["n_runs_sens"])]
    for name, pv, kk, nr in combos:
        if nr is None:
            so, bo = S_oof, base_oof
        else:
            so, bo = lobo_cv(df, pv, kk, nr)
        row = dict(Configuration=name, n_vars=len(pv), K=kk, adjR2_OOF_FIS=r2_groups(so, df.Group.values)[1])
        for k, v in bo.items():
            row[f"adjR2_OOF_{k}"] = r2_groups(v, df.Group.values)[1]
        sens.append(row)
    TABLES["sensitivity"] = pd.DataFrame(sens)
    say(TABLES["sensitivity"].round(3).to_string(index=False))

    # ---------------- 9.5 INFERENCE PER SEASON ----------------
    say("\n=== INFERENCE PER SEASON (factorial PERMANOVA, blocks as strata; OOF scores) ===")
    idx_main = [VARS.index(v) for v in pca_main]
    perm_tabs, s_perm, disp_rows = {}, {}, []
    for s_ in SEASONS:
        sub = df[df.Season == s_].reset_index(drop=True)
        x = sub[VARS].values.astype(float)[:, idx_main]
        zs = (x - x.mean(0)) / x.std(0, ddof=1)
        mv = factorial_permanova(zs, sub, CFG["n_perm"], CFG["seed"] + s_)
        uv = factorial_permanova(sub["S_Fuzzy"].values, sub, CFG["n_perm"], CFG["seed"] + s_ + 1)
        perm_tabs[s_], s_perm[s_] = mv, uv
        TABLES[f"PERMANOVA_multivar_{s_}"] = mv
        TABLES[f"PERMANOVA_S_Fuzzy_{s_}"] = uv
        glab = (sub.Cultivar + "_" + sub.Treatment).values
        f0, pd0 = permdisp(zs, glab, CFG["n_perm"], CFG["seed"] + s_ + 2)
        f1, pd1 = permdisp(sub["S_Fuzzy"].values[:, None], glab, CFG["n_perm"], CFG["seed"] + s_ + 3)
        disp_rows += [dict(Season=s_, Response="multivariate (z)", F=f0, p=pd0),
                      dict(Season=s_, Response="S_Fuzzy OOF", F=f1, p=pd1)]
        say(f"\nSeason {s_} - multivariate PERMANOVA ({len(pca_main)} z-scored variables):")
        say(mv.round(4).to_string(index=False))
        say(f"Season {s_} - permutation ANOVA on S_Fuzzy (OOF):")
        say(uv.round(4).to_string(index=False))
        say(f"PERMDISP (homogeneity of dispersion, 8 groups): multivariate p={pd0:.4f} | S_Fuzzy p={pd1:.4f}"
            + ("  [WARNING: heterogeneous dispersion; interpret PERMANOVA with caution]" if min(pd0, pd1) < 0.05 else ""))
    TABLES["PERMDISP"] = pd.DataFrame(disp_rows)

    # Conover-Iman (Holm) within cultivar; exact Mann-Whitney (Holm) between cultivars
    say("\n=== MULTIPLE COMPARISONS (Holm) ===")
    say(f"Smallest attainable exact two-sided p for Mann-Whitney with n=4 vs 4 = {2 / 70:.4f} "
        f"(after Holm with 4 comparisons: {min(1, 4 * 2 / 70):.3f}) -> very low power; prioritise the factorial PERMANOVA.")
    cld_map, kw_rows, mw_rows = {}, [], []
    for s_ in SEASONS:
        sub_a = df[df.Season == s_]
        pm, vals = [], []
        for t in TRT_ORDER:
            a = sub_a[(sub_a.Cultivar == CULTIVARS[0]) & (sub_a.Treatment == t)]["S_Fuzzy"].values
            b = sub_a[(sub_a.Cultivar == CULTIVARS[1]) & (sub_a.Treatment == t)]["S_Fuzzy"].values
            pm.append(stats.mannwhitneyu(a, b, alternative="two-sided", method="exact").pvalue)
            vals.append((a.mean(), b.mean()))
        pm_adj = holm(pm)
        upper = {}
        for t, praw, padj, (ma, mb) in zip(TRT_ORDER, pm, pm_adj, vals):
            mw_rows.append(dict(Season=s_, Treatment=t, exact_p=praw, p_Holm=padj))
            if padj < CFG["alpha"]:
                upper[(CULTIVARS[0], t)], upper[(CULTIVARS[1], t)] = ("A", "B") if ma >= mb else ("B", "A")
            else:
                upper[(CULTIVARS[0], t)] = upper[(CULTIVARS[1], t)] = "A"
        for cult in CULTIVARS:
            sc = sub_a[sub_a.Cultivar == cult]
            gd = {t: sc[sc.Treatment == t]["S_Fuzzy"].values for t in TRT_ORDER}
            H, pk = stats.kruskal(*gd.values())
            kw_rows.append(dict(Season=s_, Cultivar=cult, H=H, p_KW_descriptive=pk))
            names = sorted(gd, key=lambda k: gd[k].mean(), reverse=True)
            P = conover_pvalues([gd[n] for n in names])
            iu = np.triu_indices(len(names), 1)
            Padj = np.ones_like(P)
            Padj[iu] = holm(P[iu])
            Padj = np.minimum(Padj, Padj.T)
            np.fill_diagonal(Padj, 1)
            low = cld_letters(names, Padj, CFG["alpha"])
            for t in TRT_ORDER:
                cld_map[(s_, cult, t)] = f"{low[t]}{upper[(cult, t)]}"
    TABLES["MannWhitney_cultivars_Holm"] = pd.DataFrame(mw_rows)
    TABLES["KruskalWallis_descriptive"] = pd.DataFrame(kw_rows)
    say(TABLES["MannWhitney_cultivars_Holm"].round(4).to_string(index=False))

    # ---------------- 9.6 REVISED IEF ----------------
    ief = ief_tables(df, "S_Fuzzy")
    TABLES["IEF_revised"] = ief
    say("\n=== IEF AND NET nSi GAIN (paired by block; OOF scores) ===")
    say("Original IEF = stress/control x 100. For the score (no true zero) the ratio is not interpretable on a ratio scale;")
    say("the valid reading is the difference-in-differences (Delta_diff_in_diff, points) and the Water x nSi term of the permutation ANOVA.")
    say("IEF on TCH/TPH (ratio scale) is the analogue of Fernandez's index. 95% CI by t (df=3); exact sign-flip test (smallest possible p = 0.125).")
    say(ief.round(3).to_string(index=False))

    # ---------------- 9.7 PLANT CANE vs RATOON (raw variables) ----------------
    sr = season_raw_comparison(df)
    TABLES["seasons_raw_variables"] = sr
    say("\n=== PLANT CANE vs RATOON: raw variables (paired Wilcoxon by plot, Holm) ===")
    say(sr.round(4).to_string(index=False))
    say("(S_Fuzzy is NOT comparable between seasons: within-season z-scoring removes the season effect.)")

    # ---------------- 9.8 DESCRIPTIVE TABLES ----------------
    tab = df.groupby(["Season", "Cultivar", "Treatment"]).agg(
        S_OOF_mean=("S_Fuzzy", "mean"), S_OOF_sd=("S_Fuzzy", "std"),
        S_insample_mean=("S_Fuzzy_insample", "mean")).reset_index()
    TABLES["scores_by_treatment"] = tab
    say("\n=== SCORES BY TREATMENT ===")
    say(tab.round(2).to_string(index=False))

    # ---------------- 9.9 FIGURES ----------------
    make_figures(df, full, eig_corr, pa, cmp, ief, perm_tabs, s_perm, cld_map)

    # ---------------- 9.10 EXPORT ----------------
    out = CFG["out_dir"]
    df.to_csv(os.path.join(out, "scores_by_plot.csv"), index=False)
    with pd.ExcelWriter(os.path.join(out, "results_tables.xlsx")) as xw:
        for name, t in TABLES.items():
            t.to_excel(xw, sheet_name=name[:31], index=not isinstance(t.index, pd.RangeIndex))
    say(f"\nTotal time: {time.time() - t0:.0f} s | outputs in: {os.path.abspath(out)}")
    with open(os.path.join(out, "methodological_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(REPORT))


# =============================================================================
# 10. FIGURES
# =============================================================================
def top_loading_label(V, names, k, n=3):
    o = np.argsort(-np.abs(V[:, k]))[:n]
    return ", ".join(names[i] for i in o)


def make_figures(df, full, eig_corr, pa, cmp, ief, perm_tabs, s_perm, cld_map):
    K = full.K
    names = full.pca_vars
    palette = {"Control - nSi": "#1f77b4", "Control + nSi": "#2ca02c",
               "Stress - nSi": "#d62728", "Stress + nSi": "#ff7f0e"}
    markers = {2025: "o", 2026: "^"}
    letters = ['A','B','C','D']

    # A) biplot + scree with parallel analysis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 11), dpi=300)
    for (s_, trt), g in df.groupby(["Season", "Treatment"]):
        i = g.index.values
        ax1.scatter(full.scores[i, 0], full.scores[i, 1], c=palette[trt], marker=markers[s_], s=65,
                    alpha=0.8, edgecolors="k", linewidth=0.5, label=f"{trt} ({s_})")
    sc_arrow = 0.8 * np.abs(full.scores[:, :2]).max() / np.abs(full.V[:, :2]).max()
    for i, v in enumerate(names):
        ax1.arrow(0, 0, full.V[i, 0] * sc_arrow, full.V[i, 1] * sc_arrow, color="#444444", alpha=0.7,
                  width=0.01, head_width=0.12)
        ax1.text(full.V[i, 0] * sc_arrow * 1.1, full.V[i, 1] * sc_arrow * 1.1, v, fontsize=11, ha="center", va="center")
    ax1.axhline(0, color="gray", ls="--", lw=0.8)
    ax1.axvline(0, color="gray", ls="--", lw=0.8)
    ax1.set_xlabel(f"PCA1 ({full.evr[0] * 100:.1f}%)")
    ax1.set_ylabel(f"PCA2 ({full.evr[1] * 100:.1f}%)")
    ax1.set_title(f"(A) PCA biplot")
    ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12, frameon=True)
    ax1.grid(True, ls=":", alpha=0.5)
    nshow = min(10, len(eig_corr))
    comps = np.arange(1, nshow + 1)
    ax2.bar(comps, eig_corr[:nshow], color="#4c72b0", alpha=0.85, label="Observed eigenvalue")
    ax2.plot(comps, pa[:nshow], color="#c44e52", marker="o", lw=2, label="Parallel analysis (p95)")
    ax2.axhline(1, color="gray", ls=":", label="Kaiser (=1)")
    ax2.axvline(K + 0.5, color="darkgreen", ls="--", lw=1.2,
                label=f"FIS cut-off (PCA1-{K} = {100 * full.evr[:K].sum():.1f}% of variance)")
    ax2.set_xticks(comps)
    ax2.set_xlabel("Principal component")
    ax2.set_ylabel("Eigenvalue")
    ax2.set_title("(B) Scree plot")
    ax2.legend(fontsize=12)
    ax2.grid(True, ls=":", alpha=0.5)
    plt.tight_layout()
    finish_fig(fig, "pca_biplot_scree")

    # B) membership functions and PSO convergence (all runs)
    ncol = 2
    nrow = int(np.ceil((K + 1) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 9), dpi=300)
    axes = np.atleast_1d(axes).ravel()
    xg = np.linspace(0, 1, 300)
    k_letter = 0
    for k in range(K):
        ax = axes[k]
        for m, (lab, col) in enumerate(zip(["Low", "Medium", "High"], ["#d62728", "#ff7f0e", "#2ca02c"])):
            y = np.exp(-0.5 * ((xg - full.centers[k, m]) / full.sigmas[k, m]) ** 2)
            ax.plot(xg, y, color=col, lw=2.2, label=rf"{lab} ($c={full.centers[k, m]:.2f}$, $\sigma={full.sigmas[k, m]:.2f}$)")
            ax.fill_between(xg, 0, y, color=col, alpha=0.12)
        ax.set_title(f"({letters[k_letter]}) PCA{k + 1} ({full.evr[k] * 100:.1f}%)")
        ax.set_xlabel(f"Normalised PCA{k + 1} score")
        ax.set_ylabel(r"Membership degree ($\mu$)")
        ax.legend(loc="upper right", fontsize=12)
        ax.grid(True, ls=":", alpha=0.6)
        k_letter += 1

    axc = axes[K]
    for i, r in enumerate(full.runs):
        axc.plot(r["hist_g"], lw=1.4, alpha=0.8, label=f"Run {i + 1}")
    axc.set_title(f"({letters[k_letter]}) PSO convergence")
    axc.set_xlabel("Iteration")
    axc.set_ylabel(r"$J_{total}$")
    axc.legend(fontsize=12, ncol=2)
    axc.grid(True, ls=":", alpha=0.6)
    for ax in axes[K + 1:]:
        ax.axis("off")
    plt.tight_layout()
    finish_fig(fig, "pso_mfs_convergence")

    # C) FIS vs simple indices (in-sample vs out-of-sample)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(cmp))
    ax.bar(x - 0.19, cmp.adjR2_in_sample, 0.38, label="In-sample", color="#8da0cb", edgecolor="k")
    ax.bar(x + 0.19, cmp.adjR2_OOF, 0.38, label="Out-of-sample (LOBO-CV)", color="#fc8d62", edgecolor="k")
    ax.set_xticks(x)
    ax.set_xticklabels(cmp.Index, rotation=15)
    ax.set_ylabel(r"Adjusted $R^2$ (16 groups)")
    ax.set_title("")
    ax.legend()
    ax.grid(axis="y", ls=":", alpha=0.6)
    plt.tight_layout()
    finish_fig(fig, "fis_vs_baselines")

    # D) final statistical comparison (OOF scores)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300, sharey=True)
    pal = ["#4c72b0", "#c44e52", "#55a868", "#dd8452"]
    k_letter = 0
    for ia, s_ in enumerate(SEASONS):
        ax = axes[ia]
        sub = df[df.Season == s_]
        x = np.arange(len(CULTIVARS))
        wd = 0.18
        offs = [-1.5 * wd, -0.5 * wd, 0.5 * wd, 1.5 * wd]
        for ti, t in enumerate(TRT_ORDER):
            mm, ss = [], []
            for cult in CULTIVARS:
                v = sub[(sub.Cultivar == cult) & (sub.Treatment == t)]["S_Fuzzy"].values
                mm.append(v.mean())
                ss.append(v.std(ddof=1))
            ax.bar(x + offs[ti], mm, wd, yerr=ss, capsize=4, color=pal[ti], edgecolor="black", linewidth=0.7,
                   alpha=0.9, label=t if ia == 0 else "", error_kw={"elinewidth": 1.2})
            for ci, cult in enumerate(CULTIVARS):
                ax.text(x[ci] + offs[ti], mm[ci] + ss[ci] + 2.2, cld_map[(s_, cult, t)], ha="center", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(CULTIVARS, fontsize=12)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", ls=":", alpha=0.6)
        ax.set_title(rf"({letters[k_letter]}) Season {s_}", fontsize=12)
        k_letter += 1
        mv, uv = perm_tabs[s_], s_perm[s_]
        g = lambda tb, term: tb.loc[tb.Term == term, "p"].values[0]
        '''txt = ("Factorial PERMANOVA (multivariate, blocks as strata): "
               f"Cult p{fmt_p(g(mv, 'Cultivar'))}; Water p{fmt_p(g(mv, 'Water'))}; nSi p{fmt_p(g(mv, 'nSi'))}; "
               f"Water x nSi p{fmt_p(g(mv, 'Water x nSi'))}\n"
               "Permutation ANOVA on S_Fuzzy: "
               f"Cult p{fmt_p(g(uv, 'Cultivar'))}; Water p{fmt_p(g(uv, 'Water'))}; nSi p{fmt_p(g(uv, 'nSi'))}; "
               f"Water x nSi p{fmt_p(g(uv, 'Water x nSi'))}")
        ax.text(0.02, 0.985, txt, transform=ax.transAxes, fontsize=7.8, va="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95))'''
    axes[0].set_ylabel(r"Out-of-fold fuzzy score ($S_{Fuzzy}$, 0-100)")
    axes[0].set_xlabel("Cultivar")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=12, title="Treatments", framealpha=0.95)
    '''plt.suptitle("Lowercase: treatments (Conover-Iman, Holm) | Uppercase: cultivars (exact Mann-Whitney, Holm)",
                 fontsize=11.5, y=1.02)'''
    plt.tight_layout()
    finish_fig(fig, "fuzzy_statistics_permanova", pdf=True)

    # E) net nSi gain
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    k_letter = 0
    for ax, var, ycol, ylab in [(axes[0], "S_Fuzzy", "Delta_diff_in_diff", r"$\Delta$nSi on the score (points; diff.-in-diff.)"),
                                (axes[1], "TCH", "Delta_nSi_pp", r"$\Delta$nSi on TCH IEF (%)")]:
        sub = ief[ief.Variable == var]
        lab = [f"{r.Cultivar}\n{r.Season}" for r in sub.itertuples()]
        if var == "S_Fuzzy":
            ax.bar(lab, sub[ycol], color="#8da0cb", edgecolor="k")
            title = "(" + letters[k_letter] + ") Score"
        else:
            ax.bar(lab, sub[ycol], yerr=[sub[ycol] - sub.CI95_low, sub.CI95_high - sub[ycol]], capsize=4,
                   color="#8da0cb", edgecolor="k")
            title = "(" + letters[k_letter] + ") TCH"

        ax.axhline(0, color="k", lw=0.8)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", ls=":", alpha=0.6)
        k_letter += 1
    plt.tight_layout()
    finish_fig(fig, "nSi_net_gain")


if __name__ == "__main__":
    main()
