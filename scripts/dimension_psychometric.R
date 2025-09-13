# =================================================================================================
# Psychometric Base–Fibre Manifold + Diagnosis Mapping + Fibre Contribution Diagnostics
# -------------------------------------------------------------------------------------------------
# Purpose
#   End-to-end pipeline to:
#     1) Load and coerce mixed psychometric data into a consistent design matrix (X).
#     2) Compute Gower distances, optimise per-variable weights to minimise intrinsic dimension,
#        and select an informative, minimally redundant set of predictors (X_pred).
#     3) Derive the low-dimensional **Base manifold** via PCA on X_pred (m_star PCs fixed by heuristic).
#     4) Residualise X_pred w.r.t. Base, obtain fold-safe **Fibre features** (E), and whiten them.
#     5) Characterise each psychometric variable's role:
#          - Base-aligned, Fibre-specific, or Mixed.
#          - R²_base from GAM; R²_fibre from kNN on residuals; permutation-based p-values.
#     6) Explore independent low-dimensional structure inside Fibre residuals (optional).
#     7) For each diagnosis:
#          - Build fold-consistent Base and Fibre predictions.
#          - Evaluate predictive power (AUC, ΔAUC, bootstrap CI, DeLong p).
#          - Compare Base-only vs Fibre-only vs Stacked models.
#     8) Output maps, tables, and diagnostics to support manifold interpretation.
#
# Inputs
#   - psychometric_matrix.csv
#       Wide table, one row per subject.
#       Columns: mixed psychometric variables + optional diagnosis columns.
#       ID column required ('participant_id'), otherwise the first column is used.
#
#   - long_diagnoses.csv
#       Long table with at least:
#         • participant_id: matches IDs in the psychometric matrix.
#         • diagnosis: string label for diagnosis.
#       Diagnoses are transformed into a wide binary Dx table (multi-label, 0/1).
#
# Outputs
#   Core outputs:
#     - roles_df.csv
#         Psychometric variables with role classification:
#           * R²_base, R²_fibre, permutation p-values, effect size, PC contribution, role.
#     - dx_auc_summary.csv
#         AUC/ΔAUC table comparing Base vs Fibre vs Stacked per diagnosis.
#     - FIG_AUC_<diagnosis>.png
#         ROC plots for each diagnosis, annotated with AUC ± bootstrap CI.
#
#   Intermediate/diagnostic outputs:
#     - embedding_Base.csv          # Coordinates of subjects in m_star Base PCs.
#     - residuals_Fibre.csv        # Fold-safe Fibre features (E) per subject.
#     - fibre_self_decomp.rds      # Optional fibre-only manifold, ID & curvature metrics.
#     - roles_plots/FIG_roles_*    # Role diagnostic plots (optional).
#
# Key methods and references
#   - Gower distance for mixed data:
#       cluster::daisy(metric = "gower")
#   - Intrinsic dimension estimation:
#       TwoNN (Facco et al., Sci Rep 2017) and Levina–Bickel MLE.
#   - Variable selection:
#       Constrained weight optimisation on Gower distances.
#   - Manifold decomposition:
#       Base = PCA(m_star PCs), Fibre = residuals(X_pred | Base).
#   - Variable role classification:
#       GAM R² + permutation tests + fibre residual predictability.
#   - Diagnosis modelling:
#       Fold-safe OOF predictions for Base vs Fibre vs Stacked glmnet models.
#       ROC/AUC + bootstrap CI + DeLong comparison.
#
# Reproducibility and compute notes
#   - All random seeds centralised at the top for Base, Fibre, and bootstrap reproducibility.
#   - Threads pinned to 1 in all parallel futures to avoid race conditions.
#   - Uses fold-consistent partitioning: the same cross-validation folds propagate through
#     Base, Fibre, and Dx models → guarantees comparability and no leakage.
#   - Optional Fibre self-decomposition runs PCA on residuals to detect independent structure.
#
# Rationale for selected defaults
#   - M_STAR_FIXED = 2L: principal Base manifold assumed 2D (heuristic; TC sweep used for QC only).
#   - K_FIBRE_CAP = 3: max fibre PCs used in diagnosis models.
#   - EPS_DEDUP: chosen adaptively via knee on ε vs retained pairs curve.
#   - CV_FOLDS = 5: balances statistical power and fibre/Base comparability.
#   - BOOT_B = 2000: sufficient for stable ΔAUC bootstrap CIs.
#
# Author: Afonso Dinis Ribeiro
# Date:   2025-08-23
# =================================================================================================

# ====================== 0) Preamble: packages, seed, threads ===================
# Goal: load libraries quietly, fix randomness for reproducibility, and avoid
# thread oversubscription (BLAS/OpenMP). This keeps results stable and laptops cool.

suppressPackageStartupMessages({
  library(stats);  library(utils);  library(Matrix)
  library(cluster)      # Gower distances (`daisy`) and clustering
  library(glmnet)       # fast regularised GLMs for OOF predictions
  library(mgcv)         # smooth regressions for base residualisation
  library(RANN)         # k-NN search (fast)
  library(dplyr); library(tidyr)
  # Optional plotting / ROC packages: used when installed, otherwise code falls back
  if (!requireNamespace("ggplot2", quietly = TRUE)) NULL
  if (!requireNamespace("pROC",    quietly = TRUE)) NULL
  if (!requireNamespace("ggrepel", quietly = TRUE)) NULL
})

suppressPackageStartupMessages({ if (requireNamespace("compiler", quietly=TRUE)) compiler::enableJIT(3) })

# ---- Reproducibility ----------------------------------------------------------
SEED_GLOBAL <- 42L        # single source of truth for RNG unless otherwise noted
set.seed(SEED_GLOBAL)
RNGkind("L'Ecuyer-CMRG")

# ---- Threading (avoid nested parallelism + noisy CPU fans) -------------------
# We pin math libraries to 1 thread here. Later, when we use futures, we repeat
# these pins inside workers (belt-and-braces approach).
Sys.setenv(
  OMP_NUM_THREADS        = "1",
  OMP_PROC_BIND          = "spread",
  OMP_DISPLAY_ENV        = "FALSE",
  KMP_SETTINGS           = "0",
  OPENBLAS_NUM_THREADS   = "1",
  MKL_NUM_THREADS        = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  BLAS_NUM_THREADS       = "1"
)

# ============================ 1) User knobs (tidy) =============================
# --------------------------- CONFIGURATION PARAMETERS --------------------------

# Target selection
PREF_TARGET <- "ANY_DX"     # Target variable: positive class = "any diagnosis" by default.
# Alternative: "NODIAG" for predicting absence of diagnoses.

# Light redundancy hygiene (off for TwoNN-only pipelines)
DO_CORR_TRIM <- FALSE       # If TRUE, drop highly correlated items before modelling.
CORR_THRESH  <- 0.95        # Correlation threshold for redundancy trimming.

# Cross-validation basics
CV_FOLDS     <- 10L          # Maximum number of folds for out-of-fold (OOF) predictions.
SEED_PRED    <- 42L          # Random seed for CV fold assignment and OOF predictions.
SEED_JITTER  <- 1L           # Tiny jitter added to Gower distances to break ties deterministically.
SEED_BOOT    <- 42L          # Random seed for bootstrap CIs (e.g., ΔAUC confidence intervals).
CV_REPEATS   <- 10L          # best for 8–20
MIN_TEST_POS <- 3L           # per test fold (guard for rare dx)
MIN_TEST_NEG <- 3L

# Base dimensionality (diagnostic sweep kept, but we fix m* globally)
M_STAR_FIXED <- 2L          # Number of Base PCs to retain (heuristic)

# Subject-level duplicate collapse
DO_DEDUP        <- TRUE     # If TRUE, collapse near-duplicate subjects by ε-Gower distance.
EPS_DEDUP       <- NA_real_ # ε threshold for deduplication: NA = auto knee detection.
WRITE_DEDUP_CSV <- TRUE     # Export deduplication decisions to CSV if enabled.

# -------- Constrained Gower weight optimisation (TwoNN objective) -------------
# Each item gets a Gower weight; optimiser decays weights that *hurt* manifold quality.
W_MIN          <- 0.00      # Lower bound on any item's Gower weight.
W_STEP_GRID    <- c(0.95, 0.90, 0.75, 0.5, 0.25, 0.10, 0.05, 0.01, 0.00)
# Grid of multiplicative decay factors tested per iteration.
W_BATCH_K      <- 3         # After picking the best decay, apply up to (k-1) extra "good" decays.
W_BATCH_FACTOR <- 0.75      # Damp later decays in the same batch to avoid overshooting.
W_MAX_ITERS    <- NA_integer_
# Maximum optimiser iterations; NA lets optimiser decide automatically.

# Survivor policy (after optimisation)
KEEP_THR_FIXED <- 0.10      # Fixed tail-weight cutoff for survivor selection (not used if USE_KNEE_THR=TRUE).
USE_KNEE_THR   <- TRUE      # Use a "knee" detection algorithm to select surviving items automatically.

# Compute parallelism and sampling
NCORES_PAR     <- max(1, parallel::detectCores() - 1)
# Number of cores to use for parallelisable steps.
N_ROWS_SUB     <- NULL      # Optional row subsample size for weight optimisation (NULL = use all rows).
FIX_REP_SUBSET <- TRUE      # When subsampling reps, use fixed head() rows for reproducibility.

# Optional supervised greedy cull inside survivor set (risk of overfit ⇒ disabled)
DO_SUPERVISED_CULL <- FALSE # If TRUE, iteratively drop weakest items using supervised greedy search.
CULL_MAX_ITERS     <- 8     # Max iterations for supervised culling.
CULL_DROP_K        <- 5     # Drop up to this many items per supervised iteration.
AUC_FLOOR_DROP     <- 0.01  # Minimum tolerated AUC drop before stopping supervised drops.
CULL_OBJECTIVE     <- "TwoNN_core"
# Objective for greedy culling: options = "TwoNN_core", "LB_core", "TwoNN_all".

# Geometry diagnostics (trustworthiness / continuity / dimensionality sweeps)
KS_TC        <- 10:30       # Range of k values for trust & continuity averages.
K_BASE_NEIGH <- 40          # kNN size for Base neighbourhood diagnostics.
K_ID_LO_HI   <- c(8, 20)    # Min & max k for intrinsic dimension (ID) estimates.
PC_MAX       <- 6           # Max PCs to compute when probing manifold geometry.
M_DEFAULT    <- 4           # Default fallback embedding dimensionality if sweep fails.

# Optional diagnosis-based analyses (off by default)
DO_DX_CLUSTER_DIAG <- FALSE # If TRUE, run diagnostic clustering evaluations.
DO_DX_ASSOC        <- FALSE # If TRUE, compute pairwise diagnosis associations.
DX_DENY_NOS        <- TRUE  # If TRUE, exclude "NOS" / unspecified diagnoses from outputs.
DX_PREV_MIN        <- 0.00  # Minimum prevalence threshold for including diagnoses downstream.
DX_CASES_MIN       <- 10    # Minimum case count per diagnosis for inclusion.
N_TOP_PER_DX       <- 80    # Maximum number of items considered per diagnosis for GAM/interaction.

# If TRUE, use typed Gower distances (handles ordered vs unordered factors).
RARE_LEVEL_MIN_PROP <- 0.01
# Drop factor levels with <1% prevalence before distance calculation.
CORE_BAND <- get0("CORE_BAND", ifnotfound = c(0.20, 0.70))
# Quantile band used to define the "core set" for stable ID estimates.
CORE_KNN_K          <- get0("CORE_KNN_K", ifnotfound = 10)
# Number of neighbours used for TwoNN & core-ID metrics.

# Base dimensionality sweep early-stopping (for geometry diagnostics)
EPS_M_GAIN <- 0.003         # Minimum required gain in (trust+continuity)/2 to add another dimension.
M_PATIENCE <- 1             # Allow this many consecutive below-threshold gains before stopping.
LAMBDA_FID <- 0.00          # Optional penalty on fibre-ID during Base dimension selection (off by default).

SIG_Q <- 0.01  # FDR threshold for plotting & calls
# =========================== 2) Data prep for Gower ============================
# Why Gower? It combines distances across mixed data types:
# - numeric items → scaled absolute differences
# - unordered factors → 0/1 mismatch
# - ordered factors → treated as ranks (optionally "ordratio")
# We also "collapse" very rare factor levels to "Other" to avoid tiny categories
# dominating distance noise. A tiny jitter on low-cardinality numerics breaks
# ties so nearest-neighbour ranks are stable.

typed_lists <- function(Xdf){
  Xdf <- as.data.frame(Xdf, check.names = TRUE, stringsAsFactors = FALSE)
  ordr <- names(Xdf)[vapply(Xdf, is.ordered, logical(1))]
  asym <- names(Xdf)[vapply(Xdf, function(z) is.factor(z) && nlevels(z) == 2, logical(1))]
  list(ordratio = ordr, asymm = asym)
}

make_typed_X <- get0("make_typed_X", ifnotfound = function(X, min_prop = 0.01){
  collapse_rare <- function(f, min_prop = 0.01){
    f <- droplevels(f)
    p <- prop.table(table(f))
    rare <- names(p)[p < min_prop]
    if (!length(rare)) return(f)
    f2 <- as.character(f); f2[f2 %in% rare] <- "Other"; factor(f2)
  }
  X2 <- X
  for (nm in names(X2))
    if (is.factor(X2[[nm]]) && !is.ordered(X2[[nm]]))
      X2[[nm]] <- collapse_rare(X2[[nm]], min_prop)
  X2
})

jitter_discrete <- function(X, eps = 1e-6){
  # Add a *tiny* random noise to numerics with very few unique values.
  # This breaks distance ties without changing substantive values.
  X2 <- X
  for (nm in names(X2)){
    v <- X2[[nm]]
    if (is.factor(v)) next
    vn <- suppressWarnings(as.numeric(v))
    if (!is.numeric(vn)) next
    u <- sort(unique(na.omit(vn)))
    if (length(u) <= 12L) {
      rng <- max(u) - min(u); if (!is.finite(rng) || rng == 0) rng <- 1
      X2[[nm]] <- vn + rnorm(length(vn), sd = eps * rng)
    }
  }
  X2
}

# ---- helpers used below ----
is_binary <- function(v){
  u <- sort(unique(na.omit(as.numeric(v))))
  length(u) == 2 && all(u %in% c(0,1))
}

prep_X_for_gower <- function(X, rare_prop = 0.01, do_jitter = TRUE){
  X1 <- X
  
  # coerce chars → factors; keep ordered factors; keep numerics
  for(nm in names(X1)){
    v <- X1[[nm]]
    if (is.character(v)) X1[[nm]] <- factor(v)
  }
  
  # drop ultra-rare levels in unordered factors
  drop_rare <- function(f, prop){
    if (!is.factor(f) || is.ordered(f)) return(f)
    tb <- prop.table(table(f))
    keep <- names(tb)[tb >= prop]
    f <- factor(ifelse(f %in% keep, as.character(f), NA), exclude = NULL)
    droplevels(f)
  }
  X1 <- as.data.frame(lapply(X1, drop_rare, prop = rare_prop), stringsAsFactors = FALSE)
  
  # light jitter for pure numerics (optional)
  if (isTRUE(do_jitter)) {
    for(nm in names(X1)){
      if (is.numeric(X1[[nm]])) {
        sdv <- stats::sd(X1[[nm]], na.rm = TRUE)
        if (is.finite(sdv) && sdv > 0) X1[[nm]] <- X1[[nm]] + rnorm(length(X1[[nm]]), 0, 1e-6 * sdv)
      }
    }
  }
  
  # build daisy() 'type' list from current columns
  num_cols  <- names(X1)[vapply(X1, is.numeric,  logical(1))]
  ord_cols  <- names(X1)[vapply(X1, is.ordered, logical(1))]
  fac_cols  <- names(X1)[vapply(X1, function(z) is.factor(z) && !is.ordered(z), logical(1))]
  bin_cols  <- fac_cols[vapply(X1[fac_cols], is_binary, logical(1))]
  
  # 'type' only needs non-empty entries
  type_list <- list()
  if (length(bin_cols)) type_list$asymm   <- bin_cols    # treat 0/1 as asymmetric
  if (length(ord_cols)) type_list$ordratio<- ord_cols    # ordered factors
  
  # weights that MATCH columns exactly (named)
  w <- rep(1, ncol(X1)); names(w) <- names(X1)
  
  list(X = X1, type = type_list, weights = w)
}

gower_dist <- function(Xdf, type_list = NULL, weights = NULL){
  # prune type list to existing columns
  if (!is.null(type_list)) {
    type_list <- lapply(type_list, function(cols) intersect(cols, names(Xdf)))
    type_list <- type_list[lengths(type_list) > 0]
    if (!length(type_list)) type_list <- NULL
  }
  # reconcile weights
  if (is.null(weights)) {
    weights <- rep(1, ncol(Xdf))
  } else if (length(weights) == 1) {
    weights <- rep(weights, ncol(Xdf))
  } else if (!is.null(names(weights))) {
    # reorder & drop to match Xdf
    weights <- weights[names(Xdf)]
  }
  stopifnot(length(weights) == ncol(Xdf))
  
  cluster::daisy(Xdf, metric = "gower", type = type_list, weights = weights)
}

# ==================== 3) Geometry: cores, ID, trust/continuity =================
# "Core band": we pick points whose k-th neighbour distances sit in a middle band.
# This avoids extremely dense (duplicates) or extremely sparse (outliers) areas.
core_band_idx <- get0("core_band_idx", ifnotfound = function(D, k = 10, band = c(0.15,0.85)){
  M <- as.matrix(D); diag(M) <- Inf
  kth <- function(r, k) { rf <- r[is.finite(r)]; if (!length(rf)) return(NA_real_)
  k_eff <- min(k, length(rf)); sort(rf, partial = k_eff)[k_eff] }
  rk <- apply(M, 1, kth, k = k)
  ok <- is.finite(rk); if (!any(ok)) return(integer(0))
  q  <- stats::quantile(rk[ok], band, na.rm = TRUE)
  which(ok & rk >= q[1] & rk <= q[2])
})

# TwoNN intrinsic dimension (ID): uses the ratio of distances to 1st and 2nd NN.
# Lower ID ⇒ "flatter"/simpler manifold. We minimise this during weight tuning.
twonn_id_from_dist <- get0("twonn_id_from_dist", ifnotfound = function(D, eps=1e-8, trim=0.02){
  M <- as.matrix(D); n <- nrow(M); if (n < 3) return(NA_real_); diag(M) <- Inf
  r <- vapply(seq_len(n), function(i){
    di <- M[i, ]; di <- di[is.finite(di)]
    if (length(di) < 2) return(NA_real_)
    ds <- sort(di, partial = 2)[1:2]; d1 <- max(ds[1], eps); d2 <- max(ds[2], d1 + eps); d2/d1
  }, numeric(1))
  r <- r[is.finite(r) & r > 1]
  if (!length(r)) return(NA_real_)
  logr <- sort(log(r)); k <- floor(trim * length(logr))
  if (k > 0 && 2*k < length(logr)) logr <- logr[(k+1):(length(logr)-k)]
  1/mean(logr)
})

twonn_core_by_slope <- function(D, min_frac = 0.30, w = 20,
                                slope_tol = 0.08, rmse_tol = 0.10) {
  M <- as.matrix(D); n <- nrow(M); diag(M) <- Inf
  if (n < 8) return(seq_len(n))
  # r1, r2, mu
  r1 <- apply(M, 1L, function(r) sort(r)[1])
  r2 <- apply(M, 1L, function(r) sort(r)[2])
  mu <- pmax(r2 / pmax(r1, .Machine$double.eps), 1 + 1e-12)
  
  ord <- order(mu); x <- log(mu[ord]); m <- length(x)
  u   <- (seq_len(m) - 0.5) / (m + 1)      # avoid 0/1
  y   <- log(1 - u)
  
  k0 <- max(20L, floor(min_frac * m))
  slope <- rep(NA_real_, m); rmse <- rep(NA_real_, m)
  for (k in k0:(m - 2L)) {
    fit <- stats::lm(y[1:k] ~ x[1:k])
    slope[k] <- coef(fit)[2]
    rmse[k]  <- sqrt(mean(residuals(fit)^2))
  }
  ok <- which(is.finite(slope) & is.finite(rmse))
  if (!length(ok)) return(ord[seq_len(max(3L, k0))])
  
  # stability: last 'w' slopes must lie within a band of width 'slope_tol'
  pick <- function(k) {
    L <- max(k0, k - w + 1); s <- slope[L:k]
    (max(s, na.rm = TRUE) - min(s, na.rm = TRUE) <= slope_tol) && (rmse[k] <= rmse_tol)
  }
  ks <- ok[vapply(ok, pick, logical(1))]
  k_star <- if (length(ks)) max(ks) else floor(0.6 * m)
  
  ord[seq_len(max(3L, min(k_star, m - 2L)))]
}

# Levina–Bickel MLE ID: a more classical k-NN slope-based estimate.
lb_mle_id <- get0("lb_mle_id", ifnotfound = function(Dm, k_lo=5, k_hi=15){
  Dm <- as.matrix(Dm); n <- nrow(Dm); diag(Dm) <- Inf
  if (n <= k_lo) return(NA_real_)
  k_hi <- max(k_lo, min(k_hi, n-1))
  ids <- sapply(k_lo:k_hi, function(k){
    nn <- t(apply(Dm, 1L, function(r){
      rf <- r[is.finite(r)]; m <- length(rf); if (m < k) return(rep(NA_real_, k))
      sort(rf, partial = k)[1:k]
    }))
    if (!nrow(nn)) return(NA_real_)
    l  <- log(nn[,k,drop=TRUE]/nn[,1:(k-1),drop=FALSE])
    d  <- 1/rowMeans(l, na.rm=TRUE)
    mean(d[is.finite(d)], na.rm=TRUE)
  })
  mean(ids, na.rm=TRUE)
})

# Switch for which ID to compute: across all points vs the "core" only.
compute_id_objective <- function(D, objective="TwoNN_core", k_core=CORE_KNN_K, band=CORE_BAND,
                                 core_idx_override=NULL){
  if (!is.null(core_idx_override) && length(core_idx_override) >= 3) {
    M <- as.matrix(D); diag(M) <- Inf
    return(if (objective == "LB_core")
      lb_mle_id(M[core_idx_override, core_idx_override, drop = FALSE], 5, 15)
      else
        twonn_id_from_dist(as.dist(M[core_idx_override, core_idx_override, drop = FALSE])))
  }
  Dm <- as.matrix(D); diag(Dm) <- Inf
  if (objective == "TwoNN_all") return(twonn_id_from_dist(D))
  core <- core_band_idx(D, k=k_core, band=band)
  if (length(core) < 20) return(NA_real_)
  Dcore <- stats::as.dist(Dm[core, core, drop=FALSE])
  if (objective == "LB_core") lb_mle_id(as.matrix(Dcore), 5, 15) else twonn_id_from_dist(Dcore)
}

# Trust & Continuity (TC): "does the low-d embedding preserve local neighbours?"
# Trust: neighbours in low-d also appear in high-d. Continuity: converse.
trust_cont <- function(high, low, ks=10:30){
  high <- as.matrix(high); low <- as.matrix(low); stopifnot(nrow(high)==nrow(low))
  n <- nrow(high)
  Dh <- as.matrix(stats::dist(high)); diag(Dh) <- Inf
  Dl <- as.matrix(stats::dist(low));  diag(Dl) <- Inf
  rf <- function(D){ R <- matrix(0L, n, n); for (i in 1:n){ r <- D[i,]; ord <- order(r); R[i, ord] <- seq_len(n) } ; R }
  Rh <- rf(Dh); Rl <- rf(Dl)
  res <- lapply(ks, function(k){
    H <- t(apply(Rh, 1, function(r) order(r)[1:k]))
    L <- t(apply(Rl, 1, function(r) order(r)[1:k]))
    Tsum <- 0; Csum <- 0
    for (i in 1:n){
      U <- setdiff(L[i,], H[i,]); if (length(U)) Tsum <- Tsum + sum(pmax(Rh[i, U]-k, 0))
      V <- setdiff(H[i,], L[i,]); if (length(V)) Csum <- Csum + sum(pmax(Rl[i, V]-k, 0))
    }
    denom <- n*k*(2*n - 3*k - 1)
    data.frame(k=k, Trust=1 - (2/denom)*Tsum, Continuity=1 - (2/denom)*Csum)
  })
  do.call(rbind, res)
}
trust_cont_avg <- function(high, low, ks=10:30){
  tc <- trust_cont(high, low, ks); c(T=mean(tc$Trust), C=mean(tc$Continuity))
}

# ================= 4) Survivor selection from weight curve (knee) ==============
# We sort weights descending. The "knee" (elbow) separates the high, flat
# region (likely useful variables) from the decaying tail (likely dispensable).
# Geometry: compute each point's distance to the straight line joining the
# first and last points in the (rank, weight) curve; the maximum is the knee.

.knee_triangle <- function(w) {
  # Input: numeric vector of weights (unnamed OK). We internally sort ↓.
  y <- sort(as.numeric(w), decreasing = TRUE)
  n <- length(y)
  
  # With <3 points, a "knee" isn't meaningful. Return the last weight if any.
  if (n < 3L) return(list(
    k     = n,                             # knee index in the sorted curve
    thr   = if (n) y[n] else NA_real_,     # knee threshold (weight)
    curve = data.frame(i = seq_len(n), w = y, d = rep(0, n))
  ))
  
  # x-axis = ranks (1..n), y-axis = weights, both in descending order.
  x <- seq_len(n)
  
  # Distance from each point (x_i, y_i) to the chord between (1, y1) and (n, yn).
  # Classic "triangle method" elbow detector.
  num <- abs((y[n] - y[1]) * x - (n - 1) * y + n * y[1] - y[n])
  den <- sqrt((y[n] - y[1])^2 + (n - 1)^2)
  d   <- num / den
  
  k <- which.max(d)  # position of maximum perpendicular distance (the knee)
  list(k = k, thr = y[k], curve = data.frame(i = x, w = y, d = d))
}

.survivors_from_weights <- function(
    w,
    w_min     = W_MIN,            # hard floor to avoid literal zeros
    kmin      = NULL,             # minimum survivors (absolute count)
    eps_ceil  = 1e-4,             # anything within this of 1.0 is “ceiling”
    eps_floor = 1e-12,            # guard for numeric underflow near the floor
    make_plot = TRUE,             # quick visual sanity check
    plot_file = "FIG_weight_curve_knee.png") {
  
  # 1) Ensure numeric, clamp to the minimum, and keep names (if given)
  if (is.null(names(w)) && exists("X") && !is.null(colnames(X)))
    names(w) <- colnames(X)  # nice-to-have for reporting
  w[] <- pmax(w_min, as.numeric(w))
  p <- length(w)
  
  # 2) Quick distribution summary — helps eyeball weird outputs
  cat(sprintf("[weights] p=%d | min=%.4f  q25=%.4f  med=%.4f  q75=%.4f  max=%.4f\n",
              p, min(w), as.numeric(quantile(w, .25)), median(w),
              as.numeric(quantile(w, .75)), max(w)))
  
  # 3) Separate the "ceiling" mass (≈1.0) and the tail (strictly between floor and ceiling)
  idx_ceil <- which(w >= 1 - eps_ceil)
  idx_tail <- which(w <  1 - eps_ceil & w > w_min + eps_floor)
  
  # Knee on the tail only — we do NOT penalise variables that hit the ceiling.
  thr_tail <- NA_real_
  knee_obj <- NULL
  if (length(idx_tail) >= 3L) {
    knee_obj <- .knee_triangle(w[idx_tail])
    thr_tail <- knee_obj$thr
  } else if (length(idx_tail) > 0L) {
    # Too few points for a geometric knee: use a conservative median
    thr_tail <- max(w_min + 1e-6, median(w[idx_tail]))
  }
  
  # 4) Survivors = all ceiling vars + tail vars above the tail knee
  S_ceil <- names(w)[idx_ceil]
  S_tail <- if (is.finite(thr_tail)) names(w)[w >= thr_tail & w < 1 - eps_ceil] else character(0)
  survivors <- union(S_ceil, S_tail)
  
  # 5) Guardrail: enforce a minimum number of survivors (default ~10% of p, ≥30)
  if (is.null(kmin)) kmin <- max(30L, ceiling(0.10 * p))
  if (length(survivors) < kmin) {
    survivors <- names(sort(w, decreasing = TRUE))[seq_len(kmin)]
  }
  
  cat(sprintf("Ceiling kept: %d | tail knee thr=%s | survivors: %d / %d\n",
              length(S_ceil),
              ifelse(is.finite(thr_tail), sprintf("%.3f", thr_tail), "NA"),
              length(survivors), p))
  
  # 6) Optional plot of the weight curve and knee mark (sorted ↓)
  if (isTRUE(make_plot) && requireNamespace("ggplot2", quietly = TRUE)) {
    ord <- order(w, decreasing = TRUE)
    curve <- data.frame(i = seq_along(ord), w = as.numeric(w[ord]))
    pplt <- ggplot2::ggplot(curve, ggplot2::aes(i, w)) +
      ggplot2::geom_line() +
      ggplot2::labs(
        x = "rank (sorted ↓)",
        y = "weight",
        title = "Weight curve with knee (tail-only)"
      ) +
      ggplot2::theme_minimal()
    
    if (!is.null(knee_obj)) {
      # Mark the knee by nearest y match (visual cue; sorting may collapse duplicates)
      thr_mark <- knee_obj$thr
      knee_row <- curve[which.min(abs(curve$w - thr_mark)), , drop = FALSE]
      pplt <- pplt + ggplot2::geom_point(data = knee_row, size = 2)
    }
    print(pplt)
    try(ggplot2::ggsave(plot_file, pplt, width = 6, height = 4, dpi = 150), silent = TRUE)
  }
  
  # 7) Return survivors, tail threshold (informative), and the clamped weights
  list(
    survivors = survivors,
    thr_tail  = thr_tail,
    w_clamped = w
  )
}

# =========================== 5) Data ingest & targets ==========================
# Goal: load psychometric data (X) and diagnoses (diag_wide_full), align IDs,
# coerce item types sensibly, and build a binary target y_use.
# If PREF_TARGET is degenerate (all 0 or all 1), we flip to its complement.

# ---- 5.1 Load psychometric matrix (once) -------------------------------------
if (!exists("X") || !nrow(X)) {
  df <- readr::read_delim(
    "psychometric_matrix.csv",
    delim = ";",
    locale = readr::locale(decimal_mark = "."),
    progress = FALSE
  )
  
  # Pick ID column: prefer explicit "participant_id", otherwise the first column.
  id_col  <- if ("participant_id" %in% names(df)) "participant_id" else names(df)[1]
  ids_all <- as.character(df[[id_col]])
  
  # Items = everything except ID and any "diagnosis*" columns the file might contain.
  X <- dplyr::select(df, -dplyr::all_of(c(id_col, grep("^diagnosis", names(df), value = TRUE))))
  
  # ---- 5.2 Coerce item types (simple & robust rules) -------------------------
  # Why these rules?
  #  - Binary {0,1} → ordered factor (categorical, asymmetric for Gower)
  #  - Small integer scales (3..7 levels) → ordered factor (Likert-ish)
  #  - Other numeric → keep numeric
  #  - Everything else → unordered factor (safe fallback)
  is_small_int_scale <- function(v) {
    vn <- suppressWarnings(as.numeric(v))
    if (all(is.na(vn))) return(FALSE)
    u <- sort(unique(na.omit(vn)))
    k <- length(u)
    k >= 3 && k <= 7 && all(abs(u - round(u)) < 1e-8)
  }
  
  for (nm in names(X)) {
    v <- X[[nm]]
    
    # Try to parse characters as numeric when it’s safe (e.g., "1", "2", "3")
    if (is.character(v)) {
      vn <- suppressWarnings(as.numeric(v))
      if (!all(is.na(vn))) v <- vn
    }
    
    if (all(v %in% c(0, 1, NA))) {
      X[[nm]] <- factor(v, levels = c(0, 1), ordered = TRUE)
    } else if (is_small_int_scale(v)) {
      X[[nm]] <- factor(as.integer(round(as.numeric(v))), ordered = TRUE)
    } else if (is.numeric(v)) {
      X[[nm]] <- as.numeric(v)
    } else {
      X[[nm]] <- factor(v)
    }
  }
  
  # Drop rows with any missing item values (keeps geometry/OOF code simpler).
  keep <- stats::complete.cases(X)
  X    <- X[keep, , drop = FALSE]
  rownames(X) <- make.unique(ids_all[keep])
}

# ---- Typed-Gower: build a single, global type list (ordratio only; NO asymm) ----
make_type_list <- function(X, use_asymm = FALSE, asymm_whitelist = NULL) {
  ordr <- names(X)[vapply(X, is.ordered, logical(1))]
  tl <- list()
  if (length(ordr) >= 2) tl$ordratio <- match(ordr, names(X))
  if (use_asymm) {
    asym <- if (is.null(asymm_whitelist))
      names(X)[vapply(X, function(z) is.factor(z) && nlevels(z) == 2, logical(1))]
    else
      intersect(asymm_whitelist, names(X))
    if (length(asym)) tl$asymm <- match(asym, names(X))
  }
  tl
}
TYPE_LIST <- make_type_list(X, use_asymm = FALSE)  # <- ordratio only

# ---- 5.3 Load diagnoses (long → wide) ----------------------------------------
if (!exists("diag_wide_full")) {
  diag_long_raw <- readr::read_delim(
    "long_diagnoses.csv",
    delim = ";",
    col_types = readr::cols(),
    progress = FALSE
  )
  # Accept either a "diagnosis" column or a generic column named "Data".
  dx_col <- if ("diagnosis" %in% names(diag_long_raw)) "diagnosis" else "Data"
  
  diag_wide_full <- diag_long_raw |>
    dplyr::transmute(
      participant_id = as.character(.data[["participant_id"]]),
      dx             = as.character(.data[[dx_col]])
    ) |>
    # Normalise "No Diagnosis" variants to a single token
    dplyr::mutate(dx = ifelse(grepl("^\\s*No\\s+Diagnosis\\b", dx, ignore.case = TRUE),
                              "NODIAG", dx)) |>
    dplyr::mutate(present = 1L) |>
    dplyr::distinct(participant_id, dx, .keep_all = TRUE) |>
    tidyr::pivot_wider(
      names_from  = dx,
      values_from = present,
      values_fill = 0L
    )
}

# ---- 5.4 Align IDs and build the target y_use --------------------------------
ids_here <- rownames(X)
mm <- match(ids_here, as.character(diag_wide_full$participant_id))
if (anyNA(mm)) {
  stop(sprintf("Diagnoses join failed for %d/%d rows.", sum(is.na(mm)), length(mm)))
}

dx_cols_all <- setdiff(names(diag_wide_full), "participant_id")
# We will exclude "No Diagnosis" signals from the “any diagnosis” pool.
drop_nodiag <- grepl("^no\\s*diagnosis|^NODIAG$", dx_cols_all, ignore.case = TRUE)
dx_cols <- dx_cols_all[!drop_nodiag]

dx_mat <- if (length(dx_cols)) {
  M <- as.matrix(diag_wide_full[mm, dx_cols, drop = FALSE]); M[is.na(M)] <- 0L
  keepc <- colSums(M, na.rm = TRUE) > 0
  if (!any(keepc)) NULL else M[, keepc, drop = FALSE]
} else NULL

any_dx <- if (is.null(dx_mat)) rep.int(0L, length(ids_here)) else as.integer(rowSums(dx_mat) > 0)

make_y <- function(pref, any_dx) if (toupper(pref) == "NODIAG") 1L - any_dx else any_dx
pref        <- toupper(PREF_TARGET)
y_use       <- make_y(pref, any_dx)
n0 <- sum(y_use == 0); n1 <- sum(y_use == 1)

# If the requested target is degenerate, flip to the other target and inform.
if (n0 == 0 || n1 == 0) {
  message(sprintf("[Target %s] degenerate (n0=%d, n1=%d) — switching.", pref, n0, n1))
  pref_alt <- if (pref == "NODIAG") "ANY_DX" else "NODIAG"
  y_use    <- make_y(pref_alt, any_dx)
  n0 <- sum(y_use == 0); n1 <- sum(y_use == 1)
  if (n0 == 0 || n1 == 0) {
    tbl <- if (is.null(dx_mat)) "dx_mat=NULL (no dx cols after filtering)"
    else paste(capture.output(print(colSums(dx_mat))), collapse = "\n")
    stop(sprintf("Still degenerate after switching to %s (n0=%d, n1=%d).\n%s",
                 pref_alt, n0, n1, tbl))
  }
  PREF_TARGET <- pref_alt  # reflect the automatic fix
}

cat(sprintf("[Target %s] n=%d | n0=%d | n1=%d\n", toupper(PREF_TARGET), length(y_use), n0, n1))

# =================== 6) ε-dedup (collapse) + core selection ====================
# Why: nearest-neighbour geometry is hypersensitive to near-duplicates.
# We collapse very close subjects into one representative ("medoid"), using
# a distance threshold ε. If EPS_DEDUP is NA, we auto-pick ε via a "collapse curve".

# --- Helpers (defined if missing) ---------------------------------------------
first_nn_d1 <- get0("first_nn_d1", ifnotfound = function(D){
  Dm <- as.matrix(D); diag(Dm) <- Inf
  apply(Dm, 1L, function(r){ r <- r[is.finite(r)]; if (!length(r)) Inf else min(r) })
})

collapse_curve <- get0("collapse_curve", ifnotfound = function(D, eps_grid){
  n0 <- attr(D, "Size")
  data.frame(
    eps = eps_grid,
    n_groups = sapply(eps_grid, function(eps) length(unique(hclust(D, method = "complete") |> cutree(h = eps)))),
    prop_retained = NA_real_
  ) |>
    transform(prop_retained = n_groups / n0)
})

complete_groups <- get0("complete_groups", ifnotfound = function(D, eps){
  hclust(D, method = "complete") |> cutree(h = eps)
})

group_medoids <- get0("group_medoids", ifnotfound = function(D, groups){
  Dm <- as.matrix(D); diag(Dm) <- 0
  split_idx <- split(seq_len(nrow(Dm)), groups)
  reps <- vapply(split_idx, function(ix){ ix[ which.min(rowSums(Dm[ix, ix, drop = FALSE])) ] }, integer(1))
  list(reps = unname(reps), mult = as.integer(lengths(split_idx)))
})

# --- 6.1 Compute Gower distances on a jittered, typed view --------------------
px  <- prep_X_for_gower(X, rare_prop = RARE_LEVEL_MIN_PROP, do_jitter = TRUE)
X_for_id <- px$X
Dg <- gower_dist(X_for_id, type_list = px$type, weights = px$weights)

# --- 6.2 Pick ε (EPS_DEDUP) if not provided -----------------------------------
if (is.na(EPS_DEDUP)) {
  # Strategy: scan ε over a small grid and look where the #groups drops fastest.
  d1 <- first_nn_d1(Dg)
  # Start the grid slightly above the extreme tiny distances; stop before the
  # 30th percentile to avoid over-merging.
  qlo <- as.numeric(stats::quantile(d1[is.finite(d1)], probs = c(0.001, 0.01), na.rm = TRUE))
  eps_grid <- seq(
    from = max(0, min(qlo, na.rm = TRUE) - 0.02),
    to   = min(0.50, stats::quantile(d1, 0.30, na.rm = TRUE)),
    by   = 0.005
  )
  cc <- collapse_curve(Dg, eps_grid)
  # Slope of retained proportion vs ε; the steepest negative slope is the knee.
  dprop <- diff(cc$prop_retained) / diff(cc$eps)
  knee_i <- which.min(dprop)
  EPS_DEDUP <- mean(cc$eps[c(knee_i, knee_i + 1)])
}

# --- 6.3 Collapse near-duplicates (complete-linkage at ε) ---------------------
gr_all  <- if (isTRUE(DO_DEDUP)) complete_groups(Dg, EPS_DEDUP) else seq_len(attr(Dg, "Size"))
med_all <- group_medoids(Dg, gr_all)
reps    <- med_all$reps      # row indices (in X) of each group's representative
mult    <- med_all$mult      # cluster sizes (how many subjects were collapsed)

# --- 6.4 Core selection on representatives ------------------------------------
# We re-compute distances among reps and choose a "middle-density band" of points
# for more stable intrinsic-dimension estimates (avoids extremes).

plot_twonn <- function(D, core_idx = NULL,
                       trim_x = c(0.05, 0.80),
                       file = "FIG_twonn_slope_core_vs_all_preoptim.png") {
  M <- as.matrix(D); n <- nrow(M); diag(M) <- Inf
  if (n < 8) stop("Too few points.")
  # r1, r2, mu for ALL
  r1 <- apply(M, 1L, function(r) sort(r)[1L])
  r2 <- apply(M, 1L, function(r) sort(r)[2L])
  mu_all <- pmax(r2 / pmax(r1, .Machine$double.eps), 1 + 1e-12)
  
  # Global ECDF on ALL mu (never 0/1 by using (rank-0.5)/n)
  ord_all <- order(mu_all)
  ranks   <- seq_along(mu_all) - 0.5
  F_all   <- numeric(length(mu_all)); F_all[ord_all] <- ranks / length(mu_all)
  
  # Coords for ALL (use global ECDF)
  x_all <- log(mu_all)
  y_all <- log(pmax(1 - F_all, 1e-12))
  
  # Trim window set from ALL only (by x quantiles)
  qx <- quantile(x_all, probs = trim_x, na.rm = TRUE)
  keep_all  <- which(is.finite(x_all) & is.finite(y_all) & x_all >= qx[1] & x_all <= qx[2])
  
  # Fit on ALL (ordinary least squares)
  fit_all <- lm(y_all[keep_all] ~ x_all[keep_all])
  d_all   <- -coef(fit_all)[2]
  rmse_all <- sqrt(mean(residuals(fit_all)^2))
  
  # Core subset uses the SAME ECDF and SAME x-window
  core_idx <- if (is.null(core_idx)) seq_len(n) else as.integer(core_idx)
  keep_core <- intersect(core_idx, keep_all)
  fit_core <- lm(y_all[keep_core] ~ x_all[keep_core])
  d_core   <- -coef(fit_core)[2]
  rmse_core <- sqrt(mean(residuals(fit_core)^2))
  
  # ---- Plot
  png(file, 900, 700, res = 150)
  par(mar = c(5,5,3,2), cex.axis=1.1, cex.lab=1.25)
  plot(x_all, y_all, pch = 16, col = "#3182bd", cex = 0.55,
       xlab = "log(mu = r2/r1)", ylab = "log(1 - ECDF(mu))",
       main = "TwoNN slope (pre-optimiser): All vs Core")
  points(x_all[setdiff(seq_len(n), core_idx)], y_all[setdiff(seq_len(n), core_idx)],
         col = adjustcolor("#3182bd", 0.35), pch = 16, cex = 0.45)
  abline(fit_all, lwd = 2, col = "black")
  abline(fit_core, lwd = 2, lty = 2, col = "#3182bd")
  legend("topright", bty = "n",
         legend = c(sprintf("All:   d^ = %.2f, RMSE = %.2f", d_all,  rmse_all),
                    sprintf("Core: d^ = %.2f, RMSE = %.2f", d_core, rmse_core)))
  dev.off()
  
  invisible(list(d_all = d_all, rmse_all = rmse_all,
                 d_core = d_core, rmse_core = rmse_core,
                 keep_all = keep_all, keep_core = keep_core))
}

Dm_g   <- as.matrix(Dg); diag(Dm_g) <- Inf
Dg_rep <- stats::as.dist(Dm_g[reps, reps, drop = FALSE])

# choose the core (either band or the slope-stability core if you added it)
# core_idx_rep <- core_band_idx(Dg_rep, k = CORE_KNN_K, band = CORE_BAND)
core_idx_rep <- if (exists("twonn_core_by_slope")) twonn_core_by_slope(Dg_rep) else
  core_band_idx(Dg_rep, k = CORE_KNN_K, band = CORE_BAND)

cat(sprintf("[Dedup] eps=%.3f | reps=%d of %d | core_rep=%d\n",
            EPS_DEDUP, length(reps), nrow(X), length(core_idx_rep)))

res <- plot_twonn(Dg_rep, core_idx = core_idx_rep,
                  trim_x = c(0.05, 0.80),
                  file = "FIG_twonn_slope_core_vs_all_preoptim.png")
res$d_all; res$d_core

# --- 6.5 Optional: write the dedup map to CSV ---------------------------------
if (isTRUE(WRITE_DEDUP_CSV)) {
  mult_df <- data.frame(
    rep_row          = reps,
    representative_id= rownames(Dm_g)[reps],
    multiplicity     = mult
  )
  readr::write_csv(mult_df, sprintf("near_duplicate_groups_complete_eps%g.csv", EPS_DEDUP))
}

# ================= 7) Constrained Gower-weight optimisation ====================
# Idea in plain words:
#   - Each variable gets a non-negative Gower weight (start at 1).
#   - We try small decays to individual weights and keep any change that
#     reduces intrinsic dimension (ID) on a *stable core* of points.
#   - We freeze rows to the set of ε-dedup representatives (speed + stability).
#   - We optionally subsample rows for speed; variables not in X_pred are frozen.
#
# Why intrinsic dimension (ID)?
#   - Lower ID ≈ simpler, flatter manifold. We prefer geometries that preserve
#     coherent structure with fewer “twists”. TwoNN_core is our default score.

# ---- 7.1 Fast per-variable Gower pieces (cache) -------------------------------
# We precompute, for each variable j, its contribution to the condensed distance:
#   Dist = (Σ_j w_j * N_j) / (Σ_j w_j * S_j)
# where N_j is the per-pair dissimilarity (0..1) and S_j is availability mask.
.make_NS_cache <- function(Xdf, type = NULL){
  p  <- ncol(Xdf)
  N_list <- vector("list", p)
  S_list <- vector("list", p)
  for (j in seq_len(p)) {
    w1 <- rep(0, p); w1[j] <- 1
    Dj <- cluster::daisy(Xdf, metric = "gower", type = type, weights = w1)
    vD <- as.numeric(Dj)
    ok <- as.numeric(is.finite(vD))
    vD[!is.finite(vD)] <- 0
    N_list[[j]] <- vD
    S_list[[j]] <- ok
  }
  list(N = N_list, S = S_list, n = nrow(Xdf))
}

# ---- 7.2 The optimiser: greedy decays with small "batch" steps ----------------
optimise_gower_weights_constrained <- function(
    X,                      # full item matrix (data.frame)
    init_weights,           # named numeric vector (colnames(X)); start at 1
    allow_update,           # logical or names: which variables may change
    objective    = "TwoNN_core",  # ID objective ("TwoNN_core" default)
    w_min        = W_MIN,
    step_grid    = W_STEP_GRID,   # candidate multiplicative decays for a var
    batch_k      = W_BATCH_K,     # after best single change, apply (k-1) extra decays
    batch_factor = W_BATCH_FACTOR,# multiplicative factor for extra decays
    max_iter     = W_MAX_ITERS,   # NA ⇒ safe internal cap (≈ 3p)
    n_rows_sub   = N_ROWS_SUB,    # optional row subsample for speed
    ncores       = NCORES_PAR,    # parallel chunks across candidates
    seed_jitter  = SEED_JITTER,   # RNG for tiny numeric jitter
    reps_idx     = NULL,          # row indices for ε-dedup representatives
    core_idx_rep = NULL,          # indices *within reps* for the core
    verbose      = TRUE,
    plot_progress = TRUE){
  
  # Prevent math libraries from spawning inner threads during this loop
  restore_env <- Sys.getenv(c("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"))
  Sys.setenv(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
  on.exit({ do.call(Sys.setenv, as.list(stats::setNames(as.list(restore_env), names(restore_env)))) }, add=TRUE)
  
  set.seed(seed_jitter)
  
  # ---- Align rows: use reps (or all rows), optionally subsample for speed ----
  px  <- prep_X_for_gower(X, rare_prop = RARE_LEVEL_MIN_PROP, do_jitter = TRUE)
  X0  <- px$X              # data.frame
  typ <- px$type           # daisy() type list (may be NULL)
  wgt <- px$weights        # named weights (length = ncol(X0))
  
  row_pool <- if (!is.null(reps_idx)) reps_idx else seq_len(nrow(X0))
  ix_sub_from_reps <- if (!is.null(n_rows_sub) && length(row_pool) > n_rows_sub) {
    if (isTRUE(FIX_REP_SUBSET)) head(row_pool, n_rows_sub) else sample(row_pool, n_rows_sub)
  } else row_pool
  
  Xs   <- X0[ix_sub_from_reps, , drop = FALSE]
  vars <- colnames(Xs); p <- ncol(Xs)
  if (p < 1L) stop("[constrained] Xs has 0 columns after preprocessing")
  
  # ---- Align weights and update mask to Xs columns ---------------------------
  w <- init_weights
  if (length(w) == 1L) w <- rep(w, ncol(X))
  if (is.null(names(w))) names(w) <- colnames(X)
  if (!all(vars %in% names(w))) stop("[constrained] init_weights missing some columns found in Xs")
  w <- as.numeric(w[vars]); names(w) <- vars
  
  if (is.logical(allow_update) && length(allow_update) == p && is.null(names(allow_update))) {
    names(allow_update) <- vars
  } else if (is.character(allow_update)) {
    tmp <- rep(FALSE, p); names(tmp) <- vars
    tmp[intersect(allow_update, vars)] <- TRUE
    allow_update <- tmp
  } else if (is.logical(allow_update) && length(allow_update) == ncol(X)) {
    names(allow_update) <- colnames(X); allow_update <- allow_update[vars]
  } else if (!is.null(names(allow_update))) {
    allow_update <- allow_update[vars]
  } else stop("[constrained] allow_update must be named or aligned to colnames(X)")
  stopifnot(length(allow_update) == p)
  w[!allow_update] <- pmax(w_min, w[!allow_update])
  
  # ---- Cache variable-wise condensed pieces; build current numerator/denom ----
  cache <- .make_NS_cache(Xs, type = typ)
  eps     <- .Machine$double.eps
  num_cur <- Reduce(`+`, Map(`*`, cache$N, as.list(w)))
  den_cur <- Reduce(`+`, Map(`*`, cache$S, as.list(w)))
  
  # ---- Which pairs define the ID objective?  ---------------------------------
  # If objective == "TwoNN_all" we evaluate ID on ALL pairs of Xs;
  # otherwise we evaluate on a core subset mapped from reps.
  .pair_ix_core <- function(core_idx, n_total){
    core_idx <- sort(as.integer(core_idx))
    m <- length(core_idx)
    if (m < 2L) return(list(ix = integer(0L), m = m))
    ix  <- integer(m * (m - 1L) / 2L)
    pos <- 1L
    # 'dist' order: for j = 2..n, i = 1..j-1 → index = (j-1)*(j-2)/2 + i
    for (s in 2:m) {
      j_abs <- core_idx[s]
      base  <- (j_abs - 1L) * (j_abs - 2L) / 2L
      for (r in 1:(s-1L)) { i_abs <- core_idx[r]; ix[pos] <- base + i_abs; pos <- pos + 1L }
    }
    list(ix = ix, m = m)
  }
  .pair_ix_all <- function(n_total){
    list(ix = seq_len(n_total * (n_total - 1L) / 2L), m = n_total)
  }
  
  if (identical(objective, "TwoNN_all")) {
    # keep this for reporting/return value consistency
    core_idx_sub <- seq_len(nrow(Xs))
    pair_map     <- .pair_ix_all(nrow(Xs))
  } else {
    # Map the “core on reps” into our current subsample
    if (!is.null(reps_idx) && !is.null(core_idx_rep) && length(core_idx_rep) >= 3L) {
      used_reps    <- ix_sub_from_reps                   # rows currently used
      abs_core     <- reps_idx[core_idx_rep]             # absolute row ids
      core_idx_sub <- which(used_reps %in% abs_core)     # which used rows are core
      if (length(core_idx_sub) < 20L) core_idx_sub <- seq_len(nrow(Xs))
    } else {
      core_idx_sub <- seq_len(nrow(Xs))
    }
    pair_map <- .pair_ix_core(core_idx_sub, nrow(Xs))
  }
  
  if (verbose) cat(sprintf("[constrained] p=%d | allowed=%d | eval_rows=%d (%s)\n",
                           p, sum(allow_update), length(core_idx_sub),
                           if (identical(objective, "TwoNN_all")) "ALL pairs" else "core pairs"))
  
  # ---- From (num, den) → ID on the selected pairs ----------------------------
  id_from_numden <- (function(ix_pairs, m_pairs, objective_local = objective) {
    function(num, den) {
      if (m_pairs < 3L || !length(ix_pairs)) return(NA_real_)
      d_sel <- num[ix_pairs] / pmax(den[ix_pairs], eps)
      Dsel  <- structure(d_sel, Size = m_pairs, Diag = FALSE, Upper = FALSE, class = "dist")
      if (identical(objective_local, "LB_core")) lb_mle_id(as.matrix(Dsel), 5, 15)
      else                                       twonn_id_from_dist(Dsel)
    }
  })(pair_map$ix, pair_map$m)
  
  id0  <- id_from_numden(num_cur, den_cur)
  hist <- data.frame(iter = 0L, ID = id0, changed = NA_character_, note = NA_character_)
  if (verbose) cat(sprintf("[constrained] iter 0: %s = %.3f\n", objective, id0))
  
  # Local “apply” that honours ncores without oversubscribing
  par_apply <- function(idx, FUN, chunk = NULL){
    if (is.null(chunk)) chunk <- max(1L, ceiling(length(idx) / (2L * max(1L, ncores))))
    chunks <- split(idx, ceiling(seq_along(idx) / chunk))
    if (ncores <= 1L || length(chunks) <= 1L) {
      return(unlist(lapply(chunks, function(ii) lapply(ii, FUN)), recursive = FALSE))
    }
    if (.Platform$OS.type != "windows") {
      return(unlist(parallel::mclapply(chunks, function(ii) lapply(ii, FUN),
                                       mc.cores = ncores, mc.preschedule = TRUE),
                    recursive = FALSE))
    } else {
      cl <- parallel::makeCluster(ncores, type = "PSOCK", outfile = "")
      on.exit(parallel::stopCluster(cl), add = TRUE)
      parallel::clusterEvalQ(cl, {
        Sys.setenv(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"); NULL
      })
      parallel::clusterEvalQ(cl, { suppressPackageStartupMessages(library(cluster)); NULL })
      parallel::clusterExport(cl, varlist = c("cache","id_from_numden","eps","CORE_KNN_K","CORE_BAND","objective"),
                              envir = environment())
      return(unlist(parallel::parLapply(cl, chunks, function(ii) lapply(ii, FUN)), recursive = FALSE))
    }
  }
  
  # Safe cap for iterations if max_iter is NA/NULL
  max_iter_eff <- if (is.null(max_iter) || !is.finite(max_iter)) 3L * p else as.integer(max(1L, max_iter))
  
  id <- id0
  for (it in seq_len(max_iter_eff)){
    # Variables we’re allowed to shrink and still above floor
    can <- which(allow_update & (w > w_min + 1e-12))
    if (!length(can)) { if (verbose) cat("[constrained] nothing to update.\n"); break }
    
    # Enumerate candidate decays for each variable (monotone: only smaller weights)
    cand <- do.call(rbind, lapply(can, function(j){
      w_try <- unique(pmax(w_min, w[j] * step_grid))
      w_try <- w_try[w_try < w[j] - 1e-12]
      if (!length(w_try)) return(NULL)
      data.frame(j=j, wj=w_try)
    }))
    if (is.null(cand) || !nrow(cand)) { if (verbose) cat("[constrained] no candidates.\n"); break }
    
    # Score each candidate quickly via our cached num/den pieces
    cand$id <- as.numeric(unlist(
      par_apply(seq_len(nrow(cand)), function(i){
        j  <- cand$j[i]; wj <- cand$wj[i]
        num <- num_cur + (wj - w[j]) * cache$N[[j]]
        den <- den_cur + (wj - w[j]) * cache$S[[j]]
        id_from_numden(num, den)
      }, chunk = 64L)))
    
    best_row <- which.min(cand$id); best_id <- cand$id[best_row]; changed <- FALSE
    if (is.finite(best_id) && best_id < id - 1e-6){
      jbest <- cand$j[best_row]; wbest <- cand$wj[best_row]
      num_cur <- num_cur + (wbest - w[jbest]) * cache$N[[jbest]]
      den_cur <- den_cur + (wbest - w[jbest]) * cache$S[[jbest]]
      w[jbest] <- wbest; id <- best_id; changed <- TRUE
      hist <- rbind(hist, data.frame(iter=it, ID=id, changed=vars[jbest], note=sprintf("line %.3f", wbest)))
      if (verbose) cat(sprintf("[constrained] iter %d: ↓ %s → w=%.3f | %s = %.3f\n",
                               it, vars[jbest], wbest, objective, id))
    } else {
      if (verbose) cat("[constrained] no improving single-variable move; stopping.\n")
      break
    }
    
    # Small batch: apply a few additional gentle decays that look promising
    if (batch_k > 1){
      remain <- setdiff(unique(cand$j), cand$j[best_row])
      if (length(remain)){
        gains <- as.numeric(unlist(par_apply(remain, function(j){
          wj <- max(w_min, w[j] * batch_factor)
          num <- num_cur + (wj - w[j]) * cache$N[[j]]
          den <- den_cur + (wj - w[j]) * cache$S[[j]]
          id_from_numden(num, den)
        })))
        ord <- order(gains); take <- remain[ head(ord, min(batch_k - 1L, length(remain))) ]
        if (length(take)){
          for (j in take){
            wj <- max(w_min, w[j] * batch_factor)
            num_cur <- num_cur + (wj - w[j]) * cache$N[[j]]
            den_cur <- den_cur + (wj - w[j]) * cache$S[[j]]
            w[j] <- wj
          }
          id_new <- id_from_numden(num_cur, den_cur)
          if (is.finite(id_new) && id_new < id - 1e-6){
            id <- id_new; changed <- TRUE
            hist <- rbind(hist, data.frame(iter=it, ID=id, changed=paste(vars[take], collapse=","),
                                           note=sprintf("batch x%.2f", batch_factor)))
            if (verbose) cat(sprintf("[constrained] iter %d: batch x%.2f on %d vars | %s = %.3f\n",
                                     it, batch_factor, length(take), objective, id))
          }
        }
      }
    }
    
    if (!changed){ if (verbose) cat("[constrained] no further improvement; stopping.\n"); break }
  }
  
  if (plot_progress && requireNamespace("ggplot2", quietly=TRUE)){
    gp <- ggplot2::ggplot(hist, ggplot2::aes(iter, ID)) +
      ggplot2::geom_line() + ggplot2::geom_point() +
      ggplot2::labs(title=sprintf("Constrained Gower-weight optimisation (%s; frozen core on reps)", objective),
                    x="iteration", y="ID") + ggplot2::theme_minimal()
    print(gp); try(ggplot2::ggsave("FIG_weight_optim_constrained.png", gp, width=6, height=4, dpi=150), silent=TRUE)
  }
  
  list(weights = w, history = hist, final_ID = id,
       idx_used = ix_sub_from_reps, core_idx = core_idx_sub)
}

# ====================== 8) Apply survivor selection (glue) =====================
# 8.1 Start from all non-constant variables; optional correlation trim (OFF)
drop_constant_cols <- function(X){
  keep <- vapply(X, function(v) length(unique(na.omit(v))) >= 1L, logical(1))
  X[, keep, drop=FALSE]
}

X_pred <- drop_constant_cols(X)
if (DO_CORR_TRIM) X_pred <- corr_trim(X_pred, CORR_THRESH)
cat(sprintf("TwoNN-only start: X_pred has %d columns after const-drop%s.\n\n",
            ncol(X_pred), if (DO_CORR_TRIM) " + corr-trim" else ""))

# 8.2 Weights and update mask:
#     - Initialise all weights at 1 for all columns in X.
#     - Only variables present in X_pred are allowed to change.
w_init <- setNames(rep(1, ncol(X)), colnames(X))
allow  <- setNames(rep(FALSE, ncol(X)), colnames(X))
allow[colnames(X_pred)] <- TRUE

# 8.3 Run the optimiser on the FULL X (so type handling is consistent),
#     but with updates allowed only on X_pred columns. Frozen core on reps.
wopt <- optimise_gower_weights_constrained(
  X,
  init_weights  = w_init,
  allow_update  = allow,
  objective     = "TwoNN_all",
  w_min         = W_MIN,
  step_grid     = W_STEP_GRID,
  batch_k       = W_BATCH_K,
  batch_factor  = W_BATCH_FACTOR,
  max_iter      = W_MAX_ITERS,   # NA ⇒ safe internal cap
  n_rows_sub    = N_ROWS_SUB,
  ncores        = NCORES_PAR,
  seed_jitter   = SEED_JITTER,
  reps_idx      = reps,
  core_idx_rep  = core_idx_rep,
  verbose       = TRUE,
  plot_progress = TRUE
)

# 8.4 Choose survivors using the tail-knee rule on the optimised weights
sel <- .survivors_from_weights(
  w         = wopt$weights,                         # names preserved
  w_min     = W_MIN,
  kmin      = max(30L, ceiling(0.10 * length(wopt$weights))),
  make_plot = TRUE
)

w_full    <- sel$w_clamped
survivors <- sel$survivors

# Materialise the survivor set (freeze X to survivors and align weights)
X_pred <- X[, survivors, drop = FALSE]
X      <- X_pred
w_all  <- w_full[survivors]

cat(sprintf("[TwoNN optimiser] Survivors: p=%d\n", ncol(X_pred)))

# Fail-soft guard: if something went wrong, fall back to the top-50 by weight
if (ncol(X_pred) == 0L) {
  warning("No survivors after pruning; falling back to top-50 by weight.")
  ord  <- order(w_full, decreasing = TRUE)
  keep <- names(w_full)[head(ord, min(50, length(ord)))]
  X_pred <- X[, keep, drop = FALSE]
  X      <- X_pred
  w_all  <- w_full[keep]
  cat(sprintf("[Fallback] p=%d\n", ncol(X_pred)))
}

# 8.5 Single geometry sanity check (TwoNN IDs) — done once here
PX      <- prep_X_for_gower(X, rare_prop = RARE_LEVEL_MIN_PROP, do_jitter = TRUE)
Xg      <- PX$X                 # <- the actual data.frame for daisy()
type_g  <- PX$type              # <- typed list aligned to Xg's columns
w_use   <- w_all[colnames(Xg)]  # <- align weights to *Xg* columns

stopifnot(is.data.frame(Xg),
          length(w_use) == ncol(Xg),
          all(!is.na(w_use)))

D_final <- cluster::daisy(Xg, metric = "gower",
                          type = type_g, weights = w_use)

ID_all  <- twonn_id_from_dist(D_final)
core_ix <- twonn_core_by_slope(D_final)

DmF     <- as.matrix(D_final); diag(DmF) <- Inf
ID_core <- twonn_id_from_dist(as.dist(DmF[core_ix, core_ix, drop = FALSE]))
ID_LB   <- lb_mle_id(DmF[core_ix, core_ix, drop = FALSE], 5, 15)

cat(sprintf("[Predictive-constrained %s] TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f (n_core=%d, p_active=%d)\n",
            toupper(PREF_TARGET), ID_all, ID_core, ID_LB, length(core_ix), ncol(X_final)))

# ================= 9) Base (m*=2), whitening, neighbours, residuals ============
# Big picture:
#   1) Encode the survivor items X → design matrix Xenc with a map var → columns.
#   2) Distribute each variable’s Gower weight across its dummy/contrast columns,
#      sqrt-scale columns so PCA “feels” the weights.
#   3) Standardise, do PCA to m*=2 (policy: M_STAR_FIXED), then whiten Base so
#      Euclidean distances ≈ Mahalanobis in Base space (cleaner kNN).
#   4) Build neighbour sets on whitened Base (we’ll reuse these later).
#   5) Residualise the encoded columns on Base in an out-of-fold way to avoid
#      leakage; these residuals E carry “fibre” variation orthogonal to Base.

# --- 9.0 Hardened encoder (available if not already defined) -------------------
# Turns mixed-type data.frame into a numeric model matrix with a mapping from
# encoded columns back to original variables. This is robust to factors, ordered
# factors, booleans, numerics, etc., and drops zero-variance columns safely.

design_with_map <- get0("design_with_map", ifnotfound = function(X) {
  Xg <- as.data.frame(X, check.names = TRUE, stringsAsFactors = FALSE)
  if (!ncol(Xg)) stop("[design_with_map] input has 0 columns at entry")
  
  # Keep only columns with at least one non-NA value
  keep <- vapply(Xg, function(v) length(unique(na.omit(v))) >= 1L, logical(1))
  if (!any(keep)) stop("[design_with_map] all columns are NA-only")
  Xg <- Xg[, keep, drop = FALSE]
  
  # Coerce types deterministically
  for (nm in names(Xg)) {
    v <- Xg[[nm]]
    if (is.ordered(v))      { Xg[[nm]] <- as.numeric(v); next }
    if (is.numeric(v) ||
        is.integer(v))      next
    if (is.logical(v))      { Xg[[nm]] <- factor(v, levels = c(FALSE, TRUE)); next }
    if (is.factor(v))       next
    if (is.matrix(v))       { Xg[[nm]] <- as.numeric(v); next }
    Xg[[nm]] <- factor(as.character(v))
  }
  
  # One big no-intercept formula so assign indices line up with variables
  fml <- as.formula(paste("~", paste(colnames(Xg), collapse = " + "), "-1"))
  tm  <- terms(fml, data = Xg)
  MM  <- model.matrix(tm, data = Xg)
  storage.mode(MM) <- "double"
  if (!ncol(MM)) stop("[design_with_map] model.matrix produced 0 columns")
  
  # Drop ~zero-variance encodings (protects PCA downstream)
  ok <- apply(MM, 2L, function(col) {
    v <- stats::var(as.numeric(col), na.rm = TRUE)
    is.finite(v) && v > 1e-12
  })
  if (!any(ok)) stop("[design_with_map] all encoded columns were ~zero-variance")
  
  assign <- attr(MM, "assign")
  tl     <- attr(tm, "term.labels")     # original variable names
  varmap <- tl[assign]                  # encoded column → source variable
  
  MM <- MM[, ok, drop = FALSE]
  attr(MM, "varmap") <- varmap[ok]
  MM
})

# --- 9.1 Encode survivors and prepare weights on encodings ---------------------
# Input at this point:
#   - X      : survivor data.frame (columns already pruned)
#   - w_all  : named vector of Gower weights for columns in X
#   - M_STAR_FIXED : policy knob (we fixed it to 2 globally)

stopifnot(exists("X"), ncol(X) >= 1L, exists("w_all"))

Xenc   <- design_with_map(X)                 # numeric design (n x q)
varmap <- attr(Xenc, "varmap")               # length q; names(Xenc) → original var
vars   <- unique(varmap)                     # survivor variable names (length p*)

# Distribute each variable’s Gower weight evenly across its encoded columns,
# then sqrt-scale columns so that a variable with weight w contributes ~w in
# variance (because PCA operates on covariance).
w_enc <- setNames(rep(1, ncol(Xenc)), colnames(Xenc))
alloc <- table(varmap)
for (nm in names(alloc)) {
  idx <- which(varmap == nm)
  wj  <- w_all[nm]; if (!is.finite(wj)) wj <- 1
  w_enc[idx] <- wj / length(idx)
}
Xenc_w <- sweep(Xenc, 2, sqrt(pmax(w_enc, 0)), "*")

# --- 9.2 Standardise & PCA to m*=2 (y-agnostic Base) --------------------------
m_star <- as.integer(M_STAR_FIXED)           # policy: fixed at 2
Z  <- scale(Xenc_w, center = TRUE, scale = TRUE)
pc <- prcomp(Z, rank. = max(2L, min(m_star, nrow(Z) - 1L, ncol(Z))))
Base <- pc$x[, 1:m_star, drop = FALSE]
colnames(Base) <- paste0("b", seq_len(ncol(Base)))

# (Optional) quick report of explained variance for sanity
if (exists("ggplot2") && requireNamespace("ggplot2", quietly = TRUE)) {
  ve <- (pc$sdev[seq_len(ncol(Base))]^2) / sum(pc$sdev^2)
  cat(sprintf("[Base] m*=%d | per-PC var: %s | total (m*): %.3f\n",
              m_star, paste(round(ve, 3), collapse = ", "),
              round(sum(ve), 3)))
} else {
  cat(sprintf("[Base] m*=%d | total var (first m* PCs): %.3f\n",
              m_star, sum((pc$sdev[seq_len(ncol(Base))]^2) / sum(pc$sdev^2))))
}

# --- 9.3 Whiten Base (Mahalanobis ≈ Euclidean) --------------------------------
# Whitening makes neighbour search better conditioned (no dominant PC axis).
S      <- stats::cov(Base)
U      <- try(chol(S + diag(1e-8, ncol(Base))), silent = TRUE)
if (inherits(U, "try-error")) {
  # If near-singular (tiny n), fall back to eigen-based whitening
  eig <- eigen(S, symmetric = TRUE)
  U   <- t(eig$vectors %*% diag(sqrt(pmax(eig$values, 1e-8))) %*% t(eig$vectors))
}
Base_w <- Base %*% solve(U)

# --- 9.4 Neighbour sets on whitened Base --------------------------------------
# We build a small grid of k for later fibre CV/selection.
KS_FIBRE <- get0("KS_FIBRE", ifnotfound = c(6, 8, 10, 12, 16, 20))
nb_list  <- setNames(lapply(KS_FIBRE, function(k){
  RANN::nn2(Base_w, Base_w, k = pmin(k + 1L, nrow(Base_w)))$nn.idx[, -1L, drop = FALSE]
}), paste0("k", KS_FIBRE))

# --- 9.5 Fold-safe residualisation of encodings on Base -----------------------
# Intuition: for each encoded column, predict it from Base (smoothly). The
# out-of-fold residuals E = "what the item carries that Base doesn’t".
# We’ll standardise E for comparability across columns.

# A lightweight, fold-matched GAM residualiser (uses smooths over b1..bm).
residualise_foldsafe <- get0("residualise_foldsafe", ifnotfound = function(Xenc, Base, folds, k_gam = 6){
  n <- nrow(Base); V <- colnames(Xenc)
  E <- matrix(NA_real_, n, length(V), dimnames = list(rownames(Base), V))
  sm_terms <- paste0("s(b", seq_len(ncol(Base)), ",k=", k_gam, ")")
  for (v in V){
    z <- as.numeric(Xenc[, v])
    for (k in sort(unique(folds))){
      tr <- which(folds != k); te <- which(folds == k)
      dftr <- data.frame(v = z[tr], Base[tr, , drop = FALSE])
      fml  <- reformulate(sm_terms, response = "v")
      g <- try(mgcv::gam(fml, data = dftr, method = "REML"), silent = TRUE)
      if (inherits(g, "try-error")) next
      mu <- as.numeric(predict(g, newdata = data.frame(Base[te, , drop = FALSE]),
                               type = "response"))
      E[te, v] <- z[te] - mu
    }
  }
  E[, colSums(is.finite(E)) > 0, drop = FALSE]
})

stack_with_foldid <- get0("stack_with_foldid", ifnotfound = function(y, pB, pR, fold_id, seed = 42){
  set.seed(seed); y <- as.integer(y)
  p <- rep(NA_real_, length(y))
  for (k in sort(unique(fold_id))){
    tr <- which(fold_id != k); te <- which(fold_id == k)
    dftr <- data.frame(y = y[tr], pB = pB[tr], pR = pR[tr])
    dfte <- data.frame(pB = pB[te], pR = pR[te])
    fit  <- glm(y ~ pB + pR, data = dftr, family = binomial())
    p[te] <- as.numeric(predict(fit, newdata = dfte, type = "response"))
  }
  pmin(pmax(p, 1e-6), 1-1e-6)
})

# Simple K-fold stratification without leakage (balanced by class if available)
make_strat_folds <- get0("make_strat_folds", ifnotfound = function(y, K, group = NULL, seed = 1){
  set.seed(seed); y <- as.integer(y)
  n <- length(y); folds <- integer(n)
  if (is.null(group)) {
    idx0 <- which(y == 0); idx1 <- which(y == 1)
    f0 <- sample(rep(1:K, length.out = length(idx0)))
    f1 <- sample(rep(1:K, length.out = length(idx1)))
    folds[idx0] <- f0; folds[idx1] <- f1
  } else {
    g <- as.factor(group)
    for (lev in levels(g)){
      ix <- which(g == lev); yix <- y[ix]
      idx0 <- ix[yix == 0]; idx1 <- ix[yix == 1]
      if (length(idx0)) folds[idx0] <- sample(rep(1:K, length.out = length(idx0)))
      if (length(idx1)) folds[idx1] <- sample(rep(1:K, length.out = length(idx1)))
    }
  }
  folds
})

# Choose a conservative K (avoid tiny training splits); otherwise use CV_FOLDS.
choose_K <- get0("choose_K", ifnotfound = function(y, K_target = CV_FOLDS, min_per_class = 8){
  y <- as.integer(y); n1 <- sum(y == 1); n0 <- sum(y == 0)
  max(2, min(K_target, floor(n1 / min_per_class), floor(n0 / min_per_class)))
})

# Build fold ids (if y_use exists we stratify by class; otherwise just K groups)
if (exists("y_use") && length(unique(y_use)) >= 2) {
  K_fold  <- choose_K(y_use, K_target = CV_FOLDS, min_per_class = 8)
  fold_id <- make_strat_folds(y_use, K = K_fold, seed = SEED_PRED)
} else {
  set.seed(SEED_PRED)
  K_fold  <- min(CV_FOLDS, nrow(Base))
  fold_id <- sample(rep(1:K_fold, length.out = nrow(Base)))
}

# Residualise the **weighted** encoding on the **actual Base** (not whitened).
E         <- residualise_foldsafe(Xenc_w, Base, folds = fold_id, k_gam = 6)
E_scaled  <- scale(E, center = TRUE, scale = TRUE)

cat(sprintf("[Residuals] E matrix: %d rows × %d columns (post-OOF, scaled)\n",
            nrow(E_scaled), ncol(E_scaled)))

# --- 9.6 (For later roles) per-variable contribution to Base PCs --------------
# We’ll need a stable measure of how much each original variable builds the Base.
# This uses the *unweighted* design to avoid double counting the Gower weights.

compute_pc_contrib <- get0("compute_pc_contrib", ifnotfound = function(Xenc, varmap, m) {
  out <- try({
    Z_  <- scale(Xenc, center = TRUE, scale = TRUE)
    rk  <- max(1L, min(m, ncol(Z_), nrow(Z_) - 1L))
    pc_ <- prcomp(Z_, center = FALSE, scale. = FALSE, rank. = rk)
    var_prop <- (pc_$sdev[seq_len(rk)]^2) / sum(pc_$sdev^2)
    per_pc   <- lapply(seq_len(rk), function(j) {
      tapply(pc_$rotation[, j]^2, varmap, sum, default = 0)
    })
    contr <- Reduce(`+`, Map(function(v, w) v * w, per_pc, var_prop))
    contr / sum(contr)
  }, silent = TRUE)
  if (inherits(out, "try-error") || is.null(out)) {
    vv <- unique(varmap); out <- setNames(rep(0, length(vv)), vv)
  }
  out
})

PC_contrib <- compute_pc_contrib(Xenc, varmap, m = m_star)
# Ensure alignment to survivor variable order
if (!all(vars %in% names(PC_contrib))) {
  miss <- setdiff(vars, names(PC_contrib))
  PC_contrib <- c(PC_contrib, setNames(rep(0, length(miss)), miss))
}
PC_contrib <- PC_contrib[vars]

# --- 9.7 Small helper for item-level residual score (used in roles) -----------
# Project the residual block for a variable onto PC1 of its *standardised*
# encoding block Z to get a single “residual score” vector per item.
e_from_E <- get0("e_from_E", ifnotfound = function(nm, E_scaled, Z, varmap){
  idx <- which(varmap == nm)
  if (!length(idx)) return(rep(NA_real_, nrow(Z)))
  if (length(idx) == 1L) return(as.numeric(E_scaled[, idx]))
  pc1 <- try(prcomp(Z[, idx, drop = FALSE], rank. = 1), silent = TRUE)
  if (inherits(pc1, "try-error")) return(rep(NA_real_, nrow(Z)))
  as.numeric(as.matrix(E_scaled[, idx, drop = FALSE]) %*% pc1$rotation[, 1])
})

# Summary ping so we know the geometry we’ll hand to the roles section:
cat(sprintf("[Base] built with m*=%d; neighbours: %s; E columns: %d; vars: %d\n",
            m_star, paste(KS_FIBRE, collapse = ","), ncol(E_scaled), length(vars)))

# =================== 10) Item roles: Base vs Fibre diagnostics =================
# What we want:
#   For each survivor item (variable):
#     - How much of it is explained by the low-dim Base (m*=2)?   → R²_base + p_base
#     - How much is explained by local fibre neighbours (in Base)? → R²_fibre + p_fibre
#     - How much does the item help build Base?                    → PC_contrib
#   Then we call a role:
#     base-aligned / fibre-structured / mixed / weak
#
# Inputs expected from previous sections:
#   - Xenc        : encoded design (n × q)
#   - varmap      : length-q vector mapping encoded cols → original variables
#   - vars        : unique(varmap) (survivor item names)
#   - Base        : n × m* matrix (m*=2)  ← fixed by policy (M_STAR_FIXED)
#   - Base_w      : whitened Base (used for neighbours) — we built nb_list already
#   - nb_list     : list of k-NN index matrices on Base_w (keys: "k6","k8",…)
#   - E_scaled    : out-of-fold residuals of encoded columns on Base (standardised)
#   - PC_contrib  : per-variable contribution to Base PCs (unit-sum)
#
# We’ll also keep a *standardised unweighted* encoding to define per-item
# univariate “scores” consistently (PC1 of that item’s block), so residual
# projections do not double-count Gower weights.

# ---- 10.1 A standardised, *unweighted* design for item scoring ---------------
Z0_std <- scale(Xenc, center = TRUE, scale = TRUE)

# ---- 10.2 Small helpers (defined here if missing) ----------------------------

# Single-number “item score” aligned with Base: if an item expands into multiple
# encoded columns, take its PC1 (on the *unweighted* Z0_std block).
score_item_base <- get0("score_item_base", ifnotfound = function(nm, Z, varmap){
  idx <- which(varmap == nm)
  if (!length(idx)) return(rep(NA_real_, nrow(Z)))
  if (length(idx) == 1L) return(as.numeric(Z[, idx]))
  sc <- try(suppressWarnings(prcomp(Z[, idx, drop = FALSE], rank. = 1)$x[, 1]), silent = TRUE)
  if (inherits(sc, "try-error")) return(rep(NA_real_, nrow(Z)))
  as.numeric(sc)
})

# Fast R² to Base with permutation p-value.
# We keep it linear (no splines) for stability and speed; permutation tests
# respect the dependence across rows by permuting Base rows as a block.
r2_base_perm_fast <- get0("r2_base_perm_fast", ifnotfound = function(v, Base, B = 200,
                                                                     method = c("lm","ridge"),
                                                                     alpha_early = NULL, seed = NULL) {
  method <- match.arg(method)
  if (!any(is.finite(v))) return(c(R2 = NA_real_, p = NA_real_))
  if (!is.null(seed)) set.seed(seed)
  
  n <- length(v)
  X <- cbind(1, as.matrix(Base))
  y <- as.numeric(scale(v))
  
  fit_obs <- switch(method,
                    lm = {
                      XtX <- crossprod(X); Xty <- crossprod(X, y)
                      beta <- tryCatch(solve(XtX, Xty), error = function(e) NA_real_)
                      if (any(!is.finite(beta))) return(c(R2=NA_real_, p=NA_real_))
                      yhat <- as.vector(X %*% beta)
                      1 - sum((y - yhat)^2)/sum((y - mean(y))^2)
                    },
                    ridge = {
                      lambda <- 0.1
                      XtX <- crossprod(X); Xty <- crossprod(X, y)
                      beta <- tryCatch(solve(XtX + diag(lambda, ncol(X)), Xty), error = function(e) NA_real_)
                      if (any(!is.finite(beta))) return(c(R2=NA_real_, p=NA_real_))
                      yhat <- as.vector(X %*% beta)
                      1 - sum((y - yhat)^2)/sum((y - mean(y))^2)
                    }
  )
  if (!is.finite(fit_obs)) return(c(R2 = NA_real_, p = NA_real_))
  
  exceed <- 0L; b <- 0L
  while (b < B) {
    b <- b + 1L
    perm <- sample.int(n)
    Xp <- X; Xp[, -1] <- X[perm, -1, drop = FALSE]
    XtXb <- crossprod(Xp); Xtyb <- crossprod(Xp, y)
    beta_b <- tryCatch(solve(XtXb, Xtyb), error = function(e) NA_real_)
    if (any(!is.finite(beta_b))) next
    yhat_b <- as.vector(Xp %*% beta_b)
    R2_b   <- 1 - sum((y - yhat_b)^2)/sum((y - mean(y))^2)
    if (is.finite(R2_b) && R2_b >= fit_obs) exceed <- exceed + 1L
    
    # Optional early stopping if even the best possible p is already > α
    if (!is.null(alpha_early)) {
      p_min_possible <- (exceed + 1) / (B + 1)
      if (p_min_possible > alpha_early) break
    }
  }
  p <- (exceed + 1) / (b + 1)
  c(R2 = fit_obs, p = p)
})

# Neighbour-mean CV R² for fibre (choose k by small CV on Base_w neighbours).
r2_fibre_cv <- get0("r2_fibre_cv", ifnotfound = function(e, nb, folds = 3, seed = 1){
  set.seed(seed)
  n <- length(e)
  if (n < 6 || all(!is.finite(e))) return(NA_real_)
  fold_id <- sample(rep(1:folds, length.out = n))
  pred <- rep(NA_real_, n)
  for (f in 1:folds){
    te <- which(fold_id == f)
    e_mask <- e; e_mask[te] <- NA                # no peeking into test rows
    pr <- rowMeans(matrix(e_mask[nb], nrow = n), na.rm = TRUE)
    pr[is.na(pr)] <- mean(e[-te], na.rm = TRUE)  # safe fallback for sparse neighbourhoods
    pred[te] <- pr[te]
  }
  ve  <- stats::var(e, na.rm = TRUE)
  mse <- mean((e - pred)^2, na.rm = TRUE)
  if (!is.finite(ve) || ve <= 0) return(NA_real_)
  max(0, 1 - mse/ve)
})

choose_k_nb <- get0("choose_k_nb", ifnotfound = function(e, nb_list, folds = 3, seed = 1){
  r2s <- vapply(nb_list, function(nb) r2_fibre_cv(e, nb, folds, seed), numeric(1))
  ix  <- which.max(r2s)
  list(k = as.integer(sub("^k","", names(nb_list)[ix])),
       R2_cv = r2s[ix], nb = nb_list[[ix]], all = r2s)
})

# Fibre p-value with *coherent* null: we shuffle the *rows of neighbour sets*
# together, so the internal structure of each neighbourhood is preserved.
r2_fibre_perm_rowshuffle <- get0("r2_fibre_perm_rowshuffle", ifnotfound = function(e, nb, B = 200, seed = 1){
  set.seed(seed)
  if (!any(is.finite(e))) return(c(R2 = NA_real_, p = NA_real_))
  pred <- rowMeans(matrix(e[nb], nrow = length(e)), na.rm = TRUE)
  ve   <- stats::var(e, na.rm = TRUE)
  mse  <- mean((e - pred)^2, na.rm = TRUE)
  R2obs <- if (is.finite(ve) && ve > 0) max(0, 1 - mse/ve) else NA_real_
  if (!is.finite(R2obs) || R2obs <= 1e-12) return(c(R2 = 0, p = 1))
  
  R2null <- numeric(B)
  for (b in seq_len(B)){
    ix <- sample.int(nrow(nb))  # move neighbour sets around as intact rows
    pred_b <- rowMeans(matrix(e[nb[ix, , drop = FALSE]], nrow = length(e)), na.rm = TRUE)
    R2null[b] <- max(0, 1 - mean((e - pred_b)^2, na.rm = TRUE)/ve)
  }
  p <- (sum(is.finite(R2null) & R2null >= R2obs) + 1) / (sum(is.finite(R2null)) + 1)
  c(R2 = R2obs, p = p)
})

# Safe scalar helpers for robust data.frame construction
.scalar_or <- get0(".scalar_or", ifnotfound = function(x, default = NA_real_) {
  if (is.null(x) || length(x) == 0L) return(default)
  xx <- suppressWarnings(as.numeric(x))
  if (all(!is.finite(xx))) return(default)
  xx[1]
})
.take_named <- get0(".take_named", ifnotfound = function(x, nm, default = NA_real_) {
  if (is.null(x) || is.null(names(x)) || !(nm %in% names(x))) return(default)
  .scalar_or(x[[nm]], default)
})

# ---- 10.3 Roles per item -----------------------------------------------------

# Knobs (stable defaults; override upstream if needed)
FIBRE_PERM_B <- get0("FIBRE_PERM_B", ifnotfound = 200L)
SIG_Q        <- get0("SIG_Q",        ifnotfound = 0.01)
MIN_SD_ITEM  <- get0("MIN_SD_ITEM",  ifnotfound = 1e-6)
ADJUST_FOR_K <- get0("ADJUST_FOR_K", ifnotfound = TRUE)

roles_rows <- lapply(vars, function(nm){
  # 1) Item’s aligned score for Base
  v <- score_item_base(nm, Z0_std, varmap)
  if (stats::sd(v, na.rm = TRUE) < MIN_SD_ITEM) return(NULL)
  
  # 2) R² to Base + permutation p
  rb  <- try(r2_base_perm_fast(v, Base, B = FIBRE_PERM_B, method = "lm",
                               alpha_early = 0.05, seed = 123), silent = TRUE)
  R2b <- .take_named(rb, "R2");  pb <- .take_named(rb, "p")
  
  # 3) Residual series for this item (OOF, already in E_scaled);
  #    project its residual block through PC1 learned on Z0_std.
  e_item <- e_from_E(nm, E_scaled, Z0_std, varmap)
  if (!any(is.finite(e_item))) return(NULL)
  
  # 4) Choose neighbour count k by tiny CV on Base_w neighbours
  sel <- choose_k_nb(e_item, nb_list, folds = 3, seed = 456)
  
  # 5) R² to neighbours + p via row-shuffle null (optionally Bonferroni over k grid)
  rf  <- try(r2_fibre_perm_rowshuffle(e_item, sel$nb, B = FIBRE_PERM_B, seed = 456), silent = TRUE)
  R2f <- .take_named(rf, "R2");  pf <- .take_named(rf, "p")
  if (isTRUE(ADJUST_FOR_K) && is.finite(pf)) pf <- min(1, pf * length(nb_list))
  
  # 6) Contribution to Base PCs (already aligned and unit-sum)
  pc_c <- if (!is.null(PC_contrib) && (nm %in% names(PC_contrib))) as.numeric(PC_contrib[[nm]]) else 0
  
  data.frame(
    var        = nm,
    R2_base    = .scalar_or(R2b, NA_real_),
    p_base     = .scalar_or(pb,  NA_real_),
    R2_fibre   = .scalar_or(R2f, NA_real_),
    p_fibre    = .scalar_or(pf,  NA_real_),
    k_fibre    = .scalar_or(sel$k, NA_integer_),
    PC_contrib = pc_c,
    stringsAsFactors = FALSE
  )
})

roles_df <- dplyr::bind_rows(Filter(Negate(is.null), roles_rows))
if (!nrow(roles_df)) stop("[roles] No items produced valid role stats.")

# Clamp R² to [0,1] for neatness
roles_df$R2_base  <- pmax(0, pmin(1, as.numeric(roles_df$R2_base)))
roles_df$R2_fibre <- pmax(0, pmin(1, as.numeric(roles_df$R2_fibre)))

# ---- 10.4 Multiple testing control (Benjamini–Hochberg) ----------------------
roles_df$q_base  <- p.adjust(roles_df$p_base,  method = "BH")
roles_df$q_fibre <- p.adjust(roles_df$p_fibre, method = "BH")

# ---- 10.5 Role assignment: FDR and effect-size gates -------------------------
# FDR-based role (statistical significance)
roles_df$role_fdr <- with(
  roles_df,
  ifelse(q_base  < SIG_Q & (is.na(q_fibre) | q_fibre >= SIG_Q), "base-aligned",
         ifelse(q_fibre < SIG_Q & (is.na(q_base)  | q_base  >= SIG_Q), "fibre-structured",
                ifelse(q_base  < SIG_Q & q_fibre < SIG_Q,                      "mixed", "weak")))
)

# Effect-size thresholds (interpretability-first)
ES_BASE  <- get0("ES_BASE",  ifnotfound = 0.08)
ES_FIBRE <- get0("ES_FIBRE", ifnotfound = 0.05)
roles_df$role_es <- dplyr::case_when(
  roles_df$R2_base  >= ES_BASE  & (is.na(roles_df$R2_fibre) | roles_df$R2_fibre <  ES_FIBRE) ~ "base-aligned",
  roles_df$R2_fibre >= ES_FIBRE & (is.na(roles_df$R2_base)  | roles_df$R2_base  <  ES_BASE ) ~ "fibre-structured",
  roles_df$R2_base  >= ES_BASE  & roles_df$R2_fibre >= ES_FIBRE                               ~ "mixed",
  TRUE                                                                                         ~ "weak"
)

# Pick final scheme: prefer effect-size unless it yields zero base-aligned;
# if that happens, fall back to FDR roles.
ROLE_SCHEME <- get0("ROLE_SCHEME", ifnotfound = "es")  # "es" or "fdr"
role_col    <- if (ROLE_SCHEME == "fdr") "role_fdr" else "role_es"
if (!any(roles_df[[role_col]] == "base-aligned", na.rm = TRUE)) {
  message(sprintf("[roles] No 'base-aligned' under '%s'; falling back to the other scheme.", role_col))
  role_col <- if (role_col == "role_es") "role_fdr" else "role_es"
}

roles_df$role_final <- factor(
  roles_df[[role_col]],
  levels = c("base-aligned","mixed","fibre-structured","weak")
)

# Neat ordering for reports: strongest Base builders first
roles_df <- roles_df[order(-roles_df$PC_contrib, -roles_df$R2_base, roles_df$var), ]

cat("[roles] counts by role (FDR):\n");  print(table(roles_df$role_fdr, useNA = "ifany"))
cat("[roles] counts by role (ES):\n");   print(table(roles_df$role_es,  useNA = "ifany"))
cat(sprintf("[roles] using scheme: %s\n", role_col))

# ---- 10.6 Save + optional volcano plot ---------------------------------------
readr::write_csv(roles_df, "predictive_item_roles_diagnostics.csv")
cat(sprintf("[roles] wrote: predictive_item_roles_diagnostics.csv  (p=%d)\n", nrow(roles_df)))

if (requireNamespace("ggplot2", quietly = TRUE)) {
  sup <- roles_df %>%
    dplyr::filter(is.finite(R2_base), is.finite(R2_fibre)) %>%
    dplyr::mutate(role = role_final)
  
  pV <- ggplot2::ggplot(sup, ggplot2::aes(R2_base, R2_fibre, size = PC_contrib, color = role)) +
    ggplot2::geom_point(alpha = 0.8) +
    ggplot2::geom_vline(xintercept = 0, linetype = 3) +
    ggplot2::geom_hline(yintercept = 0, linetype = 3) +
    ggplot2::labs(title = "Item roles: Base vs Fibre (final roles)",
                  x = "R² to Base", y = "R² to fibre neighbours", color = "role") +
    ggplot2::theme_minimal()
  print(pV)
  ggplot2::ggsave("FIG_roles_volcano_base_vs_fibre.png", pV, width = 7, height = 5, dpi = 150)
}

# ========== 11) Fibre-only decomposition & clustering (diagnosis-agnostic) =====
# Purpose (plain English):
#   - We ignore the original items for a moment and look ONLY at E (the
#     out-of-fold residuals orthogonal to Base).
#   - We do a PCA on E to get a small “fibre base” B′ (f1, f2, ...).
#   - We remove B′ from E in an out-of-fold linear way to get F′ (residual
#     “fibre-of-fibre”), just to check if anything remains structured.
#   - We compute intrinsic dimensions (TwoNN / LB) for E, B′, F′.
#   - We cluster in B′ (Louvain on a kNN graph) and report a simple stability score.

stopifnot(exists("E_scaled"), is.matrix(E_scaled), nrow(E_scaled) >= 3)

# ---- 11.1 Prep: choose a safe PCA rank cap and build a normalised E ----------
Ef <- scale(E_scaled, center = TRUE, scale = TRUE)
Ef[!is.finite(Ef)] <- 0
n  <- nrow(Ef)
pE <- ncol(Ef)

if (pE < 2L || n < 4L) {
  stop("[Fibre-only] not enough columns/rows in E to proceed (need ≥2 cols and ≥4 rows).")
}

FIBRE_BASE_MAX <- min(6L, pE, n - 1L)

# ---- 11.2 PCA on E to get an unsupervised fibre-base B′ ----------------------
pc_f <- prcomp(Ef, rank. = max(2L, FIBRE_BASE_MAX))
Bprime_all <- pc_f$x[, 1:FIBRE_BASE_MAX, drop = FALSE]
colnames(Bprime_all) <- paste0("f", seq_len(ncol(Bprime_all)))

# (optional) simple scree report/plot
if (requireNamespace("ggplot2", quietly = TRUE)) {
  ve <- (pc_f$sdev^2) / sum(pc_f$sdev^2)
  df_scree <- data.frame(PC = seq_along(ve), var = ve)
  p_scree <- ggplot2::ggplot(df_scree, ggplot2::aes(PC, var)) +
    ggplot2::geom_col(width = 0.9) +
    ggplot2::geom_vline(xintercept = FIBRE_BASE_MAX + 0.5, linetype = 2) +
    ggplot2::labs(title = "Fibre PCA scree", y = "Variance explained") +
    ggplot2::theme_minimal()
  print(p_scree)
  ggplot2::ggsave("FIG_fibre_scree.png", p_scree, width = 6, height = 7, dpi = 150)
}

# ---- 11.3 Pick m_f for B′ using trust/continuity (with light penalty) --------
# We penalise trivially larger m by a tiny λ so we avoid “always pick bigger”.
trust_cont       <- get0("trust_cont",       ifnotfound = NULL)
trust_cont_avg   <- get0("trust_cont_avg",   ifnotfound = NULL)
KS_TC            <- get0("KS_TC",            ifnotfound = 10:30)

if (is.null(trust_cont_avg)) {
  stop("[Fibre-only] trust_cont_avg() not found; ensure Section 2 helpers were sourced.")
}

pick_m_via_tc <- function(Xhigh, Xlow_all, ks = 10:30, mmax = ncol(Xlow_all), lambda = 0.02) {
  vals <- lapply(2:mmax, function(m) {
    tc <- trust_cont_avg(Xhigh, Xlow_all[, 1:m, drop = FALSE], ks)
    data.frame(m = m, T = tc["T"], C = tc["C"])
  }) |> dplyr::bind_rows()
  vals$score <- with(vals, (T + C) - lambda * (m - min(m)))
  vals$m[which.max(vals$score)]
}

m_f <- pick_m_via_tc(Ef, Bprime_all, ks = KS_TC, mmax = ncol(Bprime_all), lambda = 0.02)

# Also prepare the table of TC vs m_f for reporting
tc_tbl <- lapply(2:ncol(Bprime_all), function(m){
  tc <- trust_cont_avg(Ef, Bprime_all[, 1:m, drop = FALSE], ks = KS_TC)
  data.frame(m_f = m, Trust = tc["T"], Continuity = tc["C"])
}) |> dplyr::bind_rows()

cat(sprintf("[Fibre-only] m_f* = %d (mean Trust=%.3f, Continuity=%.3f at m_f*)\n",
            m_f,
            mean(tc_tbl$Trust[tc_tbl$m_f == m_f]),
            mean(tc_tbl$Continuity[tc_tbl$m_f == m_f])))

# ---- 11.4 Intrinsic dimension of E (independent of m_f) ----------------------
twonn_id_from_dist <- get0("twonn_id_from_dist", ifnotfound = NULL)
core_band_idx      <- get0("core_band_idx",      ifnotfound = NULL)
lb_mle_id          <- get0("lb_mle_id",          ifnotfound = NULL)
CORE_KNN_K         <- get0("CORE_KNN_K",         ifnotfound = 10L)
CORE_BAND          <- get0("CORE_BAND",          ifnotfound = c(0.15, 0.85))

if (any(sapply(list(twonn_id_from_dist, core_band_idx, lb_mle_id), is.null))) {
  stop("[Fibre-only] ID helper(s) missing; ensure Section 2 helpers were sourced.")
}

Df      <- stats::dist(Ef)
IDf_all <- twonn_id_from_dist(Df)
Mf      <- as.matrix(Df); diag(Mf) <- Inf
core_f  <- core_band_idx(Df, k = CORE_KNN_K, band = CORE_BAND)
IDf_core<- if (length(core_f) >= 3) twonn_id_from_dist(stats::as.dist(Mf[core_f, core_f])) else NA_real_
IDf_LB  <- if (length(core_f) >= 3) lb_mle_id(Mf[core_f, core_f, drop = FALSE], 5, 15) else NA_real_

# ---- 11.5 Fix B′ = first m_f PCs --------------------------------------------
Bprime <- Bprime_all[, 1:m_f, drop = FALSE]
colnames(Bprime) <- paste0("f", seq_len(ncol(Bprime)))

# ---- 11.6 Out-of-fold residuals of E on B′ → F′ (linear, stable) -------------
set.seed(SEED_PRED)
Kf      <- max(2L, min(CV_FOLDS, n))         # cap by available rows
folds_f <- sample(rep(1:Kf, length.out = n))

residualise_linear_oof <- function(Y, X, folds) {
  # Y: n × p (Ef), X: n × m (Bprime)
  n <- nrow(Y); p <- ncol(Y)
  R <- matrix(NA_real_, n, p, dimnames = list(rownames(Y), colnames(Y)))
  Xdf <- as.data.frame(X)
  for (j in seq_len(p)) {
    y <- Y[, j]
    if (all(!is.finite(y))) next
    for (k in sort(unique(folds))) {
      tr <- which(folds != k); te <- which(folds == k)
      fit <- try(stats::lm(y ~ ., data = cbind.data.frame(y = y[tr], Xdf[tr, , drop = FALSE])), silent = TRUE)
      if (inherits(fit, "try-error")) next
      mu  <- as.numeric(predict(fit, newdata = Xdf[te, , drop = FALSE]))
      R[te, j] <- y[te] - mu
    }
  }
  R[, colSums(is.finite(R)) > 0, drop = FALSE]
}

Fprime <- residualise_linear_oof(Ef, Bprime, folds_f)

# ---- 11.7 Intrinsic dimension of B′ (Euclidean) and F′ -----------------------
D_Bprime  <- stats::dist(Bprime)
ID_B_all  <- twonn_id_from_dist(D_Bprime)
MB        <- as.matrix(D_Bprime); diag(MB) <- Inf
core_B    <- core_band_idx(D_Bprime, k = CORE_KNN_K, band = CORE_BAND)
ID_B_core <- if (length(core_B) >= 3) twonn_id_from_dist(stats::as.dist(MB[core_B, core_B])) else NA_real_
ID_B_LB   <- if (length(core_B) >= 3) lb_mle_id(MB[core_B, core_B, drop = FALSE], 5, 15) else NA_real_

if (ncol(Fprime) >= 2) {
  D_Fprime  <- stats::dist(Fprime)
  ID_Fp_all <- twonn_id_from_dist(D_Fprime)
  MFp       <- as.matrix(D_Fprime); diag(MFp) <- Inf
  core_Fp   <- core_band_idx(D_Fprime, k = CORE_KNN_K, band = CORE_BAND)
  ID_Fp_core<- if (length(core_Fp) >= 3) twonn_id_from_dist(stats::as.dist(MFp[core_Fp, core_Fp])) else NA_real_
  ID_Fp_LB  <- if (length(core_Fp) >= 3) lb_mle_id(MFp[core_Fp, core_Fp, drop = FALSE], 5, 15) else NA_real_
} else {
  ID_Fp_all <- ID_Fp_core <- ID_Fp_LB <- NA_real_
}

cat(sprintf("[Fibre-only] ID(E):   TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f (n_core=%d)\n",
            IDf_all, IDf_core, IDf_LB, length(core_f)))
cat(sprintf("[Fibre-only] ID(B′):  TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f (n_core=%d)\n",
            ID_B_all, ID_B_core, ID_B_LB, length(core_B)))
cat(sprintf("[Fibre-only] ID(F′):  TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f (n_core=%s)\n",
            ID_Fp_all, ID_Fp_core, ID_Fp_LB,
            if (exists("core_Fp")) length(core_Fp) else "NA"))

# ---- 11.8 Clustering in B′ (unsupervised; no diagnoses) ----------------------
# kNN graph on B′ (m_f dims), Louvain community detection.
if (!requireNamespace("igraph", quietly = TRUE)) {
  warning("[Fibre-only] package 'igraph' not available; skipping clustering.")
  clF <- rep(1L, nrow(Bprime)); kF <- 1L
} else {
  knn_graph <- function(X, k = 15) {
    idx <- RANN::nn2(X, X, k = pmin(k + 1L, nrow(X)))$nn.idx[, -1L, drop = FALSE]
    i <- rep(seq_len(nrow(X)), each = ncol(idx)); j <- as.vector(idx)
    g <- igraph::graph_from_edgelist(cbind(i, j), directed = FALSE)
    igraph::simplify(g)
  }
  gF  <- knn_graph(Bprime, k = 15)
  clF <- igraph::cluster_louvain(gF)$membership
  kF  <- length(unique(clF))
  cat(sprintf("[Fibre-only] Louvain clusters (kNN=15): %d\n", kF))
  
  # Stability via bootstrap of points (adjusted NMI; coarse but informative)
  adjNMI <- function(a, b){
    tab <- table(a, b); P <- tab / sum(tab)
    pa <- rowSums(P);   pb <- colSums(P)
    mi <- sum(P * log(pmax(P, 1e-12) / (pa %o% pb)))
    Ha <- -sum(pa * log(pmax(pa, 1e-12))); Hb <- -sum(pb * log(pmax(pb, 1e-12)))
    mi / pmax((Ha + Hb) / 2, 1e-12)
  }
  set.seed(1)
  stab <- replicate(20, {
    ix  <- sample(seq_len(nrow(Bprime)), replace = TRUE)
    g2  <- knn_graph(Bprime[ix, , drop = FALSE], k = 15)
    cl2 <- igraph::cluster_louvain(g2)$membership
    adjNMI(clF[ix], cl2)
  })
  cat(sprintf("[Fibre-only] cluster stability (median adjNMI): %.3f\n", stats::median(stab, na.rm = TRUE)))
  
  # Small 2D snapshot if m_f ≥ 2
  if (requireNamespace("ggplot2", quietly = TRUE) && ncol(Bprime) >= 2L) {
    df_sc <- data.frame(f1 = Bprime[,1], f2 = Bprime[,2], cl = factor(clF))
    p_sc  <- ggplot2::ggplot(df_sc, ggplot2::aes(f1, f2, colour = cl)) +
      ggplot2::geom_point() +
      ggplot2::labs(title = "Fibre space (first 2 PCs)", x = "f1", y = "f2", colour = "cluster") +
      ggplot2::theme_minimal()
    print(p_sc)
    ggplot2::ggsave("FIG_fibre_scatter.png", p_sc, width = 6, height = 7, dpi = 150)
  }
  
  # TC vs m_f and ID vs m_f (ID constant across m_f, but shown for context)
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    df_tc <- tidyr::pivot_longer(tc_tbl, cols = c(Trust, Continuity),
                                 names_to = "metric", values_to = "val")
    p_tc <- ggplot2::ggplot(df_tc, ggplot2::aes(m_f, val, colour = metric)) +
      ggplot2::geom_line() + ggplot2::geom_point() +
      ggplot2::geom_vline(xintercept = m_f, linetype = 2) +
      ggplot2::labs(title = "Fibre trust/continuity vs m_f", x = "m_f", y = "score") +
      ggplot2::theme_minimal()
    print(p_tc); ggplot2::ggsave("FIG_fibre_tc_vs_m.png", p_tc, width = 6, height = 7, dpi = 150)
    
    p_id <- data.frame(m_f = tc_tbl$m_f,
                       TwoNN_all = IDf_all, TwoNN_core = IDf_core, LB_core = IDf_LB) |>
      tidyr::pivot_longer(-m_f, names_to = "kind", values_to = "ID") |>
      ggplot2::ggplot(ggplot2::aes(m_f, ID, colour = kind)) +
      ggplot2::geom_line() + ggplot2::geom_point() +
      ggplot2::geom_vline(xintercept = m_f, linetype = 2) +
      ggplot2::labs(title = "Fibre intrinsic dimension vs m_f", x = "m_f", y = "ID") +
      ggplot2::theme_minimal()
    print(p_id); ggplot2::ggsave("FIG_fibre_ID_vs_m.png", p_id, width = 6, height = 7, dpi = 150)
  }
}

# ---- 11.9 Persist a compact summary for downstream/QA -------------------------
saveRDS(list(
  Bprime      = Bprime,
  Fprime      = Fprime,
  m_f         = m_f,
  tc_tbl      = tc_tbl,
  ID_E        = c(all = IDf_all, core = IDf_core, LB = IDf_LB),
  ID_Bprime   = c(all = ID_B_all, core = ID_B_core, LB = ID_B_LB),
  ID_Fprime   = c(all = ID_Fp_all, core = ID_Fp_core, LB = ID_Fp_LB),
  clusters    = if (exists("clF")) clF else NULL
), file = "fibre_only_summary.rds")

cat("[Fibre-only] wrote: FIG_fibre_*.png (if ggplot2 available) and fibre_only_summary.rds\n")

# ================= 12) Per-diagnosis predictive evaluation (OOF) ===============
# Goal (bird’s-eye):
#   For each diagnosis column (one-vs-rest):
#     1) Build **fold-safe** predictions using:
#        - Base only   (pB)
#        - Fibre only  (pR)  – from fold-matched residuals
#        - Stacked     (pBR) – logistic on (pB, pR) within the same folds
#     2) Summarise performance:
#        - AUC (+ 95% CI, one-sided p against 0.5)
#        - ΔAUC (Both − Base) with bootstrap CI
#        - Paired DeLong p (if pROC available)
#        - AUPRC (+ prevalence baseline)
#     3) Export ROC coordinates and save neat plots.
#
# Inputs required (from previous sections):
#   Base        : n × m* embedding (m*=2; fixed by M_STAR_FIXED)
#   Xenc_w      : weighted, sqrt-scaled design (for Base OOF when using PCA)
#   E_scaled    : OOF residuals of encoded columns vs Base, standardised
#   diag_wide_full : long → wide diagnoses table with participant_id
#   CV_FOLDS, SEED_PRED, DX_PREV_MIN, DX_CASES_MIN, BASE_OOF_PCA
#
# Safety note:
#   Everything here is **out-of-fold**. No row is ever scored by a model that
#   saw it during training.

stopifnot(exists("Base"), exists("Xenc_w"), exists("E_scaled"),
          exists("diag_wide_full"), is.matrix(Base))

suppressPackageStartupMessages({
  if (!requireNamespace("pROC", quietly = TRUE)) {
    message("[dx] pROC not available — using robust fallbacks (no DeLong).")
  }
  if (!requireNamespace("ggrepel", quietly = TRUE)) NULL
})

# ---------- 12.1 Utilities (define here iff missing) --------------------------

# (a) Choose K folds conservatively given class sizes
choose_K <- get0("choose_K", ifnotfound = function(y, K_target = CV_FOLDS, min_per_class = 8){
  y <- as.integer(y); n1 <- sum(y == 1); n0 <- sum(y == 0)
  max(2L, min(K_target, floor(n1 / max(1L, min_per_class)), floor(n0 / max(1L, min_per_class))))
})

# (b) Stratified folds (optionally by group); here we only need class-stratification
make_strat_folds <- get0("make_strat_folds", ifnotfound = function(y, K, group = NULL, seed = 1){
  set.seed(seed); y <- as.integer(y)
  n <- length(y); folds <- integer(n)
  if (is.null(group)) {
    idx0 <- which(y == 0); idx1 <- which(y == 1)
    f0 <- sample(rep(1:K, length.out = length(idx0)))
    f1 <- sample(rep(1:K, length.out = length(idx1)))
    folds[idx0] <- f0; folds[idx1] <- f1
  } else {
    g <- as.factor(group)
    for (lev in levels(g)){
      ix <- which(g == lev); yix <- y[ix]
      idx0 <- ix[yix == 0]; idx1 <- ix[yix == 1]
      if (length(idx0)) folds[idx0] <- sample(rep(1:K, length.out = length(idx0)))
      if (length(idx1)) folds[idx1] <- sample(rep(1:K, length.out = length(idx1)))
    }
  }
  folds
})

# (c) Base OOF probabilities from **PCA within each training fold** (no leakage)
oof_prob_base_with_foldid <- get0("oof_prob_base_with_foldid", ifnotfound = function(y, Xenc, m, fold_id, seed=42, family="binomial"){
  set.seed(seed); y <- as.integer(y)
  P <- rep(NA_real_, length(y))
  for (k in sort(unique(fold_id))){
    tr <- which(fold_id != k); te <- which(fold_id == k)
    Xtr <- Xenc[tr,,drop=FALSE]; Xte <- Xenc[te,,drop=FALSE]
    vtr <- apply(Xtr, 2, stats::var, na.rm=TRUE); keep <- is.finite(vtr) & vtr > 1e-12
    if (!any(keep) || length(unique(y[tr])) < 2L || length(tr) < 5L){ P[te] <- mean(y[tr]); next }
    Xtr <- Xtr[,keep,drop=FALSE]; Xte <- Xte[,keep,drop=FALSE]
    mu <- colMeans(Xtr); sdv <- pmax(1, apply(Xtr,2,stats::sd))
    Ztr <- sweep(sweep(Xtr,2,mu,"-"),2,sdv,"/"); Zte <- sweep(sweep(Xte,2,mu,"-"),2,sdv,"/")
    m_eff <- max(1L, min(m, ncol(Ztr), length(tr)-1L))
    pc <- try(prcomp(Ztr, center=FALSE, scale.=FALSE, rank.=m_eff), silent=TRUE)
    if (inherits(pc,"try-error")){ P[te] <- mean(y[tr]); next }
    Btr <- pc$x[,1:m_eff,drop=FALSE]
    Bte <- as.matrix(Zte) %*% pc$rotation[,1:m_eff,drop=FALSE]
    fit <- try(glm(y[tr] ~ ., data=as.data.frame(Btr), family=family), silent=TRUE)
    P[te] <- if (inherits(fit,"try-error")) mean(y[tr]) else
      as.numeric(predict(fit, newdata=as.data.frame(Bte), type="response"))
  }
  pmin(pmax(P, 1e-6), 1-1e-6)
})

# (d) Simpler OOF probability using a penalised GLM on supplied features
oof_prob_with_foldid <- get0("oof_prob_with_foldid", ifnotfound = function(y, X, fold_id, seed=42, alpha=0, lambda=NULL){
  set.seed(seed); y <- as.integer(y); X <- as.matrix(X)
  p <- rep(NA_real_, length(y))
  for (k in sort(unique(fold_id))){
    tr <- which(fold_id != k); te <- which(fold_id == k)
    if (length(unique(y[tr])) < 2L || length(tr) < 5L){ p[te] <- mean(y[tr]); next }
    fit <- if (is.null(lambda))
      glmnet::glmnet(X[tr,,drop=FALSE], y[tr], family="binomial", alpha=alpha, standardize=TRUE, lambda=0.05)
    else
      glmnet::glmnet(X[tr,,drop=FALSE], y[tr], family="binomial", alpha=alpha, standardize=TRUE, lambda=lambda)
    p[te] <- as.numeric(predict(fit, newx=X[te,,drop=FALSE], type="response"))
  }
  pmin(pmax(p, 1e-6), 1-1e-6)
})

# (e) Build a **fold-matched Base embedding** for residualisation later
oof_base_components <- get0("oof_base_components", ifnotfound = function(Xenc, m, fold_id){
  n <- nrow(Xenc); B <- matrix(NA_real_, n, m); colnames(B) <- paste0("b", seq_len(m))
  for (k in sort(unique(fold_id))){
    tr <- which(fold_id != k); te <- which(fold_id == k)
    vtr  <- apply(Xenc[tr, , drop=FALSE], 2, stats::var, na.rm=TRUE)
    keep <- is.finite(vtr) & vtr > 1e-12
    Xtr  <- Xenc[tr, keep, drop=FALSE]; Xte <- Xenc[te, keep, drop=FALSE]
    mu   <- colMeans(Xtr); sdv <- apply(Xtr, 2, stats::sd); sdv[!is.finite(sdv)|sdv<1e-9] <- 1
    Ztr  <- sweep(sweep(Xtr,2,mu,"-"),2,sdv,"/"); Zte <- sweep(sweep(Xte,2,mu,"-"),2,sdv,"/")
    m_eff <- max(1L, min(m, ncol(Ztr), length(tr)-1L))
    pc <- prcomp(Ztr, center=FALSE, scale.=FALSE, rank.=m_eff)
    B[tr, 1:m_eff] <- as.matrix(Ztr) %*% pc$rotation[, 1:m_eff, drop=FALSE]
    B[te, 1:m_eff] <- as.matrix(Zte) %*% pc$rotation[, 1:m_eff, drop=FALSE]
    if (m_eff < m) { B[tr, (m_eff+1):m] <- 0; B[te, (m_eff+1):m] <- 0 }
  }
  B
})

# (f) Residualise encodings on Base with *matching folds*
residualise_foldsafe <- get0("residualise_foldsafe", ifnotfound = function(Xenc, Base, folds, k_gam = 6){
  n <- nrow(Base); V <- colnames(Xenc)
  E <- matrix(NA_real_, n, length(V), dimnames = list(rownames(Base), V))
  sm_terms <- paste0("s(b", seq_len(ncol(Base)), ",k=", k_gam, ")")
  for (v in V){
    z <- as.numeric(Xenc[, v])
    for (k in sort(unique(folds))){
      tr <- which(folds != k); te <- which(folds == k)
      dftr <- data.frame(v = z[tr], Base[tr, , drop = FALSE])
      fml  <- reformulate(sm_terms, response = "v")
      g <- try(mgcv::gam(fml, data = dftr, method = "REML"), silent = TRUE)
      if (inherits(g, "try-error")) next
      mu <- as.numeric(predict(g, newdata = data.frame(Base[te, , drop = FALSE]), type = "response"))
      E[te, v] <- z[te] - mu
    }
  }
  E[, colSums(is.finite(E)) > 0, drop = FALSE]
})

# (g) Fibre OOF probabilities: PCA on fold-matched residuals inside each train split
oof_prob_fibre_pca_foldsafe <- function(y, E, k_fibre = 3, fold_id, seed = 42){
  set.seed(seed); y <- as.integer(y); n <- length(y)
  P <- rep(NA_real_, n)
  
  # helper para limpar matrizes (NA/Inf -> 0)
  .clean <- function(M){
    M[!is.finite(M)] <- 0
    M
  }
  
  for (k in sort(unique(fold_id))){
    tr <- which(fold_id != k); te <- which(fold_id == k)
    
    # guarda-chuva para folds muito pequenos
    if (length(unique(y[tr])) < 2L || length(tr) < 5L) { 
      P[te] <- mean(y[tr]); 
      next 
    }
    
    # estatísticas com na.rm=TRUE
    mu  <- suppressWarnings(colMeans(E[tr, , drop = FALSE], na.rm = TRUE))
    sdv <- suppressWarnings(apply(E[tr, , drop = FALSE],  2, stats::sd, na.rm = TRUE))
    sdv[!is.finite(sdv) | sdv < 1e-9] <- 1
    
    Ztr <- sweep(sweep(E[tr, , drop = FALSE], 2, mu, "-"), 2, sdv, "/")
    Zte <- sweep(sweep(E[te, , drop = FALSE], 2, mu, "-"), 2, sdv, "/")
    
    # limpa NA/Inf criados pelo centering/scaling
    Ztr <- .clean(Ztr); Zte <- .clean(Zte)
    
    rk  <- max(1L, min(k_fibre, ncol(Ztr), length(tr) - 1L))
    if (rk < 1L) { P[te] <- mean(y[tr]); next }
    
    pc <- try(prcomp(Ztr, center = FALSE, scale. = FALSE, rank. = rk), silent = TRUE)
    if (inherits(pc, "try-error")) { P[te] <- mean(y[tr]); next }
    
    XR_tr <- as.matrix(Ztr) %*% pc$rotation[, 1:rk, drop = FALSE]
    XR_te <- as.matrix(Zte) %*% pc$rotation[, 1:rk, drop = FALSE]
    
    fit <- try(stats::glm(y[tr] ~ ., data = as.data.frame(XR_tr), family = binomial()), silent = TRUE)
    P[te] <- if (inherits(fit, "try-error")) mean(y[tr]) else
      as.numeric(stats::predict(fit, newdata = as.data.frame(XR_te), type = "response"))
  }
  pmin(pmax(P, 1e-6), 1 - 1e-6)
}

# (h) Stack meta-combiner (OOF-safe) with repeat capability
oof_prob_stacked_strict <- function(y, Xb, XR = NULL, K = 5L, R = 1L,
                                    seed = 42, min_test_pos = 2L, min_test_neg = 2L){
  clip <- function(z) pmin(pmax(as.numeric(z), 1e-6), 1-1e-6)
  y  <- as.integer(y > 0); Xb <- as.data.frame(Xb)
  if (!is.null(XR)) XR <- as.data.frame(XR)
  N <- length(y)
  acc <- rep(0, N)
  
  # helper: build “good” stratified folds
  make_ok_folds <- function(r){
    tries <- 0L
    repeat {
      fid <- make_strat_folds(y, K = K, seed = seed + 1000L*r + tries)
      ok  <- TRUE
      for (k in seq_len(K)) {
        te <- (fid == k)
        if (sum(y[te]==1L) < min_test_pos || sum(y[te]==0L) < min_test_neg) { ok <- FALSE; break }
      }
      if (ok || tries >= 50L) return(fid)
      tries <- tries + 1L
    }
  }
  
  for (r in seq_len(R)) {
    fid <- make_ok_folds(r)
    out <- rep(NA_real_, N)
    
    if (is.null(XR)) {
      for (k in seq_len(K)){
        tr <- fid != k; te <- fid == k
        m <- .fit_prob_model(y[tr], Xb[tr,,drop=FALSE])
        out[te] <- .pred_prob(m, Xb[te,,drop=FALSE])
      }
      acc <- acc + clip(out)
      next
    }
    
    # stacked two-source case
    for (k in seq_len(K)){
      tr <- fid != k; te <- fid == k
      m1 <- .fit_prob_model(y[tr], Xb[tr,,drop=FALSE])
      m2 <- .fit_prob_model(y[tr], XR[tr,,drop=FALSE])
      p1 <- clip(.pred_prob(m1, Xb[tr,,drop=FALSE]))
      p2 <- clip(.pred_prob(m2, XR[tr,,drop=FALSE]))
      s  <- stats::glm(y[tr] ~ qlogis(p1) + qlogis(p2), family = binomial())
      p1e <- clip(.pred_prob(m1, Xb[te,,drop=FALSE]))
      p2e <- clip(.pred_prob(m2, XR[te,,drop=FALSE]))
      out[te] <- as.numeric(stats::predict(s,
                                           newdata = data.frame(l1=qlogis(p1e), l2=qlogis(p2e)),
                                           type="response"))
    }
    acc <- acc + clip(out)
  }
  clip(acc / R)
}

# (i) Metrics helpers
AUC <- function(y, p){
  y <- as.integer(y)
  if (!requireNamespace("pROC", quietly = TRUE)) {
    # Mann–Whitney U / Wilcoxon as AUC
    r <- rank(p); n1 <- sum(y==1); n0 <- sum(y==0)
    if (n1==0 || n0==0) return(NA_real_)
    (sum(r[y==1]) - n1*(n1+1)/2)/(n0*n1)
  } else {
    as.numeric(pROC::roc(y, p, quiet = TRUE)$auc)
  }
}

auc_ci_p <- function(y, p, B=1000, seed=42){
  y <- as.integer(y)
  ok <- is.finite(p)
  if (sum(y[ok]==1)==0 || sum(y[ok]==0)==0) return(c(auc=NA, lo=NA, hi=NA, p=NA))
  # if predictor is (nearly) constant → define AUC=0.5 and neutral CI/p
  if (length(unique(round(p[ok], 6))) <= 1L)
    return(c(auc=0.5, lo=0.5, hi=0.5, p=1))
  if (!requireNamespace("pROC", quietly = TRUE)) {
    set.seed(seed)
    idx0 <- which(y==0); idx1 <- which(y==1)
    au <- AUC(y, p)
    boots <- replicate(B, {
      s0 <- sample(idx0, length(idx0), replace=TRUE)
      s1 <- sample(idx1, length(idx1), replace=TRUE)
      AUC(y[c(s0,s1)], p[c(s0,s1)])
    })
    p_wmw <- tryCatch(stats::wilcox.test(p[y==1], p[y==0], alternative="greater")$p.value, error=function(e) 1)
    return(c(auc=au, lo=quantile(boots,.025,na.rm=TRUE), hi=quantile(boots,.975,na.rm=TRUE), p=p_wmw))
  } else {
    r  <- pROC::roc(y, p, quiet=TRUE, direction="<")
    ci <- try(pROC::ci.auc(r, conf.level=0.95), silent=TRUE)
    if (inherits(ci,"try-error")) ci <- c(NA, as.numeric(r$auc), NA)
    p_wmw <- tryCatch(stats::wilcox.test(p[y==1], p[y==0], alternative="greater")$p.value, error=function(e) 1)
    return(c(auc=as.numeric(r$auc), lo=as.numeric(ci[1]), hi=as.numeric(ci[3]), p=p_wmw))
  }
}

delong_p <- function(y, p1, p2){
  if (!requireNamespace("pROC", quietly = TRUE)) return(NA_real_)
  tryCatch({
    r1 <- pROC::roc(y, p1, quiet = TRUE, direction = "<")
    r2 <- pROC::roc(y, p2, quiet = TRUE, direction = "<")
    as.numeric(pROC::roc.test(r1, r2, paired = TRUE, method = "delong")$p.value)
  }, error = function(e) NA_real_)
}

boot_ci_delta <- function(y, p1, p2, B = 400, seed = 42){
  set.seed(seed); y <- as.integer(y)
  idx0 <- which(y==0); idx1 <- which(y==1)
  if (!length(idx0) || !length(idx1)) return(c(NA_real_, NA_real_))
  deltas <- replicate(B, {
    s0 <- sample(idx0, length(idx0), replace = TRUE)
    s1 <- sample(idx1, length(idx1), replace = TRUE)
    AUC(y[c(s0,s1)], p2[c(s0,s1)]) - AUC(y[c(s0,s1)], p1[c(s0,s1)])
  })
  stats::quantile(na.omit(deltas), c(0.025, 0.975), na.rm = TRUE)
}

auprc_simple <- function(y, p){
  o <- order(p, decreasing = TRUE)
  y <- y[o]; p <- p[o]
  tp <- cumsum(y==1); fp <- cumsum(y==0)
  P  <- sum(y==1); N <- sum(y==0)
  if (P == 0 || N == 0) return(c(AUPRC = NA_real_, baseline = NA_real_))
  recall <- tp / P
  precision <- tp / pmax(tp + fp, 1)
  # right-continuous interpolation
  auprc <- sum(diff(c(0, recall)) * zoo::na.locf0(precision, fromLast = TRUE))
  c(AUPRC = auprc, baseline = P/(P+N))
}

make_fname_safe <- function(s) {
  s <- iconv(s, to = "UTF-8", sub = "_")
  s <- gsub("[/\\\\:*?\"<>|]+", "_", s)
  s <- gsub("\\s+", "_", s)
  s <- gsub("_+", "_", s)
  gsub("^_+|_+$", "", s)
}

export_roc_coords <- function(y, p, name, dir = "roc_csv"){
  if (!requireNamespace("pROC", quietly = TRUE)) return(invisible(NULL))
  dir.create(dir, showWarnings = FALSE)
  y <- as.integer(y); ok <- is.finite(p)
  if (sum(ok & y == 1) == 0 || sum(ok & y == 0) == 0) return(invisible(NULL))
  r <- pROC::roc(y[ok], p[ok], quiet=TRUE, direction="<")
  C <- pROC::coords(r, x = "all",
                    ret = c("threshold","specificity","sensitivity"),
                    transpose = FALSE)
  safe_name <- make_fname_safe(name)
  utils::write.csv(C, file = file.path(dir, paste0("roc_", safe_name, ".csv")), row.names = FALSE)
  invisible(C)
}

# ---------- 12.2 Build per-diagnosis labels aligned to Base rows --------------

mm  <- match(rownames(Base), diag_wide_full$participant_id)
DxW <- as.data.frame(diag_wide_full[mm, setdiff(names(diag_wide_full), "participant_id"), drop = FALSE])
DxW[is.na(DxW)] <- 0L

# --- normalize DxW column names BEFORE screening ---
norm_dx <- function(x){
  x <- gsub("\\s+", " ", trimws(x))
  x <- sub("^no\\s*diagnosis.*$", "No Diagnosis on Axis I", x, ignore.case = TRUE)
  x <- sub("^schizoaffective.*$", "Schizoaffective Disorder", x, ignore.case = TRUE)
  x <- sub("^ocd\\b|obsessive\\s*-?\\s*compulsive.*$", "Obsessive-Compulsive Disorder", x, ignore.case = TRUE)
  x
}
names(DxW) <- norm_dx(names(DxW))


prev  <- colMeans(DxW > 0, na.rm = TRUE)
cases <- colSums(DxW > 0, na.rm = TRUE)
keep_dx <- names(prev)[(prev >= DX_PREV_MIN) & (cases >= DX_CASES_MIN)]
message("[dx] cases: ", paste(sprintf("%s=%d", names(cases), cases), collapse=", "))
if (!length(keep_dx)) {
  warning("[dx] No diagnosis passes prevalence/case thresholds; skipping predictive diagnostics.")
} else {
  message(sprintf("[dx] Evaluating %d diagnoses (prevalence ≥ %.2f, cases ≥ %d).",
                  length(keep_dx), DX_PREV_MIN, DX_CASES_MIN))
}


# ---- sanity: force multisession on RStudio/macOS and verify workers ----------
suppressPackageStartupMessages({
  library(future); library(future.apply); library(parallelly)
})

# 1) RStudio often reports supportsMulticore() == FALSE, but be explicit anyway.
#    For a stable experience on macOS/RStudio: always multisession here.
N_WORKERS <- get0("NCORES_PAR", ifnotfound = parallelly::availableCores(omit = 1L))
future::plan(future::multisession, workers = N_WORKERS)

# 2) Pin BLAS/OMP threads in the **main** session *before* workers spawn
Sys.setenv(
  OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
  VECLIB_MAXIMUM_THREADS="1", BLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
)

# 3) Quick probes: you should see >1 workers and distinct PIDs
cat(sprintf("[future] plan=%s | workers=%d\n", format(future::plan()), future::nbrOfWorkers()))
pids <- future.apply::future_lapply(seq_len(min(N_WORKERS, 8)), function(i){
  list(pid = Sys.getpid(), has_mgcv = requireNamespace("mgcv", quietly = TRUE))
}, future.seed = TRUE)
print(pids)
# ---------- 12.3 Loop per diagnosis: OOF predictions e métricas (PARALELO) ----

BASE_OOF_PCA <- get0("BASE_OOF_PCA", ifnotfound = TRUE)
m_star       <- if (exists("M_STAR_FIXED")) as.integer(M_STAR_FIXED) else 2L
R            <- get0("CV_REPEATS",   ifnotfound = 1L)
MIN_TEST_POS <- get0("MIN_TEST_POS", ifnotfound = 3L)
MIN_TEST_NEG <- get0("MIN_TEST_NEG", ifnotfound = 3L)

.make_ok_folds <- function(y, K, seed, min_pos = 2L, min_neg = 2L, max_tries = 50L){
  y <- as.integer(y); tries <- 0L
  repeat {
    fid <- make_strat_folds(y, K = K, seed = seed + 1000L*tries)
    ok <- TRUE
    for (k in seq_len(K)) {
      te <- (fid == k)
      if (sum(y[te] == 1L) < min_pos || sum(y[te] == 0L) < min_neg) { ok <- FALSE; break }
    }
    if (ok || tries >= max_tries) return(fid)
    tries <- tries + 1L
  }
}


# exact stratified K-folds: distribute cases as evenly as possible (e.g., 4/4/3)
make_strat_folds_exact <- function(y, K, seed = 1){
  set.seed(seed)
  f <- integer(length(y))
  i1 <- which(y==1); i0 <- which(y==0)
  i1 <- sample(i1);  i0 <- sample(i0)
  f[i1] <- rep_len(1:K, length(i1))
  f[i0] <- rep_len(1:K, length(i0))
  f
}

# >>> ALTERAÇÃO: future_lapply em vez de lapply; future.seed=TRUE para RNG estável
res <- future.apply::future_lapply(
  keep_dx,
  FUN = function(dx_name){
    
    # --- prolog in worker
    suppressPackageStartupMessages({
      library(glmnet); library(mgcv)
      if (!requireNamespace("pROC", quietly = TRUE)) NULL
    })
    Sys.setenv(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", BLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    
    y <- as.integer(DxW[[dx_name]] > 0)
    n  <- length(y); 
    n1 <- sum(y==1); 
    n0 <- sum(y==0)
    
    # keep dx if enough positives and at least one negative
    if (n1 < DX_CASES_MIN || n0 == 0) return(NULL)
    
    # choose the largest feasible K; with n1=11 & MIN_TEST_POS=3 ⇒ K_max = 3
    K_max_pos <- floor(n1 / max(1L, MIN_TEST_POS))
    K_max_neg <- floor(n0 / max(1L, MIN_TEST_NEG))
    K <- max(2L, min(CV_FOLDS, K_max_pos, K_max_neg))
    
    fold_id_r <- make_strat_folds_exact(y, K, seed = SEED_PRED + 123)
    
    # verify per-fold minima; if any fold fails, fall back to shuffle-OOF
    ok_pos <- min(tabulate(fold_id_r[y==1], nbins = K)) >= MIN_TEST_POS
    ok_neg <- min(tabulate(fold_id_r[y==0], nbins = K)) >= MIN_TEST_NEG
    use_shuffle <- !(ok_pos && ok_neg)
    
    clip <- function(z) pmin(pmax(as.numeric(z), 1e-6), 1-1e-6)
    
    R_KFOLD <- max(1L, get0("CV_REPEATS", ifnotfound = 1L))
    if (!use_shuffle) {
      pB_acc <- pR_acc <- rep(0, n)
      for (rpt in seq_len(R_KFOLD)) {
        fold_id_r <- make_strat_folds_exact(y, K, seed = SEED_PRED + 123 + 1000L*rpt)
        
        # --- classic K-fold path using fold_id_r ---
        pB_r <- if (isTRUE(BASE_OOF_PCA)) {
          oof_prob_base_with_foldid(y, Xenc_w, m = m_star, fold_id = fold_id_r, seed = SEED_PRED + rpt)
        } else {
          oof_prob_with_foldid(y, Base, fold_id = fold_id_r, seed = SEED_PRED + rpt)
        }
        
        B_oof_r <- oof_base_components(Xenc_w, m = m_star, fold_id = fold_id_r)
        E_dx_r  <- residualise_foldsafe(Xenc_w, B_oof_r, folds = fold_id_r, k_gam = 6)
        K_FIBRE_CAP <- max(1L, min(3L, ncol(E_dx_r), nrow(E_dx_r) - 1L))
        
        pR_r <- oof_prob_fibre_pca_foldsafe(
          y, E_dx_r, k_fibre = K_FIBRE_CAP, fold_id = fold_id_r, seed = SEED_PRED + rpt
        )
        
        # >>> accumulate <<<
        pB_acc <- pB_acc + clip(pB_r)
        pR_acc <- pR_acc + clip(pR_r)
      }
      # set the outputs actually used later
      pB <- pB_acc / R_KFOLD
      pR <- pR_acc / R_KFOLD
    } else {
      # --- your shuffle-OOF fallback path from earlier suggestion ---
      # (keeps ≥ MIN_TEST_POS/NEG in each holdout; averages per-subject across repeats)
      # -------------------- Shuffle-OOF (repeated stratified holdout) --------------------
      TEST_PROP    <- 0.25                          # ~25% test each repeat
      R_SHUF       <- max(40L, ceiling(2000 / max(1L, n1)))  # more repeats for rarer dx
      pB_sum <- pR_sum <- rep(0, n)
      n_seen <- rep(0L, n)
      
      for (r in seq_len(R_SHUF)) {
        set.seed(SEED_PRED + 10000L + r)
        
        # stratified test sample with hard minima
        pick_test <- function(){
          # proportional sampling by class
          idx0 <- which(y==0); idx1 <- which(y==1)
          te1  <- sample(idx1, max(MIN_TEST_POS, ceiling(length(idx1)*TEST_PROP)), replace = length(idx1)*TEST_PROP < MIN_TEST_POS)
          te0  <- sample(idx0, max(MIN_TEST_NEG, ceiling(length(idx0)*TEST_PROP)), replace = length(idx0)*TEST_PROP < MIN_TEST_NEG)
          sort(unique(c(te1, te0)))
        }
        te <- pick_test(); tries <- 0L
        while ((sum(y[te]==1) < MIN_TEST_POS || sum(y[te]==0) < MIN_TEST_NEG) && tries < 50L) {
          te <- pick_test(); tries <- tries + 1L
        }
        if (sum(y[te]==1) < MIN_TEST_POS || sum(y[te]==0) < MIN_TEST_NEG) next  # skip this repeat if impossible
        
        # Build a 2-fold label: fold 1 = test, fold 2 = train; OOF funcs will respect it
        fold_id_r <- rep.int(2L, n); fold_id_r[te] <- 1L
        
        pB_r <- if (isTRUE(BASE_OOF_PCA)) {
          oof_prob_base_with_foldid(y, Xenc_w, m = m_star, fold_id = fold_id_r, seed = SEED_PRED + r)
        } else {
          oof_prob_with_foldid(y, Base, fold_id = fold_id_r, seed = SEED_PRED + r)
        }
        
        B_oof_r <- oof_base_components(Xenc_w, m = m_star, fold_id = fold_id_r)
        E_dx_r  <- residualise_foldsafe(Xenc_w, B_oof_r, folds = fold_id_r, k_gam = 6)
        K_FIBRE_CAP <- max(1L, min(3L, ncol(E_dx_r), nrow(E_dx_r) - 1L))
        
        pR_r <- oof_prob_fibre_pca_foldsafe(y, E_dx_r, k_fibre = K_FIBRE_CAP,
                                            fold_id = fold_id_r, seed = SEED_PRED + r)
        
        # accumulate only test rows from this repeat (strict OOF)
        pB_sum[te] <- pB_sum[te] + pB_r[te]
        pR_sum[te] <- pR_sum[te] + pR_r[te]
        n_seen[te] <- n_seen[te] + 1L
      }
      
      # Final OOF by averaging over repeats per subject
      pB <- rep(NA_real_, n); pR <- rep(NA_real_, n)
      ok <- n_seen > 0
      pB[ok] <- clip(pB_sum[ok] / n_seen[ok])
      pR[ok] <- clip(pR_sum[ok] / n_seen[ok])
      
      # if any subjects never landed in test (rare), backfill with a light K-fold
      if (any(!ok)) {
        fold_id_fix <- .make_ok_folds(y, K = max(2L, min(3L, n1)), seed = SEED_PRED + 9999,
                                      min_pos = MIN_TEST_POS, min_neg = MIN_TEST_NEG)
        pB_fix <- oof_prob_base_with_foldid(y, Xenc_w, m = m_star, fold_id = fold_id_fix, seed = SEED_PRED + 9999)
        B_fix  <- oof_base_components(Xenc_w, m = m_star, fold_id = fold_id_fix)
        E_fix  <- residualise_foldsafe(Xenc_w, B_fix, folds = fold_id_fix, k_gam = 6)
        kcap   <- max(1L, min(3L, ncol(E_fix), nrow(E_fix) - 1L))
        pR_fix <- oof_prob_fibre_pca_foldsafe(y, E_fix, k_fibre = kcap, fold_id = fold_id_fix, seed = SEED_PRED + 9999)
        pB[!ok] <- clip(pB_fix[!ok]); pR[!ok] <- clip(pR_fix[!ok])
      }
    }
    
    # Meta-stacking on OOF probabilities (unchanged)
    fold_id_meta <- make_strat_folds(y, K = max(2L, min(5L, floor(n1 / max(1L, MIN_TEST_POS)))),
                                     seed = SEED_PRED + 999)
    pBR <- stack_with_foldid(y, pB, pR, fold_id = fold_id_meta, seed = SEED_PRED)
    
    # Metrics + exports (unchanged)
    mB  <- auc_ci_p(y, pB);   mR  <- auc_ci_p(y, pR);   mBR <- auc_ci_p(y, pBR)
    ci_d <- boot_ci_delta(y, pB, pBR, B = 400, seed = SEED_PRED)
    p_dl <- delong_p(y, pB, pBR)
    
    auB  <- auprc_simple(y, pB);  auR <- auprc_simple(y, pR);  auBR <- auprc_simple(y, pBR)
    
    export_roc_coords(y, pB,  paste0(dx_name, "_Base"))
    export_roc_coords(y, pR,  paste0(dx_name, "_Fibre"))
    export_roc_coords(y, pBR, paste0(dx_name, "_Both"))
    
    data.frame(
      dx = dx_name, n1 = n1, n0 = n0,
      AUC_Base  = mB["auc"],  AUC_Base_lo  = mB["lo"],  AUC_Base_hi  = mB["hi"],  p_Base  = mB["p"],
      AUC_Fibre = mR["auc"],  AUC_Fibre_lo = mR["lo"],  AUC_Fibre_hi = mR["hi"],  p_Fibre = mR["p"],
      AUC_Both  = mBR["auc"], AUC_Both_lo  = mBR["lo"], AUC_Both_hi = mBR["hi"], p_Both  = mBR["p"],
      AUPRC_Base   = unname(auB["AUPRC"]),  AUPRC_Base_baseline   = unname(auB["baseline"]),
      AUPRC_Fibre  = unname(auR["AUPRC"]),  AUPRC_Fibre_baseline  = unname(auR["baseline"]),
      AUPRC_Both   = unname(auBR["AUPRC"]), AUPRC_Both_baseline   = unname(auBR["baseline"]),
      dAUC_Base_to_Both = as.numeric(mBR["auc"] - mB["auc"]),
      dAUC_lo = ci_d[1], dAUC_hi = ci_d[2],
      p_delong = p_dl,
      stringsAsFactors = FALSE
    )
  },
  future.seed = TRUE   # RNG reprodutível por worker
)

# voltar ao plano sequencial no fim (boa higiene)
future::plan(future::sequential)

dx_auc <- dplyr::bind_rows(Filter(Negate(is.null), res))

if (!nrow(dx_auc)) {
  warning("[dx] No usable diagnoses after gating/folds; skipping plots/CSVs.")
} else {
  # Robust q-values only where p’s are finite
  for (pcol in c("p_Base","p_Fibre","p_Both","p_delong")) {
    if (!(pcol %in% names(dx_auc))) dx_auc[[pcol]] <- NA_real_
    qcol <- sub("^p_", "q_", pcol)
    qvec <- rep(NA_real_, nrow(dx_auc))
    idx  <- which(is.finite(dx_auc[[pcol]]))
    if (length(idx)) qvec[idx] <- p.adjust(dx_auc[[pcol]][idx], method = "BH")
    dx_auc[[qcol]] <- qvec
  }
  
  dx_auc <- dplyr::arrange(dx_auc, dplyr::desc(dAUC_Base_to_Both))
  readr::write_csv(dx_auc, "dx_auc_base_fibre_stacked_with_pr.csv")
  print(utils::head(dx_auc, 50))
  cat("Saved: dx_auc_base_fibre_stacked_with_pr.csv  and roc_csv/*.csv (if pROC present)\n")
  
  # ---------- 12.4 Plots ------------------------------------------------------
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    
    # ---- Canonical ordering only (no relabelling at all) ----
    desired_order <- c(
      "No Diagnosis on Axis I",
      "Attention-Deficit/Hyperactivity Disorder",
      "Bipolar I Disorder",
      "Depressive Disorder NOS",
      "Major Depressive Disorder",
      "Schizophrenia",
      "Schizoaffective Disorder",
      "Alcohol Abuse",
      "Alcohol Dependence",
      "Amphetamine Abuse",
      "Amphetamine Dependence",
      "Cannabis Abuse",
      "Cannabis Dependence",
      "Cocaine Abuse",
      "Cocaine Dependence"
    )
    
    dx_present <- unique(dx_auc$dx)
    ord_levels <- rev(c(intersect(desired_order, dx_present),
                    setdiff(dx_present, desired_order)))
    
    dx_auc <- dx_auc %>%
      dplyr::mutate(dx = factor(dx, levels = ord_levels))
    
    # Palette for significance
    pal_sig <- c("ns" = "#D06464", "q<0.05" = "#1AA6B7")
    
    # (A) ΔAUC with 95% bootstrap CI
    plot_df <- dx_auc %>%
      dplyr::arrange(dAUC_Base_to_Both) %>%
      dplyr::mutate(sig = ifelse(is.finite(q_delong) & q_delong < 0.05, "q<0.05", "ns"))
    
    p1 <- ggplot2::ggplot(
      plot_df,
      ggplot2::aes(x = dx, y = dAUC_Base_to_Both, ymin = dAUC_lo, ymax = dAUC_hi, colour = sig)
    ) +
      ggplot2::geom_hline(yintercept = 0, linetype = 2) +
      ggplot2::geom_pointrange(position = ggplot2::position_dodge(width = 0.3)) +
      ggplot2::coord_flip() +
      ggplot2::scale_colour_manual(values = pal_sig, breaks = names(pal_sig), name = NULL) +
      ggplot2::labs(title = "ΔAUC (Base → Base+Fibre) with 95% CI",
                    x = "Diagnosis", y = "ΔAUC (Both - Base)") +
      ggplot2::theme_minimal(base_size = 15) +
      ggplot2::theme(
        panel.grid.minor = ggplot2::element_blank(),
        axis.title.y = ggplot2::element_text(margin = ggplot2::margin(r = 6)),
        legend.position = "right"
      )
    print(p1)
    ggplot2::ggsave("FIG_dAUC_base_to_both.png", p1, width = 8, height = 10, dpi = 150)
    
    # (B) AUC(Base) vs AUC(Fibre)
    p2 <- ggplot2::ggplot(dx_auc, ggplot2::aes(AUC_Base, AUC_Fibre, label = dx)) +
      ggplot2::geom_abline(slope = 1, intercept = 0, linetype = 2) +
      ggplot2::geom_point() +
      {if (requireNamespace("ggrepel", quietly = TRUE))
        ggrepel::geom_text_repel(size = 3, max.overlaps = 20)
        else ggplot2::geom_text(check_overlap = TRUE, size = 3)} +
      ggplot2::coord_equal(xlim = c(0.4, 1.0), ylim = c(0.4, 1.0)) +
      ggplot2::labs(title = "AUC(Base) vs AUC(Fibre) by diagnosis",
                    x = "AUC(Base)", y = "AUC(Fibre)") +
      ggplot2::theme_minimal(base_size = 14)
    print(p2)
    ggplot2::ggsave("FIG_auc_base_vs_fibre.png", p2, width = 6, height = 5, dpi = 150)
    
    # (C) Absolute AUCs per model: horizontal dot–whisker
    abs_long <- dplyr::bind_rows(
      dx_auc %>% dplyr::transmute(dx, model = "Base",
                                  AUC = as.numeric(AUC_Base),
                                  lo  = as.numeric(AUC_Base_lo),
                                  hi  = as.numeric(AUC_Base_hi),
                                  q   = as.numeric(q_Base)),
      dx_auc %>% dplyr::transmute(dx, model = "Fibre",
                                  AUC = as.numeric(AUC_Fibre),
                                  lo  = as.numeric(AUC_Fibre_lo),
                                  hi  = as.numeric(AUC_Fibre_hi),
                                  q   = as.numeric(q_Fibre)),
      dx_auc %>% dplyr::transmute(dx, model = "Both",
                                  AUC = as.numeric(AUC_Both),
                                  lo  = as.numeric(AUC_Both_lo),
                                  hi  = as.numeric(AUC_Both_hi),
                                  q   = as.numeric(q_Both))
    ) %>%
      dplyr::mutate(sig = ifelse(is.finite(q) & q < 0.05, "q<0.05", "ns"),
                    dx  = factor(dx, levels = ord_levels),
                    model = factor(model, levels = c("Base","Fibre","Both"))) %>%
      dplyr::filter(is.finite(AUC))
    
    abs_long_pts <- abs_long %>% dplyr::filter(is.finite(AUC))
    abs_long_cis <- abs_long_pts %>% dplyr::filter(is.finite(lo), is.finite(hi))
    
    x_min <- max(0, floor(10 * min(c(abs_long_pts$AUC, abs_long_cis$lo), na.rm = TRUE)) / 10)
    x_max <- min(1, ceiling(10 * max(c(abs_long_pts$AUC, abs_long_cis$hi), na.rm = TRUE)) / 10)
    
    p3 <- ggplot2::ggplot(
      abs_long_pts,
      ggplot2::aes(x = AUC, y = dx, xmin = lo, xmax = hi, colour = sig)
    ) +
      ggplot2::geom_vline(xintercept = 0.5, linetype = 2, linewidth = 0.4) +
      ggplot2::geom_errorbarh(data = abs_long_cis, height = 0, linewidth = 0.7, alpha = 0.9) +
      ggplot2::geom_point(size = 2.6, stroke = 0) +
      ggplot2::facet_grid(. ~ model, scales = "free_x") +
      ggplot2::scale_colour_manual(values = pal_sig, breaks = names(pal_sig), name = NULL) +
      ggplot2::scale_x_continuous(
        breaks = seq(x_min, x_max, by = 0.1),
        labels = function(b) sprintf("%.1f", b),
        expand = ggplot2::expansion(mult = c(0.02, 0.02))
      ) +
      ggplot2::coord_cartesian(xlim = c(x_min, x_max)) +
      ggplot2::labs(x = NULL, y = NULL) +
      ggplot2::theme_minimal(base_size = 16) +
      ggplot2::theme(
        panel.grid.major.y = ggplot2::element_blank(),
        panel.grid.minor   = ggplot2::element_blank(),
        strip.background   = ggplot2::element_blank(),
        strip.text.x       = ggplot2::element_text(face = "italic", size = 18),
        axis.text.y        = ggplot2::element_text(size = 13, lineheight = 0.9),
        axis.text.x        = ggplot2::element_text(size = 11),
        panel.spacing.x    = grid::unit(16, "pt"),
        legend.position    = "right"
      )
    print(p3)
    ggplot2::ggsave("FIG_auc_absolute_by_model.png", p3, width = 11, height = 4.8, dpi = 300)
    
    cat("Saved: FIG_dAUC_base_to_both.png, FIG_auc_base_vs_fibre.png, FIG_auc_absolute_by_model.png\n")
  }
}

# ==================== 13) Fibre self-decomposition per item ====================
# Prereqs from earlier:
#   E_scaled  : OOF residuals of encoded columns vs Base (standardised)
#   varmap    : map encoded columns → original item
#   vars      : unique(varmap) (survivor item names)
#   Bprime    : fibre base (n × m_f) from Section 11
#   nb_list   : kNN sets on Base_w (or Bprime; we’ll build a Bprime_kNN too)
#   PC_contrib: item’s contribution to Base (context only; not used in stats)
#
# Output:
#   CSV 'fibre_self_decomp_items.csv' with per-item:
#     R2_fibre_smooth, p_smooth, R2_fibre_local, p_local, k_local, role_fibre
#   Optional plots.

stopifnot(exists("E_scaled"), exists("Bprime"), is.matrix(E_scaled), is.matrix(Bprime))

# -- 13.1 Build kNN on B′ (whiten first to balance axes), small grid of k -------
S_Bp      <- cov(Bprime)
U_Bp      <- chol(S_Bp + diag(1e-8, ncol(Bprime)))
Bprime_w  <- Bprime %*% solve(U_Bp)

KS_FIBRE  <- get0("KS_FIBRE", ifnotfound = c(6,8,10,12,16,20))
nb_list_Bp <- setNames(lapply(KS_FIBRE, function(k){
  RANN::nn2(Bprime_w, Bprime_w, k = pmin(k+1L, nrow(Bprime_w)))$nn.idx[, -1L, drop = FALSE]
}), paste0("k", KS_FIBRE))

# -- 13.2 Helper: make per-item fibre residual series e_item --------------------
# Uses the same projection rule we used before: PC1 of this item’s encoded block
# applied to the residual matrix E_scaled (OOF).
e_from_E <- get0("e_from_E", ifnotfound = function(nm, E_scaled, Z_for_pc1, varmap){
  idx <- which(varmap == nm)
  if (!length(idx)) return(rep(NA_real_, nrow(Z_for_pc1)))
  if (length(idx) == 1L) return(as.numeric(E_scaled[, idx]))
  pc1 <- try(prcomp(Z_for_pc1[, idx, drop = FALSE], rank. = 1), silent = TRUE)
  if (inherits(pc1, "try-error")) return(rep(NA_real_, nrow(Z_for_pc1)))
  as.numeric(as.matrix(E_scaled[, idx, drop = FALSE]) %*% pc1$rotation[, 1])
})

# We want PC1 directions learned on *unweighted*, standardised encodings:
Z0_std <- scale(Xenc, center = TRUE, scale = TRUE)

# -- 13.3 Smooth (global) fibre effect via GAM on B′ ---------------------------
# Fit v ~ s(f1) + s(f2) + ... on the *entire* series (no fold split needed here,
# because E_scaled is already OOF). We still keep it simple and stable.
fibre_smooth_R2_p <- function(v, Bprime, k_gam = 6, B = 300, seed = 42){
  set.seed(seed)
  if (!any(is.finite(v))) return(c(R2 = NA_real_, p = NA_real_))
  df  <- data.frame(v = as.numeric(v), Bprime)
  smt <- paste0("s(", colnames(Bprime), ",k=", k_gam, ")")
  fml <- reformulate(smt, response = "v")
  fit <- try(mgcv::gam(fml, data = df, method = "REML"), silent = TRUE)
  if (inherits(fit, "try-error")) return(c(R2 = NA_real_, p = NA_real_))
  r   <- residuals(fit, type = "response")
  ve  <- stats::var(df$v, na.rm = TRUE)
  R2o <- if (is.finite(ve) && ve > 0) max(0, 1 - mean(r^2, na.rm = TRUE)/ve) else NA_real_
  if (!is.finite(R2o)) return(c(R2 = NA_real_, p = NA_real_))
  
  # permutation null: shuffle rows of B′ together (preserves B′ structure)
  exceed <- 0L; n <- nrow(Bprime)
  for (b in seq_len(B)){
    ix  <- sample.int(n)
    dfb <- data.frame(v = df$v, Bprime = Bprime[ix, , drop = FALSE])
    colnames(dfb) <- c("v", colnames(Bprime))
    fitb <- try(mgcv::gam(fml, data = dfb, method = "REML"), silent = TRUE)
    if (inherits(fitb, "try-error")) next
    rb <- residuals(fitb, type = "response")
    R2b <- max(0, 1 - mean(rb^2, na.rm = TRUE)/ve)
    if (is.finite(R2b) && R2b >= R2o) exceed <- exceed + 1L
  }
  p <- (exceed + 1) / (B + 1)
  c(R2 = R2o, p = p)
}

# --- plano paralelo outra vez (igual ao da Secção 12) -------------------------
suppressPackageStartupMessages({
  if (!requireNamespace("future.apply", quietly = TRUE)) install.packages("future.apply")
  if (!requireNamespace("future",        quietly = TRUE)) install.packages("future")
  if (!requireNamespace("parallelly",    quietly = TRUE)) install.packages("parallelly")
})
library(future); library(future.apply)

N_WORKERS <- get0("NCORES_PAR", ifnotfound = parallelly::availableCores(omit = 1))
if (future::supportsMulticore()) {
  future::plan(multicore,    workers = N_WORKERS)
} else {
  future::plan(multisession, workers = N_WORKERS)
}
options(future.rng.onMisuse = "error")

# -- 13.4 Local fibre effect: neighbour-mean CV R² on residual-of-smooth -------
r2_fibre_cv    <- get0("r2_fibre_cv")
choose_k_nb    <- get0("choose_k_nb")
r2_fibre_perm_rowshuffle <- get0("r2_fibre_perm_rowshuffle")

fibre_rows <- future.apply::future_lapply(
  vars,
  FUN = function(nm){
    
    # prólogo do worker (multisession precisa disto)
    suppressPackageStartupMessages({
      library(mgcv)
      if (!requireNamespace("RhpcBLASctl", quietly = TRUE)) install.packages("RhpcBLASctl")
      library(RhpcBLASctl)
    })
    Sys.setenv(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", BLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    RhpcBLASctl::blas_set_num_threads(1)
    RhpcBLASctl::omp_set_num_threads(1)
    
    e_item <- e_from_E(nm, E_scaled, Z0_std, varmap)
    if (!any(is.finite(e_item)) || stats::sd(e_item, na.rm = TRUE) < get0("MIN_SD_ITEM", ifnotfound = 1e-6))
      return(NULL)
    
    # Smooth part on B′
    sm <- fibre_smooth_R2_p(e_item, Bprime, k_gam = 6)
    R2_sm <- as.numeric(sm["R2"]); p_sm <- as.numeric(sm["p"])
    
    # Residual-of-smooth
    df  <- data.frame(v = as.numeric(e_item), Bprime)
    smt <- paste0("s(", colnames(Bprime), ",k=", 6, ")")
    fml <- reformulate(smt, response = "v")
    g   <- try(mgcv::gam(fml, data = df, method = "REML"), silent = TRUE)
    rloc <- if (inherits(g, "try-error")) df$v - mean(df$v, na.rm = TRUE) else residuals(g, type = "response")
    
    # Local via neighbour means
    sel <- choose_k_nb(rloc, nb_list_Bp, folds = 3, seed = 456)
    rf  <- try(r2_fibre_perm_rowshuffle(rloc, sel$nb, B = get0("FIBRE_PERM_B", ifnotfound = 200L), seed = 456), silent = TRUE)
    R2_loc <- if (!inherits(rf, "try-error")) as.numeric(rf["R2"]) else NA_real_
    p_loc  <- if (!inherits(rf, "try-error")) as.numeric(rf["p"])  else NA_real_
    
    data.frame(
      var              = nm,
      R2_fibre_smooth  = R2_sm,    p_smooth = p_sm,
      R2_fibre_local   = R2_loc,   p_local  = p_loc,
      k_local          = sel$k,
      stringsAsFactors = FALSE
    )
  },
  future.seed = TRUE
)

# (opcional) voltar a sequencial no fim desta secção:
future::plan(sequential)

fibre_items <- dplyr::bind_rows(Filter(Negate(is.null), fibre_rows))
if (!nrow(fibre_items)) stop("[fibre-decomp] No items produced valid fibre self-decomposition.")

# Multiple testing control and role inside fibre space
fibre_items$q_smooth <- p.adjust(fibre_items$p_smooth, method = "BH")
fibre_items$q_local  <- p.adjust(fibre_items$p_local,  method = "BH")

SIG_Q <- get0("SIG_Q", ifnotfound = 0.01)
fibre_items$role_fibre <- with(
  fibre_items,
  ifelse(q_smooth < SIG_Q & (is.na(q_local) | q_local >= SIG_Q), "fibre-smooth",
         ifelse(q_local  < SIG_Q & (is.na(q_smooth)| q_smooth>= SIG_Q), "fibre-local",
                ifelse(q_smooth < SIG_Q & q_local  < SIG_Q,                    "mixed", "weak")))
)

readr::write_csv(fibre_items, "fibre_self_decomp_items.csv")
cat(sprintf("[fibre-decomp] wrote: fibre_self_decomp_items.csv (p=%d)\n", nrow(fibre_items)))

# Optional quick plot: smooth vs local R²
if (requireNamespace("ggplot2", quietly = TRUE)) {
  pFD <- ggplot2::ggplot(fibre_items, ggplot2::aes(R2_fibre_smooth, R2_fibre_local, color = role_fibre)) +
    ggplot2::geom_point(alpha = 0.8) +
    ggplot2::labs(title = "Fibre self-decomposition per item",
                  x = "R² (smooth over B′)", y = "R² (local beyond smooth)") +
    ggplot2::theme_minimal()
  print(pFD)
  ggplot2::ggsave("FIG_fibre_self_decomp_scatter.png", pFD, width = 7, height = 5, dpi = 150)
}

# =============== 14) Diagnosis probability fields over Base(b1,b2) =============
# Prereqs:
#   Base          : n × 2 (m*=2) — fixed
#   DxW (from Sect. 12): per-row diagnoses (0/1)
#   keep_dx       : diagnoses to evaluate
# Output:
#   - CSV grids in dir 'base_prob_grids/' one per dx
#   - Optional heatmaps

stopifnot(ncol(Base) == 2)
dir.create("base_prob_grids", showWarnings = FALSE)

grid_from_base <- function(Base, nx = 120, ny = 120, pad = 0.05){
  rx <- range(Base[,1]); ry <- range(Base[,2])
  wx <- diff(rx); wy <- diff(ry)
  xs <- seq(rx[1] - pad*wx, rx[2] + pad*wx, length.out = nx)
  ys <- seq(ry[1] - pad*wy, ry[2] + pad*wy, length.out = ny)
  as.matrix(expand.grid(b1 = xs, b2 = ys))
}

predict_dx_surface <- function(y, Base, gridXY, k_gam = 20){
  df <- data.frame(y = as.integer(y), b1 = Base[,1], b2 = Base[,2])
  # Thin plate spline over (b1,b2); REML is stable
  fit <- try(mgcv::gam(y ~ s(b1, b2, k = k_gam), data = df, family = binomial(), method = "REML"), silent = TRUE)
  if (inherits(fit, "try-error")) return(NULL)
  pg  <- try(as.numeric(predict(fit, newdata = data.frame(b1 = gridXY[,1], b2 = gridXY[,2]), type = "response")), silent = TRUE)
  if (inherits(pg, "try-error")) return(NULL)
  pmin(pmax(pg, 1e-6), 1 - 1e-6)
}

gridXY <- grid_from_base(Base, nx = 140, ny = 140)

dx_surface_files <- c()
for (dx in keep_dx) {
  y <- as.integer(DxW[[dx]] > 0)
  pg <- predict_dx_surface(y, Base, gridXY, k_gam = 30)
  if (is.null(pg)) { warning(sprintf("[base-field] skip dx=%s (fit failed).", dx)); next }
  out <- data.frame(b1 = gridXY[,1], b2 = gridXY[,2], p = pg, dx = dx)
  fp  <- file.path("base_prob_grids", paste0("probgrid_", make_fname_safe(dx), ".csv"))
  readr::write_csv(out, fp)
  dx_surface_files <- c(dx_surface_files, fp)
}
cat(sprintf("[base-field] wrote %d grid(s) to base_prob_grids/ \n", length(dx_surface_files)))

# Optional: quick facet heatmap (downsample for plotting speed)
if (requireNamespace("ggplot2", quietly = TRUE)) {
  show <- min(6, length(keep_dx))
  if (show > 0) {
    demo <- data.table::rbindlist(lapply(head(keep_dx, show), function(dx){
      fp <- file.path("base_prob_grids", paste0("probgrid_", make_fname_safe(dx), ".csv"))
      if (!file.exists(fp)) return(NULL)
      suppressMessages(readr::read_csv(fp, show_col_types = FALSE))
    }))
    if (nrow(demo)) {
      pH <- ggplot2::ggplot(demo, ggplot2::aes(b1, b2, fill = p)) +
        ggplot2::geom_raster() +
        ggplot2::facet_wrap(~ dx, scales = "fixed") +   # ← was "free"
        ggplot2::scale_fill_viridis_c(limits = c(0, 1)) +
        ggplot2::coord_equal(expand = FALSE) +          # keep square pixels
        ggplot2::labs(title = "Diagnosis probability fields over Base", fill = "P(dx)") +
        ggplot2::theme_minimal()
      print(pH); ggplot2::ggsave("FIG_base_prob_fields.png", pH, width = 10, height = 8, dpi = 150)
    }
  }
}

# ======== 15) Item × diagnosis interaction over Base (OOF GAM-ANOVA) ==========
# Prereqs:
#   Base (n×2), DxW + keep_dx (from Sect. 12), item scoring helper from Sect. 10
#   vars: items to consider; to keep runtime healthy we cap to top-N by PC_contrib
#
# Output:
#   CSV 'item_dx_interactions.csv' with rows (dx, var, R2_add, R2_int, dR2, p_like)

stopifnot(exists("Base"), ncol(Base) == 2, exists("vars"))

# Cap to a sensible number of items per dx (rank by PC_contrib if available)
N_TOP_PER_DX <- get0("N_TOP_PER_DX", ifnotfound = 80L)
if (!exists("PC_contrib")) PC_contrib <- setNames(rep(0, length(vars)), vars)
vars_ranked <- names(sort(PC_contrib[vars], decreasing = TRUE))
vars_probe  <- head(vars_ranked, min(N_TOP_PER_DX, length(vars_ranked)))

score_item_base <- get0("score_item_base")  # from Sect. 10

# OOF R² helper for two nested GAMs
oof_R2_two_gams <- function(v, Base, dx, K = 5, k_gam = 12, seed = 1){
  set.seed(seed)
  n <- length(v)
  if (n != nrow(Base) || n != length(dx)) return(c(R2_add = NA, R2_int = NA, p_like = NA))
  fold_id <- sample(rep(1:K, length.out = n))
  y_add <- rep(NA_real_, n); y_int <- rep(NA_real_, n)
  for (k in sort(unique(fold_id))) {
    tr <- which(fold_id != k); te <- which(fold_id == k)
    dtr <- data.frame(v = v[tr], b1 = Base[tr,1], b2 = Base[tr,2], dx = as.factor(dx[tr]))
    dte <- data.frame(b1 = Base[te,1], b2 = Base[te,2], dx = as.factor(dx[te]))
    
    f_add <- try(mgcv::gam(v ~ s(b1,b2,k=k_gam) + dx, data = dtr, method = "REML"), silent = TRUE)
    f_int <- try(mgcv::gam(v ~ s(b1,b2,k=k_gam) + dx + s(b1,b2, by = dx, k = k_gam),
                           data = dtr, method = "REML"), silent = TRUE)
    if (inherits(f_add, "try-error") || inherits(f_int, "try-error")) next
    
    y_add[te] <- as.numeric(predict(f_add, newdata = dte, type = "response"))
    y_int[te] <- as.numeric(predict(f_int, newdata = dte, type = "response"))
  }
  ve <- stats::var(v, na.rm = TRUE)
  R2_add <- if (is.finite(ve) && ve > 0) max(0, 1 - mean((v - y_add)^2, na.rm = TRUE) / ve) else NA_real_
  R2_int <- if (is.finite(ve) && ve > 0) max(0, 1 - mean((v - y_int)^2, na.rm = TRUE) / ve) else NA_real_
  # Pseudo-likelihood ratio p via paired bootstrap on OOF errors
  set.seed(seed + 1L)
  errs_add <- (v - y_add); errs_int <- (v - y_int)
  B <- 400
  diffs <- replicate(B, {
    ix <- sample.int(n, n, replace = TRUE)
    ve_b <- stats::var(v[ix], na.rm = TRUE)
    if (!is.finite(ve_b) || ve_b <= 0) return(NA_real_)
    R2a <- 1 - mean(errs_add[ix]^2, na.rm = TRUE) / ve_b
    R2i <- 1 - mean(errs_int[ix]^2, na.rm = TRUE) / ve_b
    R2i - R2a
  })
  p_like <- mean(na.omit(diffs) <= 0)  # one-sided: is ΔR² > 0?
  c(R2_add = R2_add, R2_int = R2_int, p_like = p_like)
}

rows <- list()
for (dx in keep_dx) {
  ydx <- as.integer(DxW[[dx]] > 0)
  for (nm in vars_probe) {
    v <- score_item_base(nm, scale(Xenc, TRUE, TRUE), varmap)  # stable scoring
    if (!any(is.finite(v)) || stats::sd(v, na.rm = TRUE) < 1e-6) next
    res <- oof_R2_two_gams(v, Base, ydx, K = 5, k_gam = 10, seed = SEED_PRED)
    rows[[length(rows)+1]] <- data.frame(
      dx = dx, var = nm,
      R2_add = res["R2_add"], R2_int = res["R2_int"],
      dR2    = unname(res["R2_int"] - res["R2_add"]),
      p_like = res["p_like"],
      stringsAsFactors = FALSE
    )
  }
}
int_tbl <- dplyr::bind_rows(rows)
if (!nrow(int_tbl)) {
  warning("[interactions] empty table; no items/diagnoses passed screening.")
} else {
  int_tbl$q_like <- p.adjust(int_tbl$p_like, method = "BH")
  int_tbl$sig    <- int_tbl$q_like < SIG_Q
  int_tbl <- dplyr::arrange(int_tbl, dplyr::desc(dR2))
  readr::write_csv(int_tbl, "item_dx_interactions.csv")
  cat(sprintf("[interactions] wrote: item_dx_interactions.csv (rows=%d)\n", nrow(int_tbl)))
  
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    topK <- head(int_tbl, 30)
    pI <- ggplot2::ggplot(topK, ggplot2::aes(x = stats::reorder(paste(dx, var, sep=" · "), dR2), y = dR2,
                                             color = q_like < 0.10)) +
      ggplot2::geom_point() + ggplot2::coord_flip() +
      ggplot2::labs(title = "Top ΔR² (interaction gain) — item × dx over Base",
                    x = "dx · item", y = "ΔR² (interaction − additive)") +
      ggplot2::theme_minimal()
    print(pI); ggplot2::ggsave("FIG_item_dx_interaction_top.png", pI, width = 8, height = 10, dpi = 150)
  }
}

# =================== 16) Cluster-level summaries & exports =====================
# Prereqs:
#   clF     : integer cluster labels from B′ Louvain (Sect. 11). If missing, we
#             build a fallback by clustering in Base (kNN+Louvain).
#   DxW     : diagnoses aligned to rows in Base
# Output:
#   - cluster_summary.csv   : size per cluster, Base centroid, fibre centroid
#   - cluster_dx_enrichment.csv : std residuals per (cluster, dx)
#   - Optional heatmap

if (!exists("clF")) {
  message("[clusters] fibre clusters not found; clustering on Base as fallback.")
  idx <- RANN::nn2(Base, Base, k = pmin(16, nrow(Base)))$nn.idx[, -1]
  i <- rep(seq_len(nrow(Base)), each = ncol(idx)); j <- as.vector(idx)
  g <- igraph::graph_from_edgelist(cbind(i, j), directed = FALSE)
  g <- igraph::simplify(g)
  clF <- igraph::cluster_louvain(g)$membership
}

kF <- length(unique(clF))
cat(sprintf("[clusters] using %d clusters.\n", kF))

# 16.1 Summary per cluster (robust to tapply() array outputs)
clusters <- sort(unique(clF))
cl_fac   <- factor(clF, levels = clusters)

mean_by_cluster <- function(vec, clf) {
  out <- tapply(vec, clf, function(z) mean(z, na.rm = TRUE))
  as.numeric(out)  # strip array class, keep order = levels(clf)
}

n_by_cluster <- as.integer(tabulate(cl_fac))

base_means <- cbind(
  b1_mean = mean_by_cluster(Base[, 1], cl_fac),
  b2_mean = mean_by_cluster(Base[, 2], cl_fac)
)

f_means <- NULL
if (exists("Bprime") && is.matrix(Bprime) && ncol(Bprime) > 0) {
  f_means <- sapply(seq_len(ncol(Bprime)), function(j) mean_by_cluster(Bprime[, j], cl_fac))
  if (!is.null(dim(f_means))) {
    colnames(f_means) <- paste0("f", seq_len(ncol(Bprime)), "_mean")
  } else {
    # Edge case: only one fibre PC → sapply returns vector
    f_means <- cbind(`f1_mean` = as.numeric(f_means))
  }
}

sum_tbl <- data.frame(
  cluster = clusters,
  n       = n_by_cluster,
  base_means,
  check.names = FALSE,
  stringsAsFactors = FALSE
)

if (!is.null(f_means)) {
  # ensure it's a plain numeric matrix
  f_means <- as.matrix(f_means)
  storage.mode(f_means) <- "double"
  sum_tbl <- cbind(sum_tbl, f_means, stringsAsFactors = FALSE)
}

readr::write_csv(sum_tbl, "cluster_summary.csv")

# 16.2 Diagnosis enrichment per cluster (std residuals)
DxW2 <- DxW
DxW2[is.na(DxW2)] <- 0L
prev <- colSums(DxW2 > 0)
DxW2 <- DxW2[, prev >= get0("DX_CASES_MIN", ifnotfound = 15L), drop = FALSE]
tab  <- sapply(colnames(DxW2), function(dn) tapply(DxW2[[dn]] == 1L, factor(clF), sum, na.rm = TRUE))
tab  <- as.matrix(tab); mode(tab) <- "numeric"
Eexp <- outer(rowSums(tab), colSums(tab), function(r,c) r * c / max(sum(tab), 1))
Z    <- (tab - Eexp) / sqrt(pmax(Eexp, 1e-9))  # std residuals

enrich <- as.data.frame(as.table(Z))
colnames(enrich) <- c("cluster", "diagnosis", "std_resid")
enrich$cluster <- as.integer(as.character(enrich$cluster))
readr::write_csv(enrich, "cluster_dx_enrichment.csv")

cat("Wrote: cluster_summary.csv, cluster_dx_enrichment.csv\n")

# Optional heatmap
if (requireNamespace("ggplot2", quietly = TRUE)) {
  pH <- ggplot2::ggplot(enrich, ggplot2::aes(diagnosis, factor(cluster), fill = std_resid)) +
    ggplot2::geom_tile(colour = "white") +
    ggplot2::scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
    ggplot2::labs(title = "Cluster × diagnosis enrichment (std residuals)",
                  x = "Diagnosis", y = "Cluster", fill = "Std resid") +
    ggplot2::theme_minimal() +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
  print(pH); ggplot2::ggsave("FIG_cluster_dx_enrichment.png", pH, width = 10, height = 6, dpi = 150)
}