# =================================================================================================
# Psychometric Base-Fibre Manifold + Diagnosis Mapping + Fibre Contribution Diagnostics
# -------------------------------------------------------------------------------------------------
# Purpose
#   End-to-end pipeline to:
#     (1) Load mixed psychometric data (X). 
#     (2) Gower distances + per-variable weights to minimise
#         intrinsic dimension; select informative predictors. 
#     (3) Build low-d "Base" manifold via PCA.
#     (4) Residualise predictors on Base to obtain fold-safe "Fibre" features (E). 
#     (5) Classify item
#         roles (Base-aligned / Fibre-structured / Mixed / Weak) with R² metrics and permutation tests.
#     (6) Optional: explore structure within Fibre space. 
#     (7) Per-diagnosis OOF predictions for Base,
#         Fibre, and Stacked models with AUC/CI/DeLong. 
#     (8) Export maps, tables, diagnostics.
#
# Inputs
#   - psychometric_matrix.csv  (semicolon-delimited; first col or 'participant_id' is ID)
#   - long_diagnoses.csv       (long table: participant_id, diagnosis)
#
# Outputs (main)
#   - predictive_item_roles_diagnostics.csv
#   - dx_auc_base_fibre_stacked_with_pr.csv  + roc_csv/ROC_*.csv (if pROC available)
#   - fibre_only_summary.rds
#   - base_prob_grids/probgrid_<diagnosis>.csv
#   - cluster_summary.csv, cluster_dx_enrichment.csv
#
# Author: Afonso Dinis Ribeiro
# Date:   2025-08-23
# =================================================================================================

suppressPackageStartupMessages({
  library(stats);  library(utils);  library(Matrix)
  library(cluster)      # Gower (daisy)
  library(glmnet)       # penalised GLMs
  library(mgcv)         # GAMs
  library(RANN)         # kNN
  library(dplyr); library(tidyr)
  library(readr); library(data.table)
  if (!requireNamespace("ggplot2", quietly = TRUE)) NULL
  if (!requireNamespace("ggrepel", quietly = TRUE)) NULL
  if (!requireNamespace("pROC",    quietly = TRUE)) NULL
  if (!requireNamespace("igraph",  quietly = TRUE)) NULL
  if (!requireNamespace("future",  quietly = TRUE)) NULL
  if (!requireNamespace("future.apply", quietly = TRUE)) NULL
  if (!requireNamespace("parallelly", quietly = TRUE)) NULL
  if (!requireNamespace("zoo", quietly = TRUE)) NULL
})
suppressPackageStartupMessages({ if (requireNamespace("compiler", quietly=TRUE)) compiler::enableJIT(3) })

# ---- Reproducibility
SEED_GLOBAL <- 42L
set.seed(SEED_GLOBAL)
RNGkind("L'Ecuyer-CMRG")

# ---- Threading (pin BLAS/OMP to 1 for stability)
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

# ============================ Configuration ====================================
# Target for binary modelling
PREF_TARGET   <- "ANY_DX"     # or "NODIAG"

# Redundancy hygiene (kept off; selection handled by weights)
DO_CORR_TRIM  <- FALSE
CORR_THRESH   <- 0.95

# Cross-validation
CV_FOLDS      <- 5L
SEED_PRED     <- 42L
SEED_JITTER   <- 1L
SEED_BOOT     <- 42L
CV_REPEATS    <- 1L
MIN_TEST_POS  <- 5L
MIN_TEST_NEG  <- 5L
BOOT_B        <- 1000L

# Base dimensionality
M_STAR_FIXED  <- 2L

# Deduplication on Gower
DO_DEDUP        <- TRUE
EPS_DEDUP       <- NA_real_   # auto if NA
WRITE_DEDUP_CSV <- TRUE

# Gower weight optimisation (TwoNN objective)
W_MIN          <- 0.00
W_STEP_GRID    <- c(0.95, 0.90, 0.75, 0.5, 0.25, 0.10, 0.05, 0.01, 0.00)
W_BATCH_K      <- 3
W_BATCH_FACTOR <- 0.75
W_MAX_ITERS    <- NA_integer_  # auto cap if NA

# Survivors: knee on tail
USE_KNEE_THR   <- TRUE
KEEP_THR_FIXED <- 0.10   # used only if USE_KNEE_THR=FALSE

# Parallelism / sampling
NCORES_PAR   <- max(1, parallel::detectCores() - 1)
N_ROWS_SUB   <- NULL
FIX_REP_SUBSET <- TRUE

# Geometry diagnostics
KS_TC        <- 10:30
K_BASE_NEIGH <- 40
K_ID_LO_HI   <- c(8, 20)
PC_MAX       <- 6
M_DEFAULT    <- 4

# Diagnosis gating
DO_DX_CLUSTER_DIAG <- FALSE
DO_DX_ASSOC        <- FALSE
DX_DENY_NOS        <- TRUE
DX_PREV_MIN        <- 0.00
DX_CASES_MIN       <- 10
N_TOP_PER_DX       <- 80

# Factor handling
RARE_LEVEL_MIN_PROP <- 0.01

# Core band for ID estimates
CORE_BAND   <- c(0.20, 0.70)
CORE_KNN_K  <- 10

# Role-significance threshold
SIG_Q <- 0.01

# Fibre features used in dx models
K_FIBRE_CAP <- 3

# =========================== Gower preparation =================================
is_binary <- function(v){
  u <- sort(unique(na.omit(as.numeric(v))))
  length(u) == 2 && all(u %in% c(0,1))
}

prep_X_for_gower <- function(X, rare_prop = 0.01, do_jitter = TRUE){
  X1 <- as.data.frame(X, check.names = TRUE, stringsAsFactors = FALSE)
  for(nm in names(X1)){
    v <- X1[[nm]]
    if (is.character(v)) X1[[nm]] <- factor(v)
  }
  drop_rare <- function(f, prop){
    if (!is.factor(f) || is.ordered(f)) return(f)
    tb <- prop.table(table(f))
    keep <- names(tb)[tb >= prop]
    f <- factor(ifelse(f %in% keep, as.character(f), NA), exclude = NULL)
    droplevels(f)
  }
  X1 <- as.data.frame(lapply(X1, drop_rare, prop = rare_prop), stringsAsFactors = FALSE)
  if (isTRUE(do_jitter)) {
    for(nm in names(X1)){
      if (is.numeric(X1[[nm]])) {
        sdv <- stats::sd(X1[[nm]], na.rm = TRUE)
        if (is.finite(sdv) && sdv > 0) X1[[nm]] <- X1[[nm]] + rnorm(length(X1[[nm]]), 0, 1e-6 * sdv)
      }
    }
  }
  ord_cols  <- names(X1)[vapply(X1, is.ordered, logical(1))]
  fac_cols  <- names(X1)[vapply(X1, function(z) is.factor(z) && !is.ordered(z), logical(1))]
  bin_cols  <- fac_cols[vapply(X1[fac_cols], is_binary, logical(1))]
  type_list <- list()
  if (length(bin_cols)) type_list$asymm <- bin_cols
  if (length(ord_cols)) type_list$ordratio <- ord_cols
  w <- rep(1, ncol(X1)); names(w) <- names(X1)
  list(X = X1, type = type_list, weights = w)
}

gower_dist <- function(Xdf, type_list = NULL, weights = NULL){
  if (!is.null(type_list)) {
    type_list <- lapply(type_list, function(cols) intersect(cols, names(Xdf)))
    type_list <- type_list[lengths(type_list) > 0]
    if (!length(type_list)) type_list <- NULL
  }
  if (is.null(weights)) {
    weights <- rep(1, ncol(Xdf))
  } else if (length(weights) == 1) {
    weights <- rep(weights, ncol(Xdf))
  } else if (!is.null(names(weights))) {
    weights <- weights[names(Xdf)]
  }
  stopifnot(length(weights) == ncol(Xdf))
  cluster::daisy(Xdf, metric = "gower", type = type_list, weights = weights)
}

# ============================= Geometry helpers ================================
core_band_idx <- function(D, k = 10, band = c(0.15,0.85)){
  M <- as.matrix(D); diag(M) <- Inf
  kth <- function(r, k) { rf <- r[is.finite(r)]; if (!length(rf)) return(NA_real_)
  k_eff <- min(k, length(rf)); sort(rf, partial = k_eff)[k_eff] }
  rk <- apply(M, 1, kth, k = k)
  ok <- is.finite(rk); if (!any(ok)) return(integer(0))
  q  <- stats::quantile(rk[ok], band, na.rm = TRUE)
  which(ok & rk >= q[1] & rk <= q[2])
}

twonn_id_from_dist <- function(D, eps=1e-8, trim=0.02){
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
}

twonn_core_by_slope <- function(D, min_frac = 0.30, w = 20, slope_tol = 0.08, rmse_tol = 0.10) {
  M <- as.matrix(D); n <- nrow(M); diag(M) <- Inf
  if (n < 8) return(seq_len(n))
  r1 <- apply(M, 1L, function(r) sort(r)[1])
  r2 <- apply(M, 1L, function(r) sort(r)[2])
  mu <- pmax(r2 / pmax(r1, .Machine$double.eps), 1 + 1e-12)
  ord <- order(mu); x <- log(mu[ord]); m <- length(x)
  u   <- (seq_len(m) - 0.5) / (m + 1); y <- log(1 - u)
  k0 <- max(20L, floor(min_frac * m))
  slope <- rmse <- rep(NA_real_, m)
  for (k in k0:(m - 2L)) {
    fit <- stats::lm(y[1:k] ~ x[1:k])
    slope[k] <- coef(fit)[2]; rmse[k] <- sqrt(mean(residuals(fit)^2))
  }
  ok <- which(is.finite(slope) & is.finite(rmse))
  if (!length(ok)) return(ord[seq_len(max(3L, k0))])
  pick <- function(k) {
    L <- max(k0, k - w + 1); s <- slope[L:k]
    (max(s, na.rm = TRUE) - min(s, na.rm = TRUE) <= slope_tol) && (rmse[k] <= rmse_tol)
  }
  ks <- ok[vapply(ok, pick, logical(1))]
  k_star <- if (length(ks)) max(ks) else floor(0.6 * m)
  ord[seq_len(max(3L, min(k_star, m - 2L)))]
}

lb_mle_id <- function(Dm, k_lo=5, k_hi=15){
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
}

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

# ============================ Knee selection ===================================
knee_triangle <- function(w) {
  y <- sort(as.numeric(w), decreasing = TRUE)
  n <- length(y)
  if (n < 3L) return(list(k = n, thr = if (n) y[n] else NA_real_,
                          curve = data.frame(i = seq_len(n), w = y, d = rep(0, n))))
  x <- seq_len(n)
  num <- abs((y[n] - y[1]) * x - (n - 1) * y + n * y[1] - y[n])
  den <- sqrt((y[n] - y[1])^2 + (n - 1)^2)
  d   <- num / den
  k <- which.max(d)
  list(k = k, thr = y[k], curve = data.frame(i = x, w = y, d = d))
}

survivors_from_weights <- function(w, w_min = W_MIN, kmin = NULL,
                                   eps_ceil = 1e-4, eps_floor = 1e-12,
                                   make_plot = TRUE, plot_file = "FIG_weight_curve_knee.png") {
  if (is.null(names(w))) names(w) <- paste0("V", seq_along(w))
  w[] <- pmax(w_min, as.numeric(w))
  p <- length(w)
  cat(sprintf("[weights] p=%d | min=%.4f  q25=%.4f  med=%.4f  q75=%.4f  max=%.4f\n",
              p, min(w), as.numeric(quantile(w, .25)), median(w),
              as.numeric(quantile(w, .75)), max(w)))
  idx_ceil <- which(w >= 1 - eps_ceil)
  idx_tail <- which(w <  1 - eps_ceil & w > w_min + eps_floor)
  thr_tail <- NA_real_; knee_obj <- NULL
  if (length(idx_tail) >= 3L) {
    knee_obj <- knee_triangle(w[idx_tail]); thr_tail <- knee_obj$thr
  } else if (length(idx_tail) > 0L) {
    thr_tail <- max(w_min + 1e-6, median(w[idx_tail]))
  }
  S_ceil <- names(w)[idx_ceil]
  S_tail <- if (is.finite(thr_tail)) names(w)[w >= thr_tail & w < 1 - eps_ceil] else character(0)
  survivors <- union(S_ceil, S_tail)
  if (is.null(kmin)) kmin <- max(30L, ceiling(0.10 * p))
  if (length(survivors) < kmin) {
    survivors <- names(sort(w, decreasing = TRUE))[seq_len(kmin)]
  }
  cat(sprintf("Ceiling kept: %d | tail knee thr=%s | survivors: %d / %d\n",
              length(S_ceil),
              ifelse(is.finite(thr_tail), sprintf("%.3f", thr_tail), "NA"),
              length(survivors), p))
  if (isTRUE(make_plot) && requireNamespace("ggplot2", quietly = TRUE)) {
    ord <- order(w, decreasing = TRUE)
    curve <- data.frame(i = seq_along(ord), w = as.numeric(w[ord]))
    pplt <- ggplot2::ggplot(curve, ggplot2::aes(i, w)) +
      ggplot2::geom_line() +
      ggplot2::labs(x = "rank (sorted ↓)", y = "weight", title = "Weight curve with knee (tail-only)") +
      ggplot2::theme_minimal()
    if (!is.null(knee_obj)) {
      thr_mark <- knee_obj$thr
      knee_row <- curve[which.min(abs(curve$w - thr_mark)), , drop = FALSE]
      pplt <- pplt + ggplot2::geom_point(data = knee_row, size = 2)
    }
    print(pplt)
    try(ggplot2::ggsave(plot_file, pplt, width = 6, height = 4, dpi = 150), silent = TRUE)
  }
  list(survivors = survivors, thr_tail  = thr_tail, w_clamped = w)
}

# =========================== Data ingest & target ==============================
# 1) Load psychometric matrix
if (!exists("X") || !nrow(X)) {
  df <- readr::read_delim("psychometric_matrix.csv", delim = ";",
                          locale = readr::locale(decimal_mark = "."),
                          progress = FALSE)
  id_col  <- if ("participant_id" %in% names(df)) "participant_id" else names(df)[1]
  ids_all <- as.character(df[[id_col]])
  X <- dplyr::select(df, -dplyr::all_of(c(id_col, grep("^diagnosis", names(df), value = TRUE))))
  X <- as.data.frame(X, stringsAsFactors = FALSE)
  
  is_small_int_scale <- function(v) {
    vn <- suppressWarnings(as.numeric(v))
    if (all(is.na(vn))) return(FALSE)
    u <- sort(unique(na.omit(vn))); k <- length(u)
    k >= 3 && k <= 7 && all(abs(u - round(u)) < 1e-8)
  }
  for (nm in names(X)) {
    v <- X[[nm]]
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
  keep <- stats::complete.cases(X)
  X    <- X[keep, , drop = FALSE]
  rownames(X) <- make.unique(ids_all[keep])
}

# 2) Load diagnoses (already wide 0/1)
if (!exists("diag_wide_full")) {
  diag_wide_full <- readr::read_delim("wide_diagnoses.csv", delim = ";",
                                      col_types = readr::cols(), progress = FALSE) |>
    dplyr::mutate(participant_id = as.character(participant_id)) |>
    dplyr::mutate(dplyr::across(-participant_id, ~ {
      x <- suppressWarnings(as.integer(.))
      x[is.na(x)] <- 0L
      pmin(pmax(x, 0L), 1L)
    }))
}

# 3) Target y_use
ids_here <- rownames(X)
mm <- match(ids_here, as.character(diag_wide_full$participant_id))
if (anyNA(mm)) stop(sprintf("Diagnoses join failed for %d/%d rows.", sum(is.na(mm)), length(mm)))

dx_cols_all <- setdiff(names(diag_wide_full), "participant_id")
drop_nodiag <- grepl("^no\\s*diagnosis|^NODIAG$", dx_cols_all, ignore.case = TRUE)
dx_cols <- dx_cols_all[!drop_nodiag]
dx_mat <- if (length(dx_cols)) {
  M <- as.matrix(diag_wide_full[mm, dx_cols, drop = FALSE]); M[is.na(M)] <- 0L
  keepc <- colSums(M, na.rm = TRUE) > 0
  if (!any(keepc)) NULL else M[, keepc, drop = FALSE]
} else NULL
any_dx <- if (is.null(dx_mat)) rep.int(0L, length(ids_here)) else as.integer(rowSums(dx_mat) > 0)
make_y <- function(pref, any_dx) if (toupper(pref) == "NODIAG") 1L - any_dx else any_dx
pref  <- toupper(PREF_TARGET)
y_use <- make_y(pref, any_dx)
n0 <- sum(y_use == 0); n1 <- sum(y_use == 1)
if (n0 == 0 || n1 == 0) {
  message(sprintf("[Target %s] degenerate (n0=%d, n1=%d) - switching.", pref, n0, n1))
  pref_alt <- if (pref == "NODIAG") "ANY_DX" else "NODIAG"
  y_use    <- make_y(pref_alt, any_dx)
  n0 <- sum(y_use == 0); n1 <- sum(y_use == 1)
  if (n0 == 0 || n1 == 0) stop("Target remains degenerate after switch.")
  PREF_TARGET <- pref_alt
}
cat(sprintf("[Target %s] n=%d | n0=%d | n1=%d\n", toupper(PREF_TARGET), length(y_use), n0, n1))

# ======================= ε-deduplication and core ==============================
first_nn_d1 <- function(D){
  Dm <- as.matrix(D); diag(Dm) <- Inf
  apply(Dm, 1L, function(r){ r <- r[is.finite(r)]; if (!length(r)) Inf else min(r) })
}
collapse_curve <- function(D, eps_grid){
  n0 <- attr(D, "Size")
  data.frame(
    eps = eps_grid,
    n_groups = sapply(eps_grid, function(eps) length(unique(hclust(D, method = "complete") |> cutree(h = eps)))),
    prop_retained = NA_real_
  ) |>
    transform(prop_retained = n_groups / n0)
}
complete_groups <- function(D, eps){
  hclust(D, method = "complete") |> cutree(h = eps)
}
group_medoids <- function(D, groups){
  Dm <- as.matrix(D); diag(Dm) <- 0
  split_idx <- split(seq_len(nrow(Dm)), groups)
  reps <- vapply(split_idx, function(ix){ ix[ which.min(rowSums(Dm[ix, ix, drop = FALSE])) ] }, integer(1))
  list(reps = unname(reps), mult = as.integer(lengths(split_idx)))
}

px  <- prep_X_for_gower(X, rare_prop = RARE_LEVEL_MIN_PROP, do_jitter = TRUE)
X_for_id <- px$X
Dg <- gower_dist(X_for_id, type_list = px$type, weights = px$weights)

if (is.na(EPS_DEDUP)) {
  d1 <- first_nn_d1(Dg)
  qlo <- as.numeric(stats::quantile(d1[is.finite(d1)], probs = c(0.001, 0.01), na.rm = TRUE))
  eps_grid <- seq(from = max(0, min(qlo, na.rm = TRUE) - 0.02),
                  to   = min(0.50, stats::quantile(d1, 0.30, na.rm = TRUE)),
                  by   = 0.005)
  cc <- collapse_curve(Dg, eps_grid)
  dprop <- diff(cc$prop_retained) / diff(cc$eps)
  knee_i <- which.min(dprop)
  EPS_DEDUP <- mean(cc$eps[c(knee_i, knee_i + 1)])
}

gr_all  <- if (isTRUE(DO_DEDUP)) complete_groups(Dg, EPS_DEDUP) else seq_len(attr(Dg, "Size"))
med_all <- group_medoids(Dg, gr_all)
reps    <- med_all$reps
mult    <- med_all$mult

Dm_g   <- as.matrix(Dg); diag(Dm_g) <- Inf
Dg_rep <- stats::as.dist(Dm_g[reps, reps, drop = FALSE])
core_idx_rep <- twonn_core_by_slope(Dg_rep)

cat(sprintf("[Dedup] eps=%.3f | reps=%d of %d | core_rep=%d\n",
            EPS_DEDUP, length(reps), nrow(X), length(core_idx_rep)))

if (isTRUE(WRITE_DEDUP_CSV)) {
  mult_df <- data.frame(
    rep_row           = reps,
    representative_id = rownames(Dm_g)[reps],
    multiplicity      = mult
  )
  readr::write_csv(mult_df, sprintf("near_duplicate_groups_complete_eps%g.csv", EPS_DEDUP))
}

# ================== Constrained Gower-weight optimisation ======================
make_NS_cache <- function(Xdf, type = NULL){
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

optimise_gower_weights_constrained <- function(
    X,
    init_weights,
    allow_update,
    objective     = "TwoNN_all",  # evaluate ID on ALL pairs by default
    w_min         = W_MIN,
    step_grid     = W_STEP_GRID,
    batch_k       = W_BATCH_K,
    batch_factor  = W_BATCH_FACTOR,
    max_iter      = W_MAX_ITERS,
    n_rows_sub    = N_ROWS_SUB,
    ncores        = NCORES_PAR,
    seed_jitter   = SEED_JITTER,
    reps_idx      = NULL,
    core_idx_rep  = NULL,
    verbose       = TRUE,
    plot_progress = TRUE){
  
  restore_env <- Sys.getenv(c("OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"))
  Sys.setenv(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
  on.exit({ do.call(Sys.setenv, as.list(stats::setNames(as.list(restore_env), names(restore_env)))) }, add=TRUE)
  set.seed(seed_jitter)
  
  px  <- prep_X_for_gower(X, rare_prop = RARE_LEVEL_MIN_PROP, do_jitter = TRUE)
  X0  <- px$X;  typ <- px$type
  row_pool <- if (!is.null(reps_idx)) reps_idx else seq_len(nrow(X0))
  ix_sub_from_reps <- if (!is.null(n_rows_sub) && length(row_pool) > n_rows_sub) {
    if (isTRUE(FIX_REP_SUBSET)) head(row_pool, n_rows_sub) else sample(row_pool, n_rows_sub)
  } else row_pool
  Xs   <- X0[ix_sub_from_reps, , drop = FALSE]
  vars <- colnames(Xs); p <- ncol(Xs)
  if (p < 1L) stop("[optim] Xs has 0 columns")
  
  w <- init_weights
  if (length(w) == 1L) w <- rep(w, ncol(X))
  if (is.null(names(w))) names(w) <- colnames(X)
  if (!all(vars %in% names(w))) stop("[optim] init_weights missing some cols")
  w <- as.numeric(w[vars]); names(w) <- vars
  
  if (is.logical(allow_update) && length(allow_update) == ncol(X) && is.null(names(allow_update))) {
    names(allow_update) <- colnames(X)
  }
  if (!all(vars %in% names(allow_update))) stop("[optim] allow_update missing some cols")
  allow_update <- as.logical(allow_update[vars]); stopifnot(length(allow_update) == p)
  w[!allow_update] <- pmax(w_min, w[!allow_update])
  
  cache   <- make_NS_cache(Xs, type = typ)
  eps     <- .Machine$double.eps
  num_cur <- Reduce(`+`, Map(`*`, cache$N, as.list(w)))
  den_cur <- Reduce(`+`, Map(`*`, cache$S, as.list(w)))
  
  # --- DROP-IN: objective over ALL rows vs CORE rows (on reps subset) ----------
  # Map CORE indices (defined on reps) to the working subset rows we actually use
  core_idx_sub <- NULL
  if (!is.null(core_idx_rep) && length(core_idx_rep)) {
    if (!is.null(reps_idx)) {
      # core reps in original rows → which of those are in our working subset?
      core_global <- reps_idx[core_idx_rep]
      core_idx_sub <- match(core_global, ix_sub_from_reps)
      core_idx_sub <- core_idx_sub[is.finite(core_idx_sub)]
      core_idx_sub <- sort(unique(core_idx_sub))
    }
  }
  
  # ID evaluators
  id_eval_all <- function(num, den) {
    Dfull <- num / pmax(den, eps)
    attr(Dfull, "Size") <- nrow(Xs); attr(Dfull, "Diag") <- FALSE
    attr(Dfull, "Upper") <- FALSE; class(Dfull) <- "dist"
    twonn_id_from_dist(Dfull)
  }
  
  id_eval_core <- function(num, den, use_lb = FALSE) {
    if (is.null(core_idx_sub) || length(core_idx_sub) < 3L) return(NA_real_)
    # build full and then subset to core (simple & robust)
    Dfull <- as.matrix(num / pmax(den, eps))
    diag(Dfull) <- 0
    Dcore <- stats::as.dist(Dfull[core_idx_sub, core_idx_sub, drop = FALSE])
    id2nn <- twonn_id_from_dist(Dcore)
    if (!use_lb) return(id2nn)
    # optional LB on the same core
    M <- as.matrix(Dcore); diag(M) = Inf
    idlb <- lb_mle_id(M, 5, 15)
    # weighted mean: mostly TwoNN, a dash of LB
    0.7 * id2nn + 0.3 * idlb
  }
  
  # pick objective
  id_from_numden <- switch(
    objective,
    "TwoNN_core"      = function(num, den) id_eval_core(num, den, use_lb = FALSE),
    "TwoNN_LB_core"   = function(num, den) id_eval_core(num, den, use_lb = TRUE),
    "TwoNN_all"       = function(num, den) id_eval_all(num, den),
    # default fallback
    function(num, den) id_eval_all(num, den)
  )
  
  id0  <- id_from_numden(num_cur, den_cur)
  hist <- data.frame(iter = 0L, ID = id0, changed = NA_character_, note = NA_character_)
  if (verbose) cat(sprintf("[optim] iter 0: %s = %.3f\n", objective, id0))
  
  par_apply <- function(idx, FUN, chunk = NULL){
    if (is.null(chunk)) chunk <- max(1L, ceiling(length(idx) / (2L * max(1L, ncores))))
    chunks <- split(idx, ceiling(seq_along(idx) / chunk))
    if (ncores <= 1L || length(chunks) <= 1L) return(unlist(lapply(chunks, function(ii) lapply(ii, FUN)), recursive = FALSE))
    if (.Platform$OS.type != "windows") {
      return(unlist(parallel::mclapply(chunks, function(ii) lapply(ii, FUN), mc.cores = ncores, mc.preschedule = TRUE), recursive = FALSE))
    } else {
      cl <- parallel::makeCluster(ncores, type = "PSOCK", outfile = "")
      on.exit(parallel::stopCluster(cl), add = TRUE)
      parallel::clusterEvalQ(cl, { Sys.setenv(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1"); NULL })
      parallel::clusterEvalQ(cl, { suppressPackageStartupMessages(library(cluster)); NULL })
      parallel::clusterExport(cl, varlist = c("cache","id_from_numden","eps"), envir = environment())
      return(unlist(parallel::parLapply(cl, chunks, function(ii) lapply(ii, FUN)), recursive = FALSE))
    }
  }
  
  max_iter_eff <- if (is.null(max_iter) || !is.finite(max_iter)) 3L * p else as.integer(max(1L, max_iter))
  id <- id0
  for (it in seq_len(max_iter_eff)){
    can <- which(allow_update & (w > w_min + 1e-12))
    if (!length(can)) { if (verbose) cat("[optim] nothing to update.\n"); break }
    cand <- do.call(rbind, lapply(can, function(j){
      w_try <- unique(pmax(w_min, w[j] * step_grid)); w_try <- w_try[w_try < w[j] - 1e-12]
      if (!length(w_try)) return(NULL)
      data.frame(j=j, wj=w_try)
    }))
    if (is.null(cand) || !nrow(cand)) { if (verbose) cat("[optim] no candidates.\n"); break }
    
    cand$id <- as.numeric(unlist(par_apply(seq_len(nrow(cand)), function(i){
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
      if (verbose) cat(sprintf("[optim] iter %d: ↓ %s → w=%.3f | %s = %.3f\n", it, vars[jbest], wbest, objective, id))
    } else {
      if (verbose) cat("[optim] no improving single-variable move; stopping.\n")
      break
    }
    
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
            hist <- rbind(hist, data.frame(iter=it, ID=id, changed=paste(vars[take], collapse=","), note=sprintf("batch x%.2f", batch_factor)))
            if (verbose) cat(sprintf("[optim] iter %d: batch x%.2f on %d vars | %s = %.3f\n",
                                     it, batch_factor, length(take), objective, id))
          }
        }
      }
    }
    if (!changed){ if (verbose) cat("[optim] no further improvement; stopping.\n"); break }
  }
  
  if (plot_progress && requireNamespace("ggplot2", quietly=TRUE)){
    gp <- ggplot2::ggplot(hist, ggplot2::aes(iter, ID)) +
      ggplot2::geom_line() + ggplot2::geom_point() +
      ggplot2::labs(title=sprintf("Constrained Gower-weight optimisation (%s; core on reps)", objective),
                    x="iteration", y="ID") + ggplot2::theme_minimal()
    print(gp); try(ggplot2::ggsave("FIG_weight_optim_constrained.png", gp, width=6, height=4, dpi=150), silent=TRUE)
  }
  list(weights = w, history = hist, final_ID = id, idx_used = ix_sub_from_reps, core_idx = core_idx_sub)
}

# ---- Apply optimiser on survivor candidates -----------------------------------
drop_constant_cols <- function(X){
  keep <- vapply(X, function(v) length(unique(na.omit(v))) >= 1L, logical(1))
  X[, keep, drop=FALSE]
}

X_pred <- drop_constant_cols(X)
if (DO_CORR_TRIM) {
  corr_trim <- function(X, thr=0.95){
    num <- X[, vapply(X, is.numeric, logical(1)), drop=FALSE]
    if (!ncol(num)) return(X)
    C <- suppressWarnings(cor(num, use="pairwise.complete.obs"))
    drop <- character(0)
    for (i in seq_len(ncol(C)-1)) {
      if (colnames(C)[i] %in% drop) next
      j <- which(abs(C[i,(i+1):ncol(C)]) >= thr) + i
      drop <- union(drop, colnames(C)[j])
    }
    keep <- setdiff(colnames(X), drop)
    X[, keep, drop=FALSE]
  }
  X_pred <- corr_trim(X_pred, CORR_THRESH)
}
cat(sprintf("Start: X_pred has %d columns after constant-drop%s.\n\n",
            ncol(X_pred), if (DO_CORR_TRIM) " + corr-trim" else ""))

w_init <- setNames(rep(1, ncol(X)), colnames(X))
allow  <- setNames(rep(FALSE, ncol(X)), colnames(X))
allow[colnames(X_pred)] <- TRUE

wopt <- optimise_gower_weights_constrained(
  X, init_weights = w_init, allow_update = allow, objective = "TwoNN_all",
  w_min = W_MIN, step_grid = W_STEP_GRID, batch_k = W_BATCH_K, batch_factor = W_BATCH_FACTOR,
  max_iter = W_MAX_ITERS, n_rows_sub = N_ROWS_SUB, ncores = NCORES_PAR, seed_jitter = SEED_JITTER,
  reps_idx = reps, core_idx_rep = core_idx_rep, verbose = TRUE, plot_progress = TRUE
)

sel <- survivors_from_weights(
  w = wopt$weights, w_min = W_MIN,
  kmin = max(30L, ceiling(0.10 * length(wopt$weights))), make_plot = TRUE
)
w_full    <- sel$w_clamped
survivors <- sel$survivors
X_pred <- X[, survivors, drop = FALSE]; X <- X_pred
w_all  <- w_full[survivors]
cat(sprintf("[TwoNN optimiser] Survivors: p=%d\n", ncol(X_pred)))

PX      <- prep_X_for_gower(X, rare_prop = RARE_LEVEL_MIN_PROP, do_jitter = TRUE)
Xg      <- PX$X; type_g  <- PX$type
w_use   <- w_all[colnames(Xg)]

D_final <- cluster::daisy(Xg, metric = "gower", type = type_g, weights = w_use)
ID_all  <- twonn_id_from_dist(D_final)
core_ix <- twonn_core_by_slope(D_final)
DmF     <- as.matrix(D_final); diag(DmF) <- Inf
ID_core <- twonn_id_from_dist(as.dist(DmF[core_ix, core_ix, drop = FALSE]))
ID_LB   <- lb_mle_id(DmF[core_ix, core_ix, drop = FALSE], 5, 15)
cat(sprintf("[Constrained %s] TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f (n_core=%d, p_active=%d)\n",
            toupper(PREF_TARGET), ID_all, ID_core, ID_LB, length(core_ix), ncol(Xg)))

# ================= Base (m*=2), whitening, neighbours, residuals ===============
design_with_map <- function(X) {
  Xg <- as.data.frame(X, check.names = TRUE, stringsAsFactors = FALSE)
  if (!ncol(Xg)) stop("[design_with_map] input has 0 columns")
  keep <- vapply(Xg, function(v) length(unique(na.omit(v))) >= 1L, logical(1))
  if (!any(keep)) stop("[design_with_map] all columns are NA-only")
  Xg <- Xg[, keep, drop = FALSE]
  for (nm in names(Xg)) {
    v <- Xg[[nm]]
    if (is.ordered(v))      { Xg[[nm]] <- as.numeric(v); next }
    if (is.numeric(v) || is.integer(v)) next
    if (is.logical(v))      { Xg[[nm]] <- factor(v, levels = c(FALSE, TRUE)); next }
    if (is.factor(v))       next
    if (is.matrix(v))       { Xg[[nm]] <- as.numeric(v); next }
    Xg[[nm]] <- factor(as.character(v))
  }
  fml <- as.formula(paste("~", paste(colnames(Xg), collapse = " + "), "-1"))
  tm  <- terms(fml, data = Xg)
  MM  <- model.matrix(tm, data = Xg)
  storage.mode(MM) <- "double"
  ok <- apply(MM, 2L, function(col) { v <- stats::var(as.numeric(col), na.rm = TRUE); is.finite(v) && v > 1e-12 })
  if (!any(ok)) stop("[design_with_map] all encoded columns were ~zero-variance")
  assign <- attr(MM, "assign")
  tl     <- attr(tm, "term.labels")
  varmap <- tl[assign]
  MM <- MM[, ok, drop = FALSE]
  attr(MM, "varmap") <- varmap[ok]
  MM
}

Xenc   <- design_with_map(X)
varmap <- attr(Xenc, "varmap")
vars   <- unique(varmap)

# distribute Gower weights across encodings; sqrt-scale so PCA "feels" weights
w_enc <- setNames(rep(1, ncol(Xenc)), colnames(Xenc))
alloc <- table(varmap)
for (nm in names(alloc)) {
  idx <- which(varmap == nm)
  wj  <- w_all[nm]; if (!is.finite(wj)) wj <- 1
  w_enc[idx] <- wj / length(idx)
}
Xenc_w <- sweep(Xenc, 2, sqrt(pmax(w_enc, 0)), "*")

m_star <- as.integer(M_STAR_FIXED)
Z <- scale(Xenc_w, center = TRUE, scale = TRUE)
if (is.null(rownames(Z)) || all(grepl("^[0-9]+$", rownames(Z)))) {
  rn <- rownames(Xenc_w); if (!is.null(rn) && any(nzchar(rn))) rownames(Z) <- rn
}
pc   <- prcomp(Z, rank. = max(2L, min(m_star, nrow(Z) - 1L, ncol(Z))))
Base <- pc$x[, 1:m_star, drop = FALSE]
colnames(Base) <- paste0("b", seq_len(ncol(Base)))
rnB <- rownames(Z); if (!is.null(rnB) && any(nzchar(rnB))) rownames(Base) <- rnB

if (requireNamespace("ggplot2", quietly = TRUE)) {
  ve <- (pc$sdev[seq_len(ncol(Base))]^2) / sum(pc$sdev^2)
  cat(sprintf("[Base] m*=%d | per-PC var: %s | total (m*): %.3f\n",
              m_star, paste(round(ve, 3), collapse = ", "), round(sum(ve), 3)))
} else {
  cat(sprintf("[Base] m*=%d | total var (first m* PCs): %.3f\n",
              m_star, sum((pc$sdev[seq_len(ncol(Base))]^2) / sum(pc$sdev^2))))
}

# Whitening for neighbour search
S      <- stats::cov(Base)
U      <- try(chol(S + diag(1e-8, ncol(Base))), silent = TRUE)
if (inherits(U, "try-error")) {
  eig <- eigen(S, symmetric = TRUE)
  U   <- t(eig$vectors %*% diag(sqrt(pmax(eig$values, 1e-8))) %*% t(eig$vectors))
}
Base_w <- Base %*% solve(U)

# Neighbour sets on whitened Base
KS_FIBRE <- c(6, 8, 10, 12, 16, 20)
nb_list  <- setNames(lapply(KS_FIBRE, function(k){
  RANN::nn2(Base_w, Base_w, k = pmin(k + 1L, nrow(Base_w)))$nn.idx[, -1L, drop = FALSE]
}), paste0("k", KS_FIBRE))

# Fold-safe residualisation (GAM over Base)
residualise_foldsafe <- function(Xenc, Base, folds, k_gam = 6){
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
}

make_strat_folds <- function(y, K, group = NULL, seed = SEED_GLOBAL){
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
}
choose_K <- function(y, K_target = CV_FOLDS, min_per_class = 8){
  y <- as.integer(y); n1 <- sum(y == 1); n0 <- sum(y == 0)
  max(2, min(K_target, floor(n1 / min_per_class), floor(n0 / min_per_class)))
}

if (exists("y_use") && length(unique(y_use)) >= 2) {
  K_fold  <- choose_K(y_use, K_target = CV_FOLDS, min_per_class = 8)
  fold_id <- make_strat_folds(y_use, K = K_fold, seed = SEED_GLOBAL)
} else {
  set.seed(SEED_GLOBAL)
  K_fold  <- min(CV_FOLDS, nrow(Base))
  fold_id <- sample(rep(1:K_fold, length.out = nrow(Base)))
}

E         <- residualise_foldsafe(Xenc_w, Base, folds = fold_id, k_gam = 6)
E_scaled  <- scale(E, center = TRUE, scale = TRUE)
cat(sprintf("[Residuals] E matrix: %d rows × %d columns (post-OOF, scaled)\n",
            nrow(E_scaled), ncol(E_scaled)))

compute_pc_contrib <- function(Xenc, varmap, m) {
  out <- try({
    Z_  <- scale(Xenc, center = TRUE, scale = TRUE)
    rk  <- max(1L, min(m, ncol(Z_), nrow(Z_) - 1L))
    pc_ <- prcomp(Z_, center = FALSE, scale. = FALSE, rank. = rk)
    var_prop <- (pc_$sdev[seq_len(rk)]^2) / sum(pc_$sdev^2)
    per_pc   <- lapply(seq_len(rk), function(j) tapply(pc_$rotation[, j]^2, varmap, sum, default = 0))
    contr <- Reduce(`+`, Map(function(v, w) v * w, per_pc, var_prop))
    contr / sum(contr)
  }, silent = TRUE)
  if (inherits(out, "try-error") || is.null(out)) {
    vv <- unique(varmap); out <- setNames(rep(0, length(vv)), vv)
  }
  out
}
PC_contrib <- compute_pc_contrib(Xenc, varmap, m = m_star)

e_from_E <- function(nm, E_scaled, Z, varmap){
  idx <- which(varmap == nm)
  if (!length(idx)) return(rep(NA_real_, nrow(Z)))
  if (length(idx) == 1L) return(as.numeric(E_scaled[, idx]))
  pc1 <- try(prcomp(Z[, idx, drop = FALSE], rank. = 1), silent = TRUE)
  if (inherits(pc1, "try-error")) return(rep(NA_real_, nrow(Z)))
  as.numeric(as.matrix(E_scaled[, idx, drop = FALSE]) %*% pc1$rotation[, 1])
}
cat(sprintf("[Base] built with m*=%d; neighbours: %s; E columns: %d; vars: %d\n",
            m_star, paste(KS_FIBRE, collapse = ","), ncol(E_scaled), length(vars)))

# =================== Item roles: Base vs Fibre diagnostics =====================
Z0_std <- scale(Xenc, center = TRUE, scale = TRUE)

score_item_base <- function(nm, Z, varmap){
  idx <- which(varmap == nm)
  if (!length(idx)) return(rep(NA_real_, nrow(Z)))
  if (length(idx) == 1L) return(as.numeric(Z[, idx]))
  sc <- try(suppressWarnings(prcomp(Z[, idx, drop = FALSE], rank. = 1)$x[, 1]), silent = TRUE)
  if (inherits(sc, "try-error")) return(rep(NA_real_, nrow(Z)))
  as.numeric(sc)
}

r2_base_perm_stratified <- function(y, v, fold_id = NULL, B = CV_FOLDS,
                                    stratify = c("fold", "quantile"), Q = 10L,
                                    fit_fun = function(y, v) summary(lm(v ~ y))$r.squared) {
  stratify <- match.arg(stratify)
  n <- length(y)
  stopifnot(length(v) == n)
  
  # observed
  r2_obs <- fit_fun(y, v)
  
  # index sets
  if (stratify == "fold") {
    stopifnot(!is.null(fold_id), length(fold_id) == n)
    strata <- split(seq_len(n), fold_id)
  } else {
    qs <- cut(v, breaks = unique(quantile(v, probs = seq(0, 1, length.out = Q + 1), na.rm = TRUE)),
              include.lowest = TRUE, ordered_result = TRUE)
    strata <- split(seq_len(n), qs)
  }
  
  r2_perm <- numeric(B)
  for (b in seq_len(B)) {
    y_perm <- y
    for (ix in strata) {
      if (length(ix) > 1L) y_perm[ix] <- sample(y[ix], replace = FALSE)
    }
    r2_perm[b] <- fit_fun(y_perm, v)
  }
  
  # two-sided Monte Carlo p per Phipson & Smyth
  p <- (1 + sum(r2_perm >= r2_obs)) / (B + 1)
  list(r2_obs = r2_obs, r2_perm = r2_perm, p = p)
}

r2_base_perm_stratified_base <- function(Base, v, fold_id = NULL, B = 1000L,
                                         stratify = c("fold","quantile"), Q = 10L) {
  stratify <- match.arg(stratify)
  stopifnot(nrow(Base) == length(v), ncol(Base) == 2)
  # observed R² from additive smooth ≈ linear in b1,b2
  r2_obs <- summary(lm(v ~ Base[,1] + Base[,2]))$r.squared
  # strata
  n <- length(v)
  if (stratify == "fold") {
    stopifnot(!is.null(fold_id), length(fold_id) == n)
    strata <- split(seq_len(n), fold_id)
  } else {
    qs <- cut(v, breaks = unique(quantile(v, probs = seq(0,1,length.out = Q+1), na.rm = TRUE)),
              include.lowest = TRUE, ordered_result = TRUE)
    strata <- split(seq_len(n), qs)
  }
  r2_null <- numeric(B)
  for (b in seq_len(B)) {
    v_perm <- v
    for (ix in strata) if (length(ix) > 1L) v_perm[ix] <- sample(v[ix], replace = FALSE)
    r2_null[b] <- summary(lm(v_perm ~ Base[,1] + Base[,2]))$r.squared
  }
  p <- (1 + sum(r2_null >= r2_obs)) / (B + 1)  # Phipson-Smyth
  c(R2 = r2_obs, p = p)
}

r2_fibre_cv <- function(e, nb, folds = CV_FOLDS, seed = SEED_GLOBAL){
  set.seed(seed)
  n <- length(e)
  if (n < 6 || all(!is.finite(e))) return(NA_real_)
  fold_id <- sample(rep(1:folds, length.out = n))
  pred <- rep(NA_real_, n)
  for (f in 1:folds){
    te <- which(fold_id == f)
    e_mask <- e; e_mask[te] <- NA
    pr <- rowMeans(matrix(e_mask[nb], nrow = n), na.rm = TRUE)
    pr[is.na(pr)] <- mean(e[-te], na.rm = TRUE)
    pred[te] <- pr[te]
  }
  ve  <- stats::var(e, na.rm = TRUE)
  mse <- mean((e - pred)^2, na.rm = TRUE)
  if (!is.finite(ve) || ve <= 0) return(NA_real_)
  max(0, 1 - mse/ve)
}
choose_k_nb <- function(e, nb_list, folds = CV_FOLDS, seed = SEED_GLOBAL){
  r2s <- vapply(nb_list, function(nb) r2_fibre_cv(e, nb, folds, seed), numeric(1))
  ix  <- which.max(r2s)
  list(k = as.integer(sub("^k","", names(nb_list)[ix])), R2_cv = r2s[ix], nb = nb_list[[ix]], all = r2s)
}
r2_fibre_perm_rowshuffle <- function(e, nb, B = BOOT_B, seed = SEED_GLOBAL){
  set.seed(seed)
  if (!any(is.finite(e))) return(c(R2 = NA_real_, p = NA_real_))
  pred <- rowMeans(matrix(e[nb], nrow = length(e)), na.rm = TRUE)
  ve   <- stats::var(e, na.rm = TRUE)
  mse  <- mean((e - pred)^2, na.rm = TRUE)
  R2obs <- if (is.finite(ve) && ve > 0) max(0, 1 - mse/ve) else NA_real_
  if (!is.finite(R2obs) || R2obs <= 1e-12) return(c(R2 = 0, p = 1))
  R2null <- numeric(B)
  for (b in seq_len(B)){
    ix <- sample.int(nrow(nb))
    pred_b <- rowMeans(matrix(e[nb[ix, , drop = FALSE]], nrow = length(e)), na.rm = TRUE)
    R2null[b] <- max(0, 1 - mean((e - pred_b)^2, na.rm = TRUE)/ve)
  }
  p <- (sum(is.finite(R2null) & R2null >= R2obs) + 1) / (sum(is.finite(R2null)) + 1)
  c(R2 = R2obs, p = p)
}

permute_knn_circular <- function(nn_idx) {
  nn_idx <- as.matrix(nn_idx)
  n  <- nrow(nn_idx); k <- ncol(nn_idx)
  off <- sample.int(k, size = n, replace = TRUE) - 1L
  out <- nn_idx
  for (i in seq_len(n)) {
    if (off[i] == 0L) next
    out[i, ] <- c(nn_idx[i, (off[i]+1):k], nn_idx[i, 1:off[i]])
  }
  out
}

r2_fibre_perm_circular <- function(e, nb, B = BOOT_B, seed = SEED_GLOBAL){
  set.seed(seed)
  n  <- length(e); ve <- stats::var(e, na.rm = TRUE)
  if (!is.finite(ve) || ve <= 0) return(c(R2 = NA_real_, p = NA_real_))
  pred_obs <- rowMeans(matrix(e[nb], nrow = n), na.rm = TRUE)
  R2obs <- max(0, 1 - mean((e - pred_obs)^2, na.rm = TRUE)/ve)
  if (R2obs <= 1e-12) return(c(R2 = 0, p = 1))
  R2null <- numeric(B)
  for (b in seq_len(B)){
    nb_b <- permute_knn_circular(nb)
    pred_b <- rowMeans(matrix(e[nb_b], nrow = n), na.rm = TRUE)
    R2null[b] <- max(0, 1 - mean((e - pred_b)^2, na.rm = TRUE)/ve)
  }
  p <- (sum(is.finite(R2null) & R2null >= R2obs) + 1) / (sum(is.finite(R2null)) + 1)
  c(R2 = R2obs, p = p)
}

FIBRE_PERM_B <- BOOT_B
MIN_SD_ITEM  <- 1e-6
roles_rows <- lapply(vars, function(nm){
  v <- score_item_base(nm, Z0_std, varmap)
  if (stats::sd(v, na.rm = TRUE) < MIN_SD_ITEM) return(NULL)
  rb <- try(r2_base_perm_stratified_base(Base, v,
                                         fold_id = fold_id,
                                         B = FIBRE_PERM_B,
                                         stratify = "fold"), silent = TRUE)
  R2b <- if (!inherits(rb,"try-error")) as.numeric(rb["R2"]) else NA_real_
  pb  <- if (!inherits(rb,"try-error")) as.numeric(rb["p"])  else NA_real_
  e_item <- e_from_E(nm, E_scaled, Z0_std, varmap)
  if (!any(is.finite(e_item))) return(NULL)
  sel <- choose_k_nb(e_item, nb_list, folds = CV_FOLDS, seed = SEED_GLOBAL)
  rf <- try(r2_fibre_perm_circular(e_item, sel$nb, B = FIBRE_PERM_B, seed = SEED_GLOBAL), silent = TRUE)
  R2f <- if (!inherits(rf,"try-error")) as.numeric(rf["R2"]) else NA_real_
  pf  <- if (!inherits(rf,"try-error")) as.numeric(rf["p"])  else NA_real_
  pc_c <- if (!is.null(PC_contrib) && (nm %in% names(PC_contrib))) as.numeric(PC_contrib[[nm]]) else 0
  data.frame(var = nm, R2_base = R2b, p_base = pb, R2_fibre = R2f, p_fibre = pf,
             k_fibre = sel$k, PC_contrib = pc_c, stringsAsFactors = FALSE)
})
roles_df <- dplyr::bind_rows(Filter(Negate(is.null), roles_rows))
if (!nrow(roles_df)) stop("[roles] No items produced valid role stats.")

roles_df$R2_base  <- pmax(0, pmin(1, as.numeric(roles_df$R2_base)))
roles_df$R2_fibre <- pmax(0, pmin(1, as.numeric(roles_df$R2_fibre)))
roles_df$q_base  <- p.adjust(roles_df$p_base,  method = "BH")
roles_df$q_fibre <- p.adjust(roles_df$p_fibre, method = "BH")

roles_df$role_fdr <- with(
  roles_df,
  ifelse(q_base  < SIG_Q & (is.na(q_fibre) | q_fibre >= SIG_Q), "base-aligned",
         ifelse(q_fibre < SIG_Q & (is.na(q_base)  | q_base  >= SIG_Q), "fibre-structured",
                ifelse(q_base  < SIG_Q & q_fibre < SIG_Q,                     "mixed", "weak")))
)

ES_BASE  <- 0.08
ES_FIBRE <- 0.05
roles_df$role_es <- dplyr::case_when(
  roles_df$R2_base  >= ES_BASE  & (is.na(roles_df$R2_fibre) | roles_df$R2_fibre <  ES_FIBRE) ~ "base-aligned",
  roles_df$R2_fibre >= ES_FIBRE & (is.na(roles_df$R2_base)  | roles_df$R2_base  <  ES_BASE ) ~ "fibre-structured",
  roles_df$R2_base  >= ES_BASE  & roles_df$R2_fibre >= ES_FIBRE                               ~ "mixed",
  TRUE                                                                                         ~ "weak"
)

# Prefer ES scheme; if no base-aligned appear, fall back to FDR
role_col <- "role_es"
if (!any(roles_df[[role_col]] == "base-aligned", na.rm = TRUE)) role_col <- "role_fdr"
roles_df$role_final <- factor(roles_df[[role_col]],
                              levels = c("base-aligned","mixed","fibre-structured","weak"))
roles_df <- roles_df[order(-roles_df$PC_contrib, -roles_df$R2_base, roles_df$var), ]
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
    ggplot2::labs(title = "Item roles: Base vs Fibre", x = "R² to Base", y = "R² to fibre neighbours", color = "role") +
    ggplot2::theme_minimal()
  print(pV)
  ggplot2::ggsave("FIG_roles_volcano_base_vs_fibre.png", pV, width = 7, height = 5, dpi = 150)
}

# ================= Fibre-only decomposition & clustering ======================
stopifnot(exists("E_scaled"), is.matrix(E_scaled), nrow(E_scaled) >= 3)
Ef <- scale(E_scaled, center = TRUE, scale = TRUE); Ef[!is.finite(Ef)] <- 0
n  <- nrow(Ef); pE <- ncol(Ef)
if (pE < 2L || n < 4L) stop("[Fibre-only] insufficient columns/rows in E.")
FIBRE_BASE_MAX <- min(6L, pE, n - 1L)

pc_f <- prcomp(Ef, rank. = max(2L, FIBRE_BASE_MAX))
Bprime_all <- pc_f$x[, 1:FIBRE_BASE_MAX, drop = FALSE]
colnames(Bprime_all) <- paste0("f", seq_len(ncol(Bprime_all)))

if (requireNamespace("ggplot2", quietly = TRUE)) {
  ve <- (pc_f$sdev^2) / sum(pc_f$sdev^2)
  df_scree <- data.frame(PC = seq_along(ve), var = ve)
  p_scree <- ggplot2::ggplot(df_scree, ggplot2::aes(PC, var)) +
    ggplot2::geom_col(width = 0.9) +
    ggplot2::geom_vline(xintercept = FIBRE_BASE_MAX + 0.5, linetype = 2) +
    ggplot2::labs(title = "Fibre PCA scree", y = "Variance explained") +
    ggplot2::theme_minimal()
  print(p_scree); ggplot2::ggsave("FIG_fibre_scree.png", p_scree, width = 6, height = 7, dpi = 150)
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

tc_tbl <- lapply(2:ncol(Bprime_all), function(m){
  tc <- trust_cont_avg(Ef, Bprime_all[, 1:m, drop = FALSE], ks = KS_TC)
  data.frame(m_f = m, Trust = tc["T"], Continuity = tc["C"])
}) |> dplyr::bind_rows()
cat(sprintf("[Fibre-only] m_f* = %d (mean Trust=%.3f, Continuity=%.3f at m_f*)\n",
            m_f, mean(tc_tbl$Trust[tc_tbl$m_f == m_f]), mean(tc_tbl$Continuity[tc_tbl$m_f == m_f])))

Df      <- stats::dist(Ef)
IDf_all <- twonn_id_from_dist(Df)
Mf      <- as.matrix(Df); diag(Mf) <- Inf
core_f  <- core_band_idx(Df, k = CORE_KNN_K, band = CORE_BAND)
IDf_core<- if (length(core_f) >= 3) twonn_id_from_dist(stats::as.dist(Mf[core_f, core_f])) else NA_real_
IDf_LB  <- if (length(core_f) >= 3) lb_mle_id(Mf[core_f, core_f, drop = FALSE], 5, 15) else NA_real_

Bprime <- Bprime_all[, 1:m_f, drop = FALSE]
colnames(Bprime) <- paste0("f", seq_len(ncol(Bprime)))

# OOF residuals of E on B′ → F′ (linear)
set.seed(SEED_PRED)
Kf      <- max(2L, min(CV_FOLDS, n))
folds_f <- sample(rep(1:Kf, length.out = n))
residualise_linear_oof <- function(Y, X, folds) {
  n <- nrow(Y); p <- ncol(Y)
  R <- matrix(NA_real_, n, p, dimnames = list(rownames(Y), colnames(Y)))
  Xdf <- as.data.frame(X)
  for (j in seq_len(p)) {
    y <- Y[, j]; if (all(!is.finite(y))) next
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

cat(sprintf("[Fibre-only] ID(E):  TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f (n_core=%d)\n",
            IDf_all, IDf_core, IDf_LB, length(core_f)))
cat(sprintf("[Fibre-only] ID(B′): TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f (n_core=%d)\n",
            ID_B_all, ID_B_core, ID_B_LB, length(core_B)))
cat(sprintf("[Fibre-only] ID(F′): TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f (n_core=%s)\n",
            ID_Fp_all, ID_Fp_core, ID_Fp_LB, if (exists("core_Fp")) length(core_Fp) else "NA"))

# Clustering in B′ (Louvain on kNN graph) - optional if igraph present
if (requireNamespace("igraph", quietly = TRUE)) {
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
  
  if (requireNamespace("ggplot2", quietly = TRUE) && ncol(Bprime) >= 2L) {
    df_sc <- data.frame(f1 = Bprime[,1], f2 = Bprime[,2], cl = factor(clF))
    p_sc  <- ggplot2::ggplot(df_sc, ggplot2::aes(f1, f2, colour = cl)) +
      ggplot2::geom_point() +
      ggplot2::labs(title = "Fibre space (first 2 PCs)", x = "f1", y = "f2", colour = "cluster") +
      ggplot2::theme_minimal()
    print(p_sc); ggplot2::ggsave("FIG_fibre_scatter.png", p_sc, width = 6, height = 7, dpi = 150)
  }
}

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

# =================== Per-diagnosis predictive evaluation (OOF) ================
# Prereqs in env: Base (2 cols), diag_wide_full (id + dx columns), optional E_scaled
stopifnot(exists("Base"), is.matrix(Base),
          exists("diag_wide_full"))

if (!requireNamespace("pROC", quietly = TRUE))
  message("[dx] pROC not available — AUC falls back to rank U-statistic.")

wmw_p <- function(y, prob){
  y <- as.integer(y > 0); prob <- as.numeric(prob)
  ok <- is.finite(y) & is.finite(prob); y <- y[ok]; prob <- prob[ok]
  if (sum(y==1)==0 || sum(y==0)==0) return(NA_real_)
  suppressWarnings(tryCatch(
    stats::wilcox.test(prob[y==1], prob[y==0], alternative = "greater")$p.value,
    error = function(e) NA_real_
  ))
}

# ---- Align wide dx labels to Base row order ----------------------------------
ids_base <- trimws(rownames(Base))
stopifnot(length(ids_base) == nrow(Base), all(nzchar(ids_base)))

DxW <- diag_wide_full %>%
  dplyr::transmute(participant_id = trimws(as.character(participant_id)),
                   dplyr::across(-participant_id, ~ as.integer(.x))) %>%
  dplyr::right_join(tibble::tibble(participant_id = ids_base, .row = seq_along(ids_base)),
                    by = "participant_id") %>%
  dplyr::arrange(.row) %>%
  dplyr::select(-participant_id, -.row)

DxW[is.na(DxW)] <- 0L
match_rate <- mean(ids_base %in% trimws(as.character(diag_wide_full$participant_id)))
cat(sprintf("[dx] ID match rate: %.1f%% (%d/%d)\n",
            100*match_rate, sum(ids_base %in% diag_wide_full$participant_id), length(ids_base)))

prev    <- colMeans(DxW > 0, na.rm = TRUE)
cases   <- colSums(DxW > 0, na.rm = TRUE)
keep_dx <- names(prev)[(prev >= DX_PREV_MIN) & (cases >= DX_CASES_MIN)]
if (!length(keep_dx)) {
  warning("[dx] No diagnosis passes thresholds; skipping predictive diagnostics.")
  keep_dx <- intersect(names(prev), names(prev)[cases > 0 & (1 - prev) * nrow(DxW) > 0])
}

DxW_A  <- as.data.frame(DxW)[, keep_dx, drop = FALSE]
rownames(DxW_A) <- ids_base

# ---- Features to match dimension_plots.R -------------------------------------
# Base_A: the two base coordinates the rest of the code uses (“b1”,“b2”)
Base_A <- as.data.frame(Base[, 1:2, drop = FALSE])
colnames(Base_A) <- c("b1","b2")

# XR: optional “fibre” block; if E_scaled doesn’t exist, Fibre/Both are omitted.
XR <- if (exists("E_scaled") && is.matrix(E_scaled) && ncol(E_scaled) > 0) as.data.frame(E_scaled) else NULL

# ---- Helpers copied from dimension_plots.R (lightly inlined) -----------------
fit_glm_or_glmnet <- function(y, X){
  df <- as.data.frame(X); df$.y <- as.integer(y>0)
  if (requireNamespace("glmnet", quietly = TRUE)){
    x <- as.matrix(df[setdiff(names(df), ".y")]); yb <- df$.y
    cv <- glmnet::cv.glmnet(x, yb, alpha = 0, family = "binomial",
                            standardize = TRUE, nfolds = 5)
    list(type = "glmnet", fit = cv, xnames = colnames(x))
  } else {
    # tiny jitter to avoid complete separation
    if (ncol(df) > 1L){
      for (j in setdiff(seq_along(df), which(names(df)==".y"))){
        v <- df[[j]]; if (is.numeric(v)) df[[j]] <- v + rnorm(length(v), 0, 1e-8)
      }
    }
    list(type = "glm",
         fit = stats::glm(.y ~ ., data = df, family = stats::binomial(),
                          control = list(maxit = 50)))
  }
}
pred_prob <- function(mod, newX){
  if (mod$type == "glmnet") {
    x <- as.matrix(as.data.frame(newX)[, mod$xnames, drop = FALSE])
    p <- stats::predict(mod$fit, x, s = "lambda.min", type = "response")
  } else {
    p <- stats::predict(mod$fit, newdata = as.data.frame(newX), type = "response")
  }
  as.numeric(pmin(pmax(p, 1e-6), 1 - 1e-6))
}
make_folds <- function(y, K, seed){
  set.seed(seed); y <- as.integer(y>0)
  i1 <- which(y==1); i0 <- which(y==0)
  f1 <- sample(rep(seq_len(K), length.out = length(i1)))
  f0 <- sample(rep(seq_len(K), length.out = length(i0)))
  fid <- integer(length(y)); fid[i1] <- f1; fid[i0] <- f0; fid
}
oof_prob_stacked <- function(y, Base_A, XR = NULL, K, seed){
  y  <- as.integer(y > 0)
  Xb <- as.data.frame(Base_A)
  Xr <- if (!is.null(XR)) as.data.frame(XR) else NULL
  fid <- make_folds(y, K, seed)
  
  # Level-1 (Base, Fibre) OOF
  pB <- rep(NA_real_, length(y))
  pR <- if (!is.null(Xr) && ncol(Xr) > 0) rep(NA_real_, length(y)) else NULL
  for (k in seq_len(K)){
    tr <- fid != k; te <- fid == k
    modB <- fit_glm_or_glmnet(y[tr], Xb[tr, , drop = FALSE])
    pB[te] <- pred_prob(modB, Xb[te, , drop = FALSE])
    if (!is.null(pR)){
      modR <- fit_glm_or_glmnet(y[tr], Xr[tr, , drop = FALSE])
      pR[te] <- pred_prob(modR, Xr[te, , drop = FALSE])
    }
  }
  out <- list(Base = pB)
  
  # Level-2 (stacking on logits) only if Fibre exists
  if (!is.null(pR)){
    tologit <- function(p) qlogis(pmin(pmax(p, 1e-6), 1 - 1e-6))
    pBR <- rep(NA_real_, length(y))
    for (k in seq_len(K)){
      tr <- fid != k; te <- fid == k
      Xm_tr <- data.frame(l1 = tologit(pB[tr]), l2 = tologit(pR[tr]))
      m     <- stats::glm(y[tr] ~ ., data = Xm_tr, family = binomial())
      pBR[te] <- as.numeric(stats::predict(
        m, newdata = data.frame(l1 = tologit(pB[te]), l2 = tologit(pR[te])),
        type = "response"))
    }
    out$Fibre <- pR
    out$Both  <- pmin(pmax(pBR, 1e-6), 1 - 1e-6)
  }
  out
}
auc_point <- function(y, p){
  y <- as.integer(y > 0); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  if (length(y) < 2L || !any(y == 1L) || all(y == 1L)) return(NA_real_)
  if (requireNamespace("pROC", quietly = TRUE)) {
    r <- try(pROC::roc(response = factor(y, levels = c(0, 1)),
                       predictor = p, quiet = TRUE, direction = "auto"),
             silent = TRUE)
    if (!inherits(r, "try-error")) {
      a <- suppressWarnings(as.numeric(pROC::auc(r)))
      if (is.finite(a)) return(a)
    }
  }
  rk  <- rank(p, ties.method = "average")
  P   <- sum(y == 1); N <- sum(y == 0)
  auc <- (sum(rk[y == 1]) - P * (P + 1) / 2) / (P * N)
  max(auc, 1 - auc)
}
boot_ci <- function(y, p, FUN, B = BOOT_B, seed = SEED_GLOBAL){
  set.seed(seed)
  y <- as.integer(y > 0); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  i0 <- which(y==0L); i1 <- which(y==1L)
  pt <- as.numeric(FUN(y, p))
  if (!length(i0) || !length(i1)) return(c(point = pt, lo = NA_real_, hi = NA_real_))
  vals <- replicate(B, {
    ii <- c(sample(i0, length(i0), TRUE), sample(i1, length(i1), TRUE))
    as.numeric(FUN(y[ii], p[ii]))
  })
  vf <- vals[is.finite(vals)]
  if (!length(vf)) return(c(point = pt, lo = NA_real_, hi = NA_real_))
  c(point = pt,
    lo = as.numeric(stats::quantile(vf, 0.025, names = FALSE)),
    hi = as.numeric(stats::quantile(vf, 0.975, names = FALSE)))
}

# ---- Per-diagnosis AUCs (unified) --------------------------------------------
rows <- list()
for (dx in names(DxW_A)){
  y <- as.integer(DxW_A[[dx]] > 0)
  if (length(unique(y)) < 2L) next
  
  # Same K rule as dimension_plots.R::run_dx_metrics
  K <- max(2L, min(CV_FOLDS,
                   floor(sum(y==1) / CV_FOLDS),
                   floor(sum(y==0) / CV_FOLDS)))
  
  models <- tryCatch(
    oof_prob_stacked(y, Base_A, XR, K = K, seed = SEED_GLOBAL),
    error = function(e) { warning("[dx] OOF failure for ", dx, ": ", conditionMessage(e)); list(Base = rep(mean(y), length(y))) }
  )
  
  for (m in names(models)){
    prob <- models[[m]]
    ci   <- boot_ci(y, prob, FUN = auc_point, B = BOOT_B, seed = SEED_GLOBAL)
    pv   <- wmw_p(y, prob)
    
    rows[[length(rows)+1L]] <- data.frame(
      dx = dx, model = m, prevalence = mean(y),
      AUC = as.numeric(ci["point"]),
      lo  = as.numeric(ci["lo"]),
      hi  = as.numeric(ci["hi"]),
      p_wmw = pv,
      stringsAsFactors = FALSE
    )
  }
}

AUC_UNIFIED <- if (length(rows)) dplyr::bind_rows(rows) else {
  warning("[dx] No usable diagnoses after gating; nothing to plot."); data.frame()
}

# ---- Absolute AUC by model figure (unified with dimension_plots.R) -----------
if (nrow(AUC_UNIFIED)) {
  # Optional pretty ordering (keeps your old preferred names if present)
  desired_order <- c(
    "No Diagnosis on Axis I",
    "Attention-Deficit/Hyperactivity Disorder",
    "Bipolar I Disorder",
    "Depressive Disorder NOS",
    "Major Depressive Disorder",
    "Schizophrenia",
    "Schizoaffective Disorder",
    "Alcohol Abuse","Alcohol Dependence",
    "Amphetamine Abuse","Amphetamine Dependence",
    "Cannabis Abuse","Cannabis Dependence",
    "Cocaine Abuse","Cocaine Dependence"
  )
  dx_present <- unique(AUC_UNIFIED$dx)
  ord_levels <- rev(c(intersect(desired_order, dx_present),
                      setdiff(dx_present, desired_order)))
  AUC_UNIFIED <- AUC_UNIFIED %>%
    dplyr::mutate(model = factor(model, levels = c("Base","Fibre","Both"))) %>%
    dplyr::group_by(model) %>%
    dplyr::mutate(q = ifelse(is.finite(.data$p_wmw),
                             p.adjust(.data$p_wmw, method = "BH"),
                             NA_real_)) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(sig = dplyr::case_when(
      is.finite(q) & q < 0.05 ~ "q<0.05",
      is.finite(lo) & is.finite(hi) & lo > 0.5 ~ "q<0.05",  # fallback
      TRUE ~ "ns"
    ))
  
  if (exists("desired_order")) {
    dx_present <- unique(AUC_UNIFIED$dx)
    ord_levels <- rev(c(intersect(desired_order, dx_present),
                        setdiff(dx_present, desired_order)))
    AUC_UNIFIED$dx <- factor(AUC_UNIFIED$dx, levels = ord_levels)
  }
  
  pal_sig <- c("ns" = "#9AA0A6",        # muted grey
               "q<0.05" = "#2F3B52")    # dark blue-grey
  
  abs_pts <- AUC_UNIFIED %>% dplyr::filter(is.finite(AUC))
  abs_cis <- abs_pts       %>% dplyr::filter(is.finite(lo), is.finite(hi))
  
  p_abs <- ggplot2::ggplot(abs_pts,
                           ggplot2::aes(x = AUC, y = dx, xmin = lo, xmax = hi, colour = sig)) +
    ggplot2::geom_vline(xintercept = 0.5, linetype = 2, linewidth = 0.4) +
    ggplot2::geom_errorbarh(data = abs_cis, height = 0, linewidth = 0.8, alpha = 0.95) +
    ggplot2::geom_point(size = 2.6, stroke = 0) +
    ggplot2::facet_grid(. ~ model) +
    ggplot2::scale_colour_manual(values = pal_sig, breaks = names(pal_sig), name = NULL) +
    ggplot2::scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, by = 0.1),
                                labels = function(b) sprintf("%.1f", b),
                                expand = ggplot2::expansion(mult = c(0.02, 0.02))) +
    ggplot2::labs(x = NULL, y = NULL) +
    ggplot2::theme_minimal(base_size = 16) +
    ggplot2::theme(panel.grid.major.y = ggplot2::element_blank(),
                   panel.grid.minor   = ggplot2::element_blank(),
                   strip.background   = ggplot2::element_blank(),
                   strip.text.x       = ggplot2::element_text(face = "italic", size = 16),
                   axis.text.y        = ggplot2::element_text(size = 12, lineheight = 0.9),
                   axis.text.x        = ggplot2::element_text(size = 11),
                   panel.spacing.x    = grid::unit(16, "pt"),
                   legend.position    = "right")
  
  print(p_abs)
  ggplot2::ggsave("FIG_auc_absolute_by_model.png", p_abs, width = 11, height = 4.8, dpi = 300)
  readr::write_csv(AUC_UNIFIED, "dx_auc_base_fibre_stacked_with_pr.csv")
} else {
  message("[dx] Skipped FIG_auc_absolute_by_model — no rows to plot.")
}
# ==============================================================================

# =============== Diagnosis probability fields over Base(b1,b2) =================
stopifnot(ncol(Base) == 2)
dir.create("base_prob_grids", showWarnings = FALSE)

grid_from_base <- function(Base, nx = 140, ny = 140, pad = 0.05){
  rx <- range(Base[,1]); ry <- range(Base[,2]); wx <- diff(rx); wy <- diff(ry)
  xs <- seq(rx[1] - pad*wx, rx[2] + pad*wx, length.out = nx)
  ys <- seq(ry[1] - pad*wy, ry[2] + pad*wy, length.out = ny)
  as.matrix(expand.grid(b1 = xs, b2 = ys))
}
predict_dx_surface <- function(y, Base, gridXY, k_gam = 30){
  df <- data.frame(y = as.integer(y), b1 = Base[,1], b2 = Base[,2])
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
cat(sprintf("[base-field] wrote %d grid(s) to base_prob_grids/\n", length(dx_surface_files)))

if (requireNamespace("ggplot2", quietly = TRUE) && length(dx_surface_files)) {
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
        ggplot2::facet_wrap(~ dx, scales = "fixed") +
        ggplot2::scale_fill_viridis_c(limits = c(0, 1)) +
        ggplot2::coord_equal(expand = FALSE) +
        ggplot2::labs(title = "Diagnosis probability fields over Base", fill = "P(dx)") +
        ggplot2::theme_minimal()
      print(pH); ggplot2::ggsave("FIG_base_prob_fields.png", pH, width = 10, height = 8, dpi = 150)
    }
  }
}

# ======== Item × diagnosis interaction over Base (OOF GAM-ANOVA) ===============
stopifnot(exists("Base"), ncol(Base) == 2, exists("vars"))
if (!exists("PC_contrib")) PC_contrib <- setNames(rep(0, length(vars)), vars)
vars_ranked <- names(sort(PC_contrib[vars], decreasing = TRUE))
vars_probe  <- head(vars_ranked, min(N_TOP_PER_DX, length(vars_ranked)))

choose_K_dx <- function(y, K_target = 5L, min_per_class = 6L){
  y <- as.integer(y > 0); n1 <- sum(y==1); n0 <- sum(y==0)
  max(2L, min(K_target, floor(n1/min_per_class), floor(n0/min_per_class)))
}

oof_R2_two_gams <- function(v, Base, dx, K_target = 5, k_gam = 10,
                            seed = SEED_GLOBAL){
  n <- length(v)
  if (n != nrow(Base) || n != length(dx)) return(
    c(R2_add = NA, R2_int = NA, p_like = NA, dR2 = NA))
  
  # stratified K per dx
  K <- choose_K_dx(dx, K_target = K_target, min_per_class = 6L)
  set.seed(seed)
  fold_id <- sample(rep(1:K, length.out = n))
  
  # Harmonize factor levels so predict() never sees unseen levels
  lev_all <- levels(factor(dx))
  
  y_add <- rep(NA_real_, n)
  y_int <- rep(NA_real_, n)
  
  for (k in sort(unique(fold_id))) {
    tr <- which(fold_id != k); te <- which(fold_id == k)
    dtr <- data.frame(v=v[tr], b1=Base[tr,1], b2=Base[tr,2],
                      dx=factor(dx[tr], levels=lev_all))
    dte <- data.frame(b1=Base[te,1], b2=Base[te,2],
                      dx=factor(dx[te], levels=lev_all))
    
    ctrl <- list(maxit = 100)
    f_add <- try(mgcv::gam(v ~ s(b1,b2,k=k_gam, bs="tp", m=2),
                           data=dtr, method="REML", gamma=1.4, control=ctrl), silent=TRUE)
    f_int <- try(mgcv::gam(v ~ s(b1,b2,k=k_gam, bs="tp", m=2) + dx +
                             s(b1,b2, by=dx, k=k_gam, bs="tp", m=2),
                           data=dtr, method="REML", select=TRUE, gamma=1.4, control=ctrl), silent=TRUE)
    
    mu_fallback <- mean(dtr$v, na.rm = TRUE)
    if (!inherits(f_add, "try-error")) {
      pa <- try(predict(f_add, newdata=dte, type="response"), silent=TRUE)
      if (!inherits(pa, "try-error")) y_add[te] <- as.numeric(pa)
    }
    if (!inherits(f_int, "try-error")) {
      pi <- try(predict(f_int, newdata=dte, type="response"), silent=TRUE)
      if (!inherits(pi, "try-error")) y_int[te] <- as.numeric(pi)
    }
    if (any(!is.finite(y_add[te]))) y_add[te][!is.finite(y_add[te])] <- mu_fallback
    if (any(!is.finite(y_int[te]))) y_int[te][!is.finite(y_int[te])] <- mu_fallback
  }
  
  ve <- stats::var(v, na.rm = TRUE)
  if (!is.finite(ve) || ve <= 0) return(c(R2_add=NA, R2_int=NA, p_like=NA, dR2=NA))
  
  # OOF R²
  R2_add <- max(0, 1 - mean((v - y_add)^2, na.rm = TRUE) / ve)
  R2_int <- max(0, 1 - mean((v - y_int)^2, na.rm = TRUE) / ve)
  
  # Paired error-difference test (more power than bootstrapping ΔR²)
  d_sq <- (v - y_add)^2 - (v - y_int)^2
  d_sq <- d_sq[is.finite(d_sq)]
  if (length(d_sq) < 10L || all(abs(d_sq) < .Machine$double.eps)) {
    p_like <- NA_real_
  } else {
    # H1: interaction has *lower* error → median(d_sq) > 0
    p_like <- tryCatch(
      stats::wilcox.test(d_sq, mu = 0, alternative = "greater", exact = FALSE)$p.value,
      error = function(e) NA_real_)
  }
  
  c(R2_add = R2_add, R2_int = R2_int, p_like = p_like, dR2 = R2_int - R2_add)
}

rows <- list()
for (dx in keep_dx) {
  ydx <- as.integer(DxW[[dx]] > 0)
  for (nm in vars_probe) {
    v <- score_item_base(nm, scale(Xenc, TRUE, TRUE), varmap)
    if (!any(is.finite(v)) || stats::sd(v, na.rm = TRUE) < 1e-6) next
    res <- oof_R2_two_gams(v, Base, ydx, K_target = 5, k_gam = 10, seed = SEED_GLOBAL)
    rows[[length(rows)+1]] <- data.frame(
      dx   = dx, var = nm,
      R2_add = unname(res["R2_add"]),
      R2_int = unname(res["R2_int"]),
      dR2    = unname(res["dR2"]),
      p_like = unname(res["p_like"]),
      stringsAsFactors = FALSE
    )
  }
}

int_tbl <- dplyr::bind_rows(rows) %>%
  dplyr::filter(is.finite(dR2), is.finite(p_like))

if (!nrow(int_tbl)) {
  warning("[interactions] empty after cleaning — no valid item×dx fits.")
} else {
  # Per-dx BH
  int_tbl <- int_tbl %>%
    dplyr::group_by(dx) %>%
    dplyr::mutate(q_like = p.adjust(p_like, method = "BH")) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(sig = q_like < SIG_Q) %>%
    dplyr::arrange(dplyr::desc(dR2))
  
  readr::write_csv(int_tbl, "item_dx_interactions.csv")
  cat(sprintf("[interactions] wrote: item_dx_interactions.csv (rows=%d)\n", nrow(int_tbl)))
  
  if (requireNamespace("ggplot2", quietly = TRUE) && nrow(int_tbl)) {
    topK <- utils::head(int_tbl, 30)
    pI <- ggplot2::ggplot(topK,
                          ggplot2::aes(x = stats::reorder(paste(dx, var, sep=" · "), dR2), y = dR2, color = q_like < 0.10)) +
      ggplot2::geom_point() + ggplot2::coord_flip() +
      ggplot2::labs(title = "Top ΔR² (interaction gain) - item × dx over Base",
                    x = "dx · item", y = "ΔR² (interaction − additive)") +
      ggplot2::theme_minimal()
    print(pI); ggplot2::ggsave("FIG_item_dx_interaction_top.png", pI, width = 8, height = 10, dpi = 150)
  }
}

# =================== Cluster-level summaries & exports =========================
if (!exists("clF")) {
  message("[clusters] fibre clusters not found; clustering on Base as fallback.")
  idx <- RANN::nn2(Base, Base, k = pmin(16, nrow(Base)))$nn.idx[, -1]
  i <- rep(seq_len(nrow(Base)), each = ncol(idx)); j <- as.vector(idx)
  if (requireNamespace("igraph", quietly = TRUE)) {
    g <- igraph::graph_from_edgelist(cbind(i, j), directed = FALSE)
    g <- igraph::simplify(g)
    clF <- igraph::cluster_louvain(g)$membership
  } else {
    clF <- rep(1L, nrow(Base))
  }
}
kF <- length(unique(clF))
cat(sprintf("[clusters] using %d clusters.\n", kF))

clusters <- sort(unique(clF))
cl_fac   <- factor(clF, levels = clusters)

mean_by_cluster <- function(vec, clf) {
  out <- tapply(vec, clf, function(z) mean(z, na.rm = TRUE))
  as.numeric(out)
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
    f_means <- cbind(`f1_mean` = as.numeric(f_means))
  }
}
sum_tbl <- data.frame(cluster = clusters, n = n_by_cluster, base_means, check.names = FALSE, stringsAsFactors = FALSE)
if (!is.null(f_means)) { f_means <- as.matrix(f_means); storage.mode(f_means) <- "double"; sum_tbl <- cbind(sum_tbl, f_means, stringsAsFactors = FALSE) }
readr::write_csv(sum_tbl, "cluster_summary.csv")

# ---- Cluster × diagnosis enrichment (configurable, robust, no Z clobber) ----
# Gates (override if you like)
MIN_CASES_ENRICH <- DX_CASES_MIN
MIN_PROP_ENRICH  <- 0.01
ENRICH_Q         <- 0.05           # highlight cells with q < ENRICH_Q

DxW2 <- DxW
DxW2[is.na(DxW2)] <- 0L
stopifnot(nrow(DxW2) == length(clF))  # ensure alignment with clusters

prev  <- colSums(DxW2 > 0L, na.rm = TRUE)
keepc <- prev >= max(MIN_CASES_ENRICH, ceiling(MIN_PROP_ENRICH * nrow(DxW2)))
DxW2  <- DxW2[, keepc, drop = FALSE]

if (ncol(DxW2) > 0) {
  # counts per (cluster, diagnosis)
  tab <- sapply(colnames(DxW2), function(dn)
    tapply(DxW2[[dn]] > 0L, factor(clF), sum, na.rm = TRUE))
  tab <- as.matrix(tab); storage.mode(tab) <- "double"
  
  # standardized residuals (prefer chisq.test; fall back to manual)
  suppressWarnings({
    chi <- try(chisq.test(tab), silent = TRUE)
  })
  if (inherits(chi, "try-error")) {
    Eexp <- outer(rowSums(tab), colSums(tab), function(r, c) r * c / max(sum(tab), 1))
    Z_enrich <- (tab - Eexp) / sqrt(pmax(Eexp, 1e-9))
    pmat <- 2 * pnorm(-abs(Z_enrich))
    Eexp_used <- Eexp
  } else {
    Z_enrich  <- chi$stdres
    pmat      <- 2 * pnorm(-abs(Z_enrich))
    Eexp_used <- chi$expected
  }
  
  enrich <- as.data.frame(as.table(Z_enrich))
  names(enrich) <- c("cluster", "diagnosis", "std_resid")
  enrich$cluster  <- as.integer(as.character(enrich$cluster))
  enrich$count    <- as.vector(tab)
  enrich$expected <- as.vector(Eexp_used)
  enrich$p_cell   <- as.vector(pmat)
  enrich$q_cell   <- p.adjust(enrich$p_cell, method = "BH")
  
  readr::write_csv(enrich, "cluster_dx_enrichment.csv")
  cat("Wrote: cluster_summary.csv, cluster_dx_enrichment.csv\n")
  
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    pH <- ggplot2::ggplot(enrich, ggplot2::aes(diagnosis, factor(cluster), fill = std_resid)) +
      ggplot2::geom_tile(colour = "white") +
      ggplot2::scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
      ggplot2::labs(title = "Cluster × diagnosis enrichment (std residuals)",
                    x = "Diagnosis", y = "Cluster", fill = "Std resid") +
      ggplot2::theme_minimal() +
      ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
    
    # optional: mark significant cells (q < ENRICH_Q)
    if ("q_cell" %in% names(enrich)) {
      pH <- pH + ggplot2::geom_point(
        data = subset(enrich, is.finite(q_cell) & q_cell < ENRICH_Q),
        ggplot2::aes(x = diagnosis, y = factor(cluster)),
        inherit.aes = FALSE, shape = 21, stroke = 0.7, size = 2
      )
    }
    
    print(pH)
    ggplot2::ggsave("FIG_cluster_dx_enrichment.png", pH, width = 10, height = 6, dpi = 150)
  }
} else {
  cat("No diagnosis columns pass the enrichment threshold; skipped enrichment export.\n")
}