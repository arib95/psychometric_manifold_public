# =================================================================================================
# Psychometric manifold + diagnostics + PHATE dimensional sweep
# -------------------------------------------------------------------------------------------------
# Purpose
#   End-to-end pipeline to:
#     1) Load and coerce mixed psychometric variables into a consistent matrix.
#     2) Build a kNN graph from a Gower distance matrix and compute a Laplacian Eigenmaps embedding.
#     3) Derive unsupervised clusters and run stability and quality diagnostics.
#     4) Map diagnoses (multi-label, long table -> wide 0/1) onto LE space for association tests.
#     5) Fit a principal surface in the LE space and report projection error and cross-validated metrics.
#
# Inputs
#   - psychometric_matrix.csv
#       Delimited file with row-wise subjects and column-wise variables.
#       Must contain 'participant_id' or else the first column is treated as the ID.
#       May contain diagnosis columns beginning with 'diagnosis', which are dropped from X.
#       Mixed types allowed: numeric, integer-coded ordinal, character, factors.
#   - long_diagnoses.csv
#       Long table with at least:
#         participant_id: matches the ID column above (coercible to character)
#         diagnosis or Data: diagnosis label for a present diagnosis instance
#
# Outputs
#   - embedding_LE.csv
#   - clusters_unsupervised.csv
#   - permanova_dx_in_LE.csv
#   - principal_surface_coords_and_residuals.csv
#   - bundle_frames_nodes.rds
#   - principal_angles_nodes.csv, connection_edges.csv, curvature_map_uv.csv
#   - fibre_energy_by_subject.csv, unsupervised_risk_subjects.csv
#
# Key methods and references
#   - Gower distance for mixed data: cluster::daisy(metric = "gower")
#   - kNN graph with self-tuned Gaussian weights: Zelnik-Manor and Perona, NIPS 2004
#   - Laplacian Eigenmaps: Belkin and Niyogi, 2003
#   - PHATE embedding: Moon et al., Nat Biotechnol, 2019
#   - Thin-plate spline GAMs: Wood, Generalized Additive Models
#   - Principal curves: Hastie and Stuetzle, 1989
#
# Reproducibility and compute notes
#   - DO_* flags control which stages run.
#   - Random seeds fixed where appropriate.
#   - mgcv::bam uses fREML, single-thread, discrete=FALSE for numerical stability.
#   - PHATE runs via a Conda environment to keep R dependencies minimal.
#
# Rationale for selected defaults
#   - KNN_K = 10 balances connectivity and locality for n approximately 500 to 5000.
#   - HDBSCAN minPts grid c(8,10,12,15) favours solutions with more non-noise points.
#   - Modularity permutations B=200 provide light but informative evidence.
#   - Manifold GAM basis sizes k_basis=4, k_basis_hi=6 are conservative; CV guards optimism.
#   - CV folds K=3 for nd up to 4 and K=2 for nd=5 are compromises for runtime.
#
# Author: Afonso Dinis Ribeiro
# Date:   23-08-2025
# =================================================================================================

# =================================================================================================
# A) Environment, threads, libraries
# =================================================================================================

# Optional OpenMP set-up for Apple Silicon Homebrew libomp
omp_lib <- "/opt/homebrew/opt/libomp/lib/libomp.dylib"
if (file.exists(omp_lib)) {
  Sys.setenv(DYLD_FALLBACK_LIBRARY_PATH = dirname(omp_lib))
  try(dyn.load(omp_lib), silent = TRUE)
}

suppressPackageStartupMessages({
  library(readr);   library(dplyr);  library(tidyr);   library(utils)
  library(cluster); library(RANN);   library(RSpectra); library(dbscan)
  library(igraph);  library(ggplot2);library(Matrix);  library(aricode)
  library(glmnet);  library(vegan);  library(princurve);library(mgcv)
  library(reticulate);               library(MASS);     library(R.utils)
  library(FNN);     library(expm);   library(clue);     library(FactoMineR)
  library(scales)
  # plotly is optional and used only for a 3D surface visual
})

# Threads and BLAS environment
BAM_THREADS <- max(1, parallel::detectCores(logical = TRUE) - 1)
options(mc.cores = BAM_THREADS)
Sys.setenv(
  OMP_NUM_THREADS        = as.character(BAM_THREADS),
  OMP_PROC_BIND          = "spread",
  OPENBLAS_NUM_THREADS   = "1",
  MKL_NUM_THREADS        = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  KMP_SETTINGS           = "0",
  OMP_DISPLAY_ENV        = "FALSE"
)

# =================================================================================================
# B) Global switches & constants (single source of truth)
# =================================================================================================

# ---- Run switches ----
DO_PLOTS        <- TRUE     # figures for LE, density, dx surfaces, residual maps
DO_DIAGNOSTICS  <- TRUE     # modularity, silhouette, k-sensitivity, TwoNN ID
DO_SURFACE      <- TRUE     # principal surface on LE and residual export
DO_SWEEP        <- TRUE     # PHATE nd=1..5 with CV manifold fits + residual exports
RUN_PRINSURF    <- FALSE    # try 'prinsurf' package before GAM fallback

# ---- Randomness ----
SEED            <- 42
set.seed(SEED)
RNGkind("L'Ecuyer-CMRG")

# ---- Graph/embedding ----
KNN_K           <- 10       # k for Gower kNN graph (used consistently everywhere)

# ---- Diagnosis tests ----
MIN_N_DX        <- 25       # min per-class count for PERMANOVA etc.

# ---- Principal surface on LE (Section I & J) ----
K_BASIS_UV      <- 40       # basis size for s(u,v) in mgcv
SURFACE_CV_FOLDS<- 3        # K-fold CV for LE surface (also used in Section J)
USE_SURFACE_CV   <- TRUE

# ---- Bundle / fibre discovery (Section K) ----
BF_NGR_UV       <- 35       # grid resolution for (u,v)
BF_K_BASE       <- 100      # base-space kNN per node
BF_K_RANK_CAP   <- 4        # cap for fibre rank
BF_MP_SHRINK    <- 0.90     # MP threshold shrink
BF_MIN_N_NEIGH  <- 50       # min neighbourhood size
BF_DENS_K       <- 30       # k for kNN density on base
BF_SEED         <- 42

# =================================================================================================
# C) Utilities
# =================================================================================================

# Safe and stable bam wrapper with fREML and no 'discrete' approximation
safe_bam <- function(formula, data,
                     k_threads = 1,
                     gamma = 1.2,
                     select = TRUE) {
  stopifnot(all(vapply(data, function(z) all(is.finite(z)), logical(1))))
  fit <- try(mgcv::bam(
    formula, data = data,
    method   = "fREML",
    discrete = FALSE,
    nthreads = k_threads,
    select   = select,
    gamma    = gamma,
    optimizer = c("outer","bfgs"),
    control   = mgcv::gam.control(maxit = 100)
  ), silent = TRUE)
  if (!inherits(fit, "try-error")) return(fit)
  mgcv::gam(formula, data = data, method = "REML", select = select, gamma = gamma)
}

# Simple type detectors for coercion
is_binary <- function(v){
  u <- sort(unique(na.omit(as.numeric(v))))
  length(u) == 2 && all(u %in% c(0,1))
}
is_small_int <- function(v){
  u <- sort(unique(na.omit(as.numeric(v))))
  k <- length(u)
  k >= 3 && k <= 7 && all(abs(u - round(u)) < 1e-8)
}

# Totals for R2-like metrics and a timer
r2_tot <- function(Y) sum(scale(Y, scale = FALSE)^2)
.timeit <- function(expr, msg = NULL){
  t0 <- proc.time()[3]; out <- force(expr); t1 <- proc.time()[3]
  if (!is.null(msg)) cat(msg, sprintf("(%.2fs)\n", t1 - t0))
  out
}

# Single consolidated kNN graph builder with self-tuned Gaussian weights
knn_graph <- function(Dm, ord_idx, k = 10) {
  n <- nrow(Dm)
  k_eff <- pmin(k, n - 1L)
  I <- rep(seq_len(n), each = k_eff)
  J <- as.vector(ord_idx[seq_len(k_eff), , drop = FALSE])
  dIJ <- Dm[cbind(I, J)]
  kth <- vapply(seq_len(n), function(i) Dm[i, ord_idx[k_eff, i]], numeric(1))
  sigmaIJ <- pmax(kth[I] * kth[J], 1e-12)
  W <- exp(-(dIJ^2) / sigmaIJ)
  vnames <- rownames(Dm); if (is.null(vnames)) vnames <- as.character(seq_len(n))
  df_edges <- data.frame(from = vnames[I], to = vnames[J], weight = W)
  df_verts <- data.frame(name = vnames)
  g <- igraph::graph_from_data_frame(df_edges, directed = FALSE, vertices = df_verts)
  igraph::simplify(g, remove.multiple = TRUE, remove.loops = TRUE,
                   edge.attr.comb = list(weight = "max"))
}

# Laplacian Eigenmaps with normalised Laplacian by default
le_embed <- function(g, ndim = 2, normalise = TRUE, row_normalize = FALSE) {
  attr <- if ("weight" %in% igraph::edge_attr_names(g)) "weight" else NULL
  A <- igraph::as_adjacency_matrix(g, attr = attr, sparse = TRUE)
  A <- Matrix::forceSymmetric(A, uplo = "U")
  d <- Matrix::rowSums(A)
  if (any(d == 0)) {
    nz <- which(d == 0)
    if (length(nz)) {
      A <- A + Matrix::Diagonal(n = nrow(A), x = ifelse(seq_len(nrow(A)) %in% nz, 1e-12, 0))
      d <- Matrix::rowSums(A)
    }
  }
  if (isTRUE(normalise)) {
    invsqrtd <- 1 / sqrt(d); invsqrtd[!is.finite(invsqrtd)] <- 0
    Dmhalf <- Matrix::Diagonal(x = invsqrtd)
    L <- Matrix::Diagonal(n = nrow(A)) - (Dmhalf %*% A %*% Dmhalf)
    k_take <- min(ndim + 1L, nrow(L) - 1L)
    ev <- RSpectra::eigs_sym(L, k = k_take, which = "SM")
    U  <- ev$vectors[, -1, drop = FALSE]
    U  <- U[, seq_len(min(ndim, ncol(U))), drop = FALSE]
    emb <- U
  } else {
    Dm <- Matrix::Diagonal(x = d); L <- Dm - A
    ev <- RSpectra::eigs_sym(L, k = ndim + 1, which = "SM")
    emb <- ev$vectors[, -1, drop = FALSE]
  }
  if (isTRUE(row_normalize)) {
    emb <- sweep(emb, 1, sqrt(rowSums(emb^2)) + 1e-12, "/")
  }
  rownames(emb) <- igraph::V(g)$name
  emb
}

# Intrinsic dimension estimators
twonn_id <- function(Y){
  nn <- RANN::nn2(Y, k = 3)$nn.dists
  r <- nn[,3] / nn[,2]
  r <- r[is.finite(r) & r > 0]
  1 / mean(log(r))
}
twonn_local <- function(XY, k = 3){
  nn <- RANN::nn2(XY, XY, k = k)$nn.dists
  r  <- nn[,k] / nn[,2]
  r  <- r[is.finite(r) & r > 0]
  1 / mean(log(r))
}


# =================================================================================================
# D) I/O and coercion
# =================================================================================================

# Load base matrix. Delimiter set to ';' and decimal '.'. Adjust if needed.
df <- readr::read_delim("psychometric_matrix.csv", delim = ";",
                        locale = readr::locale(decimal_mark = "."))

# Identify ID and drop diagnosis-prefixed columns to keep pure psychometric X
id_col  <- if ("participant_id" %in% names(df)) "participant_id" else names(df)[1]
ids_all <- as.character(df[[id_col]])

X0 <- dplyr::select(df, -dplyr::all_of(c(id_col, grep("^diagnosis", names(df), value = TRUE))))

# Coercion rules
X <- X0
for (nm in names(X)){
  v <- X[[nm]]
  if (is.character(v)) {
    suppressWarnings(vn <- as.numeric(v))
    if (!all(is.na(vn))) v <- vn
  }
  if      (is_binary(v) || is_small_int(v)) X[[nm]] <- factor(as.numeric(v), ordered = TRUE)
  else if (is.numeric(v))                   X[[nm]] <- as.numeric(v)
  else                                      X[[nm]] <- factor(v)
}

# Drop incomplete rows and near-zero-variance variables, align IDs
keep <- stats::complete.cases(X)
X    <- X[keep, , drop = FALSE]
nzv <- vapply(X, function(v){
  if (is.numeric(v)) stats::sd(v, na.rm = TRUE) > 1e-8 else length(unique(v)) > 1
}, logical(1))
X <- as.data.frame(X[, nzv, drop = FALSE])

ids <- ids_all[keep]
rn <- make.unique(ids)
rownames(X) <- rn

# Flat ID table for joins
psy_ids <- tibble::tibble(participant_id = as.character(df[[id_col]]))

# Load diagnoses (long) and build wide 0/1 table
diag_long_raw <- readr::read_delim("long_diagnoses.csv", delim = ";",
                                   col_types = readr::cols())

dx_col <- if ("diagnosis" %in% names(diag_long_raw)) "diagnosis" else "Data"

diag_wide <- diag_long_raw |>
  dplyr::transmute(participant_id = as.character(participant_id),
                   dx = .data[[dx_col]]) |>
  dplyr::mutate(present = 1L) |>
  dplyr::distinct(participant_id, dx, .keep_all = TRUE) |>
  tidyr::pivot_wider(names_from = dx, values_from = present, values_fill = 0L)

# Subject-aligned wide diagnoses for arbitrary row lists
dx_wide_for_rows <- function(ids_vec) {
  tibble::tibble(participant_id = as.character(ids_vec)) %>%
    dplyr::left_join(diag_wide, by = "participant_id") %>%
    dplyr::mutate(dplyr::across(-participant_id, ~ tidyr::replace_na(., 0L)))
}

# Primary diagnosis label per subject for quick grouping visuals
diag_wide_full <- tibble::tibble(participant_id = as.character(df[[id_col]])) %>%
  dplyr::left_join(diag_wide, by = "participant_id") %>%
  dplyr::mutate(dplyr::across(-participant_id, ~ tidyr::replace_na(., 0L)))

dx_cols_all <- setdiff(names(diag_wide_full), "participant_id")

if (length(dx_cols_all)) {
  ord <- dx_cols_all[order(colSums(diag_wide_full[, dx_cols_all, drop = FALSE]), decreasing = TRUE)]
  primary_dx <- apply(diag_wide_full[, ord, drop = FALSE], 1, function(r){
    ix <- which(r > 0L)
    if (length(ix)) ord[min(ix)] else NA_character_
  })
} else {
  primary_dx <- rep(NA_character_, nrow(diag_wide_full))
}

dx_for_rows <- function(row_ids){
  mm <- match(row_ids, diag_wide_full$participant_id)
  factor(primary_dx[mm], exclude = NULL)
}

# =================================================================================================
# E) Gower distance, kNN graph, Laplacian Eigenmaps embedding
# =================================================================================================
# Goal:
#   - Compute a single mixed-type distance matrix (Gower) for all downstream graph/geo operations.
#   - Build a weighted kNN graph using self-tuned Gaussian kernels (Zelnik-Manor & Perona).
#   - Produce a 3D Laplacian Eigenmaps (LE) embedding that serves as the base geometric space.
# Notes:
#   - We cache 'Dm_gower' and 'ord_idx' ONCE here. Later sections (audits) should reuse them.
#   - 'KNN_K' is the only source of truth for the k parameter (avoid drift across sections).
# =================================================================================================

# 1) Mixed-type distances (Gower)
#    cluster::daisy handles numeric / ordered factors / unordered factors consistently.
D_gower <- cluster::daisy(X, metric = "gower")

# Keep a dense matrix copy ONCE (we reuse it later); set diagonal to Inf to simplify NN lookups.
Dm_gower <- as.matrix(D_gower)
diag(Dm_gower) <- Inf

# Pre-compute, for each row, the indices of all other rows ordered by ascending distance.
# This is reused by the kNN graph builder and by later geodesic / audit routines.
ord_idx <- apply(Dm_gower, 1L, order, decreasing = FALSE)
if (is.null(dim(ord_idx))) ord_idx <- matrix(ord_idx, ncol = nrow(Dm_gower))

# 2) Build kNN graph with self-tuned Gaussian weights
#    - For each i, connect to its KNN_K nearest neighbours.
#    - Edge weights: exp(-d^2 / (σ_i σ_j)) with σ_i = distance to i's k-th neighbour.
#    - Graph is undirected, deduped, loop-free, and uses max weight on multi-edges.
g <- .timeit(
  knn_graph(Dm_gower, ord_idx, k = KNN_K),
  sprintf("[graph] kNN built with k=%d ", KNN_K)
)

# 3) Laplacian Eigenmaps (normalised Laplacian) to 3D
#    - We do NOT row-normalise the embedding; downstream fits expect raw coordinates.
#    - Row names carry through from vertex names; we ensure they exist and match X rownames.
Yle <- .timeit(
  le_embed(g, ndim = 3, row_normalize = FALSE),
  "[LE] embedded (3D)"
)
if (is.null(rownames(Yle))) rownames(Yle) <- rownames(X)

# Persist LE coordinates for external tools
emb_df <- data.frame(
  row_id = rownames(Yle),
  LE1 = Yle[, 1],
  LE2 = Yle[, 2],
  LE3 = Yle[, 3]
)
write.csv(emb_df, "embedding_LE.csv", row.names = FALSE)

# Optional quick intrinsic-dimension proxy (TwoNN) on the LE space
if (DO_DIAGNOSTICS) {
  id_le <- suppressWarnings(twonn_id(scale(Yle[, 1:3, drop = FALSE])))
  cat(sprintf("Global TwoNN intrinsic dimension (LE1-LE3) ≈ %.2f\n", id_le))
}

# Optional quick visual of LE1–LE2 coloured by point density (via KDE contours)
if (DO_PLOTS) {
  # A light scatter with cluster colours will be done after clustering (Section F).
  dens <- with(as.data.frame(Yle), MASS::kde2d(V1, V2, n = 150))
  dens_df <- with(dens, data.frame(
    x = rep(x, each = length(y)),
    y = rep(y, times = length(x)),
    z = as.vector(z)
  ))
  print(
    ggplot() +
      geom_point(data = as.data.frame(Yle), aes(V1, V2), size = 1.0, alpha = .5) +
      geom_contour(data = dens_df, aes(x, y, z = z), bins = 10, linewidth = 0.3) +
      labs(title = "LE1–LE2 with KDE contours", x = "LE1", y = "LE2") +
      theme_minimal()
  )
}

# =================================================================================================
# F) Clustering and LE diagnostics
# =================================================================================================
# Goal:
#   - Provide two unsupervised clusterings: density-based (HDBSCAN) and graph-based (Louvain).
#   - Quantify structure via modularity permutations, silhouette, and sensitivity to k.
#   - Persist cluster labels aligned to LE rows.
# Notes:
#   - HDBSCAN sweep chooses the fit with the most non-noise points (cluster > 0).
#   - Louvain uses the weighted kNN graph built above.
#   - Diagnostic prints are concise but informative; plotting is gated by DO_PLOTS.
# =================================================================================================

# 1) HDBSCAN quick sweep over minPts to stabilise against noise-only outcomes
hdb <- {
  grid <- c(8, 10, 12, 15)
  outs <- lapply(grid, function(m) {
    list(m = m, fit = dbscan::hdbscan(scale(Yle), minPts = m))
  })
  outs[[which.max(sapply(outs, function(o) sum(o$fit$cluster > 0)))]]
}

# 2) Louvain clustering on the weighted graph
cl_louvain <- igraph::cluster_louvain(g, weights = igraph::E(g)$weight)
cl_lv_vec  <- igraph::membership(cl_louvain)

# Persist cluster labels
cl_df <- data.frame(
  row_id  = rownames(Yle),
  HDBSCAN = hdb$fit$cluster,
  Louvain = cl_lv_vec[ rownames(Yle) ]
)
write.csv(cl_df, "clusters_unsupervised.csv", row.names = FALSE)

# Optional: LE1–LE2 coloured by Louvain (quick look)
if (DO_PLOTS) {
  df_plot <- data.frame(LE1 = Yle[,1], LE2 = Yle[,2], lab = factor(cl_lv_vec[ rownames(Yle) ]))
  print(
    ggplot(df_plot, aes(LE1, LE2, color = lab)) +
      geom_point(size = 1.4, alpha = .9) +
      guides(color = "none") +
      labs(title = "Laplacian Eigenmaps coloured by Louvain")
  )
}

# 3) Diagnostics (guarded by DO_DIAGNOSTICS)
if (DO_DIAGNOSTICS) {
  # 3a) Modularity permutation test (degree-sequence preserving rewiring)
  Q_obs <- igraph::modularity(g, cl_lv_vec, weights = igraph::E(g)$weight)
  B     <- 200                # light but informative
  Q_null <- numeric(B)
  took   <- 0L
  
  for (b in seq_len(B)) {
    # keeping_degseq rewiring with ~10*m edge swaps
    g0  <- igraph::rewire(g, igraph::keeping_degseq(niter = 10 * igraph::gsize(g)))
    cl0 <- igraph::cluster_louvain(g0, weights = igraph::E(g0)$weight)
    Q_null[b] <- igraph::modularity(g0, igraph::membership(cl0), weights = igraph::E(g0)$weight)
    took <- b
    
    # Early stop every 20 draws once we have at least 40:
    # if the 95% CI half-width around p-hat is < 0.02, we bail out.
    if (b >= 40 && (b %% 20 == 0)) {
      k <- sum(Q_null[1:b] >= Q_obs, na.rm = TRUE)
      p_hat <- (k + 1) / (b + 1)
      half  <- 1.96 * sqrt(p_hat * (1 - p_hat) / b)
      if (half < 0.02) break
    }
  }
  p_Q <- (1 + sum(Q_null[1:took] >= Q_obs, na.rm = TRUE)) / (took + 1)
  cat(sprintf("Modularity Q = %.3f (perm p = %.3f; draws = %d)\n", Q_obs, p_Q, took))
  
  # 3b) Mean silhouette on LE1–LE2 for Louvain labels
  D_le <- stats::dist(Yle[, 1:2, drop = FALSE])
  asw  <- mean(cluster::silhouette(as.integer(as.factor(cl_lv_vec)), D_le)[, "sil_width"])
  cat(sprintf("Mean silhouette in LE1–LE2 = %.3f\n", asw))
  
  # 3c) Stability vs k in graph construction (AMI against the k=KNN_K reference)
  stab <- sapply(c(8, 10, 12, 15), function(k) {
    gk <- knn_graph(Dm_gower, ord_idx, k = k)
    ck <- igraph::membership(igraph::cluster_louvain(gk, weights = igraph::E(gk)$weight))
    aricode::AMI(as.integer(factor(cl_lv_vec)), as.integer(factor(ck)))
  })
  cat("AMI vs k grid:", paste(sprintf("%.3f", stab), collapse = " "), "\n")
  
  # 3d) Local intrinsic dimension per Louvain community (TwoNN on LE)
  by_id <- tapply(seq_along(cl_lv_vec), cl_lv_vec, function(ix) twonn_local(Yle[ix, , drop = FALSE]))
  print(round(unlist(by_id), 2))
}

# =================================================================================================
# G) Variable ↔ LE association and LE density visual
# =================================================================================================
# Goal:
#   - Screen which original variables associate most with LE axes (ranked by |Spearman ρ|).
#   - Provide a density context plot for LE1–LE2 (already printed in E).
# Notes:
#   - Factors are coerced to numeric scores for rank-correlation (ordered → integer codes).
#   - This is purely exploratory; no multiple-testing correction is applied here.
# =================================================================================================

# Align X rows to Yle rows; coerce factors to numeric scores for rank-correlation
X_assoc <- X[rownames(Yle), , drop = FALSE]
numZ    <- as.data.frame(lapply(X_assoc, function(v) if (is.factor(v)) as.numeric(v) else v))

# Spearman correlation (robust to monotone transforms)
LE12  <- Yle[, 1:2, drop = FALSE]
cors1 <- sapply(numZ, function(v) suppressWarnings(cor(v, LE12[,1], use = "pair", method = "spearman")))
cors2 <- sapply(numZ, function(v) suppressWarnings(cor(v, LE12[,2], use = "pair", method = "spearman")))

cat("\nTop absolute Spearman correlates for LE1:\n")
print(head(sort(abs(cors1), decreasing = TRUE), 12))
cat("\nTop absolute Spearman correlates for LE2:\n")
print(head(sort(abs(cors2), decreasing = TRUE), 12))

# (Density visual was printed in E; keeping that there prevents duplicate plots.)

# =================================================================================================
# H) Diagnosis mapping and tests in LE space
# =================================================================================================
# Goal:
#   - Bring long-format diagnoses to the LE rows, optionally visualise a prevalence surface.
#   - Run PERMANOVA for each diagnosis (binary 0/1) against LE(1:3) when both classes have n>=MIN_N_DX.
#   - Summarise over-/under-representation of diagnoses within Louvain clusters via std residuals.
# Notes:
#   - We do NOT change any subject ordering; everything is matched by 'participant_id'.
#   - The "probability surface" is a quick ridge-penalised logistic model, visualised in LE1–LE2
#     at median LE3 for context (only for the most prevalent dx with n >= MIN_N_DX).
# =================================================================================================

# 1) Wide 0/1 diagnoses aligned to the LE row order
dx_wide <- dx_wide_for_rows(rownames(Yle)) %>%
  dplyr::mutate(dplyr::across(-dplyr::any_of(c("row_id","participant_id")),
                              ~ as.integer(as.logical(.x))))
dx_cols <- setdiff(names(dx_wide), c("row_id","participant_id"))

# 2) Optional probability surface for the most prevalent dx (quick logistic ridge)
if (length(dx_cols) && DO_PLOTS) {
  prev <- sort(colSums(dx_wide[, dx_cols, drop = FALSE], na.rm = TRUE), decreasing = TRUE)
  if (length(prev) && prev[1] >= MIN_N_DX) {
    target <- names(prev)[1]
    y01 <- as.integer(dx_wide[[target]] == 1L)
    
    # Basic ridge logistic on LE1:LE3 (alpha=0, small lambda)
    keep <- is.finite(Yle[,1]) & is.finite(Yle[,2]) & is.finite(Yle[,3])
    fit  <- glmnet::glmnet(as.matrix(Yle[keep, 1:3]), y01[keep],
                           family = "binomial", alpha = 0, lambda = 0.1)
    
    # Visualise on LE1–LE2 grid at median LE3
    gx <- seq(min(Yle[,1]), max(Yle[,1]), length.out = 120)
    gy <- seq(min(Yle[,2]), max(Yle[,2]), length.out = 120)
    grid <- expand.grid(LE1 = gx, LE2 = gy)
    medLE3 <- stats::median(Yle[,3], na.rm = TRUE)
    pr <- predict(fit, newx = as.matrix(cbind(grid$LE1, grid$LE2, medLE3)), type = "response")
    surf <- data.frame(grid, p = as.numeric(pr))
    
    print(
      ggplot() +
        geom_raster(data = surf, aes(LE1, LE2, fill = p), alpha = .85, interpolate = TRUE) +
        scale_fill_viridis_c() +
        geom_contour(data = surf, aes(LE1, LE2, z = p), colour = "black", bins = 8, linewidth = 0.25) +
        geom_point(data = data.frame(LE1 = Yle[,1], LE2 = Yle[,2]), aes(LE1, LE2),
                   size = 0.6, alpha = .5) +
        labs(title = paste0("P(", target, ") surface at median LE3"), fill = "prob")
    )
  }
}

# 3) PERMANOVA for each diagnosis (binary) vs LE(1:3), with sample size guard
if (length(dx_cols) && ncol(Yle) >= 3) {
  Y_dx <- Yle[, 1:3, drop = FALSE]
  perm_tbl <- lapply(dx_cols, function(dn){
    y01 <- as.integer(dx_wide[[dn]] == 1L)
    # Require both classes to have at least MIN_N_DX
    if (sum(y01) < MIN_N_DX || sum(1 - y01) < MIN_N_DX) return(NULL)
    res <- suppressWarnings(
      vegan::adonis2(stats::dist(Y_dx) ~ grp, data = data.frame(grp = factor(y01)), permutations = 999)
    )
    data.frame(diagnosis = dn, F = res$F[1], R2 = res$R2[1], p = res$`Pr(>F)`[1])
  }) %>% dplyr::bind_rows()
  
  if (nrow(perm_tbl)) {
    perm_tbl <- perm_tbl[order(perm_tbl$p), ]
    write.csv(perm_tbl, "permanova_dx_in_LE.csv", row.names = FALSE)
  }
}

# 4) Cluster × diagnosis residuals heatmap (std residuals)
#    - Compares observed counts per (Louvain cluster, diagnosis) to independence expectation.
#    - Highlights over-/under-representation patterns.
cl_fac <- factor(cl_lv_vec[ rownames(Yle) ])
if (length(dx_cols)) {
  tab <- sapply(dx_cols, function(dn) tapply(dx_wide[[dn]] == 1L, cl_fac, sum, na.rm = TRUE))
  tab <- as.matrix(tab); mode(tab) <- "numeric"
  
  # Independence expectation E_ij = (row_i total * col_j total) / grand total
  Eexp <- outer(rowSums(tab), colSums(tab), function(r, c) r * c / max(sum(tab), 1))
  Z <- (tab - Eexp) / sqrt(pmax(Eexp, 1e-9))  # std residuals
  
  if (DO_PLOTS) {
    heat_df <- as.data.frame(as.table(Z))
    colnames(heat_df) <- c("Cluster", "Diagnosis", "StdResid")
    print(
      ggplot(heat_df, aes(Diagnosis, Cluster, fill = StdResid)) +
        geom_tile(colour = "white") +
        scale_fill_gradient2(low = "blue", mid = "white", high = "red", midpoint = 0) +
        labs(title = "Cluster–Diagnosis residuals", fill = "Std resid") +
        theme_minimal() +
        theme(axis.text.x = element_text(angle = 45, hjust = 1))
    )
  }
}

# =================================================================================================
# I) Principal surface in LE space
# =================================================================================================
# Goal:
#   - Parameterise LE(1:3) by a 2D coordinate system (u,v) via PCA of Yle.
#   - Fit a smooth surface Yle_j ~ s(u,v) for each LE coordinate j=1..3 using mgcv::bam.
#   - Report apparent fit (RMSE, R2-like) and a leakage-safe K-fold CV (train-only PCA).
#   - Save per-row (u,v), fitted LE-hat, and residual norms for downstream use.
# Notes:
#   - PCA(u,v) is deterministic for given Yle, and is re-fit inside CV on TRAIN only.
#   - 'K_BASIS_UV' controls smooth flexibility; we keep it modest to reduce overfitting.
# =================================================================================================

if (DO_SURFACE) {
  # 1) 2D parameterisation: first two PCs of LE (deterministic on all data for apparent fit)
  pca_init  <- prcomp(Yle, center = TRUE, scale. = FALSE)
  uv_coords <- as.data.frame(pca_init$x[, 1:2, drop = FALSE])
  names(uv_coords) <- c("u", "v")
  
  # 2) Fit Yle_j ~ s(u,v) for j=1..ncol(Yle) with mgcv::bam (stable settings via safe_bam)
  df_uv <- uv_coords
  mfits <- lapply(seq_len(ncol(Yle)), function(j) {
    safe_bam(Yle[, j] ~ s(u, v, bs = "tp", k = K_BASIS_UV),
             data = df_uv, k_threads = 1, gamma = 1.2, select = TRUE)
  })
  Yhat <- sapply(seq_along(mfits), function(j) as.numeric(predict(mfits[[j]], df_uv)))
  colnames(Yhat) <- colnames(Yle)
  
  # 3) Residuals and apparent R2-like
  resid_mat <- as.matrix(Yle) - as.matrix(Yhat)
  resid2    <- rowSums(resid_mat^2)
  rmse      <- sqrt(mean(resid2))
  R2_like   <- 1 - sum(resid_mat^2) / r2_tot(Yle)
  
  # 4) Leakage-safe K-fold CV (train-only PCA → project test into train's (u,v))
  R2_like_cv     <- NA_real_
  
  if (isTRUE(USE_SURFACE_CV)) {
    set.seed(1) # reproductibility
    n <- nrow(Yle)
    fold <- sample(rep(1:SURFACE_CV_FOLDS, length.out = n))
    rss <- 0
    tss <- sum(scale(Yle, scale = FALSE)^2)
    
    for (k in seq_len(SURFACE_CV_FOLDS)) {
      tr <- which(fold != k); te <- which(fold == k)
      if (length(tr) < 5L || length(te) < 2L) next
      
      # PCA on train only
      eff_rank <- min(2L, ncol(Yle), length(tr) - 1L)
      pc_tr <- prcomp(Yle[tr, , drop = FALSE], center = TRUE, scale. = FALSE, rank. = eff_rank)
      if (ncol(pc_tr$x) < 2L) next  # skip tiny folds with rank < 2
      
      Utr <- as.data.frame(pc_tr$x[, 1:2, drop = FALSE]); names(Utr) <- c("u","v")
      Ute <- as.data.frame(
        scale(Yle[te, , drop = FALSE], center = pc_tr$center, scale = FALSE) %*%
          pc_tr$rotation[, 1:2, drop = FALSE]
      ); names(Ute) <- c("u","v")
      
      # Predict each LE coordinate from (u,v)
      preds <- sapply(seq_len(ncol(Yle)), function(j){
        fit <- safe_bam(Yle[tr, j] ~ s(u, v, bs = "tp", k = K_BASIS_UV),
                        data = Utr, k_threads = 1, gamma = 1.2, select = TRUE)
        as.numeric(predict(fit, Ute))
      })
      rss <- rss + sum((Yle[te, , drop = FALSE] - preds)^2)
    }
    R2_like_cv <- 1 - rss / tss
  }
  
  cat(sprintf("[Surface on LE] RMSE = %.4f | R2-like = %.3f%s\n",
              rmse, R2_like,
              if (isTRUE(USE_SURFACE_CV))
                sprintf(" | R2-like (CV,%d-fold) = %.3f", SURFACE_CV_FOLDS, R2_like_cv) else ""))
  
  # 5) Save per-row (u,v), fitted coords, and residual norm
  surf_df <- data.frame(
    row_id  = rownames(Yle),
    u       = uv_coords$u,
    v       = uv_coords$v,
    LE1_hat = Yhat[, 1],
    LE2_hat = Yhat[, 2],
    LE3_hat = if (ncol(Yhat) >= 3) Yhat[, 3] else NA_real_,
    resid   = sqrt(resid2)
  )
  write.csv(surf_df, "principal_surface_coords_and_residuals.csv", row.names = FALSE)
  
  # 6) Plots (gated): residuals in (u,v) and optional 3D surface
  if (DO_PLOTS) {
    # (a) residual magnitude scatter in (u,v)
    print(
      ggplot(data.frame(u = uv_coords$u, v = uv_coords$v, resid = sqrt(resid2)),
             aes(u, v, color = resid)) +
        geom_point(size = 1.6, alpha = 0.9) +
        scale_color_viridis_c() +
        coord_equal() +
        labs(title = "Principal surface residuals in (u,v)", x = "u", y = "v", color = "dist")
    )
    
    # (b) Optional 3D surface using plotly (skip if not installed)
    if (requireNamespace("plotly", quietly = TRUE)) {
      ngr <- 80
      ug  <- seq(min(uv_coords$u), max(uv_coords$u), length.out = ngr)
      vg  <- seq(min(uv_coords$v), max(uv_coords$v), length.out = ngr)
      grid_uv <- expand.grid(u = ug, v = vg)  # v varies fastest
      
      pred1 <- as.numeric(predict(mfits[[1]], grid_uv))
      pred2 <- as.numeric(predict(mfits[[2]], grid_uv))
      pred3 <- as.numeric(predict(mfits[[3]], grid_uv))
      
      Z1 <- matrix(pred1, nrow = length(vg), ncol = length(ug), byrow = FALSE)
      Z2 <- matrix(pred2, nrow = length(vg), ncol = length(ug), byrow = FALSE)
      Z3 <- matrix(pred3, nrow = length(vg), ncol = length(ug), byrow = FALSE)
      
      p <- plotly::plot_ly() %>%
        plotly::add_markers(x = Yle[,1], y = Yle[,2], z = Yle[,3],
                            marker = list(size = 2, opacity = 0.45),
                            name = "Points") %>%
        plotly::add_surface(x = Z1, y = Z2, z = Z3, opacity = 0.6, showscale = FALSE,
                            name = "Surface") %>%
        plotly::layout(
          scene = list(
            xaxis = list(title = "LE1"),
            yaxis = list(title = "LE2"),
            zaxis = list(title = "LE3"),
            aspectmode = "data"
          ),
          title = "Principal surface in LE (mgcv)"
        )
      print(p)
    }
  }
  
  # 7) Optional association directly in (u,v) (diagnosis factor)
  dx_lab <- dx_for_rows(rownames(Yle))
  keep_dx2 <- !is.na(dx_lab)
  if (any(keep_dx2)) {
    D_uv <- stats::dist(uv_coords[keep_dx2, , drop = FALSE])
    df   <- data.frame(dx = droplevels(dx_lab[keep_dx2]))
    print(vegan::adonis2(D_uv ~ dx, data = df, permutations = 999))
    
    # quick categorical association via Cramer's V after clustering (k=4 PAM)
    set.seed(1) #reproductibility
    pamk <- cluster::pam(uv_coords[keep_dx2, , drop = FALSE], k = 4)
    tab_uv <- table(pamk$clustering, droplevels(dx_lab[keep_dx2]))
    chi2 <- suppressWarnings(chisq.test(tab_uv, correct = FALSE)$statistic)
    n <- sum(tab_uv); mdf <- min(nrow(tab_uv) - 1, ncol(tab_uv) - 1)
    cv <- as.numeric(sqrt(chi2 / (n * mdf)))
    cat(sprintf("Cramer's V on (u,v) binned: %.3f\n", cv))
  }
}

# =================================================================================================
# J) No-mercy dimension audit (visual + quantitative, artefact-aware)
# =================================================================================================
# Goal:
#   - Stress-test the 1D-vs-2D narrative in LE space using independent surrogates:
#       * Principal curve in LE(1:3), arc-length parameter t.
#       * Geodesic distances on the Gower kNN graph (edge length = Gower distance).
#       * MDS/Isomap residual variance elbow with k matched to KNN_K.
#       * Sliding-window TwoNN along t.
#       * Leakage-safe CV: predict LE3 from surface (u,v) vs from curve t.
# Notes:
#   - Reuses Dm_gower and ord_idx computed in Section E (no recomputation).
#   - Uses the same KNN_K to ensure consistent geodesic topology.
# =================================================================================================

# --- helpers (moved earlier in design doc; repeated here for clarity of section scope) ---
arc_length_from_order <- function(P, ord){
  d <- sqrt(rowSums((P[ord[-1], , drop = FALSE] - P[ord[-length(ord)], , drop = FALSE])^2))
  c(0, cumsum(d))
}
geodesic_graph <- function(Dm, ord_idx, k = 10){
  n <- nrow(Dm); k_eff <- pmin(k, n - 1L)
  I <- rep(seq_len(n), each = k_eff)
  J <- as.vector(ord_idx[seq_len(k_eff), , drop = FALSE])
  w <- Dm[cbind(I, J)]                         # edge length = raw Gower distance (no kernel)
  vnames <- rownames(Dm); if (is.null(vnames)) vnames <- as.character(seq_len(n))
  dfE <- data.frame(from = vnames[I], to = vnames[J], len = w)
  g <- igraph::graph_from_data_frame(dfE, directed = FALSE, vertices = data.frame(name = vnames))
  igraph::simplify(g, remove.multiple = TRUE, remove.loops = TRUE,
                   edge.attr.comb = list(len = "min"))
}
pair_metrics <- function(d_true, d_model){
  d_true <- as.numeric(d_true); d_model <- as.numeric(d_model)
  keep <- is.finite(d_true) & is.finite(d_model) & d_true > 0 & d_model > 0
  d_true <- d_true[keep]; d_model <- d_model[keep]
  rho <- suppressWarnings(cor(d_true, d_model, method = "spearman"))
  s1  <- sqrt(sum((d_true - d_model)^2) / sum(d_true^2))  # Kruskal stress-1
  c(spearman = rho, stress1 = s1)
}
isomap_resvar <- function(Dm, m, k = 10){
  # Build geodesic graph and compute all-pairs shortest paths
  ggeo <- geodesic_graph(Dm, ord_idx, k = k)
  G <- igraph::distances(ggeo, weights = igraph::E(ggeo)$len)
  G[!is.finite(G)] <- max(G[is.finite(G)], 0) * 10
  fit <- stats::cmdscale(stats::as.dist(G), k = m, eig = FALSE)
  Dm_hat <- as.matrix(stats::dist(fit))
  1 - cor(as.numeric(G), as.numeric(Dm_hat))^2  # Tenenbaum residual variance
}

# --- 1) principal curve in LE(1:3) and colour-by-parameter visual ---
pcv <- princurve::principal_curve(as.matrix(Yle[, 1:3, drop = FALSE]), stretch = 0)
t   <- pcv$lambda

if (DO_PLOTS) {
  df_le <- data.frame(LE1 = Yle[,1], LE2 = Yle[,2], LE3 = Yle[,3], t = t)
  print(
    ggplot(df_le, aes(LE1, LE2, color = t)) +
      geom_point(size = 1.4, alpha = .9) +
      scale_color_viridis_c() +
      theme_minimal() +
      labs(title = "LE with principal curve (coloured by t)")
  )
}

# --- 2) geodesic vs 1D curve-distance (arc length), and vs LE-2D Euclidean (control) ---
# Geodesics on the Gower kNN graph:
ggeo <- geodesic_graph(Dm_gower, ord_idx, k = KNN_K)

# Arc length along the principal curve:
ord <- order(t)
P    <- pcv$s
dseg <- sqrt(rowSums((P[ord[-1], , drop = FALSE] - P[ord[-length(ord)], , drop = FALSE])^2))
slen <- c(0, cumsum(dseg))
slen_i <- numeric(length(t)); slen_i[ord] <- slen   # arc-length per sample (original row order)

# Efficient pair sampling to compare the three surrogates
set.seed(1) #reproductibility
n    <- nrow(Dm_gower)
nsrc <- min(400L, n)
mtgt <- 40L
src  <- sample.int(n, nsrc, replace = FALSE)

d_geo_vec <- numeric(0)  # graph geodesic (true)
d_cur_vec <- numeric(0)  # |Δ arc-length|
d_le2_vec <- numeric(0)  # Euclidean in LE(1:2)

for (s in src) {
  d_all <- igraph::distances(ggeo, v = s, weights = igraph::E(ggeo)$len)  # one-to-all geodesics
  tgt   <- sample.int(n, mtgt, replace = TRUE)
  if (any(tgt == s)) tgt[tgt == s] <- sample.int(n, sum(tgt == s), replace = TRUE)
  
  dg  <- as.numeric(d_all[1, tgt])
  dc  <- abs(slen_i[s] - slen_i[tgt])
  dle <- sqrt(rowSums((Yle[rep(s, length(tgt)), 1:2, drop = FALSE] - Yle[tgt, 1:2, drop = FALSE])^2))
  
  keep <- is.finite(dg) & is.finite(dc) & is.finite(dle) & (dg > 0) & (dc > 0) & (dle > 0)
  d_geo_vec <- c(d_geo_vec, dg[keep])
  d_cur_vec <- c(d_cur_vec, dc[keep])
  d_le2_vec <- c(d_le2_vec, dle[keep])
}

m_geocurve <- pair_metrics(d_geo_vec, d_cur_vec)
cat(sprintf("[Geo vs curve] Spearman=%.3f | Stress-1=%.3f | n=%d pairs\n",
            m_geocurve["spearman"], m_geocurve["stress1"], length(d_geo_vec)))

m_geo2d <- pair_metrics(d_geo_vec, d_le2_vec)
cat(sprintf("[Geo vs LE-2D] Spearman=%.3f | Stress-1=%.3f | n=%d pairs\n",
            m_geo2d["spearman"], m_geo2d["stress1"], length(d_le2_vec)))

if (DO_PLOTS) {
  oldpar <- par(no.readonly = TRUE); on.exit(par(oldpar))
  par(mfrow = c(1,2))
  plot(d_geo_vec, d_cur_vec, pch = 16, cex = .5,
       xlab = "Geodesic (Gower kNN)", ylab = "Curve arc-length",
       main = "Geodesic vs curve"); abline(0,1,col="gray")
  plot(d_geo_vec, d_le2_vec, pch = 16, cex = .5,
       xlab = "Geodesic (Gower kNN)", ylab = "LE(1:2) Euclidean",
       main = "Geodesic vs LE-2D");  abline(0,1,col="gray")
}

# --- 3) Isomap residual variance elbow (match k to KNN_K) ---
iso_rv <- sapply(1:3, function(m) isomap_resvar(Dm_gower, m = m, k = KNN_K))
names(iso_rv) <- paste0("m=", 1:3)
print(iso_rv)
if (DO_PLOTS) {
  barplot(iso_rv, main = "Isomap residual variance", ylab = "Residual variance")
}

# --- 4) Local TwoNN ID along principal-curve parameter t (sliding windows) ---
win <- 0.15   # fraction per window
q <- stats::quantile(t, probs = seq(0,1, by = win), na.rm = TRUE)
cent <- 0.5 * (head(q, -1) + tail(q, -1))
id_loc <- sapply(seq_along(cent), function(i){
  lo <- q[i]; hi <- q[i+1]
  ix <- which(t >= lo & t <= hi)
  if (length(ix) < 20) return(NA_real_)
  suppressWarnings(twonn_id(scale(Yle[ix, 1:3, drop = FALSE])))
})
if (DO_PLOTS) {
  plot(cent, id_loc, type = "b", pch = 16, xlab = "t (principal curve param)",
       ylab = "Local TwoNN ID", main = "Local intrinsic dimension along the curve")
  abline(h = 1, lty = 2, col = "gray")
}

# --- 5) Non-cheating CV: predict only LE3 from (u,v) surface vs from 1D curve t ---
# (A) Surface CV scoring LE3 only
set.seed(1) #reproductibility
fold <- sample(rep(1:SURFACE_CV_FOLDS, length.out = nrow(Yle)))
rss3 <- 0; tss3 <- sum((Yle[,3] - mean(Yle[,3]))^2)
for (k in 1:SURFACE_CV_FOLDS) {
  tr <- which(fold != k); te <- which(fold == k)
  pc_tr <- prcomp(Yle[tr, , drop = FALSE], center = TRUE, scale. = FALSE, rank. = 2)
  Utr <- as.data.frame(pc_tr$x[, 1:2, drop = FALSE]); names(Utr) <- c("u","v")
  Ute <- as.data.frame(scale(Yle[te, , drop = FALSE], center = pc_tr$center, scale = FALSE) %*%
                         pc_tr$rotation[, 1:2, drop = FALSE]); names(Ute) <- c("u","v")
  fit3 <- safe_bam(Yle[tr, 3] ~ s(u, v, bs = "tp", k = K_BASIS_UV),
                   data = Utr, k_threads = 1, gamma = 1.2, select = TRUE)
  pred3 <- as.numeric(predict(fit3, Ute))
  rss3  <- rss3 + sum((Yle[te, 3] - pred3)^2)
}
R2cv_surface_LE3 <- 1 - rss3 / tss3

# (B) Curve CV scoring of LE3: train-only principal curve; OOS t via NN-to-curve
R2cv_curve_LE3 <- {
  set.seed(1) # reproductibility
  Kcv  <- SURFACE_CV_FOLDS
  n    <- nrow(Yle)
  fold <- sample(rep(1:Kcv, length.out = n))
  rss  <- 0
  tss3 <- sum((Yle[,3] - mean(Yle[,3]))^2)
  
  for (k in 1:Kcv) {
    tr <- which(fold != k); te <- which(fold == k)
    pcv_tr  <- princurve::principal_curve(as.matrix(Yle[tr, 1:3, drop = FALSE]), stretch = 0)
    
    # arc-length parameter on the TRAIN curve
    ord_tr  <- order(pcv_tr$lambda)
    P_tr    <- pcv_tr$s[ord_tr, , drop = FALSE]
    dseg    <- sqrt(rowSums((P_tr[-1, , drop = FALSE] - P_tr[-nrow(P_tr), , drop = FALSE])^2))
    s_arc   <- c(0, cumsum(dseg))                 # arc-length along curve
    t_tr    <- numeric(length(tr)); t_tr[ord_tr] <- s_arc
    
    # map TEST rows to TRAIN curve parameter via 3-NN on polyline points (distance-weighted)
    nn      <- FNN::get.knnx(P_tr, as.matrix(Yle[te, 1:3, drop = FALSE]), k = 3)
    w       <- 1 / pmax(nn$nn.dist, 1e-9); w <- w / rowSums(w)
    t_te    <- rowSums(matrix(s_arc[nn$nn.index], ncol = 3) * w)
    
    fit1d   <- safe_bam(Yle[tr, 3] ~ s(t, bs = "tp", k = K_BASIS_UV),
                        data = data.frame(t = t_tr),
                        k_threads = 1, gamma = 1.2, select = TRUE)
    pred3   <- as.numeric(predict(fit1d, newdata = data.frame(t = t_te)))
    rss     <- rss + sum((Yle[te, 3] - pred3)^2)
  }
  1 - rss / tss3
}

cat(sprintf("[CV] R2(LE3 | surface u,v) = %.3f | R2(LE3 | curve t) = %.3f\n",
            R2cv_surface_LE3, R2cv_curve_LE3))

# =================================================================================================
# K) Label-free bundle discovery on the unsupervised base (no diagnoses)
# =================================================================================================
# Goal:
#   - Build a base coordinate system (auto-select 1D vs 2D vs 3D) from LE via PCA/curve.
#   - On a grid over the base, estimate local "fibre" frames by PCA on encoded X neighbourhoods.
#   - Synchronise frames over the grid via an MST "gauge" (orthogonal Procrustes on edges).
#   - Derive curvature-like and principal-angle diagnostics, per-subject fibre energy, and density.
#   - Combine into an unsupervised risk score R_u with robust z-normalisation.
# Notes:
#   - This section is self-contained and can be toggled/removed if not needed.
#   - It relies on FactoMineR MCA/FAMD encoding to compress mixed X into a numeric space.
#   - Filenames preserved for compatibility with your downstream readers.
# =================================================================================================

# --------------------------- helpers: encoding, rank, PCA ---------------------------
encode_numeric_matrix <- function(Xdf){
  num <- vapply(Xdf, is.numeric, logical(1))
  ord <- vapply(Xdf, is.ordered, logical(1))
  fac <- vapply(Xdf, function(z) is.factor(z) && !is.ordered(z), logical(1))
  M_list <- list()
  if (any(num)) {
    M_num <- scale(as.matrix(Xdf[, num, drop = FALSE]), TRUE, TRUE)
    M_list$NUM <- M_num[, apply(M_num, 2, sd, na.rm = TRUE) > 0, drop = FALSE]
  }
  if (any(ord)) {
    M_ord <- scale(as.matrix(data.frame(lapply(Xdf[, ord, drop = FALSE], as.numeric))), TRUE, TRUE)
    M_list$ORD <- M_ord[, apply(M_ord, 2, sd, na.rm = TRUE) > 0, drop = FALSE]
  }
  if (any(fac)) {
    mm <- model.matrix(~ . - 1, data = Xdf[, fac, drop = FALSE])
    keep <- apply(mm, 2, sd) > 0
    if (any(keep)) M_list$FAC <- scale(mm[, keep, drop = FALSE], TRUE, TRUE)
  }
  if (!length(M_list)) stop("encode_numeric_matrix: no encodable columns.")
  M <- do.call(cbind, M_list); storage.mode(M) <- "double"; rownames(M) <- rownames(Xdf); M
}
mp_rank <- function(Xn, cap = 4, shrink = 0.9){
  n_k <- nrow(Xn); p <- ncol(Xn); if (n_k < 5 || p < 2) return(0L)
  sv <- tryCatch(svd(scale(Xn, TRUE, FALSE), nu = 0, nv = 0)$d, error = function(e) NA)
  if (all(!is.finite(sv))) return(0L)
  lam <- (sv^2) / max(n_k - 1, 1)
  tail_idx <- ceiling(length(lam) / 2):length(lam)
  sigma2 <- stats::median(lam[tail_idx], na.rm = TRUE); if (!is.finite(sigma2) || sigma2 <= 0) sigma2 <- stats::median(lam, na.rm = TRUE)
  beta <- p / max(n_k - 1, 1); lambda_plus <- shrink * sigma2 * (1 + sqrt(beta))^2
  as.integer(max(0, min(sum(lam > lambda_plus), cap)))
}
local_pca_Q <- function(Xn, k){
  if (k <= 0) return(list(Q = matrix(0, ncol(Xn), 0), evals = numeric(0)))
  Xc <- scale(Xn, TRUE, FALSE); sv <- svd(Xc, nu = k, nv = k)
  Q  <- if (!is.null(sv$v)) sv$v else sv$u
  Q  <- apply(Q, 2, function(col) col / sqrt(sum(col^2)))
  evals <- (sv$d[seq_len(k)]^2) / max(nrow(Xn) - 1, 1)
  list(Q = Q, evals = evals)
}
build_uv_grid <- function(uv, ngr = 50){
  u_seq <- seq(min(uv$u), max(uv$u), length.out = ngr)
  v_seq <- seq(min(uv$v), max(uv$v), length.out = ngr)
  centers <- expand.grid(u = u_seq, v = v_seq); centers$node_id <- seq_len(nrow(centers))
  mat_id <- matrix(centers$node_id, nrow = ngr, ncol = ngr, byrow = FALSE)
  edges <- list()
  for (i in seq_len(ngr)) { id_row <- mat_id[i, ]; edges <- c(edges, Map(function(a,b) c(a,b), id_row[-length(id_row)], id_row[-1])) }
  for (j in seq_len(ngr)) { id_col <- mat_id[, j]; edges <- c(edges, Map(function(a,b) c(a,b), id_col[-length(id_col)], id_col[-1])) }
  edges <- do.call(rbind, edges); colnames(edges) <- c("from","to")
  cells <- list()
  for (i in seq_len(ngr - 1)) for (j in seq_len(ngr - 1)) {
    a <- mat_id[i, j]; b <- mat_id[i, j + 1]; c <- mat_id[i + 1, j + 1]; d <- mat_id[i + 1, j]
    cells[[length(cells) + 1]] <- c(a, b, c, d)
  }
  cells <- do.call(rbind, cells); colnames(cells) <- c("a","b","c","d")
  list(nodes = centers, edges = as.data.frame(edges), cells = as.data.frame(cells), ngr = ngr)
}
orth_procrustes_R <- function(Qp, Qq){
  k <- ncol(Qp); if (k == 0) return(diag(0))
  sv <- svd(t(Qp) %*% Qq); R <- sv$u %*% t(sv$v)
  if (det(R) < 0) { sv$u[, k] <- -sv$u[, k]; R <- sv$u %*% t(sv$v) }
  R
}
project_to_SO <- function(R){
  sv <- svd(R); Rp <- sv$u %*% t(sv$v)
  if (det(Rp) < 0) { sv$u[, ncol(sv$u)] <- -sv$u[, ncol(sv$u)]; Rp <- sv$u %*% t(sv$v) }
  Rp
}
safe_logSO <- function(R, tol = 1e-8){
  k <- nrow(R); Rp <- project_to_SO(R)
  if (sqrt(sum((Rp - diag(k))^2)) < tol) return(matrix(0, k, k))
  A <- try(expm::logm(Rp), silent = TRUE); if (inherits(A, "try-error") || any(!is.finite(A))) A <- matrix(0, k, k)
  0.5 * (A - t(A))
}
log_SO_fast <- function(R){
  k <- nrow(R)
  if (k == 2) { th <- atan2(R[2,1] - R[1,2], R[1,1] + R[2,2]); return(matrix(c(0, -th, th, 0), 2, 2)) }
  if (k == 3) {
    tr <- sum(diag(R)); cos_th <- max(-1, min(1, (tr - 1) / 2)); th <- acos(cos_th); if (abs(th) < 1e-10) return(matrix(0, 3, 3))
    A <- (R - t(R)) * 0.5; w <- c(A[3,2], A[1,3], A[2,1]); s <- th / (2 * sin(th))
    K <- matrix(c(0, -w[3], w[2], w[3], 0, -w[1], -w[2], w[1], 0), 3, 3, byrow = TRUE)
    return(2 * s * K)
  }
  expm::logm(R)
}
fro_norm <- function(M) sqrt(sum(M * M))

# --------------------------- base dimension auto-detect (1D vs 2D) ---------------------------
auto_base_dim <- function(Yle){
  set.seed(1) #reproductibility
  n <- nrow(Yle)
  fold <- sample(rep(1:3, length.out = n))
  rss_surf <- 0; rss_curve <- 0
  tss3 <- sum((Yle[, 3] - mean(Yle[, 3]))^2)
  
  for(k in 1:3){
    tr <- which(fold != k); te <- which(fold == k)
    
    # surface: LE3 ~ s(u,v) learned on train-PC(2), scored on test
    pc2 <- prcomp(Yle[tr, , drop = FALSE], center = TRUE, scale. = FALSE, rank. = 2)
    Utr <- as.data.frame(pc2$x[, 1:2, drop = FALSE]); names(Utr) <- c("u","v")
    Ute <- as.data.frame(scale(Yle[te, , drop = FALSE], center = pc2$center, scale = FALSE) %*%
                           pc2$rotation[, 1:2, drop = FALSE]); names(Ute) <- c("u","v")
    m3 <- safe_bam(Yle[tr, 3] ~ s(u, v, bs = "tp", k = K_BASIS_UV),
                   data = Utr, k_threads = 1, gamma = 1.2, select = TRUE)
    pred3 <- as.numeric(predict(m3, newdata = Ute))
    rss_surf <- rss_surf + sum((Yle[te, 3] - pred3)^2)
    
    # curve: fit principal curve on train, map test to nearest-train t, predict LE3 via poly(t)
    pcv_tr <- princurve::principal_curve(as.matrix(Yle[tr, 1:3, drop = FALSE]), stretch = 0)
    nn <- RANN::nn2(Yle[tr, 1:3, drop = FALSE], Yle[te, 1:3, drop = FALSE], k = 1)$nn.idx[, 1]
    t_tr <- as.numeric(pcv_tr$lambda); t_te <- t_tr[nn]
    df_tr <- data.frame(y3 = Yle[tr, 3], t = t_tr)
    m1 <- stats::lm(y3 ~ poly(t, 3, raw = TRUE), data = df_tr)
    pred_curve <- as.numeric(stats::predict(m1, newdata = data.frame(t = t_te)))
    rss_curve  <- rss_curve + sum((Yle[te, 3] - pred_curve)^2)
  }
  
  r2_s <- 1 - rss_surf / tss3
  r2_c <- 1 - rss_curve / tss3
  list(dim = if ((r2_s - r2_c) > 0.2) 2L else 1L,
       R2_surface = r2_s,
       R2_curve   = r2_c)
}

# --------------------------- base coordinates and grid ---------------------------
abd <- auto_base_dim(Yle)
BF_BASE_DIM <- abd$dim
cat(sprintf("auto_base_dim → chose %dD (R2_surface=%.3f, R2_curve=%.3f)\n",
            BF_BASE_DIM, abd$R2_surface, abd$R2_curve))

# Use the existing uv_coords from Section I if available; else derive from PCA
if (!exists("uv_coords") || !all(c("u","v") %in% names(uv_coords))) {
  pca_init <- prcomp(Yle, center = TRUE, scale. = FALSE)
  uv_coords <- as.data.frame(pca_init$x[, 1:2, drop = FALSE]); names(uv_coords) <- c("u","v")
  rownames(uv_coords) <- rownames(Yle)
}

if (BF_BASE_DIM == 1L) {
  # 1D base via principal curve
  pcv <- princurve::principal_curve(as.matrix(Yle[, 1:min(3, ncol(Yle)), drop = FALSE]), stretch = 0)
  base_coords <- data.frame(t = pcv$lambda); rownames(base_coords) <- rownames(Yle)
  
  # Build a chain of nodes along t
  t_seq <- seq(min(base_coords$t), max(base_coords$t), length.out = BF_NGR_UV * 2)
  nodes <- data.frame(node_id = seq_along(t_seq), t = t_seq)
  
  # For plotting and later subject-node matching, give nodes a smoothed (u,v)
  df_tuv <- data.frame(t = base_coords$t, u = uv_coords$u, v = uv_coords$v)
  lu <- stats::loess(u ~ t, df_tuv, span = 0.2); lv <- stats::loess(v ~ t, df_tuv, span = 0.2)
  nodes$u <- as.numeric(predict(lu, newdata = data.frame(t = t_seq)))
  nodes$v <- as.numeric(predict(lv, newdata = data.frame(t = t_seq)))
  
  edges  <- data.frame(from = head(nodes$node_id, -1), to = tail(nodes$node_id, -1))
  cells  <- data.frame(a=integer(0), b=integer(0), c=integer(0), d=integer(0))  # no plaquettes in 1D
  base_cols <- "t"
  
} else if (BF_BASE_DIM == 2L) {
  # 2D base via PCA(2)
  pc <- prcomp(Yle, center = TRUE, scale. = FALSE, rank. = 2)
  base_coords <- as.data.frame(pc$x[, 1:2, drop = FALSE]); colnames(base_coords) <- c("u","v")
  rownames(base_coords) <- rownames(Yle)
  UVG   <- build_uv_grid(base_coords, ngr = BF_NGR_UV)
  nodes <- UVG$nodes; edges <- UVG$edges; cells <- UVG$cells
  base_cols <- c("u","v")
  
} else {
  # 3D base via PCA(3) — curvature not computed by default in 3D case
  pc <- prcomp(Yle, center = TRUE, scale. = FALSE, rank. = 3)
  base_coords <- as.data.frame(pc$x[, 1:3, drop = FALSE]); colnames(base_coords) <- c("u1","u2","u3")
  rownames(base_coords) <- rownames(Yle)
  u1 <- seq(min(base_coords$u1), max(base_coords$u1), length.out = BF_NGR_UV)
  u2 <- seq(min(base_coords$u2), max(base_coords$u2), length.out = BF_NGR_UV)
  u3 <- seq(min(base_coords$u3), max(base_coords$u3), length.out = BF_NGR_UV)
  G  <- expand.grid(u1=u1, u2=u2, u3=u3); G$node_id <- seq_len(nrow(G)); nodes <- G
  idx3 <- array(G$node_id, dim = c(BF_NGR_UV, BF_NGR_UV, BF_NGR_UV))
  edges <- list()
  for (i in 1:(BF_NGR_UV-1)) edges <- c(edges, Map(c, idx3[i, , ], idx3[i+1, , ]))
  for (j in 1:(BF_NGR_UV-1)) edges <- c(edges, Map(c, idx3[, j, ], idx3[, j+1, ]))
  for (k in 1:(BF_NGR_UV-1)) edges <- c(edges, Map(c, idx3[, , k], idx3[, , k+1]))
  edges <- as.data.frame(do.call(rbind, edges)); names(edges) <- c("from","to")
  cells <- data.frame(a=integer(0), b=integer(0), c=integer(0), d=integer(0))
  base_cols <- c("u1","u2","u3")
  nodes$u <- nodes$u1; nodes$v <- nodes$u2  # provide (u,v) for mapping/plots
}

# --------------------------- nearest base-neighbourhoods for each node ---------------------------
nn_idx_nodes <- FNN::get.knnx(as.matrix(base_coords[, base_cols, drop = FALSE]),
                              as.matrix(nodes[, base_cols, drop = FALSE]),
                              k = BF_K_BASE)$nn.index

# --------------------------- encode X into numeric, with MCA/FAMD compression -------------------
X_enc <- {
  X_fmr <- X
  for (nm in names(X_fmr)) {
    v <- X_fmr[[nm]]
    if (is.ordered(v) || (is.factor(v) && !is.ordered(v))) X_fmr[[nm]] <- factor(v)
    else if (is.character(v)) X_fmr[[nm]] <- factor(v)
  }
  rownames(X_fmr) <- rownames(X)
  
  has_numeric <- any(vapply(X, is.numeric, logical(1)))
  .choose_ncp <- function(eig_percent, target = 0.99, hard_cap = 100L) {
    cum <- cumsum(eig_percent) / 100
    ncp <- which(cum >= target)[1]; if (is.na(ncp)) ncp <- length(eig_percent)
    as.integer(max(2L, min(ncp, hard_cap)))
  }
  
  if (!has_numeric) {
    mca <- FactoMineR::MCA(X_fmr, ncp = min(nrow(X_fmr) - 1L, 120L), graph = FALSE)
    eig_pct <- if ("percentage of variance" %in% colnames(mca$eig)) mca$eig[,"percentage of variance"] else mca$eig[,2]
    ncp_keep <- .choose_ncp(eig_pct, target = 0.99, hard_cap = 80L)
    enc <- mca$ind$coord[, seq_len(ncp_keep), drop = FALSE]
  } else {
    famd <- FactoMineR::FAMD(X_fmr, ncp = min(nrow(X_fmr) - 1L, 120L), graph = FALSE)
    eig_pct <- if ("percentage of variance" %in% colnames(famd$eig)) famd$eig[,"percentage of variance"] else famd$eig[,2]
    ncp_keep <- .choose_ncp(eig_pct, target = 0.99, hard_cap = 100L)
    enc <- famd$ind$coord[, seq_len(ncp_keep), drop = FALSE]
  }
  enc <- scale(enc, center = TRUE, scale = FALSE)
  rownames(enc) <- rownames(X_fmr)
  if (exists("Yle")) enc <- enc[rownames(Yle), , drop = FALSE]
  enc
}

cat(sprintf("[Compression] p_eff=%d | n=%d | method=%s\n",
            ncol(X_enc), nrow(X_enc),
            if (!any(vapply(X, is.numeric, logical(1)))) "MCA" else "FAMD"))

# --------------------------- local ranks and frames per node (trimmed MP) ------------------------
node_frames <- vector("list", nrow(nodes))
node_ranks  <- integer(nrow(nodes))
node_evals  <- vector("list", nrow(nodes))

for (i in seq_len(nrow(nodes))) {
  idx <- nn_idx_nodes[i, ]; idx <- idx[is.finite(idx)]
  if (length(idx) < BF_MIN_N_NEIGH) { node_frames[[i]] <- NULL; node_ranks[i] <- 0L; next }
  Xi  <- X_enc[idx, , drop = FALSE]
  k0  <- mp_rank(Xi, cap = BF_K_RANK_CAP, shrink = BF_MP_SHRINK)
  node_ranks[i] <- k0
  res <- local_pca_Q(Xi, k = k0)
  node_frames[[i]] <- res$Q; node_evals[[i]] <- res$evals
}

rk <- node_ranks[node_ranks > 0]
if (!length(rk)) {
  k_global <- mp_rank(X_enc, cap = BF_K_RANK_CAP, shrink = 0.80)
  K_FIBRE  <- max(1L, min(BF_K_RANK_CAP, k_global))
  message(sprintf("All local MP ranks were zero. Using global K_FIBRE=%d (fallback).", K_FIBRE))
} else {
  tb <- table(rk); K_FIBRE <- max(1L, min(as.integer(names(tb)[which.max(tb)]), BF_K_RANK_CAP))
}

# Refit frames with the consensus K_FIBRE
for (i in seq_len(nrow(nodes))) {
  idx <- nn_idx_nodes[i, ]
  if (length(idx) < max(BF_MIN_N_NEIGH, K_FIBRE + 2)) { node_frames[[i]] <- NULL; next }
  Xi <- X_enc[idx, , drop = FALSE]
  res <- local_pca_Q(Xi, k = K_FIBRE)
  node_frames[[i]] <- res$Q; node_evals[[i]] <- res$evals
}

# Optional sign/permutation alignment to a global anchor
A0 <- prcomp(X_enc, center = TRUE, scale. = FALSE, rank. = K_FIBRE)$rotation[, 1:K_FIBRE, drop = FALSE]
align_to_anchor <- function(Q, A0){
  if (is.null(Q) || !is.matrix(Q) || !ncol(Q)) return(Q); stopifnot(ncol(Q) == ncol(A0))
  C <- crossprod(A0, Q); C[!is.finite(C)] <- 0
  perm <- clue::solve_LSAP(abs(C), maximum = TRUE)
  Qp  <- Q[, perm, drop = FALSE]
  sgn <- sign(diag(crossprod(A0, Qp))); sgn[sgn == 0] <- 1
  sweep(Qp, 2, sgn, "*")
}
for(i in seq_along(node_frames)){
  if (!is.null(node_frames[[i]]) && ncol(node_frames[[i]]) == K_FIBRE)
    node_frames[[i]] <- align_to_anchor(node_frames[[i]], A0)
}

# Persist frames and grid
saveRDS(list(
  nodes = nodes, edges = edges, cells = cells,
  k_base = BF_K_BASE, k_fibre = K_FIBRE,
  frames = node_frames, evals = node_evals,
  nn_idx_nodes = nn_idx_nodes,
  base_dim = BF_BASE_DIM, base_cols = base_cols
), file = "bundle_frames_nodes.rds")

# --------------------------- discrete connection + MST gauge synchronisation ---------------------
if (K_FIBRE > 0) {
  E <- nrow(edges)
  edge_metrics <- data.frame(from = integer(0), to = integer(0),
                             angle_like = numeric(0), fro_log = numeric(0))
  R_raw <- vector("list", E); sim_w <- numeric(E)
  
  for (e in seq_len(E)) {
    a <- edges$from[e]; b <- edges$to[e]
    Qa <- node_frames[[a]]; Qb <- node_frames[[b]]
    if (is.null(Qa) || is.null(Qb) || ncol(Qa) != K_FIBRE || ncol(Qb) != K_FIBRE) {
      R_raw[[e]] <- diag(K_FIBRE); sim_w[e] <- 0
    } else {
      R <- orth_procrustes_R(Qa, Qb); R <- project_to_SO(R)
      R_raw[[e]] <- R
      s <- svd(t(Qa) %*% Qb)$d; s <- pmin(pmax(s, 0), 1)
      sim_w[e] <- mean(s^2)
    }
  }
  
  # Minimum spanning tree on the grid with weight (1 - similarity)
  g_grid <- igraph::graph_from_data_frame(
    data.frame(from = edges$from, to = edges$to, w = 1 - sim_w),
    directed = FALSE, vertices = data.frame(name = seq_len(nrow(nodes)))
  )
  Tmst <- igraph::mst(g_grid, weights = igraph::E(g_grid)$w)
  
  ekey <- function(i,j) paste0(min(i,j), "_", max(i,j))
  R_map <- setNames(R_raw, mapply(ekey, edges$from, edges$to))
  
  # Choose a central root (median (u,v) if available), propagate gauges along MST
  if (all(c("u","v") %in% names(nodes))) {
    root <- which.min((nodes$u - stats::median(nodes$u, na.rm=TRUE))^2 +
                        (nodes$v - stats::median(nodes$v, na.rm=TRUE))^2)
  } else root <- 1L
  
  G <- vector("list", nrow(nodes)); for (i in seq_len(nrow(nodes))) G[[i]] <- diag(K_FIBRE)
  visited <- rep(FALSE, nrow(nodes)); visited[root] <- TRUE
  front <- as.integer(igraph::neighbors(Tmst, root)); parent <- rep(root, length(front))
  while(length(front)){
    new_front <- integer(0); new_parent <- integer(0)
    for (k in seq_along(front)){
      child <- front[k]; p <- parent[k]
      if (visited[child]) next
      visited[child] <- TRUE
      Rij <- R_map[[ekey(p, child)]]; if (is.null(Rij)) Rij <- diag(K_FIBRE)
      G[[child]] <- t(Rij) %*% G[[p]]
      kids <- setdiff(as.integer(igraph::neighbors(Tmst, child)), which(visited))
      if (length(kids)) { new_front <- c(new_front, kids); new_parent <- c(new_parent, rep(child, length(kids))) }
    }
    front <- new_front; parent <- new_parent
  }
  
  node_frames_sync <- node_frames
  for (i in seq_along(node_frames)) {
    if (!is.null(node_frames[[i]]) && ncol(node_frames[[i]]) == K_FIBRE)
      node_frames_sync[[i]] <- node_frames[[i]] %*% G[[i]]
  }
  
  # Edge rotations + logs + θ-RMS on synchronised frames
  R_edges <- vector("list", E)
  theta_rms <- rep(NA_real_, E)
  for (e in seq_len(E)) {
    a <- edges$from[e]; b <- edges$to[e]
    Qa <- node_frames_sync[[a]]; Qb <- node_frames_sync[[b]]
    if (is.null(Qa) || is.null(Qb)) { R_edges[[e]] <- diag(K_FIBRE); next }
    s <- svd(t(Qa) %*% Qb)$d; s <- pmin(pmax(s, -1), 1); th <- acos(s)
    theta_rms[e] <- sqrt(mean(th^2))
    R <- orth_procrustes_R(Qa, Qb); R <- project_to_SO(R)
    A <- safe_logSO(R)
    ang_like <- if (K_FIBRE == 2) abs(A[1,2]) else sqrt(sum(diag(-A %*% A))) / sqrt(2)
    edge_metrics <- rbind(edge_metrics, data.frame(from=a, to=b,
                                                   angle_like=as.numeric(ang_like),
                                                   fro_log=fro_norm(A)))
    R_edges[[e]] <- R
  }
  write.csv(edge_metrics, "connection_edges.csv", row.names = FALSE)
  write.csv(data.frame(theta_rms = theta_rms), "principal_angles_edges.csv", row.names = FALSE)
  
  # Node-average θ-RMS for mapping
  deg <- rep(0L, nrow(nodes)); theta_sum <- rep(0, nrow(nodes))
  for (e in seq_len(E)) {
    a <- edges$from[e]; b <- edges$to[e]
    if (!is.finite(theta_rms[e])) next
    theta_sum[a] <- theta_sum[a] + theta_rms[e]; deg[a] <- deg[a] + 1L
    theta_sum[b] <- theta_sum[b] + theta_rms[e]; deg[b] <- deg[b] + 1L
  }
  theta_node <- theta_sum / pmax(deg, 1L)
  if (!all(c("u","v") %in% names(nodes))) { nodes$u <- NA_real_; nodes$v <- NA_real_ }
  write.csv(data.frame(node_id = seq_len(nrow(nodes)), u = nodes$u, v = nodes$v, theta = theta_node),
            "principal_angles_nodes.csv", row.names = FALSE)
  
  # Curvature map over 2D plaquettes only
  if (nrow(cells) > 0) {
    curv_map <- data.frame(u = numeric(0), v = numeric(0), curv = numeric(0))
    R_lookup <- setNames(R_edges, mapply(ekey, edges$from, edges$to))
    rget <- function(i, j) {
      key <- ekey(i, j); R <- R_lookup[[key]]; if (!is.null(R)) return(R)
      R <- R_lookup[[ekey(j, i)]]; if (!is.null(R)) return(t(R))
      diag(K_FIBRE)
    }
    for (c in seq_len(nrow(cells))) {
      a <- cells$a[c]; b <- cells$b[c]; d <- cells$d[c]; cc <- cells$c[c]
      Rab <- rget(a,b); Rbc <- rget(b,cc); Rcd <- rget(cc,d); Rda <- rget(d,a)
      H <- Rab %*% Rbc %*% Rcd %*% Rda; F <- log_SO_fast(H)
      curv_map <- rbind(curv_map, data.frame(
        u = mean(nodes$u[c(a,b,cc,d)], na.rm=TRUE),
        v = mean(nodes$v[c(a,b,cc,d)], na.rm=TRUE),
        curv = fro_norm(F)
      ))
    }
    write.csv(curv_map, "curvature_map_uv.csv", row.names = FALSE)
  } else {
    write.csv(data.frame(u=numeric(0), v=numeric(0), curv=numeric(0)), "curvature_map_uv.csv", row.names = FALSE)
  }
  
} else {
  node_frames_sync <- node_frames
  write.csv(data.frame(from=integer(0), to=integer(0), angle_like=numeric(0), fro_log=numeric(0)),
            "connection_edges.csv", row.names = FALSE)
  write.csv(data.frame(theta_rms=numeric(0)), "principal_angles_edges.csv", row.names = FALSE)
  if (!all(c("u","v") %in% names(nodes))) { nodes$u <- NA_real_; nodes$v <- NA_real_ }
  write.csv(data.frame(node_id=integer(0), u=numeric(0), v=numeric(0), theta=numeric(0)),
            "principal_angles_nodes.csv", row.names = FALSE)
  write.csv(data.frame(u = numeric(0), v = numeric(0), curv = numeric(0)), "curvature_map_uv.csv", row.names = FALSE)
}

# --------------------------- fibre energy per subject & density ---------------------------
# nearest node per subject in base space
nn_node_for_subject <- FNN::get.knnx(as.matrix(nodes[, base_cols, drop = FALSE]),
                                     as.matrix(base_coords[, base_cols, drop = FALSE]),
                                     k = 1)$nn.index[,1]

E_fibre <- rep(0, nrow(X_enc))
if (K_FIBRE > 0) {
  for (i in seq_len(nrow(X_enc))) {
    node_id <- nn_node_for_subject[i]
    Qp <- node_frames_sync[[node_id]]
    if (is.null(Qp) || ncol(Qp) == 0) { E_fibre[i] <- 0; next }
    idxN <- nn_idx_nodes[node_id, ]
    mu <- colMeans(X_enc[idxN, , drop = FALSE])
    xi <- X_enc[i, ] - mu
    zi <- as.numeric(xi %*% Qp)
    E_fibre[i] <- sum(zi^2)
  }
}

# Persist fibre energy (with (u,v) for mapping)
ids_rows <- rownames(Yle)
fibre_df <- if (all(c("u","v") %in% names(uv_coords))) {
  data.frame(row_id = ids_rows, u = uv_coords$u, v = uv_coords$v, E_fibre = E_fibre)
} else {
  data.frame(row_id = ids_rows, E_fibre = E_fibre)
}
write.csv(fibre_df, "fibre_energy_by_subject.csv", row.names = FALSE)

# kNN density in base coordinates: scale with intrinsic base dimension
den_nn <- FNN::get.knn(as.matrix(base_coords[, base_cols, drop = FALSE]), k = BF_DENS_K)$nn.dist
rk <- den_nn[, BF_DENS_K]
density_base <- 1 / pmax(rk^BF_BASE_DIM, .Machine$double.eps)

# Composite unsupervised risk: robust z-normalisation of contributors
curv_map <- tryCatch(read.csv("curvature_map_uv.csv"), error = function(e) data.frame())
nnc <- if (nrow(curv_map) && all(c("u","v") %in% names(curv_map)) && all(c("u","v") %in% names(uv_coords))) {
  FNN::get.knnx(as.matrix(curv_map[, c("u","v")]), as.matrix(uv_coords), k = 1)$nn.index[,1]
} else rep(1L, nrow(uv_coords))
curv_at_subject <- if (nrow(curv_map)) curv_map$curv[nnc] else rep(0, nrow(uv_coords))
curv_at_subject[!is.finite(curv_at_subject)] <- 0

theta_node_df <- tryCatch(read.csv("principal_angles_nodes.csv"), error = function(e) NULL)
nnn <- if (!is.null(theta_node_df) && all(c("u","v") %in% names(nodes)) && all(c("u","v") %in% names(uv_coords))) {
  FNN::get.knnx(as.matrix(nodes[,c("u","v")]), as.matrix(uv_coords), k = 1)$nn.index[,1]
} else rep(1L, nrow(uv_coords))
theta_at_subject <- if (!is.null(theta_node_df)) theta_node_df$theta[nnn] else rep(0, nrow(uv_coords))
theta_at_subject[!is.finite(theta_at_subject)] <- 0

nz <- function(x){
  z <- (x - stats::median(x, na.rm = TRUE)) / (stats::mad(x, constant = 1.4826, na.rm = TRUE) + 1e-9)
  pmin(pmax(z, -6), 6)
}
z_fibre <- nz(E_fibre)
z_theta <- nz(theta_at_subject)
z_curv  <- nz(curv_at_subject)
z_dinv  <- nz(-log(pmax(density_base, 1e-12)))

R_u <- scales::rescale(z_fibre + z_theta + z_dinv, to = c(0,1),
                       from = range(z_fibre + z_theta + z_dinv, na.rm = TRUE))
risk_df <- data.frame(
  row_id = rownames(X_enc),
  u = uv_coords$u, v = uv_coords$v,
  z_fibre = as.numeric(z_fibre), z_theta = as.numeric(z_theta),
  z_dinv  = as.numeric(z_dinv),  R_u = as.numeric(R_u)
)
write.csv(risk_df, "unsupervised_risk_subjects.csv", row.names = FALSE)

# Legacy curvature-based composite (backwards-compatible)
R_u_legacy <- scales::rescale(z_fibre + z_curv + z_dinv, to = c(0,1),
                              from = range(z_fibre + z_curv + z_dinv, na.rm = TRUE))
risk_legacy <- data.frame(
  row_id = rownames(X_enc),
  u = uv_coords$u, v = uv_coords$v,
  z_fibre = as.numeric(z_fibre), z_curv = as.numeric(z_curv),
  z_dinv = as.numeric(z_dinv), R_u = as.numeric(R_u_legacy)
)
write.csv(risk_legacy, "unsupervised_risk_subjects_legacy_curv.csv", row.names = FALSE)

# =================================================================================================
# L) Coordinates in principal curve: (t, w) around a principal curve in LE(1:3)
# =================================================================================================
# Goal:
#   - Provide an alternative 1D+1D coordinate system: t = curve parameter, w = signed deviation
#     along a locally estimated normal direction.
#   - Quick check: how much (t, w) explains LE3 via a linear model with poly(t) + w.
# Notes:
#   - This is diagnostic and does not affect outputs elsewhere.
# =================================================================================================

pcv   <- princurve::principal_curve(as.matrix(Yle[,1:3]), stretch = 0)
t     <- pcv$lambda                # curve parameter per sample
C     <- pcv$s                     # on-curve closest points per sample
Rres  <- as.matrix(Yle[,1:3]) - C  # residual vector to the curve

# Estimate a local normal direction via PCA on residuals in a sliding window along t
w <- numeric(nrow(Yle))
kwin <- 15
ord  <- order(t)
for (i in seq_along(t)) {
  j <- ord[pmax(1, which(ord == i) - kwin) : pmin(length(ord), which(ord == i) + kwin)]
  N <- prcomp(Rres[j, , drop = FALSE], center = TRUE, scale. = FALSE)$rotation[,1]
  w[i] <- as.numeric(Rres[i, ] %*% N)
}

# Simple explanatory model: does (t, w) capture LE3 variation?
fit_tw <- summary(stats::lm(Yle[,3] ~ poly(t, 3, raw = TRUE) + w))
print(fit_tw)


# =================================================================================================
# N) REPORTER — quick end-of-run summaries & sanity checks
# =================================================================================================
# Goal:
#   - Print a concise “receipt” of what was produced and a couple of sanity plots/stats
#     to catch obvious issues early (file missing, empty frames, etc.).
#   - Keep this robust: failures should not crash the whole script — they should inform.
# =================================================================================================

quiet_file_n <- function(p) if (file.exists(p)) nrow(utils::read.csv(p, nrows = 5)) else NA_integer_

cat("\n[REPORT] Bundle + Sweep pipeline summaries\n")

# Compression summary (from Section K)
if (exists("X_enc")) {
  cat(sprintf("  Compression: p_eff=%d  n=%d  method=%s\n",
              ncol(X_enc), nrow(X_enc),
              if (!any(vapply(X, is.numeric, logical(1)))) "MCA" else "FAMD"))
}

# Base dimension
if (exists("BF_BASE_DIM")) cat(sprintf("  Base dimension: d=%d\n", BF_BASE_DIM))

# Frames & connection (Section K)
bundle_rds_path <- "bundle_frames_nodes.rds"
if (!file.exists(bundle_rds_path)) {
  cat("  !! bundle_frames_nodes.rds not found — frames were not saved.\n")
} else {
  B <- readRDS(bundle_rds_path)
  K_FIBRE <- B$k_fibre; n_nodes <- nrow(B$nodes)
  hasQ <- vapply(B$frames, function(Q) is.matrix(Q) && ncol(Q) == K_FIBRE, logical(1))
  cat(sprintf("  Frames: k_fibre=%d | valid frames %d / %d nodes\n",
              K_FIBRE, sum(hasQ), n_nodes))
  if (sum(hasQ) < 4) cat("  .. too few frames for connection/curvature to be informative.\n")
}

conn_path <- "connection_edges.csv"
ne <- quiet_file_n(conn_path)
if (is.na(ne)) {
  cat("  !! connection_edges.csv missing.\n")
} else {
  conn <- read.csv(conn_path)
  if (nrow(conn) == 0) {
    cat("  Connection: file present but empty (no valid edges).\n")
  } else {
    qs <- stats::quantile(conn$fro_log, c(.25,.5,.75,.95), na.rm=TRUE)
    cat(sprintf("  Connection: |log R| Frobenius quantiles: Q2=%.4f  Q95=%.4f  (n_edges=%d)\n",
                qs[2], qs[4], nrow(conn)))
    hist(conn$fro_log, breaks = 50, main = "|log R| across edges", xlab = "Frobenius(log R)")
  }
}

curv_path <- "curvature_map_uv.csv"
nc <- quiet_file_n(curv_path)
if (is.na(nc)) {
  cat("  !! curvature_map_uv.csv missing.\n")
} else {
  curv <- read.csv(curv_path)
  if (nrow(curv) == 0) {
    cat("  Curvature: file present but empty (no plaquettes or frames / dim<2).\n")
  } else {
    qs <- stats::quantile(curv$curv, c(.25,.5,.75,.95), na.rm=TRUE)
    cat(sprintf("  Curvature: ||F|| quantiles: Q2=%.4f  Q95=%.4f  (cells=%d)\n",
                qs[2], qs[4], nrow(curv)))
    with(curv, plot(u, v, pch = 16, cex = 0.6, col = cut(curv, 12),
                    main = "Curvature heat (point view)", xlab = "u", ylab = "v"))
  }
}

pang_path <- "principal_angles_edges.csv"
if (file.exists(pang_path)) {
  pang <- read.csv(pang_path)
  if (nrow(pang)) {
    qs <- stats::quantile(pang$theta_rms, c(.25,.5,.75,.95), na.rm=TRUE)
    cat(sprintf("  Principal-angle RMS: Q2=%.4f rad  Q95=%.4f rad  (n_edges=%d)\n",
                qs[2], qs[4], nrow(pang)))
    hist(pang$theta_rms, breaks = 50, main = "RMS principal angle across edges", xlab = "radians")
  }
}

fib_path  <- "fibre_energy_by_subject.csv"
risk_path <- "unsupervised_risk_subjects.csv"
if (file.exists(fib_path) && file.exists(risk_path)) {
  fib <- read.csv(fib_path); ris <- read.csv(risk_path)
  if (nrow(fib) == nrow(ris)) {
    c12 <- suppressWarnings(cor(fib$E_fibre, ris$z_dinv, use = "pair"))
    c13 <- suppressWarnings(cor(ris$R_u, ris$z_theta, use = "pair"))
    cat(sprintf("  E_fibre vs inverse-density z: r=%.3f | Risk vs theta z: r=%.3f (n=%d)\n",
                c12, c13, nrow(fib)))
    if (all(c("u","v") %in% names(fib))) {
      plot(fib$u, fib$v, pch = 16, cex = .6, col = cut(fib$E_fibre, 12),
           main = "Fibre energy in (u,v)", xlab = "u", ylab = "v")
    }
  } else {
    cat("  !! fibre/risk row count mismatch.\n")
  }
} else {
  cat("  (fibre_energy_by_subject.csv or unsupervised_risk_subjects.csv missing)\n")
}

# PHATE sweep summary (Section M)
sweep_path <- "dimensional_sweep_principal_manifold_R2.csv"
if (file.exists(sweep_path)) {
  sw <- read.csv(sweep_path)
  best_idx <- which.max(replace(sw$R2_like, is.na(sw$R2_like), -Inf))
  if (length(best_idx) && is.finite(best_idx)) {
    cat(sprintf("  Sweep: best nd=%d (%s) with R2-like=%.3f\n",
                sw$ndim[best_idx], sw$method[best_idx], sw$R2_like[best_idx]))
  }
} else {
  cat("  (dimensional_sweep_principal_manifold_R2.csv missing)\n")
}

cat("[END REPORT]\n\n")

