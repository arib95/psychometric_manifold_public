# ==============================================================================
# DX SPACE: Identification + Clustering + Diagnostics
# ------------------------------------------------------------------------------
# Purpose
#   End-to-end pipeline to:
#     1) Load a wide 0/1 diagnosis matrix at participant level (or use DX in mem).
#     2) Deduplicate identical diagnosis profiles and keep multiplicities.
#     3) Compute asymmetric-binary Gower distances on unique non-zero profiles.
#     4) Build a weighted kNN graph with optional local scaling and multiplicity
#        upweighting; run Louvain community detection with stability and nulls.
#     5) Quantify intrinsic dimensionality (TwoNN and Levina–Bickel on a core band).
#     6) Assess clustering significance (modularity Q, silhouette S) under two
#        nulls: weight-shuffle and degree-preserving rewires; bootstrap ARI.
#     7) Select "major" diagnoses using three independent pillars:
#          A) Cluster enrichment (Fisher/binomial with FDR and OR guards).
#          B) Label localisation (assortativity and kNN purity with label-shuffle p).
#          C) One-vs-rest predictability (kNN score AUC with multiplicity weights).
#     8) Sensitivity sweep over k and union vs mutual kNN; ΔQ per diagnosis; plots.
#
# Inputs
#   - File: wide_diagnoses.csv (tab-delimited) with columns:
#       participant_id, DX1, DX2, ..., DXp  (each DX* is 0/1/NA)
#     Alternatively, an object DX can be present in the workspace with the same
#     structure, in which case DX_CSV_PATH is ignored.
#
# Outputs (written to working directory)
#   - dx_space_run_summary.csv                         (compact run summary)
#   - cluster_membership_all_participants.csv          (expanded labels)
#   - modularity_degree_null.csv                       (degree-null draws)
#   - cluster_bootstrap_ari.csv                        (bootstrap ARIs)
#   - cluster_diagnosis_enrichment.csv                 (pillar A table)
#   - selected_major_diagnoses.csv                     (majors from A)
#   - dx_label_localization.csv                        (pillar B table)
#   - dx_predictability_auc_knn.csv                    (pillar C table)
#   - selected_major_diagnoses_union.csv               (A ∪ B ∪ C, final guards)
#   - dx_deltaQ.csv                                    (ΔQ when zeroing each DX)
#   - clustering_sensitivity_grid.csv                  (k × variant sweep)
#   - major_dx_bootstrap_frequency_by_eligibility.csv  (stability of majors)
#   - Figures: FIG_* PNG/HTML as produced below.
#
# Reproducibility
#   - Random seed set at the top.
#   - Prevalence guards treat NA as 0 only for keep/drop decisions; modelling uses
#     0/1 with NA coerced to 0 conservatively where needed.
#   - All thresholds are surfaced at the Parameters block.
#
# Dependencies
#   - CRAN: readr, dplyr, tidyr, utils, cluster, RANN, RSpectra, dbscan, igraph,
#           ggplot2, Matrix, aricode, glmnet, vegan, princurve, mgcv, reticulate,
#           MASS, R.utils, FNN, expm, clue, FactoMineR, scales, plotly,
#           htmlwidgets, ggrepel, GGally (for pairs plot).
#
# Notes
#   - All distances are computed on unique non-zero diagnosis profiles only.
#   - kNN graph is symmetric by union or mutual rule; weights are Gaussian with
#     either global sigma or per-node local scale from k-th neighbour distance.
#   - Multiplicity upweights edges between frequent profiles to respect sample size.
#   - This script is compute-heavy in the bootstrap and sweep sections. Adjust B
#     and grid sizes if runtime is a concern.
# ==============================================================================

# ============================= Package imports =================================
suppressPackageStartupMessages({
  library(readr);   library(dplyr);  library(tidyr);   library(utils)
  library(cluster); library(RANN);   library(RSpectra); library(dbscan)
  library(igraph);  library(ggplot2);library(Matrix);  library(aricode)
  library(glmnet);  library(vegan);  library(princurve);library(mgcv)
  library(reticulate);               library(MASS);     library(R.utils)
  library(FNN);     library(expm);   library(clue);     library(FactoMineR)
  library(scales);  library(plotly); library(htmlwidgets); library(ggrepel)
})

set.seed(42)

# ================================ Parameters ===================================
# Data
DX_CSV_PATH        <- "wide_diagnoses.csv"  # used only if DX is not already in memory
INCLUDE_NODIAG     <- TRUE                   # if no NODIAG col, create it from rows with no DX=1
DX_MIN_PREV        <- 0.00                   # keep DX with prevalence in [min,max]
DX_MAX_PREV        <- 0.99
MIN_CASES_TOTAL    <- 10                     # minimum total cases required to keep a DX

# kNN graph parameters
K_KNN              <- 8                      # check cluster_sensitivity_grid
UNION_KNN          <- FALSE                  # TRUE: union-kNN; FALSE: mutual-kNN

# significance + stability
N_PERM             <- 200                    # permutations for Q/S nulls
N_BOOT             <- 50                     # bootstrap re-cluster repetitions
MIN_CLUSTER_SIZE   <- 10                     # on deduplicated profiles (pre-expansion)
MIN_CLUSTER_WEIGHT <- 25                     # participants (sum of mult) required to keep a community

# enrichment + major Dx selection thresholds (pillar A)
ALPHA_FDR          <- 0.05
MIN_PREV_IN_CL     <- 0.10
MIN_OR             <- 2.0

# three-pillars thresholds (B,C)
ALPHA_LOCALIZE     <- 0.05                   # min(assort p, kNN-purity p) for pillar B
AUC_MIN            <- 0.70                   # one-vs-rest predictability for pillar C
PREV_MIN           <- 0.03                   # ≥3% overall or at least NCASE_MIN cases
NCASE_MIN          <- 10
NIN_MIN            <- 5                      # ≥5 cases inside a cluster for enrichment test
NOUT_MIN           <- 5                      # ≥5 cases outside cluster for enrichment test
DENY_NOS           <- TRUE                   # drop *NOS diagnoses from majors by default

# plotting
PLOT_WIDTH         <- 7
PLOT_HEIGHT        <- 6
PLOT_DPI           <- 150

# k-sweep grid
K_GRID             <- c(5,6,7,8,9,10,11,12,13,14,15)
VARIANT            <- c("union","mutual")

# ================================ Helpers ======================================
#' Safe filename helper: replace non [A-Za-z0-9_] with underscores
safe_file <- function(s) gsub("[^A-Za-z0-9_]+","_", s)

#' n choose 2 with vectorised guard
comb2 <- function(n) ifelse(n < 2, 0, n*(n-1)/2)

#' Report basic kNN graph diagnostics
report_knn_graph <- function(g){
  comps <- igraph::components(g)
  cat(sprintf("[kNN graph] V=%d, E=%d, components=%d, largest=%d (%.1f%%), isolates=%d (%.1f%%)\n",
              igraph::gorder(g), igraph::ecount(g),
              comps$no, max(comps$csize), 100*max(comps$csize)/igraph::gorder(g),
              sum(comps$csize==1), 100*sum(comps$csize==1)/igraph::gorder(g)))
}

#' Adjusted Rand Index for integer label vectors with optional zero-drop
align_ari <- function(a, b, drop_zero = TRUE) {
  if (is.null(names(a))) names(a) <- as.character(seq_along(a))
  if (is.null(names(b))) names(b) <- as.character(seq_along(b))
  ii <- intersect(names(a), names(b))
  aa <- as.integer(a[ii]); bb <- as.integer(b[ii])
  if (drop_zero) {
    keep <- (aa != 0L) & (bb != 0L)
    aa <- aa[keep]; bb <- bb[keep]
  }
  if (length(aa) < 2L || length(unique(aa)) < 2L || length(unique(bb)) < 2L)
    return(NA_real_)
  adj_rand_index(aa, bb)
}

#' Choose MDS dimensionality by out-of-sample R^2(dist)
cv_r2_mds <- function(D, k_grid = 1:8, K = 5, seed = 42){
  set.seed(seed)
  Dm <- as.matrix(D); n <- nrow(Dm); stopifnot(ncol(Dm) == n)
  folds <- sample(rep(1:K, length.out = n))
  res <- vector("list", length(k_grid))
  for (kk in seq_along(k_grid)){
    k <- k_grid[kk]
    r2_fold <- rep(NA_real_, K)
    for (f in 1:K){
      te <- which(folds == f); tr <- setdiff(seq_len(n), te)
      if (length(tr) < 3L || length(te) < 2L) next
      # TRAIN: classical MDS on squared distances
      Dtr2 <- Dm[tr, tr, drop = FALSE]^2
      ntr  <- length(tr)
      Htr  <- diag(ntr) - matrix(1, ntr, ntr)/ntr
      Btr  <- -0.5 * Htr %*% Dtr2 %*% Htr
      E    <- eigen(Btr, symmetric = TRUE)
      pos  <- which(E$values > 1e-12)
      if (!length(pos)) next
      m      <- min(k, length(pos))
      vals   <- E$values[pos][1:m]
      V      <- E$vectors[, pos, drop = FALSE][, 1:m, drop = FALSE]
      Xtr    <- sweep(V, 2, sqrt(vals), `*`)  # train coords
      # TEST: out-of-sample projection using train centring
      Dte2 <- Dm[te, tr, drop = FALSE]^2
      cmtr <- colMeans(Dtr2)
      rmte <- rowMeans(Dte2)
      mu   <- mean(Dtr2)
      Gte  <- -0.5 * ( Dte2
                       - matrix(rmte,  nrow(Dte2), ncol(Dte2), byrow = FALSE)
                       - matrix(cmtr,  nrow(Dte2), ncol(Dte2), byrow = TRUE)
                       + mu )
      Xte <- sweep(Gte %*% V, 2, 1/sqrt(vals), `*`)
      # R^2 on combined train+test
      X    <- rbind(Xtr, Xte)
      idx  <- c(tr, te)
      dhat <- as.numeric(dist(X))
      dref <- as.numeric(as.dist(Dm[idx, idx, drop = FALSE]))
      r2_fold[f] <- suppressWarnings(cor(dhat, dref)^2)
    }
    res[[kk]] <- data.frame(k = k, r2 = mean(r2_fold, na.rm = TRUE))
  }
  do.call(rbind, res)
}

#' Counts from participant table (not unique-profiles)
build_counts <- function(DX){
  cols <- setdiff(names(DX), "participant_id")
  n1   <- vapply(cols, function(nm) sum(DX[[nm]] == 1L, na.rm = TRUE), integer(1))
  n0   <- nrow(DX) - n1
  prev <- n1 / (n1 + n0)
  data.frame(dx = cols, n1 = as.integer(n1), n0 = as.integer(n0),
             prev = as.numeric(prev), stringsAsFactors = FALSE)
}

#' Core ARI implementation used by align_ari
adj_rand_index <- function(a, b){
  a <- as.integer(as.factor(a)); b <- as.integer(as.factor(b))
  tab <- table(a, b); n <- sum(tab)
  sum_ij <- sum(comb2(tab))
  sum_i  <- sum(comb2(rowSums(tab)))
  sum_j  <- sum(comb2(colSums(tab)))
  expected <- sum_i * sum_j / comb2(n)
  max_idx  <- 0.5 * (sum_i + sum_j)
  denom    <- max_idx - expected
  if (denom <= 0) return(0)
  (sum_ij - expected) / denom
}

#' TwoNN intrinsic dimension estimate on a distance object
#' Tie-safe and trimmed on log(r2/r1)
twonn_id_from_dist <- function(D, eps=1e-8, trim=0.02){
  M <- as.matrix(D); n <- nrow(M); if (n < 3) return(NA_real_)
  diag(M) <- Inf
  r <- vapply(seq_len(n), function(i){
    di <- M[i, ]; di <- di[is.finite(di)]; if (length(di) < 2) return(NA_real_)
    ds <- sort(di, partial=2)[1:2]
    d1 <- max(ds[1], eps); d2 <- max(ds[2], d1+eps)
    d2/d1
  }, numeric(1))
  r <- r[is.finite(r) & r > 1]; if (!length(r)) return(NA_real_)
  logr <- sort(log(r))
  k <- floor(trim * length(logr))
  if (k > 0 && 2*k < length(logr)) logr <- logr[(k+1):(length(logr)-k)]
  1/mean(logr)
}

#' Levina–Bickel MLE of intrinsic dimension on a dense matrix
lb_mle_id <- function(Dm, k_lo=5, k_hi=15){
  Dm <- as.matrix(Dm); n <- nrow(Dm); diag(Dm) <- Inf
  if (n <= k_lo) return(NA_real_)
  k_hi <- max(k_lo, min(k_hi, n-1))
  ids <- sapply(k_lo:k_hi, function(k){
    nn <- t(apply(Dm, 1L, function(r){
      rf <- r[is.finite(r)]; m <- length(rf); if (m < k) return(rep(NA_real_, k))
      sort(rf, partial=k)[1:k]
    }))
    if (!nrow(nn)) return(NA_real_)
    l  <- log(nn[,k,drop=TRUE]/nn[,1:(k-1),drop=FALSE])
    d  <- 1/rowMeans(l, na.rm=TRUE)
    mean(d[is.finite(d)], na.rm=TRUE)
  })
  mean(ids, na.rm=TRUE)
}

#' Select a central band of points by k-th neighbour distance
core_band_idx <- function(D, k=10, band=c(0.15,0.85)){
  M <- as.matrix(D); n <- nrow(M); diag(M) <- Inf
  if (n < 3) return(integer(0))
  kth <- function(r){
    rf <- r[is.finite(r)]; if (!length(rf)) return(NA_real_)
    k_eff <- min(k, length(rf)); sort(rf, partial=k_eff)[k_eff]
  }
  rk <- apply(M, 1, kth); ok <- is.finite(rk); if (!any(ok)) return(integer(0))
  q <- stats::quantile(rk[ok], band, na.rm=TRUE)
  which(ok & rk >= q[1] & rk <= q[2])
}

#' Asymmetric-binary Gower distance builder for a DX table
#' Returns a dist object or NULL if fewer than 2 varying columns
gower_dist_dx <- function(DXdf){
  cols0 <- setdiff(names(DXdf), c("participant_id","mult", "ANY_DX"))
  vary <- vapply(cols0, function(nm){
    v <- DXdf[[nm]]; u <- unique(v[is.finite(v)])
    length(u) >= 2 && all(u %in% c(0L,1L))
  }, logical(1))
  cols <- cols0[vary]
  if (length(cols) < 2L) return(NULL)
  suppressWarnings(
    cluster::daisy(DXdf[, cols, drop=FALSE], metric="gower",
                   type=list(asymm=cols), stand=FALSE, weights=rep(1,length(cols)))
  )
}

#' Load DX from path unless DX exists in memory; enforce prevalence/count guards
load_dx_wide <- function(path, include_nodiag = TRUE, min_prev = 0.01, max_prev = 0.99){
  if (exists("DX") && is.data.frame(get("DX", inherits = TRUE))) {
    DX0 <- get("DX", inherits = TRUE)
  } else {
    DX0 <- readr::read_delim(path, delim = "\t", col_types = readr::cols())
  }
  stopifnot("participant_id" %in% names(DX0))
  
  # Optionally add NODIAG and ANY_DX here so both survive column filtering.
  dx_cols0 <- setdiff(names(DX0), "participant_id")
  any_dx   <- as.integer(rowSums(sapply(dx_cols0, function(nm) as.integer(DX0[[nm]] == 1)), na.rm = TRUE) > 0)
  if (include_nodiag && !"NODIAG" %in% names(DX0)) DX0$NODIAG <- 1L - any_dx
  if (!"ANY_DX" %in% names(DX0)) DX0$ANY_DX <- any_dx
  
  cols <- setdiff(names(DX0), "participant_id")  # recompute AFTER adding NODIAG/ANY_DX
  DX1  <- DX0[, c("participant_id", cols), drop = FALSE]
  
  # force 0/1/NA and coerce to integer
  for (nm in cols){
    v <- suppressWarnings(as.numeric(DX1[[nm]]))
    if (!all(v %in% c(0,1,NA))) stop(sprintf("Column %s is not 0/1.", nm))
    DX1[[nm]] <- as.integer(v)
  }
  
  n <- nrow(DX1)
  pos_counts <- vapply(cols, function(nm) sum(DX1[[nm]] == 1L, na.rm = TRUE), integer(1))
  prev_all   <- pos_counts / n
  
  # keep by prevalence/count, but never drop NODIAG/ANY_DX if present
  protected <- intersect(c("NODIAG","ANY_DX"), cols)
  keep <- unique(c(
    names(prev_all)[ prev_all >= min_prev & prev_all <= max_prev & pos_counts >= MIN_CASES_TOTAL ],
    protected
  ))
  if (!length(keep)) stop("No diagnosis columns pass prevalence/count filters.")
  DX1 <- DX1[, c("participant_id", keep), drop = FALSE]
  DX1 <- as.data.frame(DX1)
  rownames(DX1) <- make.unique(as.character(DX1$participant_id))
  DX1
}

#' Deduplicate identical diagnosis signatures; keep first row and a multiplicity
#' column called "mult"; returns list with DXu, sig vector for all rows, and mult
dedup_dx <- function(DX){
  cols <- setdiff(names(DX), "participant_id")
  
  # For dedup *only*, treat NA as 0 so missingness doesn't create fake "new" profiles
  X <- as.data.frame(DX[, cols, drop = FALSE])
  for (nm in cols) { v <- X[[nm]]; v[is.na(v)] <- 0L; X[[nm]] <- as.integer(v) }
  
  sig <- apply(X, 1, paste0, collapse = "")
  tab <- as.data.frame(table(sig), stringsAsFactors = FALSE)
  names(tab) <- c("sig", "Freq")
  
  first_idx <- tapply(seq_len(nrow(DX)), sig, `[`, 1)   # named by sig
  DXu <- DX[unname(first_idx), , drop = FALSE]
  # multiplicities aligned by signature, not by rownames
  DXu$mult <- as.integer(tab$Freq[ match(names(first_idx), tab$sig) ])
  
  list(DXu = DXu, sig = sig, mult = DXu$mult)
}

#' Build a symmetric kNN graph from a distance with Gaussian weights
#' Optional: local scaling using per-node k-th neighbour distance; multiplicity boost
knn_graph_from_dist <- function(D, k = 12, union = TRUE, mult = NULL,
                                local_scale = TRUE, add_mst = FALSE){
  M <- as.matrix(D); n <- nrow(M); diag(M) <- Inf
  # k-th neighbour distance per node
  kth <- function(r, k){
    rf <- r[is.finite(r)]
    if (!length(rf)) return(NA_real_)
    k_eff <- min(k, length(rf))
    sort(rf, partial = k_eff)[k_eff]
  }
  rk <- apply(M, 1, kth, k = k)
  # local or global scale
  if (local_scale) {
    sigma_i <- rk; sigma_i[!is.finite(sigma_i) | sigma_i <= 0] <- median(M[is.finite(M)])
  } else {
    sigma <- stats::median(rk[is.finite(rk)], na.rm = TRUE)
    if (!is.finite(sigma) || sigma <= 0) sigma <- stats::median(M[is.finite(M)], na.rm = TRUE)
    if (!is.finite(sigma) || sigma <= 0) sigma <- 1
  }
  # neighbour lists
  nbrs <- lapply(seq_len(n), function(i){
    di <- M[i, ]; ok <- which(is.finite(di))
    if (!length(ok)) integer(0) else ok[order(di[ok])[seq_len(min(k, length(ok)))]]
  })
  edges <- list()
  for (i in seq_len(n)){
    js <- nbrs[[i]]; if (!length(js)) next
    partners <- if (union) {
      unique(c(js, which(vapply(nbrs, function(x) i %in% x, logical(1)))))
    } else {
      intersect(js, which(vapply(nbrs, function(x) i %in% x, logical(1))))
    }
    partners <- partners[partners > i]
    if (!length(partners)) next
    if (local_scale) {
      sprod <- outer(sigma_i[i], sigma_i[partners], "*")
      w <- exp(-(M[i, partners]^2) / (2 * sprod))
    } else {
      w <- exp(-(M[i, partners]^2) / (2 * sigma^2))
    }
    # multiplicity upweighting
    if (!is.null(mult)) {
      m_i  <- if (is.finite(mult[i]) && mult[i] > 0) mult[i] else 1
      m_js <- ifelse(is.finite(mult[partners]) & mult[partners] > 0, mult[partners], 1)
      w <- w * sqrt(pmax(1, m_i) * pmax(1, m_js))
    }
    edges[[length(edges) + 1L]] <- data.frame(from = i, to = partners, weight = pmax(1e-12, w))
  }
  if (!length(edges)) stop("Empty kNN edge set.")
  Edf <- do.call(rbind, edges)
  # declare all vertices so isolates are retained
  g <- igraph::graph_from_data_frame(
    Edf,
    directed = FALSE,
    vertices = data.frame(name = seq_len(n))
  )
  g <- igraph::simplify(
    g, remove.multiple = TRUE, remove.loops = TRUE,
    edge.attr.comb = list(weight = "sum")
  )
  # optional: MST backbones could be added here if single-component is required
  g
}

# Louvain clustering with robustness, nulls, and guards
#
# - Does not reseed unless a non-NULL seed is provided.
# - Keeps only communities with size >= min_size; others are labelled 0.
# - Robust silhouette via try(); allows single-cluster paths.
# - Weight-shuffle nulls -> returns Q_p_upper (one-sided), Q_p_two (two-sided), S_p, and z_w.
# - Degree-preserving nulls via rewires -> returns z_deg and p_deg.
# - Restores original edge weights after shuffles.
# - NA guards when n_perm == 0 (p-values and z become NA).
louvain_with_stats <- function(
    g, D, n_perm = 200, min_size = 10, seed = NULL,
    deg_reps = 200L, deg_keep_connected = TRUE,
    null_scope = c("kept", "full"),
    v_weight = NULL,                # <— new
    min_weight = NULL               # <— new
){
  null_scope <- match.arg(null_scope)
  if (!is.null(seed)) set.seed(seed)
  
  cl_full <- igraph::cluster_louvain(g, weights = igraph::E(g)$weight)
  memb    <- as.integer(igraph::membership(cl_full))
  
  # vertex-count per community
  tab_size <- table(memb)
  
  # participant-weight per community (sum of mult); default 1 per vertex
  if (is.null(v_weight)) v_weight <- rep(1, igraph::gorder(g))
  w_by_comm <- tapply(v_weight, memb, sum)
  
  keep_flag <- (tab_size >= min_size)
  if (!is.null(min_weight)) {
    keep_flag <- keep_flag | (w_by_comm[names(tab_size)] >= min_weight)
  }
  keep_c   <- as.integer(names(tab_size)[keep_flag])
  keep_idx <- which(memb %in% keep_c)
  
  # Default outputs
  out_default <- list(
    membership = rep(0L, igraph::gorder(g)),
    Q = NA_real_, Q_p_upper = NA_real_, Q_p_two = NA_real_,
    S = NA_real_, S_p = NA_real_,
    z_w = NA_real_, z_deg = NA_real_, p_deg = NA_real_,
    Q_null = rep(NA_real_, n_perm), S_null = rep(NA_real_, n_perm)
  )
  
  # If no sizable communities, return early with zeros elsewhere
  if (length(keep_idx) < 2L) {
    return(out_default)
  }
  
  # Induced subgraph of kept communities
  g2 <- igraph::induced_subgraph(g, vids = keep_idx)
  # Silhouette distances on kept indices
  Dm  <- as.matrix(D)
  D2  <- stats::as.dist(Dm[keep_idx, keep_idx, drop = FALSE])
  
  # Membership remapped to 1..K on kept set, 0 otherwise
  memb2        <- as.integer(factor(memb[keep_idx]))
  memb_full    <- integer(igraph::gorder(g))
  memb_full[keep_idx] <- memb2
  K <- length(unique(memb2))
  
  # If edges vanish or a single cluster path
  if (igraph::ecount(g2) == 0L || K < 1L) {
    out_default$membership <- memb_full
    # compute S even for single cluster via try
    S_obs <- try({ ss <- cluster::silhouette(rep(1L, sum(keep_idx)), D2); mean(ss[, "sil_width"]) }, silent = TRUE)
    out_default$S <- if (inherits(S_obs, "try-error")) NA_real_ else S_obs
    return(out_default)
  }
  
  # Observed Q and S on kept subgraph
  Q_obs <- igraph::modularity(g2, memb2, weights = igraph::E(g2)$weight)
  S_obs <- try({ ss <- cluster::silhouette(memb2, D2); mean(ss[, "sil_width"]) }, silent = TRUE)
  S_obs <- if (inherits(S_obs, "try-error")) NA_real_ else S_obs
  
  # ---------------- Weight-shuffle nulls on kept subgraph ----------------
  Q_null <- rep(NA_real_, n_perm)
  S_null <- rep(NA_real_, n_perm)
  z_w    <- NA_real_
  Q_p_upper <- NA_real_
  Q_p_two   <- NA_real_
  S_p       <- NA_real_
  
  if (n_perm > 0L) {
    W2 <- igraph::E(g2)$weight
    on.exit({ igraph::E(g2)$weight <- W2 }, add = TRUE)
    
    for (b in seq_len(n_perm)) {
      igraph::E(g2)$weight <- sample(W2, length(W2), replace = FALSE)
      clb <- igraph::cluster_louvain(g2, weights = igraph::E(g2)$weight)
      mb2 <- as.integer(igraph::membership(clb))
      qv  <- try(igraph::modularity(g2, mb2, weights = igraph::E(g2)$weight), silent = TRUE)
      Q_null[b] <- if (inherits(qv, "try-error") || !is.finite(qv)) NA_real_ else qv
      sv  <- try({ ssb <- cluster::silhouette(mb2, D2); mean(ssb[, "sil_width"]) }, silent = TRUE)
      S_null[b] <- if (inherits(sv, "try-error")) NA_real_ else sv
    }
    igraph::E(g2)$weight <- W2
    
    # Empirical p-values with +1 smoothing
    B_Q  <- sum(is.finite(Q_null))
    B_S  <- sum(is.finite(S_null))
    if (B_Q > 0) {
      r_upper <- sum(Q_null >= Q_obs, na.rm = TRUE)
      r_lower <- sum(Q_null <= Q_obs, na.rm = TRUE)
      Q_p_upper <- (r_upper + 1) / (B_Q + 1)
      Q_p_two   <- 2 * min((r_upper + 1) / (B_Q + 1), (r_lower + 1) / (B_Q + 1))
      muQ <- mean(Q_null, na.rm = TRUE); sdQ <- stats::sd(Q_null, na.rm = TRUE)
      z_w <- if (is.finite(muQ) && is.finite(sdQ) && sdQ > 0) (Q_obs - muQ)/sdQ else NA_real_
    }
    if (B_S > 0) {
      rS <- sum(S_null >= S_obs, na.rm = TRUE)
      S_p <- (rS + 1) / (B_S + 1)
    }
  }
  
  # ---------------- Degree-preserving nulls (rewires) ----------------
  # Scope: "kept" uses g2; "full" uses g
  g_deg <- if (null_scope == "kept") g2 else g
  Q_deg <- modularity_degseq_null(
    g_deg, reps = deg_reps, niter_mult = 30L, keep_connected = deg_keep_connected
  )
  p_deg <- NA_real_
  z_deg <- NA_real_
  if (any(is.finite(Q_deg))) {
    muD <- mean(Q_deg, na.rm = TRUE); sdD <- stats::sd(Q_deg, na.rm = TRUE)
    p_deg <- (sum(Q_deg >= Q_obs, na.rm = TRUE) + 1) / (sum(is.finite(Q_deg)) + 1)
    z_deg <- if (is.finite(muD) && is.finite(sdD) && sdD > 0) (Q_obs - muD)/sdD else NA_real_
  }
  
  list(
    membership = memb_full,
    Q = Q_obs, Q_p_upper = Q_p_upper, Q_p_two = Q_p_two,
    S = S_obs, S_p = S_p,
    z_w = z_w, z_deg = z_deg, p_deg = p_deg,
    Q_null = Q_null, S_null = S_null,
    # Backward-compatibility aliases
    Q_p = Q_p_upper, z = z_w
  )
}

# Expand deduplicated membership back to all participants
expand_membership <- function(memb_u, sig_u, sig_all, ids_all){
  lu <- setNames(memb_u, sig_u)
  ma <- lu[as.character(sig_all)]
  data.frame(participant_id=ids_all, cluster=as.integer(ma))
}

# Pillar A: per-cluster diagnosis enrichment with guards and FDR
# Returns list(enrichment=table, majors=vector)
diagnosis_enrichment <- function(
    DX, clusters,
    alpha_fdr=0.05, min_prev_in=0.10, min_or=2.0,
    exclude=c("ANY_DX"),
    min_in_cases=5, min_total_cases=10, min_out_cases=5
){
  cols <- setdiff(names(DX), c("participant_id", exclude))
  df   <- merge(DX, clusters, by="participant_id", all.x=TRUE)
  df$cluster[is.na(df$cluster)] <- 0L
  out <- list()
  for (cl in setdiff(sort(unique(df$cluster)), 0L)){
    in_cl <- df$cluster==cl; n_in <- sum(in_cl); if (n_in < MIN_CLUSTER_SIZE) next
    rows <- lapply(cols, function(v){
      a <- sum(df[[v]][in_cl]  == 1L, na.rm=TRUE)
      b <- sum(df[[v]][!in_cl] == 1L, na.rm=TRUE)
      c <- sum(df[[v]][in_cl]  == 0L, na.rm=TRUE)
      d <- sum(df[[v]][!in_cl] == 0L, na.rm=TRUE)
      if ((a+b) < min_total_cases || a < min_in_cases) return(NULL)
      prev_in  <- a / max(1, a + c)
      prev_all <- (a + b) / max(1, a + b + c + d)
      pval <- NA_real_; OR <- if (b==0 || c==0) Inf else (a*d)/(b*c)
      if (b < min_out_cases || a == n_in) {
        alt <- if (prev_in >= prev_all) "greater" else "less"
        bt  <- try(stats::binom.test(a, n_in, p = prev_all, alternative = alt), silent = TRUE)
        if (inherits(bt,"try-error")) return(NULL)
        pval <- bt$p.value
      } else {
        ft <- try(stats::fisher.test(matrix(c(a,b,c,d), 2, byrow=TRUE)), silent=TRUE)
        if (inherits(ft,"try-error")) return(NULL)
        pval <- ft$p.value
        OR   <- unname(ft$estimate)
      }
      data.frame(cluster=cl, diagnosis=v, a=a,b=b,c=c,d=d,
                 prev_in=prev_in, OR=OR, p=pval)
    })
    rows <- Filter(Negate(is.null), rows)
    if (length(rows)) out[[length(out)+1L]] <- do.call(rbind, rows)
  }
  if (!length(out)) return(list(enrichment=NULL, majors=character(0)))
  res <- do.call(rbind, out); res$FDR <- p.adjust(res$p, "BH")
  majors <- res$diagnosis[ res$FDR<=alpha_fdr & res$prev_in>=min_prev_in & res$OR>=min_or ]
  list(enrichment = res[order(res$cluster, res$FDR, -res$OR),], majors=unique(majors))
}

#' Degree-sequence modularity null via degree-preserving rewires
#' - Works on a simple, undirected graph.
#' - Optionally enforces connected rewired graphs (tries up to 20 times).
#' - Shuffles edge weights each draw to decouple topology from weights.
#' - Runs Louvain and returns the vector of modularity scores Q.
modularity_degseq_null <- function(
    g, reps = 500L, niter_mult = 30L, keep_connected = TRUE
) {
  stopifnot(igraph::is_simple(g), !igraph::is_directed(g))
  m <- igraph::ecount(g)
  W <- igraph::E(g)$weight
  Q <- rep(NA_real_, reps)
  for (i in seq_len(reps)) {
    tries <- 0L
    repeat {
      g0 <- igraph::rewire(g, with = igraph::keeping_degseq(niter = niter_mult * max(1L, m)))
      tries <- tries + 1L
      if (!keep_connected || igraph::components(g0)$no == 1L || tries > 20L) break
    }
    if (igraph::ecount(g0) == 0L) next
    igraph::E(g0)$weight <- sample(W, igraph::ecount(g0), replace = TRUE)
    cl0 <- igraph::cluster_louvain(g0, weights = igraph::E(g0)$weight)
    Q[i] <- igraph::modularity(g0, igraph::membership(cl0), weights = igraph::E(g0)$weight)
  }
  Q
}

#' Column-shuffle null: preserve per-diagnosis prevalence; re-run clustering
null_column_shuffle <- function(DX, K_KNN, UNION_KNN, MIN_CLUSTER_SIZE, reps=300L){
  cols <- setdiff(names(DX), "participant_id")
  out  <- data.frame(Q=rep(NA_real_, reps), S=NA_real_, ID=NA_real_)
  for (r in seq_len(reps)){
    DXs <- DX; for (v in cols) DXs[[v]] <- sample(DXs[[v]])
    dd  <- dedup_dx(DXs); DXu <- dd$DXu
    keep <- rowSums(DXu[, setdiff(names(DXu), c("participant_id","mult")), drop=FALSE], na.rm=TRUE) > 0
    DXu_id <- DXu[keep, , drop=FALSE]; if (nrow(DXu_id) < 5) next
    D  <- gower_dist_dx(DXu_id); if (is.null(D)) next
    g  <- knn_graph_from_dist(D, k=K_KNN, union=UNION_KNN, mult=DXu_id$mult)
    lv0 <- louvain_with_stats(g, D, n_perm=0, min_size=MIN_CLUSTER_SIZE, seed=42)
    out$Q[r]  <- lv0$Q; out$S[r] <- lv0$S
    out$ID[r] <- suppressWarnings(twonn_id_from_dist(D))
  }
  out
}

#' Pillar B: label localisation metrics with label-shuffle p-values
label_localization_table <- function(g, DXu_id, B = 1000,
                                     n_pos_min = 10, n_neg_min = 10,
                                     counts_all = NULL,
                                     exclude = c("ANY_DX")){
  el <- igraph::as_edgelist(g, names = FALSE)
  cols <- setdiff(names(DXu_id), c("participant_id","mult", exclude))
  out <- lapply(cols, function(v){
    # Use participant-level counts if available (so NODIAG is not dropped)
    if (!is.null(counts_all)) {
      row <- counts_all[counts_all$dx == v, , drop = FALSE]
      pos_tot <- if (nrow(row)) row$n1[1] else sum(DXu_id[[v]] == 1L, na.rm = TRUE)
      neg_tot <- if (nrow(row)) row$n0[1] else sum(DXu_id[[v]] == 0L, na.rm = TRUE)
    } else {
      pos_tot <- sum(DXu_id[[v]] == 1L, na.rm = TRUE)
      neg_tot <- sum(DXu_id[[v]] == 0L, na.rm = TRUE)
    }
    if (pos_tot < n_pos_min || neg_tot < n_neg_min) return(NULL)
    z <- as.integer(DXu_id[[v]] == 1L); z[is.na(z)] <- 0L
    # assortativity on current graph
    K <- 2L; zi <- z[el[,1]]; zj <- z[el[,2]]
    tab <- matrix(0, K, K)
    for (i in seq_len(nrow(el))) { a <- zi[i]+1L; b <- zj[i]+1L; tab[a,b] <- tab[a,b]+1; tab[b,a] <- tab[b,a]+1 }
    e <- tab/sum(tab); a <- rowSums(e); b <- colSums(e)
    r_obs <- (sum(diag(e)) - sum(a*b)) / (1 - sum(a*b))
    r_null <- replicate(B, {
      zsh <- sample(z)
      zsi <- zsh[el[,1]]; zsj <- zsh[el[,2]]
      tb0 <- matrix(0, K, K)
      for (i in seq_len(nrow(el))) { a0 <- zsi[i]+1L; b0 <- zsj[i]+1L; tb0[a0,b0] <- tb0[a0,b0]+1; tb0[b0,a0] <- tb0[b0,a0]+1 }
      e0 <- tb0/sum(tb0); a0 <- rowSums(e0); b0 <- colSums(e0)
      (sum(diag(e0)) - sum(a0*b0)) / (1 - sum(a0*b0))
    })
    p_r <- (sum(r_null >= r_obs) + 1) / (B + 1)
    # kNN purity (positive-neighbour fraction)
    nbrs <- igraph::neighborhood(g, order=1)
    safe_mean <- function(x) if (length(x)) mean(x) else 0
    pur_obs <- mean(sapply(which(z==1L), function(i) safe_mean(z[ setdiff(nbrs[[i]], i) ])))
    pur_null <- replicate(B, {
      zz <- sample(z)
      mean(sapply(which(zz==1L), function(i) safe_mean(zz[ setdiff(nbrs[[i]], i) ])))
    })
    p_pur <- (sum(pur_null >= pur_obs) + 1) / (B + 1)
    data.frame(dx=v, assort_r=r_obs, assort_p=p_r, knn_purity=pur_obs, knn_p=p_pur)
  })
  do.call(rbind, Filter(Negate(is.null), out))
}

#' Pillar C: one-vs-rest AUC using kNN neighbour-label score with weights
auc_one_vs_rest_knn_weighted <- function(DXu_id, target, k = 10,
                                         pos_min = 10, neg_min = 10){
  stopifnot("mult" %in% names(DXu_id))
  cols0 <- setdiff(names(DXu_id), c("participant_id","mult", target))
  v01 <- vapply(cols0, function(nm){
    v <- DXu_id[[nm]]; s <- sum(v==1L, na.rm=TRUE); !(s==0L || s==sum(is.finite(v)))
  }, logical(1))
  cols <- cols0[v01]; if (!length(cols)) return(NA_real_)
  D  <- cluster::daisy(DXu_id[, cols, drop=FALSE], metric="gower",
                       type=list(asymm=cols), stand=FALSE, weights=rep(1,length(cols)))
  M  <- as.matrix(D); diag(M) <- Inf
  y   <- as.integer(DXu_id[[target]] == 1L); y[is.na(y)] <- 0L
  w   <- as.numeric(DXu_id$mult); w[!is.finite(w) | w <= 0] <- 1
  if (sum(w[y==1]) < pos_min || sum(w[y==0]) < neg_min) return(NA_real_)
  kth <- function(r, k){ ok <- which(is.finite(r)); if (!length(ok)) integer(0) else ok[order(r[ok])[seq_len(min(k, length(ok)))]] }
  nn <- lapply(seq_len(nrow(M)), function(i) setdiff(kth(M[i,], k), i))
  s  <- vapply(seq_len(nrow(M)), function(i){
    js <- nn[[i]]; if (!length(js)) return(NA_real_)
    sum(y[js] * w[js]) / sum(w[js])
  }, numeric(1))
  ok <- is.finite(s); s <- s[ok]; yy <- y[ok]; ww <- w[ok]
  if (length(unique(yy)) < 2L) return(1.0)
  df <- data.frame(s=s, y=yy, w=ww)
  df <- aggregate(w ~ s + y, data=df, sum)
  wp <- tapply(ifelse(df$y==1, df$w, 0), df$s, sum, default=0)
  wn <- tapply(ifelse(df$y==0, df$w, 0), df$s, sum, default=0)
  scores <- sort(as.numeric(names(wp)))
  wp <- wp[as.character(scores)]; wn <- wn[as.character(scores)]
  Wp <- sum(wp); Wn <- sum(wn)
  if (Wp==0 || Wn==0) return(1.0)
  cwn <- cumsum(wn)  # negatives strictly below each score
  auc_num <- sum(wp * (cwn + 0.5*wn))   # ties get 0.5
  auc <- auc_num / (Wp * Wn)
  if (auc < 0.5) auc <- 1 - auc
  auc
}

#' ΔQ when zeroing a single diagnosis (importance by ablation)
deltaQ_per_dx <- function(DX, lv_ref_Q, K_KNN, union=UNION_KNN, MIN_CLUSTER_SIZE=10){
  dx_cols <- setdiff(names(DX), c("participant_id","ANY_DX"))
  out <- lapply(dx_cols, function(v){
    DXm <- DX; DXm[[v]] <- 0L
    dd  <- dedup_dx(DXm); DXu <- dd$DXu
    keep <- rowSums(DXu[, setdiff(names(DXu), c("participant_id","mult")), drop=FALSE], na.rm=TRUE) > 0
    DXu  <- DXu[keep, , drop=FALSE]; if (nrow(DXu) < 5) return(NULL)
    Dm  <- gower_dist_dx(DXu); if (is.null(Dm)) return(NULL)
    g   <- try(knn_graph_from_dist(Dm, k=K_KNN, union=union, mult=DXu$mult), silent=TRUE)
    if (inherits(g,"try-error") || igraph::ecount(g)==0) return(NULL)
    lv0 <- louvain_with_stats(g, Dm, n_perm=0, min_size=MIN_CLUSTER_SIZE, seed=42)
    data.frame(dx=v, Q_drop = lv_ref_Q - lv0$Q)
  })
  do.call(rbind, Filter(Negate(is.null), out))
}

# ============================== Main pipeline =================================
DX <- load_dx_wide(DX_CSV_PATH, include_nodiag=INCLUDE_NODIAG,
                   min_prev=DX_MIN_PREV, max_prev=DX_MAX_PREV)

dd   <- dedup_dx(DX)
DXu  <- dd$DXu
sigU <- apply(DXu[, setdiff(names(DXu), c("participant_id","mult")), drop=FALSE], 1, paste0, collapse="")
sigA <- dd$sig

counts_all <- build_counts(DX)

# keep unique profiles with any Dx present
colsU    <- setdiff(names(DXu), c("participant_id","mult"))
keep_idU <- rowSums(DXu[, colsU, drop=FALSE], na.rm=TRUE) > 0L
DXu_id   <- DXu[keep_idU, , drop=FALSE]
sigU_id  <- sigU[keep_idU]
multU    <- as.numeric(DXu_id$mult); multU[!is.finite(multU) | multU <= 0] <- 1

cat(sprintf("[DX] unique profiles: %d / %d (%.1f%% duplicates)\n",
            nrow(DXu), nrow(DX), 100*(1 - nrow(DXu)/nrow(DX))))
cat(sprintf("[DX] unique non-zero profiles used: %d\n", nrow(DXu_id)))

# distances on unique non-zero profiles
D_dx <- gower_dist_dx(DXu_id)
if (is.null(D_dx)) stop("DX after dedup has <2 varying diagnosis columns.")
Dm <- as.matrix(D_dx); diag(Dm) <- Inf

# intrinsic dimension (all and core band)
ID_twonn_all   <- suppressWarnings(twonn_id_from_dist(D_dx))
idx_core       <- core_band_idx(D_dx, k = max(5, min(20, K_KNN)), band = c(0.15, 0.85))
ID_twonn_core  <- if (length(idx_core) >= 5) suppressWarnings(
  twonn_id_from_dist(stats::as.dist(Dm[idx_core, idx_core, drop=FALSE]))
) else NA_real_
ID_lbmle_core  <- if (length(idx_core) >= 5) suppressWarnings(
  lb_mle_id(Dm[idx_core, idx_core, drop=FALSE], 5, 15)
) else NA_real_
cat(sprintf("[ID] TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f | n_core=%s\n",
            ID_twonn_all, ID_twonn_core, ID_lbmle_core,
            if (length(idx_core)) length(idx_core) else "NA"))

# kNN graph + Louvain clustering
knn_reciprocity <- function(M, k){
  n <- nrow(M)
  get_nbrs <- function(i){
    di <- M[i,]; ok <- which(is.finite(di))
    if (!length(ok)) integer(0) else ok[order(di[ok])[seq_len(min(k, length(ok)))]]
  }
  nbrs <- lapply(seq_len(n), get_nbrs)
  rec_cnt <- 0L; dir_cnt <- 0L
  for (i in seq_len(n)){
    for (j in nbrs[[i]]) {
      dir_cnt <- dir_cnt + 1L
      if (i %in% nbrs[[j]]) rec_cnt <- rec_cnt + 1L
    }
  }
  c(reciprocal = rec_cnt, directed = dir_cnt, ratio = rec_cnt / max(1, dir_cnt))
}

g_dx <- knn_graph_from_dist(D_dx, k = K_KNN, union = UNION_KNN, mult = multU, local_scale = TRUE)
report_knn_graph(g_dx)
M <- as.matrix(D_dx); diag(M) <- Inf
print(knn_reciprocity(M, K_KNN))

lv <- louvain_with_stats(
  g_dx, D_dx, n_perm = N_PERM,
  min_size = MIN_CLUSTER_SIZE, min_weight = MIN_CLUSTER_WEIGHT,
  seed = 42, deg_reps = N_PERM, null_scope = "kept",
  v_weight = multU
)

K_kept <- length(setdiff(unique(lv$membership), 0L))
names(lv$membership) <- rownames(as.matrix(D_dx))

cat(sprintf("[Clustering] Louvain Q=%.3f | p_deg=%.5f | p_two_w=%.5f | S=%.3f (p=%.5f) | K=%d | kept=%d\n",
            lv$Q, lv$p_deg, lv$Q_p_two, lv$S, lv$S_p, K_KNN, K_kept))

# degree-preserving null (after baseline)
Q_null_deg <- modularity_degseq_null(g_dx, reps = N_PERM)
mu <- mean(Q_null_deg, na.rm=TRUE)
sd0 <- sd(Q_null_deg, na.rm=TRUE)
p_emp <- (sum(Q_null_deg >= lv$Q, na.rm=TRUE) + 1) / (sum(is.finite(Q_null_deg)) + 1)
z_deg <- (lv$Q - mu) / sd0
readr::write_csv(
  data.frame(Q_null_deg=Q_null_deg, Q_obs=lv$Q, Q_p_deg=p_emp),
  "modularity_degree_null.csv"
)
message(sprintf(
  "[Degree-null] Q=%.3f | mu=%.3f sigma=%.3f | z=%.2f | p(emp)=%.3f",
  lv$Q, mu, sd0, z_deg, p_emp
))

# column-shuffle nulls (prevalence-preserving)
ns <- null_column_shuffle(DX, K_KNN=K_KNN, UNION_KNN=UNION_KNN,
                          MIN_CLUSTER_SIZE=MIN_CLUSTER_SIZE, reps=300)
Q_p_col  <- mean(ns$Q  >= lv$Q,         na.rm=TRUE)
S_p_col  <- mean(ns$S  >= lv$S,         na.rm=TRUE)
ID_p_col <- mean(ns$ID >= ID_twonn_all, na.rm=TRUE)
ID_p_high <- mean(ns$ID >= ID_twonn_all, na.rm = TRUE)
ID_p_low  <- mean(ns$ID <= ID_twonn_all, na.rm = TRUE)
ID_p_two  <- 2 * min(ID_p_high, ID_p_low)
message(sprintf("[Column-shuffle null] p(Q)=%.3f, p(S)=%.3f, p(ID)=%.3f",
                Q_p_col, S_p_col, ID_p_col))
message(sprintf("[Column-shuffle null] p_high(ID)=%.3f, p_low(ID)=%.3f, p_two(ID)=%.3f",
                ID_p_high, ID_p_low, ID_p_two))

# bootstrap stability (ARI vs baseline)
boot_ari <- rep(NA_real_, N_BOOT)
p_boot <- multU / sum(multU)
for (b in seq_len(N_BOOT)){
  take <- sample(seq_len(nrow(DXu_id)), replace=TRUE, prob=p_boot)
  take <- sort(unique(take)); if (length(take) < 5) next
  D_b <- stats::as.dist(as.matrix(D_dx)[take, take, drop=FALSE])
  g_b <- try(knn_graph_from_dist(D_b, k=K_KNN, union=UNION_KNN, mult=multU[take]), silent=TRUE)
  if (inherits(g_b,"try-error") || igraph::ecount(g_b) == 0L) next
  clb <- igraph::cluster_louvain(g_b, weights=igraph::E(g_b)$weight)
  mb  <- as.integer(igraph::membership(clb))
  ref <- lv$membership[take]
  if (length(unique(ref)) < 2L || length(unique(mb)) < 2L) next
  ref <- setNames(lv$membership[take], as.character(take))
  mb  <- setNames(mb,                  as.character(take))
  boot_ari[b] <- align_ari(ref, mb)
}
ARI_med <- stats::median(boot_ari, na.rm=TRUE)
ARI_iqr <- stats::IQR(boot_ari, na.rm=TRUE)
cat(sprintf("[Stability] ARI_median=%.3f | ARI_IQR=%.3f\n", ARI_med, ARI_iqr))
readr::write_csv(data.frame(boot=seq_len(N_BOOT), ARI=boot_ari),
                 "cluster_bootstrap_ari.csv")

# expand cluster labels to all participants
clusters_all <- expand_membership(lv$membership, sigU_id, sigA, DX$participant_id)
readr::write_csv(clusters_all, "cluster_membership_all_participants.csv")

# ===================== NEW: Over-representation (aligned) =====================

# Guarded binariser (only define if absent)
if (!exists("binarise_dx")) {
  binarise_dx <- function(M, cutoff = 0.5){
    A <- as.data.frame(M, check.names = FALSE)
    for (j in names(A)){
      v <- suppressWarnings(as.numeric(A[[j]]))
      if (all(v %in% c(0,1,NA))) {
        A[[j]] <- as.integer(v > 0)
      } else {
        rng <- range(v, na.rm = TRUE)
        if (rng[1] >= 0 && rng[2] <= 1) A[[j]] <- as.integer(v >= cutoff) else A[[j]] <- as.integer(v > 0)
      }
    }
    A
  }
}

# Canonical aligner (names/rownames) — tight, loud on mismatch
# Drop-in replacement: permissive alignment by subsetting to `ids`
align_to_ids <- function(obj, ids){
  if (is.null(obj)) return(obj)
  
  # Data frame path with participant_id -> set rownames then strip the column
  if (is.data.frame(obj) && "participant_id" %in% names(obj)) {
    rn <- as.character(obj$participant_id)
    stopifnot(!anyDuplicated(rn), all(nzchar(rn)))
    rownames(obj) <- rn
    obj <- obj[, setdiff(names(obj), "participant_id"), drop = FALSE]
  }
  
  # Matrix/data.frame with rownames
  if (!is.null(rownames(obj))) {
    rn <- rownames(obj)
    missing <- setdiff(ids, rn)
    if (length(missing)) {
      stop(sprintf("[align] %d ids missing from object (first 10): %s",
                   length(missing), paste(head(missing, 10), collapse = ", ")))
    }
    return(obj[ids, , drop = FALSE])  # allow extras in obj; subset to ids
  }
  
  # Named vector
  if (!is.null(names(obj))) {
    nm <- names(obj)
    missing <- setdiff(ids, nm)
    if (length(missing)) {
      stop(sprintf("[align] %d ids missing from named vector (first 10): %s",
                   length(missing), paste(head(missing, 10), collapse = ", ")))
    }
    return(obj[ids])
  }
  
  stop("[align] Object has neither rownames nor names; cannot align.")
}

compute_overrep <- function(DX, clusters_all,
                            min_in = NIN_MIN %||% 5L,
                            min_out = NOUT_MIN %||% 5L,
                            cutoff = 0.5){
  stopifnot(all(c("participant_id","cluster") %in% names(clusters_all)))
  # Anchor on participant_id in DX
  if (is.null(rownames(DX))) {
    stopifnot("participant_id" %in% names(DX))
    rownames(DX) <- as.character(DX$participant_id)
  }
  ids_ref <- intersect(rownames(DX), as.character(clusters_all$participant_id))
  if (!length(ids_ref)) stop("[overrep] No overlap between DX and clusters_all.")
  DX_pts <- align_to_ids(DX, ids_ref)
  DX_pts <- DX_pts[, setdiff(names(DX_pts), "participant_id"), drop = FALSE]
  cl_map <- setNames(clusters_all$cluster, clusters_all$participant_id)
  cl_vec <- as.integer(cl_map[ids_ref]); cl_vec[is.na(cl_vec)] <- 0L
  
  DXb <- binarise_dx(DX_pts, cutoff = cutoff)
  cls <- sort(setdiff(unique(cl_vec), 0L))
  out <- list()
  for (cid in cls){
    inidx  <- which(cl_vec == cid)
    outidx <- which(cl_vec != cid)
    if (length(inidx) < min_in || length(outidx) < min_out) next
    n_in <- length(inidx); n_out <- length(outidx)
    for (dx in names(DXb)){
      v <- DXb[[dx]]
      a <- sum(v[inidx]  == 1L, na.rm = TRUE)  # pos in
      b <- n_in  - a                           # neg in
      c <- sum(v[outidx] == 1L, na.rm = TRUE)  # pos out
      d <- n_out - c                           # neg out
      if ((a + c) == 0 || (b + d) == 0) next
      in_prev  <- a / n_in
      out_prev <- c / n_out
      lift     <- if (out_prev > 0) in_prev / out_prev else Inf
      log2_lift <- log2(pmax(lift, 1e-12))
      p <- tryCatch(suppressWarnings(fisher.test(matrix(c(a,b,c,d), 2, 2))$p.value),
                    error = function(e) NA_real_)
      out[[length(out)+1L]] <- data.frame(
        cluster = cid, dx = dx,
        n_in = n_in, n_out = n_out,
        pos_in = a, pos_out = c,
        in_prev = in_prev, out_prev = out_prev,
        lift = lift, log2_lift = log2_lift, p = p,
        stringsAsFactors = FALSE
      )
    }
  }
  T <- dplyr::bind_rows(out)
  if (!nrow(T)) return(T)
  T$q <- p.adjust(T$p, method = "BH")
  T$star <- dplyr::case_when(
    !is.finite(T$p) ~ "",
    T$q < 0.001 ~ "***",
    T$q < 0.01  ~ "**",
    T$q < 0.05  ~ "*",
    TRUE ~ "ns"
  )
  T[order(T$cluster, -T$log2_lift, T$q), ]
}

# Heatmap and slope detail (identical look & feel to your earlier utility)
plot_overrep_heatmap <- function(tab, top_k_per_cluster = 12L, cap = 2){
  req <- c("cluster","dx","log2_lift","q","star"); stopifnot(all(req %in% names(tab)))
  TOP <- tab %>%
    dplyr::group_by(cluster) %>%
    dplyr::arrange(dplyr::desc(log2_lift), .by_group = TRUE) %>%
    dplyr::filter(is.finite(log2_lift)) %>%
    dplyr::slice_head(n = top_k_per_cluster) %>%
    dplyr::ungroup()
  keep_dx <- unique(TOP$dx)
  H <- tab %>% dplyr::filter(dx %in% keep_dx) %>%
    dplyr::mutate(cluster = factor(paste0("C", cluster),
                                   levels = paste0("C", sort(unique(tab$cluster))))) %>%
    dplyr::mutate(dx = factor(dx, levels = rev(sort(unique(keep_dx)))))
  H$val <- pmin(pmax(H$log2_lift, -cap), cap)
  ggplot2::ggplot(H, ggplot2::aes(x = cluster, y = dx, fill = val)) +
    ggplot2::geom_tile() +
    ggplot2::geom_text(ggplot2::aes(label = ifelse(q < 0.05, star, "")),
                       size = 3, fontface = "bold") +
    ggplot2::scale_fill_gradient2(name = "log2(lift)", limits = c(-cap, cap)) +
    ggplot2::labs(x = NULL, y = NULL, title = "Over-representation by cluster") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(panel.grid = element_blank(),
                   axis.text.x = ggplot2::element_text(size = 10),
                   axis.text.y = ggplot2::element_text(size = 9))
}

wilson_ci <- function(k, n, z = 1.96){
  k <- as.numeric(k); n <- as.numeric(n)
  # proportions
  p <- ifelse(n > 0, k / n, NA_real_)
  # Wilson pieces
  denom   <- ifelse(n > 0, 1 + z^2 / n, NA_real_)
  centre  <- ifelse(n > 0, (p + z^2 / (2 * n)) / denom, NA_real_)
  halfwid <- ifelse(n > 0, z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom, NA_real_)
  lo <- pmax(0, centre - halfwid)
  hi <- pmin(1, centre + halfwid)
  cbind(lo = lo, hi = hi)          # <- guaranteed 2-column matrix
}

plot_cluster_detail <- function(cid, tab, top_n = 15L, alpha = 0.05){
  T <- tab %>%
    dplyr::filter(cluster == cid) %>%
    dplyr::mutate(sig = is.finite(q) & q < alpha) %>%
    dplyr::arrange(dplyr::desc(log2_lift))
  if (!nrow(T)) return(ggplot2::ggplot() + ggplot2::theme_void())
  
  T$dx <- factor(T$dx, levels = rev(head(T$dx, top_n)))
  
  ci_in  <- wilson_ci(T$pos_in,  T$n_in)
  ci_out <- wilson_ci(T$pos_out, T$n_out)
  
  # Long (2 rows per dx) — for points + error bars
  D <- tibble::tibble(
    dx    = rep(T$dx, each = 2),
    group = rep(c("in-cluster","out-of-cluster"), times = nrow(T)),
    prev  = c(T$in_prev, T$out_prev),
    lo    = c(ci_in[,"lo"],  ci_out[,"lo"]),
    hi    = c(ci_in[,"hi"],  ci_out[,"hi"])
  ) %>%
    dplyr::filter(dx %in% levels(T$dx)) %>%
    dplyr::group_by(dx) %>% dplyr::slice(1:2) %>% dplyr::ungroup()
  
  # Wide (1 row per dx) — for the segment
  S <- D %>%
    tidyr::pivot_wider(id_cols = dx, names_from = group, values_from = prev) %>%
    dplyr::rename(prev_in  = `in-cluster`,
                  prev_out = `out-of-cluster`)
  
  ggplot2::ggplot(D, ggplot2::aes(y = dx)) +
    ggplot2::geom_segment(
      data = S,
      ggplot2::aes(y = dx, yend = dx, x = prev_out, xend = prev_in),
      linewidth = 0.5, alpha = 0.6, colour = "grey40", inherit.aes = FALSE
    ) +
    ggplot2::geom_errorbarh(ggplot2::aes(xmin = lo, xmax = hi, colour = group),
                            height = 0, alpha = 0.8) +
    ggplot2::geom_point(ggplot2::aes(x = prev, colour = group), size = 2.2) +
    ggplot2::scale_x_continuous(labels = scales::percent, limits = c(0, 1)) +
    ggplot2::scale_colour_manual(values = c("out-of-cluster" = "#606060",
                                            "in-cluster"   = "#1b7cff"), name = NULL) +
    ggplot2::labs(x = "Prevalence", y = NULL,
                  title = paste0("Cluster C", cid, ": Dx enrichment (top ", top_n, ")")) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(legend.position = "top", panel.grid.minor = ggplot2::element_blank())
}

# ---- Compute + save ----
tab_sig2 <- compute_overrep(DX, clusters_all,
                            min_in = NIN_MIN, min_out = NOUT_MIN, cutoff = 0.5)
if (nrow(tab_sig2)) {
  readr::write_csv(tab_sig2, "dx_overrep_full.csv")
  
  p_heat <- plot_overrep_heatmap(tab_sig2, top_k_per_cluster = 12, cap = 2)
  print(p_heat)
  ggplot2::ggsave("FIG_dx_overrep_heatmap.png", p_heat, width = 8.5, height = 7, dpi = 300, bg = "white")
  
  # pick largest kept cluster for the detail panel
  cl_sizes <- sort(table(clusters_all$cluster), decreasing = TRUE)
  cid <- as.integer(names(cl_sizes)[cl_sizes == max(cl_sizes) & names(cl_sizes) != "0"][1])
  if (is.finite(cid)) {
    p_det <- plot_cluster_detail(cid, tab_sig2, top_n = 15, alpha = 0.05)
    print(p_det)
    ggplot2::ggsave(sprintf("FIG_dx_overrep_C%d_detail.png", cid), p_det, width = 7.5, height = 6.5,
                    dpi = 300, bg = "white")
  }
  cat(sprintf("[overrep] wrote dx_overrep_full.csv; figures FIG_dx_overrep_heatmap.png and FIG_dx_overrep_C%d_detail.png\n",
              if (is.finite(cid)) cid else -1L))
} else {
  cat("[overrep] No cells passed basic counts; skipping figures.\n")
}

# ---------------- Pillar A: cluster enrichment (Fisher + guards) ---------------
enr <- diagnosis_enrichment(
  DX, clusters_all,
  alpha_fdr = ALPHA_FDR, min_prev_in = MIN_PREV_IN_CL, min_or = MIN_OR,
  exclude = c("ANY_DX"),
  min_in_cases = MIN_CASES_TOTAL, min_total_cases = MIN_CASES_TOTAL, min_out_cases = MIN_CASES_TOTAL
)
if (!is.null(enr$enrichment)) {
  readr::write_csv(enr$enrichment, "cluster_diagnosis_enrichment.csv")
  readr::write_csv(data.frame(major_diagnosis = enr$majors),
                   "selected_major_diagnoses.csv")
  cat(sprintf("[Majors A] %d diagnoses by enrichment\n", length(enr$majors)))
} else {
  cat("[Majors A] none\n")
}

# --- Columns to plot in the heatmaps ---
# Start with majors; optionally prepend NODIAG if present/desired.
dx_core <- majors_union
if (INCLUDE_NODIAG && "NODIAG" %in% names(DX)) {
  dx_core <- c("NODIAG", dx_core)
}

# Keep only columns that actually exist in DX and are not meta
dx_for_heatmap <- intersect(
  dx_core,
  setdiff(names(DX), c("participant_id", "ANY_DX"))
)

if (!length(dx_for_heatmap)) {
  stop("dx_for_heatmap ended up empty — check majors_union / DX column names.")
}

# (optional) print for sanity
cat("[heatmap] dx_for_heatmap =", paste(dx_for_heatmap, collapse = ", "), "\n")

# ---------------- Pillar B: localisation (assortativity / kNN purity) ----------
loc_tab <- label_localization_table(
  g_dx, DXu_id, B = 1000,
  n_pos_min = MIN_CASES_TOTAL, n_neg_min = MIN_CASES_TOTAL,
  counts_all = counts_all,
  exclude = c("ANY_DX")
)
maj_B <- subset(loc_tab, pmin(assort_p, knn_p) <= ALPHA_LOCALIZE)$dx
readr::write_csv(loc_tab, "dx_label_localization.csv")

# ---------------- Pillar C: predictability (kNN-score AUC) ---------------------
dx_cols <- setdiff(names(DXu_id), c("participant_id","mult","ANY_DX"))
auc_tab <- data.frame(
  dx  = dx_cols,
  AUC = vapply(dx_cols, function(v) {
    tryCatch(auc_one_vs_rest_knn_weighted(DXu_id, v, k = 10,
                                          pos_min = MIN_CASES_TOTAL, neg_min = MIN_CASES_TOTAL),
             error = function(e) NA_real_)
  }, numeric(1))
)
maj_C <- subset(auc_tab, is.finite(AUC) & AUC >= AUC_MIN)$dx
readr::write_csv(auc_tab, "dx_predictability_auc_knn.csv")

# -------------------- Assemble majors: A ∪ B ∪ C with final guards -------------
cand <- unique(c(enr$majors, maj_B, maj_C))
maj_prev_pass <- with(counts_all, dx[prev >= PREV_MIN | n1 >= NCASE_MIN])
majors_union <- sort(setdiff(intersect(cand, maj_prev_pass), c("ANY_DX")))
readr::write_csv(data.frame(major_dx = majors_union),
                 "selected_major_diagnoses_union.csv")
cat(sprintf("[Majors | union] %d diagnoses selected (A ∪ B ∪ C)\n",
            length(majors_union)))

# ---------------------- ΔQ per Dx (importance / ablation) ----------------------
dq <- deltaQ_per_dx(DX, lv$Q, K_KNN, UNION_KNN, MIN_CLUSTER_SIZE)
if (!is.null(dq) && nrow(dq)) readr::write_csv(dq, "dx_deltaQ.csv")

# -------------------------------- Visualisations -------------------------------
# 2D MDS quick plot
fit2 <- stats::cmdscale(D_dx, k=2, eig=TRUE, add=TRUE)
XY   <- fit2$points; colnames(XY) <- c("MDS1","MDS2")
eig  <- fit2$eig; pos <- eig[eig>0]
frac <- if (length(pos)) sum(pmax(0, eig[1:2]))/sum(pos) else NA_real_
r2   <- suppressWarnings(stats::cor(as.numeric(D_dx), as.numeric(stats::dist(XY)))^2)
dfp  <- data.frame(MDS1=XY[,1], MDS2=XY[,2], cluster=factor(lv$membership))
dfp <- subset(dfp, cluster != 0)
p2   <- ggplot2::ggplot(dfp, ggplot2::aes(MDS1, MDS2, color=cluster)) +
  ggplot2::geom_point(size=1.9, alpha=0.95) + ggplot2::coord_equal() +
  ggplot2::labs(title=sprintf("DX MDS2 - var=%.2f, r2(dist)=%.2f, Q=%.3f, sil=%.3f",
                              frac, r2, lv$Q, lv$S)) +
  ggplot2::theme_minimal(12)
print(p2)
ggplot2::ggsave("FIG_dxspace_clusters_mds2.png", p2, width=PLOT_WIDTH, height=PLOT_HEIGHT, dpi=PLOT_DPI)

# # 3D MDS interactive (saved as HTML)
# mds3_plot_interactive <- function(D, memb, mult=NULL,
#                                   file="FIG_dxspace_mds3_interactive.html"){
#   if (!requireNamespace("plotly", quietly=TRUE) ||
#       !requireNamespace("htmlwidgets", quietly=TRUE)) return(invisible(NULL))
#   fit <- cmdscale(D, k=3, eig=TRUE, add=TRUE)
#   XY  <- as.data.frame(fit$points); names(XY) <- paste0("MDS",1:3)
#   eig <- fit$eig; pos <- eig[eig>0]
#   frac_axes <- if (length(pos)) pmax(0, eig[1:3]) / sum(pos) else rep(NA_real_, 3)
#   r2 <- suppressWarnings(cor(as.numeric(D), as.numeric(dist(XY)))^2)
#   df <- XY; df$cluster <- factor(memb); df$mult <- if (is.null(mult)) 1 else as.numeric(mult)
#   df$size <- 6 + 2*log1p(df$mult)
#   ttl <- sprintf("DX MDS3 | var(1:3)=%.0f%% | r2(dist)=%.0f%%",
#                  100*sum(frac_axes, na.rm=TRUE), 100*r2)
#   p <- plotly::plot_ly(df, x=~MDS1, y=~MDS2, z=~MDS3, color=~cluster,
#                        type="scatter3d", mode="markers") |>
#     plotly::add_markers(size=~size, opacity=0.9) |>
#     plotly::layout(title=ttl,
#                    scene=list(
#                      xaxis=list(title=sprintf("MDS1 (%.0f%%)",100*frac_axes[1])),
#                      yaxis=list(title=sprintf("MDS2 (%.0f%%)",100*frac_axes[2])),
#                      zaxis=list(title=sprintf("MDS3 (%.0f%%)",100*frac_axes[3]))
#                    ))
#   htmlwidgets::saveWidget(plotly::as_widget(p), file, selfcontained=TRUE)
#   p
# }
# mds3_plot_interactive(D_dx, lv$membership, mult=multU,
#                       file="FIG_dxspace_mds3_interactive.html")

# Cluster × major Dx heatmap (prevalence in-cluster)
if (length(majors_union)) {
  dfx <- merge(DX, clusters_all, by="participant_id", all.x=TRUE)
  dfx$cluster[is.na(dfx$cluster)] <- 0L
  cl_levels <- setdiff(sort(unique(dfx$cluster)), 0L)
  long_prev <- do.call(
    rbind,
    lapply(cl_levels, function(cl){
      in_cl <- dfx$cluster == cl
      data.frame(
        cluster = cl,
        dx = majors_union,
        prev = vapply(majors_union, function(v){
          a  <- sum(dfx[[v]][in_cl] == 1, na.rm=TRUE)
          c0 <- sum(dfx[[v]][in_cl] == 0, na.rm=TRUE)
          a / max(1, a + c0)
        }, numeric(1))
      )
    })
  )
  # --- map codes -> pretty labels and enforce desired order ---
  code_to_label <- c(
    NODIAG                    = "No Diagnosis on Axis I",
    SCID.DIAG.ADHD            = "Attention-Deficit/Hyperactivity Disorder",
    SCID.DIAG.BipolarI        = "Bipolar I Disorder",
    SCID.DIAG.DepressNOS      = "Depressive Disorder NOS",
    SCID.DIAG.MDD             = "Major Depressive Disorder",
    SCID.DIAG.Schizophrenia   = "Schizophrenia",
    SCID.DIAG.Schizoaffective = "Schizoaffective Disorder",
    SCID.DIAG.AlcAbuse        = "Alcohol Abuse",
    SCID.DIAG.AlcDepend       = "Alcohol Dependence",
    SCID.DIAG.AmpAbuse        = "Amphetamine Abuse",
    SCID.DIAG.AmpDepend       = "Amphetamine Dependence",
    SCID.DIAG.CanAbuse        = "Cannabis Abuse",
    SCID.DIAG.CanDepend       = "Cannabis Dependence",
    SCID.DIAG.CocAbuse        = "Cocaine Abuse",
    SCID.DIAG.CocDepend       = "Cocaine Dependence"
  )
  
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
  
  # pretty label for plotting; fallback to code if unmapped
  idx <- match(long_prev$dx, names(code_to_label))
  long_prev$dx_lab <- ifelse(is.na(idx),
                             as.character(long_prev$dx),
                             code_to_label[idx])
  
  # enforce desired order (keep present ones, then any extras)
  present <- intersect(desired_order, unique(long_prev$dx_lab))
  extras  <- setdiff(unique(long_prev$dx_lab), present)
  dx_levels_plot <- c(present, sort(extras))
  long_prev$dx_lab <- factor(long_prev$dx_lab, levels = dx_levels_plot)
  
  # plot uses dx_lab
  pH <- ggplot2::ggplot(long_prev,
                        ggplot2::aes(dx_lab, factor(cluster), fill = prev)) +
    ggplot2::geom_tile() +
    ggplot2::scale_fill_viridis_c(limits = c(0, 1)) +
    ggplot2::labs(x = NULL, y = "cluster", fill = "prevalence",
                  title = "Major diagnoses across Louvain communities") +
    ggplot2::theme_minimal(12) +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 45, hjust = 1))
  
  print(pH)
  ggplot2::ggsave("FIG_cluster_by_dx_heatmap.png", pH, width = 8.5, height = 5.5, dpi = 150)
}

# ------------------------------------------------------------------------------
# Heatmap of over-representation (lift) with significance marks
#  • lift = prevalence_in_cluster / prevalence_overall   (colour uses log2)
#  • p from Fisher/binomial/hypergeo on raw counts, BH-adjusted per-dx column
#  • "**" = FDR <= 0.01, "*" = FDR <= 0.05, "ns" otherwise; blank if not tested
#  • switches:
#       only_over = TRUE  -> test only over-representation
#       min_total_cases_plot -> hide dx columns with very small totals (a+b)
#       show_under_colors = FALSE -> grey-out under-representation
# ------------------------------------------------------------------------------

plot_overrep_with_sig <- function(
    DX, clusters_all, dx_to_plot,
    min_in_cases = 5, min_out_cases = 5, min_total_cases = 10,
    use_fdr = TRUE, alpha1 = 0.05, alpha2 = 0.01,
    include_zero_cluster = FALSE, mark_underrep = FALSE,
    title = "Over-representation of major diagnoses by Louvain community",
    file  = "FIG_overrep_lift_sig.png",
    only_over = FALSE,
    min_total_cases_plot = 10,
    show_under_colors = TRUE,
    ref_ids_for_overall = NULL,   # NEW: reference population for prev_all
    drop_dx = NULL                # NEW: vector of dx names to exclude from plot
){
  
  if (is.null(rownames(DX))) rownames(DX) <- as.character(DX$participant_id)
  
  # IDs used for cluster membership and testing (may exclude C6)
  ids_cur <- intersect(rownames(DX), as.character(clusters_all$participant_id))
  
  # IDs used for overall prevalence in the lift denominator (freeze to full set if provided)
  ids_ref <- if (is.null(ref_ids_for_overall)) ids_cur else {
    intersect(rownames(DX), as.character(ref_ids_for_overall))
  }
  
  # Align
  DX_cur <- align_to_ids(DX, ids_cur)[, setdiff(names(DX), "participant_id"), drop = FALSE]
  DX_ref <- align_to_ids(DX, ids_ref)[, setdiff(names(DX), "participant_id"), drop = FALSE]
  
  # Optional DX drop (e.g., "NODIAG")
  if (!is.null(drop_dx)) {
    keep_cols <- setdiff(colnames(DX_cur), drop_dx)
    DX_cur <- DX_cur[, keep_cols, drop = FALSE]
    DX_ref <- DX_ref[, intersect(colnames(DX_ref), keep_cols), drop = FALSE]
  }
  
  # keep x-axis order exactly as provided
  cols <- intersect(dx_to_plot, colnames(DX_cur))
  if (!length(cols)) stop("dx_to_plot has no columns present in DX after filtering")
  
  # cluster vector on current ids
  cl_map <- setNames(clusters_all$cluster, clusters_all$participant_id)
  cl_cur <- as.integer(cl_map[ids_cur]); cl_cur[is.na(cl_cur)] <- 0L
  
  # overall prevalence on reference ids (fixed denominator)
  prev_all_vec <- vapply(cols, function(v){
    a <- sum(DX_ref[[v]] == 1L, na.rm = TRUE)
    c0 <- sum(DX_ref[[v]] == 0L, na.rm = TRUE)
    if (a + c0 > 0) a/(a + c0) else NA_real_
  }, numeric(1))
  names(prev_all_vec) <- cols
  
  # build long table over current set
  N <- length(ids_cur)
  cl_levels <- sort(setdiff(unique(cl_cur), if (include_zero_cluster) integer(0) else 0L))
  rows <- list()
  for (cl in cl_levels){
    in_cl  <- cl_cur == cl
    out_cl <- !in_cl
    n_in   <- sum(in_cl); n_out <- N - n_in
    for (v in cols){
      a <- sum(DX_cur[[v]][in_cl]  == 1L, na.rm = TRUE)
      c <- sum(DX_cur[[v]][in_cl]  == 0L, na.rm = TRUE)
      b <- sum(DX_cur[[v]][out_cl] == 1L, na.rm = TRUE)
      d <- sum(DX_cur[[v]][out_cl] == 0L, na.rm = TRUE)
      
      prev_in  <- if ((a + c) > 0) a/(a + c) else NA_real_
      prev_all <- prev_all_vec[[v]]                 # FIXED denominator
      lift_raw <- if (is.finite(prev_in) && is.finite(prev_all) && prev_all > 0) prev_in/prev_all else NA_real_
      
      # p-value on current split (unchanged)
      p <- NA_real_; tested <- FALSE; direction <- NA_integer_
      if (n_in > 0 && is.finite(prev_all)) {
        if (a == 0L) {
          bt <- try(stats::binom.test(a, n_in, p = prev_all, alternative = "less"), silent = TRUE)
          if (!inherits(bt, "try-error")) { p <- bt$p.value; tested <- TRUE; direction <- -1L }
        } else if (a == n_in) {
          bt <- try(stats::binom.test(a, n_in, p = prev_all, alternative = "greater"), silent = TRUE)
          if (!inherits(bt, "try-error")) { p <- bt$p.value; tested <- TRUE; direction <- +1L }
        } else if ((a + b) >= min_total_cases && a >= min_in_cases && b >= min_out_cases &&
                   all(c(n_in, n_out) > 0)){
          ft <- try(stats::fisher.test(matrix(c(a,b,c,d), 2, byrow = TRUE)), silent=TRUE)
          if (!inherits(ft,"try-error")) {
            p <- ft$p.value; tested <- TRUE
            lor <- log((a * d + 1e-9) / (b * c + 1e-9))
            direction <- ifelse(lor >= 0, +1L, -1L)
          }
        } else {
          p <- stats::phyper(a - 1, m = a + b, n = c + d, k = n_in, lower.tail = FALSE)
          tested <- is.finite(p); direction <- +1L
        }
      }
      
      eps <- 1e-12
      lift_for_colour <- if (is.finite(prev_in)) prev_in / pmax(prev_all, eps) else NA_real_
      
      rows[[length(rows) + 1L]] <- data.frame(
        cluster = cl, dx = v,
        a=a, b=b, c=c, d=d, n_in=n_in, n_out=n_out,
        prev_in = prev_in, prev_all = prev_all,
        lift = lift_raw, lift_colour = lift_for_colour,
        p_raw = if (tested) p else NA_real_, tested = tested, direction = direction,
        stringsAsFactors = FALSE
      )
    }
  }
  
  tab <- do.call(rbind, rows)
  
  # --- map codes -> pretty labels and enforce desired order ---
  code_to_label <- c(
    "NODIAG"                    = "No Diagnosis on Axis I",
    "SCID.DIAG.ADHD"            = "Attention-Deficit/Hyperactivity Disorder",
    "SCID.DIAG.BipolarI"        = "Bipolar I Disorder",
    "SCID.DIAG.DepressNOS"      = "Depressive Disorder NOS",
    "SCID.DIAG.MDD"             = "Major Depressive Disorder",
    "SCID.DIAG.Schizophrenia"   = "Schizophrenia",
    "SCID.DIAG.Schizoaffective" = "Schizoaffective Disorder",
    "SCID.DIAG.AlcAbuse"        = "Alcohol Abuse",
    "SCID.DIAG.AlcDepend"       = "Alcohol Dependence",
    "SCID.DIAG.AmpAbuse"        = "Amphetamine Abuse",
    "SCID.DIAG.AmpDepend"       = "Amphetamine Dependence",
    "SCID.DIAG.CanAbuse"        = "Cannabis Abuse",
    "SCID.DIAG.CanDepend"       = "Cannabis Dependence",
    "SCID.DIAG.CocAbuse"        = "Cocaine Abuse",
    "SCID.DIAG.CocDepend"       = "Cocaine Dependence"
  )
  
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
  
  # pretty label for plotting; fallback to code if unmapped
  tab$dx_lab <- unname(ifelse(tab$dx %in% names(code_to_label),
                              code_to_label[tab$dx],
                              as.character(tab$dx)))
  
  # keep only desired labels that are present; append any extras
  present <- intersect(desired_order, unique(tab$dx_lab))
  extras  <- setdiff(unique(tab$dx_lab), present)
  dx_levels_plot <- c(present, sort(extras))
  
  # factors for plotting
  tab$dx_lab <- factor(tab$dx_lab, levels = dx_levels_plot)
  
  # hide ultra-rare dx by totals in the reference population
  tot_by_dx <- vapply(cols, function(v){
    sum(DX_ref[[v]] == 1L, na.rm = TRUE) + sum(DX_ref[[v]] == 0L, na.rm = TRUE)
  }, integer(1))
  rare_dx <- names(tot_by_dx)[tot_by_dx < min_total_cases_plot]
  if (length(rare_dx)) {
    m <- tab$dx %in% rare_dx
    tab$p_raw[m] <- NA_real_; tab$tested[m] <- FALSE
    tab$lift_colour[m] <- NA_real_
  }
  
  # ---- finish: adjust p, labels, plot, return ----
  # BH per diagnosis (column-wise)
  tab$q <- NA_real_
  for (v in unique(tab$dx)) {
    i <- which(tab$dx == v & is.finite(tab$p_raw))
    if (length(i)) tab$q[i] <- if (use_fdr) p.adjust(tab$p_raw[i], "BH") else tab$p_raw[i]
  }
  
  # significance labels
  tab$label <- ifelse(!tab$tested | !is.finite(tab$q), "",
                      ifelse(tab$q < alpha2, "**",
                             ifelse(tab$q < alpha1, "*", "ns")))
  
  # colour value = log2(lift) using the fixed denominator
  eps <- 1e-12
  tab$val <- log2(pmax(tab$lift, eps))   # <-- use raw lift, not lift_colour
  cap <- 2
  tab$val_c <- pmin(pmax(tab$val, -cap), cap)
  if (!show_under_colors) tab$val_c[tab$val_c < 0] <- NA_real_
  
  # honour input order for x-axis
  tab$dx <- factor(tab$dx, levels = cols)
  # clusters as C1, C2, ...
  cl_lev <- sort(unique(tab$cluster))
  tab$cluster_f <- factor(paste0("C", tab$cluster), levels = paste0("C", cl_lev))
  
  # draw heatmap
  suppressWarnings({
    p <- ggplot2::ggplot(tab, ggplot2::aes(x = dx_lab, y = cluster_f, fill = val_c)) +  # <- dx_lab
      ggplot2::geom_tile() +
      ggplot2::geom_text(
        ggplot2::aes(
          label = label,
          alpha = ifelse(label == "ns", 0.25, 1),
          colour = ifelse(label == "ns", "black", "black")
        ),
        size = 3.3, fontface = "bold"
      ) +
      ggplot2::scale_alpha_identity(guide = "none") +
      ggplot2::scale_colour_identity(guide = "none") +
      ggplot2::scale_fill_gradient2(
        name = "log2(lift)",
        limits = c(-cap, cap),
        midpoint = 0,
        low = "#1f3a93", 
        mid = "white", 
        high = "#b33939",
      ) +
      ggplot2::labs(x = NULL, y = NULL, title = title) +
      ggplot2::theme_minimal(base_size = 11) +
      ggplot2::theme(panel.grid = ggplot2::element_blank(),
                     axis.text.x = ggplot2::element_text(angle = 55, hjust = 1, vjust = 1))
    print(p)
    ggplot2::ggsave(file, p, width = 9, height = 7, dpi = 300, bg = "white")
  })
  
  # return table matching heatmap (keep dx order)
  out <- tab[, c("cluster","dx","lift","q","label")]
  out$cluster <- as.integer(out$cluster)
  out$dx <- factor(out$dx, levels = cols)
  out <- out[order(out$cluster, as.integer(out$dx)), ]
  rownames(out) <- NULL
  return(invisible(out))

}

# Freeze denominator to full set once
ids_full <- as.character(clusters_all$participant_id)

# Exclude NODIAG from the no-C6 column set
dx_for_heatmap_no6 <- setdiff(dx_for_heatmap, "NODIAG")

# 1) Full figure (uses fixed denominator)
lift_tab <- plot_overrep_with_sig(
  DX = DX, clusters_all = clusters_all, dx_to_plot = dx_for_heatmap,
  min_in_cases = 5, min_out_cases = 5, min_total_cases = 10,
  use_fdr = TRUE, alpha1 = 0.05, alpha2 = 0.01,
  only_over = FALSE, min_total_cases_plot = 10, show_under_colors = TRUE,
  ref_ids_for_overall = ids_full,
  file = "FIG_overrep_lift_sig.png",
  title = "Over-representation of major diagnoses by Louvain community"
)

# 2) Excluding cluster 6 and the NODIAG column (same denominator)
clusters_no6 <- subset(clusters_all, cluster != 6L)
lift_tab_no6 <- plot_overrep_with_sig(
  DX = DX, clusters_all = clusters_no6, dx_to_plot = dx_for_heatmap_no6,
  min_in_cases = 5, min_out_cases = 5, min_total_cases = 10,
  use_fdr = TRUE, alpha1 = 0.05, alpha2 = 0.01,
  only_over = FALSE, min_total_cases_plot = 10, show_under_colors = TRUE,
  ref_ids_for_overall = ids_full,
  file = "FIG_overrep_lift_sig_noC6.png",
  title = "Over-representation by community - excluding nodiag (C6)"
)

# --- Make a printable "lift table" that matches FIG_overrep_lift_sig ---
plot_overrep_lift_table <- function(tab_out,
                                    file = "FIG_overrep_lift_table.png",
                                    digits = 2){
  stopifnot(all(c("cluster","dx","lift","label") %in% names(tab_out)))
  df <- tab_out
  
  # keep same ordering as the heatmap
  df$cluster <- factor(df$cluster, levels = sort(unique(df$cluster)))
  df$dx      <- factor(df$dx,      levels = levels(df$dx))
  
  # text to show in each cell
  lift_txt <- ifelse(is.finite(df$lift),
                     sprintf(paste0("%.", digits, "f"), df$lift), "")
  sig_txt  <- ifelse(df$label %in% c("*","**"), paste0(" ", df$label), "")
  df$cell  <- paste0(lift_txt, sig_txt)
  
  # draw as a clean “table-like” plot
  p <- ggplot2::ggplot(df, ggplot2::aes(x = dx, y = cluster)) +
    ggplot2::geom_tile(fill = "white", colour = "grey85") +
    ggplot2::geom_text(ggplot2::aes(label = cell), size = 3.2) +
    ggplot2::labs(title = "Lift (in / overall) with significance",
                  x = NULL, y = "cluster",
                  caption = "Cells show lift; * FDR≤0.05, ** FDR≤0.01 (same tests as heatmap).") +
    ggplot2::theme_minimal(12) +
    ggplot2::theme(panel.grid = ggplot2::element_blank(),
                   axis.text.x = ggplot2::element_text(angle = 55, hjust = 1, vjust = 1))
  
  print(p)
  ggplot2::ggsave(file, p, width = 9, height = 7, dpi = 150, bg = "white")
}

# --- Usage (right after you make the heatmap) ---
# lift_tab is returned (invisibly) by plot_overrep_with_sig(...)
# Re-run with the same arguments you used for the figure:
lift_tab <- plot_overrep_with_sig(
  DX = DX, clusters_all = clusters_all, dx_to_plot = dx_for_heatmap,
  min_in_cases = 5, min_out_cases = 5, min_total_cases = 10,
  use_fdr = TRUE, alpha1 = 0.05, alpha2 = 0.01,
  only_over = FALSE, min_total_cases_plot = 10, show_under_colors = TRUE,
  file = "FIG_overrep_lift_sig.png"
)

# Save a CSV too (optional)
readr::write_csv(lift_tab, "overrep_lift_table.csv")

# Plot the matching table
plot_overrep_lift_table(lift_tab, "FIG_overrep_lift_table.png")

# CV curve of out-of-sample R^2(dist) vs MDS dimensionality
r2k <- cv_r2_mds(D_dx, k_grid = 1:8, K = 5, seed = 42)
if (nrow(r2k)) {
  p_r2 <- ggplot2::ggplot(r2k, ggplot2::aes(k, r2)) +
    ggplot2::geom_line() + ggplot2::geom_point() +
    ggplot2::scale_x_continuous(breaks = r2k$k) +
    ggplot2::labs(x = "MDS dimensionality (k)",
                  y = expression(R^2~"(distances, 5-fold CV)"),
                  title = "Out-of-sample distance preservation vs k") +
    ggplot2::theme_minimal(12)
  print(p_r2)
  ggplot2::ggsave("FIG_mds_cv_r2.png", p_r2, width = 6.5, height = 4.5, dpi = 150)
  k_star <- r2k$k[which.max(r2k$r2)]
  message(sprintf("[MDS CV] best k ≈ %d (CV r^2=%.2f)", k_star, max(r2k$r2, na.rm=TRUE)))
}

# Pairs matrix of MDS (axes 1..6) with 50% ellipses
plot_mds_pairs <- function(D, memb, max_axes = 6,
                           file = "FIG_dxspace_mds_pairs.png"){
  if (!requireNamespace("GGally", quietly=TRUE)) return(invisible(NULL))
  if (!requireNamespace("ggplot2", quietly=TRUE)) return(invisible(NULL))
  fit <- cmdscale(D, k = max_axes, eig = TRUE, add = TRUE)
  XY  <- as.data.frame(fit$points)
  pve <- { pos <- fit$eig[fit$eig > 0]; if (length(pos)) pmax(0, fit$eig[seq_len(ncol(XY))]) / sum(pos) else rep(NA, ncol(XY)) }
  names(XY) <- paste0("MDS", seq_len(ncol(XY)), " (", sprintf("%.0f", 100*pve), "%)")
  XY$cluster <- factor(memb)
  lower_fun <- function(data, mapping, ...){
    ggplot2::ggplot(data, mapping) +
      ggplot2::geom_point(size = 0.9, alpha = .85) +
      ggplot2::stat_ellipse(type = "norm", level = .5, linewidth = .25,
                            linetype = 2, show.legend = FALSE, na.rm = TRUE) +
      ggplot2::coord_equal() + ggplot2::theme_minimal(9)
  }
  diag_fun <- function(data, mapping, ...){
    ggplot2::ggplot(data, mapping) +
      ggplot2::geom_density(ggplot2::aes(fill = cluster, colour = cluster),
                            alpha = .35, linewidth = .3, position = "identity") +
      ggplot2::theme_minimal(9) + ggplot2::theme(legend.position = "none")
  }
  p <- GGally::ggpairs(
    XY, columns = 1:(min(max_axes, ncol(XY)-1)),
    ggplot2::aes(color = cluster), lower = list(continuous = lower_fun),
    diag = list(continuous = diag_fun), upper = list(continuous = "blank")
  ) + ggplot2::theme(strip.text = ggplot2::element_text(size = 9))
  ggplot2::ggsave(file, p, width = 10, height = 10, dpi = 150)
  print(p)
}
plot_mds_pairs(D_dx, lv$membership, max_axes = 6)

#' Helper to compute membership plus null statistics for a given k and variant
#' - Never resets RNG inside (seed is used only for reproducible graph choices outside).
#' - Names membership by rownames of D so that ARIs align across settings.
#' - null_scope = "full" ensures degree-null comparability across k.
#' - Reports p preferred from degree-null (p_deg); keeps weight-null z for diagnostics.
get_membership <- function(
    D, k = K_KNN, union = UNION_KNN, mult, B = 500, seed = NULL,
    deg_reps = 200L, null_scope = "full"
){
  g  <- knn_graph_from_dist(D, k = k, union = union, mult = mult)
  lv <- louvain_with_stats(
    g, D, n_perm = B, min_size = MIN_CLUSTER_SIZE,
    min_weight = MIN_CLUSTER_WEIGHT, v_weight = mult,
    seed = seed, deg_reps = deg_reps, null_scope = null_scope
  )
  m  <- lv$membership
  names(m) <- rownames(as.matrix(D))
  q_mean <- mean(lv$Q_null, na.rm = TRUE); q_sd <- stats::sd(lv$Q_null, na.rm = TRUE)
  z_w    <- if (is.finite(q_mean) && is.finite(q_sd) && q_sd > 0) (lv$Q - q_mean)/q_sd else NA_real_
  # Prefer degree-null p for stability when available
  p_pref <- if (is.finite(lv$p_deg)) lv$p_deg else lv$Q_p_upper
  tab <- table(lv$membership)
  n_kept <- length(setdiff(unique(lv$membership), 0L))
  list(
    m = m,
    Q = lv$Q,
    p_pref = p_pref,
    Q_p = lv$Q_p_upper,
    Q_p_two = lv$Q_p_two,
    S = lv$S,
    S_p = lv$S_p,
    z_w = z_w,
    z_deg = lv$z_deg,
    n_kept = n_kept
  )
}

base <- get_membership(D_dx, k = K_KNN, union = UNION_KNN,
                       mult = multU, B = 500, seed = 42,
                       deg_reps = 200L, null_scope = "full")

grid <- expand.grid(K = K_GRID, variant = VARIANT, stringsAsFactors = FALSE)
res  <- vector("list", nrow(grid))
for (i in seq_len(nrow(grid))) {
  k <- grid$K[i]; u <- grid$variant[i] == "union"
  gi <- get_membership(D_dx, k = k, union = u, mult = multU,
                       B = 500, seed = NULL,  # do not reseed within the sweep
                       deg_reps = 200L, null_scope = "full")
  res[[i]] <- data.frame(
    K = k, variant = grid$variant[i],
    Q = gi$Q, p_pref = gi$p_pref, Q_p_two = gi$Q_p_two,
    S = gi$S, S_p = gi$S_p,
    z_w = gi$z_w, z_deg = gi$z_deg, n_kept = gi$n_kept,
    ARI_vs_base = align_ari(base$m, gi$m)
  )
}
sens <- do.call(rbind, res)
readr::write_csv(sens, "clustering_sensitivity_grid.csv")
print(sens)

# --------------- Bootstrap frequency of “majors” (by-eligibility) -------------
set.seed(42)
B <- 300

dx_pool <- setdiff(names(DX), c("participant_id","ANY_DX"))
success_by_dx <- setNames(integer(length(dx_pool)), dx_pool)
trials_by_dx  <- setNames(integer(length(dx_pool)), dx_pool)

n_valid <- 0L
for (b in seq_len(B)){
  idx  <- sample(seq_len(nrow(DX)), replace = TRUE)
  DX_b <- DX[idx, , drop = FALSE]
  dd_b  <- dedup_dx(DX_b); DXu_b <- dd_b$DXu
  colsB <- setdiff(names(DXu_b), c("participant_id","mult","ANY_DX"))
  keepB <- rowSums(DXu_b[, colsB, drop=FALSE], na.rm=TRUE) > 0L
  DXu_id_b <- DXu_b[keepB, , drop=FALSE]
  if (nrow(DXu_id_b) < 5) next
  D_b <- gower_dist_dx(DXu_id_b); if (is.null(D_b)) next
  g_b <- try(knn_graph_from_dist(D_b, k = K_KNN, union = UNION_KNN, mult = DXu_id_b$mult), silent = TRUE)
  if (inherits(g_b,"try-error") || igraph::ecount(g_b) == 0L) next
  lv_b <- louvain_with_stats(g_b, D_b, n_perm = 0, min_size = MIN_CLUSTER_SIZE, seed = 42)
  n_valid <- n_valid + 1L
  sigU_b <- apply(DXu_id_b[, colsB, drop=FALSE], 1, paste0, collapse = "")
  cl_all_b <- expand_membership(lv_b$membership, sigU_b, dd_b$sig, rownames(DX_b))
  enr_b <- diagnosis_enrichment(
    DX_b, cl_all_b,
    alpha_fdr = ALPHA_FDR, min_prev_in = MIN_PREV_IN_CL, min_or = MIN_OR,
    exclude = c("ANY_DX"),
    min_in_cases = MIN_CASES_TOTAL, min_total_cases = MIN_CASES_TOTAL, min_out_cases = MIN_CASES_TOTAL
  )
  mA <- if (!is.null(enr_b$majors)) enr_b$majors else character(0)
  counts_b <- build_counts(DX_b)
  loc_b <- label_localization_table(
    g_b, DXu_id_b, B = 500,
    n_pos_min = MIN_CASES_TOTAL, n_neg_min = MIN_CASES_TOTAL,
    counts_all = counts_b
  )
  mB <- subset(loc_b, pmin(assort_p, knn_p) <= ALPHA_LOCALIZE)$dx
  auc_b <- data.frame(
    dx = colsB,
    AUC = vapply(colsB, function(v)
      suppressWarnings(auc_one_vs_rest_knn_weighted(
        DXu_id_b, v, k = 10, pos_min = MIN_CASES_TOTAL, neg_min = MIN_CASES_TOTAL
      )), numeric(1))
  )
  mC <- subset(auc_b, is.finite(AUC) & AUC >= AUC_MIN)$dx
  eligible_b <- with(counts_b, dx[prev >= PREV_MIN | n1 >= NCASE_MIN])
  elig_here <- intersect(eligible_b, names(trials_by_dx))
  trials_by_dx[elig_here] <- trials_by_dx[elig_here] + 1L
  majors_b <- sort(intersect(unique(c(mA, mB, mC)), eligible_b))
  sel_here <- intersect(majors_b, names(success_by_dx))
  success_by_dx[sel_here] <- success_by_dx[sel_here] + 1L
}

stab <- data.frame(
  dx     = names(success_by_dx),
  count  = as.integer(success_by_dx),
  trials = as.integer(trials_by_dx),
  stringsAsFactors = FALSE
)
stab <- subset(stab, trials > 0)
stab$freq <- with(stab, count / trials)
stab <- stab[order(stab$freq, decreasing = TRUE), ]
stab$lo <- qbinom(0.025, size = stab$trials, prob = stab$freq) / stab$trials
stab$hi <- qbinom(0.975, size = stab$trials, prob = stab$freq) / stab$trials
readr::write_csv(stab, "major_dx_bootstrap_frequency_by_eligibility.csv")
print(stab)

# Plot bootstrap stability with denominators in labels
if (requireNamespace("ggplot2", quietly = TRUE) && requireNamespace("scales", quietly = TRUE)) {
  stab$coverage <- stab$trials / max(1, n_valid)
  stab$dx_lab <- ifelse(
    stab$trials < n_valid,
    sprintf("%s (%d/%d)", stab$dx, stab$trials, n_valid),
    stab$dx
  )
  p_ci <- ggplot2::ggplot(stab, ggplot2::aes(x = reorder(dx_lab, freq), y = freq)) +
    ggplot2::geom_col(fill = "grey30") +
    ggplot2::geom_errorbar(ggplot2::aes(ymin = lo, ymax = hi), width = .2) +
    ggplot2::coord_flip() +
    ggplot2::scale_y_continuous(labels = scales::percent, limits = c(0, 1)) +
    ggplot2::labs(
      x = NULL,
      y = "bootstrap frequency (among eligible resamples)",
      title = sprintf("Major Dx stability (per-dx denominator) — valid bootstraps: %d", n_valid),
      subtitle = "Denominators shown in labels for Dx with fewer eligible resamples."
    ) + ggplot2::theme_minimal(12)
  print(p_ci)
  ggplot2::ggsave("FIG_major_dx_bootstrap_frequency_CI_by_eligibility.png", p_ci,
                  width = 7, height = 5, dpi = 150)
}

# ------------------------------ Run summary -----------------------------------
run_summary <- data.frame(
  n_total         = nrow(DX),
  n_unique        = nrow(DXu),
  n_used_nonzero  = nrow(DXu_id),
  K_KNN           = K_KNN,
  Q_modularity    = lv$Q,
  Q_pvalue        = lv$Q_p,
  silhouette      = lv$S,
  silhouette_p    = lv$S_p,
  ARI_median      = ARI_med,
  ARI_IQR         = ARI_iqr,
  ID_TwoNN_all    = ID_twonn_all,
  ID_TwoNN_core   = ID_twonn_core,
  ID_LB_core      = ID_lbmle_core,
  n_core          = if (length(idx_core)) length(idx_core) else NA_integer_,
  n_clusters_kept = length(setdiff(unique(lv$membership), 0L))
)
readr::write_csv(run_summary, "dx_space_run_summary.csv")
print(run_summary)
