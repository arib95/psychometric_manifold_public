# ==============================================================================
# PROBABILISTIC, SMOOTH CLUSTERS from DxW_A (no CSV lookups, no crosswalks)
# ==============================================================================

suppressPackageStartupMessages({
  library(mgcv); library(igraph); library(Matrix); library(grid)
  library(ggplot2); library(dplyr); library(tidyr); library(readr)
  library(scales); library(cluster); library(tibble); library(clue)
})

set.seed(42)
`%||%` <- function(a,b) if (!is.null(a)) a else b

# --- Optional imports from dimension_clusters.R ---
LEIDEN_MEMBERSHIP_CSV <- "cluster_membership_all_participants.csv"     # cols: participant_id, cluster
MAJOR_DX_CSV          <- "selected_major_diagnoses_union.csv"          # cols: dx_code (or 'dx')
USE_LEIDEN_K          <- TRUE         # if TRUE, force K = number of Leiden clusters kept in CSV
LEIDEN_LABEL_PREFIX   <- "L"          # renaming prefix for aligned clusters

# ---- Palette (for all fills) -------------------------------------------------
palette_cfg <- list(
  engine    = "scico",     # "scico" | "colorspace" | "brewer" | "paletteer" | "manual"
  name      = "lipari",   # e.g. scico: "lajolla","batlow","oslo" ; brewer: "YlGnBu"
  direction = 1,           # 1 or -1 (reverse)
  colours   = NULL         # used only if engine == "manual"
)

psych_theme <- function(base_size = 15){
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(linewidth = 0.3),
      plot.title = element_text(face = "bold", margin = margin(b = 6)),
      axis.title = element_text(margin = margin(t = 2, r = 2)),
      strip.background = element_blank(),
      strip.text = element_text(face = "italic"),
      legend.position = "right",
      legend.key.height = unit(12, "pt"),
      legend.key.width  = unit(12, "pt")
    )
}

scale_fill_psych <- function(lims = NULL, name = NULL, percent_labels = FALSE){
  labs_fun <- if (isTRUE(percent_labels))
    scales::percent_format(accuracy = 1)
  else
    scales::label_number(accuracy = 0.01)
  
  eng <- tolower(palette_cfg$engine %||% "scico")
  dir <- palette_cfg$direction %||% 1
  nm  <- palette_cfg$name %||% "lajolla"
  
  if (eng == "scico" && requireNamespace("scico", quietly = TRUE)) {
    scico::scale_fill_scico(
      palette = nm, direction = dir, limits = lims,
      na.value = NA, name = name, labels = labs_fun
    )
  } else if (eng == "colorspace" && requireNamespace("colorspace", quietly = TRUE)) {
    colorspace::scale_fill_continuous_sequential(
      palette = nm, rev = (dir == -1), limits = lims,
      na.value = NA, name = name, labels = labs_fun
    )
  } else if (eng == "brewer") {
    ggplot2::scale_fill_distiller(
      palette = nm, type = "seq", direction = if (dir==1) 1 else -1,
      limits = lims, na.value = NA, name = name, labels = labs_fun
    )
  } else if (eng == "paletteer" && requireNamespace("paletteer", quietly = TRUE)) {
    paletteer::scale_fill_paletteer_c(
      nm, direction = dir, limits = lims,
      na.value = NA, name = name, labels = labs_fun
    )
  } else if (eng == "manual" && !is.null(palette_cfg$colours)) {
    ggplot2::scale_fill_gradientn(
      colours = palette_cfg$colours, limits = lims,
      na.value = NA, name = name, labels = labs_fun
    )
  } else {
    # conservative fallback
    ggplot2::scale_fill_gradient(
      low = "#f7fbff", high = "#08306b", limits = lims,
      na.value = NA, name = name, labels = labs_fun
    )
  }
}

save_png <- function(file, plot, width = 8, height = 6, dpi = 300){
  ggsave(file, plot, width = width, height = height, dpi = dpi, bg = "white")
}

# --------------------------- Parameters ---------------------------------------
# K-grid will be centered around Louvain K (computed later); these are hard caps
K_MIN <- 2; K_MAX <- 4

LAMBDA_GRID <- c(0, 0.05, 0.1, 0.2, 0.4, 0.8)
KNN_BASE    <- 12
LOCAL_SCALE <- TRUE
N_START     <- 3
N_ITER      <- 700
TOL         <- 1e-5
EPS         <- 1e-10
MIN_POS_DEF <- 10L
MIN_NEG_DEF <- 10L

# usage/regularisation knobs
MASS_FLOOR_FRAC <- 0.05   # stricter alive threshold (was 0.02)
BALANCE_EVERY   <- 8L
RESEED_MAX      <- 3L
ENTROPY_PUSH    <- 1e-3   # gentler than 1e-3

# selection weights/penalties
W_Q   <- 0.45
W_SNN <- 0.50
W_REC <- 0.20
W_SIL <- 0.15
PEN_EMPTY <- 0.60
PEN_IMB   <- 0.15
PEN_RED   <- 0.30   # redundancy penalty for nearly-duplicate columns of V

# --------------------------- Helpers ------------------------------------------
as_base2 <- function(B){
  A <- as.data.frame(B)[,1:2,drop=FALSE]; names(A) <- c("b1","b2")
  A$b1 <- as.numeric(A$b1); A$b2 <- as.numeric(A$b2)
  as.matrix(A)
}
get_base_ids <- function(){
  if (exists("Base_A")) return(rownames(Base_A) %||% rownames(Base) %||% NULL)
  if (exists("Base"))   return(rownames(Base))
  NULL
}

# point-in-polygon for convex-hull masking (ray casting + on-edge)
inside_hull <- function(px, py, poly, eps = 1e-12){
  n <- nrow(poly); j <- n; inside <- rep(FALSE, length(px)); on_edge <- rep(FALSE, length(px))
  for (i in seq_len(n)){
    xi <- poly$b1[i]; yi <- poly$b2[i]; xj <- poly$b1[j]; yj <- poly$b2[j]
    dx <- xj - xi; dy <- yj - yi
    seg_len2 <- dx*dx + dy*dy + eps
    t <- pmin(1, pmax(0, ((px - xi)*dx + (py - yi)*dy) / seg_len2))
    projx <- xi + t*dx; projy <- yi + t*dy
    on_edge <- on_edge | ((px - projx)^2 + (py - projy)^2 <= (1e-9)^2)
    cross <- ((yi > py) != (yj > py)) & (px < (xj - xi) * (py - yi) / (yj - yi + eps) + xi)
    inside <- xor(inside, cross); j <- i
  }
  inside | on_edge
}

# Gaussian kNN graph on base (symmetric by union)
knn_graph_base <- function(B, k=12, local_scale=TRUE){
  B <- as.matrix(B); n <- nrow(B)
  D <- as.matrix(dist(B)); diag(D) <- Inf
  kth <- function(v,k) sort(v,partial=k)[k]
  r_k <- apply(D,1,kth,k=min(k, n-1))
  if (!local_scale) sigma <- stats::median(r_k[is.finite(r_k)], na.rm = TRUE)
  nbrs <- lapply(1:n, function(i) { o <- order(D[i,])[1:min(k, n-1)]; o[is.finite(D[i,o])] })
  edges <- list()
  for (i in 1:n){
    js <- nbrs[[i]]
    partners <- unique(c(js, which(vapply(nbrs, function(x) i %in% x, logical(1)))))
    partners <- partners[partners > i]
    if (!length(partners)) next
    w <- if (local_scale){
      sprod <- outer(r_k[i], r_k[partners], "*")
      exp(-(D[i, partners]^2)/(2*sprod))
    } else exp(-(D[i, partners]^2)/(2*sigma^2))
    edges[[length(edges)+1L]] <- data.frame(from=i,to=partners,weight=pmax(EPS,w))
  }
  Edf <- do.call(rbind, edges)
  igraph::simplify(igraph::graph_from_data_frame(Edf, FALSE, vertices=data.frame(name=1:n)),
                   edge.attr.comb=list(weight="sum"))
}

laplacian_from_graph <- function(g, normalized=FALSE){
  n <- gorder(g)
  w <- Matrix(0, n, n, sparse=TRUE)
  el <- as_edgelist(g, names=FALSE)
  if (!nrow(el)) stop("Empty graph.")
  w[cbind(el[,1], el[,2])] <- E(g)$weight
  w[cbind(el[,2], el[,1])] <- E(g)$weight
  d <- Matrix::Diagonal(n, x = Matrix::rowSums(w))
  L <- d - w
  if (normalized) {
    di <- 1/sqrt(pmax(Matrix::diag(d), EPS))
    Dm <- Matrix::Diagonal(n, di)
    L <- Dm %*% L %*% Dm
  }
  L
}

# Standardise to unit disk (fallback)
if (!exists("standardise_to_circle")){
  standardise_to_circle <- function(Base2, cover=0.995){
    X <- as.matrix(Base2[,1:2,drop=FALSE]); storage.mode(X) <- "double"
    mu <- colMeans(X); S <- stats::cov(X); if (!all(is.finite(S)) || det(S)<=0) S <- S + diag(1e-8,2)
    eig <- eigen(S, symmetric=TRUE); V <- eig$vectors; L <- pmax(eig$values, 1e-12)
    S_half <- V %*% diag(sqrt(L),2) %*% t(V)
    S_hi   <- V %*% diag(1/sqrt(L),2) %*% t(V)
    U0 <- t(S_hi %*% t(sweep(X,2,mu,"-"))); r <- sqrt(rowSums(U0^2))
    s  <- as.numeric(stats::quantile(r, probs=cover, na.rm=TRUE)); if (!is.finite(s) || s<=0) s <- max(r, na.rm=TRUE)
    fwd <- function(xb1, xb2){ Uq <- t(S_hi %*% t(cbind(xb1,xb2) - matrix(mu, nrow(xb1), 2, byrow=TRUE)))/s; colnames(Uq) <- c("u1","u2"); Uq }
    inv <- function(u1,u2){ Xq <- t(S_half %*% t(cbind(u1,u2)*s)) + matrix(mu, length(u1), 2, byrow=TRUE); colnames(Xq) <- c("b1","b2"); Xq }
    list(mu=mu,S_half=S_half,S_half_inv=S_hi,s=s,fwd=fwd,inv=inv)
  }
}

# ------------------------ Build P from DxW_A ----------------------------------
build_P_from_DxW_A <- function(Base_in, Dx_in,
                               min_pos = MIN_POS_DEF, min_neg = MIN_NEG_DEF,
                               dx_schema = NULL){
  Base_A <- as_base2(Base_in)
  id_candidates <- c("participant_id","subject_id","id","PID","participant")
  DX <- as.data.frame(Dx_in, check.names = FALSE)
  if (is.null(rownames(DX))) {
    id_in <- intersect(id_candidates, colnames(DX))
    if (length(id_in)) {
      rn <- as.character(DX[[id_in[1]]]); if (!anyDuplicated(rn)) rownames(DX) <- rn
    }
  }
  if (is.null(rownames(Base_A)) && !is.null(rownames(DX))) rownames(Base_A) <- rownames(DX)
  stopifnot(!is.null(rownames(Base_A)))
  DX <- DX[rownames(Base_A), , drop = FALSE]
  
  # choose which Dx to model
  dx_cols <- setdiff(colnames(DX), id_candidates)
  if (!length(dx_cols)) stop("No diagnosis columns found in DxW_A.")
  
  # audit counts (unchanged)
  pos <- vapply(dx_cols, function(nm) sum(as.integer(DX[[nm]] > 0), na.rm = TRUE), integer(1))
  neg <- vapply(dx_cols, function(nm) sum(as.integer(DX[[nm]] == 0), na.rm = TRUE), integer(1))
  audit <- data.frame(dx = dx_cols, n_pos = pos, n_neg = neg, n_total = nrow(DX))
  try(readr::write_csv(audit[order(-audit$n_pos, audit$dx), ], "diagnosis_counts_audit.csv"), silent = TRUE)
  
  # keep set honours schema order if provided
  if (!is.null(dx_schema)) {
    want <- unique(dx_schema)
    keep <- want[want %in% dx_cols]                    # preserve schema order
    missing <- setdiff(want, dx_cols)
    if (length(missing)) {
      warning(sprintf("[DX schema] %d diagnoses not found and will be dropped: %s",
                      length(missing), paste(missing, collapse = ", ")))
    }
  } else {
    keep <- names(which(pos >= min_pos & neg >= min_neg))
    if (!length(keep)) for (th in c(8L,5L,3L,1L)) {
      cand <- names(which(pos >= th & neg >= th)); if (length(cand)) { keep <- cand; break }
    }
    if (!length(keep)) {
      cand <- names(which(pos > 0 & neg > 0))
      if (!length(cand)) stop("No diagnosis has both classes present (≥1 pos & ≥1 neg).")
      warning(sprintf("Proceeding with %d dx that merely have both classes; results may be unstable.", length(cand)))
      keep <- cand
    }
  }
  
  B1 <- Base_A[,1]; B2 <- Base_A[,2]
  fits <- setNames(vector("list", length(keep)), keep)
  for (dx in keep){
    y <- as.integer(DX[[dx]] > 0); y[is.na(y)] <- 0L
    dat <- data.frame(y=y, b1=B1, b2=B2)
    f <- mgcv::bam(y ~ s(b1, b2, bs="tp", k=60), family=binomial(),
                   data=dat, method="fREML", discrete=TRUE)
    vary <- diff(range(predict(f, type="response"), na.rm=TRUE))
    edf  <- tryCatch(summary(f)$s.table[1,"edf"], error=function(e) NA_real_)
    if (!is.finite(edf) || edf <= 1.05 || vary < 1e-4){
      f <- mgcv::bam(y ~ s(b1, bs="tp", k=40), family=binomial(),
                     data=dat, method="fREML", discrete=TRUE)
    }
    fits[[dx]] <- f
  }
  P <- sapply(keep, function(dx){
    p <- as.numeric(predict(fits[[dx]], newdata = data.frame(b1=B1,b2=B2), type="response"))
    pmin(pmax(p, 1e-6), 1-1e-6)
  })
  storage.mode(P) <- "double"
  list(P = as.matrix(P), dx = colnames(P), fits = fits, Base_A = Base_A)
}

# Graph-regularised NMF
nmf_graph <- function(P, L, K, lambda = 0.1, n_iter = 500, tol = 1e-5, n_start = 2,
                      seed = 42, mass_floor_frac = MASS_FLOOR_FRAC,
                      balance_every = BALANCE_EVERY, reseed_max = RESEED_MAX,
                      entropy_push = ENTROPY_PUSH, eps = EPS){
  set.seed(seed)
  n <- nrow(P); m <- ncol(P)
  L <- as(L, "dgCMatrix")
  S <- -L; diag(S) <- 0
  Dg <- Matrix::Diagonal(n, x = Matrix::rowSums(S))
  best <- NULL; best_obj <- Inf
  for (s in seq_len(n_start)){
    km  <- kmeans(P, centers = K, nstart = 1, iter.max = 30)
    U   <- matrix(eps, n, K); U[cbind(seq_len(n), km$cluster)] <- 1
    U   <- U / pmax(rowSums(U), eps)
    V   <- matrix(runif(m*K, 0, 1), m, K)
    obj_prev <- NA_real_; n_reseed <- 0L
    for (it in seq_len(n_iter)){
      UVt  <- U %*% t(V)
      numV <- t(P) %*% U
      denV <- (t(UVt) %*% U) + eps
      V    <- V * (numV / denV); V[V < eps] <- eps
      VVt  <- t(V) %*% V
      numU <- P %*% V + lambda * (S %*% U)
      denU <- U %*% VVt + lambda * (Dg %*% U) + eps
      U    <- U * (numU / denU); U[U < eps] <- eps
      if (entropy_push > 0) U <- U + entropy_push / K
      if (balance_every > 0L && (it %% balance_every) == 0L){
        col_sums <- colSums(U) + eps
        U <- sweep(U, 2, col_sums, FUN = "/")
      }
      U <- U / pmax(rowSums(U), eps)
      dead <- which(colSums(U) < mass_floor_frac * n)
      if (length(dead) && n_reseed < reseed_max){
        for (k in dead){
          idx <- order(U[,k])[seq_len(min(10L, n))]
          U[idx, k] <- runif(length(idx), 0.5, 1.0)
          V[, k]    <- pmax(runif(m), eps)
        }
        U <- U / pmax(rowSums(U), eps)
        n_reseed <- n_reseed + 1L
      }
      R   <- P - U %*% t(V)
      obj <- sum(R*R)
      gu  <- sum(U * (as.matrix(L %*% U)))
      if (is.finite(gu)) obj <- obj + lambda * gu
      if (!is.finite(obj)) obj <- .Machine$double.xmax/4
      if (is.finite(obj_prev)) {
        delta <- abs(obj_prev - obj) / max(1, abs(obj_prev))
        if (is.finite(delta) && (delta < tol)) break
      }
      obj_prev <- obj
    }
    if (obj < best_obj) { best <- list(U = U, V = V, obj = obj, it = it); best_obj <- obj }
  }
  best
}

# Safe silhouette
silhouette_safe <- function(U, P){
  hard <- max.col(U, ties.method = "first")
  tab  <- table(hard)
  if (length(tab) < 2 || any(tab < 2)) return(NA_real_)
  Dp <- tryCatch(as.dist(as.matrix(dist(P))), error = function(e) NULL)
  if (is.null(Dp)) return(NA_real_)
  out <- tryCatch(cluster::silhouette(hard, Dp), error = function(e) NULL)
  if (is.null(out)) return(NA_real_)
  mean(out[,"sil_width"])
}

# Soft structure metrics
soft_scores <- function(g, U, P){
  w  <- E(g)$weight
  el <- igraph::as_edgelist(g, names = FALSE)
  cs <- rowSums(U[el[,1], , drop = FALSE] * U[el[,2], , drop = FALSE])
  S_nn <- stats::weighted.mean(cs, w)
  deg <- igraph::strength(g, weights = w)
  m2  <- sum(w)
  Q_soft <- (sum(w * cs) - sum(deg[el[,1]] * deg[el[,2]] * mean(rowSums(U)^2) / (2*m2))) / (2*m2)
  sil <- silhouette_safe(U, P)
  data.frame(S_nn = S_nn, Q_soft = Q_soft, S_hard = sil)
}

# --------------------------- Main --------------------------------------------
BA <- if (exists("Base_A")) Base_A else { stop("Base_A not found.") }
DX <- if (exists("DxW_A"))  DxW_A  else { stop("DxW_A not found.") }

dx_schema <- NULL # no longer necessary, was used for matching different dx label encodings

# 1) Build probability matrix
pack <- build_P_from_DxW_A(BA, DX, dx_schema = dx_schema)
P  <- pack$P; dx <- pack$dx; BA <- pack$Base_A

# 2) kNN graph & Laplacian + Louvain K prior
gB <- knn_graph_base(as_base2(BA), k = KNN_BASE, local_scale = LOCAL_SCALE)
cat(sprintf("[kNN-Base] V=%d, E=%d\n", gorder(gB), gsize(gB)))
L  <- laplacian_from_graph(gB, normalized = FALSE)
lvn <- igraph::cluster_louvain(gB, weights = E(gB)$weight)
K0  <- length(sizes(lvn)); K0 <- max(K_MIN, min(K_MAX, K0))
K_GRID <- sort(unique(pmax(K_MIN, pmin(K_MAX, c(K0-1, K0, K0+1, K0+2)))))
cat(sprintf("[Louvain prior] K0=%d -> K_GRID={%s}\n", K0, paste(K_GRID, collapse=", ")))

# 3) Use the DXs from dimension_clusters.R for comparability
leiden_df <- NULL
if (isTRUE(USE_LEIDEN_K) && file.exists(LEIDEN_MEMBERSHIP_CSV)) {
  leiden_df <- readr::read_csv(LEIDEN_MEMBERSHIP_CSV, show_col_types = FALSE) |>
    dplyr::select(participant_id, cluster) |>
    dplyr::mutate(cluster = as.integer(cluster))
  cat(sprintf("[Leiden import] read %d clusters but NOT forcing K\n",
              dplyr::n_distinct(leiden_df$cluster)))
}

# 4) Sweep K × lambda
grid <- expand.grid(K = K_GRID, lambda = LAMBDA_GRID, stringsAsFactors = FALSE)
res  <- vector("list", nrow(grid))
arch_rows <- list()
best_all <- list(score = -Inf)
.mass_str <- function(v) paste0(format(round(v, 1), trim=TRUE, nsmall=1), collapse = "/")
ids_here <- get_base_ids() %||% rownames(BA) %||% as.character(seq_len(nrow(BA)))

for (i in seq_len(nrow(grid))){
  K   <- grid$K[i]; lam <- grid$lambda[i]
  fit <- nmf_graph(P, L, K = K, lambda = lam, n_iter = N_ITER, tol = TOL,
                   n_start = N_START, seed = 42 + i)
  U <- as.matrix(fit$U); V <- as.matrix(fit$V)

  sc   <- soft_scores(gB, U, P)
  den  <- mean((P - mean(P))^2)
  recon <- if (den > 0) 1 - mean((P - U %*% t(V))^2) / den else NA_real_
  soft_mass <- colSums(U)
  effK      <- sum(soft_mass > MASS_FLOOR_FRAC * nrow(U))
  hard_lab  <- max.col(U, ties.method = "first")
  hard_tab  <- tabulate(hard_lab, nbins = K)
  imb       <- (max(soft_mass) / mean(soft_mass)) - 1

  # redundancy penalty (discourage duplicate factors)
  Vn   <- sweep(V, 2, sqrt(colSums(V*V)) + EPS, "/")
  Rv   <- crossprod(Vn)
  over <- mean(Rv[upper.tri(Rv)]^2, na.rm = TRUE)

  # gentle prior centered at Louvain K0
  prior_K <- exp(-0.5 * ((K - K0)/1.2)^2)
  Ssil    <- if (is.finite(sc$S_hard)) sc$S_hard else 0

  score_core <- W_Q*sc$Q_soft + W_SNN*sc$S_nn + W_REC*recon + W_SIL*Ssil
  score <- score_core * (1 + 0.05*prior_K) -
           PEN_RED * over -
           PEN_EMPTY * max(0, K_MIN - effK) / K -
           PEN_IMB   * ((max(soft_mass)/mean(soft_mass)) - 1)

  if (effK < max(2L, ceiling(0.6*K))) score <- score - 1e3  # hard reject splitty fits

  res[[i]] <- cbind(
    data.frame(K=K, lambda=lam, recon=recon, effK=effK, imb=imb,
               iters=fit$it, obj=fit$obj, score=score, V_redundancy=over),
    sc
  )

  Vdf <- as.data.frame(V); if (is.null(colnames(Vdf))) colnames(Vdf) <- paste0("V", seq_len(ncol(Vdf)))
  Vdf$dx <- dx
  arch_rows[[i]] <- tidyr::pivot_longer(Vdf, -dx, names_to="cluster", values_to="loading") |>
    dplyr::mutate(cluster = paste0("C", readr::parse_integer(gsub("^V","", cluster))),
                  K = K, lambda = lam)

  cat(sprintf("[NMF] K=%d λ=%.2f | recon=%.3f | S_nn=%.3f | Q_soft=%.3f | S_hard=%s | effK=%d | Vred=%.3f | soft_mass=%s | hard_n=%s | score=%.3f\n",
              K, lam, recon, sc$S_nn, sc$Q_soft,
              ifelse(is.na(sc$S_hard), "NA", sprintf("%.3f", sc$S_hard)),
              effK, over, .mass_str(soft_mass), .mass_str(hard_tab), score))

  if (score > best_all$score)
    best_all <- list(K=K, lambda=lam, U=U, V=V, metrics=res[[i]], score=score)
}

metrics <- dplyr::bind_rows(res)
readr::write_csv(metrics, "nmf_sweep_metrics.csv")
arche <- dplyr::bind_rows(arch_rows)
readr::write_csv(arche, "cluster_archetypes_dx_by_k.csv")

cat(sprintf("[BEST] K=%d, λ=%.2f | recon=%.3f | S_nn=%.3f | Q_soft=%.3f | S_hard=%s | Vred=%.3f | score=%.3f\n",
            best_all$K, best_all$lambda, best_all$metrics$recon,
            best_all$metrics$S_nn, best_all$metrics$Q_soft,
            ifelse(is.na(best_all$metrics$S_hard),"NA",sprintf("%.3f",best_all$metrics$S_hard)),
            best_all$metrics$V_redundancy, best_all$score))

# (Postcheck only; no refit)
if (!is.null(best_all$V)) {
  Vn_best <- sweep(best_all$V, 2, sqrt(colSums(best_all$V^2))+EPS, "/")
  over_b  <- mean(crossprod(Vn_best)[upper.tri(diag(ncol(Vn_best)))]^2, na.rm=TRUE)
  if (over_b > 0.75 && best_all$K > 3) {
    cat(sprintf("[postcheck] V columns quite redundant (%.2f). Consider K in {%d,%d}.\n",
                over_b, best_all$K-2, best_all$K-1))
  }
}

# 4) Save memberships
U_best <- best_all$U
U_best <- U_best / pmax(rowSums(U_best), EPS)
colnames(U_best) <- paste0("C", seq_len(ncol(U_best)))
ids <- get_base_ids() %||% rownames(BA) %||% seq_len(nrow(BA))
mem_df <- cbind(participant_id = ids, as.data.frame(U_best, check.names = FALSE))
mem_df$K_best <- best_all$K; mem_df$lambda_best <- best_all$lambda
readr::write_csv(mem_df, "soft_memberships.csv")

# ---------------- Align to Leiden (if available) ------------------------------
# Pre-requirements: mem_df has participant_id + C1..CK; best_all$V exists.
cluster_cols <- grep("^C\\d+$", names(mem_df), value = TRUE)

if (!is.null(leiden_df)) {
  # hard Base labels from U_best (pre-normalised choice is equivalent after row-normalisation)
  hard_base <- max.col(as.matrix(mem_df[, cluster_cols, drop = FALSE]), ties.method = "first")
  base_lab_df <- tibble::tibble(participant_id = mem_df$participant_id,
                                base = hard_base)
  
  X <- dplyr::inner_join(base_lab_df,
                         dplyr::rename(leiden_df, leiden = cluster),
                         by = "participant_id")
  
  if (nrow(X) > 0 && length(unique(X$leiden)) >= 2L) {
    tab <- table(X$base, X$leiden)              # rows: Base components, cols: Leiden ids
    A <- as.matrix(tab)
    cost <- max(A) - A             # non-negative, larger overlap -> smaller cost
    assign <- clue::solve_LSAP(cost)
    base_lvls   <- as.integer(rownames(tab))
    leiden_lvls <- as.integer(colnames(tab))
    # Map: Base component i -> Leiden label j
    map <- setNames(leiden_lvls[as.vector(assign)], base_lvls)
    
    # Reorder U_best/V to Leiden order and rename to L{j}
    perm <- match(sort(unique(unname(map))), unname(map))          # indices in Base -> Leiden order
    U_aligned <- as.matrix(mem_df[, cluster_cols, drop = FALSE])
    U_aligned <- U_aligned[, perm, drop = FALSE]
    new_names <- paste0(LEIDEN_LABEL_PREFIX, sort(unique(unname(map))))
    colnames(U_aligned) <- new_names
    
    # Replace mem_df soft columns
    mem_df <- dplyr::bind_cols(
      tibble::tibble(participant_id = mem_df$participant_id),
      as.data.frame(U_aligned, check.names = FALSE)
    ) |>
      dplyr::mutate(K_best = best_all$K, lambda_best = best_all$lambda)
    
    # Align V (dx loadings) the same way and rename columns
    V_aligned <- best_all$V[, perm, drop = FALSE]
    colnames(V_aligned) <- new_names
    best_all$V <- V_aligned
    
    # Persist aligned files + mapping
    map_df <- tibble::tibble(
      base_component = paste0("C", base_lvls),
      leiden_cluster = paste0(LEIDEN_LABEL_PREFIX, leiden_lvls[as.vector(assign)])
    )
    readr::write_csv(map_df, "base_to_leiden_mapping.csv")
    readr::write_csv(mem_df, "soft_memberships_aligned_to_leiden.csv")
    
    cat("[align] Base components relabelled to Leiden as:\n"); print(map_df)
  } else {
    cat("[align] Skipped: no overlap rows or <2 Leiden clusters present.\n")
  }
}

if (!is.null(leiden_df)) {
  # Recompute hard aligned labels from the current mem_df column order
  hard_now <- max.col(as.matrix(mem_df[, grep(paste0("^(", LEIDEN_LABEL_PREFIX, "|C)\\d+$"),
                                              names(mem_df), value = TRUE), drop = FALSE]),
                      ties.method = "first")
  lab_now <- tibble::tibble(participant_id = mem_df$participant_id, base_now = hard_now)
  Y <- dplyr::inner_join(lab_now, dplyr::rename(leiden_df, leiden = cluster), by = "participant_id")
  cont <- as.data.frame.matrix(table(Y$base_now, Y$leiden))
  readr::write_csv(tibble::rownames_to_column(cont, "base_component"),
                   "contingency_base_vs_leiden.csv")
}

# 5) Visualisations ------------------------------------------------------------
UD <- {
  Xstd <- standardise_to_circle(BA, cover = 0.995)
  Uxy  <- as.data.frame(Xstd$fwd(BA[,1], BA[,2])); names(Uxy) <- c("u1","u2")
  u1 <- seq(-1, 1, length.out = 550); u2 <- seq(-1, 1, length.out = 550)
  grid <- expand.grid(u1 = u1, u2 = u2)
  mask <- with(grid, sqrt(u1^2 + u2^2) <= 1 + 1e-9)
  list(Xstd=Xstd, Uxy=Uxy, grid=grid, mask=mask)
}

cluster_names <- grep(paste0("^(", LEIDEN_LABEL_PREFIX, "|C)\\d+$"), names(mem_df), value = TRUE)
Pi_long <- mem_df %>%
  dplyr::mutate(u1 = UD$Uxy$u1, u2 = UD$Uxy$u2) %>%
  tidyr::pivot_longer(all_of(cluster_names), names_to = "cluster", values_to = "pi") %>%
  dplyr::mutate(pi_clip = pmin(pmax(pi, 1e-6), 1-1e-6),
                logit_pi = qlogis(pi_clip))

clusters <- cluster_names

# fit once per cluster in (u1,u2)
fits_u <- vector("list", length(clusters)); names(fits_u) <- clusters
for (cl in clusters){
  dfc <- Pi_long[Pi_long$cluster == cl, c("logit_pi","u1","u2")]
  f <- try(
    mgcv::bam(logit_pi ~ s(u1, u2, bs = "tp", k = 80),
              data = dfc, family = gaussian(), method = "fREML", discrete = TRUE),
    silent = TRUE
  )
  if (inherits(f, "try-error")) f <- NULL
  fits_u[[cl]] <- f
}

# --- UNIT DISK (unmasked) ---
Fstd_cl <- dplyr::bind_rows(lapply(clusters, function(cl){
  f <- fits_u[[cl]]
  if (is.null(f))
    return(data.frame(cluster = cl, u1 = UD$grid$u1, u2 = UD$grid$u2, p = NA_real_))
  eta <- as.numeric(predict(f, newdata = UD$grid, type = "link"))
  p   <- plogis(eta); p[!UD$mask] <- NA
  data.frame(cluster = cl, u1 = UD$grid$u1, u2 = UD$grid$u2, p = p)
}))

# --- UNIT SQUARE (unmasked) ---
Fsq_cl <- dplyr::bind_rows(lapply(clusters, function(cl){
  f <- fits_u[[cl]]
  if (is.null(f))
    return(data.frame(cluster = cl, u1 = UD$grid$u1, u2 = UD$grid$u2, p = NA_real_))
  eta <- as.numeric(predict(f, newdata = UD$grid, type = "link"))
  p   <- plogis(eta)
  data.frame(cluster = cl, u1 = UD$grid$u1, u2 = UD$grid$u2, p = p)
}))

# --- BASE CONVEX HULL ---
B1 <- BA[,1]; B2 <- BA[,2]
hidx  <- grDevices::chull(B1, B2)
hpoly <- data.frame(b1 = B1[hidx], b2 = B2[hidx])

n_base_grid <- 150L
qx <- range(B1); qy <- range(B2)
gridB_full <- expand.grid(
  b1 = seq(qx[1], qx[2], length.out = n_base_grid),
  b2 = seq(qy[1], qy[2], length.out = n_base_grid)
)
mask_hull <- inside_hull(gridB_full$b1, gridB_full$b2, hpoly)

# map base grid -> (u1,u2), predict with the same fits
Ub <- as.data.frame(UD$Xstd$fwd(gridB_full$b1, gridB_full$b2)); names(Ub) <- c("u1","u2")

Fbase_cl <- dplyr::bind_rows(lapply(clusters, function(cl){
  f <- fits_u[[cl]]
  if (is.null(f))
    return(data.frame(cluster = cl, b1 = gridB_full$b1, b2 = gridB_full$b2, p = NA_real_))
  eta <- as.numeric(predict(f, newdata = Ub, type = "link"))
  p   <- plogis(eta); p[!mask_hull] <- NA
  data.frame(cluster = cl, b1 = gridB_full$b1, b2 = gridB_full$b2, p = p)
}))

# ---- Soft cluster fields (UNIT DISK) -----------------------------------------
br_fixed <- c(.15,.25,.40,.60,.80)

p_fields <- ggplot(Fstd_cl, aes(u1, u2)) +
  geom_raster(aes(fill = p), interpolate = TRUE) +
  geom_contour(aes(z = p), breaks = br_fixed, colour = "white",
               linewidth = 0.28, alpha = 0.95, na.rm = TRUE) +
  geom_point(data = UD$Uxy, aes(u1, u2), inherit.aes = FALSE,
             shape = 20, size = 0.28, colour = scales::alpha("black", 0.35)) +
  coord_equal(xlim = c(-1,1), ylim = c(-1,1), expand = FALSE) +
  facet_wrap(~ cluster, ncol = 4, scales = "fixed") +
  scale_fill_psych(lims = c(0,1), name = "π", percent_labels = TRUE) +
  labs(
    title = sprintf("Soft cluster fields (K=%d, λ=%.2f)", best_all$K, best_all$lambda),
    x = "u1 (whitened b1,b2)",
    y = "u2"
  ) +
  psych_theme(15)

save_png("FIG_softcluster_fields_Kbest.png", p_fields, width = 8.0, height = 6.5, dpi = 300)

# Square [-1,1]^2
p_fields_sq <- ggplot(Fsq_cl, aes(u1, u2)) +
  geom_raster(aes(fill = p), interpolate = TRUE) +
  geom_contour(aes(z = p), breaks = br_fixed, colour = "white",
               linewidth = 0.28, alpha = 0.95, na.rm = TRUE) +
  geom_point(data = UD$Uxy, aes(u1, u2), inherit.aes = FALSE,
             shape = 20, size = 0.28, colour = scales::alpha("black", 0.35)) +
  coord_equal(xlim = c(-1, 1), ylim = c(-1, 1), expand = FALSE) +
  facet_wrap(~ cluster, ncol = 4, scales = "fixed") +
  scale_fill_psych(lims = c(0,1), name = "π", percent_labels = TRUE) +
  labs(
    title = sprintf("Soft cluster fields — unit square (K=%d, λ=%.2f)", best_all$K, best_all$lambda),
    x = "u1 (whitened b1,b2)", y = "u2"
  ) +
  psych_theme(15)

save_png("FIG_softcluster_fields_UNITSQUARE.png", p_fields_sq, width = 8.0, height = 6.5, dpi = 300)

# Base convex hull
df_base <- data.frame(b1 = B1, b2 = B2)
p_fields_base <- ggplot(Fbase_cl, aes(b1, b2)) +
  geom_raster(aes(fill = p), interpolate = TRUE, na.rm = TRUE) +
  geom_point(data = df_base, aes(b1, b2), inherit.aes = FALSE,
             shape = 20, size = 0.26, colour = scales::alpha("black", 0.35)) +
  geom_contour(aes(z = p), breaks = br_fixed, colour = "white",
               linewidth = 0.28, alpha = 0.95, na.rm = TRUE) +
  coord_equal() +
  facet_wrap(~ cluster, ncol = 4, scales = "fixed") +
  scale_fill_psych(lims = c(0,1), name = "π", percent_labels = TRUE) +
  labs(
    title = sprintf("Soft cluster fields — base convex hull (K=%d, λ=%.2f)", best_all$K, best_all$lambda),
    x = "b1", y = "b2"
  ) +
  psych_theme(15)

save_png("FIG_softcluster_fields_BASEHULL.png", p_fields_base, width = 8.0, height = 6.5, dpi = 300)

# ---- Diagnosis archetypes heatmap --------------------------------------------
# keep raw order from pack$dx (unique by construction)
levels_dx <- unique(dx)

H <- as.data.frame(Vb) |>
  tibble::rownames_to_column("dx") |>
  tidyr::pivot_longer(-dx, names_to = "cluster", values_to = "loading") |>
  dplyr::mutate(dx = factor(dx, levels = .env$levels_dx))  # note .env

# wrap only for display
labs_wrapped <- setNames(wrap_dx(levels_dx), levels_dx)

p_heat <- ggplot(H, aes(cluster, dx, fill = loading)) +
  geom_tile() +
  scale_fill_psych(name = "loading", percent_labels = FALSE) +
  scale_y_discrete(labels = labs_wrapped) +
  labs(x = "cluster", y = NULL, title = "Diagnosis archetypes per soft cluster") +
  psych_theme(15) +
  theme(axis.text.y = element_text(lineheight = 0.9),
        panel.grid = element_blank())

save_png("FIG_cluster_by_dx_heatmap_Kbest.png", p_heat, width = 9.0, height = 7.0, dpi = 300)

print(p_fields); print(p_fields_sq); print(p_fields_base); print(p_heat)
