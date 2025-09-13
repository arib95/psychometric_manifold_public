# phate_sweep.R

# =================================================================================================
# M) PHATE sweep (nd = 1..5) + cross-validated manifold fits
# =================================================================================================
# Goal:
#   - Compute a single PHATE embedding with 5 components (once), then evaluate nd = 1..5 views.
#   - For nd >= 3, fit a principal (nd-1)D hypersurface with K-fold CV using tensor-product splines.
#   - Record baseline linear (PCA) R2, CV R2-like, orthogonal/Z-score R2, curvature proxy (nd=3),
#     and save residuals/gradient fields.
# Design notes:
#   - We *do not* loop PHATE per nd — we run it ONCE at n_components = 5 and reuse prefixes.
#   - We cache to phate_embeddings_wide.csv; if present and compatible, we reuse it.
#   - The manifold fitting uses r2_manifold_te() defined in Section “Utility functions”.
#   - Curvature metrics are computed only when m = nd-1 = 2 (i.e., nd == 3), where a 2D parameter
#     grid is well-defined.
#   - Runtime guards: modest K for CV; low basis sizes to avoid overfitting.
# =================================================================================================
# 6) Run a PHATE dimensional sweep (1..5) and, for each embedding, fit a principal manifold:
#        - nd=2: principal curve in R^2
#        - nd>=3: tensor thin-plate spline hypersurface parameterised by the first m=nd-1 PCs
#       Report cross-validated R2-like, linear baseline R2, curvature proxy, and save residuals.
#
# Author: Afonso Dinis Ribeiro
# Date:   31-08-2025
# =================================================================================================

suppressPackageStartupMessages({
  library(mgcv); library(readr)
})

# ---- PHATE sweep (Section M) ----
PHATE_CONDA_ENV <- Sys.getenv("PHATE_CONDA_ENV", "r-phate")
PHATE_K_BASIS   <- 4        # tensor-product low basis
PHATE_K_BASIS_HI<- 6        # hi/backup basis
PHATE_K_EXTRA   <- 3        # extra axis basis for nd>=4
PHATE_KCV_ND5   <- 2        # CV folds when nd = 5
PHATE_KCV_ELSE  <- 3        # CV folds when nd <= 4

# ---------- internal helpers (PHATE only) ----------
r2_tot <- function(Y) sum(scale(Y, scale = FALSE)^2)

r2_linear_baseline_m <- function(X, m){
  m <- min(m, ncol(X))
  pc   <- prcomp(X, rank. = m, center = TRUE, scale. = FALSE)
  Xhat <- pc$x %*% t(pc$rotation)
  Xhat <- sweep(Xhat, 2, pc$center, "+")
  1 - sum((X - Xhat)^2) / sum((X - colMeans(X))^2)
}

r2_curve_pc <- function(Y){
  pc <- princurve::principal_curve(as.matrix(Y), stretch = 0)
  1 - sum((Y - pc$s)^2) / r2_tot(Y)
}

r2_surface_orth <- function(Y, Yhat){
  1 - sum(rowSums((Y - Yhat)^2)) / r2_tot(Y)
}
r2_zscore <- function(Y, Yhat){
  Yz <- scale(Y, TRUE, TRUE)
  Yhatz <- scale(Yhat, center = attr(Yz,"scaled:center"), scale = attr(Yz,"scaled:scale"))
  1 - sum((Yz - Yhatz)^2) / sum(Yz^2)
}
residual_summary <- function(Y, Yhat){
  d_res <- sqrt(rowSums((Y - Yhat)^2))
  list(med_resid = median(d_res), mean_resid = mean(d_res), med_pair = median(as.numeric(dist(Y))))
}

safe_bam <- function(formula, data, k_threads = 1, gamma = 1.0, select = TRUE){
  stopifnot(all(vapply(data, function(z) all(is.finite(z)), logical(1))))
  fit <- try(mgcv::bam(
    formula, data = data,
    method = "fREML", discrete = FALSE,
    nthreads = k_threads, select = select, gamma = gamma,
    optimizer = c("outer","bfgs"),
    control = mgcv::gam.control(maxit = 100)
  ), silent = TRUE)
  if (!inherits(fit, "try-error")) return(fit)
  mgcv::gam(formula, data = data, method = "REML", select = select, gamma = gamma)
}


# Utility: principal (m = nd-1) manifold via tensor-product GAMs + scoring helpers
# r2_manifold_te(Y, m, ...)
#   Y : n x d embedding matrix (rows = subjects, cols = coordinates to be modelled)
#   m : intrinsic dimension of the parameter space (usually nd-1)
#
# Modes:
#   - mode="apparent": fit on all rows, score on all rows (returns Yhat, U, and models if requested)
#   - mode="cv":       K-fold CV with TRAIN-only PCA for U; score on TEST (no models returned)
#
# PCA parameterisation:
#   - U = PC scores of Y with rank m
#   - In CV mode, PCA is fit on TRAIN only; TEST is projected into TRAIN PCs (no leakage)
#
# Smooth:
#   - m==1: s(u1, bs="tp", k=k_basis)
#   - m>=2: te(u1, u2[, ..., um], bs="tp", k=c(k_basis, k_basis_hi, rep(k_extra, m-2)))
#
# Returns:
#   mode="apparent": list(R2_like, R2_adj, GCV, Yhat, U, models)
#   mode="cv":       named numeric vector c(R2_like=..., R2_adj=NA, GCV=NA)
r2_manifold_te <- function(
    Y, m, mode = c("cv","apparent"), K = 3,
    k_basis = 4, k_basis_hi = 6, k_extra = 3,
    soft_extra = FALSE, gamma = 1.0, select_smooth = TRUE,
    seed = 1, return_models = FALSE
){
  mode <- match.arg(mode)
  Y <- as.matrix(Y); storage.mode(Y) <- "double"
  n <- nrow(Y); d <- ncol(Y); m <- min(m, d, n - 1L)
  
  .build_formula <- function(m, k1, k2, kx){
    if (m == 1L) sprintf("y ~ s(u1, bs='tp', k=%d)", k1) else {
      kvec <- c(k1, k2, rep(kx, max(0L, m - 2L)))
      uv   <- paste0("u", 1:m, collapse = ", ")
      sprintf("y ~ te(%s, bs='tp', k=c(%s))", uv, paste(kvec, collapse = ","))
    }
  }
  .fit_one <- function(y, Udf, k1, k2, kx, gamma, select, soft_extra){
    form <- as.formula(.build_formula(ncol(Udf), k1, k2, kx))
    fit  <- safe_bam(form, data = cbind(y = as.numeric(y), Udf),
                     k_threads = 1, gamma = gamma, select = if (soft_extra) TRUE else select)
    yhat <- as.numeric(stats::predict(fit, newdata = Udf))
    list(fit = fit, yhat = yhat)
  }
  
  if (mode == "apparent"){
    pc  <- prcomp(Y, center = TRUE, scale. = FALSE, rank. = m)
    U   <- pc$x[, 1:m, drop = FALSE]; colnames(U) <- paste0("u", 1:ncol(U))
    Udf <- as.data.frame(U)
    fits <- vector("list", d); Yhat <- matrix(NA_real_, n, d); r2s <- gcv <- rep(NA_real_, d)
    for (j in seq_len(d)){
      fr <- .fit_one(Y[, j], Udf, k_basis, k_basis_hi, k_extra, gamma, select_smooth, soft_extra)
      fits[[j]] <- fr$fit; Yhat[, j] <- fr$yhat
      sm <- try(suppressWarnings(summary(fr$fit)), silent = TRUE)
      r2s[j] <- if (!inherits(sm, "try-error") && !is.null(sm$r.sq)) sm$r.sq else NA_real_
      gcv[j] <- suppressWarnings(if (!is.null(fr$fit$gcv.ubre)) fr$fit$gcv.ubre else NA_real_)
    }
    R2_like <- 1 - sum((Y - Yhat)^2) / r2_tot(Y)
    out <- list(R2_like = as.numeric(R2_like),
                R2_adj  = if (all(is.na(r2s))) NA_real_ else mean(r2s, na.rm = TRUE),
                GCV     = if (all(is.na(gcv))) NA_real_ else mean(gcv, na.rm = TRUE),
                Yhat    = Yhat, U = U)
    if (isTRUE(return_models)) out$models <- fits
    return(out)
  }
  
  set.seed(seed)
  fold <- sample(rep(seq_len(K), length.out = n))
  rss  <- 0; tss <- r2_tot(Y)
  for (k in seq_len(K)){
    tr <- which(fold != k); te <- which(fold == k)
    if (length(tr) < (m + 2L) || length(te) < 2L) next
    pc_tr <- prcomp(Y[tr, , drop = FALSE], center = TRUE, scale. = FALSE, rank. = m)
    Utr   <- pc_tr$x[, 1:m, drop = FALSE]; colnames(Utr) <- paste0("u", 1:ncol(Utr))
    Ute   <- scale(Y[te, , drop = FALSE], center = pc_tr$center, scale = FALSE) %*%
      pc_tr$rotation[, 1:m, drop = FALSE]
    colnames(Ute) <- paste0("u", 1:ncol(Ute))
    Utr <- as.data.frame(Utr); Ute <- as.data.frame(Ute)
    Ypred <- matrix(NA_real_, nrow = length(te), ncol = d)
    for (j in seq_len(d)){
      fr <- .fit_one(Y[tr, j], Utr, k_basis, k_basis_hi, k_extra, gamma, select_smooth, soft_extra)
      Ypred[, j] <- as.numeric(stats::predict(fr$fit, newdata = Ute))
    }
    rss <- rss + sum((Y[te, , drop = FALSE] - Ypred)^2)
  }
  c(R2_like = as.numeric(1 - rss / tss), R2_adj = NA_real_, GCV = NA_real_)
}

# =================================================================================================
# Helper: gradient field on (u,v) parameter grid for a list of GAMs (nd=3 case)
# -------------------------------------------------------------------------------------------------
# gradient_field_uv(models, U, ngr=80, h_frac=1e-3)
#   models : list of GAMs for Y1..Yd with formula in te(u,v, ...)
#   U      : n x 2 matrix/data.frame of (u,v) used to derive grid bounds
# Returns a data.frame with u, v, du_norm, dv_norm, jac_frob
# =================================================================================================
gradient_field_uv <- function(models, U, ngr = 80, h_frac = 1e-3){
  U <- as.data.frame(U); names(U)[1:2] <- c("u","v")
  ug <- seq(min(U$u), max(U$u), length.out = ngr)
  vg <- seq(min(U$v), max(U$v), length.out = ngr)
  G  <- expand.grid(u = ug, v = vg)
  hu <- max(1e-9, diff(range(U$u)) * h_frac)
  hv <- max(1e-9, diff(range(U$v)) * h_frac)
  P0 <- sapply(models, function(m) as.numeric(predict(m, newdata = G)))
  Pu <- sapply(models, function(m) as.numeric(predict(m, newdata = transform(G, u = u + hu))))
  Pv <- sapply(models, function(m) as.numeric(predict(m, newdata = transform(G, v = v + hv))))
  dYu <- (Pu - P0) / hu; dYv <- (Pv - P0) / hv
  du_norm  <- sqrt(rowSums(dYu^2)); dv_norm <- sqrt(rowSums(dYv^2))
  jac_frob <- sqrt(du_norm^2 + dv_norm^2)
  data.frame(u = G$u, v = G$v, du_norm = du_norm, dv_norm = dv_norm, jac_frob = jac_frob)
}

curvature_proxy <- function(U, Y, k = 6, gamma = 1.0, ngr = 60){
  stopifnot(ncol(U) >= 2)
  df <- data.frame(u = U[,1], v = U[,2])
  ug <- seq(min(U[,1]), max(U[,1]), length.out = ngr)
  vg <- seq(min(U[,2]), max(U[,2]), length.out = ngr)
  G  <- expand.grid(u = ug, v = vg)
  pred_mat <- sapply(seq_len(ncol(Y)), function(j){
    as.numeric(predict(
      mgcv::gam(Y[, j] ~ te(u, v, bs = "tp", k = c(k, k)),
                data = df, method = "REML", select = TRUE, gamma = gamma),
      G))
  })
  mats <- lapply(seq_len(ncol(Y)), function(j) matrix(pred_mat[, j], nrow = ngr, ncol = ngr, byrow = FALSE))
  hess_rms <- function(Zm){
    ngr <- nrow(Zm)
    Zu_p <- Zm[3:ngr, 2:(ngr-1)]; Zu_0 <- Zm[2:(ngr-1), 2:(ngr-1)]; Zu_m <- Zm[1:(ngr-2), 2:(ngr-1)]
    d2u  <- Zu_p - 2*Zu_0 + Zu_m
    Zv_p <- Zm[2:(ngr-1), 3:ngr];       Zv_0 <- Zm[2:(ngr-1), 2:(ngr-1)]; Zv_m <- Zm[2:(ngr-1), 1:(ngr-2)]
    d2v  <- Zv_p - 2*Zv_0 + Zv_m
    sqrt(d2u^2 + d2v^2)
  }
  Hsum <- Reduce("+", lapply(mats, hess_rms))
  c(curv_median = stats::median(Hsum), curv_p95 = as.numeric(stats::quantile(Hsum, 0.95)))
}

# ---------- PHATE CLI wrapper ----------
run_phate_cli <- function(Xnum, ndim = 5, env = "r-phate"){
  # Xnum must be numeric matrix, rows aligned to ids, scaled beforehand
  # This function assumes you have a small CLI helper `phate_cli.py` on PATH or in project.
  # If you already have one, keep it. If not, adapt to your wrapper.
  stop("run_phate_cli is a placeholder. Point this to your existing PHATE CLI wrapper.")
}

# ---------- public entry point ----------
run_phate_sweep <- function(
    X, ids,
    cache_path = "phate_embeddings_wide.csv",
    conda_env = "r-phate",
    kcv_nd5 = 2, kcv_else = 3,
    k_basis = 4, k_basis_hi = 6, k_extra = 3,
    soft_extra_nd = 4
){
  # 1) numeric-coded input
  ph_in <- as.data.frame(lapply(X, function(v) if (is.factor(v)) as.numeric(v) else as.numeric(v)))
  keep <- vapply(ph_in, function(col) stats::sd(col, na.rm = TRUE) > 0, logical(1))
  ph_in <- ph_in[, keep, drop = FALSE]
  stopifnot(ncol(ph_in) >= 1)
  ph_mat <- scale(as.matrix(ph_in)); rownames(ph_mat) <- ids
  
  # 2) load or compute PHATE with n_components=5 once
  phate_store <- vector("list", 5); need_run <- TRUE
  if (file.exists(cache_path)) {
    phw <- try(suppressWarnings(readr::read_csv(cache_path, show_col_types = FALSE)), silent = TRUE)
    if (!inherits(phw, "try-error") && "participant_id" %in% names(phw)) {
      rowmap <- match(ids, phw$participant_id)
      has5 <- all(paste0("PHATE", 1:5, "_nd5") %in% names(phw))
      if (all(is.finite(rowmap)) && has5) {
        for (nd in 1:5) {
          cols <- paste0("PHATE", 1:nd, "_nd", nd)
          if (all(cols %in% names(phw))) {
            emb <- as.matrix(phw[rowmap, cols, drop = FALSE])
            rownames(emb) <- ids; phate_store[[nd]] <- emb
          }
        }
        if (all(vapply(phate_store, function(e) is.matrix(e) && nrow(e) == length(ids), logical(1))))
          need_run <- FALSE
      }
    }
  }
  if (need_run) {
    message("[PHATE] computing n_components=5 via conda env: ", conda_env)
    ph_all <- run_phate_cli(ph_mat, ndim = 5, env = conda_env)
    if (!is.matrix(ph_all) || nrow(ph_all) != length(ids)) stop("PHATE run failed.")
    rownames(ph_all) <- ids
    for (nd in 1:5) phate_store[[nd]] <- ph_all[, seq_len(nd), drop = FALSE]
    ph_wide <- data.frame(participant_id = ids)
    for (nd in 1:5) {
      df_nd <- as.data.frame(phate_store[[nd]])
      colnames(df_nd) <- paste0("PHATE", seq_len(ncol(df_nd)), "_nd", nd)
      ph_wide <- cbind(ph_wide, df_nd)
    }
    readr::write_csv(ph_wide, cache_path)
  }
  
  # 3) metrics per nd and outputs identical to previous behaviour
  mk_row <- function(nd, method,
                     R2_like = NA_real_, R2_adj = NA_real_, GCV = NA_real_,
                     R2_linear = NA_real_, R2_z = NA_real_, R2_orth = NA_real_,
                     Delta_R2 = NA_real_, Curv_med = NA_real_, Curv_p95 = NA_real_,
                     TwoNN_ID = NA_real_){
    data.frame(ndim = nd, method = method, R2_like = R2_like, R2_adj = R2_adj, GCV = GCV,
               R2_linear = R2_linear, R2_z = R2_z, R2_orth = R2_orth, Delta_R2 = Delta_R2,
               Curv_med = Curv_med, Curv_p95 = Curv_p95, TwoNN_ID = TwoNN_ID)
  }
  
  fit_summary <- data.frame()
  
  for (nd in 1:5){
    emb <- phate_store[[nd]]
    stopifnot(is.matrix(emb) && nrow(emb) == length(ids))
    id_ph <- if (ncol(emb) >= 2) suppressWarnings({
      nn <- RANN::nn2(scale(emb), k = 3)$nn.dists
      r <- nn[,3] / nn[,2]; r <- r[is.finite(r) & r > 0]; 1 / mean(log(r))
    }) else 1.0
    
    if (nd == 1){
      fit_summary <- rbind(fit_summary, mk_row(nd, "line_1D", TwoNN_ID = id_ph))
      next
    }
    if (nd == 2){
      base <- r2_linear_baseline_m(emb, m = 2)
      r2c  <- r2_curve_pc(emb)
      fit_summary <- rbind(fit_summary,
                           mk_row(nd, "principal_curve_2D",
                                  R2_like = r2c, R2_linear = base, TwoNN_ID = id_ph))
      next
    }
    
    m <- nd - 1
    base <- r2_linear_baseline_m(emb, m = m)
    use_soft <- (nd == soft_extra_nd)
    Kcv <- if (nd >= 5) kcv_nd5 else kcv_else
    
    res_cv <- r2_manifold_te(emb, m, mode = "cv", K = Kcv,
                             k_basis = k_basis, k_basis_hi = k_basis_hi, k_extra = k_extra,
                             soft_extra = use_soft, gamma = 1.0, select_smooth = use_soft, seed = 1)
    app <- r2_manifold_te(emb, m, mode = "apparent",
                          k_basis = k_basis, k_basis_hi = k_basis_hi, k_extra = k_extra,
                          soft_extra = use_soft, gamma = 1.0, select_smooth = use_soft,
                          seed = 1, return_models = TRUE)
    Yhat_app <- app$Yhat; U_app <- app$U
    r2_orth <- tryCatch(r2_surface_orth(emb, Yhat_app), error = function(e) NA_real_)
    r2_z    <- tryCatch(r2_zscore(emb, Yhat_app),      error = function(e) NA_real_)
    deltaR2 <- unname(res_cv["R2_like"] - base)
    
    # residuals export
    write.csv(
      data.frame(
        row_id = rownames(emb), nd = nd, m = m,
        u = if (is.matrix(U_app) && ncol(U_app) >= 1) U_app[,1] else NA_real_,
        v = if (is.matrix(U_app) && ncol(U_app) >= 2) U_app[,2] else NA_real_,
        resid = sqrt(rowSums((emb - Yhat_app)^2))
      ),
      sprintf("phate_nd%d_surface_residuals.csv", nd), row.names = FALSE
    )
    
    # gradients and curvature only when m == 2
    curv_median <- NA_real_; curv_p95 <- NA_real_
    if (m == 2 && !is.null(app$models)){
      gf <- tryCatch(gradient_field_uv(app$models, U_app[,1:2, drop = FALSE], ngr = 80, h_frac = 1e-3),
                     error = function(e) NULL)
      if (!is.null(gf)) write.csv(gf, sprintf("phate_nd%d_surface_gradient_field.csv", nd), row.names = FALSE)
      cv <- tryCatch(curvature_proxy(U_app[,1:2, drop = FALSE], Yhat_app, k = 6, gamma = 1.0, ngr = 60),
                     error = function(e) c(curv_median = NA_real_, curv_p95 = NA_real_))
      curv_median <- unname(cv["curv_median"]); curv_p95 <- unname(cv["curv_p95"])
    }
    
    fit_summary <- rbind(
      fit_summary,
      mk_row(
        nd, paste0("principal_", m, "D_hypersurface_CV"),
        R2_like = unname(res_cv["R2_like"]),
        R2_adj  = unname(res_cv["R2_adj"]),
        GCV     = unname(res_cv["GCV"]),
        R2_linear = base, R2_z = r2_z, R2_orth = r2_orth,
        Delta_R2 = deltaR2, Curv_med = curv_median, Curv_p95 = curv_p95,
        TwoNN_ID = id_ph
      )
    )
  }
  
  write.csv(fit_summary, "dimensional_sweep_principal_manifold_R2.csv", row.names = FALSE)
  list(summary = fit_summary, cache = cache_path)
}