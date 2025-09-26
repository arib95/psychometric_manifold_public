# dimension_plots.R
# -----------------------------------------------------------------------------
# Purpose: End-to-end generation of probability fields and diagnostics over a
# 2D base manifold (b1,b2) and its whitened unit-disk transform (u1,u2).
# Design: single config list; pure functions; guarded optional features;
#         minimal globals; consistent I/O; deduplicated metrics.
# R >= 4.2 recommended. macOS/Linux preferred for parallel.
# -----------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(tidyr)
  library(mgcv);    library(ggrepel); library(readr)
  library(parallelly); library(future); library(future.apply)
  library(pROC);   library(PRROC); library(progressr);
  library(robustbase)
})

# ============================== 0) Config =====================================

default_cfg <- function(){
  list(
    # toggles
    make_3d_surfaces = FALSE,
    make_lift_maps   = FALSE,
    add_u_hulls      = TRUE,
    
    # modelling
    dx_min_pos = 10L,
    dx_min_neg = 10L,
    cv_folds   = 5L,
    seed_pred  = 42L,
    
    # significance (clusters)
    sig_method = "perm",   # perm | se2 (se2 = fixed ±2)
    sig_B      = 200L,
    sig_fwer   = 0.95,
    
    # grids
    nu_unit    = 550L,      # unit square grid side
    n_base_grid= 200L,      # base hull grid side
    
    # plotting
    br_fixed   = c(.15,.25,.40,.60,.80),
    use_ragg   = isTRUE(requireNamespace("ragg", quietly = TRUE)),
    mm         = TRUE,
    
    # parallel
    ncores     = parallelly::availableCores(omit = 1L),
    
    # output
    out_dir    = ".",
    verbose    = TRUE
  )
}

progressr::handlers(global = TRUE)  # pick your favorite handler in your env

log_msg <- function(..., .cfg){ if (isTRUE(.cfg$verbose)) message(sprintf(...)) }

# ---- Safe plan setter ---------------------------------------------------------
set_future_plan <- function(cfg){
  options(future.rng.onMisuse = "warn",
          future.globals.maxSize = 1024^3)
  if (!isTRUE(cfg$ncores > 1)) {
    future::plan(future::sequential); return(invisible())
  }
  ok <- try({
    # Prefer multicore on POSIX if allowed; else multisession
    if (.Platform$OS.type != "windows" && parallelly::supportsMulticore()) {
      future::plan(future::multicore, workers = cfg$ncores)
    } else {
      future::plan(future::multisession, workers = cfg$ncores)
    }
  }, silent = TRUE)
  if (inherits(ok, "try-error")) {
    message("[future] parallel plan failed; falling back to sequential (",
            conditionMessage(attr(ok, "condition")), ")")
    future::plan(future::sequential)
  }
}

# ---- Robust apply shims (future -> base fallback) -----------------------------
.drop_future_args <- function(dots){
  if (is.null(names(dots))) return(dots)
  dots[!grepl("^future\\.", names(dots), perl = TRUE)]
}

FUTURE_LAPPLY <- function(X, FUN, ...) {
  dots <- list(...)
  # try future path
  ok <- try(do.call(future.apply::future_lapply, c(list(X, FUN), dots)), silent = TRUE)
  if (!inherits(ok, "try-error")) return(ok)
  message("[parallel] future_lapply failed, falling back to lapply: ",
          conditionMessage(attr(ok, "condition")))
  # strip future.* before calling base lapply
  dots2 <- .drop_future_args(dots)
  do.call(base::lapply, c(list(X, FUN), dots2))
}

FUTURE_SAPPLY <- function(X, FUN, ..., simplify = TRUE, USE.NAMES = TRUE) {
  dots <- list(...)
  ok <- try(do.call(future.apply::future_sapply,
                    c(list(X, FUN, simplify = simplify, USE.NAMES = USE.NAMES), dots)),
            silent = TRUE)
  if (!inherits(ok, "try-error")) return(ok)
  message("[parallel] future_sapply failed, falling back to sapply: ",
          conditionMessage(attr(ok, "condition")))
  dots2 <- .drop_future_args(dots)
  do.call(base::sapply, c(list(X, FUN, simplify = simplify, USE.NAMES = USE.NAMES), dots2))
}

FUTURE_MAPPLY <- function(FUN, ..., MoreArgs = NULL, SIMPLIFY = TRUE, USE.NAMES = TRUE) {
  dots <- list(...)
  ok <- try(do.call(future.apply::future_mapply,
                    c(list(FUN, ..., MoreArgs = MoreArgs, SIMPLIFY = SIMPLIFY, USE.NAMES = USE.NAMES), dots)),
            silent = TRUE)
  if (!inherits(ok, "try-error")) return(ok)
  message("[parallel] future_mapply failed, falling back to mapply: ",
          conditionMessage(attr(ok, "condition")))
  dots2 <- .drop_future_args(dots)
  do.call(base::mapply, c(list(FUN, ..., MoreArgs = MoreArgs, SIMPLIFY = SIMPLIFY, USE.NAMES = USE.NAMES), dots2))
}

# unify ggsave with ragg + mm sizes
save_plot <- function(filename, plot, width, height, cfg){
  path <- file.path(cfg$out_dir, filename)
  if (cfg$use_ragg){
    ragg::agg_png(path, width = width, height = height,
                  units = if (cfg$mm) "mm" else "in", res = 450, background = "white")
    on.exit(grDevices::dev.off(), add = TRUE)
    print(plot)
  } else {
    ggplot2::ggsave(path, plot, width = if (cfg$mm) width/25.4 else width,
                    height = if (cfg$mm) height/25.4 else height,
                    dpi = 300, bg = "white")
  }
  path
}

# consistent theme
theme_pub <- function(base_size = 11, base_family = "sans"){
  ggplot2::theme_minimal(base_size = base_size, base_family = base_family) +
    ggplot2::theme(
      panel.grid       = element_blank(),
      panel.background = element_rect(fill = "white", colour = NA),
      plot.background  = element_rect(fill = "white", colour = NA),
      strip.background = element_blank(),
      legend.title     = element_text(size = base_size),
      legend.text      = element_text(size = base_size - 1)
    )
}

scale_prob_fill <- function(cfg){
  lims <- c(0,1)
  lab  <- scales::percent_format(accuracy = 1)
  eng  <- tolower(cfg$palette$engine %||% "scico")
  dir  <- cfg$palette$direction %||% 1
  nm   <- cfg$palette$name %||% "lajolla"
  if (eng == "scico" && requireNamespace("scico", quietly = TRUE)) {
    scico::scale_fill_scico(palette = nm, direction = dir, limits = lims, na.value = NA, labels = lab)
  } else if (eng == "colorspace" && requireNamespace("colorspace", quietly = TRUE)) {
    colorspace::scale_fill_continuous_sequential(palette = nm, rev = (dir == -1), limits = lims, na.value = NA, labels = lab)
  } else if (eng == "brewer") { # ColorBrewer (via distiller → continuous)
    ggplot2::scale_fill_distiller(palette = nm, type = "seq", direction = if (dir==1) 1 else -1,
                                  limits = lims, na.value = NA, labels = lab)
  } else if (eng == "paletteer" && requireNamespace("paletteer", quietly = TRUE)) {
    paletteer::scale_fill_paletteer_c(nm, limits = lims, na.value = NA, labels = lab, direction = dir)
  } else if (eng == "manual" && !is.null(cfg$palette$colours)) {
    ggplot2::scale_fill_gradientn(colours = cfg$palette$colours, limits = lims, na.value = NA, labels = lab)
  } else {
    ggplot2::scale_fill_gradient(low = "#f7fbff", high = "#08306b", limits = lims, na.value = NA, labels = lab)
  }
}

# (Only used in fig2 density colouring)
scale_level_colour <- function(cfg){
  eng <- tolower(cfg$palette$engine %||% "scico")
  dir <- cfg$palette$direction %||% 1
  nm  <- cfg$palette$name %||% "oslo"
  if (eng == "scico" && requireNamespace("scico", quietly = TRUE)) {
    scico::scale_colour_scico(palette = nm, direction = dir, guide = "none")
  } else if (eng == "colorspace" && requireNamespace("colorspace", quietly = TRUE)) {
    colorspace::scale_colour_continuous_sequential(palette = nm, rev = (dir == -1), guide = "none")
  } else if (eng == "brewer") {
    ggplot2::scale_colour_distiller(palette = nm, type = "seq", direction = if (dir==1) 1 else -1, guide = "none")
  } else if (eng == "paletteer" && requireNamespace("paletteer", quietly = TRUE)) {
    paletteer::scale_colour_paletteer_c(nm, direction = dir, guide = "none")
  } else {
    ggplot2::scale_colour_gradient(guide = "none")
  }
}

trim_gam_safe <- function(fit, keep_model = TRUE){
  fn <- try(getFromNamespace("trim.gam", "mgcv"), silent = TRUE)
  if (is.function(fn)) {
    out <- try(fn(fit), silent = TRUE)
    if (!inherits(out, "try-error")) return(out)
  }
  # Keep minimal bits for predict(se=TRUE); optionally drop model frame
  drop <- c(
    "y","fitted.values","linear.predictors","working.weights",
    "weights","offset","residuals","prior.weights","hat","qr","X","R"
  )
  if (!keep_model) drop <- c(drop, "model")
  for (nm in intersect(names(fit), drop)) fit[[nm]] <- NULL
  fit
}

predict_link_se <- function(fit, newdata){
  pr <- try(predict(fit, newdata = newdata, type = "link", se.fit = TRUE), silent = TRUE)
  if (!inherits(pr, "try-error") && is.list(pr) && !is.null(pr$se.fit)) {
    return(list(eta = as.numeric(pr$fit), se = pmax(as.numeric(pr$se.fit), 1e-9)))
  }
  Xp <- try(predict(fit, newdata = newdata, type = "lpmatrix"), silent = TRUE)
  if (!inherits(Xp, "try-error") && !is.null(fit$Vp)) {
    eta <- as.numeric(predict(fit, newdata = newdata, type = "link"))
    Vp  <- fit$Vp
    Xp  <- as.matrix(Xp)
    if (ncol(Xp) != ncol(Vp) || nrow(Vp) != ncol(Vp)) {
      # dimensions off; return eta with NA se
      return(list(eta = eta, se = rep(NA_real_, nrow(newdata))))
    }
    se  <- sqrt(pmax(rowSums((Xp %*% Vp) * Xp), 1e-12))
    return(list(eta = eta, se = se))
  }
  list(eta = as.numeric(predict(fit, newdata = newdata, type = "link")),
       se  = rep(NA_real_, nrow(newdata)))
}

# ======================= 1) Alignment & Encoding ==============================

`%||%` <- function(a,b) if (!is.null(a)) a else b

as_base2_numeric <- function(Base_in){
  A <- as.data.frame(Base_in[,1:2,drop=FALSE])
  for (j in 1:2){
    A[[j]] <- suppressWarnings(as.numeric(A[[j]]))
    if (any(!is.finite(A[[j]]))){ m <- mean(A[[j]], na.rm = TRUE); if (!is.finite(m)) m <- 0; A[[j]][!is.finite(A[[j]])] <- m }
  }
  B <- as.matrix(A); storage.mode(B) <- "double"; colnames(B) <- c("b1","b2"); rownames(B) <- rownames(Base_in); B
}

align_by_rownames <- function(..., .allow_index_align = TRUE){
  L <- list(...); if (length(L)==1L && is.list(L[[1]]) && !is.data.frame(L[[1]])) L <- L[[1]]
  nm <- names(L) %||% paste0("obj", seq_along(L))
  L  <- lapply(L, function(A) as.data.frame(A, check.names = FALSE, stringsAsFactors = FALSE))
  rn_list <- lapply(L, rownames); had_ids <- vapply(rn_list, function(x) !is.null(x) && any(nzchar(x)), logical(1))
  if (any(had_ids)){
    id_sets <- lapply(seq_along(L), function(i) if (had_ids[i]) rn_list[[i]] else character(0))
    ids <- Reduce(intersect, id_sets)
    if (!length(ids)){
      if (.allow_index_align && length(unique(vapply(L, nrow, integer(1)))) == 1L){
        ids <- sprintf("idx_%d", seq_len(nrow(L[[1]]))); L <- lapply(L, function(A){ rownames(A) <- ids; A })
      } else stop("[align] no overlap and differing nrows.")
    } else {
      L <- lapply(L, function(A){ if (is.null(rownames(A))) stop("[align] some objects lack IDs; cannot intersect."); A[ids,,drop=FALSE] })
    }
  } else {
    n <- unique(vapply(L, nrow, integer(1))); if (length(n)==1L && .allow_index_align){ ids <- sprintf("idx_%d", seq_len(n)); L <- lapply(L, function(A){ rownames(A) <- ids; A }) } else stop("[align] no rownames and differing nrows; cannot align.")
  }
  names(L) <- nm; L
}

# encoder: either take precomputed Z with varmap attr, or rebuild with design_with_map()
encode_Z <- function(Base, X_pred, Z = NULL, w_all = NULL){
  fallback_rebuild <- function(Base, X_pred, w_all){
    if (!exists("design_with_map")) stop("[encode] design_with_map() must exist to build Z.")
    AL <- align_by_rownames(Base = Base[,1:2,drop=FALSE], X = X_pred, .allow_index_align = TRUE)
    Base_A <- as_base2_numeric(AL$Base)
    X      <- as.data.frame(AL$X, check.names = TRUE)
    keep   <- vapply(X, function(v){ v2 <- suppressWarnings(as.numeric(v)); length(unique(na.omit(v2))) > 1L }, logical(1))
    if (!any(keep)) stop(sprintf("[encode] all predictors NA-only/constant (n=%d).", nrow(Base_A)))
    X      <- X[, keep, drop = FALSE]
    Z0     <- design_with_map(X)
    varmap <- attr(Z0, "varmap")
    if (is.null(varmap)) stop("[encode] design_with_map() must set varmap attribute.")
    if (length(varmap) != ncol(Z0))
      stop(sprintf("[encode] varmap length (%d) must equal ncol(Z) (%d).",
                   length(varmap), ncol(Z0)))
    if (any(!nzchar(as.character(varmap))))
      stop("[encode] varmap must have non-empty names for every encoded column.")
    w_all  <- w_all %||% setNames(rep(1, length(unique(varmap))), unique(varmap))
    w_enc  <- setNames(rep(1, ncol(Z0)), colnames(Z0))
    for (nm in unique(varmap)){
      idx <- which(varmap == nm); wj <- if (!is.null(w_all[nm]) && is.finite(w_all[nm])) w_all[nm] else 1
      w_enc[idx] <- wj / length(idx)
    }
    Zw <- sweep(Z0, 2, sqrt(pmax(w_enc, 0)), "*")
    Zs <- scale(Zw, TRUE, TRUE); attr(Zs, "varmap") <- varmap
    stopifnot(identical(rownames(Zs), rownames(Base_A)))
    list(Base_A = Base_A, Z_A = Zs, varmap = varmap)
  }
  
  # If no Z, build it.
  if (is.null(Z)) return(fallback_rebuild(Base, X_pred, w_all))
  
  # Try to align provided Z; if that fails for any reason, rebuild.
  try_align <- try({
    AL <- align_by_rownames(Base = Base[,1:2,drop=FALSE], Z = Z, .allow_index_align = TRUE)
    Base_A <- as_base2_numeric(AL$Base)
    Z_A    <- as.matrix(AL$Z); storage.mode(Z_A) <- "double"
    varmap <- attr(Z, "varmap") %||% attr(Z_A, "varmap")
    if (is.null(varmap)) stop("[encode] Z provided without varmap attribute.")
    list(Base_A = Base_A, Z_A = Z_A, varmap = varmap)
  }, silent = TRUE)
  
  if (inherits(try_align, "try-error")) {
    message("[encode] Provided Z could not be aligned to Base (", conditionMessage(attr(try_align, "condition")), "). Rebuilding from X_pred.")
    return(fallback_rebuild(Base, X_pred, w_all))
  }
  try_align
}

# ============================ 2) Geometry =====================================

# Robust whitening to unit disk
# Robust whitening with trimming + shrinkage
standardise_to_circle <- function(Base2,
                                  cover     = 0.98,  # map this inlier-quantile radius to 1
                                  h_frac    = 0.80,  # central fraction used for shape
                                  shrink    = 0.15,  # shrinkage to identity (0–0.5 is sane)
                                  ridge     = 1e-6   # tiny PD floor
){
  X <- as.matrix(Base2[, 1:2, drop = FALSE])
  storage.mode(X) <- "double"
  # clean NAs per column
  for (j in 1:2) {
    v <- suppressWarnings(as.numeric(X[, j]))
    m <- stats::median(v, na.rm = TRUE)
    if (!is.finite(m)) m <- 0
    v[!is.finite(v)] <- m
    X[, j] <- v
  }
  n <- nrow(X); h <- max(ceiling(h_frac * n), 3L)
  
  # ---- robust centre + provisional scatter for distances ---------------------
  have_mcd <- requireNamespace("robustbase", quietly = TRUE)
  have_trob <- requireNamespace("MASS", quietly = TRUE)
  
  if (have_mcd) {
    cm <- robustbase::covMcd(X, alpha = h_frac)
    mu0 <- as.numeric(cm$center)
    S0  <- cm$cov
  } else if (have_trob) {
    ct <- MASS::cov.trob(X)
    mu0 <- as.numeric(ct$center)
    S0  <- ct$cov
  } else {
    mu0 <- apply(X, 2, stats::median)
    mad2 <- stats::mad(X[,1], constant = 1.4826)^2 + ridge
    mad2b<- stats::mad(X[,2], constant = 1.4826)^2 + ridge
    S0   <- diag(c(mad2, mad2b), 2)
  }
  
  # stabilise S0
  if (!all(is.finite(S0))) S0[] <- 0
  e0 <- eigen(S0, symmetric = TRUE)
  lam0 <- pmax(e0$values, ridge)
  V0   <- e0$vectors
  S0_inv <- V0 %*% diag(1/lam0, 2) %*% t(V0)
  
  # robust distances; keep central h for shape
  D2 <- rowSums((X - matrix(mu0, n, 2, byrow = TRUE)) %*% S0_inv * (X - matrix(mu0, n, 2, byrow = TRUE)))
  idx <- order(D2, na.last = NA)[seq_len(min(h, length(D2)))]
  
  Xin <- X[idx, , drop = FALSE]
  mu  <- colMeans(Xin)
  
  # classical cov on inliers, then shrink toward identity (trace/2)
  Sin <- stats::cov(Xin)
  if (!all(is.finite(Sin))) Sin <- diag(1, 2)
  
  tr  <- sum(diag(Sin))
  tau <- if (is.finite(tr) && tr > 0) tr/2 else 1
  Ssh <- (1 - shrink) * Sin + shrink * (tau * diag(2))
  Ssh <- Ssh + ridge * tau * diag(2)
  
  # eigendecompose; build whitening and its inverse
  e  <- eigen(Ssh, symmetric = TRUE)
  lam <- pmax(e$values, ridge)
  V   <- e$vectors
  S_half <- V %*% diag(sqrt(lam), 2) %*% t(V)
  S_hi   <- V %*% diag(1/sqrt(lam), 2) %*% t(V)
  
  # whiten all points
  U0 <- t(S_hi %*% t(sweep(X, 2, mu, "-")))
  r  <- sqrt(rowSums(U0^2))
  
  # scale by inlier radius quantile (measured on inliers)
  rin <- sqrt(rowSums( (t(S_hi %*% t(sweep(Xin, 2, mu, "-"))))^2 ))
  s   <- as.numeric(stats::quantile(rin, probs = cover, na.rm = TRUE))
  if (!is.finite(s) || s <= 0) s <- max(rin, na.rm = TRUE)
  
  fwd <- function(xb1, xb2){
    Uq <- t(S_hi %*% t(sweep(cbind(xb1, xb2), 2, mu, "-"))) / s
    colnames(Uq) <- c("u1","u2"); Uq
  }
  inv <- function(u1, u2){
    Xq <- t(S_half %*% t(cbind(u1, u2) * s)) + matrix(mu, length(u1), 2, byrow = TRUE)
    colnames(Xq) <- c("b1","b2"); Xq
  }
  
  list(mu = mu, S_half = S_half, S_half_inv = S_hi, s = s, fwd = fwd, inv = inv)
}

make_unitdisk_square <- function(nu){
  u1 <- seq(-1, 1, length.out = nu); u2 <- seq(-1, 1, length.out = nu)
  g  <- expand.grid(u1 = u1, u2 = u2)
  mask_vec <- with(g, sqrt(u1^2 + u2^2) <= 1 + 1e-9)
  list(u1 = u1, u2 = u2, grid = g, mask_vec = mask_vec)
}

inside_hull <- function(px, py, poly, eps = 1e-12){
  n <- nrow(poly); j <- n; inside <- rep(FALSE, length(px))
  on_edge <- rep(FALSE, length(px))
  for (i in seq_len(n)){
    xi <- poly$b1[i]; yi <- poly$b2[i]; xj <- poly$b1[j]; yj <- poly$b2[j]
    # point-on-segment check (projection + distance)
    dx <- xj - xi; dy <- yj - yi
    seg_len2 <- dx*dx + dy*dy + eps
    t <- pmin(1, pmax(0, ((px - xi)*dx + (py - yi)*dy) / seg_len2))
    projx <- xi + t*dx; projy <- yi + t*dy
    on_edge <- on_edge | ((px - projx)^2 + (py - projy)^2 <= (1e-9)^2)
    
    cross <- ((yi > py) != (yj > py)) &
      (px < (xj - xi) * (py - yi) / (yj - yi + eps) + xi)
    inside <- xor(inside, cross)
    j <- i
  }
  inside | on_edge
}

draw_disk_outline <- function(){ th <- seq(0, 2*pi, length.out = 361); data.frame(x = cos(th), y = sin(th)) }

# ============================ 3) CV & metrics =================================

# folds (stratified)
make_folds <- function(y, K, seed){ set.seed(seed); y <- as.integer(y>0); i1 <- which(y==1); i0 <- which(y==0); f1 <- sample(rep(seq_len(K), length.out=length(i1))); f0 <- sample(rep(seq_len(K), length.out=length(i0))); fid <- integer(length(y)); fid[i1] <- f1; fid[i0] <- f0; fid }

fit_glm_or_glmnet <- function(y, X){
  df <- as.data.frame(X); df$.y <- as.integer(y>0)
  if (requireNamespace("glmnet", quietly = TRUE)){
    x <- as.matrix(df[setdiff(names(df), ".y")]); yb <- df$.y
    cv <- glmnet::cv.glmnet(x, yb, alpha = 0, family = "binomial", standardize = TRUE, nfolds = 5)
    list(type = "glmnet", fit = cv, xnames = colnames(x))
  } else {
    # jitter for separation
    if (ncol(df) > 1L){ for (j in setdiff(seq_along(df), which(names(df)==".y"))){ v <- df[[j]]; if (is.numeric(v)) df[[j]] <- v + rnorm(length(v), 0, 1e-8) } }
    f <- stats::glm(.y ~ ., data = df, family = stats::binomial(), control = list(maxit = 50))
    list(type = "glm", fit = f)
  }
}

# --- prediction helpers ---
pred_prob <- function(mod, newX){
  if (mod$type == "glmnet") {
    x <- as.matrix(as.data.frame(newX)[, mod$xnames, drop = FALSE])
    p <- stats::predict(mod$fit, x, s = "lambda.min", type = "response")
  } else {
    p <- stats::predict(mod$fit, newdata = as.data.frame(newX), type = "response")
  }
  as.numeric(pmin(pmax(p, 1e-6), 1 - 1e-6))
}

oof_prob <- function(y, X, K, seed){
  y  <- as.integer(y > 0)
  X  <- as.data.frame(X)
  fid <- make_folds(y, K, seed)
  p   <- rep(NA_real_, length(y))
  for (k in seq_len(K)) {
    tr <- fid != k; te <- fid == k
    mod <- fit_glm_or_glmnet(y[tr], X[tr, , drop = FALSE])
    p[te] <- pred_prob(mod, X[te, , drop = FALSE])
  }
  p[!is.finite(p)] <- mean(y)
  pmin(pmax(p, 1e-6), 1 - 1e-6)
}

orient_scores <- function(y, p){
  y <- as.integer(y > 0); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  P <- sum(y); N <- length(y) - P
  if (P == 0L || N == 0L) return(p)
  r <- rank(p, ties.method = "average")
  auc_raw <- (sum(r[y == 1]) - P * (P + 1) / 2) / (P * N)
  if (is.finite(auc_raw) && auc_raw < 0.5) 1 - p else p
}

# --- AUROC (symmetric) with pROC-first, rank fallback ---
auc_point <- function(y, p){
  y <- as.integer(y > 0); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  if (length(y) < 2L || !any(y == 1L) || all(y == 1L)) return(NA_real_)
  
  if (requireNamespace("pROC", quietly = TRUE)) {
    r <- try(
      pROC::roc(response = factor(y, levels = c(0, 1)),
                predictor = p, quiet = TRUE, direction = "auto"),
      silent = TRUE
    )
    if (!inherits(r, "try-error")) {
      a <- suppressWarnings(as.numeric(pROC::auc(r)))
      # if (is.finite(a)) return(max(a, 1 - a))
      if (is.finite(a)) return(a)
    }
  }
  # Mann–Whitney fallback (then symmetrise)
  rk  <- rank(p, ties.method = "average")
  P   <- sum(y == 1); N <- sum(y == 0)
  auc <- (sum(rk[y == 1]) - P * (P + 1) / 2) / (P * N)
  max(auc, 1 - auc)
}

# ---- AUPRC point estimate (robust, PRROC if available, else sweep) ----------
auprc_point <- function(y, p){
  y <- as.integer(y > 0); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  n <- length(y); P <- sum(y); N <- n - P
  if (n < 2L || P == 0L || N == 0L) return(NA_real_)
  if (requireNamespace("PRROC", quietly = TRUE)) {
    r <- try(PRROC::pr.curve(scores.class0 = p[y==1], scores.class1 = p[y==0], curve = FALSE), silent = TRUE)
    if (!inherits(r, "try-error")) {
      ap <- as.numeric(r$auc.integral)
      if (is.finite(ap)) return(ap)
    }
  }
  if (diff(range(p)) < 1e-14) return(P/n)
  s  <- sort(unique(p), decreasing = TRUE)
  tp <- fp <- rec <- prec <- numeric(length(s))
  for (i in seq_along(s)) {
    sel <- p >= s[i]
    tp[i] <- sum(y[sel]); fp[i] <- sum(1 - y[sel])
    rec[i]  <- tp[i] / P
    prec[i] <- tp[i] / pmax(tp[i] + fp[i], 1e-12)
  }
  drec <- c(rec[1], diff(rec))
  sum(prec * drec)
}

# ---- PR-Gain (Flach & Kull) with anchors; returns auprg ∈ [0,1] + curve -----
prg_flach <- function(y, p, eps = 1e-12){
  y <- as.integer(y > 0)
  ok <- is.finite(p); y <- y[ok]; p <- p[ok]
  n <- length(y); if (n == 0L || length(unique(y)) < 2L)
    return(list(curve = data.frame(recG = numeric(0), precG = numeric(0)), auprg = NA_real_))
  pi <- mean(y == 1)
  
  # --- PR curve from scores (decreasing thresholds) ---
  o  <- order(p, decreasing = TRUE)
  y  <- y[o]
  tp <- cumsum(y == 1); fp <- cumsum(y == 0)
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  
  R  <- tp / pmax(n1, 1L)
  P  <- tp / pmax(tp + fp, 1L)
  
  # add anchors; then sort by R and keep max P at each R
  R  <- c(0, R, 1)
  P  <- c(pi, P, pi)
  PR <- data.frame(R = R, P = P)
  PR <- PR[order(PR$R, PR$P, decreasing = FALSE), ]
  PR <- aggregate(P ~ R, PR, max)            # dedup on recall
  
  # --- PR -> PRG (Flach & Kull definitions) ---
  recG  <- (PR$R - pi) / ((1 - pi) * pmax(PR$R, eps))
  precG <- (PR$P - pi) / ((1 - pi) * pmax(PR$P, eps))
  
  # clip to plotting domain for area (RG in [0,1], PG clipped below at 0)
  recG_clip  <- pmin(pmax(recG, 0), 1)
  precG_clip <- pmin(pmax(precG, 0), 1)
  
  # make RG monotone increasing (it should be after the PR clean-up)
  o2 <- order(recG_clip, precG_clip)
  recG_clip  <- recG_clip[o2]
  precG_clip <- precG_clip[o2]
  
  # trapezoid AUPRG in PRG space
  auprg <- if (length(recG_clip) >= 2)
    sum(0.5 * (precG_clip[-1] + precG_clip[-length(precG_clip)]) *
          diff(recG_clip)) else NA_real_
  
  list(
    curve = data.frame(recG = recG, precG = precG),   # raw (can exceed [0,1] on the left)
    auprg = as.numeric(auprg)
  )
}

# ---- Generic stratified bootstrap for a scalar metric FUN(y,p) ---------------
boot_ci <- function(y, p, FUN, B = 1000L, seed = 42L){
  set.seed(seed)
  y <- as.integer(y > 0); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  i0 <- which(y==0L); i1 <- which(y==1L)
  pt <- as.numeric(FUN(y, p))
  if (!length(i0) || !length(i1)) return(c(point = pt, lo = NA_real_, hi = NA_real_))
  vals_raw <- FUTURE_SAPPLY(
    seq_len(B),
    function(i){
      ii <- c(sample(i0, length(i0), TRUE), sample(i1, length(i1), TRUE))
      out <- FUN(y[ii], p[ii])              # may be NA, list, etc.
      out <- suppressWarnings(as.numeric(out))
      if (length(out) != 1L || !is.finite(out)) NA_real_ else out
    },
    future.seed = TRUE
  )
  
  # Force atomic numeric and drop non-finite
  vals <- suppressWarnings(as.numeric(unlist(vals_raw, use.names = FALSE)))
  vals <- vals[is.finite(vals)]
  
  if (!length(vals)) {
    return(c(point = pt, lo = NA_real_, hi = NA_real_))
  }
  
  c(
    point = pt,
    lo = as.numeric(stats::quantile(vals, 0.025, na.rm = TRUE, names = FALSE)),
    hi = as.numeric(stats::quantile(vals, 0.975, na.rm = TRUE, names = FALSE))
  )
}

# helper: robust equal-count binning for calibration
make_calib_bins <- function(pc, n_min = 10L, max_bins = 8L) {
  # pc must be numeric in [0,1]
  pc <- as.numeric(pc)
  pc <- pmin(pmax(pc, 1e-12), 1 - 1e-12)
  
  # deterministic tie-break so pj is strictly increasing on ties
  rk <- rank(pc, ties.method = "first")
  pj <- pc + 1e-12 * rk
  
  n <- length(pj)
  k_target <- max(2L, min(max_bins, floor(n / max(n_min, 1L))))
  
  # primary breaks
  br <- unique(stats::quantile(pj, probs = seq(0, 1, length.out = k_target + 1L),
                               type = 7, na.rm = TRUE))
  
  # last-ditch fallback to 2 bins if quantiles collapse
  if (length(br) < 3L) {
    br <- unique(stats::quantile(pj, probs = c(0, .5, 1),
                                 type = 7, na.rm = TRUE))
  }
  
  # if still not enough distinct breaks, return a 2-level logical split
  if (length(br) < 2L) {
    g <- rk > stats::median(rk, na.rm = TRUE)
    return(list(g = as.factor(g), n_bins = length(unique(g[is.finite(g)]))))
  }
  
  g <- cut(pj, breaks = br, include.lowest = TRUE, right = TRUE)
  
  # if cut() produced <2 effective groups (can happen with severe ties), logical split fallback
  if (length(levels(g)) < 2L || length(unique(g[!is.na(g)])) < 2L) {
    g <- rk > stats::median(rk, na.rm = TRUE)
    g <- as.factor(g)
  }
  list(g = g, n_bins = length(levels(g)))
}

# ---- Simple, dependency-free AECE with robust binning ------------------------
# Uses calibrated-to-prevalence probabilities (order-preserving), then equal-count quantile bins.
robust_aece <- function(y, p, min_bin = 10L, max_bins = 8L){
  y <- as.integer(y > 0); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  n <- length(y); if (n < 2L) return(list(aece = NA_real_, n_bins = 0L))
  
  prev <- mean(y)
  
  # calib-in-the-large on logit scale
  pclip <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  lp <- qlogis(pclip); adj <- qlogis(prev) - mean(lp)
  pc <- plogis(lp + adj)
  
  # flat after calibration? aECE is zero with 1 bin
  if (!any(is.finite(pc)) || diff(range(pc, na.rm = TRUE)) < 1e-12) {
    return(list(aece = 0, n_bins = 1L))
  }
  
  B <- make_calib_bins(pc, n_min = min_bin, max_bins = max_bins)
  g <- B$g
  
  T <- data.frame(
    n    = as.integer(tapply(y, g, length)),
    ybar = as.numeric(tapply(y, g, mean)),
    phat = as.numeric(tapply(pc, g, mean))
  )
  T <- T[is.finite(T$n) & T$n > 0, , drop = FALSE]
  if (!nrow(T)) return(list(aece = NA_real_, n_bins = 0L))
  
  list(
    aece   = as.numeric(weighted.mean(abs(T$phat - T$ybar), w = T$n)),
    n_bins = as.integer(nrow(T))
  )
}

# ---- Brier score + Brier R^2 (skill) -----------------------------------------
brier_score <- function(y, p){
  y <- as.integer(y > 0)
  p <- pmin(pmax(as.numeric(p), 1e-6), 1 - 1e-6)
  ok <- is.finite(y) & is.finite(p)
  y <- y[ok]; p <- p[ok]
  if (length(y) < 2L) return(NA_real_)
  mean((p - y)^2)
}

# R^2_Brier = 1 - BS(model) / BS(null), null = constant forecast = prevalence
brier_R2 <- function(y, p){
  y <- as.integer(y > 0)
  p <- pmin(pmax(as.numeric(p), 1e-6), 1 - 1e-6)
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  n <- length(y); if (n < 2L) return(NA_real_)
  pi <- mean(y)
  bs_null <- pi * (1 - pi)                # BS of constant prevalence predictor
  if (!is.finite(bs_null) || bs_null <= 0) return(NA_real_)  # degenerate class
  bs_mod <- mean((p - y)^2)
  1 - bs_mod / bs_null
}

# Stratified bootstrap CI for scalar metric FUN(y, p) (reuses your parallel shims)
boot_brier_R2 <- function(y, p, B = 1000L, seed = 42L){
  set.seed(seed)
  y <- as.integer(y > 0); p <- as.numeric(p)
  ok <- is.finite(y) & is.finite(p); y <- y[ok]; p <- p[ok]
  i0 <- which(y == 0L); i1 <- which(y == 1L)
  pt <- as.numeric(brier_R2(y, p))
  if (!length(i0) || !length(i1)) return(c(point = pt, lo = NA_real_, hi = NA_real_))
  vals <- FUTURE_SAPPLY(
    seq_len(B),
    function(i, y, p, i0, i1){
      ii <- c(sample(i0, length(i0), TRUE), sample(i1, length(i1), TRUE))
      as.numeric(brier_R2(y[ii], p[ii]))
    },
    y = y, p = p, i0 = i0, i1 = i1,
    future.seed = TRUE, future.globals = TRUE
  )
  vf <- vals[is.finite(vals)]
  if (!length(vf)) return(c(point = pt, lo = NA_real_, hi = NA_real_))
  c(point = pt,
    lo = as.numeric(stats::quantile(vf, 0.025, names = FALSE)),
    hi = as.numeric(stats::quantile(vf, 0.975, names = FALSE)))
}

# --- OOF helpers: additive vs full surface on (b1,b2) -------------------------
oof_prob_gam_add <- function(y, B, K, seed){
  y <- as.integer(y>0); fid <- make_folds(y, K, seed); p <- rep(NA_real_, length(y))
  for(k in seq_len(K)){
    tr <- fid!=k; te <- fid==k
    dtr <- data.frame(y=y[tr], b1=B[tr,1], b2=B[tr,2])
    fit <- mgcv::bam(y ~ s(b1, bs="tp", k=40) + s(b2, bs="tp", k=40),
                     family=binomial(), data=dtr, method="fREML", discrete=TRUE)
    p[te] <- as.numeric(predict(fit, newdata=data.frame(b1=B[te,1], b2=B[te,2]), type="response"))
  }
  pmin(pmax(p,1e-6),1-1e-6)
}

oof_prob_gam_full <- function(y, B, K, seed){
  y <- as.integer(y>0); fid <- make_folds(y, K, seed); p <- rep(NA_real_, length(y))
  for(k in seq_len(K)){
    tr <- fid!=k; te <- fid==k
    dtr <- data.frame(y=y[tr], b1=B[tr,1], b2=B[tr,2])
    fit <- mgcv::bam(y ~ s(b1, b2, bs="tp", k=60),
                     family=binomial(), data=dtr, method="fREML", discrete=TRUE)
    p[te] <- as.numeric(predict(fit, newdata=data.frame(b1=B[te,1], b2=B[te,2]), type="response"))
  }
  pmin(pmax(p,1e-6),1-1e-6)
}

# --- Write Brier R² table in your exact format --------------------------------
write_brierR2_table <- function(DxW_A, Base_A, K, seed, B = 1000L, out_csv){
  rows <- lapply(names(DxW_A), function(dx){
    y_all <- as.integer(DxW_A[[dx]] > 0)
    # need at least both classes present (after NA drop)
    if (length(unique(y_all[is.finite(y_all)])) < 2L) return(NULL)
    
    # choose folds based on available counts
    n1_all <- sum(y_all == 1L, na.rm = TRUE)
    n0_all <- sum(y_all == 0L, na.rm = TRUE)
    Kx <- max(2L, min(K, floor(n1_all/8), floor(n0_all/8)))
    
    # OOF probabilities (may include NAs downstream)
    p_add_all  <- oof_prob_gam_add (y_all, Base_A, K = Kx, seed = seed)
    p_full_all <- oof_prob_gam_full(y_all, Base_A, K = Kx, seed = seed)
    
    # effective cases actually scored by BOTH models
    ok <- is.finite(y_all) & is.finite(p_add_all) & is.finite(p_full_all)
    if (!any(ok)) return(NULL)
    
    y     <- y_all[ok]
    p_add <- p_add_all[ok]
    p_full<- p_full_all[ok]
    
    n_eff <- length(y)
    n1    <- sum(y == 1L)
    n0    <- sum(y == 0L)
    
    # Brier R² with bootstrap CIs (on the filtered set)
    r2_add  <- boot_brier_R2(y, p_add,  B = B, seed = seed)
    r2_full <- boot_brier_R2(y, p_full, B = B, seed = seed)
    
    # ΔR² bootstrap (paired, stratified, on the filtered set)
    set.seed(seed)
    i0 <- which(y == 0L); i1 <- which(y == 1L)
    if (!length(i0) || !length(i1)) return(NULL)
    dboot <- replicate(B, {
      ii <- c(sample(i0, length(i0), TRUE), sample(i1, length(i1), TRUE))
      brier_R2(y[ii], p_full[ii]) - brier_R2(y[ii], p_add[ii])
    })
    dboot <- dboot[is.finite(dboot)]
    d_pt <- r2_full["point"] - r2_add["point"]
    d_lo <- as.numeric(stats::quantile(dboot, 0.025, names = FALSE))
    d_hi <- as.numeric(stats::quantile(dboot, 0.975, names = FALSE))
    star <- if (!is.na(d_lo) && !is.na(d_hi) && (d_lo > 0 || d_hi < 0)) "*" else "ns"
    
    fmt3 <- function(x) sprintf("%.3f", as.numeric(x))
    data.frame(
      Diagnosis = dx,
      n         = n_eff,
      n_pos     = n1,
      n_neg     = n0,
      R2add     = sprintf("%s [%s, %s]", fmt3(r2_add["point"]),  fmt3(r2_add["lo"]),  fmt3(r2_add["hi"])),
      `R²full`  = sprintf("%s [%s, %s]", fmt3(r2_full["point"]), fmt3(r2_full["lo"]), fmt3(r2_full["hi"])),
      `ΔR²`     = sprintf("%s [%s, %s]", fmt3(d_pt), fmt3(d_lo), fmt3(d_hi)),
      `Sig ΔR²` = star,
      stringsAsFactors = FALSE
    )
  })
  T <- dplyr::bind_rows(rows)
  utils::write.csv(T, out_csv, row.names = FALSE, fileEncoding = "UTF-8")
  T
}

# ---- Core prepare: align & encode → Base_A, Z_A, varmap ---------------------
prepare_data <- function(Base, X_pred, Z = NULL, w_all = NULL){
  enc <- encode_Z(Base = Base, X_pred = X_pred, Z = Z, w_all = w_all)
  list(Base_A = enc$Base_A, Z_A = enc$Z_A, varmap = enc$varmap)
}

# ---- Geometry & grids from Base_A -------------------------------------------
build_geometry <- function(Base_A, cfg){
  B1 <- Base_A[,1]; B2 <- Base_A[,2]
  df_base <- data.frame(b1 = as.numeric(B1), b2 = as.numeric(B2))
  Xstd <- standardise_to_circle(Base_A, cover = 0.995)
  U    <- as.data.frame(Xstd$fwd(B1, B2)); names(U) <- c("u1","u2"); rownames(U) <- rownames(Base_A)
  UD   <- make_unitdisk_square(cfg$nu_unit)
  gridX_sq <- as.data.frame(Xstd$inv(UD$grid$u1, UD$grid$u2))
  mask_sq  <- UD$mask_vec
  
  # base convex hull grid
  hidx   <- grDevices::chull(df_base$b1, df_base$b2)
  hpoly  <- df_base[hidx, , drop = FALSE]
  qx <- range(df_base$b1); qy <- range(df_base$b2)
  gridB_full <- expand.grid(
    b1 = seq(qx[1], qx[2], length.out = cfg$n_base_grid),
    b2 = seq(qy[1], qy[2], length.out = cfg$n_base_grid)
  )
  mask_hull <- inside_hull(gridB_full$b1, gridB_full$b2, hpoly)
  
  U$rin  <- sqrt(U$u1^2 + U$u2^2) <= 1 + 1e-9
  U$insq <- abs(U$u1) <= 1 + 1e-9 & abs(U$u2) <= 1 + 1e-9
  
  list(
    Xstd = Xstd, U = U, UD = UD,
    gridX_sq = gridX_sq, mask_sq = mask_sq,
    df_base = df_base, gridB_full = gridB_full, mask_hull = mask_hull
  )
}

# ---- Choose diagnoses to model by min pos/neg --------------------------------
select_diagnoses <- function(DxW, cfg){
  M <- as.data.frame(DxW, check.names = FALSE)
  pos <- colSums(M > 0, na.rm = TRUE)
  neg <- colSums(M == 0, na.rm = TRUE)
  keep <- names(which(pos >= cfg$dx_min_pos & neg >= cfg$dx_min_neg))
  if (!length(keep)){
    # relax thresholds progressively
    for (th in c(8L, 5L, 3L, 1L)) {
      cand <- names(which(pos >= th & neg >= th))
      if (length(cand)) { keep <- cand; break }
    }
  }
  if (!length(keep)){
    cand <- names(which(pos > 0 & neg > 0))
    if (!length(cand)) stop("[dx] No diagnosis has both classes present (≥1 pos & ≥1 neg).")
    keep <- cand
  }
  list(keep = keep, pos = pos, neg = neg)
}

# ---- Fit per-dx GAM on Base coordinates --------------------------------------
fit_dx_gams <- function(DxW_A, Base_A, dx_keep, cfg){
  B1 <- Base_A[,1]; B2 <- Base_A[,2]
  
  res <- FUTURE_LAPPLY(
    dx_keep,
    function(dx, DxW_A, B1, B2){
      y <- as.integer(DxW_A[[dx]] > 0); prev <- mean(y)
      d <- data.frame(y = y, b1 = B1, b2 = B2)
      
      fit <- mgcv::bam(y ~ s(b1, b2, bs="tp", k=60),
                       family = binomial(), data = d,
                       method = "fREML", discrete = TRUE)
      
      edf <- tryCatch(summary(fit)$s.table[1,"edf"], error = function(e) NA_real_)
      if (!is.finite(edf) || edf <= 1.05 ||
          diff(range(predict(fit, type="response"), na.rm=TRUE)) < 1e-4){
        fit <- mgcv::bam(y ~ s(b1, bs="tp", k=40),
                         family = binomial(), data = d,
                         method = "fREML", discrete = TRUE)
      }
      fit <- trim_gam_safe(fit)
      list(fit = fit, prev = prev)
    },
    DxW_A = as.data.frame(DxW_A),
    B1    = as.numeric(B1),
    B2    = as.numeric(B2),
    future.seed    = TRUE,
    future.globals = TRUE
  )
  
  fits <- lapply(res, `[[`, "fit"); names(fits) <- dx_keep
  prev <- vapply(res, `[[`, numeric(1), "prev"); names(prev) <- dx_keep
  list(fits = fits, prev = prev)
}

# ---- Predict fields (unit disk masked, base hull, unit square) ----------------
predict_dx_fields <- function(fits_dx, dx_prev, geom, cfg){
  keys   <- names(fits_dx)
  shards <- lapply(keys, function(k) list(id = k, fit = fits_dx[[k]], prev = dx_prev[[k]]))
  
  progressr::with_progress({
    p <- progressr::progressor(steps = 5L, label = "dx fields (95% CIs)")
    
    # --- 1) Per-dx threshold on the unit-square grid (link-scale) -------------
    # We compute a single thr per dx (FWER via permutation if requested).
    THR_LIST <- FUTURE_LAPPLY(
      shards,
      function(sh, gridX_sq, mask_sq, method, B, fwer){
        if (tolower(method) == "perm") {
          S <- perm_thr_maxabs_z(
            sh$fit, sh$prev, gridX_sq,
            mask_vec = mask_sq,
            B = B, fwer = fwer,
            perm_grid_stride = 3L,
            use_parallel = FALSE,
            reuse_sp = TRUE
          )
          list(dx = sh$id, thr = as.numeric(S$thr))
        } else {
          list(dx = sh$id, thr = stats::qnorm(0.975))  # 1.96 exact 95%
        }
      },
      gridX_sq = geom$gridX_sq, mask_sq = geom$mask_sq,
      method = cfg$sig_method, B = cfg$sig_B, fwer = cfg$sig_fwer,
      future.seed = TRUE, future.globals = TRUE
    )
    THR <- dplyr::bind_rows(THR_LIST)
    p("thr")
    
    # Handy joiner
    thr_of <- function(dx) THR$thr[match(dx, THR$dx)]
    
    # --- 2) UNIT DISK (masked) ------------------------------------------------
    Fstd <- dplyr::bind_rows(
      FUTURE_LAPPLY(
        shards,
        function(sh, gridX_sq, mask_sq, UD, thr_of){
          prs   <- predict_link_se(sh$fit, gridX_sq)
          eta   <- prs$eta
          seeta <- pmax(prs$se, 1e-9)
          thr   <- thr_of(sh$id)
          
          p_hat <- plogis(eta)
          p_lo  <- plogis(eta - thr * seeta)
          p_hi  <- plogis(eta + thr * seeta)
          
          # mask outside disk
          p_hat[!mask_sq] <- NA_real_
          p_lo [!mask_sq] <- NA_real_
          p_hi [!mask_sq] <- NA_real_
          
          data.frame(
            dx = sh$id,
            u1 = UD$grid$u1, u2 = UD$grid$u2,
            p  = pmin(pmax(p_hat, 1e-6), 1 - 1e-6),
            p_lo = pmin(pmax(p_lo ,  1e-9), 1 - 1e-9),
            p_hi = pmin(pmax(p_hi ,  1e-9), 1 - 1e-9),
            stringsAsFactors = FALSE
          )
        },
        gridX_sq = geom$gridX_sq, mask_sq = geom$mask_sq, UD = geom$UD, thr_of = thr_of,
        future.seed = TRUE, future.globals = TRUE
      )
    )
    p("Fstd")
    
    # --- 3) BASE HULL (same thr; CI on probability scale) ---------------------
    Fbase <- dplyr::bind_rows(
      FUTURE_LAPPLY(
        shards,
        function(sh, gridB_full, mask_hull, thr_of){
          prs   <- predict_link_se(sh$fit, gridB_full)
          eta   <- prs$eta
          seeta <- pmax(prs$se, 1e-9)
          thr   <- thr_of(sh$id)
          
          # probability surface + 95% simultaneous CI on p-scale
          p_hat <- plogis(eta)
          p_lo  <- plogis(eta - thr * seeta)
          p_hi  <- plogis(eta + thr * seeta)
          
          # link-scale z vs prevalence, then normalise by thr (FWER-adj “1σ”)
          z_link <- (eta - qlogis(sh$prev)) / seeta
          z_unit <- z_link / pmax(thr, 1e-9)
          
          # mask outside hull
          p_hat[!mask_hull] <- NA_real_
          p_lo [!mask_hull] <- NA_real_
          p_hi [!mask_hull] <- NA_real_
          z_link[!mask_hull] <- NA_real_
          z_unit[!mask_hull] <- NA_real_
          
          data.frame(
            dx = sh$id,
            b1 = gridB_full$b1, b2 = gridB_full$b2,
            p  = pmin(pmax(p_hat, 1e-6), 1 - 1e-6),
            p_lo = pmin(pmax(p_lo ,  1e-9), 1 - 1e-9),
            p_hi = pmin(pmax(p_hi ,  1e-9), 1 - 1e-9),
            z_link = z_link,
            z_unit = z_unit,
            z = z_unit,                # <— compatibility for plots_for_dx()
            stringsAsFactors = FALSE
          )
        },
        gridB_full = geom$gridB_full, mask_hull = geom$mask_hull, thr_of = thr_of,
        future.seed = TRUE, future.globals = TRUE
      )
    )
    p("Fbase")
    
    # --- 4) FULL UNIT SQUARE (unmasked; same thr) -----------------------------
    Fsq <- dplyr::bind_rows(
      FUTURE_LAPPLY(
        shards,
        function(sh, gridX_sq, UD, thr_of){
          prs   <- predict_link_se(sh$fit, gridX_sq)
          eta   <- prs$eta
          seeta <- pmax(prs$se, 1e-9)
          thr   <- thr_of(sh$id)
          
          p_hat <- plogis(eta)
          p_lo  <- plogis(eta - thr * seeta)
          p_hi  <- plogis(eta + thr * seeta)
          
          data.frame(
            dx = sh$id,
            u1 = UD$grid$u1, u2 = UD$grid$u2,
            p  = pmin(pmax(p_hat, 1e-6), 1 - 1e-6),
            p_lo = pmin(pmax(p_lo ,  1e-9), 1 - 1e-9),
            p_hi = pmin(pmax(p_hi ,  1e-9), 1 - 1e-9),
            stringsAsFactors = FALSE
          )
        },
        gridX_sq = geom$gridX_sq, UD = geom$UD, thr_of = thr_of,
        future.seed = TRUE, future.globals = TRUE
      )
    )
    p("Fsq")
    
    # --- 5) Export thresholds for plotting convenience ------------------------
    # Keep both the per-dx thr and (optionally) a normalised z for contours.
    Fsig <- dplyr::bind_rows(
      FUTURE_LAPPLY(
        shards,
        function(sh, gridX_sq, UD, mask_sq, thr_of){
          pr  <- predict(sh$fit, newdata = gridX_sq, type = "link", se.fit = TRUE)
          eta <- as.numeric(pr$fit); se <- pmax(as.numeric(pr$se.fit), 1e-9)
          z   <- (eta - qlogis(sh$prev)) / se
          thr <- thr_of(sh$id)
          data.frame(
            kind = c("disk","square"),
            dx   = sh$id,
            u1   = UD$grid$u1,
            u2   = UD$grid$u2,
            z    = c({zd <- z; zd[!mask_sq] <- NA_real_; zd}, z),
            thr  = thr
          )
        },
        gridX_sq = geom$gridX_sq, UD = geom$UD, mask_sq = geom$mask_sq, thr_of = thr_of,
        future.seed = TRUE, future.globals = TRUE
      )
    )
    p("Fsig")
    
    list(Fstd = Fstd, Fbase = Fbase, Fsq = Fsq, Fsig = Fsig, Thr = THR)
  })
}

# ---- Plot builders (return ggplot objects; saving handled by caller) ----------
add_contours_safe <- function(F, facet_col, xcol = "u1", ycol = "u2", breaks = c(.15,.25,.40,.60,.80)){
  F <- dplyr::mutate(F, p = ifelse(is.finite(p), p, NA_real_))
  ok <- F |>
    dplyr::group_by(.data[[facet_col]]) |>
    dplyr::summarise(ok = sum(is.finite(p)) > 5 &&
                       diff(range(p, na.rm = TRUE)) > 1e-5,
                     .groups = "drop") |>
    dplyr::filter(ok) |>
    dplyr::pull(1)
  if (!length(ok)) return(NULL)
  FF <- dplyr::filter(F, .data[[facet_col]] %in% ok)
  FF$x <- FF[[xcol]]; FF$y <- FF[[ycol]]
  FF <- dplyr::distinct(FF, .data[[facet_col]], x, y, .keep_all = TRUE)
  ggplot2::geom_contour(
    data = FF,
    ggplot2::aes(x = x, y = y, z = p, group = .data[[facet_col]]),
    breaks = breaks, colour = "white", linewidth = .28, alpha = .95,
    na.rm = TRUE, inherit.aes = FALSE
  )
}

plots_for_dx <- function(fields, geom, cfg){
  br <- cfg$br_fixed
  p_std <- ggplot(fields$Fstd, aes(u1, u2)) +
    geom_raster(aes(fill = p), interpolate = TRUE) +
    add_contours_safe(fields$Fstd, facet_col = "dx", xcol = "u1", ycol = "u2", breaks = br) +
    geom_contour(data = subset(fields$Fsig, kind=="disk"),
                 aes(u1, u2, z = z), breaks = c(-2, 2),
                 colour = "black", linetype = 3, linewidth = 0.35, alpha = .9) +
    geom_point(data = subset(geom$U, rin), aes(u1, u2), inherit.aes = FALSE,
               shape = 20, size = 0.30, colour = scales::alpha("black", 0.45), alpha = 1) +
    coord_equal(xlim = c(-1,1), ylim = c(-1,1), expand = FALSE) +
    facet_wrap(~ dx, ncol = 4, scales = "fixed") +
    scale_prob_fill(cfg) +
    labs(x = "u1 (whitened b1,b2)", y = "u2", fill = "Pr(dx)") + theme_pub(11)
  
  p_base <- ggplot(fields$Fbase, aes(b1, b2)) +
    geom_raster(aes(fill = p), interpolate = TRUE, na.rm = TRUE) +
    geom_point(data = geom$df_base, aes(b1, b2), inherit.aes = FALSE, shape = 20, size = 0.28,
               colour = scales::alpha("black", 0.45), alpha = 1) +
    geom_contour(aes(z = p), breaks = br, colour = "white", linewidth = .28, alpha = .9, na.rm = TRUE) +
    geom_contour(aes(z = z, linetype = after_stat(ifelse(..level.. < 0, "neg", "pos"))),
                 breaks = c(-2, 2), colour = "black", linewidth = 0.35, alpha = .9, na.rm = TRUE) +
    scale_linetype_manual(values = c(neg = "dashed", pos = "solid"), guide = "none") +
    scale_prob_fill(cfg) + 
    coord_equal() + facet_wrap(~ dx, ncol = 4, scales = "fixed") +
    labs(x = "b1", y = "b2", fill = "Pr(dx)") + theme_pub(11)
  
  p_sq <- ggplot(fields$Fsq, aes(u1, u2)) +
    geom_raster(aes(fill = p), interpolate = TRUE) +
    add_contours_safe(fields$Fsq, facet_col = "dx", xcol = "u1", ycol = "u2", breaks = br) +
    geom_contour(data = subset(fields$Fsig, kind=="square"),
                 aes(u1, u2, z = z), breaks = c(-2, 2),
                 colour = "black", linetype = 3, linewidth = 0.35, alpha = .9) +
    geom_point(data = subset(geom$U, insq), aes(u1, u2), inherit.aes = FALSE,
               shape = 20, size = 0.30, colour = scales::alpha("black", 0.45), alpha = 1) +
    coord_equal(xlim = c(-1,1), ylim = c(-1,1), expand = FALSE) +
    facet_wrap(~ dx, ncol = 4, scales = "fixed") +
    scale_prob_fill(cfg) +
    labs(x = "u1 (whitened b1,b2)", y = "u2", fill = "Pr(dx)") + theme_pub(11)
  
  list(p_std = p_std, p_base = p_base, p_sq = p_sq)
}

# ---- Read cluster membership and align to Base IDs ---------------------------
read_clusters <- function(clusters_csv, Base_A){
  if (is.null(clusters_csv)) stop("[clusters] clusters_csv must be provided.")
  cls <- readr::read_csv(clusters_csv, show_col_types = FALSE)
  stopifnot(all(c("participant_id","cluster") %in% names(cls)))
  cls$participant_id <- as.character(cls$participant_id)
  cls$cluster        <- as.integer(cls$cluster)
  base_ids <- if (!is.null(rownames(Base_A))) rownames(Base_A) else stop("[clusters] Base_A needs rownames for ID merge.")
  ovl <- intersect(base_ids, cls$participant_id)
  if (!length(ovl)) stop("[clusters] No overlap between Base IDs and cluster CSV.")
  map_cl <- setNames(cls$cluster, cls$participant_id)
  cl_vec <- as.integer(map_cl[base_ids]); cl_vec[is.na(cl_vec)] <- 0L
  cl_vec
}

# ---- Select clusters with min pos/neg ----------------------------------------
select_clusters <- function(cl_vec, min_pos = 10L, min_neg = 10L){
  all_cls  <- sort(setdiff(unique(cl_vec), 0L))
  keep     <- all_cls[vapply(all_cls, function(k) sum(cl_vec==k) >= min_pos && sum(cl_vec!=k) >= min_neg, logical(1))]
  if (!length(keep)) stop("[clusters] No cluster passes min counts.")
  keep
}

# ---- Fit per-cluster GAM on Base coords --------------------------------------
fit_cluster_gams <- function(cl_vec, Base_A, keep_cls){
  B1 <- Base_A[,1]; B2 <- Base_A[,2]
  res <- FUTURE_LAPPLY(keep_cls, function(cid){
    y <- as.integer(cl_vec == cid); prev <- mean(y)
    d <- data.frame(y=y, b1=B1, b2=B2)
    fit <- mgcv::bam(y ~ s(b1, b2, bs="tp", k=60), family=binomial(),
                     data=d, method="fREML", discrete=TRUE)
    edf <- tryCatch(summary(fit)$s.table[1,"edf"], error=function(e) NA_real_)
    if (!is.finite(edf) || edf <= 1.05 ||
        diff(range(predict(fit, type="response"), na.rm=TRUE)) < 1e-4){
      fit <- mgcv::bam(y ~ s(b1, bs="tp", k=40), family=binomial(),
                       data=d, method="fREML", discrete=TRUE)
    }
    list(fit = fit, prev = prev)
  }, future.seed = TRUE)
  fits <- lapply(res, `[[`, "fit"); names(fits) <- as.character(keep_cls)
  prev <- vapply(res, `[[`, numeric(1), "prev"); names(prev) <- as.character(keep_cls)
  list(fits = fits, prev = prev)
}

# ---- Permutation threshold for max |z| on grid (FWER) ------------------------
perm_thr_maxabs_z <- function(fit, y_prev, gridX, mask_vec = NULL,
                              B = 200, fwer = 0.95,
                              chunk = 10, use_parallel = TRUE,
                              reuse_sp = TRUE,
                              perm_grid_stride = 2L) {
  # --- Require mgcv::gam/bam; otherwise return neutral significance
  if (!inherits(fit, "gam")) {
    pr  <- try(predict(fit, newdata = gridX, type = "link", se.fit = TRUE), silent = TRUE)
    eta <- if (!inherits(pr, "try-error")) as.numeric(pr$fit) else as.numeric(predict(fit, newdata = gridX, type = "link"))
    se  <- if (!inherits(pr, "try-error") && !is.null(pr$se.fit)) pmax(as.numeric(pr$se.fit), 1e-9) else rep(NA_real_, nrow(gridX))
    z_obs <- (eta - qlogis(y_prev)) / se
    if (!is.null(mask_vec)) z_obs[!mask_vec] <- NA_real_
    return(list(z = z_obs, thr = Inf))
  }
  
  # 1) Observed z on FULL grid
  pr  <- predict(fit, newdata = gridX, type = "link", se.fit = TRUE)
  eta <- as.numeric(pr$fit)
  se  <- pmax(as.numeric(pr$se.fit), 1e-9)
  z_obs <- (eta - qlogis(y_prev)) / se
  if (!is.null(mask_vec)) z_obs[!mask_vec] <- NA_real_
  
  # Early exits
  if (!is.finite(B) || B <= 0) return(list(z = z_obs, thr = 2.0))
  
  # 2) THIN the prediction grid for permuted refits only
  if (perm_grid_stride > 1L) {
    take <- which(((seq_len(nrow(gridX)) - 1L) %% perm_grid_stride) == 0L)
    grid_perm <- gridX[take, , drop = FALSE]
    mask_perm <- if (is.null(mask_vec)) NULL else mask_vec[take]
  } else {
    grid_perm <- gridX
    mask_perm <- mask_vec
  }
  
  # 3) Permutation loop (predict only on grid_perm)
  dat <- fit$model
  # make sure columns are named b1,b2,y as in the fit calls we use elsewhere
  stopifnot(all(c("b1","b2") %in% names(dat)))
  dat$y <- as.integer(dat$y > 0)
  sp0 <- if (isTRUE(reuse_sp) && !is.null(fit$sp)) fit$sp else NULL
  q0  <- qlogis(mean(dat$y))
  
  .perm_chunk <- function(n_b){
    Sys.setenv(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", BLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    out <- rep(NA_real_, n_b)
    for (i in seq_len(n_b)) {
      dat$.y <- sample(dat$y)
      fit_b <- try(mgcv::bam(.y ~ s(b1,b2,bs="tp",k=40),
                             family=binomial(), data=dat,
                             sp=sp0, method="fREML", discrete=FALSE),
                   silent = TRUE)
      if (inherits(fit_b, "try-error")) next
      prb <- predict(fit_b, newdata = grid_perm, type = "link", se.fit = TRUE)
      zb  <- (as.numeric(prb$fit) - q0) / pmax(as.numeric(prb$se.fit), 1e-9)
      if (!is.null(mask_perm)) zb[!mask_perm] <- NA_real_
      out[i] <- suppressWarnings(max(abs(zb), na.rm = TRUE))
    }
    out
  }
  
  batches <- rep(chunk, length.out = ceiling(B / chunk))
  batches[length(batches)] <- B - sum(batches[-length(batches)])
  batches <- batches[batches > 0]
  
  maxabs <- numeric(sum(batches))
  pos <- 1L
  
  fill_chunk <- function(n_b){
    Sys.setenv(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               VECLIB_MAXIMUM_THREADS="1", BLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    out <- rep(NA_real_, n_b)
    for (i in seq_len(n_b)) {
      dat$.y <- sample(dat$y)
      fit_b <- try(mgcv::bam(.y ~ s(b1,b2,bs="tp",k=40),
                             family=binomial(), data=dat,
                             sp=sp0, method="fREML", discrete=FALSE),
                   silent = TRUE)
      if (inherits(fit_b, "try-error")) next
      prb <- predict(fit_b, newdata = grid_perm, type = "link", se.fit = TRUE)
      zb  <- (as.numeric(prb$fit) - q0) / pmax(as.numeric(prb$se.fit), 1e-9)
      if (!is.null(mask_perm)) zb[!mask_perm] <- NA_real_
      out[i] <- suppressWarnings(max(abs(zb), na.rm = TRUE))
    }
    out
  }
  
  if (isTRUE(use_parallel) && requireNamespace("future.apply", quietly = TRUE)) {
    chunks <- FUTURE_LAPPLY(batches, function(n_b) fill_chunk(n_b), future.seed = TRUE)
    for (vec in chunks) {
      n <- length(vec); maxabs[pos:(pos+n-1L)] <- vec; pos <- pos + n
    }
  } else {
    for (n_b in batches) {
      vec <- fill_chunk(n_b)
      n <- length(vec); maxabs[pos:(pos+n-1L)] <- vec; pos <- pos + n
    }
  }
  
  thr <- stats::quantile(na.omit(maxabs), probs = fwer, na.rm = TRUE, names = FALSE, type = 7)
  list(z = z_obs, thr = as.numeric(thr))
}

# ---- Predict cluster fields + significance in u-space -------------------------
predict_cluster_fields <- function(fits_cl, cl_prev, cl_vec, geom, cfg){
  labs <- paste0("C", names(fits_cl))
  shards <- lapply(names(fits_cl), function(k) list(id = paste0("C", k),
                                                    fit = fits_cl[[k]],
                                                    prev = cl_prev[[k]]))
  
  Fstd <- dplyr::bind_rows(
    FUTURE_LAPPLY(
      shards,
      function(sh, gridX_sq, mask_sq, UD){
        p <- as.numeric(predict(sh$fit, newdata = gridX_sq, type="response"))
        p <- pmin(pmax(p,1e-6),1-1e-6); p[!mask_sq] <- NA_real_
        data.frame(cluster = sh$id, u1 = UD$grid$u1, u2 = UD$grid$u2, p = p)
      },
      gridX_sq = geom$gridX_sq, mask_sq = geom$mask_sq, UD = geom$UD,
      future.seed = TRUE, future.globals = TRUE
    )
  )
  Fstd$cluster <- factor(Fstd$cluster, levels = labs)
  
  Fbase <- dplyr::bind_rows(
    FUTURE_LAPPLY(
      shards,
      function(sh, gridB_full, mask_hull){
        prs <- predict_link_se(sh$fit, gridB_full)
        eta    <- prs$eta
        se_eta <- prs$se
        p_hat  <- plogis(eta); dp_deta <- p_hat*(1-p_hat); se_p <- pmax(se_eta*dp_deta, 1e-9)
        z_p    <- (p_hat - sh$prev)/se_p
        p_hat[!mask_hull] <- NA_real_; z_p[!mask_hull] <- NA_real_
        data.frame(cluster = sh$id, b1 = gridB_full$b1, b2 = gridB_full$b2,
                   p = pmin(pmax(p_hat,1e-6),1-1e-6), z = z_p)
      },
      gridB_full = geom$gridB_full, mask_hull = geom$mask_hull,
      future.seed = TRUE, future.globals = TRUE
    )
  )
  Fbase$cluster <- factor(Fbase$cluster, levels = labs)
  
  Fsq <- dplyr::bind_rows(
    FUTURE_LAPPLY(
      shards,
      function(sh, gridX_sq, UD){
        p <- as.numeric(predict(sh$fit, newdata = gridX_sq, type="response"))
        data.frame(cluster = sh$id, u1 = UD$grid$u1, u2 = UD$grid$u2,
                   p = pmin(pmax(p,1e-6),1-1e-6))
      },
      gridX_sq = geom$gridX_sq, UD = geom$UD,
      future.seed = TRUE, future.globals = TRUE
    )
  )
  Fsq$cluster <- factor(Fsq$cluster, levels = labs)
  
  sig_list <- FUTURE_LAPPLY(
    shards,
    function(sh, gridX_sq, mask_sq, method, B, fwer, UD){
      if (tolower(method) != "perm") {
        pr  <- predict(sh$fit, newdata = gridX_sq, type = "link", se.fit = TRUE)
        eta <- as.numeric(pr$fit); se <- pmax(as.numeric(pr$se.fit), 1e-9)
        z   <- (eta - qlogis(sh$prev)) / se
        rbind(
          data.frame(kind="disk",   cluster = sh$id,
                     u1 = UD$grid$u1, u2 = UD$grid$u2,
                     z = {zd <- z; zd[!mask_sq] <- NA_real_; zd}, thr = 2.0),
          data.frame(kind="square", cluster = sh$id,
                     u1 = UD$grid$u1, u2 = UD$grid$u2,
                     z = z, thr = 2.0)
        )
      } else {
        S <- perm_thr_maxabs_z(sh$fit, sh$prev, gridX_sq,
                               mask_vec = mask_sq, B = B, fwer = fwer,
                               perm_grid_stride = 2L)
        data.frame(kind="disk", cluster = sh$id,
                   u1 = UD$grid$u1, u2 = UD$grid$u2,
                   z = S$z, thr = S$thr)
      }
    },
    gridX_sq = geom$gridX_sq, mask_sq = geom$mask_sq,
    method = cfg$sig_method, B = cfg$sig_B, fwer = cfg$sig_fwer, UD = geom$UD,
    future.seed = TRUE, future.globals = TRUE
  )
  Fsig <- dplyr::bind_rows(sig_list)
  Fsig$cluster <- factor(Fsig$cluster, levels = labs)
  
  list(
    Fstd = Fstd, Fbase = Fbase, Fsq = Fsq, Fsig = Fsig, labs = labs,
    U_pts = transform(geom$U, cluster = factor(ifelse(cl_vec==0L, NA, paste0("C", cl_vec)), levels = labs))
  )
}

# ---- Optional hulls in u-space -----------------------------------------------
make_cluster_hulls_u <- function(Udf, keep_levels, mask_col = "rin"){
  stopifnot(all(c("u1","u2","cluster") %in% names(Udf)))
  if (!mask_col %in% names(Udf)) Udf[[mask_col]] <- TRUE
  labs <- paste0("C", keep_levels)
  hulls <- lapply(labs, function(lbl){
    mask <- as.logical(Udf[[mask_col]]); mask[is.na(mask)] <- FALSE
    D <- Udf[Udf$cluster == lbl & mask, c("u1","u2"), drop = FALSE]
    D <- unique(stats::na.omit(D))
    if (nrow(D) < 3) return(NULL)
    idx <- grDevices::chull(D$u1, D$u2)
    data.frame(u1 = D$u1[idx], u2 = D$u2[idx], cluster = lbl, row.names = NULL)
  })
  do.call(rbind, hulls)
}

# ---- Cluster plots ------------------------------------------------------------
plots_for_clusters <- function(CF, geom, cfg, add_hulls = TRUE){
  br <- cfg$br_fixed
  Fsig_nd <- CF$Fsig |>
    dplyr::mutate(zunit = z / pmax(thr, 1e-9)) |>
    dplyr::filter(kind == "disk") |>
    dplyr::distinct(cluster, u1, u2, .keep_all = TRUE)
  
  hulls_u <- if (add_hulls) make_cluster_hulls_u(CF$U_pts, sub("^C","",CF$labs), mask_col = "rin") else NULL
  
  p_std <- ggplot(CF$Fstd, aes(u1, u2)) +
    geom_raster(aes(fill = p), interpolate = TRUE) +
    add_contours_safe(CF$Fstd, facet_col = "cluster", xcol = "u1", ycol = "u2", breaks = br) +
    geom_contour(data = Fsig_nd, aes(x = u1, y = u2, z = zunit, group = cluster),
                 breaks = 1, colour = "black", linetype = 3, linewidth = 0.35, alpha = .9) +
    geom_contour(data = Fsig_nd, aes(x = u1, y = u2, z = -zunit, group = cluster),
                 breaks = 1, colour = "black", linetype = 3, linewidth = 0.35, alpha = .9) +
    geom_point(data = subset(CF$U_pts, !is.na(cluster) & insq), aes(u1, u2), inherit.aes = FALSE,
               shape = 20, size = 0.30, colour = scales::alpha("black", 0.85), alpha = 1) +
    { if (!is.null(hulls_u)) geom_path(data = hulls_u, aes(u1, u2, group = cluster),
                                       inherit.aes = FALSE, colour = "black", linewidth = 0.45) } +
    coord_equal(xlim = c(-1,1), ylim = c(-1,1), expand = FALSE) +
    facet_wrap(~ cluster, ncol = 4) +
    scale_prob_fill(cfg) +
    labs(x = "u1 (whitened b1,b2)", y = "u2", fill = "Pr(cluster)") + theme_pub(11)
  
  p_base <- ggplot(CF$Fbase, aes(b1, b2)) +
    geom_raster(aes(fill = p), interpolate = TRUE, na.rm = TRUE) +
    geom_point(data = geom$df_base, aes(b1, b2), inherit.aes = FALSE, shape = 20, size = 0.28,
               colour = scales::alpha("black", 0.45), alpha = 1) +
    geom_contour(aes(z = p), breaks = br, colour = "white", linewidth = .28, alpha = .9, na.rm = TRUE) +
    geom_contour(aes(z = z, linetype = after_stat(ifelse(..level.. < 0, "neg", "pos"))),
                 breaks = c(-2,2), colour = "black", linewidth = 0.35, alpha = .9, na.rm = TRUE) +
    scale_linetype_manual(values = c(neg="dashed", pos="solid"), guide = "none") +
    scale_prob_fill(cfg) +
    coord_equal() + facet_wrap(~ cluster, ncol = 4) +
    labs(x = "b1", y = "b2", fill = "Pr(cluster)") + theme_pub(11)
  
  p_sq <- ggplot(CF$Fsq, aes(u1, u2)) +
    geom_raster(aes(fill = p), interpolate = TRUE) +
    add_contours_safe(CF$Fsq, facet_col = "cluster", xcol = "u1", ycol = "u2", breaks = br) +
    geom_contour(data = Fsig_nd, aes(x = u1, y = u2, z = zunit, group = cluster),
                 breaks = 1, colour = "black", linetype = 3, linewidth = 0.35, alpha = .9) +
    geom_contour(data = Fsig_nd, aes(x = u1, y = u2, z = -zunit, group = cluster),
                 breaks = 1, colour = "black", linetype = 3, linewidth = 0.35, alpha = .9) +
    geom_point(data = subset(geom$U, insq), aes(u1, u2), inherit.aes = FALSE,
               shape = 20, size = 0.30, colour = scales::alpha("black", 0.45), alpha = 1) +
    coord_equal(xlim = c(-1,1), ylim = c(-1,1), expand = FALSE) +
    facet_wrap(~ cluster, ncol = 4) +
    scale_prob_fill(cfg) +
    labs(x = "u1 (whitened b1,b2)", y = "u2", fill = "Pr(cluster)") + theme_pub(11)
  
  list(p_std = p_std, p_base = p_base, p_sq = p_sq)
}

# ---- 1D score per original item using first PC of its encoded columns --------
score_item_1d <- function(nm, Z, varmap){
  idx <- which(varmap == nm); if (!length(idx)) return(rep(NA_real_, nrow(Z)))
  v <- if (length(idx) == 1L) as.numeric(Z[, idx]) else {
    sc <- try(suppressWarnings(prcomp(Z[, idx, drop = FALSE], rank. = 1)$x[, 1]), silent = TRUE)
    if (inherits(sc, "try-error")) rep(NA_real_, nrow(Z)) else as.numeric(sc)
  }
  as.numeric(scale(v))
}

pearson_r <- function(a,b){
  if (length(a)!=length(b)) return(NA_real_)
  suppressWarnings(cor(as.numeric(a), as.numeric(b), use="complete.obs", method="pearson"))
}

build_biplot_data <- function(Z_A, varmap, Base_A, U){
  B1 <- Base_A[,1]; B2 <- Base_A[,2]
  items <- unique(varmap)
  Rtab <- dplyr::bind_rows(lapply(items, function(nm){
    v <- score_item_1d(nm, Z_A, varmap); if (!any(is.finite(v))) return(NULL)
    data.frame(item = nm,
               r_b1 = pearson_r(v, B1), r_b2 = pearson_r(v, B2),
               r_u1 = pearson_r(v, U$u1), r_u2 = pearson_r(v, U$u2),
               stringsAsFactors = FALSE)
  }))
  if (!nrow(Rtab)) stop("[biplot] No item correlations computed.")
  Rtab <- Rtab |>
    dplyr::mutate(across(c(r_b1, r_b2, r_u1, r_u2), ~ suppressWarnings(as.numeric(.))),
                  r_b1 = ifelse(is.finite(r_b1), r_b1, 0), r_b2 = ifelse(is.finite(r_b2), r_b2, 0),
                  r_u1 = ifelse(is.finite(r_u1), r_u1, 0), r_u2 = ifelse(is.finite(r_u2), r_u2, 0),
                  mag_r_base = sqrt(r_b1^2 + r_b2^2),
                  mag_r_disk = sqrt(r_u1^2 + r_u2^2))
  Rtab
}

plot_biplots <- function(Rtab, Base_A, U){
  B1n <- suppressWarnings(as.numeric(Base_A[,1])); B2n <- suppressWarnings(as.numeric(Base_A[,2]))
  B1n[!is.finite(B1n)] <- mean(B1n, na.rm = TRUE)
  B2n[!is.finite(B2n)] <- mean(B2n, na.rm = TRUE)
  Hidx <- grDevices::chull(B1n, B2n); H <- data.frame(b1 = B1n[Hidx], b2 = B2n[Hidx])
  
  cx <- mean(B1n); cy <- mean(B2n)
  Rscale <- 0.80 * min(diff(range(B1n)), diff(range(B2n)))
  n_arrows <- min(30L, nrow(Rtab))
  
  S_base <- Rtab |>
    dplyr::arrange(dplyr::desc(mag_r_base)) |>
    dplyr::slice_head(n = n_arrows) |>
    dplyr::mutate(x0 = cx, y0 = cy, x1 = cx + Rscale * r_b1, y1 = cy + Rscale * r_b2) |>
    dplyr::mutate(across(c(x0,y0,x1,y1), as.numeric))
  
  p_base <- ggplot() +
    geom_polygon(data = H, aes(b1, b2), fill = NA, colour = "black", linewidth = 0.4) +
    geom_segment(data = S_base, aes(x = x0, y = y0, xend = x1, yend = y1), linewidth = 0.8, colour = "firebrick",
                 arrow = grid::arrow(length = grid::unit(0.14, "cm"))) +
    ggrepel::geom_label_repel(
      data = transform(S_base, xlab = x1, ylab = y1),
      aes(x = xlab, y = ylab, label = item),
      size = 3.1, max.overlaps = Inf, label.size = 0,
      label.padding = grid::unit(0.10, "lines")
    ) +
    coord_equal() + labs(x = "b1", y = "b2") + theme_pub(12)
  
  Rdisk <- 0.85
  S_disk <- Rtab |>
    dplyr::arrange(dplyr::desc(mag_r_disk)) |>
    dplyr::slice_head(n = n_arrows) |>
    dplyr::transmute(item, u0 = 0, v0 = 0, u1 = as.numeric(Rdisk * r_u1), v1 = as.numeric(Rdisk * r_u2))
  
  p_disk <- ggplot() +
    geom_path(data = draw_disk_outline(), aes(x, y)) +
    geom_segment(data = S_disk, aes(x = u0, y = v0, xend = u1, yend = v1), linewidth = 0.8, colour = "firebrick",
                 arrow = grid::arrow(length = grid::unit(0.14, "cm"))) +
    ggrepel::geom_text_repel(data = transform(S_disk, xlab = u1, ylab = v1), aes(x = xlab, y = ylab, label = item),
                             size = 3.1, max.overlaps = Inf, label.padding = grid::unit(0.1, "lines")) +
    coord_equal(xlim = c(-1, 1), ylim = c(-1, 1), expand = FALSE) +
    labs(x = "u1 (whitened b1,b2)", y = "u2") + theme_pub(12)
  
  list(p_base = p_base, p_disk = p_disk)
}

# ---- OOF for Base, optional XR, and stacked model ----------------------------
oof_prob_stacked <- function(y, Base_A, XR = NULL, K, seed){
  y  <- as.integer(y > 0)
  Xb <- as.data.frame(Base_A)
  Xr <- if (!is.null(XR)) as.data.frame(XR) else NULL
  fid <- make_folds(y, K, seed)
  
  # base OOF
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
  
  # Meta OOF (only if we have XR)
  if (!is.null(pR)){
    pBR <- rep(NA_real_, length(y))
    for (k in seq_len(K)){
      tr <- fid != k; te <- fid == k
      Xm_tr <- data.frame(
        l1 = qlogis(pmin(pmax(pB[tr], 1e-6), 1 - 1e-6)),
        l2 = qlogis(pmin(pmax(pR[tr], 1e-6), 1 - 1e-6))
      )
      m <- stats::glm(y[tr] ~ ., data = Xm_tr, family = binomial())
      Xm_te <- data.frame(
        l1 = qlogis(pmin(pmax(pB[te], 1e-6), 1 - 1e-6)),
        l2 = qlogis(pmin(pmax(pR[te], 1e-6), 1 - 1e-6))
      )
      pBR[te] <- as.numeric(stats::predict(m, newdata = Xm_te, type = "response"))
    }
    out$Fibre <- pR
    out$Both  <- pmin(pmax(pBR, 1e-6), 1 - 1e-6)
  }
  out
}

# ---- Main loop: per-dx metrics, CSVs, PR & PR-Gain PDFs ----------------------
run_dx_metrics <- function(DxW_A, Base_A, XR = NULL, cfg, out_prefix = ""){
  Kdef <- cfg$cv_folds; seed <- cfg$seed_pred; Bci <- 1000L
  dx_keep <- names(DxW_A)
  AUPRC_ALL <- AUPRG_ALL <- AUC_ALL <- list()
  AECE_ALL <- list()
  BRIER_ALL <- list()     # <— add this collector
  
  # open a single PDF and print ALL panels into it
  grDevices::pdf(
    file.path(cfg$out_dir, paste0(out_prefix, "PR_per_diagnosis.pdf")),
    width = 6, height = 5, onefile = TRUE, bg = "white"
  )
  
  for (dx in dx_keep){
    y <- as.integer(DxW_A[[dx]] > 0)
    if (length(unique(y)) < 2L) next
    
    K <- max(2L, min(Kdef,
                     floor(sum(y==1) / Kdef),
                     floor(sum(y==0) / Kdef)))
    
    # OOF predictions for each variant (Base / Fibre / Both)
    models <- tryCatch(
      oof_prob_stacked(y, Base_A, XR, K = K, seed = seed),
      error = function(e) list(Base = rep(mean(y), length(y)))
    )
    
    # ---- evaluate & plot EACH model variant ----
    for (m in names(models)) {
      p <- models[[m]]  # no orient_scores() for PR/AUPRC/AUPRG
      
      # --- metrics + bootstrap CIs ---
      ci_pr  <- boot_ci(y, p, FUN = auprc_point, B = Bci, seed = seed)
      ci_auc <- boot_ci(y, p, FUN = auc_point,   B = Bci, seed = seed)
      ci_prg <- boot_ci(y, p, FUN = function(yy,pp) prg_flach(yy,pp)$auprg, B = Bci, seed = seed)
      
      AUPRC_ALL[[length(AUPRC_ALL)+1L]] <- data.frame(
        dx = dx, model = m, prevalence = mean(y),
        AUPRC = as.numeric(ci_pr["point"]),
        lo = as.numeric(ci_pr["lo"]), hi = as.numeric(ci_pr["hi"])
      )
      AUC_ALL[[length(AUC_ALL)+1L]] <- data.frame(
        dx = dx, model = m, prevalence = mean(y),
        AUC = as.numeric(ci_auc["point"]),
        lo = as.numeric(ci_auc["lo"]), hi = as.numeric(ci_auc["hi"])
      )
      AUPRG_ALL[[length(AUPRG_ALL)+1L]] <- data.frame(
        dx = dx, model = m, prevalence = mean(y),
        AUPRG = as.numeric(ci_prg["point"]),
        lo = as.numeric(ci_prg["lo"]), hi = as.numeric(ci_prg["hi"])
      )
      # aECE (point + bootstrap CI)
      ae      <- robust_aece(y, p)  # returns list(aece, n_bins)
      ci_ece  <- boot_ci(y, p,
                         FUN  = function(yy, pp) robust_aece(yy, pp)$aece,
                         B    = Bci,
                         seed = seed)
      
      AECE_ALL[[length(AECE_ALL) + 1L]] <- data.frame(
        dx         = dx,
        model      = m,
        prevalence = mean(y),
        aECE       = as.numeric(ci_ece["point"]),
        lo         = as.numeric(ci_ece["lo"]),
        hi         = as.numeric(ci_ece["hi"]),
        n_bins     = as.integer(ae$n_bins)
      )
      
      # --- Brier R^2 (skill) -------------------------------------------
      ci_br <- boot_brier_R2(y, p, B = Bci, seed = seed)
      BRIER_ALL[[length(BRIER_ALL) + 1L]] <- data.frame(
        dx         = dx,
        model      = m,
        prevalence = mean(y),
        R2_Brier   = as.numeric(ci_br["point"]),
        lo         = as.numeric(ci_br["lo"]),
        hi         = as.numeric(ci_br["hi"]),
        stringsAsFactors = FALSE
      )
      
      # --- PR curve ---
      if (requireNamespace("PRROC", quietly = TRUE) && any(is.finite(p))){
        rr <- PRROC::pr.curve(scores.class0 = p[y==1], scores.class1 = p[y==0], curve = TRUE)
        prdf <- data.frame(recall = rr$curve[,1], precision = rr$curve[,2])
        g_pr <- ggplot(prdf, aes(recall, precision)) +
          geom_path() +
          geom_hline(yintercept = mean(y), linetype = 3) +
          coord_equal(xlim = c(0,1), ylim = c(0,1), expand = FALSE) +
          labs(title = paste0("PR — ", dx, " (", m, ")"),
               x = "Recall", y = "Precision") +
          theme_minimal(12)
        print(g_pr)
      }
      
      # --- PR-Gain curve ---
      PG <- prg_flach(y, p)$curve
      if (nrow(PG)){
        g_prg <- ggplot(PG, aes(recG, precG)) +
          geom_path() +
          geom_abline(slope = -1, intercept = 1,   linetype = 3, linewidth = 0.35) +
          geom_abline(slope = -1, intercept = 0.75, linetype = 3, linewidth = 0.35) +
          geom_abline(slope = -1, intercept = 0.50, linetype = 3, linewidth = 0.35) +
          coord_equal(xlim = c(0,1), ylim = c(0,1), expand = FALSE) +
          labs(title = paste0("PR-Gain — ", dx, " (", m, ")"),
               x = "Recall gain", y = "Precision gain") +
          theme_minimal(12)
        print(g_prg)
      }
    }
  }
  grDevices::dev.off()
  
  # write CSVs (no clamping of AUPRG)
  if (length(AUPRC_ALL))
    utils::write.csv(dplyr::bind_rows(AUPRC_ALL),
       file.path(cfg$out_dir, paste0(out_prefix, "AUPRC_bootstrap_CI.csv")),
       row.names = FALSE)
  if (length(AUPRG_ALL))
    utils::write.csv(dplyr::bind_rows(AUPRG_ALL),
     file.path(cfg$out_dir, paste0(out_prefix, "AUPRG_bootstrap_CI.csv")),
     row.names = FALSE)
  if (length(AUC_ALL))
    utils::write.csv(dplyr::bind_rows(AUC_ALL),
     file.path(cfg$out_dir, paste0(out_prefix, "AUC_bootstrap_CI.csv")),
     row.names = FALSE)
  if (length(AECE_ALL))
    utils::write.csv(
      dplyr::bind_rows(AECE_ALL),
      file.path(cfg$out_dir, paste0(out_prefix, "AECE_bootstrap_CI.csv")),
      row.names = FALSE
    )
  # Brier R² (additive vs full) in paper-style table
  Btab <- write_brierR2_table(
    DxW_A, Base_A, K = cfg$cv_folds, seed = cfg$seed_pred,
    B = 1000L,
    out_csv = file.path(cfg$out_dir, "BrierR2_table.csv")
  )
  print(Btab)
}

# Monotone calibration curve per diagnosis (facetted), with optional bin points
calibration_curve_dx <- function(DxW_A, Base_A, cfg, XR = NULL,
                                 show_points = TRUE, band_B = 0L){
  dx_keep <- names(DxW_A)
  curves  <- list()
  ptslist <- list()
  bands   <- list()
  set.seed(cfg$seed_pred)
  
  # helper: isotonic step coordinates + (optional) bootstrap band
  .iso_curve <- function(p, y, B = 0L){
    pclip <- pmin(pmax(p, 1e-6), 1 - 1e-6)
    lp <- qlogis(pclip)
    pc <- plogis(lp + (qlogis(mean(y)) - mean(lp)))  # calib-in-the-large
    if (!any(is.finite(pc)) || length(unique(pc[is.finite(pc)])) < 2L) return(NULL)
    
    o  <- order(pc); xs <- pc[o]; ys <- y[o]
    iso <- stats::isoreg(xs, ys)                      # monotone on observed x
    S   <- aggregate(yf ~ x, data.frame(x = iso$x, yf = iso$yf), mean)
    
    out <- list(step = dplyr::rename(S, p = x, y = yf))  # <- yf exists here
    
    if (B > 0L) {
      xg <- seq(min(xs), max(xs), length.out = 101)      # band only where we have support
      eval_iso <- function(x, y){
        o  <- order(x); xi <- x[o]; yi <- y[o]
        ii <- stats::isoreg(xi, yi)
        f  <- stats::approxfun(ii$x, ii$yf, method = "constant", f = 1,
                               yleft = ii$yf[1L], yright = ii$yf[length(ii$yf)])
        f(xg)
      }
      i0 <- which(ys == 0L); i1 <- which(ys == 1L)
      M  <- replicate(B, {
        jj <- c(sample(i0, length(i0), TRUE), sample(i1, length(i1), TRUE))
        eval_iso(xs[jj], ys[jj])
      })
      lo <- apply(M, 1, stats::quantile, 0.025, na.rm = TRUE)
      hi <- apply(M, 1, stats::quantile, 0.975, na.rm = TRUE)
      out$band <- data.frame(p = xg, y_lo = lo, y_hi = hi)
    }
    out
  }
  
  # equal-count bins (to optionally overlay points)
  .bins_df <- function(pc, y){
    B <- make_calib_bins(pc, n_min = 10L, max_bins = 8L)
    g <- B$g
    T <- data.frame(
      n    = as.integer(tapply(y, g, length)),
      ybar = as.numeric(tapply(y, g, mean)),
      phat = as.numeric(tapply(pc, g, mean))
    )
    T[is.finite(T$n) & T$n > 0, , drop = FALSE]
  }
  
  for (dx in dx_keep){
    y <- as.integer(DxW_A[[dx]] > 0)
    if (length(unique(y)) < 2L) next
    
    K <- max(2L, min(cfg$cv_folds, floor(sum(y==1)/8), floor(sum(y==0)/8)))
    p <- oof_prob(y, Base_A, K = K, seed = cfg$seed_pred)
    
    # calib-in-the-large pc for points overlay
    pclip <- pmin(pmax(p, 1e-6), 1 - 1e-6)
    pc <- plogis(qlogis(pclip) + (qlogis(mean(y)) - mean(qlogis(pclip))))
    
    ISO <- .iso_curve(p, y, B = band_B)
    if (is.null(ISO)) next
    curves[[dx]] <- transform(ISO$step, dx = dx)
    
    if (show_points) {
      T <- .bins_df(pc, y)
      if (nrow(T)) ptslist[[dx]] <- data.frame(dx = dx, n = T$n, p_hat = T$phat, y_bar = T$ybar)
    }
    if (!is.null(ISO$band)) bands[[dx]] <- transform(ISO$band, dx = dx)
  }
  
  if (!length(curves)) return(NULL)
  CUR <- dplyr::bind_rows(curves)
  PTS <- if (length(ptslist)) dplyr::bind_rows(ptslist) else NULL
  BND <- if (length(bands))   dplyr::bind_rows(bands)   else NULL
  
  gg <- ggplot() +
    geom_abline(slope = 1, intercept = 0, linetype = 3, linewidth = 0.35, colour = "grey45") +
    { if (!is.null(BND)) geom_ribbon(data = BND, aes(x = p, ymin = y_lo, ymax = y_hi),
                                     inherit.aes = FALSE, alpha = 0.18) } +
    geom_step(data = CUR, aes(x = p, y = y), direction = "vh", linewidth = 0.7) +
    { if (!is.null(PTS) && show_points)
      geom_point(data = PTS, aes(x = p_hat, y = y_bar), size = 1.6,
                 shape = 21, stroke = 0.35, fill = "black", colour = "white", alpha = 0.95,
                 show.legend = FALSE) } +
    facet_wrap(~ dx, ncol = 4) +
    coord_equal(xlim = c(0,1), ylim = c(0,1), expand = FALSE) +
    labs(x = "Predicted probability (calib-in-the-large)",
         y = "Observed frequency") +
    geom_rug(data = CUR, aes(x = p), sides = "b", alpha = 0.25, linewidth = 0.2, inherit.aes = FALSE) +
    theme_pub(11) +
    theme(legend.position = if (!is.null(PTS) && show_points) "right" else "none")
  gg
}

# ---- 10.1 Neighbour stability under Gaussian noise ---------------------------
knn_idx <- function(X, k){
  if (requireNamespace("RANN", quietly = TRUE)) {
    nn <- RANN::nn2(X, query = X, k = k + 1L)
    idx <- nn$nn.idx[, -1, drop = FALSE] # drop self
    return(idx)
  } else {
    D <- as.matrix(dist(X)); diag(D) <- Inf
    t(apply(D, 1, function(d) order(d)[1:k]))
  }
}

knn_stability_plot <- function(Base_A, k = 15L,
                               sd_grid = c(0, 0.05, 0.10, 0.15, 0.20),
                               R = 600L, seed = 42L){
  B <- tryCatch(as.matrix(Base_A[, 1:2]), error = function(e) NULL)
  if (is.null(B)) return(NULL)
  storage.mode(B) <- "double"
  
  jaccard_row <- function(a, b){
    inter <- length(intersect(a, b)); uni <- length(union(a, b))
    if (uni == 0) 1 else inter / uni
  }
  mean_jaccard <- function(N0, N1){
    n <- nrow(N0); mean(vapply(seq_len(n), function(i) jaccard_row(N0[i,], N1[i,]), numeric(1)))
  }
  
  set.seed(seed)
  N0 <- knn_idx(B, k = k)
  rows <- lapply(sd_grid, function(sigma){
    vals <- if (sigma == 0) {
      rep(1, R)
    } else {
      FUTURE_SAPPLY(
        seq_len(R),
        function(i, B, k){
          Bb <- B + matrix(rnorm(nrow(B)*ncol(B), 0, sigma), nrow(B), ncol(B))
          N1 <- knn_idx(Bb, k = k)
          mean_jaccard(N0, N1)
        },
        B = B, k = k,
        future.seed = TRUE, future.globals = TRUE
      )
    }
    c(mean = mean(vals),
      lo = as.numeric(quantile(vals, 0.025, names = FALSE)),
      hi = as.numeric(quantile(vals, 0.975, names = FALSE)))
  })
  ST <- as.data.frame(do.call(rbind, rows))
  ST$noise <- sd_grid
  names(ST) <- c("mean_jaccard","lo","hi","noise")
  
  p <- ggplot(ST, aes(x = noise, y = mean_jaccard)) +
    geom_ribbon(aes(ymin = lo, ymax = hi), alpha = 0.18, fill = "grey50") +
    geom_line() + geom_point() +
    scale_x_continuous(breaks = sd_grid, limits = range(sd_grid)) +
    scale_y_continuous(limits = c(0.7, 1), breaks = seq(0.7, 1, 0.05)) +
    labs(x = "Noise σ (Base units)", y = sprintf("Mean Jaccard (k=%d)", k)) +
    theme_pub(12)
  list(data = ST, plot = p)
}

knn_stability_plot_kband <- function(Base_A,
                                     k_range = 10:20,
                                     sd_grid = c(0, 0.05, 0.10, 0.15, 0.20),
                                     R = 600L, seed = 42L){
  set.seed(seed)
  B <- as.matrix(Base_A[, 1:2])
  if (!is.matrix(B) || ncol(B) != 2) stop("Base_A must have 2 columns")
  N <- nrow(B)
  
  res <- lapply(sd_grid, function(sigma){
    J_per_point <- replicate(R, {
      noise <- matrix(rnorm(N * 2, 0, sigma), N, 2)
      Bp <- B + noise
      rowMeans(sapply(k_range, function(k){
        N0 <- knn_idx(B,  k = k)
        N1 <- knn_idx(Bp, k = k)
        sapply(seq_len(N), function(i){
          a <- N0[i,]; b <- N1[i,]
          inter <- length(intersect(a,b))
          union <- length(unique(c(a,b)))
          if (union == 0) 1 else inter/union
        })
      }))
    })
    rowMeans(J_per_point)
  })
  
  M <- do.call(cbind, res)
  
  stats <- apply(M, 2, function(col) c(
    mean = mean(col),
    p10  = unname(quantile(col, 0.10, names = FALSE)),
    p90  = unname(quantile(col, 0.90, names = FALSE))
  ))
  stats <- t(stats)
  colnames(stats) <- c("mean","p10","p90")
  
  list(sd = sd_grid, stats = stats, per_point = M)   # ← no extra t()
}

# ---- 10.2 Co-occurrence lift (upper triangle, diagonal=1) --------------------
lift_map_plot <- function(DxW_A, order = c("original","prevalence","cluster"),
                          adjust_p = FALSE, title = "Co-occurrence (lift)"){
  order <- match.arg(order)
  if (is.null(DxW_A) || !ncol(DxW_A)) return(NULL)
  
  M  <- as.matrix(DxW_A > 0)
  dx_all <- colnames(M); n <- nrow(M)
  p  <- colMeans(M)
  
  keep <- which(is.finite(p) & p > 0 & p < 1)
  if (!length(keep)) {
    return(ggplot() + labs(title = paste0(title, " — no eligible diagnoses"), x = NULL, y = NULL) +
             theme_minimal(12) + theme(panel.grid = element_blank()))
  }
  M <- M[, keep, drop = FALSE]
  dx <- dx_all[keep]; p <- p[keep]
  
  Pij <- (t(M) %*% M) / n
  L   <- Pij / (p %o% p)
  
  ord <- switch(order,
                prevalence = order(p, decreasing = TRUE),
                cluster    = { S <- suppressWarnings(cor(M)); S[!is.finite(S)] <- 0; hclust(as.dist(1 - S), "average")$order },
                original   = seq_along(dx))
  dx <- dx[ord]; p <- p[ord]; L <- L[ord, ord, drop = FALSE]
  
  L_pairs <- L; diag(L_pairs) <- NA_real_; L_pairs[lower.tri(L_pairs, TRUE)] <- NA_real_
  U <- as.data.frame(as.table(L_pairs), responseName = "lift") |>
    dplyr::rename(r = Var1, c = Var2) |>
    dplyr::filter(!is.na(lift))
  
  ij <- U |> dplyr::transmute(i = match(r, dx), j = match(c, dx))
  pv <- FUTURE_MAPPLY(
    function(i, j, M){
      a <- sum(M[, i] & M[, j]); b <- sum(M[, i] & !M[, j])
      c <- sum(!M[, i] & M[, j]); d <- sum(!M[, i] & !M[, j])
      mat <- matrix(c(a,b,c,d), 2, 2)
      if (any(mat < 0) || sum(mat) == 0) return(NA_real_)
      suppressWarnings(fisher.test(mat)$p.value)
    },
    ij$i, ij$j, MoreArgs = list(M = M),
    future.seed = TRUE, future.globals = TRUE
  )
  pv <- if (adjust_p) p.adjust(pv, "BH") else pv
  U$star <- ifelse(is.finite(pv) & pv < .01, "**",
                   ifelse(is.finite(pv) & pv < .05, "*", "ns"))
  
  cap_hi <- stats::quantile(U$lift, 0.99, na.rm = TRUE)
  cap_lo <- min(U$lift, na.rm = TRUE)
  U <- U |>
    dplyr::mutate(
      lift_clip = pmin(pmax(lift, cap_lo), cap_hi),
      s = (lift_clip - cap_lo) / (cap_hi - cap_lo + 1e-12),
      star_col = ifelse(s > .55, "black", "white")
    )
  
  D <- tibble::tibble(r = dx, c = dx, lift = 1, star = "–", star_col = "white")
  U$r <- factor(U$r, levels = dx); U$c <- factor(U$c, levels = dx)
  D$r <- factor(D$r, levels = dx); D$c <- factor(D$c, levels = dx)
  
  ggplot() +
    geom_tile(data = D, aes(c, r, fill = lift), colour = NA) +
    geom_tile(data = U, aes(c, r, fill = lift), colour = NA) +
    geom_text(data = U, aes(c, r, label = star, colour = star_col),
              fontface = "bold", size = 3.6) +
    scale_colour_identity() +
    scale_fill_viridis_c(name = "lift") +
    coord_equal(expand = FALSE) +
    labs(title = title, x = NULL, y = NULL) +
    theme_minimal(base_size = 12) +
    theme(panel.grid = element_blank(),
          axis.text.x = element_text(angle = 35, hjust = 1, vjust = 1),
          axis.ticks = element_blank(),
          plot.title = element_text(face = "bold", size = 16, hjust = .5))
}

# ---- 10.3 Figure 2 (A: unit-square scatter; B: six dx base contours) --------
# Expects: geom list (U, df_base, gridB_full, mask_hull), Fbase_dx (dx, b1,b2,p)
figure2_plots <- function(geom, Fbase_dx, br_fixed){
  U_sq <- subset(geom$U, insq)
  fig2A <- ggplot() +
    stat_density_2d(data = U_sq, aes(u1, u2, colour = after_stat(level)), bins = 6, linewidth = 0.45, alpha = 0.9) +
    scale_prob_fill(cfg) +
    geom_point(data = U_sq, aes(u1, u2), shape = 16, size = 0.9, colour = scales::alpha("black", 0.35)) +
    annotate("rect", xmin = -1, xmax = 1, ymin = -1, ymax = 1, fill = NA, colour = "black", linewidth = 0.35) +
    coord_equal(xlim = c(-1, 1), ylim = c(-1, 1), expand = FALSE) +
    labs(x = "u1 (whitened b1,b2)", y = "u2") + theme_pub(12)
  
  dx_stats <- Fbase_dx |>
    dplyr::group_by(dx) |>
    dplyr::summarise(n = dplyr::n(),
                     range_p = diff(range(p, na.rm = TRUE)),
                     iqr_p = IQR(p, na.rm = TRUE), .groups = "drop")
  dx_keep6 <- dx_stats |> dplyr::arrange(dplyr::desc(iqr_p), dplyr::desc(range_p), dx) |> dplyr::slice_head(n = 6) |> dplyr::pull(dx)
  F6 <- Fbase_dx |> dplyr::filter(dx %in% dx_keep6, is.finite(p)) |> dplyr::mutate(dx = droplevels(factor(dx)))
  
  fig2B <- ggplot(F6, aes(b1, b2)) +
    geom_point(data = geom$df_base, aes(b1, b2), inherit.aes = FALSE,
               shape = 16, size = 0.7, colour = scales::alpha("black", 0.25), na.rm = TRUE) +
    geom_contour(aes(z = p), breaks = br_fixed, colour = "white",
                 linewidth = 0.50, na.rm = TRUE) +
    coord_equal() + facet_wrap(~ dx, ncol = 3, scales = "fixed") +
    labs(x = "b1", y = "b2") + theme_pub(12) +
    theme(strip.text = element_text(size = 11, face = "bold"))
  
  list(fig2A = fig2A, fig2B = fig2B)
}

# ---- 10.4 Direction wheel (HCL, smoothed density) ---------------------------
direction_wheel_plot <- function(geom){
  U_sq <- subset(geom$U, insq)
  if (!nrow(U_sq)) return(NULL)
  suppressPackageStartupMessages({ library(MASS) })
  
  nu <- 500; pad <- 0.75
  gx <- seq(-1 - pad, 1 + pad, length.out = nu)
  gy <- seq(-1 - pad, 1 + pad, length.out = nu)
  kd <- with(U_sq, MASS::kde2d(u1, u2, n = nu, lims = c(-1 - pad, 1 + pad, -1 - pad, 1 + pad)))
  D <- kd$z
  D <- log1p(D / max(D, na.rm = TRUE))
  D <- D / quantile(D, 0.99, na.rm = TRUE)
  D[D>1] <- 1; D[D<0] <- 0
  ALPHA <- D^0.70
  
  G <- expand.grid(u1 = gx, u2 = gy)
  theta <- atan2(G$u2, G$u1); r <- sqrt(G$u1^2 + G$u2^2)
  H0 <- -200; L0 <- 60; Cmax <- 90; betaC <- 0.70
  H <- (H0 + theta * 180/pi) %% 360
  C <- pmin(Cmax * (pmin(r, 1)^betaC), Cmax)
  L <- pmax(0, pmin(100, L0 - 6*(pmin(r,1)^1.1)))
  G$fill  <- grDevices::hcl(H, C, L)
  G$alpha <- as.vector(ALPHA)
  
  feather1d <- function(x, lo = -1, hi = 1, w = 0.08){
    tL <- pmin(pmax((x - lo)/w, 0), 1); tR <- pmin(pmax((hi - x)/w, 0), 1)
    fL <- (cos(tL * pi/2))^2; fR <- (cos(tR * pi/2))^2; pmin(fL, fR)
  }
  Fx <- feather1d(G$u1, lo = -1, hi =  1, w = 0.08)
  Fy <- feather1d(G$u2, lo = -1, hi =  1, w = 0.08)
  G$alpha <- G$alpha * Fx * Fy
  
  anchor <- data.frame(x=c(0.85,-0.85, 0.02, 0.02),
                       y=c(0.02,0.02, 0.85,-0.85),
                       lab=c("+b1","-b1","+b2","-b2"),
                       col=grDevices::hcl((H0 + c(0,180,90,-90)) %% 360, Cmax, L0))
  ggplot() +
    geom_raster(data = G, aes(u1, u2, fill = I(fill), alpha = alpha), interpolate = TRUE) +
    scale_alpha(range = c(0,1), guide = "none") +
    geom_point(data = U_sq, aes(u1, u2), shape = 16, size = 0.8, colour = scales::alpha("black", 0.32)) +
    coord_equal(xlim = c(-1.5, 1.5), ylim = c(-1.5, 1.5), expand = FALSE, clip = "on") +
    geom_point(data = anchor, aes(x, y), shape = 15, size = 3, colour = anchor$col) +
    geom_text(data = anchor, aes(x, y, label = lab), nudge_x = 0.07, size = 3.2) +
    labs(x = "u1 (whitened b1,b2)", y = "u2") + theme_pub(12)
}

# Uses your Section 3 metrics: auc_point(), auprc_point(), prg_flach(), boot_ci()
# If any of those aren’t present, define light fallbacks before calling this.

base_performance_table <- function(DxW_A, Base_A, cfg, out_prefix = "TAB_"){
  if (!ncol(DxW_A)) return(NULL)
  CV  <- cfg$cv_folds; SEED <- cfg$seed_pred; B_CI <- 1000L
  
  make_stratified_folds <- function(y, K, seed){
    y <- as.integer(y > 0)
    set.seed(seed)
    i0 <- which(y == 0L); i1 <- which(y == 1L)
    f0 <- sample(rep(seq_len(K), length.out = length(i0)))
    f1 <- sample(rep(seq_len(K), length.out = length(i1)))
    fid <- integer(length(y)); fid[i0] <- f0; fid[i1] <- f1; fid
  }
  oof_prob_safe <- function(y, X, K, seed){
    y <- as.integer(y > 0); Xdf <- as.data.frame(X)
    fid <- make_stratified_folds(y, K, seed); p <- rep(NA_real_, length(y))
    for (k in seq_len(K)){
      tr <- fid != k; te <- fid == k
      sd_tr <- vapply(Xdf[tr,,drop=FALSE], function(v) sd(as.numeric(v), na.rm=TRUE), 0)
      keep  <- which(is.finite(sd_tr) & sd_tr > 0)
      if (!length(keep)) { p[te] <- mean(y[tr]); next }
      f <- try(stats::glm(y ~ ., data = data.frame(y=y[tr], Xdf[tr, keep, drop=FALSE]),
                          family = stats::binomial()), silent = TRUE)
      if (inherits(f, "try-error")) { p[te] <- mean(y[tr]); next }
      p[te] <- as.numeric(stats::predict(f, newdata = Xdf[te, keep, drop=FALSE], type="response"))
    }
    p[!is.finite(p)] <- mean(y)
    pmin(pmax(p, 1e-6), 1 - 1e-6)
  }
  
  rows <- list()
  diag_dbg <- list()
  
  for (dx in names(DxW_A)){
    y <- as.integer(DxW_A[[dx]] > 0)
    n1 <- sum(y==1L); n0 <- sum(y==0L)
    if (length(unique(y)) < 2L) {
      diag_dbg[[dx]] <- data.frame(dx, n=length(y), n1, n0, reason="one_class")
      next
    }
    K <- max(2L, min(CV, floor(n1/8), floor(n0/8)))
    pB <- oof_prob_safe(y, Base_A, K = K, seed = SEED)
    
    rng <- diff(range(pB, na.rm = TRUE))
    if (!is.numeric(pB) || anyNA(pB) || !is.finite(rng) || rng < 1e-8) {
      diag_dbg[[dx]] <- data.frame(dx, n=length(y), n1, n0, reason="degenerate_pB", rng_pB = rng)
      next
    }
    
    ci_auc <- boot_ci(y, pB, FUN = auc_point,   B = B_CI, seed = SEED)
    ci_pr  <- boot_ci(y, pB, FUN = auprc_point, B = B_CI, seed = SEED)
    ci_prg <- boot_ci(y, pB, FUN = function(yy,pp) prg_flach(yy,pp)$auprg, B = B_CI, seed = SEED)
    
    ae    <- robust_aece(y, pB)
    ci_ece  <- boot_ci(y, pB,
                       FUN  = function(yy, pp) robust_aece(yy, pp)$aece,
                       B    = B_CI,
                       seed = SEED)
    AECE_mean <- as.numeric(ci_ece["point"])
    AECE_lo   <- as.numeric(ci_ece["lo"])
    AECE_hi   <- as.numeric(ci_ece["hi"])
    
    prev  <- mean(y)
    
    clamp01 <- function(x) pmin(pmax(x, 0), 1)
    ap_pt   <- as.numeric(ci_pr["point"])
    ap_lo   <- as.numeric(ci_pr["lo"]); ap_hi <- as.numeric(ci_pr["hi"])
    apn_pt  <- if (is.finite(ap_pt) && prev < 1) (ap_pt  - prev)/max(1e-9, 1 - prev) else NA_real_
    apn_lo  <- if (is.finite(ap_lo) && prev < 1) (ap_lo  - prev)/max(1e-9, 1 - prev) else NA_real_
    apn_hi  <- if (is.finite(ap_hi) && prev < 1) (ap_hi  - prev)/max(1e-9, 1 - prev) else NA_real_
    apn_pt  <- clamp01(apn_pt); apn_lo <- clamp01(apn_lo); apn_hi <- clamp01(apn_hi)
    
    rows[[length(rows)+1L]] <- data.frame(
      Diagnosis   = dx,
      n_pos       = n1,
      n_bins      = ae$n_bins,
      prevalence  = prev,
      AUC_mean    = as.numeric(ci_auc["point"]),
      AUC_lo      = as.numeric(ci_auc["lo"]),
      AUC_hi      = as.numeric(ci_auc["hi"]),
      AUPRC_mean  = ap_pt,
      AUPRC_lo    = ap_lo,
      AUPRC_hi    = ap_hi,
      AUPRCn_mean = apn_pt,
      AUPRCn_lo   = apn_lo,
      AUPRCn_hi   = apn_hi,
      AUPRG_mean  = as.numeric(ci_prg["point"]),
      AUPRG_lo    = as.numeric(ci_prg["lo"]),
      AUPRG_hi    = as.numeric(ci_prg["hi"]),
      AECE_mean   = AECE_mean,
      AECE_lo     = AECE_lo,
      AECE_hi     = AECE_hi,
      stringsAsFactors = FALSE
    )
    diag_dbg[[dx]] <- data.frame(dx, n=length(y), n1, n0, reason="ok")
  }
  
  tbl <- if (length(rows)) dplyr::bind_rows(rows) else return(NULL)
  dbg <- if (length(diag_dbg)) dplyr::bind_rows(diag_dbg) else data.frame()
  if (nrow(dbg)) utils::write.csv(dbg, file.path(cfg$out_dir, paste0(out_prefix, "base_perf_diag.csv")), row.names = FALSE)
  
  pretty_tbl <- tbl %>%
    dplyr::transmute(
      Diagnosis, n_pos, n_bins,
      `AUC [95% CI]`        = sprintf("%.3f [%.3f, %.3f]", AUC_mean,    AUC_lo,    AUC_hi),
      `AUPRC [95% CI]`      = sprintf("%.3f [%.3f, %.3f]", AUPRC_mean,  AUPRC_lo,  AUPRC_hi),
      `AUPRC (norm) [95%]`  = sprintf("%.3f [%.3f, %.3f]", AUPRCn_mean, AUPRCn_lo, AUPRCn_hi),
      `AUPRG [95% CI]`      = sprintf("%.3f [%.3f, %.3f]", AUPRG_mean,  AUPRG_lo,  AUPRG_hi),
      Baseline              = sprintf("%.3f", prevalence),
      `aECE [95% CI]`       = sprintf("%.3f [%.3f, %.3f]", AECE_mean,   AECE_lo,   AECE_hi)
    ) %>%
    dplyr::arrange(dplyr::desc(n_pos))
  
  if (requireNamespace("flextable", quietly = TRUE) && requireNamespace("officer", quietly = TRUE)) {
    ft <- flextable::flextable(pretty_tbl) |>
      flextable::theme_vanilla() |>
      flextable::fontsize(part = "all", size = 10) |>
      flextable::bold(part = "header", bold = TRUE) |>
      flextable::align(j = 2:8, align = "center", part = "all") |>
      flextable::autofit()
    flextable::save_as_docx("Performance_Base_only" = ft,
                            path = file.path(cfg$out_dir, paste0(out_prefix, "perf_base.docx")))
  }
  utils::write.csv(pretty_tbl, file.path(cfg$out_dir, paste0(out_prefix, "perf_base.csv")), row.names = FALSE)
  invisible(pretty_tbl)
}

# ====================== SEQUENTIAL DRIVER (no wrapper) =======================

B_coords <- as.matrix(Base[, 1:2, drop = FALSE])
colnames(B_coords) <- c("B1","B2")

# tell the encoder which Base axes these are:
attr(B_coords, "varmap") <- c(B1 = "b1", B2 = "b2")
# (alternatively, some versions accept integer indices)
# attr(B_coords, "varmap") <- c(B1 = 1L, B2 = 2L)

stopifnot(identical(rownames(B_coords), rownames(Base)))

# 0) Config (use your default_cfg already defined)
cfg <- default_cfg()
cfg$out_dir <- "out"
cfg$verbose <- TRUE
cfg$add_u_hulls <- TRUE
cfg$make_lift_maps <- FALSE
cfg$sig_B <- 50L
cfg$nu_unit <- 400L
cfg$n_base_grid <- 140L
dir.create(cfg$out_dir, showWarnings = FALSE, recursive = TRUE)

cfg$palette <- list(
  engine    = "scico",   # "scico" | "colorspace" | "brewer" | "paletteer" | "manual"
  name      = "lapaz", # e.g. scico: "lajolla","batlow","oslo" ; brewer: "YlGnBu"
  direction = 1,         # 1 or -1 (reverse)
  colours   = NULL       # only used if engine == "manual"
)

Sys.setenv(
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS      = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  BLAS_NUM_THREADS     = "1",
  OMP_NUM_THREADS      = "1"
)

# If any of your helpers internally use future.apply, force sequential:
# if (requireNamespace("future", quietly = TRUE)) {
#   future::plan(future::sequential)
# }
set_future_plan(cfg)

# 1) Encode / align ------------------------------------------------------------
ENC    <- encode_Z(Base = Base, X_pred = X_pred, Z = Z %||% NULL, w_all = w_all %||% NULL)
Base_A <- ENC$Base_A
Z_A    <- ENC$Z_A
varmap <- ENC$varmap

# 2) Geometry & grids ----------------------------------------------------------
geom <- build_geometry(Base_A, cfg)

# 3) Choose diagnoses & align wide labels -------------------------------------
pos <- colSums(DxW > 0, na.rm = TRUE); neg <- colSums(DxW == 0, na.rm = TRUE)
keep_dx <- names(which(pos >= cfg$dx_min_pos & neg >= cfg$dx_min_neg))
if (!length(keep_dx)) keep_dx <- names(which(pos > 0 & neg > 0))
DxW_A <- as.data.frame(DxW)[, keep_dx, drop = FALSE]
rownames(DxW_A) <- rownames(Base_A)

# 4) Fit per-dx models, predict fields, make plots ----------------------------
DXFIT   <- fit_dx_gams(DxW_A, Base_A, dx_keep = keep_dx, cfg = cfg)
DXGRIDS <- predict_dx_fields(DXFIT$fits, DXFIT$prev, geom, cfg)
DXPL    <- plots_for_dx(DXGRIDS, geom, cfg)

p1 <- save_plot("FIG_dx_FIELDS_UNITDISK.png",   DXPL$p_std,  180, 150, cfg)
p2 <- save_plot("FIG_dx_FIELDS_BASE.png",       DXPL$p_base, 180, 150, cfg)
p3 <- save_plot("FIG_dx_FIELDS_UNITSQUARE.png", DXPL$p_sq,   180, 150, cfg)

# 5) Clusters (optional) -------------------------------------------------------
artefacts <- list(
  FIG_dx_FIELDS_UNITDISK   = p1,
  FIG_dx_FIELDS_BASE       = p2,
  FIG_dx_FIELDS_UNITSQUARE = p3
)

if (exists("clusters_csv") && !is.null(clusters_csv)) {
  cl_vec   <- read_clusters(clusters_csv, Base_A)
  keep_cls <- select_clusters(cl_vec, min_pos = cfg$dx_min_pos, min_neg = cfg$dx_min_neg)
  CLFIT    <- fit_cluster_gams(cl_vec, Base_A, keep_cls)
  CLGRIDS  <- predict_cluster_fields(CLFIT$fits, CLFIT$prev, cl_vec, geom, cfg)
  CLPL     <- plots_for_clusters(CLGRIDS, geom, cfg, add_hulls = cfg$add_u_hulls)
  
  artefacts$FIG_cl_FIELDS_UNITDISK   <- save_plot("FIG_cl_FIELDS_UNITDISK.png",   CLPL$p_std,  180, 150, cfg)
  artefacts$FIG_cl_FIELDS_BASE       <- save_plot("FIG_cl_FIELDS_BASE.png",       CLPL$p_base, 180, 150, cfg)
  artefacts$FIG_cl_FIELDS_UNITSQUARE <- save_plot("FIG_cl_FIELDS_UNITSQUARE.png", CLPL$p_sq,   180, 150, cfg)
}

# 6) Item biplots --------------------------------------------------------------
Rtab <- build_biplot_data(Z_A, varmap, Base_A, geom$U)
utils::write.csv(Rtab, file.path(cfg$out_dir, "items_vs_base_and_unitdisk_correlations.csv"), row.names = FALSE)
BIP  <- plot_biplots(Rtab, Base_A, geom$U)

artefacts$FIG_biplot_items_BASE     <- save_plot("FIG_biplot_items_BASE.png",     BIP$p_base, 140, 110, cfg)
artefacts$FIG_biplot_items_UNITDISK <- save_plot("FIG_biplot_items_UNITDISK.png", BIP$p_disk, 120, 120, cfg)

# 7) Per-dx metrics (+ curves PDF/CSVs) ---------------------------------------
run_dx_metrics(DxW_A, Base_A, XR = E %||% NULL, cfg = cfg, out_prefix = "")

# 8) Calibration panel ---------------------------------------------------------
p_cal <- calibration_curve_dx(DxW_A, Base_A, cfg, XR = E %||% NULL,
                              show_points = TRUE, band_B = 0L)  # set band_B=200 for a 95% band
if (inherits(p_cal, "ggplot")) {
  artefacts$FIG_calibration_per_dx <- save_plot("FIG_calibration_per_dx.png", p_cal, 180, 140, cfg)
}

# 9) KNN stability (k-band): mean line with p10–p90 ribbon ---------------------
KB <- knn_stability_plot_kband(
  Base_A,
  k_range = 10:20,
  sd_grid = c(0, 0.05, 0.10, 0.15, 0.20),
  R = 600L,
  seed = cfg$seed_pred
)

ST <- data.frame(
  sigma = KB$sd,
  mean  = KB$stats[, "mean"],
  p10   = KB$stats[, "p10"],
  p90   = KB$stats[, "p90"]
)

p_kn <- ggplot(ST, aes(x = sigma, y = mean)) +
  geom_ribbon(aes(ymin = p10, ymax = p90), alpha = 0.20) +
  geom_line(size = 0.9) +
  geom_point(size = 1.6) +
  scale_x_continuous(breaks = KB$sd, limits = range(KB$sd)) +
  scale_y_continuous(limits = c(0, 1)) +
  labs(
    x = expression(sigma ~ "(Base noise, SD)"),
    y = "Neighbour overlap (mean Jaccard)",
    title = "k-NN stability across k \u2208 [10, 20] (mean with p10–p90)"
  ) +
  theme_pub(12)

artefacts$FIG_knn_stability_vs_noise <- save_plot(
  "FIG_knn_stability_vs_noise.png", p_kn, 130, 95, cfg
)

# 10) Lift map (optional) ------------------------------------------------------
if (isTRUE(cfg$make_lift_maps)) {
  p_co <- lift_map_plot(DxW_A, order = "cluster", adjust_p = FALSE)
  if (inherits(p_co, "ggplot")) {
    artefacts$FIG_cooccurrence_lift <- save_plot("FIG_cooccurrence_lift_uppertri_diagprev.png", p_co, 190, 150, cfg)
  }
}

# 11) Figure 2 panels (needs Fbase from dx grids) ------------------------------
if (!is.null(DXGRIDS$Fbase) && nrow(DXGRIDS$Fbase)) {
  F2 <- figure2_plots(geom, DXGRIDS$Fbase, cfg$br_fixed)
  artefacts$FIG2A_unitsquare_scatter <- save_plot("FIG2A_unitsquare_scatter.png", F2$fig2A, 140, 120, cfg)
  artefacts$FIG2B_dx_contours_6      <- save_plot("FIG2B_dx_contours_6.png",     F2$fig2B, 175, 130, cfg)
}

# 12) Direction wheel ----------------------------------------------------------
p_dir <- direction_wheel_plot(geom)
if (inherits(p_dir, "ggplot")) {
  artefacts$FIG_uv_direction_density_HCL_smooth <- save_plot("FIG_uv_direction_density_HCL_smooth.png", p_dir, 140, 120, cfg)
}

# 13) Base-only performance table ---------------------------------------------
base_performance_table(DxW_A, Base_A, cfg, out_prefix = "TAB_")

# 14) Done — show output paths -------------------------------------------------
print(artefacts)