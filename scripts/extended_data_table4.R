
suppressPackageStartupMessages({ library(dplyr); library(readr); library(tibble) })

# ---------------- Config ----------------
B_REPS <- 1000L
SEED   <- 42L
K_CAP  <- 5L
OUT    <- "ExtendedData_Table4.csv"

# ------------- Load helpers -------------

# expect the same residual/features matrix we used for the plots
if (!exists("XR") && exists("E")) XR <- E
if (!exists("XR")) stop("XR/E matrix required for Fibre/Combined. Load the residual feature matrix used for the plots (e.g., XR or E).")

# ---- safer fitter to avoid glmnet crashes on tiny folds (shadow repo one) ----
fit_glm_or_glmnet <- function(y, X){
  df <- as.data.frame(X); df$.y <- as.integer(y > 0)
  
  # helper: make stratified fold ids with >=2 per class per fold when possible
  make_foldid <- function(yb, k){
    yb <- as.integer(yb > 0)
    n1 <- sum(yb == 1L); n0 <- sum(yb == 0L)
    # shrink k until each fold can hold at least 2 of each class (if class exists)
    k_eff <- max(2L, min(k, 5L))
    while (k_eff > 2L && ( (n1 > 0L && n1 < 2L * k_eff) || (n0 > 0L && n0 < 2L * k_eff) )) {
      k_eff <- k_eff - 1L
    }
    # assign folds separately within each class (round-robin)
    idx0 <- which(yb == 0L); idx1 <- which(yb == 1L)
    f0 <- if (length(idx0)) sample(rep_len(seq_len(k_eff), length(idx0))) else integer(0)
    f1 <- if (length(idx1)) sample(rep_len(seq_len(k_eff), length(idx1))) else integer(0)
    foldid <- integer(length(yb)); foldid[idx0] <- f0; foldid[idx1] <- f1
    list(foldid = foldid, k_eff = k_eff)
  }
  
  ok_glmnet <- requireNamespace("glmnet", quietly = TRUE) &&
    nrow(df) >= 20L && length(unique(df$.y)) == 2L &&
    min(table(df$.y)) >= 5L
  
  if (ok_glmnet) {
    x <- as.matrix(df[setdiff(names(df), ".y")])
    yb <- as.numeric(df$.y)
    
    # drop constant/near-constant columns to avoid degeneracy
    keep <- which(apply(x, 2, function(v) isTRUE(sd(v, na.rm = TRUE) > 0)))
    if (length(keep) >= 1L) x <- x[, keep, drop = FALSE]
    
    nfolds0 <- min(5L, max(2L, floor(min(table(df$.y))/2L)))
    sf <- make_foldid(yb, nfolds0)
    
    res <- tryCatch({
      cv <- glmnet::cv.glmnet(
        x, yb, alpha = 0, family = "binomial",
        standardize = TRUE,
        foldid = sf$foldid,
        nfolds = sf$k_eff
      )
      list(type = "glmnet", fit = cv, xnames = colnames(x))
    }, error = function(e) {
      # Fallback: plain GLM with tiny jitter to dodge separation
      jcols <- setdiff(names(df), ".y")
      for (nm in jcols) {
        v <- df[[nm]]
        if (is.numeric(v)) df[[nm]] <- v + rnorm(length(v), 0, 1e-8)
      }
      f <- stats::glm(.y ~ ., data = df, family = stats::binomial(), control = list(maxit = 50))
      list(type = "glm", fit = f)
    })
    return(res)
  }
  
  # GLM fallback (+ tiny jitter)
  jcols <- setdiff(names(df), ".y")
  for (nm in jcols) {
    v <- df[[nm]]
    if (is.numeric(v)) df[[nm]] <- v + rnorm(length(v), 0, 1e-8)
  }
  f <- stats::glm(.y ~ ., data = df, family = stats::binomial(), control = list(maxit = 50))
  list(type = "glm", fit = f)
}

# -------- shared bootstrap indices & helpers (paired across models) ----------
mk_boot_idx <- function(y, B, seed = SEED){
  set.seed(seed)
  y <- as.integer(y > 0); N <- length(y)
  i0 <- which(y == 0L); i1 <- which(y == 1L)
  if (!length(i0) || !length(i1)) return(NULL)
  idx <- matrix(NA_integer_, nrow = N, ncol = B)
  for (b in seq_len(B)) {
    idx[, b] <- c(sample(i0, length(i0), TRUE), sample(i1, length(i1), TRUE))
  }
  idx
}
boot_auc_ci_from_idx <- function(y, p, idx){
  y <- as.integer(y > 0)
  if (is.null(idx)) return(c(point = NA_real_, lo = NA_real_, hi = NA_real_))
  draws <- apply(idx, 2, function(ii) auc_point(y[ii], p[ii]))
  c(point = auc_point(y, p),
    lo    = unname(quantile(draws, 0.025, names = FALSE)),
    hi    = unname(quantile(draws, 0.975, names = FALSE)))
}
boot_delta_from_idx <- function(y, pA, pB, idx){
  y <- as.integer(y > 0)
  if (is.null(idx)) return(c(point = NA_real_, lo = NA_real_, hi = NA_real_, p = NA_real_, win = NA_real_))
  draws <- apply(idx, 2, function(ii) auc_point(y[ii], pA[ii]) - auc_point(y[ii], pB[ii]))
  pt  <- auc_point(y, pA) - auc_point(y, pB)
  lo  <- unname(quantile(draws, 0.025, names = FALSE))
  hi  <- unname(quantile(draws, 0.975, names = FALSE))
  if (all(abs(draws) < 1e-15)) {
    win <- 0.5
    p2  <- 1.0
  } else {
    win <- mean(draws > 0)
    p2  <- min(1.0, 2 * min(mean(draws >= 0), mean(draws <= 0)))
  }
  c(point = pt, lo = lo, hi = hi, p = p2, win = win)
}
fmt3 <- function(x) ifelse(is.finite(x), sprintf("%.3f", as.numeric(x)), NA_character_)

# ---------------- Expect DxW/Base in env; select same dx used for plots -------
stopifnot(exists("DxW"), exists("Base"))
stopifnot(nrow(DxW) == nrow(Base))
if (is.null(rownames(DxW)))  rownames(DxW)  <- seq_len(nrow(DxW))
if (is.null(rownames(Base))) rownames(Base) <- rownames(DxW)

cfg <- default_cfg()  # uses dx_min_pos = 10, dx_min_neg = 10 by default
sel <- select_diagnoses(DxW, cfg)
dx_keep <- sel$keep
diags <- dx_keep
if (!length(diags)) stop("No diagnoses pass the plot thresholds (≥10 pos & ≥10 neg).")

excluded <- setdiff(names(DxW), diags)
if (length(excluded)) cat("Excluding rare/one-class labels (not used in plots):\n  -", paste(excluded, collapse = "\n  - "), "\n")

rows <- vector("list", length(diags))
cat("Building Extended Data Table 4 for", length(diags), "diagnoses…\n")

# ---------------- Loop sequentially (no 'Dominant model') ---------------------
for (j in seq_along(diags)){
  dx <- diags[j]
  y  <- as.integer(DxW[[dx]] > 0)
  n1 <- sum(y == 1L); n0 <- sum(y == 0L); N <- length(y); prev <- n1 / N
  if (length(unique(y)) < 2L) { cat(sprintf("[%d/%d] %s -> skipped (one-class)\n", j, length(diags), dx)); next }
  
  K  <- max(2L, min(K_CAP, floor(n1/2L), floor(n0/2L)))
  cat(sprintf("[%d/%d] %s  K=%d  n+=%d  N=%d\n", j, length(diags), dx, K, n1, N))
  
  mods <- oof_prob_stacked(
    y = y,
    Base_A = as.data.frame(Base),
    XR     = as.data.frame(XR),
    K      = K,
    seed   = SEED
  )
  
  # probs (clip + impute so boot never sees NA/Inf)
  fixp <- function(p){ p[!is.finite(p)] <- mean(y); pmin(pmax(p, 1e-6), 1 - 1e-6) }
  pB <- fixp(as.numeric(mods$Base))
  pF <- fixp(as.numeric(mods$Fibre))
  pC <- fixp(as.numeric(mods$Both))   # Combined
  
  if (sd(pB) == 0) warning(sprintf("[%s] Base predictions are constant", dx))
  if (sd(pF) == 0) warning(sprintf("[%s] Fibre predictions are constant", dx))
  if (sd(pC) == 0) warning(sprintf("[%s] Combined predictions are constant", dx))
  
  # shared bootstrap indices per diagnosis
  IDX <- mk_boot_idx(y, B_REPS, seed = SEED)
  
  # AUCs + CIs (paired resamples)
  ciB <- boot_auc_ci_from_idx(y, pB, IDX)
  ciF <- boot_auc_ci_from_idx(y, pF, IDX)
  ciC <- boot_auc_ci_from_idx(y, pC, IDX)
  
  # ΔAUCs + p, win (paired resamples)
  dCB <- boot_delta_from_idx(y, pC, pB, IDX)  # Combined - Base
  dCF <- boot_delta_from_idx(y, pC, pF, IDX)  # Combined - Fibre
  dBF <- boot_delta_from_idx(y, pB, pF, IDX)  # Base - Fibre
  
  # points for formatting
  ptB <- as.numeric(ciB["point"]); loB <- as.numeric(ciB["lo"]); hiB <- as.numeric(ciB["hi"])
  ptF <- as.numeric(ciF["point"]); loF <- as.numeric(ciF["lo"]); hiF <- as.numeric(ciF["hi"])
  ptC <- as.numeric(ciC["point"]); loC <- as.numeric(ciC["lo"]); hiC <- as.numeric(ciC["hi"])
  
  rows[[j]] <- tibble(
    Diagnosis = dx,
    `n₊/N (%)`              = sprintf("%d/%d (%.1f%%)", n1, N, 100 * prev),
    
    `Base AUC [95% CI]`     = sprintf("%.3f [%.3f, %.3f]", ptB, loB, hiB),
    `Fibre AUC [95% CI]`    = sprintf("%.3f [%.3f, %.3f]", ptF, loF, hiF),
    `Combined AUC [95% CI]` = sprintf("%.3f [%.3f, %.3f]", ptC, loC, hiC),
    
    `ΔAUC (Combined−Base)`  = sprintf("%.3f [%.3f, %.3f]", dCB["point"], dCB["lo"], dCB["hi"]),
    `p (C vs B)`            = fmt3(dCB["p"]),   `q (C vs B)` = NA_character_, `Pr(C>B)` = fmt3(dCB["win"]),
    
    `ΔAUC (Combined−Fibre)` = sprintf("%.3f [%.3f, %.3f]", dCF["point"], dCF["lo"], dCF["hi"]),
    `p (C vs F)`            = fmt3(dCF["p"]),   `q (C vs F)` = NA_character_, `Pr(C>F)` = fmt3(dCF["win"]),
    
    `ΔAUC (Base−Fibre)`     = sprintf("%.3f [%.3f, %.3f]", dBF["point"], dBF["lo"], dBF["hi"]),
    `p (B vs F)`            = fmt3(dBF["p"]),   `q (B vs F)` = NA_character_,
    
    # raw p's for FDR later (will be dropped after BH)
    p_CvsB = as.numeric(dCB["p"]),
    p_CvsF = as.numeric(dCF["p"]),
    p_BvsF = as.numeric(dBF["p"])
  )
}

ED4 <- bind_rows(rows)
if (!nrow(ED4)) stop("No rows produced (check DxW/Base and dx_keep).")

ED4$`q (C vs B)` <- fmt3(p.adjust(ED4$p_CvsB, method = "BH"))
ED4$`q (C vs F)` <- fmt3(p.adjust(ED4$p_CvsF, method = "BH"))
ED4$`q (B vs F)` <- fmt3(p.adjust(ED4$p_BvsF, method = "BH"))

# Drop the raw p columns (base R; immune to masking)
ED4 <- ED4[, !(names(ED4) %in% c("p_CvsB", "p_CvsF", "p_BvsF")), drop = FALSE]

readr::write_csv(ED4, OUT)
cat("Extended Data Table 4 written to:", OUT, "\n")
print(utils::head(ED4, 6))

# Packages
library(gt); library(readr); library(dplyr); library(stringr);
library(pdftools)

# ---- 1) Load your CSV ----
df <- readr::read_csv("ExtendedData_Table4.csv", show_col_types = FALSE)

# ---- 2) Build the columns EXACTLY as you want them ----
# Create n_pos from the "n₊/N (%)" field (left of the slash)
df <- df %>%
  mutate(
    n_pos = as.integer(str_trim(str_extract(`n₊/N (%)`, "^[0-9]+")))
  )

# Normalise diagnosis names to match your requested labels (if needed)
df <- df %>%
  mutate(
    Diagnosis = recode(
      Diagnosis,
      "Bipolar I Disorder" = "Bipolar Disorder, type I"
      # add other mappings if any label spelling differs from your list
    )
  )

# Desired order
order_vec <- c(
  "No Diagnosis on Axis I",
  "ADHD",
  "Bipolar Disorder, type I",
  "Major Depressive Disorder",
  "Depressive Disorder NOS",
  "Schizoaffective Disorder",
  "Schizophrenia",
  "Alcohol Abuse",
  "Alcohol Dependence",
  "Amphetamine Dependence",
  "Cannabis Abuse",
  "Cannabis Dependence",
  "Cocaine Abuse",
  "Cocaine Dependence"
)

# Keep only those present (quietly) & order
df$Diagnosis <- factor(df$Diagnosis, levels = order_vec)
df <- df %>% filter(!is.na(Diagnosis)) %>% arrange(Diagnosis)

# Build the ED table with EXACT headers and order
df_ed <- df %>%
  transmute(
    Diagnosis,
    n_pos,
    `Base AUC [95% CI]`,
    `Fibre AUC [95% CI]`,
    `Combined AUC [95% CI]`,
    `ΔAUC (Combined−Base)`,
    `p (C vs B)`,
    `q (C vs B)`,
    `Pr(C>B)`,
    `ΔAUC (Combined−Fibre)`,
    `p (C vs F)`,
    `q (C vs F)`,
    `Pr(C>F)`,
    `ΔAUC (Base−Fibre)`,
    `p (B vs F)`,
    `q (B vs F)`
  )

# ---- 3) Format with gt (Arial/Helvetica 7 pt; horizontal rules only) ----
last_row <- nrow(df_ed)

tab <- gt(df_ed) |>
  cols_align(align = "left", columns = "Diagnosis") |>
  cols_align(align = "center", columns = -Diagnosis) |>
  tab_options(
    table.font.names = c("Helvetica","Arial","Liberation Sans"),
    table.font.size  = px(7),
    data_row.padding = px(2),
    heading.align    = "left"
  ) |>
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels(everything())
  ) |>
  # horizontal rules: top of header + bottom of table
  tab_style(
    style = cell_borders(sides = "top", weight = px(1)),
    locations = cells_column_labels(everything())
  ) |>
  tab_style(
    style = cell_borders(sides = "bottom", weight = px(1)),
    locations = cells_body(rows = last_row, columns = everything())
  ) |>
  opt_table_lines(extent = "none")

# 1) Save vector PDF
gtsave(tab, "ED_Table4.pdf")   # vector; no pixelation

# 2) Convert to high-DPI TIFF (or PNG/JPEG) using pdftools
# install.packages("pdftools") # if needed
pdftools::pdf_convert("ED_Table4.pdf",
                      format    = "tiff",
                      dpi       = 600,          # 600–900 dpi is fine
                      filenames = "ED_Table4.tif")