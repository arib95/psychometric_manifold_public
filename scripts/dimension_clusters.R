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
# Reproducibility notes
#   - Random seed set at the top.
#   - Prevalence guards treat NA as 0 only for keep/drop decisions; modelling uses
#     0/1 with NA coerced to 0 conservatively where needed.
#   - All thresholds are surfaced at the Parameters block.
# ==============================================================================


# ============================= 1) Package imports ==============================
suppressPackageStartupMessages({
  library(readr);   library(dplyr);  library(tidyr);   library(utils)
  library(cluster); library(RANN);   library(RSpectra); library(dbscan)
  library(igraph);  library(ggplot2);library(Matrix);  library(aricode)
  library(glmnet);  library(vegan);  library(princurve);library(mgcv)
  library(reticulate);               library(MASS);     library(R.utils)
  library(FNN);     library(expm);   library(clue);     library(FactoMineR)
  library(scales);  library(plotly); library(htmlwidgets); library(ggrepel)
  library(future); library(future.apply); library(progressr); library(dplyr)
  library(tibble); library(ragg)
})

# =============================== 2) Parameters =================================
# Data
DX_CSV_PATH        <- "wide_diagnoses.csv"  # used only if DX not in memory
INCLUDE_NODIAG     <- FALSE                  # if no NODIAG col, create from rows with no DX=1
DX_MIN_PREV        <- 0.00                  # keep DX with prevalence in [min,max]
DX_MAX_PREV        <- 0.99
MIN_CASES_TOTAL    <- 10                    # minimum total cases required to keep a DX
MIN_CASES_IN       <- 5
MIN_CASES_OUT      <- 5

# kNN graph parameters
K_KNN              <- 9                     # check cluster_sensitivity_grid and seed bootstrap
UNION_KNN          <- FALSE                 # TRUE: union-kNN; FALSE: mutual-kNN

# significance + stability
N_PERM             <- 200L                   # permutations for Q/S nulls & deg-null reps (where used)
N_BOOT             <- 200L                    # bootstrap re-cluster repetitions
MIN_CLUSTER_SIZE   <- 2L                    # on deduplicated profiles!! (pre-expansion)
MIN_CLUSTER_WEIGHT <- 8L                    # participants (sum of mult) required to keep a community

# enrichment + major Dx selection thresholds (pillar A)
ALPHA_FDR          <- 0.05
MIN_PREV_IN_CL     <- 0.00
MIN_OR             <- 2.0

# three-pillars thresholds (B, C)
ALPHA_LOCALIZE     <- 0.05                  # min(assort p, kNN-purity p) for pillar B
AUC_MIN            <- 0.70                  # one-vs-rest predictability for pillar C
PREV_MIN           <- 0.03                  # ≥3% overall or at least NCASE_MIN cases
NCASE_MIN          <- 5L
NIN_MIN            <- 5L                     # ≥5 cases inside a cluster for enrichment test
NOUT_MIN           <- 5L                     # ≥5 cases outside cluster for enrichment test
DENY_NOS           <- FALSE                 # drop NOS diagnoses by default (was debug)

# plotting
PLOT_WIDTH         <- 7
PLOT_HEIGHT        <- 6
PLOT_DPI           <- 150

# k-sweep grid
K_GRID             <- c(5,6,7,8,9,10,11,12,13,14,15)
VARIANT            <- c("union","mutual")

# Reproducibility
SEED_GLOBAL        <- 42L
set.seed(SEED_GLOBAL)
RNGkind("L'Ecuyer-CMRG");
Sys.setenv(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
if (requireNamespace("RhpcBLASctl", quietly=TRUE)) RhpcBLASctl::blas_set_num_threads(1)

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

cfg <- default_cfg()
cfg$palette <- list(
  engine    = "scico",
  name      = "vik",      # sequential fallback
  name_div  = "vik",
  direction = 1
)
# ds000030 vik / ds005237 cork

# --- Community detection toggle ---
COMMUNITY_ALGO <- "leiden"  # "leiden" or "louvain"
# --- Parameters (add near the top) ---
LEIDEN_OBJECTIVE <- "modularity"   # or "CPM"
LEIDEN_GAMMA     <- 0.75            # <1 → fewer, larger clusters; >1 → more, smaller
LEIDEN_N_ITERS   <- -1             # default

# --- testing / plotting options ---
FISHER_ALT      <- "greater"  # one-sided tests for over-representation
PLOT_MAJOR_SET  <- "union"        # "A" = enrichment only; "union" = A ∪ B ∪ C
SHOW_NS_LABELS  <- TRUE      # hide "ns" on tiles/tables

# ============================ 3) Small utilities ===============================
# Safe filename: replace non [A-Za-z0-9_] with underscores
safe_file <- function(s) gsub("[^A-Za-z0-9_]+","_", s)

# Vectorised n choose 2
comb2 <- function(n) ifelse(n < 2, 0, n*(n-1)/2)

# Coalesce for scalars (used in defaults)
`%||%` <- function(a, b) if (!is.null(a)) a else b

cluster_community <- function(g){
  if (tolower(COMMUNITY_ALGO) == "leiden") {
    igraph::cluster_leiden(
      g,
      weights = igraph::E(g)$weight,
      objective_function  = LEIDEN_OBJECTIVE,
      resolution_parameter = LEIDEN_GAMMA,
      n_iterations = LEIDEN_N_ITERS
    )
  } else {
    igraph::cluster_louvain(g, weights = igraph::E(g)$weight)
  }
}
COMM_LABEL <- if (tolower(COMMUNITY_ALGO) == "leiden") "Leiden" else "Louvain"

library(colorspace); library(scico)

# Hues from ends of a scico sequential (works with colorspace ≥2.x)
end_hues <- function(pal) {
  ends <- scico::scico(2, palette = pal)
  obj  <- try(methods::as(colorspace::hex2RGB(ends), "polarLUV"), silent = TRUE)
  if (inherits(obj, "try-error"))  # fallback for very old colorspace
    obj <- methods::as(colorspace::hex2RGB(ends), "HCL")
  as.numeric(colorspace::coords(obj)[, "H"])
}

lapaz_div_hcl  <- function(n = 255)  diverging_hcl(n, h = end_hues("lapaz"),
                                                   c = 70, l = c(30, 96), power = 1.1)
lipari_div_hcl <- function(n = 255)  diverging_hcl(n, h = end_hues("lipari"),
                                                   c = 70, l = c(30, 96), power = 1.1)

get_shared_palette_cfg <- function(){
  # default (safe) palette config
  pal <- list(engine = "scico", name = cfg$palette$name, name_div = cfg$palette$name_div, direction = 1L)
  if (!is.null(PALETTE_SOURCE_FILE) && file.exists(PALETTE_SOURCE_FILE)) {
    e <- new.env(parent = emptyenv())
    ok <- try(suppressWarnings(source(PALETTE_SOURCE_FILE, local = e, chdir = FALSE)), silent = TRUE)
    if (!inherits(ok, "try-error") && exists("default_cfg", envir = e)) {
      cfg <- try(e$default_cfg(), silent = TRUE)
      if (is.list(cfg) && !is.null(cfg$palette)) {
        pp <- cfg$palette
        pal$engine    <- tolower(pp$engine %||% pal$engine)
        pal$name      <- pp$name     %||% pal$name
        pal$name_div  <- pp$name_div %||% pal$name_div  # allow separate diverging choice
        pal$direction <- pp$direction %||% pal$direction
      }
    }
  }
  pal
}

# --- Diverging versions of sequential scico maps (lapaz/lipari) ---
scico_divergent <- function(base, n = 255, neutral = "#FAFAFA") {
  # Use native diverging palettes as-is
  diverging_names <- c("vik","broc","cork","lisbon","tofino","bam","berlin","roma","vikO","brocO","corkO")
  if (base %in% diverging_names) return(scico::scico(n, palette = base))
  # Fallback: build diverging from sequential base
  seq <- scico::scico(n, palette = base)
  L <- seq[1:floor(n/2)]
  R <- rev(seq)[1:floor(n/2)]
  c(L, neutral, R)
}

# Sequential (0–1) scale, e.g., probability fields if you ever need it here
scale_shared_seq <- function(name = NULL, limits = c(0,1)){
  pal <- get_shared_palette_cfg()
  eng <- pal$engine; dir <- pal$direction; nm <- pal$name
  if (eng == "scico" && requireNamespace("scico", quietly = TRUE)) {
    scico::scale_fill_scico(palette = nm, direction = dir, limits = limits,
                            na.value = NA, name = name)
  } else if (requireNamespace("ggplot2", quietly = TRUE)) {
    ggplot2::scale_fill_gradient(low = "#f7fbff", high = "#08306b",
                                 limits = limits, na.value = NA, name = name)
  } else stop("ggplot2 not available")
}

# Diverging scale for log2(lift) (symmetrically bounded)
scale_shared_div <- function(name = "log2(lift)", limits = c(-2, 2)) {
  pal <- get_shared_palette_cfg()
  if (pal$engine == "scico" && requireNamespace("scico", quietly = TRUE)) {
    base <- pal$name_div %||% pal$name
    scico::scale_fill_scico(
      palette = base,
      direction = pal$direction %||% 1,
      limits = limits,
      oob = scales::squish,
      name = name
    )
  } else {
    ggplot2::scale_fill_gradient2(midpoint = 0, limits = limits,
                                  oob = scales::squish, name = name)
  }
}

# Basic kNN graph diagnostics
report_knn_graph <- function(g){
  comps <- igraph::components(g)
  cat(sprintf("[kNN graph] V=%d, E=%d, components=%d, largest=%d (%.1f%%), isolates=%d (%.1f%%)\n",
              igraph::gorder(g), igraph::ecount(g),
              comps$no, max(comps$csize), 100*max(comps$csize)/igraph::gorder(g),
              sum(comps$csize==1), 100*sum(comps$csize==1)/igraph::gorder(g)))
}

# Map a DX code -> vertex indices in the UNIQUE-profile graph (requires DXu_id in scope)
vertex_set <- function(dx_code) {
  stopifnot(dx_code %in% names(DXu_id))
  which(DXu_id[[dx_code]] == 1L)
}

# Participant IDs for a logical mask over DX (participant table)
pt_ids <- function(mask) as.character(DX$participant_id)[ which(mask) ]

# Wilson interval for a binomial proportion
wilson_ci <- function(k, n, z = 1.96){
  k <- as.numeric(k); n <- as.numeric(n)
  p <- ifelse(n > 0, k / n, NA_real_)
  denom   <- ifelse(n > 0, 1 + z^2 / n, NA_real_)
  centre  <- ifelse(n > 0, (p + z^2 / (2 * n)) / denom, NA_real_)
  halfwid <- ifelse(n > 0, z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / denom, NA_real_)
  lo <- pmax(0, centre - halfwid)
  hi <- pmin(1, centre + halfwid)
  cbind(lo = lo, hi = hi)
}

# ======================= 4) Alignment + binarisation helpers ===================
# Align anything (data frame / matrix / named vector) to a set of IDs.
# - If a data frame contains `participant_id`, it is promoted to rownames.
# - Errors loud if any required IDs are missing.
align_to_ids <- function(obj, ids) {
  if (is.null(obj)) return(obj)
  if (is.data.frame(obj) && "participant_id" %in% names(obj)) {
    rn <- as.character(obj$participant_id)
    stopifnot(!anyDuplicated(rn), all(nzchar(rn)))
    rownames(obj) <- rn
    obj <- obj[, setdiff(names(obj), "participant_id"), drop = FALSE]
  }
  if (!is.null(rownames(obj))) {
    rn <- rownames(obj)
    miss <- setdiff(ids, rn)
    if (length(miss)) stop(sprintf("[align] %d ids missing; e.g., %s", length(miss), paste(head(miss, 8), collapse=", ")))
    return(obj[ids, , drop = FALSE])
  }
  if (!is.null(names(obj))) {
    nm <- names(obj)
    miss <- setdiff(ids, nm)
    if (length(miss)) stop(sprintf("[align] %d ids missing (named vector); e.g., %s", length(miss), paste(head(miss, 8), collapse=", ")))
    return(obj[ids])
  }
  stop("[align] Object has neither rownames nor names.")
}

# Guarded binariser for 0/1 or [0,1] scores (keeps 0/1; thresholds probabilities at cutoff)
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


# ============================ 5) Distances + ID ================================
# Asymmetric-binary Gower distance on 0/1 columns; returns a `dist` or NULL if <2 varying columns.
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

# TwoNN intrinsic dimension (tie-safe, trimmed on log(r2/r1))
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

# Levina–Bickel MLE of intrinsic dimension (dense matrix input)
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

# Select central band by k-th neighbour distance (robust core)
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

# =========================== 6) Data loading + dedup ===========================
# Load DX unless already in memory; enforce prevalence/count guards; ensure 0/1/NA.
load_dx_wide <- function(path,
                         include_nodiag = TRUE,
                         min_prev = 0.01,
                         max_prev = 0.99,
                         delim_override = NULL) {
  # 0) If DX already in memory, use it
  if (exists("DX") && is.data.frame(get("DX", inherits = TRUE))) {
    DX0 <- get("DX", inherits = TRUE)
  } else {
    # 1) Read with optional explicit delimiter, else try common formats, else sniff
    read_with_delim <- function(p, d) readr::read_delim(p, delim = d, col_types = readr::cols())
    
    suppressWarnings({
      DX0 <- NULL
      
      # 1a) Explicit override wins
      if (!is.null(delim_override)) {
        DX0 <- try(read_with_delim(path, delim_override), silent = TRUE)
        if (inherits(DX0, "try-error") || ncol(DX0) <= 1) DX0 <- NULL
      }
      
      # 1b) Fallbacks if still NULL
      if (is.null(DX0)) {
        DX0 <- try(readr::read_tsv(path, col_types = readr::cols()), silent = TRUE)
        if (inherits(DX0, "try-error") || ncol(DX0) <= 1) {
          DX0 <- try(readr::read_csv(path, col_types = readr::cols()), silent = TRUE)
        }
        if (inherits(DX0, "try-error") || ncol(DX0) <= 1) {
          DX0 <- try(readr::read_csv2(path, col_types = readr::cols()), silent = TRUE) # semicolon
        }
      }
      
      # 1c) Heuristic sniff if still failing
      if (is.null(DX0) || inherits(DX0, "try-error") || ncol(DX0) <= 1) {
        l1 <- try(readLines(path, n = 1L), silent = TRUE)
        if (!inherits(l1, "try-error") && length(l1)) {
          cand <- c("\t", ";", ",", "|")
          cnt  <- vapply(cand, function(s){
            m <- gregexpr(s, l1, fixed = TRUE)[[1]]
            if (identical(m, -1L)) 0L else length(m)
          }, integer(1))
          sep <- if (max(cnt) > 0L) cand[ which.max(cnt) ] else ","
          DX0 <- read_with_delim(path, sep)
        } else {
          stop("Could not read file: ", path)
        }
      }
    })
  }
  
  stopifnot(is.data.frame(DX0))
  
  # 2) Clean column names (trim, drop BOM)
  nm_raw <- names(DX0)
  nm_clean <- trimws(gsub("\ufeff", "", nm_raw))
  names(DX0) <- nm_clean
  
  # 3) Find/standardise participant_id
  id_aliases <- c("participant_id","participantid","participant","subject_id","subjectid",
                  "subject","id","pid","IID","FID","eid")
  hit <- nm_clean[ tolower(nm_clean) %in% tolower(id_aliases) ]
  if (length(hit) == 0L) {
    c1 <- DX0[[1]]
    uniq_enough <- !anyDuplicated(c1) && all(nchar(as.character(c1)) > 0)
    if (uniq_enough) {
      names(DX0)[1] <- "participant_id"
    } else {
      stop('No "participant_id" column found and could not infer an ID column. ',
           'Inspect the header with readLines("', path, '", n = 2).')
    }
  } else {
    id_col <- if ("participant_id" %in% hit) "participant_id" else hit[1]
    names(DX0)[names(DX0) == id_col] <- "participant_id"
  }
  
  # 4) Basic ID checks
  if (anyDuplicated(DX0$participant_id)) {
    dups <- DX0$participant_id[duplicated(DX0$participant_id)]
    stop(sprintf("Duplicate participant_id values found (e.g., %s).",
                 paste(head(dups, 3), collapse = ", ")))
  }
  if (!all(nzchar(as.character(DX0$participant_id)))) stop("Empty participant_id values found.")
  
  # 5) Add ANY_DX / NODIAG, force 0/1/NA in DX columns
  dx_cols0 <- setdiff(names(DX0), "participant_id")
  if (!length(dx_cols0)) stop("No diagnosis columns found after ID.")
  
  for (nm in dx_cols0) {
    v <- suppressWarnings(as.numeric(DX0[[nm]]))
    if (!all(v %in% c(0, 1, NA))) {
      rng_min <- suppressWarnings(min(v, na.rm = TRUE))
      rng_max <- suppressWarnings(max(v, na.rm = TRUE))
      stop(sprintf("Column %s is not 0/1 (found range: [%s, %s]).", nm, rng_min, rng_max))
    }
    DX0[[nm]] <- as.integer(v)
  }
  
  any_dx <- as.integer(rowSums(sapply(dx_cols0, function(nm) as.integer(DX0[[nm]] == 1L)),
                               na.rm = TRUE) > 0)
  if (!"ANY_DX" %in% names(DX0)) DX0$ANY_DX <- any_dx
  if (include_nodiag && !"NODIAG" %in% names(DX0)) DX0$NODIAG <- 1L - any_dx
  
  cols <- setdiff(names(DX0), "participant_id")
  
  # 6) Prevalence/count filter (protect ANY_DX/NODIAG)
  n <- nrow(DX0)
  pos_counts <- vapply(cols, function(nm) sum(DX0[[nm]] == 1L, na.rm = TRUE), integer(1))
  prev_all   <- pos_counts / n
  protected  <- intersect(c("NODIAG", "ANY_DX"), cols)
  min_cases  <- as.integer(get0("MIN_CASES_TOTAL", ifnotfound = 0L, inherits = TRUE))
  
  keep <- unique(c(
    names(prev_all)[ prev_all >= min_prev & prev_all <= max_prev & pos_counts >= min_cases ],
    protected
  ))
  if (!length(keep)) stop("No diagnosis columns pass prevalence/count filters.")
  
  DX1 <- DX0[, c("participant_id", keep), drop = FALSE]
  rownames(DX1) <- make.unique(as.character(DX1$participant_id))
  DX1
}

# Deduplicate identical diagnosis signatures; add `mult`; keep first occurrence.
# Returns list(DXu, sig, mult)
dedup_dx <- function(DX){
  cols <- setdiff(names(DX), "participant_id")
  X <- as.data.frame(DX[, cols, drop = FALSE])
  # For dedup only, treat NA as 0 to avoid spurious profiles from missingness
  for (nm in cols) { v <- X[[nm]]; v[is.na(v)] <- 0L; X[[nm]] <- as.integer(v) }
  sig <- apply(X, 1, paste0, collapse = "")
  tab <- as.data.frame(table(sig), stringsAsFactors = FALSE)
  names(tab) <- c("sig", "Freq")
  first_idx <- tapply(seq_len(nrow(DX)), sig, `[`, 1)   # named by sig
  DXu <- DX[unname(first_idx), , drop = FALSE]
  DXu$mult <- as.integer(tab$Freq[ match(names(first_idx), tab$sig) ])
  list(DXu = DXu, sig = sig, mult = DXu$mult)
}

# Participant-level counts/prevalence summary (for guards and pillar B)
build_counts <- function(DX){
  cols <- setdiff(names(DX), "participant_id")
  n1   <- vapply(cols, function(nm) sum(DX[[nm]] == 1L, na.rm = TRUE), integer(1))
  n0   <- nrow(DX) - n1
  prev <- n1 / (n1 + n0)
  data.frame(dx = cols, n1 = as.integer(n1), n0 = as.integer(n0),
             prev = as.numeric(prev), stringsAsFactors = FALSE)
}

# ============================= 7) kNN graph builder ============================
# Symmetric kNN graph (union or mutual) with Gaussian weights.
# - local_scale: per-node σ_i from k-th neighbour distance, using exp(-d^2/2σ_iσ_j)
# - mult: multiplicity upweights edges by sqrt(mult_i * mult_j)
# - add_mst: placeholder if you want to enforce connectivity later
knn_graph_from_dist <- function(D, k = 12, union = TRUE, mult = NULL,
                                local_scale = TRUE, add_mst = FALSE){
  M <- as.matrix(D); n <- nrow(M); diag(M) <- Inf
  kth <- function(r, k){
    rf <- r[is.finite(r)]
    if (!length(rf)) return(NA_real_)
    k_eff <- min(k, length(rf))
    sort(rf, partial = k_eff)[k_eff]
  }
  rk <- apply(M, 1, kth, k = k)
  if (local_scale) {
    sigma_i <- rk; sigma_i[!is.finite(sigma_i) | sigma_i <= 0] <- median(M[is.finite(M)])
  } else {
    sigma <- stats::median(rk[is.finite(rk)], na.rm = TRUE)
    if (!is.finite(sigma) || sigma <= 0) sigma <- stats::median(M[is.finite(M)], na.rm = TRUE)
    if (!is.finite(sigma) || sigma <= 0) sigma <- 1
  }
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
    if (!is.null(mult)) {
      m_i  <- if (is.finite(mult[i]) && mult[i] > 0) mult[i] else 1
      m_js <- ifelse(is.finite(mult[partners]) & mult[partners] > 0, mult[partners], 1)
      w <- w * sqrt(pmax(1, m_i) * pmax(1, m_js))
    }
    edges[[length(edges) + 1L]] <- data.frame(from = i, to = partners, weight = pmax(1e-12, w))
  }
  if (!length(edges)) stop("Empty kNN edge set.")
  Edf <- do.call(rbind, edges)
  g <- igraph::graph_from_data_frame(Edf, directed = FALSE,
                                     vertices = data.frame(name = seq_len(n)))
  g <- igraph::simplify(g, remove.multiple = TRUE, remove.loops = TRUE,
                        edge.attr.comb = list(weight = "sum"))
  g
}


# ========================== 8) Clustering + statistics =========================
# Degree-sequence modularity null via degree-preserving rewires.
# - On each draw: rewire topology (keeping degree), then shuffle weights.
modularity_degseq_null <- function(
    g, reps = 500L, niter_mult = 30L, keep_connected = TRUE, seed = NULL
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
    cl0 <- cluster_community(g0)
    Q[i] <- igraph::modularity(g0, igraph::membership(cl0), weights = igraph::E(g0)$weight)
  }
  Q
}

# Adjusted Rand Index (core) + alignment wrapper
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

# Louvain with robustness, nulls and guards; returns membership + stats.
louvain_with_stats <- function(
    g, D, n_perm = 200, min_size = 10, seed = NULL,
    deg_reps = 200L, deg_keep_connected = TRUE,
    null_scope = c("kept", "full"),
    v_weight = NULL,                # optional vertex weights (multiplicity)
    min_weight = NULL               # optional min participant-weight per community
){
  null_scope <- match.arg(null_scope)
  if (!is.null(seed)) set.seed(seed)
  
  cl_full <- cluster_community(g)
  memb    <- as.integer(igraph::membership(cl_full))
  tab_size <- table(memb)
  
  if (is.null(v_weight)) v_weight <- rep(1, igraph::gorder(g))
  w_by_comm <- tapply(v_weight, memb, sum)
  
  keep_flag <- (tab_size >= min_size)
  if (!is.null(min_weight)) keep_flag <- keep_flag & (w_by_comm[names(tab_size)] >= min_weight)
  # set as AND
  keep_c   <- as.integer(names(tab_size)[keep_flag])
  keep_idx <- which(memb %in% keep_c)
  
  out_default <- list(
    membership = rep(0L, igraph::gorder(g)),
    Q = NA_real_, Q_p_upper = NA_real_, Q_p_two = NA_real_,
    S = NA_real_, S_p = NA_real_,
    z_w = NA_real_, z_deg = NA_real_, p_deg = NA_real_,
    Q_null = rep(NA_real_, n_perm), S_null = rep(NA_real_, n_perm),
    Q_p = NA_real_, z = NA_real_
  )
  if (length(keep_idx) < 2L) return(out_default)
  
  g2 <- igraph::induced_subgraph(g, vids = keep_idx)
  Dm  <- as.matrix(D)
  D2  <- stats::as.dist(Dm[keep_idx, keep_idx, drop = FALSE])
  memb2        <- as.integer(factor(memb[keep_idx]))
  memb_full    <- integer(igraph::gorder(g)); memb_full[keep_idx] <- memb2
  K <- length(unique(memb2))
  
  if (igraph::ecount(g2) == 0L || K < 1L) {
    out_default$membership <- memb_full
    S_obs <- try({ ss <- cluster::silhouette(rep(1L, sum(keep_idx)), D2); mean(ss[, "sil_width"]) }, silent = TRUE)
    out_default$S <- if (inherits(S_obs, "try-error")) NA_real_ else S_obs
    return(out_default)
  }
  
  Q_obs <- igraph::modularity(g2, memb2, weights = igraph::E(g2)$weight)
  S_obs <- try({ ss <- cluster::silhouette(memb2, D2); mean(ss[, "sil_width"]) }, silent = TRUE)
  S_obs <- if (inherits(S_obs, "try-error")) NA_real_ else S_obs
  
  # ---- Weight-shuffle nulls on kept subgraph ----
  Q_null <- rep(NA_real_, n_perm)
  S_null <- rep(NA_real_, n_perm)
  z_w    <- NA_real_; Q_p_upper <- NA_real_; Q_p_two <- NA_real_; S_p <- NA_real_
  if (n_perm > 0L) {
    W2 <- igraph::E(g2)$weight
    on.exit({ igraph::E(g2)$weight <- W2 }, add = TRUE)
    for (b in seq_len(n_perm)) {
      igraph::E(g2)$weight <- sample(W2, length(W2), replace = FALSE)
      clb <- cluster_community(g2)
      mb2 <- as.integer(igraph::membership(clb))
      qv  <- try(igraph::modularity(g2, mb2, weights = igraph::E(g2)$weight), silent = TRUE)
      Q_null[b] <- if (inherits(qv, "try-error") || !is.finite(qv)) NA_real_ else qv
      sv  <- try({ ssb <- cluster::silhouette(mb2, D2); mean(ssb[, "sil_width"]) }, silent = TRUE)
      S_null[b] <- if (inherits(sv, "try-error")) NA_real_ else sv
    }
    igraph::E(g2)$weight <- W2
    B_Q  <- sum(is.finite(Q_null)); B_S <- sum(is.finite(S_null))
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
  
  # ---- Degree-preserving nulls (scope chosen) ----
  g_deg <- if (null_scope == "kept") g2 else g
  Q_deg <- modularity_degseq_null(g_deg, reps = deg_reps, niter_mult = 30L, keep_connected = deg_keep_connected)
  p_deg <- NA_real_; z_deg <- NA_real_
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
    Q_p = Q_p_upper, z = z_w
  )
}

# Expand deduplicated membership back to all participants (by signature)
expand_membership <- function(memb_u, sig_u, sig_all, ids_all){
  lu <- setNames(memb_u, sig_u)
  ma <- lu[as.character(sig_all)]
  data.frame(participant_id=ids_all, cluster=as.integer(ma))
}
# --- 8A) Absorb leftovers helper ------
absorb_leftovers_defensible <- function(g, D, memb, DXu_id, mult,
                                        tau = 0.70, k_min = 3, shared_min = 1,
                                        B = 200, alpha = 0.05, seed = 1) {
  stopifnot(igraph::is.igraph(g))
  set.seed(seed)
  
  n <- igraph::vcount(g)
  kept <- sort(setdiff(unique(memb), 0L))
  if (!length(kept)) return(list(membership = memb, absorbed = integer(0), q = numeric(0)))
  
  # distances + per-cluster intra distance thresholds
  M <- as.matrix(D); diag(M) <- NA_real_
  q75 <- sapply(kept, function(c){
    idx <- which(memb == c)
    if (length(idx) < 2) Inf else stats::quantile(as.dist(M[idx, idx]), 0.75, na.rm = TRUE)
  }); names(q75) <- kept
  
  # adjacency with weights
  nbrs <- igraph::adjacent_vertices(g, igraph::V(g), mode = "all")
  get_w <- function(i) {
    inc <- igraph::incident(g, i, mode = "all")
    if (!length(inc)) return(list(j = integer(0), w = numeric(0)))
    ed  <- igraph::ends(g, inc, names = FALSE)
    js  <- ifelse(ed[,1] == i, ed[,2], ed[,1])
    list(j = js, w = igraph::E(g)$weight[inc])
  }
  
  # binary matrix of diagnoses (NA -> 0)
  cols <- setdiff(names(DXu_id), c("participant_id", "mult"))
  X <- as.matrix(data.frame(lapply(DXu_id[, cols, drop = FALSE], function(v){ v <- as.integer(v == 1L); v[is.na(v)] <- 0L; v })))
  w_mult <- as.numeric(mult); w_mult[!is.finite(w_mult) | w_mult <= 0] <- 1
  
  cand <- which(memb == 0L)
  if (!length(cand)) return(list(membership = memb, absorbed = integer(0), q = numeric(0)))
  
  assign_to <- rep(0L, length(cand))
  names(assign_to) <- as.character(cand)
  pvals <- rep(NA_real_, length(cand))
  
  kept_idx <- which(memb > 0L)
  kept_labels <- memb[kept_idx]
  
  for (ii in seq_along(cand)) {
    i <- cand[ii]
    nb <- get_w(i)
    if (!length(nb$j)) next
    
    # restrict to neighbours in kept clusters
    lab_nb <- memb[nb$j]
    ok <- which(lab_nb %in% kept)
    if (length(ok) < k_min) next
    
    j_ok <- nb$j[ok]
    w_ok <- nb$w[ok] * w_mult[j_ok]
    
    # support per cluster (affinity)
    supp <- tapply(w_ok, lab_nb[ok], sum)
    supp[setdiff(as.character(kept), names(supp))] <- 0
    supp <- supp[as.character(kept)]
    tot  <- sum(supp)
    if (tot <= 0) next
    
    c_star <- kept[ which.max(supp) ]
    purity <- as.numeric(max(supp) / tot)
    
    if (purity < tau) next
    
    # distance guard
    js_c <- j_ok[ lab_nb[ok] == c_star ]
    med_d <- stats::median(M[i, js_c], na.rm = TRUE)
    if (!is.finite(med_d) || med_d > q75[as.character(c_star)]) next
    
    # shared-dx guard (at least one 1 in common with neighbours of c*)
    xi <- X[i, ]
    xN <- colSums(X[js_c, , drop = FALSE]) > 0
    if (sum(xi & xN) < shared_min) next
    
    # permutation p-value: shuffle kept labels, compute max support each time
    s_obs <- max(supp)
    s_null <- numeric(B)
    for (b in seq_len(B)) {
      sh <- sample(kept_labels, length(kept_labels), replace = FALSE)
      memb_sh <- memb; memb_sh[kept_idx] <- sh
      lab_nb_sh <- memb_sh[nb$j]
      okb <- which(lab_nb_sh %in% kept)
      if (!length(okb)) { s_null[b] <- 0; next }
      j_okb <- nb$j[okb]; w_okb <- nb$w[okb] * w_mult[j_okb]
      s_b <- tapply(w_okb, lab_nb_sh[okb], sum)
      s_null[b] <- if (length(s_b)) max(s_b) else 0
    }
    pvals[ii] <- (1 + sum(s_null >= s_obs, na.rm = TRUE)) / (1 + sum(is.finite(s_null)))
    assign_to[ii] <- c_star
  }
  
  # BH across candidates we actually tested
  tested <- which(is.finite(pvals) & assign_to > 0)
  if (!length(tested)) return(list(membership = memb, absorbed = integer(0), q = numeric(0)))
  q <- rep(NA_real_, length(cand)); q[tested] <- p.adjust(pvals[tested], method = "BH")
  
  accept <- tested[ which(q[tested] <= alpha) ]
  m2 <- memb
  if (length(accept)) m2[ as.integer(names(assign_to)[accept]) ] <- assign_to[accept]
  
  list(membership = m2,
       absorbed = as.integer(names(assign_to)[accept]),
       q = stats::setNames(q[accept], names(assign_to)[accept]))
}

# =============================== 9) Pillar A ===================================
# Single-cell audit (diagnosis × cluster) — returns a,b,c,d, lifts and exact p
audit_overrep_cell <- function(dx_code, cl_id,
                               min_in_cases = 5, min_out_cases = 5, min_total_cases = 10) {
  stopifnot(all(c("participant_id","cluster") %in% names(clusters_all)))
  dfx <- merge(DX, clusters_all, by = "participant_id", all.x = TRUE)
  dfx$cluster[is.na(dfx$cluster)] <- 0L
  
  in_cl  <- dfx$cluster == cl_id
  out_cl <- !in_cl
  n_in   <- sum(in_cl); n_out <- sum(out_cl)
  
  a <- sum(dfx[[dx_code]][in_cl]  == 1L, na.rm = TRUE)
  c <- sum(dfx[[dx_code]][in_cl]  == 0L, na.rm = TRUE)
  b <- sum(dfx[[dx_code]][out_cl] == 1L, na.rm = TRUE)
  d <- sum(dfx[[dx_code]][out_cl] == 0L, na.rm = TRUE)
  
  prev_in  <- a / max(1, a + c)
  prev_all <- (a + b) / max(1, a + b + c + d)
  lift     <- if (prev_all > 0) prev_in / prev_all else NA_real_
  
  p <- NA_real_
  if ((a + b) >= min_total_cases && a >= min_in_cases && b >= min_out_cases &&
      all(c(n_in, n_out) > 0)) {
    p <- fisher.test(matrix(c(a,b,c,d), 2, byrow = TRUE))$p.value
  } else if (n_in > 0 && is.finite(prev_all)) {
    alt <- if (prev_in >= prev_all) "greater" else "less"
    p <- binom.test(a, n_in, p = prev_all, alternative = alt)$p.value
  }
  data.frame(dx = dx_code, cluster = cl_id, a,b,c,d, n_in, n_out,
             prev_in, prev_all, lift, p, stringsAsFactors = FALSE)
}

# Enrichment table per cluster with FDR guards; also returns majors (pillar A).
diagnosis_enrichment <- function(
    DX, clusters,
    alpha_fdr       = 0.05,
    min_prev_in     = 0.10,
    min_or          = 2.0,
    exclude         = c("ANY_DX"),
    # cell-level guards (per-diagnosis)
    min_in_cases    = MIN_CASES_IN,
    min_total_cases = MIN_CASES_TOTAL,
    min_out_cases   = MIN_CASES_OUT,
    # cluster-level guards (participant weight, i.e., expanded rows)
    kept_clusters        = get0("KEPT_CLUSTERS", ifnotfound = NULL, inherits = TRUE),
    min_cluster_weight   = get0("MIN_CLUSTER_WEIGHT", ifnotfound = 0L, inherits = TRUE)
){
  stopifnot(all(c("participant_id","cluster") %in% names(clusters)))
  
  cols <- setdiff(names(DX), c("participant_id", exclude))
  
  # merge to participant level; treat unlabeled as 0 (dropped)
  df <- merge(DX, clusters, by = "participant_id", all.x = TRUE)
  df$cluster[is.na(df$cluster)] <- 0L
  
  # only evaluate the clusters you kept (if provided), and never cl==0
  cl_list <- setdiff(sort(unique(df$cluster)), 0L)
  if (!is.null(kept_clusters)) cl_list <- intersect(cl_list, kept_clusters)
  if (!length(cl_list)) return(list(enrichment = NULL, majors = character(0)))
  
  out <- vector("list", length(cl_list))
  t <- 0L
  
  for (cl in cl_list) {
    in_cl <- df$cluster == cl
    n_in  <- sum(in_cl)                # PARTICIPANT WEIGHT (expanded)
    if (n_in < min_cluster_weight) next
    
    rows <- lapply(cols, function(v){
      a <- sum(df[[v]][in_cl]  == 1L, na.rm = TRUE)
      b <- sum(df[[v]][!in_cl] == 1L, na.rm = TRUE)
      c <- sum(df[[v]][in_cl]  == 0L, na.rm = TRUE)
      d <- sum(df[[v]][!in_cl] == 0L, na.rm = TRUE)
      
      # hard eligibility for the cell
      if ((a + b) < min_total_cases || a < min_in_cases || b < min_out_cases) return(NULL)
      
      prev_in  <- a / max(1, a + c)
      prev_all <- (a + b) / max(1, a + b + c + d)
      
      pval <- NA_real_
      OR   <- if (b == 0 || c == 0) Inf else (a * d) / (b * c)
      
      # test choice: fallback to binomial when Fisher is degenerate
      if (b < min_out_cases || a == n_in) {
        alt <- if (is.finite(prev_all) && prev_in >= prev_all) "greater" else "less"
        bt  <- try(stats::binom.test(a, n_in, p = prev_all, alternative = alt), silent = TRUE)
        if (inherits(bt, "try-error")) return(NULL)
        pval <- bt$p.value
      } else {
        ft <- try(stats::fisher.test(
          matrix(c(a, b, c, d), nrow = 2, byrow = TRUE),
          alternative = FISHER_ALT),
          silent = TRUE)
        if (inherits(ft, "try-error")) return(NULL)
        pval <- ft$p.value
        OR   <- unname(ft$estimate)
      }
      
      data.frame(
        cluster = cl, diagnosis = v,
        a = a, b = b, c = c, d = d,
        n_in = n_in,
        prev_in = prev_in,
        OR = OR,
        p = pval,
        stringsAsFactors = FALSE
      )
    })
    
    rows <- Filter(Negate(is.null), rows)
    if (length(rows)) { t <- t + 1L; out[[t]] <- do.call(rbind, rows) }
  }
  
  out <- Filter(Negate(is.null), out)
  if (!length(out)) return(list(enrichment = NULL, majors = character(0)))
  
  res <- do.call(rbind, out)
  res <- res %>% dplyr::group_by(cluster) %>%
    dplyr::mutate(FDR = p.adjust(p, "BH")) %>%
    dplyr::ungroup()
  
  majors <- res$diagnosis[
    res$FDR <= alpha_fdr & res$prev_in >= min_prev_in & res$OR >= min_or
  ]
  
  list(
    enrichment = res[order(res$cluster, res$FDR, -res$OR), ],
    majors = unique(majors)
  )
}

# =============================== 10) Pillar B ==================================
# Label localisation: assortativity and kNN purity with label-shuffle p-values.
label_localization_table <- function(g, DXu_id, B = 1000,
                                     n_pos_min = 10, n_neg_min = 10,
                                     counts_all = NULL,
                                     exclude = c("ANY_DX")) {
  stopifnot(igraph::is.igraph(g))
  nV <- igraph::vcount(g)
  if (nV == 0L) return(data.frame())
  
  # Edgelist once (undirected count convention handled below)
  el <- if (igraph::ecount(g) > 0L) igraph::as_edgelist(g, names = FALSE) else matrix(integer(), ncol = 2)
  
  # Neighbor list once; "all" makes it safe for directed graphs
  nbrs <- igraph::adjacent_vertices(g, V(g), mode = "all")
  
  # Helper for purity calculation on vertex i given 0/1 labels 'lab'
  frac_pos_neigh <- function(i, lab) {
    nb <- as.integer(nbrs[[i]])
    nb <- setdiff(nb, i)                    # drop self if present
    if (length(nb) == 0L) return(NA_real_)  # isolated
    mean(lab[nb], na.rm = TRUE)
  }
  
  cols <- setdiff(names(DXu_id), c("participant_id", "mult", exclude))
  
  out <- lapply(cols, function(v) {
    # Tally positives/negatives for gating (optionally from an external count table)
    if (!is.null(counts_all)) {
      row <- counts_all[counts_all$dx == v, , drop = FALSE]
      pos_tot <- if (nrow(row)) row$n1[1] else sum(DXu_id[[v]] == 1L, na.rm = TRUE)
      neg_tot <- if (nrow(row)) row$n0[1] else sum(DXu_id[[v]] == 0L, na.rm = TRUE)
    } else {
      pos_tot <- sum(DXu_id[[v]] == 1L, na.rm = TRUE)
      neg_tot <- sum(DXu_id[[v]] == 0L, na.rm = TRUE)
    }
    if (pos_tot < n_pos_min || neg_tot < n_neg_min) return(NULL)
    
    # Binary labels (0/1), NA -> 0 (conservative)
    z <- as.integer(DXu_id[[v]] == 1L); z[is.na(z)] <- 0L
    if (length(z) != nV) return(NULL)  # safety: label length must match graph
    
    ## --- Assortativity via 2x2 edge table (symmetrized) ---
    if (nrow(el) > 0L) {
      K <- 2L
      zi <- z[el[, 1]]; zj <- z[el[, 2]]
      tab <- matrix(0, K, K)
      for (i in seq_len(nrow(el))) {
        a <- zi[i] + 1L; b <- zj[i] + 1L
        tab[a, b] <- tab[a, b] + 1L
        tab[b, a] <- tab[b, a] + 1L  # symmetrize
      }
      e <- tab / sum(tab)
      a <- rowSums(e); b <- colSums(e)
      denom <- 1 - sum(a * b)
      r_obs <- if (abs(denom) < 1e-12) NA_real_ else (sum(diag(e)) - sum(a * b)) / denom
      
      r_null <- replicate(B, {
        zsh <- sample(z, length(z), replace = FALSE)
        zsi <- zsh[el[, 1]]; zsj <- zsh[el[, 2]]
        tb0 <- matrix(0, K, K)
        for (i in seq_len(nrow(el))) {
          a0 <- zsi[i] + 1L; b0 <- zsj[i] + 1L
          tb0[a0, b0] <- tb0[a0, b0] + 1L
          tb0[b0, a0] <- tb0[b0, a0] + 1L
        }
        e0 <- tb0 / sum(tb0); a0 <- rowSums(e0); b0 <- colSums(e0)
        d0 <- 1 - sum(a0 * b0)
        if (abs(d0) < 1e-12) NA_real_ else (sum(diag(e0)) - sum(a0 * b0)) / d0
      })
      r_null <- r_null[is.finite(r_null)]
      p_r <- if (!is.finite(r_obs) || length(r_null) == 0L) NA_real_
      else (sum(r_null >= r_obs) + 1) / (length(r_null) + 1)
    } else {
      r_obs <- NA_real_; p_r <- NA_real_
    }
    
    ## --- kNN purity: mean frac of positive neighbors among positive vertices ---
    pos_idx <- which(z == 1L)
    pur_obs <- if (length(pos_idx)) {
      mean(vapply(pos_idx, frac_pos_neigh, numeric(1), lab = z), na.rm = TRUE)
    } else NA_real_
    
    pur_null <- replicate(B, {
      zsh <- sample(z, length(z), replace = FALSE)     # label shuffle preserves class count
      pos_sh <- which(zsh == 1L)
      if (!length(pos_sh)) return(NA_real_)
      mean(vapply(pos_sh, frac_pos_neigh, numeric(1), lab = zsh), na.rm = TRUE)
    })
    pur_null <- pur_null[is.finite(pur_null)]
    p_pur <- if (!is.finite(pur_obs) || length(pur_null) == 0L) NA_real_
    else (sum(pur_null >= pur_obs) + 1) / (length(pur_null) + 1)
    
    data.frame(
      dx = v,
      assort_r = r_obs, assort_p = p_r,
      knn_purity = pur_obs, knn_p = p_pur,
      stringsAsFactors = FALSE
    )
  })
  
  do.call(rbind, Filter(Negate(is.null), out))
}

# =============================== 11) Pillar C ==================================
# One-vs-rest AUC from neighbour-label scores; multiplicity-weighted.
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


# ====================== 12) Over-representation visual tools ===================
# Compact heatmap of top Dx per cluster (lift + stars from q)
compute_overrep <- function(DX, clusters_all,
                            min_in = NIN_MIN %||% 5L,
                            min_out = NOUT_MIN %||% 5L,
                            cutoff = 0.5){
  stopifnot(all(c("participant_id","cluster") %in% names(clusters_all)))
  
  # make sure rownames assignment works inside align_to_ids()
  DX <- as.data.frame(DX)
  
  ids_ref <- intersect(
    as.character(DX$participant_id),
    as.character(clusters_all$participant_id)
  )
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
      p <- tryCatch(
        suppressWarnings(
          fisher.test(matrix(c(a, b, c, d), nrow = 2, byrow = TRUE),
                      alternative = FISHER_ALT)$p.value
        ),
        error = function(e) NA_real_
      )
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
    T$q < 0.01  ~ "**",
    T$q < 0.05  ~ "*",
  TRUE ~ if (SHOW_NS_LABELS) {"ns"} else {""}
  )
  T[order(T$cluster, -T$log2_lift, T$q), ]
}

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
    scale_shared_div(name = "log2(lift)", limits = c(-cap, cap)) +
    ggplot2::labs(x = NULL, y = NULL, title = "Over-representation by cluster") +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(panel.grid = element_blank(),
                   axis.text.x = ggplot2::element_text(size = 10),
                   axis.text.y = ggplot2::element_text(size = 9))
}

# Numbers table for lift with significance marks; also writes PNG
plot_overrep_lift_table <- function(tab_out,
                                    file = "FIG_overrep_lift_table.png",
                                    digits = 2) {
  stopifnot(all(c("cluster","dx","lift","label") %in% names(tab_out)))
  df <- tab_out
  df$cluster <- factor(df$cluster, levels = sort(unique(df$cluster)))
  df$dx      <- factor(df$dx,      levels = levels(df$dx))
  lift_txt <- ifelse(is.finite(df$lift), sprintf(paste0("%.", digits, "f"), df$lift), "")
  sig_txt  <- ifelse(df$label %in% c("*","**"), paste0(" ", df$label), "")
  df$cell  <- paste0(lift_txt, sig_txt)
  p <- ggplot2::ggplot(df, ggplot2::aes(x = dx, y = cluster)) +
    ggplot2::geom_tile(fill = "white", colour = "grey85") +
    ggplot2::geom_text(ggplot2::aes(label = cell), size = 3.2) +
    ggplot2::labs(title = "Lift (in / overall) with significance",
                  x = NULL, y = "cluster",
                  caption = "Cells show lift; * FDR≤0.05, ** FDR≤0.01.") +
    ggplot2::theme_minimal(12) +
    ggplot2::theme(panel.grid = ggplot2::element_blank(),
                   axis.text.x = ggplot2::element_text(angle = 55, hjust = 1, vjust = 1))
  print(p)
  ggplot2::ggsave(file, p, width = 9, height = 7, dpi = 300, bg = "white")
}

# Full over-representation heatmap with fixed denominator for overall prevalence.
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
    ref_ids_for_overall = NULL,
    drop_dx = NULL,
    clusters_on_y = TRUE          # <- new: controls orientation only
){
  DX <- as.data.frame(DX)
  
  ids_cur <- intersect(as.character(DX$participant_id),
                       as.character(clusters_all$participant_id))
  DX_cur <- align_to_ids(DX, ids_cur)[, setdiff(names(DX), "participant_id"), drop = FALSE]
  
  ids_ref <- if (is.null(ref_ids_for_overall)) ids_cur else
    intersect(as.character(DX$participant_id), as.character(ref_ids_for_overall))
  DX_ref <- align_to_ids(DX, ids_ref)[, setdiff(names(DX), "participant_id"), drop = FALSE]
  
  cols <- intersect(dx_to_plot, colnames(DX_cur))
  if (!length(cols)) stop("dx_to_plot has no columns present in DX after filtering")
  
  cl_map <- setNames(clusters_all$cluster, clusters_all$participant_id)
  cl_cur <- as.integer(cl_map[ids_cur])
  
  # overall prevalence for fixed denominator
  prev_all_vec <- vapply(cols, function(v){
    a <- sum(DX_ref[[v]] == 1L, na.rm = TRUE)
    c0 <- sum(DX_ref[[v]] == 0L, na.rm = TRUE)
    if (a + c0 > 0) a/(a + c0) else NA_real_
  }, numeric(1)); names(prev_all_vec) <- cols
  
  N <- length(ids_cur)
  cl_levels <- sort(unique(cl_cur))
  
  rows <- list()
  for (cl in cl_levels){
    in_cl  <- cl_cur == cl
    out_cl <- !in_cl
    n_in   <- sum(in_cl); n_out <- N - n_in
    for (v in cols){
      a <- sum(DX_cur[[v]][in_cl]  == 1L, na.rm = TRUE)  # pos in
      b <- n_in  - a                                     # neg in
      c <- sum(DX_cur[[v]][out_cl] == 1L, na.rm = TRUE)  # pos out
      d <- n_out - c                                     # neg out
      
      prev_in  <- if ((a + b) > 0) a/(a + b) else NA_real_
      prev_all <- prev_all_vec[[v]]
      
      # lift vs fixed overall prevalence
      lift_raw <- if (is.finite(prev_in) && is.finite(prev_all) && prev_all > 0) prev_in/prev_all else NA_real_
      
      p <- NA_real_
      tested <- FALSE
      
      if ((a + c) >= min_total_cases && a >= min_in_cases && c >= min_out_cases &&
          all(c(n_in, n_out) > 0)) {
        # --- Fisher with the SAME fill convention as compute_overrep(): by column
        # matrix(c(a,b,c,d), 2, 2) -> columns = in/out; rows = pos/neg
        ft <- try(stats::fisher.test(
          matrix(c(a, b, c, d), nrow = 2, byrow = TRUE),
          alternative = FISHER_ALT),
          silent = TRUE)
        if (!inherits(ft, "try-error")) { p <- ft$p.value; tested <- TRUE }
      } else if (n_in > 0 && is.finite(prev_all)) {
        alt <- if (is.finite(prev_in) && prev_in >= prev_all) "greater" else "less"
        bt  <- try(stats::binom.test(a, n_in, p = prev_all, alternative = alt), silent = TRUE)
        if (!inherits(bt, "try-error")) { p <- bt$p.value; tested <- TRUE }
      }
      
      rows[[length(rows) + 1L]] <- data.frame(
        cluster = cl, dx = v,
        a=a,b=b,c=c,d=d, n_in=n_in, n_out=n_out,
        prev_in = prev_in, prev_all = prev_all,
        lift = lift_raw, p_raw = if (tested) p else NA_real_,
        stringsAsFactors = FALSE
      )
    }
  }
  tab <- do.call(rbind, rows)
  
  # pretty labels (fallback to code)
  code_to_label <- c(
    "NODIAG" = "NODIAG" #keep nodiag, no correction
    # (keep/add your long pretty-name map here if you want)
  )
  tab$dx_lab <- unname(ifelse(tab$dx %in% names(code_to_label), code_to_label[tab$dx], as.character(tab$dx)))
  
  # hide ultra-rare from plotting
  tot_by_dx <- vapply(cols, function(v){
    sum(DX_ref[[v]] == 1L, na.rm = TRUE) + sum(DX_ref[[v]] == 0L, na.rm = TRUE)
  }, integer(1))
  rare_dx <- names(tot_by_dx)[tot_by_dx < min_total_cases_plot]
  if (length(rare_dx)) {
    m <- tab$dx %in% rare_dx
    tab$p_raw[m] <- NA_real_
    tab$lift[m]  <- NA_real_
  }
  
  # --- compute OR and do FDR per *cluster* (match Pillar A) ---
  tab$OR <- with(tab, ifelse(b == 0 | c == 0, Inf, (a * d) / (b * c)))
  
  tab$q <- NA_real_
  for (cl in unique(tab$cluster)) {
    i <- which(tab$cluster == cl & is.finite(tab$p_raw))
    if (length(i)) tab$q[i] <- if (use_fdr) p.adjust(tab$p_raw[i], "BH") else tab$p_raw[i]
  }
  
  tab$label <- dplyr::case_when(
    is.finite(tab$q) & tab$q < alpha2 ~ "**",
    is.finite(tab$q) & tab$q < alpha1 ~ "*",
    TRUE ~ "ns"
  )
  
  # colour scale values
  eps <- 1e-12
  tab$val   <- log2(pmax(tab$lift, eps))
  cap <- 2
  tab$val_c <- pmin(pmax(tab$val, -cap), cap)
  
  thresh <- 0.6                     # fraction of cap at which tiles get "dark"
  tab$text_col <- ifelse(is.na(tab$val_c), NA,
                         ifelse(abs(tab$val_c) > cap*thresh, "white", "black"))
  
  if (!show_under_colors) {
    idx <- which(is.finite(tab$val) & tab$val < 0)
    if (length(idx)) tab$val_c[idx] <- NA_real_
  }
  
  # lock factor orders consistently
  tab$dx      <- factor(tab$dx, levels = cols)
  tab$dx_lab  <- factor(tab$dx_lab, levels = unique(tab$dx_lab[match(cols, tab$dx)]))
  cl_lev      <- sort(unique(tab$cluster))
  tab$cluster_f <- factor(paste0("C", tab$cluster), levels = paste0("C", cl_lev))
  
  # draw  (replace this whole block)
  # build numeric grid to use geom_rect with tiny overlap
  if (clusters_on_y) {
    tab$x_i <- as.numeric(tab$dx_lab);    tab$y_i <- as.numeric(tab$cluster_f)
    x_breaks <- seq_along(levels(tab$dx_lab));    x_labels <- levels(tab$dx_lab)
    y_breaks <- seq_along(levels(tab$cluster_f)); y_labels <- levels(tab$cluster_f)
  } else {
    tab$x_i <- as.numeric(tab$cluster_f); tab$y_i <- as.numeric(tab$dx_lab)
    x_breaks <- seq_along(levels(tab$cluster_f)); x_labels <- levels(tab$cluster_f)
    y_breaks <- seq_along(levels(tab$dx_lab));    y_labels <- levels(tab$dx_lab)
  }
  
  eps <- 1e-2  # tiny overlap to kill vector hairlines
  
  p <- ggplot2::ggplot(tab) +
    ggplot2::geom_rect(
      ggplot2::aes(xmin = x_i - 0.5 - eps,
                   xmax = x_i + 0.5 + eps,
                   ymin = y_i - 0.5 - eps,
                   ymax = y_i + 0.5 + eps,
                   fill = val_c),
      colour = NA
    ) +
    ggplot2::geom_text(
      ggplot2::aes(x = x_i, y = y_i, label = label, colour = text_col),
      size = 3.3, fontface = "bold", na.rm = TRUE
    ) +
    ggplot2::scale_colour_identity(guide = "none") +
    ggplot2::scale_x_continuous(breaks = x_breaks, labels = x_labels, expand = c(0,0)) +
    ggplot2::scale_y_continuous(breaks = y_breaks, labels = y_labels, expand = c(0,0)) +
    scale_shared_div(name = "log2(lift)", limits = c(-cap, cap)) +
    ggplot2::labs(x = NULL, y = NULL, title = title) +
    ggplot2::theme_minimal(base_size = 11) +
    ggplot2::theme(panel.grid = ggplot2::element_blank(),
                   axis.text.x = ggplot2::element_text(angle = 55, hjust = 1, vjust = 1))
  print(p)
  if (requireNamespace("ragg", quietly = TRUE)) {
    ggplot2::ggsave(
      filename = file,
      plot = p,
      width = 9, height = 7, dpi = 300, bg = "white",
      device = ragg::agg_png
    )
  } else {
    ggplot2::ggsave(
      filename = file,
      plot = p,
      width = 9, height = 7, dpi = 300, bg = "white"
    )
  }  
  ggplot2::ggsave(
    "FIG_overrep_lift_sig.pdf", p,
    width = 9, height = 7, bg = "white",
    device = grDevices::cairo_pdf
  )
  # compact table (same as before, but with stars only)
  out <- tab[, c("cluster","dx","lift","q","label")]
  out$cluster <- as.integer(out$cluster)
  out$dx <- factor(out$dx, levels = cols)
  out <- out[order(out$cluster, as.integer(out$dx)), ]
  rownames(out) <- NULL
  invisible(out)
}

# ============================== 13) Sensitivity tools ==========================
# CV curve for MDS distance preservation
cv_r2_mds <- function(D, k_grid = 1:8, K = 5, seed = NULL){
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
      Xtr    <- sweep(V, 2, sqrt(vals), `*`)
      Dte2 <- Dm[te, tr, drop = FALSE]^2
      cmtr <- colMeans(Dtr2)
      rmte <- rowMeans(Dte2)
      mu   <- mean(Dtr2)
      Gte  <- -0.5 * ( Dte2 - matrix(rmte,  nrow(Dte2), ncol(Dte2), byrow = FALSE)
                       - matrix(cmtr,  nrow(Dte2), ncol(Dte2), byrow = TRUE) + mu )
      Xte <- sweep(Gte %*% V, 2, 1/sqrt(vals), `*`)
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

# Helper to run membership + null stats for a (k,variant) pair
get_membership <- function(D, k = K_KNN, union = UNION_KNN, mult,
                           B = 500, seed = 42,
                           deg_reps = 200L, null_scope = "kept") {
  g  <- knn_graph_from_dist(D, k = k, union = union, mult = mult)
  lv <- louvain_with_stats(g, D, n_perm = B, min_size = MIN_CLUSTER_SIZE,
                           min_weight = MIN_CLUSTER_WEIGHT, v_weight = mult,
                           seed = seed, deg_reps = deg_reps, null_scope = null_scope
  )
  m  <- lv$membership
  names(m) <- rownames(as.matrix(D))
  q_mean <- mean(lv$Q_null, na.rm = TRUE); q_sd <- stats::sd(lv$Q_null, na.rm = TRUE)
  z_w    <- if (is.finite(q_mean) && is.finite(q_sd) && q_sd > 0) (lv$Q - q_mean)/q_sd else NA_real_
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

# ============================ 14) Main pipeline ================================
# Load data (from memory if DX exists, otherwise from DX_CSV_PATH) and
# apply prevalence/count guards + 0/1/NA validation.
DX <- load_dx_wide(
  path       = DX_CSV_PATH,
  include_nodiag = INCLUDE_NODIAG,
  min_prev   = DX_MIN_PREV,
  max_prev   = DX_MAX_PREV
)
# Deduplicate participant rows into unique diagnosis signatures (DXu), keeping
# multiplicities (mult). Also retain per-row signature strings for expansion.
dd    <- dedup_dx(DX)
DXu   <- dd$DXu
sigU  <- apply(DXu[, setdiff(names(DXu), c("participant_id","mult")), drop = FALSE],
               1, paste0, collapse = "")
sigA  <- dd$sig

# Participant-level counts (n1, n0, prev) used for guards/pillar B.
counts_all <- build_counts(DX)

# Keep only unique profiles with at least one diagnosis present (exclude all-zero).
colsU     <- setdiff(names(DXu), c("participant_id","mult"))
keep_idU  <- rowSums(DXu[, colsU, drop = FALSE], na.rm = TRUE) > 0L
DXu_id    <- DXu[keep_idU, , drop = FALSE]
sigU_id   <- sigU[keep_idU]
multU     <- as.numeric(DXu_id$mult); multU[!is.finite(multU) | multU <= 0] <- 1

cat(sprintf("[DX] unique profiles: %d / %d (%.1f%% duplicates)\n",
            nrow(DXu), nrow(DX), 100 * (1 - nrow(DXu) / nrow(DX))))
cat(sprintf("[DX] unique non-zero profiles used: %d\n", nrow(DXu_id)))

# ------------------------ Distances + intrinsic dimension ---------------------
# Main asymmetric-binary Gower distance on the unique non-zero profiles.
D_dx <- gower_dist_dx(DXu_id)
if (is.null(D_dx)) stop("DX after dedup has <2 varying diagnosis columns.")
Dm <- as.matrix(D_dx); diag(Dm) <- Inf

# Intrinsic dimension on all points and on a robust "core band".
ID_twonn_all  <- suppressWarnings(twonn_id_from_dist(D_dx))
idx_core      <- core_band_idx(D_dx, k = max(5, min(20, K_KNN)), band = c(0.15, 0.85))
ID_twonn_core <- if (length(idx_core) >= 5){
  suppressWarnings(twonn_id_from_dist(stats::as.dist(Dm[idx_core, idx_core, drop = FALSE])))
} else { NA_real_ }
ID_lbmle_core <- if (length(idx_core) >= 5) {
  suppressWarnings(lb_mle_id(Dm[idx_core, idx_core, drop = FALSE], 5, 15))
} else { NA_real_ }

cat(sprintf("[ID] TwoNN_all=%.2f | TwoNN_core=%.2f | LB_core=%.2f | n_core=%s\n",
            ID_twonn_all, ID_twonn_core, ID_lbmle_core,
            if (length(idx_core)) length(idx_core) else "NA"))

# ---------------------------- kNN graph + clustering --------------------------
# Build symmetric (union/mutual) kNN graph with Gaussian weights; local scaling
# and multiplicity upweighting are enabled.
g_dx <- knn_graph_from_dist(
  D = D_dx,
  k = K_KNN,
  union = UNION_KNN,
  mult = multU,
  local_scale = TRUE
)
report_knn_graph(g_dx)

# Run Louvain and compute null statistics (weight-shuffle + degree-null).
lv <- louvain_with_stats(
  g = g_dx,
  D = D_dx,
  n_perm = N_PERM,
  min_size = MIN_CLUSTER_SIZE,
  min_weight = MIN_CLUSTER_WEIGHT,
  v_weight = multU,
  seed = SEED_GLOBAL,
  deg_reps = N_PERM,
  null_scope = "kept"   # degree-null on kept subgraph for baseline
)

# --- Absorb leftovers (0-labeled) to nearest kept community (defensible) ---
zeros_before <- sum(lv$membership == 0L)
abs <- absorb_leftovers_defensible(
  g = g_dx, D = D_dx, memb = lv$membership,
  DXu_id = DXu_id, mult = multU,
  tau = 0.70, k_min = 3, shared_min = 1,
  B = 200, alpha = 0.05, seed = SEED_GLOBAL
)
lv$membership <- abs$membership
zeros_after  <- sum(lv$membership == 0L)
cat(sprintf("[Absorb] assigned %d / %d leftover profiles (BH q≤0.05)\n",
            zeros_before - zeros_after, zeros_before))
if (length(abs$absorbed)) {
  readr::write_csv(
    data.frame(v_idx = abs$absorbed,
               q = as.numeric(abs$q[as.character(abs$absorbed)])),
    "absorbed_vertices_q.csv"
  )
}
KEPT_CLUSTERS <- sort(setdiff(unique(lv$membership), 0L))

K_kept <- length(setdiff(unique(lv$membership), 0L))
names(lv$membership) <- rownames(as.matrix(D_dx))

cat(sprintf("[Clustering] %s Q=%.3f | p_deg=%.5f | p_two_w=%.5f | S=%.3f (p=%.5f) | K=%d | kept=%d\n",
            COMM_LABEL, lv$Q, lv$p_deg, lv$Q_p_two, lv$S, lv$S_p, K_KNN, K_kept))

cl_kept    <- KEPT_CLUSTERS
size_by_cl <- as.integer(tabulate(factor(lv$membership, levels = cl_kept)))
mass_by_cl <- as.numeric(tapply(multU, factor(lv$membership, levels = cl_kept), sum))

audit_tab <- data.frame(cluster = cl_kept, size = size_by_cl, mass = mass_by_cl)
print(audit_tab)

# Degree-preserving null on the KEPT subgraph
kept_idx    <- which(lv$membership != 0L)
g_deg_kept  <- igraph::induced_subgraph(g_dx, vids = kept_idx)
Q_null_deg  <- modularity_degseq_null(g_deg_kept, reps = N_PERM, niter_mult = 30L)
mu          <- mean(Q_null_deg, na.rm = TRUE)
sd0         <- stats::sd(Q_null_deg, na.rm = TRUE)
p_emp       <- (sum(Q_null_deg >= lv$Q, na.rm = TRUE) + 1) / (sum(is.finite(Q_null_deg)) + 1)
z_deg       <- (lv$Q - mu) / sd0

readr::write_csv(
  data.frame(Q_null_deg = Q_null_deg, Q_obs = lv$Q, Q_p_deg = p_emp),
  "modularity_degree_null_kept.csv"
)
message(sprintf("[Degree-null | kept] Q=%.3f | mu=%.3f sigma=%.3f | z=%.2f | p(emp)=%.3f",
                lv$Q, mu, sd0, z_deg, p_emp))

# Column-shuffle null: preserve per-diagnosis prevalence; re-run clustering
# 
# For each replicate:
#   - independently permute each diagnosis column (keeps its 0/1 prevalence),
#   - deduplicate profiles (with multiplicity),
#   - keep non-zero profiles,
#   - rebuild distances and kNN graph,
#   - run Louvain once (no internal nulls), and
#   - record Q, S, and TwoNN ID.
#
# Returns a data.frame with one row per replicate.
null_column_shuffle <- function(DX, K_KNN, UNION_KNN, MIN_CLUSTER_SIZE, reps = 300L, seed = 42){
  stopifnot(is.data.frame(DX), "participant_id" %in% names(DX))
  cols <- setdiff(names(DX), "participant_id")
  if (!length(cols)) stop("[null_column_shuffle] No diagnosis columns found.")
  
  do_one <- function(r){
    # 1) Prevalence-preserving column-wise shuffles
    DXs <- DX
    for (v in cols) DXs[[v]] <- sample(DXs[[v]])
    
    # 2) Dedup + keep only non-zero profiles
    dd   <- dedup_dx(DXs)
    DXu  <- dd$DXu
    keep <- rowSums(DXu[, setdiff(names(DXu), c("participant_id","mult")), drop = FALSE],
                    na.rm = TRUE) > 0L
    DXu_id <- DXu[keep, , drop = FALSE]
    if (nrow(DXu_id) < 5L) return(list(Q = NA_real_, S = NA_real_, ID = NA_real_))
    
    # 3) Distances
    D <- gower_dist_dx(DXu_id)
    if (is.null(D)) return(list(Q = NA_real_, S = NA_real_, ID = NA_real_))
    
    # 4) kNN graph (multiplicity-weighted), single Louvain pass
    g <- try(knn_graph_from_dist(D, k = K_KNN, union = UNION_KNN, mult = DXu_id$mult),
             silent = TRUE)
    if (inherits(g, "try-error") || igraph::ecount(g) == 0L)
      return(list(Q = NA_real_, S = NA_real_, ID = NA_real_))
    
    lv0 <- louvain_with_stats(g, D, n_perm = 0,
                              min_size = MIN_CLUSTER_SIZE, seed = seed + r)
    
    # 5) Return stats
    list(Q = lv0$Q,
         S = lv0$S,
         ID = suppressWarnings(twonn_id_from_dist(D)))
  }
  
  progressr::handlers(global = TRUE)
  progressr::with_progress({
    p <- progressr::progressor(steps = reps)
    out <- future.apply::future_lapply(seq_len(reps), function(r){
      res <- do_one(r); p(); res
    }, future.seed = TRUE)
    as.data.frame(do.call(rbind, lapply(out, as.data.frame)))
  })
}

# ------------------------- Column-shuffle (prevalence) nulls ------------------
# Preserve per-column prevalence by permuting columns independently; re-run
# the entire clustering stack to get nulls for Q/S and an ID reference.
# ------------------------- Column-shuffle (prevalence) nulls ------------------
nulls_col <- (function(){
  old_plan <- future::plan()                     # save current strategy
  on.exit(future::plan(old_plan), add = TRUE)    # restore on exit
  future::plan(future::multisession,
               workers = max(1, parallel::detectCores() - 1))
  
  ns <- null_column_shuffle(
    DX,
    K_KNN = K_KNN,
    UNION_KNN = UNION_KNN,
    MIN_CLUSTER_SIZE = MIN_CLUSTER_SIZE,
    reps = 300
  )
  
  Q_p_col  <- mean(ns$Q  >= lv$Q, na.rm = TRUE)
  S_p_col  <- mean(ns$S  >= lv$S, na.rm = TRUE)
  ID_p_high <- mean(ns$ID >= ID_twonn_all, na.rm = TRUE)
  ID_p_low  <- mean(ns$ID <= ID_twonn_all, na.rm = TRUE)
  ID_p_two  <- 2 * min(ID_p_high, ID_p_low)
  
  message(sprintf("[Column-shuffle null] p(Q)=%.3f, p(S)=%.3f", Q_p_col, S_p_col))
  message(sprintf("[Column-shuffle null] p_high(ID)=%.3f, p_low(ID)=%.3f, p_two(ID)=%.3f",
                  ID_p_high, ID_p_low, ID_p_two))
  ns
})()

# --------------------------------- Bootstrap ARI -------------------------------
# Resample unique profiles with probability proportional to multiplicity, rebuild
# the graph/partition, and compute ARI against the baseline partition.
boot_ari <- rep(NA_real_, N_BOOT)
p_boot   <- multU / sum(multU)

for (b in seq_len(N_BOOT)) {
  take <- sample(seq_len(nrow(DXu_id)), replace = TRUE, prob = p_boot)
  take <- sort(unique(take))
  if (length(take) < 5) next
  D_b <- stats::as.dist(as.matrix(D_dx)[take, take, drop = FALSE])
  g_b <- try(knn_graph_from_dist(D_b, k = K_KNN, union = UNION_KNN, mult = multU[take]),
             silent = TRUE)
  if (inherits(g_b, "try-error") || igraph::ecount(g_b) == 0L) next
  clb <- cluster_community(g_b)
  mb  <- as.integer(igraph::membership(clb))
  ref <- lv$membership[take]
  if (length(unique(ref)) < 2L || length(unique(mb)) < 2L) next
  ref <- setNames(lv$membership[take], as.character(take))
  mb  <- setNames(mb,                  as.character(take))
  boot_ari[b] <- align_ari(ref, mb)
}

ARI_med <- stats::median(boot_ari, na.rm = TRUE)
ARI_iqr <- stats::IQR(boot_ari, na.rm = TRUE)
cat(sprintf("[Stability] ARI_median=%.3f | ARI_IQR=%.3f\n", ARI_med, ARI_iqr))
readr::write_csv(data.frame(boot = seq_len(N_BOOT), ARI = boot_ari),
                 "cluster_bootstrap_ari.csv")

# -------------------------- Expand to all participants -------------------------
# Assign cluster labels to *all* participants by expanding membership from the
# unique-profile graph back through the signature mapping.
clusters_all <- expand_membership(lv$membership, sigU_id, sigA, DX$participant_id)
clusters_all <- subset(clusters_all, cluster %in% KEPT_CLUSTERS)
readr::write_csv(clusters_all, "cluster_membership_all_participants.csv")
message("[clusters_all] kept clusters: ", paste(sort(unique(clusters_all$cluster)), collapse=", "))
# ======================== Seed-bootstrapped sweep ==============================
# Purpose: quantify robustness to Louvain random initialisation for each (k, variant).

# 1) Fix a baseline partition for ARI comparisons
baseline <- get_membership(
  D = D_dx, k = 8, union = FALSE, mult = multU,
  B = 500, seed = 42, deg_reps = 200L, null_scope = "kept"
)
m_base <- baseline$m

# 2) One-run wrapper with a specified seed
run_cell_once <- function(D, k, union, mult, seed,
                          B = 500, deg_reps = 200L, null_scope = "kept") {
  gi <- get_membership(D, k = k, union = union, mult = mult,
                       B = B, seed = seed, deg_reps = deg_reps, null_scope = null_scope)
  # Align ARI to fixed baseline (drop zeros inside align_ari)
  ARI_vs_base <- align_ari(m_base, gi$m)
  data.frame(
    K = k,
    variant = if (union) "union" else "mutual",
    seed = seed,
    Q = gi$Q,
    p_pref = gi$p_pref,        # degree-null p (preferred)
    Q_p_two = gi$Q_p_two,      # weight-shuffle two-sided p
    S = gi$S,
    S_p = gi$S_p,
    z_w = gi$z_w,
    z_deg = gi$z_deg,
    n_kept = gi$n_kept,
    ARI_vs_base = ARI_vs_base,
    stringsAsFactors = FALSE
  )
}

# 3) Multi-seed bootstrap per grid cell
bootstrap_sweep <- function(D, K_GRID, VARIANT, mult,
                            R = 20,              # number of seeds per cell
                            seed0 = 202,         # base seed offset
                            B = 400,             # weight-null reps per run (lower if slow)
                            deg_reps = 150L,     # degree-null rewires per run (lower if slow)
                            null_scope = "kept") {
  grid <- expand.grid(K = K_GRID, variant = VARIANT, stringsAsFactors = FALSE)
  out_list <- vector("list", nrow(grid) * R)
  t <- 0L
  for (i in seq_len(nrow(grid))) {
    k <- grid$K[i]
    union_flag <- (grid$variant[i] == "union")
    for (r in seq_len(R)) {
      t <- t + 1L
      out_list[[t]] <- run_cell_once(
        D = D, k = k, union = union_flag, mult = mult,
        seed = seed0 + r, B = B, deg_reps = deg_reps, null_scope = null_scope
      )
    }
  }
  do.call(rbind, out_list)
}

# 4) Run the bootstrap sweep
plan(multisession, workers = max(1, parallel::detectCores() - 1))

boot_raw <- {
  grid <- expand.grid(K = K_GRID, variant = VARIANT, stringsAsFactors = FALSE)
  # progress handler
  handlers(global = TRUE)
  with_progress({
    p <- progressor(steps = nrow(grid) * 20)  # grid cells × seeds
    do.call(rbind, lapply(seq_len(nrow(grid)), function(i){
      k  <- grid$K[i]; union_flag <- grid$variant[i] == "union"
      seeds <- 202 + seq_len(20)
      future_rbind <- function(...) do.call(rbind, list(...))
      future_rbind(future_lapply(seeds, function(s) {
        res <- run_cell_once(D_dx, k, union_flag, multU,
                             seed = s, B = 300, deg_reps = 150L,
                             null_scope = "kept")
        p()   # increment progress
        res
      }, future.seed = TRUE))
    }))
  })
}

plan(sequential)

# 5) Summarise per (k, variant)
#    - share_signif_deg: fraction of seeds with degree-null p < 0.05
#    - share_signif_w:   fraction of seeds with weight-null two-sided p < 0.05
#    - share_both:       both nulls significant
#    - n_kept_mode:      modal number of kept clusters across seeds
# make sure boot_raw is a tidy data frame
boot_raw_df <- {
  x <- boot_raw
  if (is.data.frame(x)) x
  else if (is.list(x) && all(vapply(x, is.data.frame, logical(1)))) bind_rows(x)
  else if (is.matrix(x) || is.array(x)) as_tibble(as.data.frame(x, stringsAsFactors = FALSE))
  else as_tibble(x)
}

# (optional) ensure expected types
num_cols <- c("Q","S","z_deg","ARI_vs_base","Q_p_two","p_pref","n_kept")
for (nm in intersect(num_cols, names(boot_raw_df))) {
  if (!is.numeric(boot_raw_df[[nm]])) boot_raw_df[[nm]] <- suppressWarnings(as.numeric(boot_raw_df[[nm]]))
}
if ("K" %in% names(boot_raw_df))       boot_raw_df$K       <- as.integer(boot_raw_df$K)
if ("variant" %in% names(boot_raw_df)) boot_raw_df$variant <- as.character(boot_raw_df$variant)

summ_boot <- boot_raw_df %>%
  group_by(K, variant) %>%
  summarise(
    n_runs = n(),
    Q_med  = median(Q, na.rm = TRUE),
    Q_sd   = sd(Q, na.rm = TRUE),
    S_med  = median(S, na.rm = TRUE),
    S_sd   = sd(S, na.rm = TRUE),
    z_deg_med = median(z_deg, na.rm = TRUE),
    ARI_med   = median(ARI_vs_base, na.rm = TRUE),
    ARI_IQR   = IQR(ARI_vs_base, na.rm = TRUE),
    share_signif_deg = mean(p_pref < 0.05, na.rm = TRUE),
    share_signif_w   = mean(Q_p_two < 0.05, na.rm = TRUE),
    share_both       = mean((p_pref < 0.05) & (Q_p_two < 0.05), na.rm = TRUE),
    n_kept_mode = {
      x <- stats::na.omit(n_kept)
      if (length(x)) as.integer(names(sort(table(x), decreasing = TRUE))[1]) else NA_integer_
    },
    .groups = "drop"
  ) %>%
  arrange(desc(share_both), desc(Q_med), desc(ARI_med))

# 6) Save both long and summary tables
readr::write_csv(boot_raw_df, "clustering_sensitivity_seed_bootstrap_long.csv")
readr::write_csv(summ_boot,   "clustering_sensitivity_seed_bootstrap_summary.csv")

print(summ_boot)

# ============================ 15) Pillars A, B, C ==============================
# -------- Pillar A: per-cluster enrichment (Fisher/binomial with guards) -------
enr <- diagnosis_enrichment(
  DX, clusters_all,
  alpha_fdr = ALPHA_FDR, min_prev_in = MIN_PREV_IN_CL, min_or = MIN_OR,
  exclude = c("ANY_DX"),
  min_in_cases = MIN_CASES_IN, min_total_cases = MIN_CASES_TOTAL, min_out_cases = MIN_CASES_OUT
)
if (!is.null(enr$enrichment)) {
  readr::write_csv(enr$enrichment, "cluster_diagnosis_enrichment.csv")
  readr::write_csv(data.frame(major_diagnosis = enr$majors),
                   "selected_major_diagnoses.csv")
  cat(sprintf("[Majors A] %d diagnoses by enrichment\n", length(enr$majors)))
} else {
  cat("[Majors A] none\n")
}

# ------------- Pillar B: localisation (assortativity / kNN purity) -------------
loc_tab <- label_localization_table(
  g_dx, DXu_id, B = 1000,
  n_pos_min = MIN_CASES_TOTAL, n_neg_min = MIN_CASES_TOTAL,
  counts_all = counts_all,
  exclude = c("ANY_DX")
)
maj_B <- subset(loc_tab, pmin(assort_p, knn_p) <= ALPHA_LOCALIZE)$dx
readr::write_csv(loc_tab, "dx_label_localization.csv")

# --------------- Pillar C: predictability (kNN-score AUC, weighted) -----------
dx_cols <- setdiff(names(DXu_id), c("participant_id","mult","ANY_DX"))
auc_tab <- data.frame(
  dx  = dx_cols,
  AUC = vapply(
    dx_cols,
    function(v) {
      tryCatch(
        auc_one_vs_rest_knn_weighted(
          DXu_id, v, k = 10,
          pos_min = MIN_CASES_TOTAL, neg_min = MIN_CASES_TOTAL
        ),
        error = function(e) NA_real_
      )
    },
    numeric(1)
  )
)
maj_C <- subset(auc_tab, is.finite(AUC) & AUC >= AUC_MIN)$dx
readr::write_csv(auc_tab, "dx_predictability_auc_knn.csv")

# --------- Assemble major set: A ∪ B ∪ C with eligibility guards --------------
cand          <- unique(c(enr$majors, maj_B, maj_C))
maj_prev_pass <- with(counts_all, dx[prev >= PREV_MIN | n1 >= NCASE_MIN])
majors_union  <- sort(setdiff(intersect(cand, maj_prev_pass), c("ANY_DX","NODIAG")))
if (DENY_NOS) majors_union <- majors_union[!grepl("NOS$", majors_union, ignore.case = TRUE)]
readr::write_csv(data.frame(major_dx = majors_union),
                 "selected_major_diagnoses_union.csv")
cat(sprintf("[Majors | union] %d diagnoses selected (A ∪ B ∪ C)\n",
            length(majors_union)))

# ========================== 16) Over-representation figs =======================
# Columns to plot (include NODIAG if requested and present).
dx_core <- if (PLOT_MAJOR_SET == "A") enr$majors else majors_union
if (INCLUDE_NODIAG && "NODIAG" %in% names(DX)) dx_core <- c("NODIAG", dx_core)
dx_for_heatmap <- intersect(dx_core, setdiff(names(DX), c("participant_id","ANY_DX")))
if (!length(dx_for_heatmap))
  stop("dx_for_heatmap ended up empty — check majors_union / DX column names.")
cat("[heatmap] dx_for_heatmap =", paste(dx_for_heatmap, collapse = ", "), "\n")

# Freeze denominator (reference IDs) to the full overlap once.
if (is.null(rownames(DX))) rownames(DX) <- as.character(DX$participant_id)
ids_full <- intersect(as.character(DX$participant_id),
                      as.character(clusters_all$participant_id))

# Full figure with fixed denominator (saved as PNG + CSV table).
lift_tab_full <- plot_overrep_with_sig(
  DX = DX,
  clusters_all = clusters_all,
  dx_to_plot = dx_for_heatmap,
  min_in_cases = 5, min_out_cases = 5, min_total_cases = 10,
  use_fdr = TRUE, alpha1 = 0.05, alpha2 = 0.01,
  only_over = FALSE, min_total_cases_plot = 10, show_under_colors = TRUE,
  ref_ids_for_overall = as.character(DX$participant_id),
  file = "FIG_overrep_lift_sig.png",
  title = sprintf("Over-representation of major diagnoses by %s community", COMM_LABEL)
  )
readr::write_csv(lift_tab_full, "overrep_lift_table_full.csv")
plot_overrep_lift_table(lift_tab_full, "FIG_overrep_lift_table_full.png")

# Identify the "NoDiag" cluster and replot excluding it.
enr_tab <- diagnosis_enrichment(
  DX, clusters_all,
  alpha_fdr = 1, min_prev_in = 0, min_or = 0,
  exclude = "ANY_DX",
  min_in_cases = 0, min_total_cases = 0, min_out_cases = 0,
  kept_clusters = KEPT_CLUSTERS
)$enrichment

c_nodiag <- NA_integer_
if (!is.null(enr_tab)) {
  nd <- subset(enr_tab, diagnosis == "NODIAG")
  if (nrow(nd)) {
    nd$prev_all <- with(nd, (a + b) / pmax(1, a + b + c + d))
    nd$lift     <- with(nd, ifelse(prev_all > 0, prev_in / prev_all, NA_real_))
    nd <- subset(nd, a > 0 & is.finite(lift) & lift > 1)   # <- guard
    if (nrow(nd)) c_nodiag <- nd$cluster[which.max(nd$lift)]
  }
}
clusters_no_nodiag <- if (is.finite(c_nodiag)) subset(clusters_all, cluster != c_nodiag) else clusters_all

if (is.finite(c_nodiag)) {
  dx_for_heatmap_no_nodiag <- setdiff(dx_for_heatmap, "NODIAG")
  lift_tab_no_nodiag <- plot_overrep_with_sig(
    DX = DX,
    clusters_all = clusters_no_nodiag,
    dx_to_plot = dx_for_heatmap_no_nodiag,
    ref_ids_for_overall = as.character(DX$participant_id),
    file = "FIG_overrep_lift_sig_noNODIAG.png",
    title = sprintf("Over-representation by %s community — excluding NoDiag cluster", COMM_LABEL)
    )
  readr::write_csv(lift_tab_no_nodiag, "overrep_lift_table_noNODIAG.csv")
  plot_overrep_lift_table(lift_tab_no_nodiag, "FIG_overrep_lift_table_noNODIAG.png")
} else {
  message("[NoDiag] No qualifying NoDiag cluster — skipping 'exclude' plots.")
}

# Compact heatmap of top Dx per cluster (counts-based, same stars).
tab_sig2 <- compute_overrep(DX, clusters_all,
                            min_in = NIN_MIN, min_out = NOUT_MIN, cutoff = 0.5)

if (nrow(tab_sig2)) {
  readr::write_csv(tab_sig2, "dx_overrep_full.csv")
  p_heat <- plot_overrep_heatmap(tab_sig2, top_k_per_cluster = 12, cap = 2)
  print(p_heat)
  ggplot2::ggsave("FIG_dx_overrep_heatmap.png", p_heat,
                  width = 8.5, height = 7, dpi = 300, bg = "white")
  # Detail panel for the largest non-zero cluster.
  cl_sizes <- sort(table(clusters_all$cluster), decreasing = TRUE)
  cid <- as.integer(names(cl_sizes)[cl_sizes == max(cl_sizes) & names(cl_sizes) != "0"][1])
  if (is.finite(cid)) {
    # Reuse the slope-style detail plot from earlier section (function defined above).
    p_det <- (function(cid_local, tab_local, top_n = 15L, alpha = 0.05){
      T <- tab_local %>%
        dplyr::filter(cluster == cid_local) %>%
        dplyr::mutate(sig = is.finite(q) & q < alpha) %>%
        dplyr::arrange(dplyr::desc(log2_lift))
      if (!nrow(T)) return(ggplot2::ggplot() + ggplot2::theme_void())
      T$dx <- factor(T$dx, levels = rev(head(T$dx, top_n)))
      ci_in  <- wilson_ci(T$pos_in,  T$n_in)
      ci_out <- wilson_ci(T$pos_out, T$n_out)
      D <- tibble::tibble(
        dx    = rep(T$dx, each = 2),
        group = rep(c("in-cluster","out-of-cluster"), times = nrow(T)),
        prev  = as.numeric(t(cbind(T$in_prev,  T$out_prev))),
        lo    = as.numeric(t(cbind(ci_in[, "lo"],  ci_out[, "lo"]))),
        hi    = as.numeric(t(cbind(ci_in[, "hi"],  ci_out[, "hi"])))
      )
      S <- tibble::tibble(
        dx = T$dx,
        prev_in  = T$in_prev,
        prev_out = T$out_prev
      )
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
                      title = paste0("Cluster C", cid_local, ": Dx enrichment (top ", top_n, ")")) +
        ggplot2::theme_minimal(base_size = 11) +
        ggplot2::theme(legend.position = "top",
                       panel.grid.minor = ggplot2::element_blank())
    })(cid, tab_sig2, top_n = 15, alpha = 0.05)
    print(p_det)
    ggplot2::ggsave(sprintf("FIG_dx_overrep_C%d_detail.png", cid), p_det,
                    width = 7.5, height = 6.5, dpi = 300, bg = "white")
  }
  cat(sprintf("[overrep] wrote dx_overrep_full.csv; figures FIG_dx_overrep_heatmap.png and FIG_dx_overrep_C%d_detail.png\n",
              if (is.finite(cid)) cid else -1L))
} else {
  cat("[overrep] No cells passed basic counts; skipping figures.\n")
}

# ================================ 17) Visuals =================================
# 2D MDS quicklook (variance share and r^2 distance preservation in title).
fit2 <- stats::cmdscale(D_dx, k = 2, eig = TRUE, add = TRUE)
XY   <- fit2$points; colnames(XY) <- c("MDS1","MDS2")
eig  <- fit2$eig; pos <- eig[eig > 0]
frac <- if (length(pos)) sum(pmax(0, eig[1:2])) / sum(pos) else NA_real_
r2   <- suppressWarnings(stats::cor(as.numeric(D_dx), as.numeric(stats::dist(XY)))^2)
dfp  <- data.frame(MDS1 = XY[,1], MDS2 = XY[,2], cluster = factor(lv$membership))
dfp  <- subset(dfp, cluster != 0)
p2   <- ggplot2::ggplot(dfp, ggplot2::aes(MDS1, MDS2, color = cluster)) +
  ggplot2::geom_point(size = 1.9, alpha = 0.95) + ggplot2::coord_equal() +
  ggplot2::labs(title = sprintf("DX MDS2 - var=%.2f, r2(dist)=%.2f, Q=%.3f, sil=%.3f",
                                frac, r2, lv$Q, lv$S)) +
  ggplot2::theme_minimal(12)
print(p2)
ggplot2::ggsave("FIG_dxspace_clusters_mds2.png", p2,
                width = PLOT_WIDTH, height = PLOT_HEIGHT, dpi = PLOT_DPI)

# Cross-validated out-of-sample R^2(dist) vs MDS dimensionality (k = 1..8).
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
  message(sprintf("[MDS CV] best k ≈ %d (CV r^2=%.2f)", k_star, max(r2k$r2, na.rm = TRUE)))
}


# Optional: pairs matrix of first 6 MDS axes with 50% ellipses (requires GGally).
if (requireNamespace("GGally", quietly = TRUE)) {
  plot_mds_pairs <- function(D, memb, max_axes = 6,
                             file = "FIG_dxspace_mds_pairs.png"){
    fit <- stats::cmdscale(D, k = max_axes, eig = TRUE, add = TRUE)
    XY  <- as.data.frame(fit$points)
    pve <- { pos <- fit$eig[fit$eig > 0]
    if (length(pos)) pmax(0, fit$eig[seq_len(ncol(XY))]) / sum(pos)
    else rep(NA, ncol(XY)) }
    names(XY) <- paste0("MDS", seq_len(ncol(XY)), " (", sprintf("%.0f", 100 * pve), "%)")
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
      XY, columns = 1:(min(max_axes, ncol(XY) - 1)),
      ggplot2::aes(color = cluster), lower = list(continuous = lower_fun),
      diag = list(continuous = diag_fun), upper = list(continuous = "blank")
    ) + ggplot2::theme(strip.text = ggplot2::element_text(size = 9))
    ggplot2::ggsave(file, p, width = 10, height = 10, dpi = 150)
    print(p)
  }
  plot_mds_pairs(D_dx, lv$membership, max_axes = 6)
}

# ========================== 18) ΔQ per diagnosis (ablation) ====================
# Importance by ablation: set a given dx column to 0, rebuild clustering, and
# measure Q drop relative to the baseline Q.
deltaQ_per_dx <- function(DX, lv_ref_Q, K_KNN, union = UNION_KNN, MIN_CLUSTER_SIZE = 10){
  dx_cols <- setdiff(names(DX), c("participant_id","ANY_DX"))
  out <- lapply(dx_cols, function(v){
    DXm <- DX; DXm[[v]] <- 0L
    dd  <- dedup_dx(DXm); DXu <- dd$DXu
    keep <- rowSums(DXu[, setdiff(names(DXu), c("participant_id","mult")), drop = FALSE],
                    na.rm = TRUE) > 0
    DXu  <- DXu[keep, , drop = FALSE]; if (nrow(DXu) < 5) return(NULL)
    Dm  <- gower_dist_dx(DXu); if (is.null(Dm)) return(NULL)
    g   <- try(knn_graph_from_dist(Dm, k = K_KNN, union = union, mult = DXu$mult), silent = TRUE)
    if (inherits(g, "try-error") || igraph::ecount(g) == 0L) return(NULL)
    lv0 <- louvain_with_stats(g, Dm, n_perm = 0, min_size = MIN_CLUSTER_SIZE, seed = 42)
    data.frame(dx = v, Q_drop = lv_ref_Q - lv0$Q)
  })
  do.call(rbind, Filter(Negate(is.null), out))
}

dq <- deltaQ_per_dx(DX, lv$Q, K_KNN, UNION_KNN, MIN_CLUSTER_SIZE)
if (!is.null(dq) && nrow(dq)) readr::write_csv(dq, "dx_deltaQ.csv")

# ====================== 19) k × variant sensitivity sweep ======================
# Compute a baseline and then sweep across (K, variant), recording statistics and
# ARI vs baseline. Degree-null is run with null_scope="kept" for comparability.

base <- get_membership(
  D = D_dx, k = K_KNN, union = UNION_KNN,
  mult = multU, B = 500, seed = 42,
  deg_reps = 200L, null_scope = "kept"
)

grid <- expand.grid(K = K_GRID, variant = VARIANT, stringsAsFactors = FALSE)

# Fixed per-cell seeds, deterministic even when parallel
seed_vec <- 1000L + seq_len(nrow(grid))

# Run in parallel, then restore plan
old_plan <- future::plan()
on.exit(future::plan(old_plan), add = TRUE)
future::plan(future::multisession, workers = max(1, parallel::detectCores() - 1))

progressr::handlers(global = TRUE)
sens <- progressr::with_progress({
  p <- progressr::progressor(steps = nrow(grid))
  out <- future.apply::future_lapply(seq_len(nrow(grid)), function(i){
    k <- grid$K[i]
    u <- grid$variant[i] == "union"
    gi <- get_membership(
      D = D_dx, k = k, union = u, mult = multU,
      B = 500, seed = seed_vec[i],         # <- deterministic per cell
      deg_reps = 200L, null_scope = "kept"
    )
    p()
    data.frame(
      K = k, variant = grid$variant[i],
      Q = gi$Q, p_pref = gi$p_pref, Q_p_two = gi$Q_p_two,
      S = gi$S, S_p = gi$S_p,
      z_w = gi$z_w, z_deg = gi$z_deg,
      n_kept = gi$n_kept,
      ARI_vs_base = align_ari(base$m, gi$m)
    )
  }, future.seed = TRUE)  # stable RNG across workers
  do.call(rbind, out)
})

readr::write_csv(sens, "clustering_sensitivity_grid.csv")
print(sens)

# =========== 20) Bootstrap frequency of “majors” (by eligibility) =============
dx_pool <- setdiff(names(DX), c("participant_id","ANY_DX"))
zero_vec <- function() setNames(integer(length(dx_pool)), dx_pool)

boot_one <- function(b) {
  suppressPackageStartupMessages(library(igraph))
  p()  # <- progress tick
  
  success_by_dx <- zero_vec()
  trials_by_dx  <- zero_vec()
  n_valid <- 0L
  
  idx  <- sample(seq_len(nrow(DX)), replace = TRUE)
  DX_b <- DX[idx, , drop = FALSE]
  dd_b <- dedup_dx(DX_b); DXu_b <- dd_b$DXu
  colsB <- setdiff(names(DXu_b), c("participant_id","mult","ANY_DX"))
  keepB <- rowSums(DXu_b[, colsB, drop = FALSE], na.rm = TRUE) > 0L
  DXu_id_b <- DXu_b[keepB, , drop = FALSE]
  if (nrow(DXu_id_b) < 5) return(list(success=success_by_dx, trials=trials_by_dx, valid=n_valid))
  
  D_b <- gower_dist_dx(DXu_id_b); if (is.null(D_b)) return(list(success=success_by_dx, trials=trials_by_dx, valid=n_valid))
  g_b <- try(knn_graph_from_dist(D_b, k=K_KNN, union=UNION_KNN, mult=DXu_id_b$mult), silent=TRUE)
  if (inherits(g_b, "try-error") || igraph::ecount(g_b)==0L) return(list(success=success_by_dx, trials=trials_by_dx, valid=n_valid))
  
  lv_b <- louvain_with_stats(g_b, D_b, n_perm=0, min_size=MIN_CLUSTER_SIZE, seed=42)
  n_valid <- 1L
  
  sigU_b   <- apply(DXu_id_b[, colsB, drop = FALSE], 1, paste0, collapse = "")
  cl_all_b <- expand_membership(lv_b$membership, sigU_b, dd_b$sig, rownames(DX_b))
  
  # Pillar A
  enr_b <- diagnosis_enrichment(
    DX_b, cl_all_b,
    alpha_fdr=ALPHA_FDR, min_prev_in=MIN_PREV_IN_CL, min_or=MIN_OR,
    exclude=c("ANY_DX"),
    min_in_cases=MIN_CASES_TOTAL, min_total_cases=MIN_CASES_TOTAL, min_out_cases=MIN_CASES_TOTAL
  )
  mA <- if (!is.null(enr_b$majors)) enr_b$majors else character(0)
  
  # Pillar B
  counts_b <- build_counts(DX_b)
  loc_b <- label_localization_table(
    g_b, DXu_id_b, B=500,
    n_pos_min=MIN_CASES_TOTAL, n_neg_min=MIN_CASES_TOTAL,
    counts_all=counts_b
  )
  mB <- subset(loc_b, pmin(assort_p, knn_p) <= ALPHA_LOCALIZE)$dx
  
  # Pillar C
  auc_b <- data.frame(
    dx = colsB,
    AUC = vapply(
      colsB,
      function(v) suppressWarnings(
        auc_one_vs_rest_knn_weighted(
          DXu_id_b, v, k=10,
          pos_min=MIN_CASES_TOTAL, neg_min=MIN_CASES_TOTAL
        )
      ),
      numeric(1)
    ),
    stringsAsFactors = FALSE
  )
  mC <- subset(auc_b, is.finite(AUC) & AUC >= AUC_MIN)$dx
  
  # Eligibility & tallies
  eligible_b <- with(counts_b, dx[prev >= PREV_MIN | n1 >= NCASE_MIN])
  elig_here <- intersect(eligible_b, dx_pool)
  if (length(elig_here)) trials_by_dx[elig_here] <- trials_by_dx[elig_here] + 1L
  majors_b <- sort(intersect(unique(c(mA, mB, mC)), eligible_b))
  sel_here <- intersect(majors_b, dx_pool)
  if (length(sel_here)) success_by_dx[sel_here] <- success_by_dx[sel_here] + 1L
  
  list(success=success_by_dx, trials=trials_by_dx, valid=n_valid)
}

future::plan(future::multisession, workers = NCORES_PAR)

with_progress({
  p <- progressor(along = 1:N_BOOT)
  res <- future_lapply(
    seq_len(N_BOOT), boot_one,
    future.seed = TRUE,
    future.packages = c("igraph","progressr")  # ensure V()/E() exist + progressr hooks
  )
})

success_by_dx <- Reduce(`+`, lapply(res, `[[`, "success"), init = zero_vec())
trials_by_dx  <- Reduce(`+`, lapply(res, `[[`, "trials"),  init = zero_vec())
n_valid       <- sum(vapply(res, `[[`, integer(1), "valid"))

future::plan(future::sequential)

stab <- data.frame(
  dx     = names(success_by_dx),
  count  = as.integer(success_by_dx),
  trials = as.integer(trials_by_dx),
  stringsAsFactors = FALSE
)
stab <- subset(stab, trials > 0)
stab$freq <- with(stab, count / trials)
stab <- stab[order(stab$freq, decreasing = TRUE), ]
stab$lo <- stats::qbinom(0.025, size = stab$trials, prob = stab$freq) / stab$trials
stab$hi <- stats::qbinom(0.975, size = stab$trials, prob = stab$freq) / stab$trials
readr::write_csv(stab, "major_dx_bootstrap_frequency_by_eligibility.csv")
print(stab)

# Plot bootstrap stability with denominators in labels (optional, if ggplot2/scales present).
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

# =============================== 21) Run summary ===============================
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

# ===================== Cluster diagnostics (deduped graph level) =====================
# - n_profiles: # unique non-zero profiles (vertices)
# - weight: sum of multiplicities (participants)
# - pct_weight: share of participants
# - E_internal / W_internal: # and total weight of edges inside the cluster
# - cut_weight & conductance: separation vs the rest (lower conductance = clearer split)
# - density: unweighted edge density inside the cluster
# - intra_dist_mean: mean intra-cluster distance (Gower) across vertices
# - sil_mean / sil_median: silhouette for that cluster (computed vs all kept)
# - nodiag_prev / nodiag_lift: prevalence & lift of NODIAG (if present in DX)
# - top_dx: data.frame of top enriched DX (lift, BH q)

cluster_diagnostics <- function(
    DXu_id, multU, D_dx, g_dx, lv_membership,
    DX_full = NULL, clusters_all = NULL,
    top_dx = 5, min_in = 5, min_out = 5, min_total = 10
){
  stopifnot(igraph::is.igraph(g_dx))
  memb <- as.integer(lv_membership)
  cl_ids <- sort(setdiff(unique(memb), 0L))           # drop 0 if present
  if (!length(cl_ids)) cl_ids <- sort(unique(memb))   # fallback if no zeros
  M <- as.matrix(D_dx); diag(M) <- Inf
  Vn <- igraph::gorder(g_dx)
  el <- if (igraph::ecount(g_dx) > 0L) igraph::as_data_frame(g_dx, "edges") else NULL
  Wtot <- if (!is.null(el)) sum(el$weight) else NA_real_
  
  out_rows <- vector("list", length(cl_ids))
  top_list <- vector("list", length(cl_ids))
  
  for (t in seq_along(cl_ids)){
    cid <- cl_ids[t]
    idx <- which(memb == cid)
    if (!length(idx)) next
    w <- multU[idx]
    nV <- length(idx)
    mass <- sum(w)
    pct_mass <- mass / sum(multU)
    
    # induced subgraph + internal edges/weights
    g_c <- igraph::induced_subgraph(g_dx, vids = idx)
    E_c <- igraph::ecount(g_c)
    W_c <- if (E_c) sum(igraph::E(g_c)$weight) else 0
    
    # cut weight & conductance
    S <- idx; Tset <- setdiff(seq_len(Vn), S)
    cut_w <- if (!is.null(el) && length(S) && length(Tset)) {
      eid <- igraph::E(g_dx)[.inc(S) & .inc(Tset)]
      sum(igraph::E(g_dx)$weight[eid])
    } else 0
    volS <- sum(igraph::strength(g_dx, vids = S, weights = igraph::E(g_dx)$weight))
    conductance <- if (volS > 0) cut_w / volS else NA_real_
    
    density <- if (nV > 1) (2 * E_c) / (nV * (nV - 1)) else NA_real_
    intra <- if (nV > 1) mean(M[idx, idx][upper.tri(M[idx, idx])], na.rm = TRUE) else NA_real_
    
    # silhouette for this cluster (computed against all nonzero clusters)
    sil_mean <- sil_median <- NA_real_
    keep_idx <- which(memb != 0L)
    if (length(keep_idx) >= 3) {
      D_kept <- stats::as.dist(M[keep_idx, keep_idx, drop = FALSE])
      sil_all <- try(cluster::silhouette(as.integer(factor(memb[keep_idx])), D_kept), silent = TRUE)
      if (!inherits(sil_all, "try-error")) {
        s_vals <- sil_all[which(memb[keep_idx] == cid), "sil_width"]
        sil_mean <- mean(s_vals, na.rm = TRUE)
        sil_median <- stats::median(s_vals, na.rm = TRUE)
      }
    }
    
    # NODIAG prevalence/lift at participant-level (optional)
    nod_prev <- nod_lift <- NA_real_
    if (!is.null(DX_full) && !is.null(clusters_all) && "NODIAG" %in% names(DX_full)) {
      df <- merge(DX_full, clusters_all, by = "participant_id", all.x = TRUE)
      df$cluster[is.na(df$cluster)] <- 0L
      in_c <- df$cluster == cid
      prev_in <- mean(df$NODIAG[in_c] == 1L, na.rm = TRUE)
      prev_all <- mean(df$NODIAG == 1L, na.rm = TRUE)
      nod_prev <- prev_in
      nod_lift <- if (is.finite(prev_all) && prev_all > 0) prev_in / prev_all else NA_real_
    }
    
    out_rows[[t]] <- data.frame(
      cluster = cid,
      n_profiles = nV,
      weight = mass,
      pct_weight = pct_mass,
      E_internal = E_c,
      W_internal = W_c,
      cut_weight = cut_w,
      conductance = conductance,
      density = density,
      intra_dist_mean = intra,
      sil_mean = sil_mean,
      sil_median = sil_median,
      nodiag_prev = nod_prev,
      nodiag_lift = nod_lift,
      stringsAsFactors = FALSE
    )
    
    # Top enriched diagnoses (quick binomial/Fisher) — optional but handy
    if (!is.null(DX_full) && !is.null(clusters_all)) {
      cols <- setdiff(names(DX_full), c("participant_id","ANY_DX"))
      df <- merge(DX_full, clusters_all, by = "participant_id", all.x = TRUE)
      df$cluster[is.na(df$cluster)] <- 0L
      in_c <- df$cluster == cid; out_c <- !in_c
      rows <- lapply(cols, function(v){
        a <- sum(df[[v]][in_c]  == 1L, na.rm = TRUE)
        b <- sum(df[[v]][out_c] == 1L, na.rm = TRUE)
        c0 <- sum(df[[v]][in_c]  == 0L, na.rm = TRUE)
        d  <- sum(df[[v]][out_c] == 0L, na.rm = TRUE)
        n_in <- sum(in_c)
        if ((a + b) < min_total || a < min_in || b < min_out) return(NULL)
        prev_in  <- a / max(1, a + c0)
        prev_all <- (a + b) / max(1, a + b + c0 + d)
        lift <- if (prev_all > 0) prev_in / prev_all else NA_real_
        p <- try(
          if (b == 0L || c0 == 0L)
            stats::binom.test(a, n_in, p = prev_all, alternative = "greater")$p.value
          else
            stats::fisher.test(matrix(c(a, b, c0, d), nrow = 2, byrow = TRUE),
                               alternative = FISHER_ALT)$p.value,
          silent = TRUE
        )
        if (inherits(p, "try-error")) p <- NA_real_
        data.frame(dx = v, lift = lift, p = p, stringsAsFactors = FALSE)
      })
      tab <- do.call(rbind, Filter(Negate(is.null), rows))
      if (!is.null(tab) && nrow(tab)) {
        tab$q <- p.adjust(tab$p, "BH")
        tab <- tab[order(-tab$lift, tab$q), ]
        top_list[[t]] <- head(tab[, c("dx","lift","q")], top_dx)
      } else {
        top_list[[t]] <- data.frame()
      }
    }
  }
  
  diag_tbl <- do.call(rbind, out_rows)
  if (!is.null(diag_tbl) && nrow(diag_tbl)) diag_tbl$top_dx <- top_list
  diag_tbl[order(diag_tbl$cluster), ]
}

# -------------------------- Run & save diagnostics --------------------------
diag_tbl <- cluster_diagnostics(
  DXu_id = DXu_id, multU = multU, D_dx = D_dx, g_dx = g_dx,
  lv_membership = lv$membership,
  DX_full = DX, clusters_all = clusters_all,
  top_dx = 5, min_in = 3, min_out = 3, min_total = 6  # lighter guards for listing
)

# Summary CSV (without the nested 'top_dx' column)
readr::write_csv(dplyr::select(diag_tbl, -top_dx), "cluster_diagnostics_summary.csv")

# Per-cluster top-dx tables
dir.create("cluster_diagnostics", showWarnings = FALSE)
invisible(lapply(seq_len(nrow(diag_tbl)), function(i){
  cid <- diag_tbl$cluster[i]
  tab <- diag_tbl$top_dx[[i]]
  if (is.data.frame(tab) && nrow(tab))
    readr::write_csv(tab, file.path("cluster_diagnostics", sprintf("C%d_top_dx.csv", cid)))
}))

# Pretty printer for quick console skim
print_cluster_diagnostics <- function(diag_tbl, k = 3){
  for (i in seq_len(nrow(diag_tbl))){
    r <- diag_tbl[i,]
    cat(sprintf(
      "\nC%-3d | profiles=%-4d weight=%-5d (%.1f%%)  sil=%.2f  cond=%.2f  intra=%.3f",
      r$cluster, r$n_profiles, r$weight, 100*r$pct_weight, r$sil_mean, r$conductance, r$intra_dist_mean
    ))
    if (is.finite(r$nodiag_prev)) cat(sprintf("  NODIAG: %.1f%% (lift=%.2f)", 100*r$nodiag_prev, r$nodiag_lift))
    td <- r$top_dx[[1]]
    if (is.data.frame(td) && nrow(td)) {
      top_str <- paste(utils::head(sprintf("%s (×%.2f, q=%.3g)", td$dx, td$lift, td$q), k), collapse = " | ")
      cat("\n   Top dx:", top_str)
    }
    cat("\n")
  }
}

print_cluster_diagnostics(diag_tbl, k = 5)