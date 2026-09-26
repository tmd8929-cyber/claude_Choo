# 서울 유입 수도권 고령인구 생활이동 — 목적별 교차분류 다수준 음이항 모형
#
# 수준 구조
#   L1 코호트 셀 (목적 H/W/E × 5세 연령 × 성별 × 주중/주말 × 낮/밤)
#   L2 출발지-도착지 쌍(od)  ⊂  출발 시군구(orig_sgg) × 도착 자치구(dest_gu)  [교차분류]
#
# 모형 (NB2, log link, offset = log(일수 × 시간))
#   M0  빈 모형: 수준별 분산 분해(VPC)
#   M1  + 개인·시간 변수 주효과
#   M2  + 목적 × (연령·성별·주중주말·낮밤) 상호작용  ← 이전 연구의 W/E 분리모형을 하나로 통합
#   M3  + 목적별 무선기울기 (0 + purpose | 도착구), (0 + purpose | 출발지)
#         → "목적마다 어느 지역 수준이 유입 편차를 만드는가"
#   M4  + 출발지(o_)·도착지(d_) 지역변수와 목적 × 도착지 변수 교차수준 상호작용 (지역변수 있을 때)
#
# 사용법: Rscript R/mlm_glmmtmb.R [cohort.csv] [out_dir] [d_변수,d_변수,...]

suppressPackageStartupMessages({
  library(glmmTMB)
  library(data.table)
})

args <- commandArgs(trailingOnly = TRUE)
cohort_path <- if (length(args) >= 1) args[1] else "data/processed/cohort.csv"
out_dir <- if (length(args) >= 2) args[2] else "outputs"
dvars_arg <- if (length(args) >= 3) strsplit(args[3], ",")[[1]] else NULL
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

d <- fread(cohort_path, colClasses = list(character = c("orig_sgg", "dest_gu", "od")))
d[, purpose := factor(purpose, levels = c("H", "W", "E"))]
d[, age5 := factor(age5)]
d[, sex := factor(sex)]
d[, daytype := factor(daytype, levels = c("wd", "wk"))]
d[, daynight := factor(daynight, levels = c("day", "night"))]
d[, log_exp := log(exposure)]
cat(sprintf("cells=%d, zeros=%.1f%%, origins=%d, dests=%d\n", nrow(d), 100 * mean(d$y == 0),
            uniqueN(d$orig_sgg), uniqueN(d$dest_gu)))

ctrl <- glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4),
                      parallel = max(1L, parallel::detectCores() - 1L))
fit <- function(f, label) {
  t0 <- Sys.time()
  m <- glmmTMB(f, data = d, family = nbinom2, control = ctrl)
  message(sprintf("[%s] %.1f분, AIC=%.1f", label,
                  as.numeric(difftime(Sys.time(), t0, units = "mins")), AIC(m)))
  if (!isTRUE(m$sdr$pdHess)) warning("Hessian not positive definite: ", deparse(f)[1])
  m
}

# ---- 분산성분 / VPC (Nakagawa et al. 2017, NB2 잠재척도) -------------------
vpc_table <- function(m, label) {
  vc <- VarCorr(m)$cond
  rows <- rbindlist(lapply(names(vc), function(g) {
    v <- diag(vc[[g]])
    data.table(model = label, level = g, term = names(v), var = as.numeric(v))
  }))
  theta <- sigma(m)
  lambda <- exp(fixef(m)$cond[1] + mean(d$log_exp) + sum(rows$var) / 2)
  dist_var <- log(1 + 1 / lambda + 1 / theta)
  rows[, `:=`(sd = sqrt(var), VPC = var / (sum(var) + dist_var), theta = theta)]
  rows
}

irr_table <- function(m, label) {
  s <- summary(m)$coefficients$cond
  data.table(model = label, term = rownames(s), coef = s[, 1], se = s[, 2],
             IRR = exp(s[, 1]), lo = exp(s[, 1] - 1.96 * s[, 2]),
             hi = exp(s[, 1] + 1.96 * s[, 2]), p = s[, 4])
}

re_base <- "(1 | orig_sgg) + (1 | dest_gu) + (1 | od)"
f0 <- as.formula(paste("y ~ 1 + offset(log_exp) +", re_base))
f1 <- update(f0, . ~ . + purpose + age5 + sex + daytype + daynight)
f2 <- update(f1, . ~ . + purpose:(age5 + sex + daytype + daynight))
f3 <- as.formula(paste(
  "y ~ offset(log_exp) + purpose * (age5 + sex + daytype + daynight) +",
  "(0 + purpose | orig_sgg) + (0 + purpose | dest_gu) + (1 | od)"))

models <- list()
models$M0 <- fit(f0, "M0")
models$M1 <- fit(f1, "M1")
models$M2 <- fit(f2, "M2")
models$M3 <- tryCatch(fit(f3, "M3"), error = function(e) {
  message("M3 (unstructured) 실패 → 대각 공분산으로 재적합: ", conditionMessage(e))
  fit(update(f3, . ~ . - (0 + purpose | orig_sgg) - (0 + purpose | dest_gu) +
               diag(0 + purpose | orig_sgg) + diag(0 + purpose | dest_gu)), "M3-diag")
})

# ---- M4: 지역변수 (출발/도착 분리, 도착 25개 구이므로 변수 수 제한) ------------
dcols <- grep("^d_", names(d), value = TRUE)
ocols <- grep("^o_", names(d), value = TRUE)
if (length(dcols)) {
  if (!is.null(dvars_arg)) {
    dcols <- intersect(dcols, dvars_arg)
    ocols <- intersect(ocols, sub("^d_", "o_", dvars_arg))
  }
  if (length(dcols) > 4) {
    message("도착지 변수 ", length(dcols), "개: 도착 단위가 25개뿐이라 과적합 위험. ",
            "세 번째 인자로 3~4개를 지정하세요.")
  }
  f4 <- as.formula(paste(
    deparse(f3, width.cutoff = 500L), "+",
    paste(c(dcols, ocols), collapse = " + "), "+",
    paste0("purpose:", dcols, collapse = " + ")))
  models$M4 <- fit(f4, "M4")
}

# ---- 결과 저장 ----------------------------------------------------------------
cmp <- rbindlist(lapply(names(models), function(n) {
  m <- models[[n]]
  data.table(model = n, logLik = as.numeric(logLik(m)), df = attr(logLik(m), "df"),
             AIC = AIC(m), BIC = BIC(m), pdHess = isTRUE(m$sdr$pdHess))
}))
fwrite(cmp, file.path(out_dir, "mlm_model_comparison.csv"))
print(cmp)

vpc <- rbindlist(lapply(names(models), function(n) vpc_table(models[[n]], n)))
fwrite(vpc, file.path(out_dir, "mlm_variance_components.csv"))
print(vpc[model %in% c("M0", "M3")])

irr <- rbindlist(lapply(names(models)[-1], function(n) irr_table(models[[n]], n)))
fwrite(irr, file.path(out_dir, "mlm_fixed_effects_irr.csv"))

# 목적별 자치구·출발지 무선효과 (M3): exp(값) = 평균 대비 유입 배수
re3 <- ranef(models$M3)$cond
for (g in c("dest_gu", "orig_sgg")) {
  r <- as.data.table(re3[[g]], keep.rownames = g)
  fwrite(r, file.path(out_dir, sprintf("mlm_random_effects_%s.csv", g)))
}
cor_dest <- attr(VarCorr(models$M3)$cond$dest_gu, "correlation")
if (!is.null(cor_dest)) fwrite(as.data.table(cor_dest, keep.rownames = "purpose"),
                               file.path(out_dir, "mlm_dest_purpose_correlation.csv"))

saveRDS(models, file.path(out_dir, "mlm_models.rds"))
cat("done →", out_dir, "\n")
