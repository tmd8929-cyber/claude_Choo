# 서울 유입 수도권 고령인구 — 출발 시군구 × 도착 자치구 교차분류 목적별 다수준 음이항 모형
#
# 수준: L1 코호트 셀  ⊂  L2 출발–도착 쌍(od)  ⊂  출발 시군구 × 도착 자치구 [교차분류]
# 모형 (NB2, offset = log(일수 × 시간))
#   M0  빈 모형: 출발지·도착지·쌍 수준 분산 분해
#   M1  + 개인·시간 주효과 (+ 평균 이동시간)
#   M2  + 목적 × (연령·성별·주중주말·시간대)
#   M3  + 목적별 무선기울기 (0 + purpose | 출발), (0 + purpose | 도착)
#   M4  + 출발지(o_)·도착 구(g_) 지역변수 + 목적 × g_ 교차수준 상호작용
#        도착 구는 25개뿐이므로 g_ 변수는 2~3개로 제한할 것
#
# 사용법: Rscript R/mlm_gu.R cohort_gu.csv out_dir "g_acc_hospital,g_acc_welfare" ["o_dens_hospital"]

args <- commandArgs(trailingOnly = TRUE)
script_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(script_dir, "common.R"))

cohort_path <- if (length(args) >= 1) args[1] else "data/processed/cohort_gu.csv"
out_dir <- if (length(args) >= 2) args[2] else "outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

d <- load_cohort(cohort_path)
gvars <- if (length(args) >= 3 && nzchar(args[3])) strsplit(args[3], ",")[[1]] else character(0)
ovars <- if (length(args) >= 4 && nzchar(args[4])) strsplit(args[4], ",")[[1]] else character(0)
stopifnot(all(c(gvars, ovars) %in% names(d)))
if (length(gvars) > 3) message("경고: 도착 구 25개에 g_ 변수 ", length(gvars), "개 → 과적합 위험")
has_travel <- "z_od_travel" %in% names(d)
cat(sprintf("cells=%d, zeros=%.1f%%, origins=%d, dests=%d\n", nrow(d), 100 * mean(d$y == 0),
            uniqueN(d$orig_sgg), uniqueN(d$dest_gu)))

re_base <- "+ (1 | orig_sgg) + (1 | dest_gu) + (1 | od)"
tr <- if (has_travel) "+ z_od_travel" else ""
f0 <- as.formula(paste("y ~ 1 + offset(log_exp)", re_base))
f1 <- as.formula(paste("y ~ offset(log_exp) + purpose + age5 + sex + daytype + time", tr, re_base))
fixed2 <- paste("offset(log_exp) + purpose * (age5 + sex + daytype + time)", tr)
f2 <- as.formula(paste("y ~", fixed2, re_base))
re_us <- "+ (0 + purpose | orig_sgg) + (0 + purpose | dest_gu) + (1 | od)"
re_diag <- "+ diag(0 + purpose | orig_sgg) + diag(0 + purpose | dest_gu) + (1 | od)"

models <- list()
models$M0 <- fit_nb(f0, d, "M0")
models$M1 <- fit_nb(f1, d, "M1")
models$M2 <- fit_nb(f2, d, "M2")
models$M3 <- fit_slopes(as.formula(paste("y ~", fixed2, re_us)),
                        as.formula(paste("y ~", fixed2, re_diag)), d, "M3")
if (length(c(gvars, ovars))) {
  re3 <- if (grepl("diag", deparse1(formula(models$M3)))) re_diag else re_us
  fixed4 <- paste(fixed2, "+", paste(c(gvars, ovars), collapse = " + "),
                  if (length(gvars)) paste("+", paste0("purpose:", gvars, collapse = " + ")))
  models$M4 <- fit_nb(as.formula(paste("y ~", fixed4, re3)), d, "M4")
}

cmp <- model_comparison(models)
fwrite(cmp, file.path(out_dir, "gu_model_comparison.csv"))
print(cmp)
vpc <- rbindlist(lapply(names(models), function(n) vpc_table(models[[n]], n, d)))
fwrite(vpc, file.path(out_dir, "gu_variance_components.csv"))
print(vpc[model %in% c("M0", "M3")], digits = 3)
irr <- rbindlist(lapply(names(models)[-1], function(n) irr_table(models[[n]], n)))
fwrite(irr, file.path(out_dir, "gu_fixed_effects_irr.csv"))
if (!is.null(models$M4) && length(gvars)) {
  fwrite(purpose_slopes(models$M4, gvars, "M4"), file.path(out_dir, "gu_purpose_specific_regional_irr.csv"))
}
re <- ranef(models$M3)$cond
for (g in names(re)) fwrite(as.data.table(re[[g]], keep.rownames = g),
                            file.path(out_dir, sprintf("gu_random_effects_%s.csv", g)))
saveRDS(models, file.path(out_dir, "gu_models.rds"))
cat("done →", out_dir, "\n")
