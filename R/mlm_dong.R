# 서울 유입 수도권 고령인구 — 도착 행정동 단위 목적별 다수준 음이항 모형
#
# 수준: L1 코호트 셀(목적 × 5세 연령 × 성별 × 주중/주말 × 시간대)
#       L2 도착 행정동(427)  ⊂  L3 도착 자치구(25)     [출발지는 합산]
# 모형 (NB2, offset = log(일수 × 시간))
#   D0  빈 모형: 동·구 수준 분산 분해
#   D1  + 개인·시간 주효과 (+ 평균 이동시간)
#   D2  + 목적 × (연령·성별·주중주말·시간대)
#   D3  + 목적별 무선기울기 (0 + purpose | 구), (0 + purpose | 동)
#   D4  + 동 지역변수(d_) 주효과 + 목적 × d_ 교차수준 상호작용 (+ 선택: 구 지역변수 g_)
#
# 사용법: Rscript R/mlm_dong.R cohort_dong.csv out_dir "d_acc_hospital,d_acc_welfare" ["g_elderly_share"]
#         d_ 변수를 비우면 d_acc_* 와 d_dist_subway, d_elderly_share 를 모두 사용

args <- commandArgs(trailingOnly = TRUE)
script_dir <- dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)))
source(file.path(script_dir, "common.R"))

cohort_path <- if (length(args) >= 1) args[1] else "data/processed/cohort_dong.csv"
out_dir <- if (length(args) >= 2) args[2] else "outputs"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

d <- load_cohort(cohort_path)
all_d <- grep("^d_", names(d), value = TRUE)
dvars <- if (length(args) >= 3 && nzchar(args[3])) strsplit(args[3], ",")[[1]] else
  grep("^d_(acc_|dist_subway|elderly_share)", all_d, value = TRUE)
gvars <- if (length(args) >= 4 && nzchar(args[4])) strsplit(args[4], ",")[[1]] else character(0)
stopifnot(all(c(dvars, gvars) %in% names(d)))
has_travel <- "z_od_travel" %in% names(d)
cat(sprintf("cells=%d, zeros=%.1f%%, dong=%d, gu=%d, d_vars=%s\n", nrow(d), 100 * mean(d$y == 0),
            uniqueN(d$dest), uniqueN(d$dest_gu), paste(dvars, collapse = ",")))

indiv <- "purpose + age5 + sex + daytype + time"
f0 <- y ~ 1 + offset(log_exp) + (1 | dest_gu) + (1 | dest)
f1 <- as.formula(paste("y ~ offset(log_exp) +", indiv, if (has_travel) "+ z_od_travel",
                       "+ (1 | dest_gu) + (1 | dest)"))
fixed2 <- paste("offset(log_exp) + purpose * (age5 + sex + daytype + time)",
                if (has_travel) "+ z_od_travel")
f2 <- as.formula(paste("y ~", fixed2, "+ (1 | dest_gu) + (1 | dest)"))
re_us <- "+ (0 + purpose | dest_gu) + (0 + purpose | dest)"
re_diag <- "+ diag(0 + purpose | dest_gu) + diag(0 + purpose | dest)"

models <- list()
models$D0 <- fit_nb(f0, d, "D0")
models$D1 <- fit_nb(f1, d, "D1")
models$D2 <- fit_nb(f2, d, "D2")
models$D3 <- fit_slopes(as.formula(paste("y ~", fixed2, re_us)),
                        as.formula(paste("y ~", fixed2, re_diag)), d, "D3")
if (length(dvars)) {
  re3 <- if (grepl("diag", deparse1(formula(models$D3)))) re_diag else re_us
  fixed4 <- paste(fixed2, "+", paste(c(dvars, gvars), collapse = " + "), "+",
                  paste0("purpose:", dvars, collapse = " + "))
  models$D4 <- fit_nb(as.formula(paste("y ~", fixed4, re3)), d, "D4")
}

cmp <- model_comparison(models)
fwrite(cmp, file.path(out_dir, "dong_model_comparison.csv"))
print(cmp)

vpc <- rbindlist(lapply(names(models), function(n) vpc_table(models[[n]], n, d)))
fwrite(vpc, file.path(out_dir, "dong_variance_components.csv"))
print(vpc[model %in% c("D0", "D3")], digits = 3)

irr <- rbindlist(lapply(names(models)[-1], function(n) irr_table(models[[n]], n)))
fwrite(irr, file.path(out_dir, "dong_fixed_effects_irr.csv"))

if (!is.null(models$D4)) {
  ps <- purpose_slopes(models$D4, dvars, "D4")
  fwrite(ps, file.path(out_dir, "dong_purpose_specific_regional_irr.csv"))
  print(ps[, .(var, purpose, IRR = round(IRR, 3), lo = round(lo, 3), hi = round(hi, 3), p = signif(p, 2))])
}

re <- ranef(models$D3)$cond
for (g in names(re)) fwrite(as.data.table(re[[g]], keep.rownames = g),
                            file.path(out_dir, sprintf("dong_random_effects_%s.csv", g)))
saveRDS(models, file.path(out_dir, "dong_models.rds"))
cat("done →", out_dir, "\n")
