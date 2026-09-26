# 다수준 모형 공통 함수 (mlm_gu.R, mlm_dong.R 에서 source)
suppressPackageStartupMessages({
  library(glmmTMB)
  library(data.table)
})

load_cohort <- function(path) {
  d <- fread(path, colClasses = list(character = intersect(
    c("orig_sgg", "dest_gu", "dest", "od"), names(fread(path, nrows = 0)))))
  d[, purpose := factor(purpose, levels = c("H", "W", "E"))]
  d[, age5 := factor(age5)]
  d[, sex := factor(sex)]
  d[, daytype := factor(daytype, levels = c("wd", "wk"))]
  tl <- sort(unique(d$time))
  d[, time := factor(time, levels = c(intersect(c("day", "t09_12"), tl), setdiff(tl, c("day", "t09_12"))))]
  d[, log_exp := log(exposure)]
  d
}

make_ctrl <- function() {
  glmmTMBControl(optCtrl = list(iter.max = 1e4, eval.max = 1e4),
                 parallel = max(1L, parallel::detectCores() - 1L))
}

fit_nb <- function(f, d, label, ctrl = make_ctrl()) {
  t0 <- Sys.time()
  m <- glmmTMB(f, data = d, family = nbinom2, control = ctrl)
  message(sprintf("[%s] %.1f분, AIC=%.1f, pdHess=%s", label,
                  as.numeric(difftime(Sys.time(), t0, units = "mins")), AIC(m), isTRUE(m$sdr$pdHess)))
  m
}

# 분산성분 / VPC (Nakagawa et al. 2017, NB2 잠재척도 관측수준 분산 log(1 + 1/λ + 1/θ))
vpc_table <- function(m, label, d) {
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

# 목적별 조건부 효과: 기준(H) 기울기 + 목적 상호작용 → 목적마다 IRR과 CI
purpose_slopes <- function(m, vars, label) {
  b <- fixef(m)$cond
  V <- vcov(m)$cond
  rbindlist(lapply(vars, function(v) rbindlist(lapply(c("H", "W", "E"), function(p) {
    terms <- v
    if (p != "H") {
      it <- intersect(c(paste0("purpose", p, ":", v), paste0(v, ":purpose", p)), names(b))
      terms <- c(v, it)
    }
    est <- sum(b[terms])
    se <- sqrt(sum(V[terms, terms]))
    data.table(model = label, var = v, purpose = p, coef = est, se = se, IRR = exp(est),
               lo = exp(est - 1.96 * se), hi = exp(est + 1.96 * se),
               p = 2 * pnorm(-abs(est / se)))
  }))))
}

model_comparison <- function(models) {
  rbindlist(lapply(names(models), function(n) {
    m <- models[[n]]
    data.table(model = n, logLik = as.numeric(logLik(m)), df = attr(logLik(m), "df"),
               AIC = AIC(m), BIC = BIC(m), pdHess = isTRUE(m$sdr$pdHess))
  }))
}

# 무선기울기 모형: unstructured 실패/비정칙이면 대각 공분산으로
fit_slopes <- function(f_us, f_diag, d, label) {
  m <- tryCatch(fit_nb(f_us, d, label), error = function(e) NULL)
  if (is.null(m) || !isTRUE(m$sdr$pdHess)) {
    message(label, ": unstructured 공분산 수렴 실패 → 대각 공분산으로 재적합")
    m <- fit_nb(f_diag, d, paste0(label, "-diag"))
  }
  m
}
