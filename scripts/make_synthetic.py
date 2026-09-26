"""파이프라인 검증용 합성 생활이동 CSV 생성 (원자료 형식 모사, CP949)."""
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
out = Path(__file__).resolve().parents[1] / "data" / "synthetic" / "synthetic_sample.csv"
out.parent.mkdir(parents=True, exist_ok=True)

seoul_gu = [f"11{i:03d}" for i in range(110, 360, 10)]  # 25개 자치구
seoul_dong = [f"{g}{j:03d}" for g in seoul_gu for j in range(510, 510 + 10 * 8, 10)]
outside = [f"41{i:03d}" for i in range(110, 420, 10)] + [f"28{i:03d}" for i in range(110, 200, 10)]
outside_dong = [f"{s}{j:03d}" for s in outside for j in (510, 520, 530)]
ages = list(range(0, 90, 5))
types = ["HW", "HE", "WH", "EH", "WE", "EW", "EE", "WW"]

gu_eff = dict(zip(seoul_gu, rng.normal(0, 0.8, len(seoul_gu))))
dong_eff = dict(zip(seoul_dong, rng.normal(0, 0.5, len(seoul_dong))))
orig_eff = dict(zip(outside, rng.normal(0, 1.0, len(outside))))

n = 300_000
o = rng.choice(outside_dong + seoul_dong, n)
d = rng.choice(seoul_dong + outside_dong[:20], n)
lam = np.exp(
    1.0
    + np.array([gu_eff.get(x[:5], 0) + dong_eff.get(x, 0) for x in d])
    + np.array([orig_eff.get(x[:5], 0) for x in o])
)
flow = rng.poisson(lam).astype(float)
flow_s = np.where(flow < 3, "*", flow.astype(int).astype(str))
dates = pd.date_range("2026-08-01", "2026-08-31").strftime("%Y%m%d")
df = pd.DataFrame({
    "대상연월": rng.choice(dates, n),
    "요일": rng.choice(list("월화수목금토일"), n),
    "도착시간": rng.integers(0, 24, n),
    "출발 행정동 코드": o,
    "도착 행정동 코드": d,
    "성별": rng.choice(["M", "F"], n),
    "나이": rng.choice(ages, n),
    "이동유형": rng.choice(types, n),
    "평균 이동 시간(분)": rng.integers(10, 120, n),
    "이동인구(합)": flow_s,
})
df.to_csv(out, index=False, encoding="cp949")
print(f"wrote {out} ({len(df):,} rows)")
