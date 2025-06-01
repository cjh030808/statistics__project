import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

pd.options.display.float_format = '{:,.0f}'.format

df = pd.read_csv(r".\korea_rental_housing.csv", encoding="cp949")
#다운로드 한 데이터를 절대경로를 이용해 가져옴

df_seoul = df[df['광역시도'] == '서울특별시']      #전국의 지역 중 서울특별시

sample_sizes = [10, 30, 100]   # 요구된 샘플 크기들
population = df_seoul['월임대료'].dropna().astype(float).values  # 모집단 (서울 월임대료)
alpha = 0.01  # 99% 신뢰수준 → 유의수준 1%

for size in sample_sizes:
    np.random.seed(42)  # 재현 가능성 위해 고정
    sample = np.random.choice(population, size=size, replace=False)  # 랜덤 샘플링

    sample_mean = np.mean(sample)
    sample_var = np.var(sample, ddof=1)  # 표본분산 (n-1로 나눔)
    sample_std = np.sqrt(sample_var)

    # 모평균에 대한 99% 신뢰구간
    t_crit = stats.t.ppf(1 - alpha / 2, df=size - 1)  # t 임계값
    mean_margin = t_crit * (sample_std / np.sqrt(size))
    ci_mean = (sample_mean - mean_margin, sample_mean + mean_margin)

    # 모분산에 대한 99% 신뢰구간 (chi-squared 분포 사용)
    chi2_lower = stats.chi2.ppf(alpha / 2, df=size - 1)
    chi2_upper = stats.chi2.ppf(1 - alpha / 2, df=size - 1)
    ci_var = ((size - 1) * sample_var / chi2_upper, (size - 1) * sample_var / chi2_lower)

    # 향후 관측값 하나에 대한 99% 예측구간
    pred_margin = t_crit * sample_std * np.sqrt(1 + 1 / size)
    prediction_interval = (sample_mean - pred_margin, sample_mean + pred_margin)

    # 실제 관측값 하나와 비교 (무작위로 하나 선택)
    actual_value = np.random.choice(population, 1)[0]
    in_interval = prediction_interval[0] <= actual_value <= prediction_interval[1]

    # 소수점 2자리까지 반올림해서 출력
    print(f"\n== 샘플 크기: {size} ==")
    print(f"표본평균 (X̄): {sample_mean:,.2f}")
    print(f"표본분산 (S²): {sample_var:,.2f}")
    print(f"99% 신뢰구간 (모평균): ({ci_mean[0]:,.2f}, {ci_mean[1]:,.2f})")
    print(f"99% 신뢰구간 (모분산): ({ci_var[0]:,.2f}, {ci_var[1]:,.2f})")
    print(f"99% 예측구간 (향후 관측값): ({prediction_interval[0]:,.2f}, {prediction_interval[1]:,.2f})")
    print(f"실제 관측값: {actual_value:,.2f} → 예측구간 내 포함 여부: {in_interval}")