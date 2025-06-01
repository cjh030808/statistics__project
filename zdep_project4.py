import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Font settings for plots
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

# 모집단 파라미터 설정
n_samples = 1200  # 샘플 수 (1000개 이상)
mu1, mu2 = 50, 70  # 두 모집단의 평균
sigma = 10  # 공통 표준편차

# 시드 설정 (재현 가능한 결과를 위해)
np.random.seed(42)

# 두 개의 정규분포 모집단 생성
population1 = np.random.normal(mu1, sigma, n_samples)
population2 = np.random.normal(mu2, sigma, n_samples)

print(f"Population 1: Mean = {np.mean(population1):.2f}, Std = {np.std(population1):.2f}")
print(f"Population 2: Mean = {np.mean(population2):.2f}, Std = {np.std(population2):.2f}")
print(f"Sample size for each population: {n_samples}")

# 그래프 그리기
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 히스토그램 (실제 데이터)
axes[0, 0].hist(population1, bins=30, alpha=0.7, color='skyblue', 
                density=True, label=f'Population 1 (μ={mu1}, σ={sigma})')
axes[0, 0].hist(population2, bins=30, alpha=0.7, color='lightcoral', 
                density=True, label=f'Population 2 (μ={mu2}, σ={sigma})')
axes[0, 0].set_title('Histogram of Real Data')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Probability Density')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. 이론적 확률분포 곡선
x = np.linspace(10, 110, 1000)
pdf1 = stats.norm.pdf(x, mu1, sigma)
pdf2 = stats.norm.pdf(x, mu2, sigma)

axes[0, 1].plot(x, pdf1, 'b-', linewidth=2, label=f'Population 1 (μ={mu1}, σ={sigma})')
axes[0, 1].plot(x, pdf2, 'r-', linewidth=2, label=f'Population 2 (μ={mu2}, σ={sigma})')
axes[0, 1].set_title('Theoretical Probability Density Function')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Probability Density')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. 실제 데이터와 이론 분포 비교 (모집단 1)
axes[1, 0].hist(population1, bins=30, alpha=0.7, color='skyblue', 
                density=True, label='Real Data')
axes[1, 0].plot(x, pdf1, 'b-', linewidth=2, label='Theoretical Distribution')
axes[1, 0].set_title(f'Population 1 Comparison (μ={mu1}, σ={sigma})')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Probability Density')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. 실제 데이터와 이론 분포 비교 (모집단 2)
axes[1, 1].hist(population2, bins=30, alpha=0.7, color='lightcoral', 
                density=True, label='Real Data')
axes[1, 1].plot(x, pdf2, 'r-', linewidth=2, label='Theoretical Distribution')
axes[1, 1].set_title(f'Population 2 Comparison (μ={mu2}, σ={sigma})')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Probability Density')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional statistics
print("\n=== Descriptive Statistics ===")
print(f"Population 1:")
print(f"  Mean: {np.mean(population1):.3f}")
print(f"  Std Dev: {np.std(population1, ddof=1):.3f}")
print(f"  Min: {np.min(population1):.3f}")
print(f"  Max: {np.max(population1):.3f}")

print(f"\nPopulation 2:")
print(f"  Mean: {np.mean(population2):.3f}")
print(f"  Std Dev: {np.std(population2, ddof=1):.3f}")
print(f"  Min: {np.min(population2):.3f}")
print(f"  Max: {np.max(population2):.3f}")

# Box plot comparison
# plt.figure(figsize=(10, 6))
# data_to_plot = [population1, population2]
# labels = [f'Population 1\n(μ={mu1}, σ={sigma})', f'Population 2\n(μ={mu2}, σ={sigma})']

# plt.boxplot(data_to_plot, labels=labels)
# plt.title('Box Plot Comparison of Two Populations')
# plt.ylabel('Value')
# plt.grid(True, alpha=0.3)
# plt.show()

# # Q-Q plot for normality test
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# # Population 1 Q-Q plot
# stats.probplot(population1, dist="norm", plot=ax1)
# ax1.set_title('Population 1 Q-Q Plot')

# # Population 2 Q-Q plot
# stats.probplot(population2, dist="norm", plot=ax2)
# ax2.set_title('Population 2 Q-Q Plot')

# plt.tight_layout()
# plt.show()

# # Shapiro-Wilk normality test (using smaller sample for large datasets)
# sample_size_for_test = 100
# sample1 = np.random.choice(population1, sample_size_for_test)
# sample2 = np.random.choice(population2, sample_size_for_test)

# stat1, p_value1 = stats.shapiro(sample1)
# stat2, p_value2 = stats.shapiro(sample2)

# print(f"\n=== Shapiro-Wilk Normality Test (sample size: {sample_size_for_test}) ===")
# print(f"Population 1: Statistic = {stat1:.4f}, p-value = {p_value1:.4f}")
# print(f"Population 2: Statistic = {stat2:.4f}, p-value = {p_value2:.4f}")
# print("If p-value > 0.05, the data follows normal distribution.")