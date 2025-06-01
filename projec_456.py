import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, f

##################
# start: project3
np.random.seed(42)  #시드 설정으로 재현가능하게 

mean1 = 70      # population 1 - mean
mean2 = 50      # population 2 - mean
variance = 10   # same variance


##################
# start: histograms
# Make population 1, 2
population1 = np.random.normal(mean1, np.sqrt(variance), 2000)  
population2 = np.random.normal(mean2, np.sqrt(variance), 3000)

plt.figure(figsize=(10, 4))
plt.hist(population1, bins=100, density=True, alpha=0.7, label='Population 1', color="lightcoral")
plt.hist(population2, bins=100, density=True, alpha=0.7, label='Population 2', color="skyblue" )

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Histogram of Populations')
plt.legend()
# end: histograms
##################

##################
# start: Probability density function
x = np.linspace(min(population1.min(), population2.min()), max(population1.max(), population2.max()), 1000)
y1 = norm.pdf(x, mean1, np.sqrt(variance))
y2 = norm.pdf(x, mean2, np.sqrt(variance))

plt.figure(figsize=(10, 4))
plt.plot(x, y1, label='Population 1', color="lightcoral")
plt.plot(x, y2, label='Population 2', color="skyblue") 

plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.title('Probability Distribution')
plt.legend()
# end: Probability density function
##################

plt.show()  


##################
# start: project5
n1 = 81
n2 = 101
confidence_level = 0.95

# Sample extraction
sample1 = np.random.choice(population1, size=n1, replace=True)  # Randomly sample 81 in population1
sample2 = np.random.choice(population2, size=n2, replace=True)  # Randomly sample 101 in population2

# Sample mean difference  𝜇1 − 𝜇2
mean_diff = np.mean(sample1) - np.mean(sample2)
print(f"Mean Difference (Sample Mean 1 - Sample Mean 2): {mean_diff}\n")

# Sample variances
S1 = np.var(sample1, ddof=1)
S2 = np.var(sample2, ddof=1)
print(f"Sample variance1: {S1}")
print(f"Sample variance2: {S2}\n")

##################
# 1. Assuming we know the population variance (𝜎1 ,𝜎2) 
print("1. Assuming we know the population variance (𝜎1² , 𝜎2²) ")

std_diff_known_var = np.sqrt(variance / n1 + variance / n2)
z_score = norm.ppf(1 - (1 - confidence_level) / 2)
lower_bound_known_var = mean_diff - z_score * std_diff_known_var
upper_bound_known_var = mean_diff + z_score * std_diff_known_var

print(f"95% confidence interval for μ1 - μ2: P({lower_bound_known_var} < μ1 - μ2 < {upper_bound_known_var}) \n")

##################
# 2. Assuming we don’t know the population variance but we know both are same
print("2. Assuming we don’t know the population variance but we know both are same")

sp = np.sqrt(((n1 - 1) * S1 + (n1 - 1) * S2) / (n1 + n2 - 2))
degrees_of_freedom = n1 + n2 - 2
t_score = t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)
lower_bound_equal_unknown_var = mean_diff - t_score * sp * np.sqrt(1 / n1 + 1 / n2)
upper_bound_equal_unknown_var = mean_diff + t_score * sp * np.sqrt(1 / n1 + 1 / n2)

print(f"95% confidence interval for μ1 - μ2: P({lower_bound_equal_unknown_var} < μ1 - μ2 < {upper_bound_equal_unknown_var}) \n")

##################
# 3. Assuming we don’t know the population variance
print("3. Assuming we don’t know the population variance")

degrees_of_freedom = ((S1 / n1 + S2 / n2) ** 2) / ((S1**2 / (n1**2 * (n1 - 1))) + (S2**2 / (n2**2 * (n2 - 1))))
t_score = t.ppf(1 - (1 - confidence_level) / 2, degrees_of_freedom)
lower_bound_unknown_var = mean_diff - t_score * np.sqrt(S1 / n1 + S2 / n2)
upper_bound_unknown_var = mean_diff + t_score * np.sqrt(S1 / n1 + S2 / n2)

print(f"95% confidence interval for μ1 - μ2: P({lower_bound_unknown_var} < μ1 - μ2 < {upper_bound_unknown_var}) \n")
print("##################\n")


##################
# start: project6
ratio_var = S1 / S2

# 95% 신뢰구간 계산 (F-분포 사용)
alpha = 0.05
v_1 = n1 - 1
v_2 = n2 - 1

f_critical1 = f.ppf(alpha / 2, v_1, v_2)
f_critical2 = f.ppf(1 - alpha / 2, v_1, v_2)

ci_lower = ratio_var / f_critical2
ci_upper = ratio_var * f_critical2

# 결과 출력
print(f"Sample variance1: {S1}")
print(f"Sample variance2: {S2}")
print(f"95% confidence interval for σ1²/σ2²: P({ci_lower:.4f} < σ1²/σ2² < {ci_upper:.4f})")
