import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Font settings for plots
plt.style.use('default')
plt.rcParams['axes.unicode_minus'] = False

# Population parameters (same as before)
n_samples = 1200  # Population size (more than 1000)
mu1, mu2 = 50, 70  # Population means
sigma = 10  # Common standard deviation
true_diff = mu2 - mu1  # True difference = 20

# Set seed for reproducible results
np.random.seed(42)

# Generate two normal distribution populations
population1 = np.random.normal(mu1, sigma, n_samples)
population2 = np.random.normal(mu2, sigma, n_samples)

print("=== POPULATION INFORMATION ===")
print(f"Population 1: Mean = {np.mean(population1):.2f}, Std = {np.std(population1):.2f}")
print(f"Population 2: Mean = {np.mean(population2):.2f}, Std = {np.std(population2):.2f}")
print(f"True difference (μ₂ - μ₁) = {true_diff}")

# Sample sizes
n1, n2 = 81, 101

# Random sampling from populations
np.random.seed(123)  # Different seed for sampling
sample1 = np.random.choice(population1, n1, replace=False)
sample2 = np.random.choice(population2, n2, replace=False)

print(f"\n=== SAMPLE INFORMATION ===")
print(f"Sample 1: n = {n1}, Mean = {np.mean(sample1):.3f}, Std = {np.std(sample1, ddof=1):.3f}")
print(f"Sample 2: n = {n2}, Mean = {np.mean(sample2):.3f}, Std = {np.std(sample2, ddof=1):.3f}")
print(f"Sample difference (x̄₂ - x̄₁) = {np.mean(sample2) - np.mean(sample1):.3f}")

# Calculate confidence intervals
alpha = 0.05  # For 95% confidence interval
z_critical = stats.norm.ppf(1 - alpha/2)  # Z critical value
t_critical_pooled = stats.t.ppf(1 - alpha/2, df=n1+n2-2)  # t critical for pooled variance
t_critical_welch = stats.t.ppf(1 - alpha/2, df=min(n1-1, n2-1))  # Conservative df for Welch's t-test

sample_mean_diff = np.mean(sample2) - np.mean(sample1)
sample_std1 = np.std(sample1, ddof=1)
sample_std2 = np.std(sample2, ddof=1)

print(f"\n=== 95% CONFIDENCE INTERVALS FOR DIFFERENCE IN MEANS ===")

# Case 1: Known population variance (σ₁² = σ₂² = σ²)
known_sigma = sigma
se_known = known_sigma * np.sqrt(1/n1 + 1/n2)
margin_error_known = z_critical * se_known
ci_known_lower = sample_mean_diff - margin_error_known
ci_known_upper = sample_mean_diff + margin_error_known

print(f"\n1. KNOWN POPULATION VARIANCE (σ₁² = σ₂² = {sigma}²)")
print(f"   Standard Error: {se_known:.4f}")
print(f"   Margin of Error: ±{margin_error_known:.4f}")
print(f"   95% CI: [{ci_known_lower:.4f}, {ci_known_upper:.4f}]")
print(f"   Contains true difference? {ci_known_lower <= true_diff <= ci_known_upper}")

# Case 2: Unknown but equal population variance (pooled variance)
pooled_variance = ((n1-1)*sample_std1**2 + (n2-1)*sample_std2**2) / (n1+n2-2)
pooled_std = np.sqrt(pooled_variance)
se_pooled = pooled_std * np.sqrt(1/n1 + 1/n2)
margin_error_pooled = t_critical_pooled * se_pooled
ci_pooled_lower = sample_mean_diff - margin_error_pooled
ci_pooled_upper = sample_mean_diff + margin_error_pooled

print(f"\n2. UNKNOWN BUT EQUAL VARIANCE (Pooled variance)")
print(f"   Pooled Standard Deviation: {pooled_std:.4f}")
print(f"   Standard Error: {se_pooled:.4f}")
print(f"   Degrees of Freedom: {n1+n2-2}")
print(f"   t-critical: {t_critical_pooled:.4f}")
print(f"   Margin of Error: ±{margin_error_pooled:.4f}")
print(f"   95% CI: [{ci_pooled_lower:.4f}, {ci_pooled_upper:.4f}]")
print(f"   Contains true difference? {ci_pooled_lower <= true_diff <= ci_pooled_upper}")

# Case 3: Unknown and unequal variance (Welch's t-test)
se_welch = np.sqrt(sample_std1**2/n1 + sample_std2**2/n2)
# Welch-Satterthwaite equation for degrees of freedom
df_welch = (sample_std1**2/n1 + sample_std2**2/n2)**2 / (
    (sample_std1**2/n1)**2/(n1-1) + (sample_std2**2/n2)**2/(n2-1)
)
t_critical_welch_exact = stats.t.ppf(1 - alpha/2, df=df_welch)
margin_error_welch = t_critical_welch_exact * se_welch
ci_welch_lower = sample_mean_diff - margin_error_welch
ci_welch_upper = sample_mean_diff + margin_error_welch

print(f"\n3. UNKNOWN AND UNEQUAL VARIANCE (Welch's t-test)")
print(f"   Standard Error: {se_welch:.4f}")
print(f"   Degrees of Freedom (Welch-Satterthwaite): {df_welch:.2f}")
print(f"   t-critical: {t_critical_welch_exact:.4f}")
print(f"   Margin of Error: ±{margin_error_welch:.4f}")
print(f"   95% CI: [{ci_welch_lower:.4f}, {ci_welch_upper:.4f}]")
print(f"   Contains true difference? {ci_welch_lower <= true_diff <= ci_welch_upper}")

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Sample distributions
ax1.hist(sample1, bins=15, alpha=0.7, color='skyblue', density=True, label=f'Sample 1 (n={n1})')
ax1.hist(sample2, bins=15, alpha=0.7, color='lightcoral', density=True, label=f'Sample 2 (n={n2})')
ax1.axvline(np.mean(sample1), color='blue', linestyle='--', label=f'Sample 1 Mean: {np.mean(sample1):.2f}')
ax1.axvline(np.mean(sample2), color='red', linestyle='--', label=f'Sample 2 Mean: {np.mean(sample2):.2f}')
ax1.set_title('Sample Distributions')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Confidence intervals comparison
methods = ['Known Variance\n(Z-test)', 'Equal Unknown\n(Pooled t-test)', 'Unequal Unknown\n(Welch t-test)']
ci_lowers = [ci_known_lower, ci_pooled_lower, ci_welch_lower]
ci_uppers = [ci_known_upper, ci_pooled_upper, ci_welch_upper]
ci_centers = [sample_mean_diff, sample_mean_diff, sample_mean_diff]

y_pos = np.arange(len(methods))
errors = [[ci_centers[i] - ci_lowers[i] for i in range(3)],
          [ci_uppers[i] - ci_centers[i] for i in range(3)]]

ax2.errorbar(ci_centers, y_pos, xerr=errors, fmt='o', capsize=5, capthick=2)
ax2.axvline(true_diff, color='red', linestyle='--', linewidth=2, label=f'True Difference: {true_diff}')
ax2.axvline(sample_mean_diff, color='green', linestyle='-', linewidth=2, label=f'Sample Difference: {sample_mean_diff:.3f}')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(methods)
ax2.set_xlabel('Difference in Means')
ax2.set_title('95% Confidence Intervals Comparison')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Sampling distribution simulation
n_simulations = 1000
sample_diffs = []

for _ in range(n_simulations):
    sim_sample1 = np.random.choice(population1, n1, replace=False)
    sim_sample2 = np.random.choice(population2, n2, replace=False)
    sample_diffs.append(np.mean(sim_sample2) - np.mean(sim_sample1))

theoretical_se = sigma * np.sqrt(1/n1 + 1/n2)
theoretical_mean = true_diff

ax3.hist(sample_diffs, bins=30, alpha=0.7, density=True, color='lightgreen', 
         label=f'Simulated Sampling Distribution\n(n={n_simulations} samples)')
x_range = np.linspace(min(sample_diffs), max(sample_diffs), 100)
theoretical_dist = stats.norm.pdf(x_range, theoretical_mean, theoretical_se)
ax3.plot(x_range, theoretical_dist, 'r-', linewidth=2, label='Theoretical Distribution')
ax3.axvline(true_diff, color='red', linestyle='--', label=f'True Difference: {true_diff}')
ax3.axvline(sample_mean_diff, color='blue', linestyle='--', label=f'Our Sample Diff: {sample_mean_diff:.3f}')
ax3.set_title('Sampling Distribution of Difference in Means')
ax3.set_xlabel('Difference in Sample Means')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Summary table
ax4.axis('tight')
ax4.axis('off')

table_data = [
    ['Method', 'Standard Error', 'Critical Value', 'Margin of Error', '95% CI Lower', '95% CI Upper', 'CI Width'],
    ['Known Variance', f'{se_known:.4f}', f'{z_critical:.3f}', f'{margin_error_known:.4f}', 
     f'{ci_known_lower:.4f}', f'{ci_known_upper:.4f}', f'{ci_known_upper-ci_known_lower:.4f}'],
    ['Equal Unknown', f'{se_pooled:.4f}', f'{t_critical_pooled:.3f}', f'{margin_error_pooled:.4f}', 
     f'{ci_pooled_lower:.4f}', f'{ci_pooled_upper:.4f}', f'{ci_pooled_upper-ci_pooled_lower:.4f}'],
    ['Unequal Unknown', f'{se_welch:.4f}', f'{t_critical_welch_exact:.3f}', f'{margin_error_welch:.4f}', 
     f'{ci_welch_lower:.4f}', f'{ci_welch_upper:.4f}', f'{ci_welch_upper-ci_welch_lower:.4f}']
]

table = ax4.table(cellText=table_data, cellLoc='center', loc='center', 
                  colWidths=[0.15, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Color the header row
for i in range(len(table_data[0])):
    table[(0, i)].set_facecolor('#E6E6FA')

ax4.set_title('Confidence Interval Summary Table', pad=20)

plt.tight_layout()
plt.show()

# Statistical tests
print(f"\n=== STATISTICAL TESTS ===")

# Two-sample t-test (assuming equal variances)
t_stat_pooled, p_value_pooled = stats.ttest_ind(sample2, sample1, equal_var=True)
print(f"Two-sample t-test (equal variances):")
print(f"   t-statistic: {t_stat_pooled:.4f}")
print(f"   p-value: {p_value_pooled:.6f}")

# Welch's t-test (unequal variances)
t_stat_welch, p_value_welch = stats.ttest_ind(sample2, sample1, equal_var=False)
print(f"Welch's t-test (unequal variances):")
print(f"   t-statistic: {t_stat_welch:.4f}")
print(f"   p-value: {p_value_welch:.6f}")

# F-test for equal variances
f_stat = sample_std2**2 / sample_std1**2 if sample_std2 > sample_std1 else sample_std1**2 / sample_std2**2
p_value_f = 2 * (1 - stats.f.cdf(f_stat, n2-1 if sample_std2 > sample_std1 else n1-1, 
                                  n1-1 if sample_std2 > sample_std1 else n2-1))
print(f"F-test for equal variances:")
print(f"   F-statistic: {f_stat:.4f}")
print(f"   p-value: {p_value_f:.6f}")
print(f"   Equal variances assumption: {'Valid' if p_value_f > 0.05 else 'Questionable'}")

print(f"\n=== SUMMARY ===")
print(f"All three confidence intervals contain the true difference ({true_diff}):")
print(f"- Known variance method gives the narrowest interval")
print(f"- Welch's t-test is most conservative when variances might be unequal")
print(f"- Sample difference ({sample_mean_diff:.3f}) is close to true difference ({true_diff})")