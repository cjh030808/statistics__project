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
sigma1, sigma2 = 10, 10  # Population standard deviations (equal in this case)
true_variance_ratio = (sigma1**2) / (sigma2**2)  # True ratio = 1.0

# Set seed for reproducible results
np.random.seed(42)

# Generate two normal distribution populations
population1 = np.random.normal(mu1, sigma1, n_samples)
population2 = np.random.normal(mu2, sigma2, n_samples)

print("=== POPULATION INFORMATION ===")
print(f"Population 1: Mean = {np.mean(population1):.2f}, Std = {np.std(population1):.2f}")
print(f"Population 2: Mean = {np.mean(population2):.2f}, Std = {np.std(population2):.2f}")
print(f"True variance ratio (σ₁²/σ₂²) = {sigma1**2}/{sigma2**2} = {true_variance_ratio:.3f}")

# Sample sizes
n1, n2 = 81, 101

# Random sampling from populations
np.random.seed(123)  # Different seed for sampling
sample1 = np.random.choice(population1, n1, replace=False)
sample2 = np.random.choice(population2, n2, replace=False)

# Calculate sample statistics
sample_mean1 = np.mean(sample1)
sample_mean2 = np.mean(sample2)
sample_var1 = np.var(sample1, ddof=1)  # Sample variance (unbiased)
sample_var2 = np.var(sample2, ddof=1)  # Sample variance (unbiased)
sample_std1 = np.sqrt(sample_var1)
sample_std2 = np.sqrt(sample_var2)

print(f"\n=== SAMPLE INFORMATION ===")
print(f"Sample 1: n = {n1}")
print(f"   Mean = {sample_mean1:.3f}")
print(f"   Sample Variance (s₁²) = {sample_var1:.3f}")
print(f"   Sample Std Dev (s₁) = {sample_std1:.3f}")

print(f"Sample 2: n = {n2}")
print(f"   Mean = {sample_mean2:.3f}")
print(f"   Sample Variance (s₂²) = {sample_var2:.3f}")
print(f"   Sample Std Dev (s₂) = {sample_std2:.3f}")

# Calculate sample variance ratio
sample_variance_ratio = sample_var1 / sample_var2
print(f"\nSample variance ratio (s₁²/s₂²) = {sample_variance_ratio:.4f}")

# Degrees of freedom
df1 = n1 - 1  # degrees of freedom for sample 1
df2 = n2 - 1  # degrees of freedom for sample 2

print(f"\nDegrees of freedom: df₁ = {df1}, df₂ = {df2}")

# 95% Confidence Interval for Variance Ratio using F-distribution
alpha = 0.05
f_lower = stats.f.ppf(alpha/2, df1, df2)        # Lower critical value
f_upper = stats.f.ppf(1 - alpha/2, df1, df2)    # Upper critical value

print(f"\nF-distribution critical values:")
print(f"   F₀.₀₂₅({df1},{df2}) = {f_lower:.4f}")
print(f"   F₀.₉₇₅({df1},{df2}) = {f_upper:.4f}")

# Confidence interval for σ₁²/σ₂²
ci_lower = sample_variance_ratio / f_upper
ci_upper = sample_variance_ratio / f_lower

print(f"\n=== 95% CONFIDENCE INTERVAL FOR VARIANCE RATIO (σ₁²/σ₂²) ===")
print(f"Sample variance ratio (s₁²/s₂²) = {sample_variance_ratio:.4f}")
print(f"95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Interval width: {ci_upper - ci_lower:.4f}")
print(f"Contains true ratio ({true_variance_ratio:.3f})? {ci_lower <= true_variance_ratio <= ci_upper}")

# F-test for equal variances
f_statistic = max(sample_var1, sample_var2) / min(sample_var1, sample_var2)
df_num = (n1-1) if sample_var1 > sample_var2 else (n2-1)
df_den = (n2-1) if sample_var1 > sample_var2 else (n1-1)
p_value_f_test = 2 * (1 - stats.f.cdf(f_statistic, df_num, df_den))

print(f"\n=== F-TEST FOR EQUAL VARIANCES ===")
print(f"F-statistic = {f_statistic:.4f}")
print(f"Degrees of freedom: ({df_num}, {df_den})")
print(f"p-value = {p_value_f_test:.6f}")
print(f"Conclusion at α=0.05: {'Reject H₀' if p_value_f_test < 0.05 else 'Fail to reject H₀'} (equal variances)")

# Visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

# Plot 1: Sample distributions
ax1.hist(sample1, bins=15, alpha=0.7, color='skyblue', density=True, 
         label=f'Sample 1 (n={n1})\ns² = {sample_var1:.2f}')
ax1.hist(sample2, bins=15, alpha=0.7, color='lightcoral', density=True, 
         label=f'Sample 2 (n={n2})\ns² = {sample_var2:.2f}')
ax1.set_title('Sample Distributions')
ax1.set_xlabel('Value')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: F-distribution and confidence interval
x_f = np.linspace(0, 3, 1000)
f_pdf = stats.f.pdf(x_f, df1, df2)

ax2.plot(x_f, f_pdf, 'b-', linewidth=2, label=f'F({df1},{df2}) distribution')
ax2.axvline(sample_variance_ratio, color='red', linestyle='-', linewidth=2, 
           label=f'Sample ratio: {sample_variance_ratio:.3f}')
ax2.axvline(ci_lower, color='green', linestyle='--', linewidth=2, 
           label=f'CI Lower: {ci_lower:.3f}')
ax2.axvline(ci_upper, color='green', linestyle='--', linewidth=2, 
           label=f'CI Upper: {ci_upper:.3f}')
ax2.axvline(true_variance_ratio, color='orange', linestyle=':', linewidth=2, 
           label=f'True ratio: {true_variance_ratio:.3f}')

# Shade the confidence interval
x_fill = x_f[(x_f >= ci_lower) & (x_f <= ci_upper)]
y_fill = stats.f.pdf(x_fill, df1, df2)
ax2.fill_between(x_fill, y_fill, alpha=0.3, color='green', label='95% CI region')

ax2.set_title('F-Distribution and Confidence Interval')
ax2.set_xlabel('Variance Ratio (σ₁²/σ₂²)')
ax2.set_ylabel('Probability Density')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 3)

# Plot 3: Simulation of sampling distribution of variance ratio
n_simulations = 1000
variance_ratios = []

np.random.seed(456)
for _ in range(n_simulations):
    sim_sample1 = np.random.choice(population1, n1, replace=False)
    sim_sample2 = np.random.choice(population2, n2, replace=False)
    sim_var1 = np.var(sim_sample1, ddof=1)
    sim_var2 = np.var(sim_sample2, ddof=1)
    variance_ratios.append(sim_var1 / sim_var2)

ax3.hist(variance_ratios, bins=30, alpha=0.7, density=True, color='lightgreen', 
         label=f'Simulated Sampling Distribution\n(n={n_simulations} samples)')

# Overlay theoretical F-distribution (scaled)
x_range = np.linspace(min(variance_ratios), max(variance_ratios), 100)
theoretical_pdf = stats.f.pdf(x_range, df1, df2)
ax3.plot(x_range, theoretical_pdf, 'r-', linewidth=2, label='Theoretical F-distribution')

ax3.axvline(true_variance_ratio, color='orange', linestyle=':', linewidth=2, 
           label=f'True ratio: {true_variance_ratio:.3f}')
ax3.axvline(sample_variance_ratio, color='red', linestyle='-', linewidth=2, 
           label=f'Our sample ratio: {sample_variance_ratio:.3f}')

ax3.set_title('Sampling Distribution of Variance Ratio')
ax3.set_xlabel('Sample Variance Ratio (s₁²/s₂²)')
ax3.set_ylabel('Density')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Summary statistics and confidence intervals
ax4.axis('tight')
ax4.axis('off')

# Create summary table
table_data = [
    ['Statistic', 'Value'],
    ['Sample 1 size (n₁)', f'{n1}'],
    ['Sample 2 size (n₂)', f'{n2}'],
    ['Sample 1 variance (s₁²)', f'{sample_var1:.4f}'],
    ['Sample 2 variance (s₂²)', f'{sample_var2:.4f}'],
    ['Sample variance ratio (s₁²/s₂²)', f'{sample_variance_ratio:.4f}'],
    ['True variance ratio (σ₁²/σ₂²)', f'{true_variance_ratio:.3f}'],
    ['', ''],
    ['Degrees of freedom', f'({df1}, {df2})'],
    ['F₀.₀₂₅', f'{f_lower:.4f}'],
    ['F₀.₉₇₅', f'{f_upper:.4f}'],
    ['', ''],
    ['95% CI Lower bound', f'{ci_lower:.4f}'],
    ['95% CI Upper bound', f'{ci_upper:.4f}'],
    ['CI Width', f'{ci_upper - ci_lower:.4f}'],
    ['Contains true ratio?', f'{"Yes" if ci_lower <= true_variance_ratio <= ci_upper else "No"}'],
    ['', ''],
    ['F-test p-value', f'{p_value_f_test:.6f}'],
    ['Equal variances?', f'{"Yes" if p_value_f_test > 0.05 else "No"}']
]

table = ax4.table(cellText=table_data, cellLoc='left', loc='center', 
                  colWidths=[0.6, 0.4])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Color every other row
for i in range(len(table_data)):
    if i % 2 == 0:
        for j in range(2):
            table[(i, j)].set_facecolor('#F0F0F0')

ax4.set_title('Variance Ratio Analysis Summary', pad=20, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# Additional analysis
print(f"\n=== ADDITIONAL ANALYSIS ===")

# Confidence interval for individual variances
# For σ₁²
chi2_lower_1 = stats.chi2.ppf(alpha/2, df1)
chi2_upper_1 = stats.chi2.ppf(1-alpha/2, df1)
var1_ci_lower = (df1 * sample_var1) / chi2_upper_1
var1_ci_upper = (df1 * sample_var1) / chi2_lower_1

print(f"95% CI for σ₁²: [{var1_ci_lower:.4f}, {var1_ci_upper:.4f}]")
print(f"Contains true σ₁² ({sigma1**2})? {var1_ci_lower <= sigma1**2 <= var1_ci_upper}")

# For σ₂²
chi2_lower_2 = stats.chi2.ppf(alpha/2, df2)
chi2_upper_2 = stats.chi2.ppf(1-alpha/2, df2)
var2_ci_lower = (df2 * sample_var2) / chi2_upper_2
var2_ci_upper = (df2 * sample_var2) / chi2_lower_2

print(f"95% CI for σ₂²: [{var2_ci_lower:.4f}, {var2_ci_upper:.4f}]")
print(f"Contains true σ₂² ({sigma2**2})? {var2_ci_lower <= sigma2**2 <= var2_ci_upper}")

print(f"\n=== INTERPRETATION ===")
print(f"The 95% confidence interval for the variance ratio (σ₁²/σ₂²) is [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"This means we are 95% confident that the true ratio lies within this interval.")
print(f"Since the true ratio is {true_variance_ratio:.3f} and our interval {'contains' if ci_lower <= true_variance_ratio <= ci_upper else 'does not contain'} this value,")
print(f"our confidence interval {'successfully captures' if ci_lower <= true_variance_ratio <= ci_upper else 'fails to capture'} the true parameter.")

if abs(true_variance_ratio - 1.0) < 0.001:  # If variances are approximately equal
    if ci_lower <= 1.0 <= ci_upper:
        print(f"Since 1.0 is within our confidence interval, we cannot reject the hypothesis of equal variances.")
    else:
        print(f"Since 1.0 is not within our confidence interval, we reject the hypothesis of equal variances.")