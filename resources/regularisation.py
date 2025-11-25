import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, validation_curve
from sklearn.metrics import mean_squared_error
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji']

# Data from the user
age = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
height = np.array([70.56, 67.68, 80.88, 82.32, 84.00, 90.00, 93.60, 105.36, 109.92])

# Create polynomial features (for demonstration of regularization benefits)
def create_polynomial_features(x, degree=2):
    """Create polynomial features up to given degree"""
    X = np.ones((len(x), 1))  # intercept
    for d in range(1, degree + 1):
        X = np.column_stack([X, x**d])
    return X

# Prepare data
X_poly = create_polynomial_features(age, degree=2)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly[:, 1:])  # Don't scale intercept
X_scaled = np.column_stack([np.ones(len(age)), X_scaled])  # Add back intercept

print("REGULARIZATION EVALUATION DEMONSTRATION")
print("="*50)

# Create comprehensive figure
fig = plt.figure(figsize=(20, 16))

# 1. PROPER EVALUATION: Cross-validation comparison
ax1 = plt.subplot(3, 4, 1)
models = {
    'No Regularization': LinearRegression(),
    'Ridge (λ=1.0)': Ridge(alpha=1.0),
    'Lasso (λ=0.1)': Lasso(alpha=0.1, max_iter=1000)
}

cv_scores = {}
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, height, cv=3, scoring='neg_mean_squared_error')
    cv_scores[name] = -scores.mean()
    print(f"{name}: CV MSE = {cv_scores[name]:.2f} ± {scores.std():.2f}")

ax1.bar(cv_scores.keys(), cv_scores.values(), alpha=0.7)
ax1.set_title('Cross-Validation MSE\n(Lower is Better)')
ax1.set_ylabel('MSE')
plt.xticks(rotation=45)

# 2. REGULARIZATION PATH: Ridge
ax2 = plt.subplot(3, 4, 2)
alphas = np.logspace(-3, 2, 50)
train_scores, val_scores = validation_curve(
    Ridge(), X_scaled, height, param_name='alpha', param_range=alphas,
    cv=3, scoring='neg_mean_squared_error'
)

ax2.semilogx(alphas, -train_scores.mean(axis=1), 'b-', label='Training')
ax2.semilogx(alphas, -val_scores.mean(axis=1), 'r-', label='Validation')
ax2.fill_between(alphas, -val_scores.mean(axis=1) - val_scores.std(axis=1),
                 -val_scores.mean(axis=1) + val_scores.std(axis=1), alpha=0.2, color='red')
optimal_alpha = alphas[np.argmin(-val_scores.mean(axis=1))]
ax2.axvline(optimal_alpha, color='green', linestyle='--', label=f'Optimal λ={optimal_alpha:.3f}')
ax2.set_xlabel('Regularization Parameter (λ)')
ax2.set_ylabel('MSE')
ax2.set_title('Ridge Regularization Path')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. COEFFICIENT BEHAVIOR
ax3 = plt.subplot(3, 4, 3)
coefficients = []
for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_scaled, height)
    coefficients.append(ridge.coef_[1:])  # Exclude intercept

coefficients = np.array(coefficients)
for i in range(coefficients.shape[1]):
    ax3.semilogx(alphas, coefficients[:, i], label=f'β{i+1}')
ax3.set_xlabel('Regularization Parameter (λ)')
ax3.set_ylabel('Coefficient Value')
ax3.set_title('Ridge: Coefficient Shrinkage')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. LASSO PATH
ax4 = plt.subplot(3, 4, 4)
lasso_alphas = np.logspace(-3, 1, 50)
lasso_coeffs = []
for alpha in lasso_alphas:
    try:
        lasso = Lasso(alpha=alpha, max_iter=2000)
        lasso.fit(X_scaled, height)
        lasso_coeffs.append(lasso.coef_[1:])
    except:
        lasso_coeffs.append([0] * (X_scaled.shape[1] - 1))

lasso_coeffs = np.array(lasso_coeffs)
for i in range(lasso_coeffs.shape[1]):
    ax4.semilogx(lasso_alphas, lasso_coeffs[:, i], label=f'β{i+1}')
ax4.set_xlabel('Regularization Parameter (λ)')
ax4.set_ylabel('Coefficient Value')
ax4.set_title('Lasso: Coefficient Path (Sparsity)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5-6. GEOMETRIC INTERPRETATION: RIDGE
def plot_ridge_geometry():
    """Create geometric interpretation for Ridge regression"""
    # Simplified 2D case for visualization
    # We'll use the first two non-intercept coefficients
    
    # Fit models to get approximate solution
    ridge_weak = Ridge(alpha=0.1).fit(X_scaled, height)
    ridge_strong = Ridge(alpha=10.0).fit(X_scaled, height)
    ols = LinearRegression().fit(X_scaled, height)
    
    beta1_range = np.linspace(-10, 15, 100)
    beta2_range = np.linspace(-5, 10, 100)
    B1, B2 = np.meshgrid(beta1_range, beta2_range)
    
    # Compute loss surface (simplified for 2D visualization)
    Z = np.zeros_like(B1)
    for i in range(len(beta1_range)):
        for j in range(len(beta2_range)):
            beta_temp = np.array([ols.intercept_, B1[j,i], B2[j,i]])
            if len(beta_temp) < X_scaled.shape[1]:
                beta_temp = np.pad(beta_temp, (0, X_scaled.shape[1] - len(beta_temp)))
            pred = X_scaled @ beta_temp[:X_scaled.shape[1]]
            Z[j,i] = np.mean((height - pred)**2)
    
    ax5 = plt.subplot(3, 4, 5)
    contour = ax5.contour(B1, B2, Z, levels=20, alpha=0.6)
    ax5.clabel(contour, inline=True, fontsize=8)
    
    # Ridge constraint circles
    for alpha, color, label in [(0.1, 'blue', 'λ=0.1'), (2.0, 'red', 'λ=2.0')]:
        constraint_radius = np.sqrt(15/alpha)  # Adjusted for visibility
        circle = plt.Circle((0, 0), constraint_radius, fill=False, 
                          color=color, linewidth=2, label=f'Ridge {label}')
        ax5.add_patch(circle)
    
    # Mark solutions
    ax5.plot(ols.coef_[1], ols.coef_[2], 'ko', markersize=8, label='OLS')
    ax5.plot(ridge_weak.coef_[1], ridge_weak.coef_[2], 'bo', markersize=8, label='Ridge λ=0.1')
    ax5.plot(ridge_strong.coef_[1], ridge_strong.coef_[2], 'ro', markersize=8, label='Ridge λ=10')
    
    ax5.set_xlabel('β₁ (Age coefficient)')
    ax5.set_ylabel('β₂ (Age² coefficient)')
    ax5.set_title('Ridge: Circular Constraint')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-10, 15)
    ax5.set_ylim(-5, 10)

plot_ridge_geometry()

# 6. GEOMETRIC INTERPRETATION: LASSO
def plot_lasso_geometry():
    """Create geometric interpretation for Lasso regression"""
    
    lasso_weak = Lasso(alpha=0.1, max_iter=2000).fit(X_scaled, height)
    lasso_strong = Lasso(alpha=1.0, max_iter=2000).fit(X_scaled, height)
    ols = LinearRegression().fit(X_scaled, height)
    
    beta1_range = np.linspace(-10, 15, 100)
    beta2_range = np.linspace(-5, 10, 100)
    B1, B2 = np.meshgrid(beta1_range, beta2_range)
    
    # Compute loss surface
    Z = np.zeros_like(B1)
    for i in range(len(beta1_range)):
        for j in range(len(beta2_range)):
            beta_temp = np.array([ols.intercept_, B1[j,i], B2[j,i]])
            if len(beta_temp) < X_scaled.shape[1]:
                beta_temp = np.pad(beta_temp, (0, X_scaled.shape[1] - len(beta_temp)))
            pred = X_scaled @ beta_temp[:X_scaled.shape[1]]
            Z[j,i] = np.mean((height - pred)**2)
    
    ax6 = plt.subplot(3, 4, 6)
    contour = ax6.contour(B1, B2, Z, levels=20, alpha=0.6)
    
    # Lasso constraint diamonds
    for alpha, color, label in [(0.1, 'blue', 'λ=0.1'), (1.0, 'red', 'λ=1.0')]:
        constraint_size = 10/alpha  # Adjusted for visibility
        diamond_x = [-constraint_size, 0, constraint_size, 0, -constraint_size]
        diamond_y = [0, constraint_size, 0, -constraint_size, 0]
        ax6.plot(diamond_x, diamond_y, color=color, linewidth=2, label=f'Lasso {label}')
    
    # Mark solutions
    ax6.plot(ols.coef_[1], ols.coef_[2], 'ko', markersize=8, label='OLS')
    ax6.plot(lasso_weak.coef_[1], lasso_weak.coef_[2], 'bo', markersize=8, label='Lasso λ=0.1')
    ax6.plot(lasso_strong.coef_[1], lasso_strong.coef_[2], 'ro', markersize=8, label='Lasso λ=1.0')
    
    ax6.set_xlabel('β₁ (Age coefficient)')
    ax6.set_ylabel('β₂ (Age² coefficient)')
    ax6.set_title('Lasso: Diamond Constraint')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-10, 15)
    ax6.set_ylim(-8, 12)

plot_lasso_geometry()

# 7. PREDICTION COMPARISON
ax7 = plt.subplot(3, 4, 7)
age_smooth = np.linspace(0.5, 5.5, 100)
X_smooth = create_polynomial_features(age_smooth, degree=2)
X_smooth_scaled = scaler.transform(X_smooth[:, 1:])
X_smooth_scaled = np.column_stack([np.ones(len(age_smooth)), X_smooth_scaled])

# Fit models
ols = LinearRegression().fit(X_scaled, height)
ridge = Ridge(alpha=optimal_alpha).fit(X_scaled, height)
lasso = Lasso(alpha=0.1, max_iter=2000).fit(X_scaled, height)

# Predictions
pred_ols = ols.predict(X_smooth_scaled)
pred_ridge = ridge.predict(X_smooth_scaled)
pred_lasso = lasso.predict(X_smooth_scaled)

ax7.scatter(age, height, color='black', s=60, alpha=0.8, label='Data', zorder=5)
ax7.plot(age_smooth, pred_ols, 'g-', linewidth=2, label='OLS', alpha=0.8)
ax7.plot(age_smooth, pred_ridge, 'b-', linewidth=2, label=f'Ridge (λ={optimal_alpha:.3f})', alpha=0.8)
ax7.plot(age_smooth, pred_lasso, 'r-', linewidth=2, label='Lasso (λ=0.1)', alpha=0.8)
ax7.set_xlabel('Age')
ax7.set_ylabel('Height')
ax7.set_title('Model Predictions Comparison')
ax7.legend()
ax7.grid(True, alpha=0.3)

# 8. COEFFICIENT COMPARISON
ax8 = plt.subplot(3, 4, 8)
models_coef = {'OLS': ols.coef_[1:], 'Ridge': ridge.coef_[1:], 'Lasso': lasso.coef_[1:]}
coef_names = ['β₁ (Age)', 'β₂ (Age²)']

x_pos = np.arange(len(coef_names))
width = 0.25

for i, (name, coefs) in enumerate(models_coef.items()):
    ax8.bar(x_pos + i*width, coefs, width, alpha=0.8, label=name)

ax8.set_xlabel('Coefficients')
ax8.set_ylabel('Coefficient Value')
ax8.set_title('Coefficient Magnitudes')
ax8.set_xticks(x_pos + width)
ax8.set_xticklabels(coef_names)
ax8.legend()
ax8.grid(True, alpha=0.3)

# 9. STABILITY ANALYSIS (Bootstrap)
ax9 = plt.subplot(3, 4, 9)
n_bootstrap = 50
ols_coefs = []
ridge_coefs = []

np.random.seed(42)
for _ in range(n_bootstrap):
    # Bootstrap sample
    indices = np.random.choice(len(age), size=len(age), replace=True)
    X_boot = X_scaled[indices]
    y_boot = height[indices]
    
    # Fit models
    ols_boot = LinearRegression().fit(X_boot, y_boot)
    ridge_boot = Ridge(alpha=optimal_alpha).fit(X_boot, y_boot)
    
    ols_coefs.append(ols_boot.coef_[1:])
    ridge_coefs.append(ridge_boot.coef_[1:])

ols_coefs = np.array(ols_coefs)
ridge_coefs = np.array(ridge_coefs)

ax9.scatter(ols_coefs[:, 0], ols_coefs[:, 1], alpha=0.6, label='OLS', s=30)
ax9.scatter(ridge_coefs[:, 0], ridge_coefs[:, 1], alpha=0.6, label='Ridge', s=30)
ax9.set_xlabel('β₁ (Age coefficient)')
ax9.set_ylabel('β₂ (Age² coefficient)')
ax9.set_title('Coefficient Stability\n(Bootstrap Samples)')
ax9.legend()
ax9.grid(True, alpha=0.3)

# 10. ERROR ANALYSIS
ax10 = plt.subplot(3, 4, 10)
train_mse_ols = mean_squared_error(height, ols.predict(X_scaled))
train_mse_ridge = mean_squared_error(height, ridge.predict(X_scaled))
train_mse_lasso = mean_squared_error(height, lasso.predict(X_scaled))

mse_comparison = {
    'OLS': train_mse_ols,
    'Ridge': train_mse_ridge, 
    'Lasso': train_mse_lasso
}

bars = ax10.bar(mse_comparison.keys(), mse_comparison.values(), alpha=0.7)
ax10.set_ylabel('Training MSE')
ax10.set_title('Training Error Comparison')
ax10.grid(True, alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, mse_comparison.values()):
    ax10.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{value:.2f}', ha='center', va='bottom')

# 11. REGULARIZATION EFFECT VISUALIZATION
ax11 = plt.subplot(3, 4, 11)
lambdas = [0, 0.1, 1.0, 10.0]
complexity_measures = []
training_errors = []

for lam in lambdas:
    if lam == 0:
        model = LinearRegression()
    else:
        model = Ridge(alpha=lam)
    
    model.fit(X_scaled, height)
    pred = model.predict(X_scaled)
    
    training_errors.append(mean_squared_error(height, pred))
    complexity_measures.append(np.sum(model.coef_[1:]**2))  # L2 norm of coefficients

ax11.plot(complexity_measures, training_errors, 'bo-', linewidth=2, markersize=8)
for i, lam in enumerate(lambdas):
    ax11.annotate(f'λ={lam}', (complexity_measures[i], training_errors[i]), 
                 xytext=(5, 5), textcoords='offset points')

ax11.set_xlabel('Model Complexity (||β||²)')
ax11.set_ylabel('Training MSE')
ax11.set_title('Regularization Effect:\nComplexity vs. Error')
ax11.grid(True, alpha=0.3)

# 12. KEY INSIGHTS SUMMARY
ax12 = plt.subplot(3, 4, 12)
ax12.axis('off')
insights_text = """
KEY INSIGHTS:

✓ Regularization reduces coefficient 
  magnitudes and model complexity

✓ Ridge shrinks coefficients smoothly
  toward zero (circular constraint)

✓ Lasso can set coefficients exactly 
  to zero (diamond constraint creates
  sparse solutions at corners)

✓ Optimal λ balances bias-variance
  tradeoff (validation curve)

✓ Regularized models are more stable
  across different training samples

✓ Training error alone is misleading -
  use cross-validation!
"""

ax12.text(0.05, 0.95, insights_text, transform=ax12.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
         facecolor='lightblue', alpha=0.7))

plt.tight_layout()
plt.show()

# Print numerical summary
print("\n" + "="*50)
print("NUMERICAL SUMMARY")
print("="*50)
print(f"Optimal Ridge λ: {optimal_alpha:.4f}")
print(f"\nCoefficient Comparison:")
print(f"{'Model':<12} {'β₁ (Age)':<12} {'β₂ (Age²)':<12} {'||β||²':<12}")
print("-" * 50)
print(f"{'OLS':<12} {ols.coef_[1]:<12.3f} {ols.coef_[2]:<12.3f} {np.sum(ols.coef_[1:]**2):<12.3f}")
print(f"{'Ridge':<12} {ridge.coef_[1]:<12.3f} {ridge.coef_[2]:<12.3f} {np.sum(ridge.coef_[1:]**2):<12.3f}")
print(f"{'Lasso':<12} {lasso.coef_[1]:<12.3f} {lasso.coef_[2]:<12.3f} {np.sum(lasso.coef_[1:]**2):<12.3f}")

print(f"\nStability Analysis (Bootstrap std dev):")
print(f"OLS:   β₁ std = {np.std(ols_coefs[:, 0]):.3f}, β₂ std = {np.std(ols_coefs[:, 1]):.3f}")
print(f"Ridge: β₁ std = {np.std(ridge_coefs[:, 0]):.3f}, β₂ std = {np.std(ridge_coefs[:, 1]):.3f}")
print(f"\nRegularization reduces coefficient variance by {(1 - np.std(ridge_coefs)/np.std(ols_coefs))*100:.1f}% on average")