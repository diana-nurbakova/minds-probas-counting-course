import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Set random seed for reproducibility
np.random.seed(42)

def true_function(x):
    """The true underlying function we want to learn"""
    return 1.5 * x**2 + 0.5 * x + 0.3

def generate_data(n_samples=50, noise_std=0.3):
    """Generate synthetic data with noise"""
    x = np.linspace(-1, 1, n_samples)
    y = true_function(x) + np.random.normal(0, noise_std, n_samples)
    return x, y

def fit_polynomial_models(x_train, y_train, x_test, y_test, max_degree=15):
    """Fit polynomial models of different degrees"""
    degrees = range(1, max_degree + 1)
    train_errors = []
    test_errors = []
    models = {}
    
    for degree in degrees:
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
        X_test_poly = poly_features.transform(x_test.reshape(-1, 1))
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_poly)
        y_test_pred = model.predict(X_test_poly)
        
        # Calculate errors
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        
        train_errors.append(train_mse)
        test_errors.append(test_mse)
        models[degree] = (model, poly_features)
        
        print(f"Degree {degree:2d}: Train MSE = {train_mse:.4f}, Test MSE = {test_mse:.4f}")
    
    return degrees, train_errors, test_errors, models

# Generate data
print("Generating synthetic data...")
x, y = generate_data(n_samples=30, noise_std=0.2)

# Split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
print("\nFitting polynomial models of different degrees...")

# Fit models of different complexity
degrees, train_errors, test_errors, models = fit_polynomial_models(
    x_train, y_train, x_test, y_test, max_degree=12
)

# Create comprehensive visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Original data and true function
x_smooth = np.linspace(-1, 1, 200)
y_true = true_function(x_smooth)

ax1.plot(x_smooth, y_true, 'g-', linewidth=3, label='True Function', alpha=0.8)
ax1.scatter(x_train, y_train, color='blue', s=60, alpha=0.7, label=f'Training Data (n={len(x_train)})')
ax1.scatter(x_test, y_test, color='red', s=60, alpha=0.7, label=f'Test Data (n={len(x_test)})')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Original Data and True Function')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Training vs Test Error (Bias-Variance Tradeoff)
ax2.plot(degrees, train_errors, 'b-o', linewidth=2, markersize=6, label='Training Error')
ax2.plot(degrees, test_errors, 'r-s', linewidth=2, markersize=6, label='Test Error')
ax2.set_xlabel('Polynomial Degree (Model Complexity)')
ax2.set_ylabel('Mean Squared Error')
ax2.set_title('Train Error vs. Test Error')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_yscale('log')

# Find optimal degree (minimum test error)
optimal_degree = degrees[np.argmin(test_errors)]
min_test_error = min(test_errors)
ax2.axvline(x=optimal_degree, color='green', linestyle='--', alpha=0.7, 
           label=f'Optimal Degree: {optimal_degree}')
ax2.legend()

# Plot 3: Comparison of different complexity models
models_to_show = [2, optimal_degree, 10]  # Simple, optimal, overfitted
colors = ['blue', 'green', 'red']
labels = ['Underfitted (Degree 2)', f'Well-fitted (Degree {optimal_degree})', 'Overfitted (Degree 10)']

x_smooth = np.linspace(-1, 1, 200)
ax3.plot(x_smooth, true_function(x_smooth), 'black', linewidth=3, label='True Function', alpha=0.8)

for i, degree in enumerate(models_to_show):
    model, poly_features = models[degree]
    X_smooth_poly = poly_features.transform(x_smooth.reshape(-1, 1))
    y_smooth_pred = model.predict(X_smooth_poly)
    
    ax3.plot(x_smooth, y_smooth_pred, color=colors[i], linewidth=2, 
            label=labels[i], alpha=0.8)

ax3.scatter(x_train, y_train, color='blue', s=40, alpha=0.6, label='Training Data')
ax3.scatter(x_test, y_test, color='red', s=40, alpha=0.6, label='Test Data')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Model Comparison: Different Complexity Levels')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(-1, 3)

# Plot 4: Residuals for the overfitted model
overfitted_model, overfitted_poly = models[10]
X_train_poly = overfitted_poly.transform(x_train.reshape(-1, 1))
X_test_poly = overfitted_poly.transform(x_test.reshape(-1, 1))

y_train_pred = overfitted_model.predict(X_train_poly)
y_test_pred = overfitted_model.predict(X_test_poly)

train_residuals = y_train - y_train_pred
test_residuals = y_test - y_test_pred

ax4.scatter(y_train_pred, train_residuals, color='blue', alpha=0.7, s=60, label='Training Residuals')
ax4.scatter(y_test_pred, test_residuals, color='red', alpha=0.7, s=60, label='Test Residuals')
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4.set_xlabel('Predicted Values')
ax4.set_ylabel('Residuals')
ax4.set_title('Residuals for Overfitted Model (Degree 10)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print summary statistics
print(f"\n{'='*60}")
print("OVERFITTING ANALYSIS SUMMARY")
print(f"{'='*60}")
print(f"Optimal polynomial degree: {optimal_degree}")
print(f"Minimum test error: {min_test_error:.4f}")
print(f"\nComparison of key models:")
for degree in [2, optimal_degree, 10]:
    train_idx = degree - 1
    print(f"Degree {degree:2d}: Train MSE = {train_errors[train_idx]:.4f}, Test MSE = {test_errors[train_idx]:.4f}, "
          f"Gap = {test_errors[train_idx] - train_errors[train_idx]:.4f}")

print(f"\n{'='*60}")
print("KEY OBSERVATIONS:")
print("• Low-degree models (underfitting): High bias, low variance")
print("• High-degree models (overfitting): Low bias, high variance") 
print("• Optimal model: Best balance between bias and variance")
print("• Training error always decreases with complexity")
print("• Test error first decreases, then increases (U-shaped curve)")
print(f"{'='*60}")

# Additional analysis: Show coefficient magnitudes for different degrees
print(f"\nCOEFFICIENT ANALYSIS:")
for degree in [2, optimal_degree, 10]:
    model, _ = models[degree]
    coeffs = model.coef_
    print(f"Degree {degree}: Max coefficient magnitude = {np.max(np.abs(coeffs)):.2f}")
    if degree == 10:
        print(f"   Coefficients: {coeffs[:5]}... (showing first 5 of {len(coeffs)})")