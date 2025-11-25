import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve, learning_curve
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.font_manager as fm

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji']



def generate_nonlinear_data(n=50, noise_level=0.1, scenario='quadratic'):
    """Generate data with different non-linear relationships"""
    x = np.random.uniform(-2, 2, n)
    
    if scenario == 'quadratic':
        # U-shaped relationship
        y = 1 + 2*x - 3*x**2 + np.random.normal(0, noise_level, n)
        true_func = lambda x: 1 + 2*x - 3*x**2
        title = "Quadratic Relationship"
        
    elif scenario == 'cubic':
        # S-shaped relationship
        y = x**3 - 2*x + np.random.normal(0, noise_level, n)
        true_func = lambda x: x**3 - 2*x
        title = "Cubic Relationship"
        
    elif scenario == 'oscillatory':
        # Periodic-like relationship
        y = np.sin(2*np.pi*x) + 0.5*x + np.random.normal(0, noise_level, n)
        true_func = lambda x: np.sin(2*np.pi*x) + 0.5*x
        title = "Oscillatory Relationship"
        
    elif scenario == 'interaction':
        # Two variables with interaction
        x2 = np.random.uniform(-1, 1, n)
        x = np.column_stack([x, x2])
        y = 1 + 2*x[:, 0] + 3*x[:, 1] + 4*x[:, 0]*x[:, 1] + np.random.normal(0, noise_level, n)
        true_func = lambda x1, x2: 1 + 2*x1 + 3*x2 + 4*x1*x2
        title = "Interaction Effect"
        
    return x, y, true_func, title

def gen_scenarios(axes):
    
    scenarios = ['quadratic', 'cubic', 'oscillatory']
    
    true_func_text = [r"$y = 1 + 2x - 3x^2 + \epsilon$", 
                      r"$y = x^3 - 2x + \epsilon$", 
                      r"$y = \sin(2\pi x) + 0.5x + \epsilon$"]
    
    residual_patterns = ["U-shaped pattern\n(Missing x² term)", 
                        "S-shaped pattern\n(Missing x³ term)",
                        "Wave pattern\n(Missing periodic terms)"]
    
    # Store model results for residual analysis
    model_results = []
    
    for i, scenario in enumerate(scenarios):
        x, y, true_func, title = generate_nonlinear_data(100, 0.3, scenario)
        
        # Main plot (top row)
        ax_main = axes[0, i]
        
        # Plot data
        ax_main.scatter(x, y, alpha=0.6, s=30, label='Data')
        
        # Fit linear model
        linear_model = LinearRegression().fit(x.reshape(-1, 1), y)
        
        # Calculate residuals and fitted values
        y_fitted = linear_model.predict(x.reshape(-1, 1))
        residuals = y - y_fitted
        
        # Store for residual plots
        model_results.append({
            'x': x, 'y': y, 'fitted': y_fitted, 'residuals': residuals,
            'model': linear_model, 'true_func': true_func
        })
        
        # Plot true function and linear fit
        x_plot = np.linspace(x.min(), x.max(), 200)
        y_true = true_func(x_plot)
        y_linear = linear_model.predict(x_plot.reshape(-1, 1))
        
        ax_main.plot(x_plot, y_true, 'g-', linewidth=3, label='True Function:' + true_func_text[i], alpha=0.8)
        ax_main.plot(x_plot, y_linear, 'r--', linewidth=2, label='Linear Fit', alpha=0.8)
        
        # Calculate R²
        r2_linear = linear_model.score(x.reshape(-1, 1), y)
        mse = mean_squared_error(y, y_fitted)
        
        ax_main.set_title(f'{title}\nLinear R² = {r2_linear:.3f}, MSE = {mse:.3f}', fontsize=10)
        ax_main.legend(fontsize=9)
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(axis='x', labelsize=7)
        ax_main.tick_params(axis='y', labelsize=7)
        
        # Residual plot (bottom row)
        ax_resid = axes[1, i]
        
        # Plot residuals vs fitted values
        ax_resid.scatter(y_fitted, residuals, alpha=0.7, s=35, color='red', 
                        edgecolors='darkred', linewidth=0.5)
        ax_resid.axhline(y=0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
        
        # Add trend line to highlight pattern
        if len(y_fitted) > 10:  # Only if we have enough points
            # Sort by fitted values for smooth trend line
            sorted_indices = np.argsort(y_fitted)
            fitted_sorted = y_fitted[sorted_indices]
            residuals_sorted = residuals[sorted_indices]
            
            # Use a simple moving average to show pattern
            window_size = max(5, len(fitted_sorted) // 10)
            if len(fitted_sorted) >= window_size:
                trend = np.convolve(residuals_sorted, np.ones(window_size)/window_size, mode='valid')
                trend_x = fitted_sorted[window_size//2:len(trend)+window_size//2]
                ax_resid.plot(trend_x, trend, 'blue', linewidth=2, alpha=0.8, label='Pattern Trend')
        
        # Customize residual plot
        ax_resid.set_xlabel('Fitted Values')
        ax_resid.set_ylabel('Residuals')
        ax_resid.set_title(f'{residual_patterns[i]}', fontsize=11, color='red')
        ax_resid.grid(True, alpha=0.3)
        ax_resid.tick_params(labelsize=9)
        
        # Add diagnostic text box
        residual_std = np.std(residuals)
        residual_range = np.ptp(residuals)  # peak-to-peak range
        
        if residual_std < 0.5:
            std_quality = "EXCELLENT"
            std_meaning = "Very tight residuals around zero"
        elif residual_std < 1.0:
            std_quality = "GOOD"
            std_meaning = "Reasonable residual spread"
        elif residual_std < 2.0:
            std_quality = "MODERATE"
            std_meaning = "Noticeable residual spread"
        else:
            std_quality = "POOR"
            std_meaning = "Large residual spread - model issues"
        
        ratio_std_ptp = residual_range / (6 * residual_std)  # Theoretical ratio for normal distribution ≈ 1
        
        if ratio_std_ptp < 0.8:
            ptp_interpretation = "No extreme outliers (NORMAL)"
        elif ratio_std_ptp < 1.5:
            ptp_interpretation = "Expected for normal errors (REASONABLE)"
        elif ratio_std_ptp < 2.5:
            ptp_interpretation = "Possible outliers or heavy tails (WIDE)"
        else:
            ptp_interpretation = "Likely outliers or model problems (VERY WIDE)"
        
        diagnostic_text = f"Std: {residual_std:.2f}" + r"$\Rightarrow$" + f"{std_meaning} ({std_quality})\nPeak-to-peak range: {residual_range:.2f}" + r"$\Rightarrow$" + f"{ptp_interpretation}"
    
        if i == 0:    
            ax_resid.text(0.10, 0.20, diagnostic_text, transform=ax_resid.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8),
                        fontsize=8, verticalalignment='top')
        else: 
            ax_resid.text(0.05, 0.95, diagnostic_text, transform=ax_resid.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.85),
                        fontsize=8, verticalalignment='top')
    
    return axes, model_results

def get_scenarios():
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    fig.suptitle('Linear Models vs Non-Linear Data: Issues', 
                fontsize=16)
    
    
    axes, model_results = gen_scenarios(axes)
    
    # Add row labels
    fig.text(0.01, 0.75, 'Model Fits', rotation=90, fontsize=14, 
             verticalalignment='center', weight='bold')
    fig.text(0.01, 0.25, 'Residual Diagnostics', rotation=90, fontsize=14, 
             verticalalignment='center', weight='bold', color='red')
    
    #plt.subplots_adjust(left=0.2, top=0.8)
    plt.tight_layout(rect=[0.02, 0.02, 0.97, 0.97])
    plt.show()
    
    return fig, model_results

def gen_polynomials_examples(axes):
    # Generate quadratic data for detailed analysis
    x_demo, y_demo, true_func_demo, _ = generate_nonlinear_data(50, 0.2, 'quadratic')

    degrees = [1, 2, 3, 5]
    colors = ['red', 'blue', 'green', 'orange']

    for i, degree in enumerate(degrees):
        ax = axes[i]
        
        # Fit polynomial model
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(x_demo.reshape(-1, 1))
        poly_model = LinearRegression().fit(X_poly, y_demo)
        
        # Plot data and fits
        ax.scatter(x_demo, y_demo, alpha=0.6, s=30, label='Data')
        
        x_plot = np.linspace(x_demo.min(), x_demo.max(), 200)
        X_plot_poly = poly_features.transform(x_plot.reshape(-1, 1))
        y_poly_pred = poly_model.predict(X_plot_poly)
        
        # True function
        y_true_plot = true_func_demo(x_plot)
        ax.plot(x_plot, y_true_plot, 'black', linewidth=3, label='True Function', alpha=0.7)
        
        # Polynomial fit
        ax.plot(x_plot, y_poly_pred, colors[i], linewidth=2, linestyle='dashed', label=f'Degree {degree}')
        
        # Calculate metrics
        r2 = poly_model.score(X_poly, y_demo)
        train_mse = mean_squared_error(y_true_plot, y_poly_pred)
        
        ax.set_title(f'Polynomial Degree {degree}\n(R² = {r2:.3f}, MSE = {train_mse:.3f})', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-4, 3)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)
        
    return axes

def get_poly_fit():
    fig, axes = plt.subplots(1, 4, figsize=(10, 4))
    
    fig.suptitle('Polynomial Solution', 
                fontsize=16)
    
    
    axes = gen_polynomials_examples(axes)
    
    # Add row labels
    #fig.text(0.01, 0.75, 'Model Fits', rotation=90, fontsize=14, 
    #         verticalalignment='center', weight='bold')
    # fig.text(0.01, 0.25, 'Residual Diagnostics', rotation=90, fontsize=14, 
    #         verticalalignment='center', weight='bold', color='red')
    
    #plt.subplots_adjust(left=0.2, top=0.8)
    #plt.tight_layout(rect=[0.02, 0.02, 0.97, 0.97])
    plt.tight_layout()
    plt.show()
    
    return fig


def gen_overfit_example(axes, x_train, x_test, y_train, y_test, true_func_over):

    # Define degrees to visualize
    degrees_to_show = [1, 4, 6, 14]  # Key degrees that show the progression
    colors = ['red', 'blue', 'orange', 'purple']
    
    # Row 1: Individual polynomial fits
    polynomial_results = {}  # Store results for later analysis
    
    for i, degree in enumerate(degrees_to_show):
        ax = axes[0, i]
        
        # Fit polynomial model
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
        X_test_poly = poly_features.transform(x_test.reshape(-1, 1))
        
        model = LinearRegression().fit(X_train_poly, y_train)
        
        # Calculate scores
        train_r2 = model.score(X_train_poly, y_train)
        test_r2 = model.score(X_test_poly, y_test)
        
        train_mse = mean_squared_error(y_train, model.predict(X_train_poly))
        test_mse = mean_squared_error(y_test, model.predict(X_test_poly))
        
        # Store results
        polynomial_results[degree] = {
            'model': model,
            'poly_features': poly_features,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse
        }
        
        # Plot data points
        ax.scatter(x_train, y_train, alpha=0.8, s=50, color='blue', 
                  label=f'Train Data (n={len(x_train)})', zorder=5, edgecolors='white')
        ax.scatter(x_test, y_test, alpha=0.6, s=20, color='red', 
                  label=f'Test Data (n={len(x_test)})', zorder=4)
        
        # Plot true function
        x_plot = np.linspace(min(x_train.min(), x_test.min()), 
                           max(x_train.max(), x_test.max()), 200)
        y_true_plot = true_func_over(x_plot)
        ax.plot(x_plot, y_true_plot, 'green', linewidth=3, 
               label='True Function', alpha=0.8, zorder=3)
        
        # Plot polynomial fit
        X_plot_poly = poly_features.transform(x_plot.reshape(-1, 1))
        y_pred_plot = model.predict(X_plot_poly)
        ax.plot(x_plot, y_pred_plot, colors[i], linewidth=2.5, 
               label=f'Degree {degree} Fit', zorder=2)
        
        # Add performance metrics and status
        gap = train_r2 - test_r2
        
        if degree == 1:
            status = "❌ Underfit"
            status_color = 'red'
        elif degree == 4:
            status = "✅ Good Fit"
            status_color = 'green'
        elif degree <= 6:
            status = "⚠️ Starting to Overfit"
            status_color = 'orange'
        else:
            status = "🚨 Severe Overfit"
            status_color = 'red'
        
        ax.set_title(f'{status}\nDegree {degree}: Train R² = {train_r2:.3f}, Test R² = {test_r2:.3f}\n'
                    f'Gap = {gap:.3f}\n Train MSE = {train_mse:.3f}, Test MSE = {test_mse:.3f}', 
                    fontsize=11, color=status_color, weight='bold')
        
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Set consistent y-limits, but allow for extreme overfitting
        if degree <= 5:
            ax.set_ylim(-4, 4)
        else:
            # For high degrees, show the wild behavior but constrain somewhat
            y_min = min(y_train.min(), y_test.min()) - 1
            y_max = max(y_train.max(), y_test.max()) + 1
            ax.set_ylim(max(-8, y_min), min(8, y_max))
             
    return axes, polynomial_results

def gen_bias_var_plot(axes, x_train, x_test, y_train, y_test):
    
    degrees_to_show = [1, 4, 6, 14]
    
    degrees_range = range(1, 16)
    train_scores = []
    test_scores = []
    train_scores_mse = []
    test_scores_mse = []
    model_complexities = []  # Track number of parameters
    
    for degree in degrees_range:
        # Fit model
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
        X_test_poly = poly_features.transform(x_test.reshape(-1, 1))
        
        model = LinearRegression().fit(X_train_poly, y_train)
        
        # Calculate scores
        train_score = model.score(X_train_poly, y_train)
        test_score = model.score(X_test_poly, y_test)
        
        train_mse = mean_squared_error(y_train, model.predict(X_train_poly))  # Decreases
        test_mse = mean_squared_error(y_test, model.predict(X_test_poly))
        
        train_scores.append(train_score)
        test_scores.append(test_score)
        
        train_scores_mse.append(train_mse)
        test_scores_mse.append(test_mse)
        
        model_complexities.append(X_train_poly.shape[1])  # Number of features
    
    # Enhanced bias-variance tradeoff plot
    ax_tradeoff = axes[0][1,1]  
    
    # Plot the curves
    #ax_tradeoff.plot(degrees_range, train_scores, 'b-o', linewidth=3, 
    #                label='Training R²', markersize=6, alpha=0.8)
    #ax_tradeoff.plot(degrees_range, test_scores, 'r-s', linewidth=3, 
    #                label='Test R²', markersize=6, alpha=0.8)
    
    ax_tradeoff.plot(degrees_range, train_scores_mse, 'b-o', linewidth=3, 
                    label='Train MSE', markersize=6, alpha=0.8)
    ax_tradeoff.plot(degrees_range, test_scores_mse, 'r-s', linewidth=3, 
                    label='Test MSE', markersize=6, alpha=0.8)
    
    # Mark optimal degree
    #print(test_scores_mse)
    optimal_degree = degrees_range[np.argmin(test_scores_mse)]
    ax_tradeoff.axvline(optimal_degree, color='green', linestyle='--', 
                       linewidth=2, alpha=0.8, 
                       label=f'Optimal Degree: {optimal_degree}')
    
    # Highlight regions
    ax_tradeoff.axvspan(1, optimal_degree, alpha=0.1, color='red', label='Underfitting Zone')
    ax_tradeoff.axvspan(optimal_degree, max(degrees_range), alpha=0.1, color='orange', 
                       label='Overfitting Zone')
    
    # Mark the degrees we visualized
    for degree in degrees_to_show:
        idx = degree - 1
        train_score = train_scores_mse[idx]
        test_score = test_scores_mse[idx]
        
        ax_tradeoff.plot(degree, train_score, 'bo', markersize=10, alpha=0.7)
        ax_tradeoff.plot(degree, test_score, 'rs', markersize=10, alpha=0.7)
        
        # Add labels for key degrees
        if degree in [2, 10]:
            ax_tradeoff.annotate(f'Degree {degree}\n(shown above)', 
                               xy=(degree, test_score), 
                               xytext=(degree + 1.5, test_score - 0.1),
                               arrowprops=dict(arrowstyle='->', color='black'),
                               fontsize=9, ha='center')
    
    ax_tradeoff.set_xlabel('Polynomial Degree (Model Complexity)', fontsize=12)
    ax_tradeoff.set_ylabel('MSE', fontsize=12)
    ax_tradeoff.set_title('Overfitting Detection: Train Error vs. Test Error', fontsize=14, weight='bold')
    ax_tradeoff.legend(fontsize=10, loc='center right')
    ax_tradeoff.grid(True, alpha=0.3)
    ax_tradeoff.tick_params(labelsize=10)
    ax_tradeoff.set_xticks(list(degrees_range))
    
    return axes

def get_overfit_example():
    
    np.random.seed(42) # for reproducibility    
    # Generate training and test data    
    x_train, y_train, true_func_over, _ = generate_nonlinear_data(30, 0.3, 'cubic')
    x_test, y_test, _, _ = generate_nonlinear_data(100, 0.3, 'cubic')
    # Create figure with subplots
    #fig = plt.figure(figsize=(20, 12))
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))    
    fig.suptitle('The Overfitting Problem', fontsize=16)
    
    axes[1, 0].axis('off')
    axes[1, 1].axis('off')
    axes[1, 2].axis('off')
    axes[1, 3].axis('off')
    
    axes = gen_overfit_example(axes, x_train, x_test, y_train, y_test, true_func_over)
    axes = gen_bias_var_plot(axes, x_train, x_test, y_train, y_test)
    
    plt.tight_layout()
    #plt.show()
    
    return fig
    

def generate_polynomials_plots():    
    
    # Create comprehensive teaching visualization
    fig, axes = plt.subplots(4, 5, figsize=(20, 20))
    plt.subplots_adjust(wspace=0.267, hspace=0.41, right=0.945, bottom=0.06, top=0.962)

    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    axes[2, 0].axis('off')
    axes[3, 0].axis('off')

    # PHASE 2: POLYNOMIAL SOLUTION
    #print("\nPHASE 2: POLYNOMIAL SOLUTION - How polynomial features help")

    ax_intro = axes[0, 0]
    ax_intro.axis('off')

    intro_text = """
    POLYNOMIAL FEATURES INTRODUCTION:

    🎯 THE PROBLEM:
    • Real relationships are rarely linear
    • Linear models underfit curved data
    • Poor predictions and """ + r"$R^2" + """

    💡 THE SOLUTION:
    • Add polynomial terms: """ + r"$x, x^2, x^3, ...$" + """
    • Model becomes: """ + r"$y = \beta_0 + \beta_1x + \beta_2x^2 + \beta_3x^3$" + """
    • Still "linear" in parameters!

    🔄 THE TRANSFORMATION:
    Original: """ + r"$[x]$" + """
    Degree 2: """ + r"$[1, x, x^2]$" + """
    Degree 3: """ + r"$[1, x, x^2, x^3]$" + """

    ⚖️ THE TRADEOFF:
    • Higher degree → Better fit
    • But risk of overfitting!
    • Need to find sweet spot
    """

    ax_intro.text(0.05, 0.95, intro_text, #ha="left", 
                transform=ax_intro.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.7", facecolor='lightblue', alpha=0.8))

    # PHASE 1: MOTIVATION - Linear models failing on non-linear data
    #print("\nPHASE 1: MOTIVATION - Why linear models struggle")

    scenarios = ['quadratic', 'cubic', 'oscillatory']
    true_func_text = [r"$y = 1 + 2x - 3x^2 + \epsilon$", r"$y = x^3 - 2x + \epsilon$", r"$y = sin{(2\pi x)} + 0.5x +\epsilon$"]
    for i, scenario in enumerate(scenarios):
        x, y, true_func, title = generate_nonlinear_data(100, 0.3, scenario)
        
        ax = axes[0, i+1]
        
        # Plot data
        ax.scatter(x, y, alpha=0.6, s=30, label='Data')
        
        # Fit linear model
        linear_model = LinearRegression().fit(x.reshape(-1, 1), y)
        
        # Plot true function and linear fit
        x_plot = np.linspace(x.min(), x.max(), 200)
        y_true = true_func(x_plot)
        y_linear = linear_model.predict(x_plot.reshape(-1, 1))
        
        ax.plot(x_plot, y_true, 'g-', linewidth=3, label='True Function:\n' + true_func_text[i], alpha=0.8)
        ax.plot(x_plot, y_linear, 'r--', linewidth=2, label='Linear Fit', alpha=0.8)
        
        # Calculate R²
        r2_linear = linear_model.score(x.reshape(-1, 1), y)
        
        ax.set_title(f'{title} (Linear R² = {r2_linear:.3f})', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

    axes[0, 4].axis('off')

    # PHASE 3: STEP-BY-STEP POLYNOMIAL DEMONSTRATION
    #print("\nPHASE 3: STEP-BY-STEP demonstration of polynomial fitting")

    # Generate quadratic data for detailed analysis
    x_demo, y_demo, true_func_demo, _ = generate_nonlinear_data(50, 0.2, 'quadratic')

    degrees = [1, 2, 3, 5]
    colors = ['red', 'blue', 'green', 'orange']

    for i, degree in enumerate(degrees):
        ax = axes[1, i+1]
        
        # Fit polynomial model
        poly_features = PolynomialFeatures(degree=degree)
        X_poly = poly_features.fit_transform(x_demo.reshape(-1, 1))
        poly_model = LinearRegression().fit(X_poly, y_demo)
        
        # Plot data and fits
        ax.scatter(x_demo, y_demo, alpha=0.6, s=30, label='Data')
        
        x_plot = np.linspace(x_demo.min(), x_demo.max(), 200)
        X_plot_poly = poly_features.transform(x_plot.reshape(-1, 1))
        y_poly_pred = poly_model.predict(X_plot_poly)
        
        # True function
        y_true_plot = true_func_demo(x_plot)
        ax.plot(x_plot, y_true_plot, 'black', linewidth=3, label='True Function', alpha=0.7)
        
        # Polynomial fit
        ax.plot(x_plot, y_poly_pred, colors[i], linewidth=2, label=f'Degree {degree}')
        
        # Calculate metrics
        r2 = poly_model.score(X_poly, y_demo)
        
        ax.set_title(f'Polynomial Degree {degree} (R² = {r2:.3f})', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-4, 3)
        ax.tick_params(axis='x', labelsize=7)
        ax.tick_params(axis='y', labelsize=7)

    # PHASE 4: THE OVERFITTING PROBLEM
    #print("\nPHASE 4: OVERFITTING demonstration")

    # Generate training and test data
    x_train, y_train, true_func_over, _ = generate_nonlinear_data(30, 0.3, 'quadratic')
    x_test, y_test, _, _ = generate_nonlinear_data(100, 0.3, 'quadratic')

    degrees_range = range(1, 16)
    train_scores = []
    test_scores = []

    for degree in degrees_range:
        # Fit model
        poly_features = PolynomialFeatures(degree=degree)
        X_train_poly = poly_features.fit_transform(x_train.reshape(-1, 1))
        X_test_poly = poly_features.transform(x_test.reshape(-1, 1))
        
        model = LinearRegression().fit(X_train_poly, y_train)
        
        # Calculate scores
        train_score = model.score(X_train_poly, y_train)
        test_score = model.score(X_test_poly, y_test)
        
        train_scores.append(train_score)
        test_scores.append(test_score)

    # Plot bias-variance tradeoff
    ax_bias_var = axes[2, 1]
    ax_bias_var.plot(degrees_range, train_scores, 'b-o', linewidth=2, label='Training R²', markersize=4)
    ax_bias_var.plot(degrees_range, test_scores, 'r-s', linewidth=2, label='Test R²', markersize=4)

    # Mark optimal degree
    optimal_degree = degrees_range[np.argmax(test_scores)]
    ax_bias_var.axvline(optimal_degree, color='green', linestyle='--', alpha=0.7, 
                    label=f'Optimal Degree: {optimal_degree}')

    ax_bias_var.set_xlabel('Polynomial Degree', fontsize=8)
    ax_bias_var.set_ylabel('R² Score', fontsize=8)
    ax_bias_var.set_title('Bias-Variance Tradeoff (Training vs Test Performance)', fontsize=9)
    ax_bias_var.legend(fontsize=8)
    ax_bias_var.grid(True, alpha=0.3)
    ax_bias_var.tick_params(axis='x', labelsize=7)
    ax_bias_var.tick_params(axis='y', labelsize=7)

    # Show extreme overfitting case
    ax_extreme = axes[2, 2]
    x_extreme, y_extreme, true_func_extreme, _ = generate_nonlinear_data(15, 0.2, 'quadratic')

    # Fit very high degree polynomial
    poly_extreme = PolynomialFeatures(degree=12)
    X_extreme_poly = poly_extreme.fit_transform(x_extreme.reshape(-1, 1))
    model_extreme = LinearRegression().fit(X_extreme_poly, y_extreme)

    # Plot
    ax_extreme.scatter(x_extreme, y_extreme, alpha=0.8, s=50, label='Training Data', zorder=5)

    x_plot_extreme = np.linspace(x_extreme.min(), x_extreme.max(), 200)
    X_plot_extreme_poly = poly_extreme.transform(x_plot_extreme.reshape(-1, 1))
    y_pred_extreme = model_extreme.predict(X_plot_extreme_poly)

    ax_extreme.plot(x_plot_extreme, true_func_extreme(x_plot_extreme), 'g-', 
                linewidth=3, label='True Function', alpha=0.8)
    ax_extreme.plot(x_plot_extreme, y_pred_extreme, 'r-', linewidth=2, 
                label='Degree 12 Fit', alpha=0.8)

    ax_extreme.set_title('Severe Overfitting (Degree 12 on 15 points)', fontsize=9)
    ax_extreme.legend(fontsize=8)
    ax_extreme.grid(True, alpha=0.3)
    ax_extreme.set_ylim(-5, 5)
    ax_extreme.tick_params(axis='x', labelsize=7)
    ax_extreme.tick_params(axis='y', labelsize=7)

    # PHASE 5: REGULARIZATION SAVES THE DAY
    #print("\nPHASE 5: REGULARIZATION as solution to polynomial overfitting")

    # Compare unregularized vs regularized high-degree polynomial
    degree_high = 10
    poly_high = PolynomialFeatures(degree=degree_high)
    X_train_high = poly_high.fit_transform(x_train.reshape(-1, 1))
    X_test_high = poly_high.transform(x_test.reshape(-1, 1))

    # Unregularized
    model_unreg = LinearRegression().fit(X_train_high, y_train)

    # Regularized (Ridge)
    model_ridge = Ridge(alpha=1.0).fit(X_train_high, y_train)

    # Plot comparison
    ax_reg = axes[2, 3]
    ax_reg.scatter(x_train, y_train, alpha=0.6, s=30, label='Training Data')

    x_plot_reg = np.linspace(x_train.min(), x_train.max(), 200)
    X_plot_reg = poly_high.transform(x_plot_reg.reshape(-1, 1))

    y_unreg_pred = model_unreg.predict(X_plot_reg)
    y_ridge_pred = model_ridge.predict(X_plot_reg)

    ax_reg.plot(x_plot_reg, true_func_over(x_plot_reg), 'black', linewidth=3, 
            label='True Function', alpha=0.8)
    ax_reg.plot(x_plot_reg, y_unreg_pred, 'r--', linewidth=2, 
            label='Unregularized', alpha=0.7)
    ax_reg.plot(x_plot_reg, y_ridge_pred, 'b-', linewidth=2, 
            label='Ridge Regularized', alpha=0.8)

    ax_reg.set_title('Regularization Tames. High-Degree Polynomials', fontsize=9)
    ax_reg.legend(fontsize=8)
    ax_reg.grid(True, alpha=0.3)
    ax_reg.set_ylim(-3, 3)
    ax_reg.tick_params(axis='x', labelsize=7)
    ax_reg.tick_params(axis='y', labelsize=7)

    # Coefficient comparison
    ax_coef = axes[2, 4]
    coef_unreg = model_unreg.coef_
    coef_ridge = model_ridge.coef_

    x_coef = np.arange(len(coef_unreg))
    width = 0.35

    ax_coef.bar(x_coef - width/2, coef_unreg, width, label='Unregularized', alpha=0.7)
    ax_coef.bar(x_coef + width/2, coef_ridge, width, label='Ridge', alpha=0.7)

    ax_coef.set_xlabel('Coefficient Index', fontsize=8)
    ax_coef.set_ylabel('Coefficient Value', fontsize=8)
    ax_coef.set_title('Coefficient Magnitudes (Regularization Shrinks)', fontsize=9)
    ax_coef.legend(fontsize=8)
    ax_coef.grid(True, alpha=0.3)
    ax_coef.tick_params(axis='x', labelsize=7)
    ax_coef.tick_params(axis='y', labelsize=7)

    # PHASE 6: INTERACTION TERMS
    # print("\nPHASE 6: INTERACTION TERMS and multi-variable polynomials")

    # Generate 2D data with interaction
    np.random.seed(42)
    n_2d = 100
    x1_2d = np.random.uniform(-1, 1, n_2d)
    x2_2d = np.random.uniform(-1, 1, n_2d)
    y_2d = 1 + 2*x1_2d + 3*x2_2d + 4*x1_2d*x2_2d + np.random.normal(0, 0.2, n_2d)

    X_2d = np.column_stack([x1_2d, x2_2d])

    # Compare models
    models_2d = {
        'Linear Only': PolynomialFeatures(degree=1, include_bias=True),
        'With Interactions': PolynomialFeatures(degree=2, include_bias=True),
        'Full Degree 3': PolynomialFeatures(degree=3, include_bias=True)
    }

    ax_interaction = axes[3, 1]
    model_names = []
    r2_scores = []

    for name, poly_transformer in models_2d.items():
        X_poly_2d = poly_transformer.fit_transform(X_2d)
        model_2d = LinearRegression().fit(X_poly_2d, y_2d)
        r2 = model_2d.score(X_poly_2d, y_2d)
        
        model_names.append(name)
        r2_scores.append(r2)

    bars = ax_interaction.bar(model_names, r2_scores, alpha=0.7)
    ax_interaction.set_ylabel('R² Score', fontsize=8)
    ax_interaction.set_title('Interaction Terms Matter! (2D Polynomial Features)', fontsize=9)
    ax_interaction.tick_params(axis='x', #rotation=45, 
                            labelsize=7)
    ax_interaction.grid(True, alpha=0.3)

    # Add value labels
    for bar, score in zip(bars, r2_scores):
        ax_interaction.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')

    ax_interaction.tick_params(axis='y', labelsize=7)

    # Show feature expansion
    ax_features = axes[2, 0]
    ax_features.axis('off')

    feature_text = """
    POLYNOMIAL FEATURE EXPANSION:

    Original features: """ + r"$[x_1, x_2]$" + """

    Degree 1 (Linear): """ + r"$[1, x_1, x_2]$" + """

    Degree 2 (Quadratic + Interactions): """ + r"$[1, x_1, x_2, x_1^2, x_1x_2, x_2^2]$" + """

    Degree 3 (Cubic): """ + r"$[1, x_1, x_2, x_1^2, x_1x_2, x_2^2, x_1^3, x_1^2x_2, x_1x_2^2, x_2^3]$" + """

    ⚠️ CURSE OF DIMENSIONALITY:
    • Original: 2 features
    • Degree 2: 6 features  
    • Degree 3: 10 features
    • Degree d: C(d+p, d) features

    With 10 original features:
    • Degree 2: 66 features
    • Degree 3: 286 features!
    """



    ax_features.text(0.05, 0.85, feature_text, transform=ax_features.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8))

    # PHASE 7: PRACTICAL GUIDELINES
    ax_guidelines = axes[3, 2]
    ax_guidelines.axis('off')

    guidelines_text = """
    ✅ WHEN TO USE POLYNOMIAL FEATURES:
    • Non-linear relationships suspected
    • Domain knowledge suggests curves
    • Residual plots show patterns
    • Small number of features (< 10)

    ⚠️ WHEN TO BE CAREFUL:
    • High-dimensional data
    • Limited training data
    • Want interpretable models
    • Computational constraints
    """

    ax_guidelines.text(0.05, 0.95, guidelines_text, transform=ax_guidelines.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))


    ax_guidelines_2 = axes[3, 3]
    ax_guidelines_2.axis('off')

    guidelines_2_text = """
    🎯 BEST PRACTICES:
    • Start with degree 2-3
    • Always use train/validation/test split
    • Apply regularization with high degrees
    • Consider splines or other alternatives
    • Standardize features before expansion

    🔄 ALTERNATIVES TO CONSIDER:
    • Splines (smoother curves)
    • Kernel methods (implicit feature mapping)
    • Tree-based methods (automatic interactions)
    • Neural networks (learned features)
    """
    ax_guidelines_2.text(0.05, 0.95, guidelines_2_text, transform=ax_guidelines_2.transAxes, 
                    fontsize=8, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))

    # PHASE 8: REAL-WORLD EXAMPLE
    ax_real_world = axes[3, 4]

    # Simulate house price data with non-linear age effect
    np.random.seed(42)
    n_houses = 200
    house_age = np.random.uniform(0, 50, n_houses)
    # House value decreases with age, but not linearly (older houses plateau)
    house_price = (300000 - 8000*house_age + 50*house_age**2 - 0.5*house_age**3 + 
                np.random.normal(0, 15000, n_houses))

    # Fit models
    linear_house = LinearRegression().fit(house_age.reshape(-1, 1), house_price)
    poly_house = Pipeline([
        ('poly', PolynomialFeatures(degree=3)),
        ('linear', LinearRegression())
    ])
    poly_house.fit(house_age.reshape(-1, 1), house_price)

    # Plot
    ax_real_world.scatter(house_age, house_price, alpha=0.6, s=20, label='House Data')

    age_plot = np.linspace(0, 50, 200)
    price_linear = linear_house.predict(age_plot.reshape(-1, 1))
    price_poly = poly_house.predict(age_plot.reshape(-1, 1))

    ax_real_world.plot(age_plot, price_linear, 'r--', linewidth=2, label='Linear')
    ax_real_world.plot(age_plot, price_poly, 'b-', linewidth=2, label='Polynomial')

    ax_real_world.set_xlabel('House Age (years)', fontsize=8)
    ax_real_world.set_ylabel('Price ($)', fontsize=8)
    ax_real_world.set_title('Real Example: House Price vs Age (Non-linear Depreciation)', fontsize=9)
    ax_real_world.legend(fontsize=8)
    ax_real_world.grid(True, alpha=0.3)
    ax_real_world.tick_params(axis='x', labelsize=7)
    ax_real_world.tick_params(axis='y', labelsize=7)

    plt.tight_layout()
    plt.show()

    return fig

def main():
    #fig = generate_polynomials_plots()
    #get_scenarios()
    #get_poly_fit()
    get_overfit_example()
    
if __name__ == "__main__":
    main()

