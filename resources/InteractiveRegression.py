import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches
from ipywidgets import interact, FloatSlider, Button, VBox, HBox, Output, fixed
import IPython.display as display
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats

class InteractiveRegression:
    """
    Creates an interactive view to demonstrate linear regression that fits the data points and the impact of slope and intercept on errors.
    """
    def __init__(self, data_x, data_y):
        
        self.data_x = data_x
        self.data_y = data_y
        
        # Calculate optimal line parameters
        self.optimal_slope, self.optimal_intercept = self.calculate_optimal_line()
        
        # Set up the figure and axes
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        plt.subplots_adjust(bottom=0.25, left=0.1, right=0.95, top=0.9)
        
        # Initial line parameters
        self.current_slope = 0.0
        self.current_intercept = 90.0
        
        # Plot setup
        self.setup_plot()
        self.create_widgets()
        self.update_plot(None)
        
        # Show the plot
        plt.show()
    
    def calculate_optimal_line(self):
        """Calculate the optimal slope and intercept using least squares"""
        n = len(self.data_x)
        sum_x = np.sum(self.data_x)
        sum_y = np.sum(self.data_y)
        sum_xy = np.sum(self.data_x * self.data_y)
        sum_x2 = np.sum(self.data_x * self.data_x)
        
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        # Using your formula: a = Cov(X,Y) / Var(X)
        slope = (sum_xy - n * mean_x * mean_y) / (sum_x2 - n * mean_x * mean_x)
        intercept = mean_y - slope * mean_x
        
        return slope, intercept
    
    def setup_plot(self):
        """Set up the main plot area"""
        self.ax.set_xlim(0.5, 5.5)
        self.ax.set_ylim(60, 115)
        self.ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
        self.ax.set_ylabel('Height (cm)', fontsize=12, fontweight='bold')
        self.ax.set_title('Interactive Linear Regression: Age vs Height\nAdjust sliders to find the best fitting line', 
                         fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # Plot data points
        self.scatter = self.ax.scatter(self.data_x, self.data_y, c='blue', s=80, alpha=0.8, 
                                      edgecolors='white', linewidth=2, zorder=5)
        
        # Initialize line and error lines (will be updated)
        x_line = np.linspace(0.5, 5.5, 100)
        y_line = self.current_slope * x_line + self.current_intercept
        self.line, = self.ax.plot(x_line, y_line, 'r-', linewidth=3, alpha=0.8, label='Fitted Line')
        
        # Error lines will be stored here
        self.error_lines = []
        
        # Statistics text box
        self.stats_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                      fontsize=10, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def create_widgets(self):
        """Create interactive widgets"""
        # Slider axes
        ax_slope = plt.axes([0.2, 0.15, 0.5, 0.03])
        ax_intercept = plt.axes([0.2, 0.1, 0.5, 0.03])
        
        # Sliders
        self.slider_slope = Slider(ax_slope, 'Slope (a)', -10, 20, 
                                  valinit=self.current_slope, valfmt='%.1f')
        self.slider_intercept = Slider(ax_intercept, 'Intercept (b)', 40, 100, 
                                      valinit=self.current_intercept, valfmt='%.1f')
        
        # Buttons
        ax_reset = plt.axes([0.8, 0.15, 0.08, 0.04])
        ax_optimal = plt.axes([0.8, 0.1, 0.08, 0.04])
        
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_optimal = Button(ax_optimal, 'Optimal')
        
        # Connect events
        self.slider_slope.on_changed(self.update_plot)
        self.slider_intercept.on_changed(self.update_plot)
        self.button_reset.on_clicked(self.reset_line)
        self.button_optimal.on_clicked(self.show_optimal)
    
    def calculate_statistics(self, slope, intercept):
        """Calculate regression statistics"""
        # Predictions
        y_pred = slope * self.data_x + intercept
        
        # Errors
        errors = self.data_y - y_pred
        
        # Statistics
        sse = np.sum(errors**2)  # Sum of Squared Errors
        mse = sse / len(self.data_y)  # Mean Squared Error
        rmse = np.sqrt(mse)     # Root Mean Squared Error
        
        # R-squared
        y_mean = np.mean(self.data_y)
        ss_tot = np.sum((self.data_y - y_mean)**2)
        r_squared = 1 - (sse / ss_tot)
        
        return sse, mse, rmse, r_squared, y_pred
    
    def update_plot(self, val):
        """Update the plot when sliders change"""
        # Get current values
        slope = self.slider_slope.val
        intercept = self.slider_intercept.val
        
        # Update line
        x_line = np.linspace(0.5, 5.5, 100)
        y_line = slope * x_line + intercept
        self.line.set_ydata(y_line)
        
        # Remove old error lines
        for line in self.error_lines:
            line.remove()
        self.error_lines = []
        
        # Calculate statistics and predictions
        sse, mse, rmse, r_squared, y_pred = self.calculate_statistics(slope, intercept)
        
        # Draw error lines
        for i in range(len(self.data_x)):
            error_line = self.ax.plot([self.data_x[i], self.data_x[i]], [self.data_y[i], self.y_pred[i]], 
                                    'r--', alpha=0.6, linewidth=1)[0]
            self.error_lines.append(error_line)
        
        # Update statistics text
        stats_str = f'''Current Line: y = {slope:.1f}x + {intercept:.1f}
        
Statistics:
SSE = {sse:.2f}
MSE = {mse:.2f}
RMSE = {rmse:.2f}
R² = {r_squared:.4f}

Optimal Line: y = {self.optimal_slope:.1f}x + {self.optimal_intercept:.1f}'''
        
        self.stats_text.set_text(stats_str)
        
        # Redraw
        self.fig.canvas.draw()
    
    def reset_line(self, event):
        """Reset to initial values"""
        self.slider_slope.reset()
        self.slider_intercept.reset()
    
    def show_optimal(self, event):
        """Show the optimal line"""
        self.slider_slope.set_val(self.optimal_slope)
        self.slider_intercept.set_val(self.optimal_intercept)
        

def demonstrate_least_squares_derivation(data_x, data_y):
    """
    Demonstrate the mathematical derivation of least squares solution
    """
    print("=== LEAST SQUARES DERIVATION ===")
    print(f"Data points: {len(data_x)} observations")
    print(f"Age (X): {data_x}")
    print(f"Height (Y): {data_y}")
    print()
    
    n = len(data_x)
    sum_x = np.sum(data_x)
    sum_y = np.sum(data_y)
    sum_xy = np.sum(data_x * data_y)
    sum_x2 = np.sum(data_x * data_x)
    
    mean_x = sum_x / n
    mean_y = sum_y / n
    
    print(f"n = {n}")
    print(f"Σx = {sum_x:.2f}")
    print(f"Σy = {sum_y:.2f}")
    print(f"Σxy = {sum_xy:.2f}")
    print(f"Σx² = {sum_x2:.2f}")
    print(f"x̄ = {mean_x:.2f}")
    print(f"ȳ = {mean_y:.2f}")
    print()
    
    # Calculate covariance and variance
    cov_xy = (sum_xy - n * mean_x * mean_y) / n
    var_x = (sum_x2 - n * mean_x * mean_x) / n
    
    print(f"Cov(X,Y) = {cov_xy:.2f}")
    print(f"Var(X) = {var_x:.2f}")
    print()
    
    # Calculate slope and intercept using your formulas
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    
    print(f"a = Cov(X,Y) / Var(X) = {slope:.2f}")
    print(f"b = ȳ - a·x̄ = {intercept:.2f}")
    print()
    print(f"OPTIMAL LINE: y = {slope:.2f}x + {intercept:.2f}")
    

def create_static_visualization(data_x, data_y, x_label_text='Age (years)', y_label_text='Height (cm)'):
    """
    Create a static visualization showing the optimal fit
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Calculate optimal line
    slope, intercept = calculate_optimal_line(data_x, data_y)
    x_min = data_x.min() - 0.5
    x_max = data_x.max() + 0.5
    x_line = np.linspace(x_min, x_max, 100)
    y_line = slope * x_line + intercept
    y_pred = slope * data_x + intercept
    
    # Plot 1: Data with optimal line and errors
    ax1.scatter(data_x, data_y, c='blue', s=80, alpha=0.8, edgecolors='white', linewidth=2)
    ax1.plot(x_line, y_line, 'r-', linewidth=3, alpha=0.8, label=f'y = {slope:.1f}x + {intercept:.1f}')
    
    # Draw error lines
    for i in range(len(data_x)):
        ax1.plot([data_x[i], data_x[i]], [data_y[i], y_pred[i]], 'r--', alpha=0.6, linewidth=1)
    
    y_min = data_y.min() - 10
    y_max = data_y.max() + 10
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel(x_label_text)
    ax1.set_ylabel(y_label_text)
    ax1.set_title('Optimal Linear Regression Fit')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Residuals
    residuals = data_y - y_pred
    ax2.scatter(data_x, residuals, c='red', s=80, alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax2.set_xlabel(x_label_text)
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def calculate_optimal_line(data_x, data_y):
    """Standalone function to calculate optimal line parameters"""
    n = len(data_x)
    sum_x = np.sum(data_x)
    sum_y = np.sum(data_y)
    sum_xy = np.sum(data_x * data_y)
    sum_x2 = np.sum(data_x * data_x)
    
    mean_x = sum_x / n
    mean_y = sum_y / n
    
    slope = (sum_xy - n * mean_x * mean_y) / (sum_x2 - n * mean_x * mean_x)
    intercept = mean_y - slope * mean_x
    
    return slope, intercept

def calculate_statistics(data_x, data_y, slope, intercept):
    """Calculate regression statistics"""
    y_pred = slope * data_x + intercept
    errors = data_y - y_pred
    
    sse = np.sum(errors**2)
    mse = sse / len(data_y)
    rmse = np.sqrt(mse)
    
    y_mean = np.mean(data_y)
    ss_tot = np.sum((data_y - y_mean)**2)
    r_squared = 1 - (sse / ss_tot)
    
    return sse, mse, rmse, r_squared, y_pred 

def plot_regression(data_x, data_y, slope=0.0, intercept=90.0):
    """Plot the regression with current parameters"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate predictions and statistics
    sse, mse, rmse, r_squared, y_pred = calculate_statistics(data_x, data_y, slope, intercept)
    
    # Plot data points
    ax.scatter(data_x, data_y, c='blue', s=100, alpha=0.8, 
               edgecolors='white', linewidth=2, zorder=5, label='Data points')
    
    # Plot regression line
    x_line = np.linspace(0.5, 5.5, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=3, alpha=0.8, 
            label=f'y = {slope:.1f}x + {intercept:.1f}')
    
    # Plot error lines
    for i in range(len(data_x)):
        ax.plot([data_x[i], data_x[i]], [data_y[i], y_pred[i]], 
                'r--', alpha=0.6, linewidth=1)
    
    # Formatting
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(60, 115)
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height (cm)', fontsize=12, fontweight='bold')
    ax.set_title('Interactive Linear Regression: Age vs Height', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics text
    stats_text = f'''Statistics:
SSE = {sse:.2f}
MSE = {mse:.2f}
RMSE = {rmse:.2f}
R² = {r_squared:.4f}'''
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def create_interactive_regression(data_x, data_y):
    """Create the interactive widget"""
    # Calculate optimal values for reference
    optimal_slope, optimal_intercept = calculate_optimal_line(data_x, data_y)
    
    print(f"Optimal line: y = {optimal_slope:.2f}x + {optimal_intercept:.2f}")
    print("Use the sliders below to find the best fitting line!")
    print("Try to minimize the Sum of Squared Errors (SSE)")
    print("-" * 50)
    
    # Create interactive plot
    interact(plot_regression,
             data_x=fixed(data_x),
             data_y=fixed(data_y), 
             slope=FloatSlider(value=0.0, min=-10.0, max=20.0, step=0.1, 
                              description='Slope (a):'),
             intercept=FloatSlider(value=90.0, min=40.0, max=100.0, step=0.1, 
                                 description='Intercept (b):'))

# Alternative: Manual step-by-step approach
def demonstrate_step_by_step(data_x, data_y):
    """Step-by-step demonstration without widgets"""
    optimal_slope, optimal_intercept = calculate_optimal_line(data_x, data_y)
    
    print("STEP-BY-STEP LINEAR REGRESSION DEMONSTRATION")
    print("=" * 50)
    
    # Step 1: Show initial horizontal line
    print("\nStep 1: Initial horizontal line (y = 90)")
    plot_regression(data_x, data_y, slope=0.0, intercept=90.0)
    
    # Step 2: Show improved line with some slope
    print("\nStep 2: Adding some slope (y = 5x + 75)")
    plot_regression(data_x, data_y, slope=5.0, intercept=75.0)
    
    # Step 3: Show better line
    print("\nStep 3: Better approximation (y = 8x + 65)")
    plot_regression(data_x, data_y, slope=8.0, intercept=65.0)
    
    # Step 4: Show optimal line
    print(f"\nStep 4: Optimal line (y = {optimal_slope:.1f}x + {optimal_intercept:.1f})")
    plot_regression(data_x, data_y, slope=optimal_slope, intercept=optimal_intercept)
    
    print(f"\nThe optimal line minimizes the sum of squared errors!")
    print(f"Mathematical solution: a = {optimal_slope:.3f}, b = {optimal_intercept:.3f}")

def compare_lines(data_x, data_y, lines_to_compare):
    """Compare multiple lines on the same plot"""
    
    
    if lines_to_compare is None:
        optimal_slope, optimal_intercept = calculate_optimal_line(data_x, data_y)
        lines_to_compare = [
            (0, 90, "Baseline"),
            (2, 85, "Gentle slope"),
            (5, 75, "Medium slope"), 
            (10, 60, "Steep slope"),
            (15, 45, "Too steep"),
            (optimal_slope, optimal_intercept, "Optimal")
        ]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data points
    ax.scatter(data_x, data_y, c='blue', s=100, alpha=0.8, 
               edgecolors='white', linewidth=2, zorder=5, label='Data points')
    
    colors = ['red', 'green', 'orange', 'purple', 'yellow', 'cyan', 'brown']
    x_line = np.linspace(0.5, 5.5, 100)
    
    print("Comparison of different lines:")
    
    for i, (slope, intercept, label) in enumerate(lines_to_compare):
        y_line = slope * x_line + intercept
        color = colors[i % len(colors)]
        ax.plot(x_line, y_line, color=color, linewidth=2, alpha=0.8, 
                label=f'{label}: y = {slope:.1f}x + {intercept:.1f}')
        
        # Calculate statistics
        sse, mse, rmse, r_squared, _ = calculate_statistics(data_x, data_y, slope, intercept)
        print(f"{label}: SSE = {sse:.1f}, R² = {r_squared:.3f}")
    
    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(60, 115)
    ax.set_xlabel('Age (years)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Height (cm)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Different Regression Lines', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
def plot_error_example(data_x, data_y, slope=0, intercept=90):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot data points
    ax.scatter(data_x, data_y, c='blue', s=100, alpha=0.8, 
               edgecolors='white', linewidth=2, zorder=5, label='Data points')
    
    # Plot regression line
    x_line = np.linspace(0.5, 5.5, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r-', linewidth=3, alpha=0.8, 
            label=f'y = {slope:.1f}x + {intercept:.1f}')
    
    # Plot error lines    
    y_pred = slope * data_x + intercept
    for i in range(len(data_x)):
        ax.plot([data_x[i], data_x[i]], [data_y[i], y_pred[i]], 
                'r--', alpha=0.6, linewidth=1)
        
    # Add annotation for the 4th data point (index 3)
    if len(data_x) > 3:  # Make sure we have at least 4 points
        x_4th = data_x[3]
        y_4th_actual = data_y[3]
        y_4th_pred = y_pred[3]
        
        # Calculate the midpoint of the error line for brace placement
        mid_y = (y_4th_actual + y_4th_pred) / 2
        
        # Draw a wheat-colored arrow pointing to the error line
        arrow_offset = 0.15  # Horizontal offset from the error line
        
        # Add text annotation with wheat-colored arrow
        ax.annotate('The distance between\n' + r'the line $y = ax + b$' + '\nand the 4th data point,\n' + r'$(ax_4 + b)-y_4$', 
                    xy=(x_4th, mid_y),  # Point to the middle of the error line
                    xytext=(x_4th + arrow_offset + 0.3, y_4th_actual - 5),  # Text position
                    fontsize=10,
                    ha='left',
                    va='center',
                    color='black',
                    bbox=dict(boxstyle="round, pad=0.3", facecolor="wheat", alpha=0.8),
                    arrowprops=dict(arrowstyle='->', lw=2, color='wheat'))
    
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Height (cm)', fontsize=12)
    ax.set_title("Age vs Height")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

def estimate_3d_regression_parameters(data_x_age, data_x_weight, data_y_height):
    """
    Estimate optimal parameters for 3D regression using normal equation
    Height = β₀ + β₁(Age) + β₂(Weight)
    
    Returns: β₀ (intercept), β₁ (age coefficient), β₂ (weight coefficient)
    """
    import numpy as np
    
    # Create design matrix X with intercept column
    n = len(data_x_age)
    X = np.column_stack([
        np.ones(n),        # β₀ (intercept)
        data_x_age,        # β₁ (age coefficient)
        data_x_weight      # β₂ (weight coefficient)
    ])
    
    # Response vector
    y = np.array(data_y_height)
    
    # Normal equation: β = (X'X)⁻¹X'y
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    
    beta = XtX_inv @ Xty
    
    return beta[0], beta[1], beta[2]  # β₀, β₁, β₂

    
def plot_3d_regression(data_x, data_y, data_z):
    
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot data points
    age = data_x
    height = data_y
    weight = data_z 
    
    # Estimate optimal parameters
    beta0, beta1, beta2 = estimate_3d_regression_parameters(age, weight, height)
        
    ax.scatter(age, weight, height, c='blue', s=100)
    
    # Create plane
    age_range = np.linspace(1, 5, 10)
    weight_range = np.linspace(12, 24, 10)
    AGE, WEIGHT = np.meshgrid(age_range, weight_range)
    HEIGHT_PRED = beta0 + beta1*AGE + beta2*WEIGHT
    y_pred = beta0 + beta1*age + beta2*weight 
    
    r_squared, mse, rmse, sse = calculate_stats_from_residuals(height, y_pred)
    
    # Plot plane
    ax.plot_surface(AGE, WEIGHT, HEIGHT_PRED, alpha=0.3, 
                    label=f'f = {beta0:.1f} + {beta1:.1f}age + {beta2:.1f}weight')
    
        # Add statistics text
    stats_text = f'''Statistics:
SSE = {sse:.2f}
MSE = {mse:.2f}
RMSE = {rmse:.2f}
R² = {r_squared:.4f}'''
    
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax.set_xlabel('Age')
    ax.set_ylabel('Weight') 
    ax.set_zlabel('Height', labelpad=1)
    ax.legend()
    plt.subplots_adjust(left=0.4)
    plt.tight_layout()
    
def calculate_stats_from_residuals(y_observed, y_pred):
    """Calculates statistics (R-squared, MSE, RMSE) given observed and predicted values.

    Args:
        y_observed (_type_): observed values
        y_pred (_type_): predicted values
    """
    # calculate residuals
    residuals = y_observed - y_pred 
    # Calculate R²
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_observed - np.mean(y_observed)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate other statistics
    mse = ss_res / len(y_observed)
    rmse = np.sqrt(mse)
    
    return r_squared, mse, rmse, ss_res

def calculate_confidence_intervals(X: np.array, y: np.array, confidence_level=0.95) -> np.array:
    """_summary_

    Args:
        X (np.array): _description_
        y (np.array): _description_
        confidence_level (float, optional): _description_. Defaults to 0.95.

    Returns:
        np.array: _description_
    """
    # Add intercept column
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    
    # Calculate coefficients
    beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ (X_with_intercept.T @ y)
    
    # Calculate residuals and error variance
    y_pred = X_with_intercept @ beta
    residuals = y - y_pred
    n, p = X_with_intercept.shape
    sigma_squared = np.sum(residuals**2) / (n - p)
    
    # Calculate standard errors
    var_covar_matrix = sigma_squared * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
    standard_errors = np.sqrt(np.diag(var_covar_matrix))
    
    # Calculate confidence intervals
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, n - p)
    
    margin_error = t_critical * standard_errors
    ci_lower = beta - margin_error
    ci_upper = beta + margin_error
    
    return beta, standard_errors, ci_lower, ci_upper

def plot_regression_with_intervals(x, y, confidence_level=0.95):
    """Plot regression line with confidence and prediction intervals"""
    
    # Prepare data
    X = np.column_stack([np.ones(len(x)), x])  # Add intercept
    
    # Calculate regression coefficients
    beta = np.linalg.inv(X.T @ X) @ (X.T @ y)
    y_pred = X @ beta
    
    # Calculate residual standard error
    residuals = y - y_pred
    n, p = X.shape
    sigma_hat = np.sqrt(np.sum(residuals**2) / (n - p))
    
    # Create prediction points
    x_new = np.linspace(x.min(), x.max(), 100)
    X_new = np.column_stack([np.ones(len(x_new)), x_new])
    y_new_pred = X_new @ beta
    
    # Calculate standard errors for new predictions
    var_covar_matrix = np.linalg.inv(X.T @ X)
    se_pred = []
    
    for i in range(len(x_new)):
        x0 = X_new[i:i+1, :]  # Single row
        se_mean = sigma_hat * np.sqrt(x0 @ var_covar_matrix @ x0.T)[0,0]
        se_pred.append(se_mean)
    
    se_pred = np.array(se_pred)
    
    # Critical t-value
    alpha = 1 - confidence_level
    t_critical = stats.t.ppf(1 - alpha/2, n - p)
    
    # Calculate intervals
    margin_conf = t_critical * se_pred  # Confidence interval
    margin_pred = t_critical * np.sqrt(sigma_hat**2 + se_pred**2)  # Prediction interval
    
    conf_lower = y_new_pred - margin_conf
    conf_upper = y_new_pred + margin_conf
    pred_lower = y_new_pred - margin_pred
    pred_upper = y_new_pred + margin_pred
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Data points
    plt.scatter(x, y, color='blue', s=60, alpha=0.7, zorder=5, label='Data points')
    
    # Regression line
    plt.plot(x_new, y_new_pred, 'red', linewidth=2, label='Fitted line')
    
    # Confidence interval
    plt.fill_between(x_new, conf_lower, conf_upper, alpha=0.3, color='red', 
                     label=f'{confidence_level:.0%} Confidence interval')
    
    # Prediction interval
    plt.fill_between(x_new, pred_lower, pred_upper, alpha=0.2, color='gray',
                     label=f'{confidence_level:.0%} Prediction interval')
    
    plt.xlabel('Age (years)')
    plt.ylabel('Height (cm)')
    plt.title('Linear Regression with Confidence and Prediction Intervals')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add equation and statistics
    r_squared = 1 - np.sum(residuals**2) / np.sum((y - np.mean(y))**2)
    equation_text = f'y = {beta[1]:.2f}x + {beta[0]:.2f}\nR² = {r_squared:.3f}'
    plt.text(0.05, 0.95, equation_text, transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
             verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    return beta, sigma_hat, r_squared