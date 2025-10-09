import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import comb
import pandas as pd
from ipywidgets import interact, FloatSlider, IntSlider
import warnings
warnings.filterwarnings('ignore')
import math
from scipy.stats import t

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def generate_click_data(py=False):
    
    np.random.seed(42)
    clicks_data = np.random.poisson(lam=2, size=1000)

    print(f"\nData summary: Mean = {np.mean(clicks_data):.2f}, Variance = {np.var(clicks_data):.2f}")
    print("Keep these statistics in mind - they'll help you identify the distribution!\n")

    fig = plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.hist(clicks_data, bins=range(0, max(clicks_data)+1), density=True, 
            alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Clicks per Day')
    plt.ylabel('Probability Density')
    plt.title('Mystery Distribution: User Click Data')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(clicks_data, bins=range(0, max(clicks_data)+1), density=True, 
            alpha=0.7, edgecolor='black', cumulative=True)
    plt.xlabel('Number of Clicks per Day')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
 
    return fig

def visualize_rv_concept(py=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sample space visualization
    outcomes = ['Heads', 'Tails']
    values = [1, 0]
    colors = ['gold', 'silver']
    
    ax1.bar(outcomes, [1, 1], color=colors, alpha=0.6, edgecolor='black')
    ax1.set_ylabel('Outcome Space Ω')
    ax1.set_title('Abstract Sample Space', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.5)
    
    # Random variable mapping
    ax2.bar(values, [0.5, 0.5], color=colors, alpha=0.6, edgecolor='black')
    ax2.set_xlabel('Value')
    ax2.set_ylabel('Probability P(X=x)')
    ax2.set_title('Random Variable X: Ω → R', fontsize=14, fontweight='bold')
    ax2.set_xticks([0, 1])
    ax2.set_ylim(0, 0.7)
    
    # Add arrow annotations
    fig.text(0.5, 0.2, '→', fontsize=40, ha='center', va='center')
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

def discrete_vs_cont_rv(py=False):
    # Visualization: Discrete vs Continuous
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Discrete example
    x_discrete = np.arange(0, 11)
    pmf_discrete = stats.binom.pmf(x_discrete, n=10, p=0.5)
    ax1.stem(x_discrete, pmf_discrete, basefmt=' ')
    ax1.set_xlabel('Value x')
    ax1.set_ylabel('P(X = x)')
    ax1.set_title('Discrete R.V.: Probability Mass Function (PMF)', 
                fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Continuous example preview
    x_continuous = np.linspace(-4, 4, 200)
    pdf_continuous = stats.norm.pdf(x_continuous, 0, 1)
    ax2.fill_between(x_continuous, pdf_continuous, alpha=0.3)
    ax2.plot(x_continuous, pdf_continuous, color='salmon', linewidth=2)
    ax2.set_xlabel('Value x')
    ax2.set_ylabel('f(x) - Density')
    ax2.set_title('Continuous R.V.: Probability Density Function (PDF)', 
                fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def coffee_example(py=False):
    # Example: Coffee consumption (from your French materials, adapted)
    coffee_values = np.array([1, 2, 3, 4, 5])
    coffee_probs = np.array([0.4, 0.25, 0.2, 0.1, 0.05])

    print("\nExample: Number of coffees before noon")
    print(pd.DataFrame({
        'Coffees (x)': coffee_values,
        'P(X=x)': coffee_probs
    }))

    # Verify it's a valid PMF
    print(f"\nVerification: Σp(x) = {coffee_probs.sum():.2f} ✓")
    print(f"\nVerification: p(xᵢ) ≥ 0: {all(coffee_probs) >= 0} ✓")

    # Visualize PMF
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.stem(coffee_values, coffee_probs, basefmt=' ')
    plt.xlabel('Number of Coffees (x)')
    plt.ylabel('P(X = x)')
    plt.xticks(coffee_values)
    plt.title('PMF: Coffee Consumption', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Cumulative Distribution Function (CDF)
    coffee_cdf = np.cumsum(coffee_probs)
    plt.subplot(1, 2, 2)
    plt.step(coffee_values, coffee_cdf, where='post', linewidth=2)
    plt.scatter(coffee_values, coffee_cdf, s=100, zorder=5)
    plt.xlabel('Number of Coffees (x)')
    plt.ylabel('F(x) = P(X ≤ x)')
    plt.xticks(coffee_values)
    plt.title('CDF: Cumulative Distribution', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def coffee_example_mean_var(py=False):
    coffee_values = np.array([1, 2, 3, 4, 5])
    coffee_probs = np.array([0.4, 0.25, 0.2, 0.1, 0.05])
    # Calculate for coffee example
    expectation = np.sum(coffee_values * coffee_probs)
    variance = np.sum((coffee_values - expectation)**2 * coffee_probs)
    std_dev = np.sqrt(variance)

    print(f"\nCoffee example:")
    print(f"E[X] = {expectation:.2f} coffees")
    print(f"Var(X) = {variance:.2f}")
    print(f"Std(X) = {std_dev:.2f} coffees")

    # Visualize expectation and variance
    fig = plt.figure(figsize=(10, 5))
    plt.stem(coffee_values, coffee_probs, basefmt=' ', label='PMF')
    plt.axvline(expectation, color='red', linestyle='--', linewidth=2, 
                label=f'E[X] = {expectation:.2f}')
    plt.axvspan(expectation - std_dev, expectation + std_dev, 
                alpha=0.2, color='orange', label=f'±1 Std Dev')
    plt.xlabel('Number of Coffees (x)')
    plt.ylabel('P(X = x)')
    plt.title('Expectation and Standard Deviation', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def demo_markov_ineq(py=False):
    # Simulate this scenario
    np.random.seed(42)

    # Typical training times (exponential-ish distribution)
    training_times = np.random.exponential(scale=10, size=10000)

    a_threshold = 50
    actual_prob = np.mean(training_times >= a_threshold)
    markov_bound = np.mean(training_times) / a_threshold

    print(f"Average training time: {np.mean(training_times):.2f} min")
    print(f"Actual P(time ≥ {a_threshold}): {actual_prob:.4f}")
    print(f"Markov bound: {markov_bound:.4f}")
    print(f"Bound is valid: {actual_prob <= markov_bound}")

    # Visualize
    fig = plt.figure(figsize=(10, 5))
    plt.hist(training_times, bins=50, density=True, alpha=0.6, edgecolor='black')
    plt.axvline(np.mean(training_times), color='blue', linestyle='--', 
                linewidth=2, label=f'Mean = {np.mean(training_times):.1f} min')
    plt.axvline(a_threshold, color='red', linestyle='-', linewidth=2, 
                label=f'Threshold a = {a_threshold} min')
    plt.fill_betweenx([0, plt.ylim()[1]], a_threshold, plt.xlim()[1], 
                    alpha=0.2, color='red', 
                    label=f'P(X≥{a_threshold}) ≤ {markov_bound:.2f}')
    plt.xlabel('Training Time (minutes)')
    plt.ylabel('Density')
    plt.title('Markov\'s Inequality: Bounding Long Training Times')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig
    
def demo_markov_ineq_2(py=False):
    # Demonstrate Markov's inequality visually
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Example 1: Exponential distribution
    x = np.linspace(0, 10, 1000)
    lam = 1
    pdf = lam * np.exp(-lam * x)
    mean = 1/lam  # E[X] = 1

    a_values = [2, 3, 4]
    colors = ['red', 'orange', 'purple']

    axes[0].fill_between(x, pdf, alpha=0.3, label='PDF')
    axes[0].axvline(mean, color='blue', linestyle='--', linewidth=2, label=f'E[X] = {mean}')

    for a, color in zip(a_values, colors):
        # Actual probability
        actual_prob = np.exp(-lam * a)
        # Markov bound
        markov_bound = mean / a
        
        # Shade region X ≥ a
        mask = x >= a
        axes[0].fill_between(x[mask], pdf[mask], alpha=0.5, color=color, 
                            label=f'P(X≥{a})={actual_prob:.3f} ≤ {markov_bound:.3f}')
        axes[0].axvline(a, color=color, linestyle=':', linewidth=2)

    axes[0].set_xlabel('x', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].set_title('Markov\'s Inequality: Upper Bound on Tail Probability', 
                    fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 8)

    # Example 2: Tightness comparison
    a_range = np.linspace(0.1, 5, 100)
    mean = 1

    # For exponential: actual P(X ≥ a) = exp(-a)
    actual_probs = np.exp(-a_range)
    markov_bounds = mean / a_range

    axes[1].plot(a_range, actual_probs, 'b-', linewidth=2, label='Actual P(X≥a)')
    axes[1].plot(a_range, markov_bounds, 'r--', linewidth=2, label='Markov Bound: E[X]/a')
    axes[1].fill_between(a_range, actual_probs, markov_bounds, alpha=0.2, color='yellow',
                        label='Gap (bound looseness)')
    axes[1].set_xlabel('Threshold a', fontsize=12)
    axes[1].set_ylabel('Probability', fontsize=12)
    axes[1].set_title('Markov Bound vs Reality: Always an Upper Bound', 
                    fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def demo_schebyshev_ineq(py=False):
    # Visualize Chebyshev's inequality
    fig, axes = plt.subplots(1, 1, figsize=(8, 5))
    
    mu, sigma = 75, 5
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = stats.norm.pdf(x, mu, sigma)

    k_values = [2, 3]
    colors = ['red', 'orange']

    axes.fill_between(x, pdf, alpha=0.3, label='Distribution')
    axes.axvline(mu, color='blue', linestyle='--', linewidth=2, 
                    label=f'Mean μ={mu}')

    for k, color in zip(k_values, colors):
        alpha = k * sigma
        # Actual probability for normal
        actual_prob = 2 * (1 - stats.norm.cdf(mu + alpha, mu, sigma))
        # Chebyshev bound
        chebyshev_bound = 1 / (k**2)
        
        # Shade tails
        mask_left = x <= mu - alpha
        mask_right = x >= mu + alpha
        axes.fill_between(x[mask_left], pdf[mask_left], alpha=0.4, color=color)
        axes.fill_between(x[mask_right], pdf[mask_right], alpha=0.4, color=color,
                            label=f'|X-μ|≥{k}σ: P={actual_prob:.4f} ≤ {chebyshev_bound:.4f}')
        axes.axvline(mu - alpha, color=color, linestyle=':', linewidth=2)
        axes.axvline(mu + alpha, color=color, linestyle=':', linewidth=2)

    axes.set_xlabel('Score', fontsize=11)
    axes.set_ylabel('Density', fontsize=11)
    axes.set_title('Chebyshev on Grades Distribution\n(Bound is loose)', 
                        fontsize=12, fontweight='bold')
    axes.legend(fontsize=9)
    axes.grid(True, alpha=0.3)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def demo_chebyshev_ineq_2(py=False):
    # Visualize Chebyshev's inequality
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Example 1: Normal distribution (Chebyshev is loose here)
    mu, sigma = 75, 5
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    pdf = stats.norm.pdf(x, mu, sigma)

    k_values = [2, 3]
    colors = ['red', 'orange']

    axes[0, 0].fill_between(x, pdf, alpha=0.3, label='Distribution')
    axes[0, 0].axvline(mu, color='blue', linestyle='--', linewidth=2, 
                    label=f'Mean μ={mu}')

    for k, color in zip(k_values, colors):
        alpha = k * sigma
        # Actual probability for normal
        actual_prob = 2 * (1 - stats.norm.cdf(mu + alpha, mu, sigma))
        # Chebyshev bound
        chebyshev_bound = 1 / (k**2)
        
        # Shade tails
        mask_left = x <= mu - alpha
        mask_right = x >= mu + alpha
        axes[0, 0].fill_between(x[mask_left], pdf[mask_left], alpha=0.4, color=color)
        axes[0, 0].fill_between(x[mask_right], pdf[mask_right], alpha=0.4, color=color,
                            label=f'|X-μ|≥{k}σ: P={actual_prob:.4f} ≤ {chebyshev_bound:.4f}')
        axes[0, 0].axvline(mu - alpha, color=color, linestyle=':', linewidth=2)
        axes[0, 0].axvline(mu + alpha, color=color, linestyle=':', linewidth=2)

    axes[0, 0].set_xlabel('Score', fontsize=11)
    axes[0, 0].set_ylabel('Density', fontsize=11)
    axes[0, 0].set_title('Chebyshev on Normal Distribution\n(Bound is loose)', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Example 2: Uniform distribution (Chebyshev is tighter)
    a_unif, b_unif = 65, 85
    mu_unif = (a_unif + b_unif) / 2
    var_unif = ((b_unif - a_unif)**2) / 12
    sigma_unif = np.sqrt(var_unif)

    x_unif = np.linspace(60, 90, 1000)
    pdf_unif = np.where((x_unif >= a_unif) & (x_unif <= b_unif), 1/(b_unif-a_unif), 0)

    axes[0, 1].fill_between(x_unif, pdf_unif, alpha=0.3, label='Uniform Distribution')
    axes[0, 1].axvline(mu_unif, color='blue', linestyle='--', linewidth=2, 
                    label=f'Mean μ={mu_unif:.1f}')

    for k, color in zip([1.5, 2], colors):
        alpha = k * sigma_unif
        # Actual probability for uniform
        left_tail = max(0, (a_unif - (mu_unif - alpha))) / (b_unif - a_unif)
        right_tail = max(0, ((mu_unif + alpha) - b_unif)) / (b_unif - a_unif)
        actual_prob = left_tail + right_tail
        chebyshev_bound = 1 / (k**2)
        
        axes[0, 1].axvline(mu_unif - alpha, color=color, linestyle=':', linewidth=2)
        axes[0, 1].axvline(mu_unif + alpha, color=color, linestyle=':', linewidth=2,
                        label=f'|X-μ|≥{k:.1f}σ: P={actual_prob:.4f} ≤ {chebyshev_bound:.4f}')

    axes[0, 1].set_xlabel('Value', fontsize=11)
    axes[0, 1].set_ylabel('Density', fontsize=11)
    axes[0, 1].set_title('Chebyshev on Uniform Distribution\n(Tighter bound)', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(60, 90)

    # Example 3: k-sigma rule visualization
    k_range = np.linspace(1, 5, 100)
    chebyshev_bound = 1 / (k_range**2)
    normal_actual = 2 * (1 - stats.norm.cdf(k_range))

    axes[1, 0].plot(k_range, chebyshev_bound, 'r-', linewidth=2.5, 
                label='Chebyshev Bound: 1/k²')
    axes[1, 0].plot(k_range, normal_actual, 'b--', linewidth=2, 
                label='Normal (actual)')
    axes[1, 0].fill_between(k_range, normal_actual, chebyshev_bound, 
                            alpha=0.2, color='yellow', label='Gap')

    # Add specific k values
    for k in [1, 2, 3]:
        cheb = 1/k**2
        norm = 2 * (1 - stats.norm.cdf(k))
        axes[1, 0].scatter([k], [cheb], s=100, color='red', zorder=5)
        axes[1, 0].scatter([k], [norm], s=100, color='blue', zorder=5)
        axes[1, 0].text(k, cheb + 0.05, f'k={k}\n{cheb:.3f}', 
                    ha='center', fontsize=9, color='red')

    axes[1, 0].set_xlabel('k (number of standard deviations)', fontsize=11)
    axes[1, 0].set_ylabel('P(|X-μ| ≥ kσ)', fontsize=11)
    axes[1, 0].set_title('Chebyshev Bound vs Reality', fontsize=12, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(1, 5)
    axes[1, 0].set_ylim(0, 1)

    # Example 4: The "rule of thumb" table
    k_vals = np.array([1, 2, 3, 4, 5])
    chebyshev_vals = 1 / (k_vals**2)
    normal_vals = 2 * (1 - stats.norm.cdf(k_vals))

    axes[1, 1].axis('off')
    table_data = []
    for k, cheb, norm in zip(k_vals, chebyshev_vals, normal_vals):
        table_data.append([f'{k}σ', f'{(1-cheb)*100:.1f}%', f'{(1-norm)*100:.2f}%'])

    table = axes[1, 1].table(cellText=table_data,
                            colLabels=['Distance', 'Chebyshev\n(≥% within)', 'Normal\n(actual)'],
                            cellLoc='center',
                            loc='center',
                            bbox=[0.1, 0.2, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    axes[1, 1].text(0.5, 0.9, 'Chebyshev\'s Guarantees', 
                ha='center', fontsize=13, fontweight='bold', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.5, 0.05, 'At least X% of data within k standard deviations', 
                ha='center', fontsize=9, style='italic', transform=axes[1, 1].transAxes)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig


def demo_pdf_cdf_discrete(py=False):
    # Example: Number of successful API calls out of 5 attempts
    # X ~ Binomial(n=5, p=0.6)
    n_trials = 5
    p_success = 0.6

    x_discrete = np.arange(0, n_trials + 1)
    pmf_discrete = stats.binom.pmf(x_discrete, n_trials, p_success)
    cdf_discrete = stats.binom.cdf(x_discrete, n_trials, p_success)

    print(f"\nExample: X ~ Binomial(n={n_trials}, p={p_success})")
    print("X = number of successful API calls out of 5 attempts\n")

    # Create probability table
    print("Probability Distribution:")
    print("─" * 50)
    print(f"{'x':<5} {'P(X=x)':<12} {'F(x)=P(X≤x)':<15}")
    print("─" * 50)
    for x_val, pmf_val, cdf_val in zip(x_discrete, pmf_discrete, cdf_discrete):
        print(f"{x_val:<5} {pmf_val:<12.4f} {cdf_val:<15.4f}")
    print("─" * 50)

    # Visualization: Building the CDF step by step
    fig_1, axes = plt.subplots(2, 3, figsize=(16, 9))

    for idx in range(6):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Plot PMF as stems
        ax.stem(x_discrete, pmf_discrete, linefmt='b-', markerfmt='bo', 
                basefmt=' ', label='PMF p(x)')
        
        # Highlight accumulated probability up to current point
        current_x = idx
        accumulated_x = x_discrete[x_discrete <= current_x]
        accumulated_pmf = pmf_discrete[x_discrete <= current_x]
        
        # Shade the accumulated bars
        for x_val, pmf_val in zip(accumulated_x, accumulated_pmf):
            ax.bar(x_val, pmf_val, width=0.4, alpha=0.5, color='lightblue',
                edgecolor='blue', linewidth=2)
        
        # Calculate and display CDF value
        cdf_value = np.sum(accumulated_pmf)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(current_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.text(0.05, 0.95, f'F({current_x}) = P(X≤{current_x})\n= {cdf_value:.4f}', 
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('x (number of successes)', fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(f'CDF at x = {current_x}', fontsize=11, fontweight='bold')
        ax.set_xticks(x_discrete)
        ax.set_ylim(0, 0.4)
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(fontsize=9)

    plt.suptitle('Building Discrete CDF: Accumulating Probability Mass', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    # Show PMF and CDF side by side for discrete case
    fig_2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PMF
    ax1.stem(x_discrete, pmf_discrete, linefmt='b-', markerfmt='bo', 
            basefmt=' ', label='PMF')
    for x_val, pmf_val in zip(x_discrete, pmf_discrete):
        ax1.text(x_val, pmf_val + 0.015, f'{pmf_val:.3f}', 
                ha='center', fontsize=9, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('p(x) = P(X = x)', fontsize=12)
    ax1.set_title('PMF: Probability Mass Function', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_discrete)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # CDF
    ax2.step(x_discrete, cdf_discrete, where='post', linewidth=2.5, color='red', label='CDF')
    ax2.scatter(x_discrete, cdf_discrete, s=100, color='red', zorder=5)

    # Show the step function clearly
    for i in range(len(x_discrete)):
        if i < len(x_discrete) - 1:
            ax2.plot([x_discrete[i], x_discrete[i+1]], [cdf_discrete[i], cdf_discrete[i]], 
                    'r-', linewidth=2.5)
            # Open circle at right endpoint
            ax2.scatter([x_discrete[i+1]], [cdf_discrete[i]], s=80, 
                    facecolors='none', edgecolors='red', linewidth=2, zorder=4)

    for x_val, cdf_val in zip(x_discrete, cdf_discrete):
        ax2.text(x_val + 0.15, cdf_val, f'{cdf_val:.3f}', 
                fontsize=9, fontweight='bold')

    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(1, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('F(x) = P(X ≤ x)', fontsize=12)
    ax2.set_title('CDF: Step Function (jumps at each x)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_discrete)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig_1, fig_2

def demo_pdf_cdf_discrete(py=False):
    # Example: Number of successful API calls out of 5 attempts
    # X ~ Binomial(n=5, p=0.6)
    n_trials = 5
    p_success = 0.6

    x_discrete = np.arange(0, n_trials + 1)
    pmf_discrete = stats.binom.pmf(x_discrete, n_trials, p_success)
    cdf_discrete = stats.binom.cdf(x_discrete, n_trials, p_success)

    print(f"\nExample: X ~ Binomial(n={n_trials}, p={p_success})")
    print("X = number of successful API calls out of 5 attempts\n")

    # Create probability table
    print("Probability Distribution:")
    print("─" * 50)
    print(f"{'x':<5} {'P(X=x)':<12} {'F(x)=P(X≤x)':<15}")
    print("─" * 50)
    for x_val, pmf_val, cdf_val in zip(x_discrete, pmf_discrete, cdf_discrete):
        print(f"{x_val:<5} {pmf_val:<12.4f} {cdf_val:<15.4f}")
    print("─" * 50)

    # Visualization: Building the CDF step by step
    fig_1, axes = plt.subplots(2, 3, figsize=(16, 9))

    for idx in range(6):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Plot PMF as stems
        ax.stem(x_discrete, pmf_discrete, linefmt='b-', markerfmt='bo', 
                basefmt=' ', label='PMF p(x)')
        
        # Highlight accumulated probability up to current point
        current_x = idx
        accumulated_x = x_discrete[x_discrete <= current_x]
        accumulated_pmf = pmf_discrete[x_discrete <= current_x]
        
        # Shade the accumulated bars
        for x_val, pmf_val in zip(accumulated_x, accumulated_pmf):
            ax.bar(x_val, pmf_val, width=0.4, alpha=0.5, color='lightblue',
                edgecolor='blue', linewidth=2)
        
        # Calculate and display CDF value
        cdf_value = np.sum(accumulated_pmf)
        ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
        ax.axvline(current_x, color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        ax.text(0.05, 0.95, f'F({current_x}) = P(X≤{current_x})\n= {cdf_value:.4f}', 
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.set_xlabel('x (number of successes)', fontsize=10)
        ax.set_ylabel('Probability', fontsize=10)
        ax.set_title(f'CDF at x = {current_x}', fontsize=11, fontweight='bold')
        ax.set_xticks(x_discrete)
        ax.set_ylim(0, 0.4)
        ax.grid(True, alpha=0.3, axis='y')
        if idx == 0:
            ax.legend(fontsize=9)

    plt.suptitle('Building Discrete CDF: Accumulating Probability Mass', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    # Show PMF and CDF side by side for discrete case
    fig_2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # PMF
    ax1.stem(x_discrete, pmf_discrete, linefmt='b-', markerfmt='bo', 
            basefmt=' ', label='PMF')
    for x_val, pmf_val in zip(x_discrete, pmf_discrete):
        ax1.text(x_val, pmf_val + 0.015, f'{pmf_val:.3f}', 
                ha='center', fontsize=9, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('p(x) = P(X = x)', fontsize=12)
    ax1.set_title('PMF: Probability Mass Function', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_discrete)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # CDF
    ax2.step(x_discrete, cdf_discrete, where='post', linewidth=2.5, color='red', label='CDF')
    ax2.scatter(x_discrete, cdf_discrete, s=100, color='red', zorder=5)

    # Show the step function clearly
    for i in range(len(x_discrete)):
        if i < len(x_discrete) - 1:
            ax2.plot([x_discrete[i], x_discrete[i+1]], [cdf_discrete[i], cdf_discrete[i]], 
                    'r-', linewidth=2.5)
            # Open circle at right endpoint
            ax2.scatter([x_discrete[i+1]], [cdf_discrete[i]], s=80, 
                    facecolors='none', edgecolors='red', linewidth=2, zorder=4)

    for x_val, cdf_val in zip(x_discrete, cdf_discrete):
        ax2.text(x_val + 0.15, cdf_val, f'{cdf_val:.3f}', 
                fontsize=9, fontweight='bold')

    ax2.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(1, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('F(x) = P(X ≤ x)', fontsize=12)
    ax2.set_title('CDF: Step Function (jumps at each x)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x_discrete)
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig_1, fig_2

def demo_cdf_interval_discrete(py=False):
    # X ~ Binomial(n=5, p=0.6)
    n_trials = 5
    p_success = 0.6

    x_discrete = np.arange(0, n_trials + 1)
    pmf_discrete = stats.binom.pmf(x_discrete, n_trials, p_success)
    cdf_discrete = stats.binom.cdf(x_discrete, n_trials, p_success)
    # Calculate interval: P(1 < X ≤ 4) = P(X ∈ {2, 3, 4})
    a_disc, b_disc = 1, 4

    F_a_disc = stats.binom.cdf(a_disc, n_trials, p_success)
    F_b_disc = stats.binom.cdf(b_disc, n_trials, p_success)
    prob_interval_disc = F_b_disc - F_a_disc

    # Direct calculation for verification
    prob_direct = sum(pmf_discrete[(x_discrete > a_disc) & (x_discrete <= b_disc)])

    print(f"\nExample: Calculate P({a_disc} < X ≤ {b_disc})")
    print(f"This means: P(X ∈ {{{', '.join(map(str, x_discrete[(x_discrete > a_disc) & (x_discrete <= b_disc)]))}}})")

    print(f"\nMethod 1: Using CDF Formula")
    print(f"  F({b_disc}) = P(X ≤ {b_disc}) = {F_b_disc:.4f}")
    print(f"  F({a_disc}) = P(X ≤ {a_disc}) = {F_a_disc:.4f}")
    print(f"  P({a_disc} < X ≤ {b_disc}) = F({b_disc}) - F({a_disc})")
    print(f"  P({a_disc} < X ≤ {b_disc}) = {F_b_disc:.4f} - {F_a_disc:.4f} = {prob_interval_disc:.4f}")

    print(f"\nMethod 2: Direct Summation (verification)")
    print(f"  P(X={a_disc+1}) + P(X={a_disc+2}) + P(X={a_disc+3})")
    values_in_interval = x_discrete[(x_discrete > a_disc) & (x_discrete <= b_disc)]
    pmf_in_interval = pmf_discrete[(x_discrete > a_disc) & (x_discrete <= b_disc)]
    sum_str = " + ".join([f"{p:.4f}" for p in pmf_in_interval])
    print(f"  = {sum_str}")
    print(f"  = {prob_direct:.4f}")
    print(f"\n✓ Both methods agree!")

    # Comprehensive visualization
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 2, hspace=0.4, wspace=0.3)

    # Panel 1: F(b) - Cumulative up to b
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.stem(x_discrete, pmf_discrete, linefmt='b-', markerfmt='bo', 
            basefmt=' ', label='PMF')

    # Highlight values up to b
    for x_val, pmf_val in zip(x_discrete[x_discrete <= b_disc], 
                            pmf_discrete[x_discrete <= b_disc]):
        ax1.bar(x_val, pmf_val, width=0.4, alpha=0.6, color='lightgreen',
                edgecolor='green', linewidth=2)

    ax1.axvline(b_disc, color='green', linestyle='--', linewidth=2.5)
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('p(x)', fontsize=11)
    ax1.set_title(f'Step 1: F({b_disc}) = P(X ≤ {b_disc}) = {F_b_disc:.4f}', 
                fontsize=12, fontweight='bold')
    ax1.set_xticks(x_discrete)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.text(b_disc, ax1.get_ylim()[1]*0.95, f'  x={b_disc}', 
            fontsize=10, color='green', va='top')

    # Panel 2: F(a) - Cumulative up to a
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.stem(x_discrete, pmf_discrete, linefmt='b-', markerfmt='bo', 
            basefmt=' ', label='PMF')

    # Highlight values up to a
    for x_val, pmf_val in zip(x_discrete[x_discrete <= a_disc], 
                            pmf_discrete[x_discrete <= a_disc]):
        ax2.bar(x_val, pmf_val, width=0.4, alpha=0.6, color='lightcoral',
                edgecolor='red', linewidth=2)

    ax2.axvline(a_disc, color='red', linestyle='--', linewidth=2.5)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('p(x)', fontsize=11)
    ax2.set_title(f'Step 2: F({a_disc}) = P(X ≤ {a_disc}) = {F_a_disc:.4f}', 
                fontsize=12, fontweight='bold')
    ax2.set_xticks(x_discrete)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.text(a_disc, ax2.get_ylim()[1]*0.95, f'x={a_disc}  ', 
            fontsize=10, color='red', va='top', ha='right')

    # Panel 3: The interval (a, b]
    ax3 = fig.add_subplot(gs[1, :])
    ax3.stem(x_discrete, pmf_discrete, linefmt='b-', markerfmt='bo', 
            basefmt=' ', label='PMF')

    # Highlight the interval
    interval_mask = (x_discrete > a_disc) & (x_discrete <= b_disc)
    for x_val, pmf_val in zip(x_discrete[interval_mask], pmf_discrete[interval_mask]):
        ax3.stem([x_val], [pmf_val], linefmt='orange', markerfmt='o', 
                basefmt=' ', label='_nolegend_')
        ax3.bar(x_val, pmf_val, width=0.5, alpha=0.8, color='gold',
                edgecolor='orange', linewidth=3)
        ax3.text(x_val, pmf_val + 0.02, f'{pmf_val:.4f}', 
                ha='center', fontsize=10, fontweight='bold', color='orange')

    ax3.axvline(a_disc, color='red', linestyle='--', linewidth=2.5, alpha=0.7,
            label=f'a = {a_disc}')
    ax3.axvline(b_disc, color='green', linestyle='--', linewidth=2.5, alpha=0.7,
            label=f'b = {b_disc}')

    ax3.set_xlabel('x', fontsize=12)
    ax3.set_ylabel('p(x)', fontsize=12)
    ax3.set_title(f'Step 3: P({a_disc} < X ≤ {b_disc}) = F({b_disc}) - F({a_disc}) = {prob_interval_disc:.4f}', 
                fontsize=13, fontweight='bold')
    ax3.set_xticks(x_discrete)
    ax3.legend(fontsize=11, loc='upper left')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add bracket showing interval
    y_bracket = ax3.get_ylim()[1] * 0.15
    ax3.plot([a_disc, b_disc], [y_bracket, y_bracket], 'orange', linewidth=3)
    ax3.plot([a_disc, a_disc], [y_bracket-0.01, y_bracket+0.01], 'orange', linewidth=3)
    ax3.plot([b_disc, b_disc], [y_bracket-0.01, y_bracket+0.01], 'orange', linewidth=3)
    ax3.text((a_disc + b_disc)/2, y_bracket + 0.02, 
            f'Interval ({a_disc}, {b_disc}] contains: {{{", ".join(map(str, values_in_interval))}}}',
            ha='center', fontsize=11, fontweight='bold', color='orange',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    # Panel 4: CDF view
    ax4 = fig.add_subplot(gs[2, :])

    # Plot CDF with step function
    ax4.step(x_discrete, cdf_discrete, where='post', linewidth=3, 
            color='purple', label='CDF F(x)')
    ax4.scatter(x_discrete, cdf_discrete, s=120, color='purple', zorder=5)

    # Show open circles at discontinuities
    for i in range(len(x_discrete) - 1):
        ax4.scatter([x_discrete[i+1]], [cdf_discrete[i]], s=100, 
                facecolors='none', edgecolors='purple', linewidth=2, zorder=4)

    # Mark F(a) and F(b)
    ax4.plot([a_disc], [F_a_disc], 'ro', markersize=15, 
            label=f'F({a_disc}) = {F_a_disc:.4f}', zorder=6)
    ax4.plot([b_disc], [F_b_disc], 'go', markersize=15, 
            label=f'F({b_disc}) = {F_b_disc:.4f}', zorder=6)

    # Draw lines to axes
    ax4.plot([a_disc, a_disc], [0, F_a_disc], 'r--', linewidth=2, alpha=0.7)
    ax4.plot([b_disc, b_disc], [0, F_b_disc], 'g--', linewidth=2, alpha=0.7)
    ax4.plot([0, a_disc], [F_a_disc, F_a_disc], 'r--', linewidth=1, alpha=0.5)
    ax4.plot([0, b_disc], [F_b_disc, F_b_disc], 'g--', linewidth=1, alpha=0.5)

    # Show the difference with arrow
    arrow_x = b_disc + 0.4
    ax4.annotate('', xy=(arrow_x, F_a_disc), xytext=(arrow_x, F_b_disc),
                arrowprops=dict(arrowstyle='<->', color='orange', lw=4))
    ax4.text(arrow_x + 0.25, (F_a_disc + F_b_disc)/2, 
            f'F({b_disc}) - F({a_disc})\n= {prob_interval_disc:.4f}',
            fontsize=12, fontweight='bold', color='orange',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax4.set_xlabel('x', fontsize=12)
    ax4.set_ylabel('F(x) = P(X ≤ x)', fontsize=12)
    ax4.set_title('CDF View: Discrete Step Function', fontsize=13, fontweight='bold')
    ax4.set_xticks(x_discrete)
    ax4.set_ylim(-0.05, 1.05)
    ax4.legend(fontsize=11, loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.suptitle(f'Discrete Case: P({a_disc} < X ≤ {b_disc}) = F({b_disc}) - F({a_disc})', 
                fontsize=15, fontweight='bold', y=0.995)
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def comparison_discrete_rv(py=False):
    # COMPARISON 1: PMF Shapes Side-by-Side
    # ============================================================================

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # --- Bernoulli ---
    ax1 = axes[0, 0]
    p_bern = 0.7
    x_bern = [0, 1]
    pmf_bern = [1-p_bern, p_bern]

    bars1 = ax1.bar(x_bern, pmf_bern, width=0.4, alpha=0.7, 
                edgecolor='black', linewidth=2,
                color=['salmon', 'lightgreen'])
    ax1.set_xlabel('x', fontsize=12, fontweight='bold')
    ax1.set_ylabel('P(X = x)', fontsize=12, fontweight='bold')
    ax1.set_title('Bernoulli(p=0.7)\nSingle Binary Trial', 
                fontsize=13, fontweight='bold')
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['Failure (0)', 'Success (1)'])
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (x, y) in enumerate(zip(x_bern, pmf_bern)):
        ax1.text(x, y + 0.03, f'{y:.2f}', ha='center', fontsize=11, fontweight='bold')

    # Add info box
    info_text1 = f'E[X] = {p_bern:.2f}\nVar(X) = {p_bern*(1-p_bern):.3f}\nUse: Single event'
    ax1.text(0.95, 0.95, info_text1, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # --- Binomial ---
    ax2 = axes[0, 1]
    n_binom = 20
    p_binom = 0.3
    x_binom = np.arange(0, n_binom + 1)
    pmf_binom = stats.binom.pmf(x_binom, n_binom, p_binom)

    ax2.stem(x_binom, pmf_binom, basefmt=' ', linefmt='blue', markerfmt='bo')
    ax2.fill_between(x_binom, pmf_binom, alpha=0.3)
    ax2.axvline(n_binom * p_binom, color='red', linestyle='--', 
            linewidth=2, label=f'Mean = {n_binom*p_binom:.1f}')
    ax2.set_xlabel('x (number of successes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('P(X = x)', fontsize=12, fontweight='bold')
    ax2.set_title(f'Binomial(n={n_binom}, p={p_binom})\n# Successes in n Trials', 
                fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(fontsize=10)

    info_text2 = f'E[X] = {n_binom*p_binom:.1f}\nVar(X) = {n_binom*p_binom*(1-p_binom):.2f}\nUse: Fixed n trials'
    ax2.text(0.95, 0.95, info_text2, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # --- Poisson ---
    ax3 = axes[1, 0]
    lambda_pois = 4.5
    x_pois = np.arange(0, 15)
    pmf_pois = stats.poisson.pmf(x_pois, lambda_pois)

    ax3.stem(x_pois, pmf_pois, basefmt=' ', linefmt='green', markerfmt='go')
    ax3.fill_between(x_pois, pmf_pois, alpha=0.3, color='lightgreen')
    ax3.axvline(lambda_pois, color='red', linestyle='--', 
            linewidth=2, label=f'Mean = Var = {lambda_pois}')
    ax3.set_xlabel('x (number of events)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('P(X = x)', fontsize=12, fontweight='bold')
    ax3.set_title(f'Poisson(λ={lambda_pois})\nRare Events in Interval', 
                fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.legend(fontsize=10)

    info_text3 = f'E[X] = {lambda_pois}\nVar(X) = {lambda_pois}\nUse: Rare events'
    ax3.text(0.95, 0.95, info_text3, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # --- Geometric ---
    ax4 = axes[1, 1]
    p_geom = 0.3
    x_geom = np.arange(1, 16)
    pmf_geom = stats.geom.pmf(x_geom, p_geom)  

    ax4.stem(x_geom, pmf_geom, basefmt=' ', linefmt='purple', markerfmt='mo')
    ax4.fill_between(x_geom, pmf_geom, alpha=0.3, color='plum')
    ax4.axvline(1/p_geom, color='red', linestyle='--', 
            linewidth=2, label=f'Mean = {1/p_geom:.2f}')
    ax4.set_xlabel('x (trials until first success)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('P(X = x)', fontsize=12, fontweight='bold')
    ax4.set_title(f'Geometric(p={p_geom})\nWaiting for First Success', 
                fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.legend(fontsize=10)

    var_geom = (1-p_geom) / (p_geom**2)
    info_text4 = f'E[X] = {1/p_geom:.2f}\nVar(X) = {var_geom:.2f}\nUse: First occurrence'
    ax4.text(0.95, 0.95, info_text4, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))

    plt.suptitle('Discrete Distributions: Shape Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def mystery_prob(py=False):
    np.random.seed(42)
    clicks_data = np.random.poisson(lam=2, size=1000)
    
    # Generate comparison data
    x_range = np.arange(0, 15)
    binomial_probs = stats.binom.pmf(x_range, n=50, p=0.04)
    poisson_probs = stats.poisson.pmf(x_range, mu=2)
    normal_probs = stats.norm.pdf(x_range, loc=2, scale=5)

    # Calculate fit statistics
    def calculate_fit(observed_data, theoretical_probs, x_values):
        obs_counts, _ = np.histogram(observed_data, bins=np.append(x_values, x_values[-1]+1))
        obs_probs = obs_counts / len(observed_data)
        
        # Chi-square-like measure (simplified)
        valid_idx = theoretical_probs > 0
        chi_sq = np.sum((obs_probs[valid_idx] - theoretical_probs[valid_idx])**2 / 
                        theoretical_probs[valid_idx])
        return chi_sq

    binomial_fit = calculate_fit(clicks_data, binomial_probs, x_range)
    poisson_fit = calculate_fit(clicks_data, poisson_probs, x_range)

    print(f"\nFit Statistics (lower is better):")
    print(f"Binomial(50, 0.04): χ² ≈ {binomial_fit:.4f}")
    print(f"Poisson(2): χ² ≈ {poisson_fit:.4f}")

    # Visual comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Actual data
    axes[0, 0].hist(clicks_data, bins=range(0, 15), density=True, 
                alpha=0.7, edgecolor='black', label='Observed Data')
    axes[0, 0].set_title('Mystery Data', fontweight='bold', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Binomial fit
    axes[0, 1].hist(clicks_data, bins=range(0, 15), density=True, 
                alpha=0.5, edgecolor='black', label='Observed')
    axes[0, 1].stem(x_range, binomial_probs, linefmt='r-', markerfmt='ro', 
                basefmt=' ', label=f'Binomial(50, 0.04)')
    axes[0, 1].set_title(f'Binomial Fit (χ²={binomial_fit:.3f})', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Poisson fit
    axes[1, 0].hist(clicks_data, bins=range(0, 15), density=True, 
                alpha=0.5, edgecolor='black', label='Observed')
    axes[1, 0].stem(x_range, poisson_probs, linefmt='g-', markerfmt='go', 
                basefmt=' ', label=f'Poisson(2)')
    axes[1, 0].set_title(f'Poisson Fit (χ²={poisson_fit:.3f}) ' + r'$\checkmark$'  +' BEST', 
                        fontweight='bold', color='green')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Normal comparison (for educational purposes)
    axes[1, 1].hist(clicks_data, bins=range(0, 15), density=True, 
                alpha=0.5, edgecolor='black', label='Observed (Discrete)')
    x_cont = np.linspace(0, 14, 100)
    axes[1, 1].plot(x_cont, stats.norm.pdf(x_cont, 2, 5), 'b-', 
                linewidth=2, label='Normal(2, 25)')
    axes[1, 1].set_title('Normal Distribution (Wrong for Discrete Data!)', 
                        fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def demo_pmf_limitations_cont_rv(py=False):
    # Show histogram with increasing bins
    data = np.random.normal(0, 1, 1000)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    bin_counts = [10, 50, 100, 500]

    for ax, bins in zip(axes, bin_counts):
        counts, edges, _ = ax.hist(data, bins=bins, density=False, alpha=0.7)
        ax.set_title(f'{bins} bins')
        ax.set_ylabel('Count')
        
        # Show that probability per bin → 0 as bins increase
        max_count = np.max(counts)
        ax.text(0.5, 0.95, f'Max count: {max_count:.0f}\nPer bin: {max_count/1000:.3f}',
                transform=ax.transAxes, verticalalignment='top')

    plt.suptitle('As bins increase, probability per bin approaches 0!')
    if py:
        plt.show() 
    plt.close() # to prevent automatic display


    return fig

def demo_pdf(py=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Example 1: Normal (f(x) always < 1)
    x1 = np.linspace(-4, 4, 200)
    pdf1 = stats.norm.pdf(x1, 0, 1)
    ax1.fill_between(x1, pdf1, alpha=0.3)
    ax1.plot(x1, pdf1, color="salmon", linewidth=2)
    ax1.axhline(1, color='red', linestyle='--', label='Probability = 1 line')
    ax1.set_title('Normal(0,1): f(x) always < 1')
    ax1.set_ylabel('f(x) - DENSITY (not probability!)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Example 2: Beta where f(x) > 1
    x2 = np.linspace(0, 1, 200)
    pdf2 = stats.beta.pdf(x2, 3, 2)
    ax2.fill_between(x2, pdf2, alpha=0.3, color='orange')
    ax2.plot(x2, pdf2, 'r-', linewidth=2)
    ax2.axhline(1, color='red', linestyle='--', label='f(x) can exceed 1!')
    ax2.set_title('Beta(3,2): f(x) can be > 1 (still valid!)')
    ax2.set_ylabel('f(x) - DENSITY')
    ax2.text(0.2, 1.5, 'f(x) > 1 here!\nBut area under curve = 1',
            bbox=dict(boxstyle='round', facecolor='yellow'))
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def demo_cdf_disrete_vs_cont(py=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Continuous: Normal
    x_cont = np.linspace(-4, 4, 200)
    pdf_cont = stats.norm.pdf(x_cont, 0, 1)
    cdf_cont = stats.norm.cdf(x_cont, 0, 1)

    axes[0, 0].fill_between(x_cont, pdf_cont, alpha=0.3)
    axes[0, 0].plot(x_cont, pdf_cont, color="salmon", linewidth=2)
    axes[0, 0].set_title('Continuous: PDF (smooth)', fontweight='bold')
    axes[0, 0].set_ylabel('f(x)')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_cont, cdf_cont, color="salmon", linewidth=2)
    axes[0, 1].set_title('Continuous: CDF (smooth)', fontweight='bold')
    axes[0, 1].set_ylabel('F(x) = P(X≤x)')
    axes[0, 1].grid(True, alpha=0.3)

    # Discrete: Binomial (for comparison)
    x_disc = np.arange(0, 11)
    pmf_disc = stats.binom.pmf(x_disc, 10, 0.5)
    cdf_disc = stats.binom.cdf(x_disc, 10, 0.5)

    axes[1, 0].stem(x_disc, pmf_disc, basefmt=' ')
    axes[1, 0].set_title('Discrete: PMF (bars)', fontweight='bold')
    axes[1, 0].set_ylabel('p(x) = P(X=x)')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].step(x_disc, cdf_disc, where='post', linewidth=2)
    axes[1, 1].scatter(x_disc, cdf_disc, s=50, zorder=5)
    axes[1, 1].set_title('Discrete: CDF (steps)', fontweight='bold')
    axes[1, 1].set_ylabel('F(x) = P(X≤x)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Continuous vs Discrete: PDF/PMF and CDF', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def get_val_by_x(x, mean=0, sd=1):
    var = float(sd)**2
    denom = (2*math.pi*var)**.5
    num = math.exp(-(float(x)-float(mean))**2/(2*var))
    return num/denom
    
def cdf_interval(py=False):
    # Continuous: Normal
    x_cont = np.linspace(-4, 4, 200)
    pdf_cont = stats.norm.pdf(x_cont, 0, 1)
    cdf_cont = stats.norm.cdf(x_cont, 0, 1)
    # Show P(a < X ≤ b) = F(b) - F(a) geometrically
    a, b = -1, 1.5
    F_a = stats.norm.cdf(a, 0, 1)
    F_b = stats.norm.cdf(b, 0, 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # PDF with shaded area
    mask = (x_cont > a) & (x_cont <= b)
    #axes[0].fill_between(x_cont, pdf_cont, alpha=0.2, color='blue')
    axes[0].fill_between(x_cont[mask], pdf_cont[mask], alpha=0.4, color='salmon')
    axes[0].plot(x_cont, pdf_cont, color="salmon", linewidth=2)
    axes[0].axvline(a, color='red', linestyle='--', linewidth=2)
    axes[0].axvline(b, color='green', linestyle='--', linewidth=2)
    axes[0].set_title(f'PDF: Area = P({a} < X ≤ {b})')
    axes[0].set_ylabel('f(x)')

    # CDF showing F(b)
    axes[1].plot(x_cont, cdf_cont, color="salmon", linewidth=2)
    axes[1].axvline(b, color='green', linestyle='--', linewidth=2)
    axes[1].axhline(F_b, color='green', linestyle=':', alpha=0.7)
    axes[1].plot(b, F_b, 'go', markersize=12)
    axes[1].set_title(f'F({b}) = {F_b:.4f}')
    axes[1].set_ylabel('F(x)')

    # CDF showing F(b) - F(a)
    axes[2].plot(x_cont, cdf_cont, color="salmon", linewidth=2)
    axes[2].axvline(a, color='red', linestyle='--', linewidth=2)
    axes[2].axvline(b, color='green', linestyle='--', linewidth=2)
    axes[2].axhline(F_a, color='red', linestyle=':', alpha=0.7)
    axes[2].axhline(F_b, color='green', linestyle=':', alpha=0.7)
    axes[2].plot(a, F_a, 'ro', markersize=12)
    axes[2].plot(b, F_b, 'go', markersize=12)

    # Arrow showing difference
    axes[2].annotate('', xy=(b+0.5, F_b), xytext=(b+0.5, F_a),
                    arrowprops=dict(arrowstyle='<->', color='orange', lw=3))
    axes[2].text(b+0.7, (F_a+F_b)/2, f'F({b})-F({a})\n={F_b-F_a:.4f}',
                fontweight='bold', color='orange')
    axes[2].set_title(f'P({a}<X≤{b}) = F({b})-F({a})')

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def show_complementary_event_cdf(py=False):
    col = '#EF9A9A'
    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)

    rv1 = t(df=1, loc=0, scale=1)
    y1 = rv1.pdf(x) 


    fig, ax = plt.subplots(1,1, figsize=(10, 5))
    plt.plot(x, stats.norm.pdf(x, mu, sigma), color='tab:blue', label='Normal Dist')


    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('zero')

    ax.vlines(1.645, ymin=0, ymax=get_val_by_x(1.645,  mu, sigma), linestyles="--", color=col)

    a3 = np.linspace(1.645, mu + 4*sigma)
    b3 = [get_val_by_x(aa,  mu, sigma) for aa in a3]

    plt.fill_between(a3, b3, 0, where = (a3 >= 1.645), 
                    color = col, alpha=0.3)


    ax.set_ylabel('', position=(0,1), fontsize=16, rotation='horizontal')
    ax.set_xlabel('', position=(1,0), fontsize=16)


    ax.set_yticks([])
    ax.set_yticklabels([])

    ax.set_xticks([1.645])
    ax.set_xticklabels([r'$a$'], fontsize=14)


    ax.annotate("", xy=(1.8, 0.03), xytext=(2.5, 0.12), arrowprops=dict(arrowstyle="->"))
    ax.annotate(r"$P(X > a)$", 
                xy=(mu+2*sigma, get_val_by_x(mu+2*sigma)+0.25), 
                xytext=(2.1, 0.12), fontsize=18)

    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig

def demo_weights(py=False):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Panel 1: Why P(W=x) = 0
    ax1 = axes[0, 0]
    x = np.linspace(-3, 3, 1000)
    pdf = stats.norm.pdf(x, 0, 1)
    ax1.plot(x, pdf, 'b-', linewidth=2)
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2)
    ax1.plot(0.5, stats.norm.pdf(0.5, 0, 1), 'ro', markersize=10)
    ax1.text(1.1, stats.norm.pdf(0.5, 0, 1),
            'f(0.5) = height\n≠ probability!',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow'))
    ax1.fill_between([0.5-0.001, 0.5+0.001], 0,
                    [stats.norm.pdf(0.5-0.001, 0, 1), stats.norm.pdf(0.5+0.001, 0, 1)],
                    alpha=0.5, color='red')
    ax1.text(0.5, 0.05, 'Area ≈ 0\n(zero width)', ha='center', fontsize=9)
    ax1.set_title('P(W = 0.5 exactly) = 0', fontweight='bold')
    ax1.set_ylabel('f(w) - Density')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Intervals have non-zero probability
    ax2 = axes[0, 1]
    ax2.plot(x, pdf, 'b-', linewidth=2)
    a, b = -0.5, 0.5
    mask = (x >= a) & (x <= b)
    ax2.fill_between(x[mask], pdf[mask], alpha=0.5, color='green')
    prob_interval = stats.norm.cdf(b, 0, 1) - stats.norm.cdf(a, 0, 1)
    ax2.text(0, 0.2, f'P({a} ≤ W ≤ {b})\n= {prob_interval:.4f}\n(has area!)',
            ha='center', fontsize=11, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax2.set_title('P(a ≤ W ≤ b) > 0 (interval)', fontweight='bold')
    ax2.set_ylabel('f(w)')
    ax2.grid(True, alpha=0.3)
    
    # Panel 3: Different σ values
    ax3 = axes[1, 0]
    sigmas = [0.1, 0.5, 1.0, 2.0]
    colors_sigma = ['red', 'orange', 'blue', 'purple']
    for sigma, color in zip(sigmas, colors_sigma):
        pdf_sigma = stats.norm.pdf(x, 0, sigma)
        ax3.plot(x, pdf_sigma, linewidth=2, label=f'σ={sigma}', color=color)
    ax3.set_title('Effect of σ on Weight Distribution', fontweight='bold')
    ax3.set_ylabel('f(w)')
    ax3.set_xlabel('Weight Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.text(1.5, 0.5, 'Small σ: Concentrated\nLarge σ: Spread out',
            ha='center', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='lightyellow'))

    # Panel 4: Xavier initialization example
    ax4 = axes[1, 1]
    n_in_example = 100
    sigma_xavier = 1/np.sqrt(n_in_example)
    x_weights = np.linspace(-0.5, 0.5, 1000)
    pdf_xavier = stats.norm.pdf(x_weights, 0, sigma_xavier)
    ax4.fill_between(x_weights, pdf_xavier, alpha=0.3, color='green')
    ax4.plot(x_weights, pdf_xavier, 'g-', linewidth=2)
    ax4.set_title(f'Xavier Init: W ~ N(0, 1/{n_in_example})', fontweight='bold')
    ax4.set_ylabel('f(w)')
    ax4.set_xlabel('Weight Value')
    ax4.text(0, max(pdf_xavier)*0.7,
            f'σ = 1/√{n_in_example} = {sigma_xavier:.3f}\n\nKeeps variance\nstable!',
            ha='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen'))
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Understanding Continuous Distributions in Weight Initialization',
                    fontsize=14, fontweight='bold')
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig
    

if __name__ == "__main__":
    #generate_click_data(py=True)
    #visualize_rv_concept(py=True)
    #discrete_vs_cont_rv(py=True)
    #coffee_example(py=True)
    #coffee_example_mean_var(py=True)
    #demo_markov_ineq(py=True)
    #demo_markov_ineq_2(py=True)
    #demo_schebyshev_ineq(py=True)
    #demo_chebyshev_ineq_2(py=True)
    #demo_pdf_cdf_discrete(py=True)
    #demo_cdf_interval_discrete(py=True)
    #comparison_discrete_rv(py=True)
    #mystery_prob(py=True)
    #demo_pmf_limitations_cont_rv(py=True)
    #demo_pdf(py=True)
    #demo_cdf_disrete_vs_cont(py=True)
    #cdf_interval(py=True)
    #show_complementary_event_cdf(py=True)
    demo_weights(py=True)