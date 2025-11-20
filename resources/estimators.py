import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from ipywidgets import interact, FloatSlider, IntSlider
from scipy.special import comb

plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji']

def generate_hook_data(n_experiments=50, py=False):
    """
    Simulates and visualises data from the hook problem

    Args:
        n_experiments (int, optional): number of experiments. Defaults to 50.
        py (bool, optional): flag to show or close the plot. Defaults to False (close).

    Returns:
        _type_: generated figure
    """
    np.random.seed(42)
    true_optimal_sigma = 0.14  # Unknown to students initially
    base_accuracy = 95 # maximum possible accuracy at the optimal point in percentage. Represents best-case scenario. Corresponds to sigma = 0.14
    
    drop_magnitude = 400 # controls HOW FAST accuracy drops as we move from optimal
    noise = np.random.normal(0, 1.0, n_experiments) # noise term adding realistic variability
    
    #Simulate experiments (accuracy depends on how close sigma is to optimal)
    sigmas_tested = np.random.uniform(0.05, 0.25, n_experiments)
    accuracies = base_accuracy - drop_magnitude * (sigmas_tested - true_optimal_sigma)**2 + noise

    # Find empirical best
    best_idx = np.argmax(accuracies)
    estimated_sigma = sigmas_tested[best_idx]
    
    #Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    #Left plot: Scatter of experiments
    ax1.scatter(sigmas_tested, accuracies, alpha=0.6, s=80, c='steelblue', edgecolors='black')
    ax1.scatter(estimated_sigma, accuracies[best_idx], s=300, c='red', marker='*',
                edgecolors='darkred', linewidths=2, label=f'Best: ' + r'$\hat{\sigma}$'+ f'={estimated_sigma:.3f}', zorder=5)
    
    ax1.axvline(estimated_sigma, color='red', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Weight Initialization Scale (σ)', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_title(f'{n_experiments} Neural Network Training Experiments', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.set_ylim(85, 100)
    ax1.grid(True, alpha=0.3)
    
    #Right plot: Question marks
    ax2.text(0.5, 0.7, '?', fontsize=120, ha='center', va='center', alpha=0.3, fontweight='bold', color='darkred')
    ax2.text(0.5, 0.3, 'Is this the TRUE optimal σ?', fontsize=16, ha='center', va='center',
            fontweight='bold', color='darkred')
    ax2.text(0.5, 0.15, 'How confident should we be?', fontsize=14, ha='center', va='center',
            style='italic', color='gray')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()

    return fig
    

def demonstrate_estimator_concept(py=False):
    """Interactive demo: Different ways to estimate the center of a target"""
    
    np.random.seed(42)
    true_center = np.array([0, 0])  # True bullseye
    n_throws = 20
    
    # Simulate dart throws (with some noise)
    throws = np.random.multivariate_normal(true_center, [[1, 0], [0, 1]], n_throws)
    
    # Three different estimators
    estimator_mean = np.mean(throws, axis=0)  # Sample mean
    estimator_median = np.median(throws, axis=0)  # Sample median
    estimator_first = throws[0]  # Just use first throw (bad estimator!)
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw target circles
    circles = [plt.Circle((0, 0), r, fill=False, color='gray', linewidth=2, alpha=0.3) 
               for r in [1, 2, 3, 4]]
    for circle in circles:
        ax.add_patch(circle)
    
    # Plot throws
    ax.scatter(throws[:, 0], throws[:, 1], s=100, alpha=0.6, c='steelblue', 
               edgecolors='black', linewidth=1.5, label='Dart throws (data)')
    
    # Plot true center
    ax.scatter(0, 0, s=500, marker='*', c='gold', edgecolors='darkgoldenrod', 
               linewidths=3, label='TRUE center (θ)', zorder=10)
    
    # Plot different estimators
    ax.scatter(*estimator_mean, s=300, marker='s', c='red', edgecolors='darkred', 
               linewidths=2, label='Estimator 1: Mean (θ̂₁)', zorder=5)
    ax.scatter(*estimator_median, s=300, marker='^', c='green', edgecolors='darkgreen', 
               linewidths=2, label='Estimator 2: Median (θ̂₂)', zorder=5)
    ax.scatter(*estimator_first, s=300, marker='D', c='orange', edgecolors='darkorange', 
               linewidths=2, label='Estimator 3: First throw (θ̂₃)', zorder=5)
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)
    ax.set_title('Different Estimators for the Same Parameter', fontsize=14, fontweight='bold')
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    return fig

def explore_sampling_distribution(true_mean=5, true_std=2, sample_size=30, n_experiments=1000):
    """Show that an estimator is itself a random variable"""
    
    # Run many experiments
    estimates = []
    for _ in range(n_experiments):
        sample = np.random.normal(true_mean, true_std, sample_size)
        estimate = np.mean(sample)  # Sample mean estimator
        estimates.append(estimate)
    
    estimates = np.array(estimates)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Left: Show a few sample experiments
    ax = axes[0]
    for i in range(5):
        sample = np.random.normal(true_mean, true_std, sample_size)
        ax.scatter([i]*sample_size, sample, alpha=0.5, s=30)
        sample_mean = np.mean(sample)
        ax.scatter(i, sample_mean, s=200, marker='_', linewidths=4, c='red')
    
    ax.axhline(true_mean, color='gold', linewidth=3, label=f'True μ = {true_mean}', linestyle='--')
    ax.set_xlabel('Experiment Number', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('5 Different Samples → 5 Different Estimates', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Middle: Distribution of estimator
    ax = axes[1]
    ax.hist(estimates, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Overlay theoretical distribution
    x_range = np.linspace(estimates.min(), estimates.max(), 200)
    theoretical_std = true_std / np.sqrt(sample_size)
    ax.plot(x_range, stats.norm.pdf(x_range, true_mean, theoretical_std), 
            'r-', linewidth=3, label='Theoretical distribution')
    
    ax.axvline(true_mean, color='gold', linewidth=3, linestyle='--', label=f'True μ = {true_mean}')
    ax.axvline(np.mean(estimates), color='green', linewidth=2, linestyle='--', 
               label=f'Mean of estimates = {np.mean(estimates):.2f}')
    
    ax.set_xlabel('Estimated μ̂', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Sampling Distribution of μ̂\n({n_experiments} experiments)', 
                 fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Right: Statistics
    ax = axes[2]
    ax.axis('off')
    
    stats_text = f"""
    ESTIMATOR STATISTICS
    ━━━━━━━━━━━━━━━━━━━━━━━
    
    True parameter: μ = {true_mean}
    Sample size: n = {sample_size}
    Number of experiments: {n_experiments}
    
    RESULTS:
    ├─ Mean of estimates: {np.mean(estimates):.4f}
    ├─ Std of estimates: {np.std(estimates):.4f}
    └─ Theoretical std: {theoretical_std:.4f}
    
    KEY INSIGHT:
    The estimator """+ r"$\hat{\mu}$" + """is centered 
    around the true value
    
    But it varies from sample 
    to sample (uncertainty).
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=11, family='monospace', 
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.show()


def visualize_bias_variance_tradeoff():
    """Show different estimators with different bias-variance characteristics"""
    
    np.random.seed(42)
    true_theta = 10
    n_samples = 20
    n_experiments = 500
    
    # Define four different estimators
    def unbiased_low_var(data):
        return np.mean(data)  # Sample mean
    
    def unbiased_high_var(data):
        return data[0]  # Just first observation
    
    def biased_low_var(data):
        return np.mean(data) + 2  # Mean with systematic shift
    
    def biased_high_var(data):
        return np.mean(data[:5]) + 3  # Mean of first 5 + shift
    
    estimators = {
        'Unbiased, Low Variance\n(BEST)': unbiased_low_var,
        'Unbiased, High Variance (Good)': unbiased_high_var,
        'Biased, Low Variance (Problematic)': biased_low_var,
        'Biased, High Variance\n(WORST)': biased_high_var
    }
    
    # Run experiments
    results = {}
    for name, estimator in estimators.items():
        estimates = []
        for _ in range(n_experiments):
            data = np.random.normal(true_theta, 3, n_samples)
            estimates.append(estimator(data))
        results[name] = np.array(estimates)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, (name, estimates) in enumerate(results.items()):
        ax = axes[idx]
        
        # Calculate statistics
        bias = np.mean(estimates) - true_theta
        variance = np.var(estimates)
        mse = np.mean((estimates - true_theta)**2)
        
        # Plot distribution
        ax.hist(estimates, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        
        # Add vertical lines
        ax.axvline(true_theta, color='gold', linewidth=4, label=f'True θ = {true_theta}', 
                   linestyle='--', zorder=5)
        ax.axvline(np.mean(estimates), color='red', linewidth=3, 
                   label=f'E[θ̂] = {np.mean(estimates):.2f}', linestyle='--')
        
        # Add shaded region for variance
        mean_est = np.mean(estimates)
        std_est = np.std(estimates)
        ax.axvspan(mean_est - std_est, mean_est + std_est, alpha=0.2, color='red')
        
        ax.set_xlabel('θ̂', fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Add text box with statistics
        stats_text = f'Bias: {bias:.2f}\nVar: {variance:.2f}\nMSE: {mse:.2f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        
    plt.suptitle('Bias-Variance Tradeoff: Comparing Different Estimators', 
                 fontsize=15, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("=" * 60)
    print("BIAS-VARIANCE DECOMPOSITION: MSE = Bias² + Variance")
    print("=" * 60)
    for name, estimates in results.items():
        bias = np.mean(estimates) - true_theta
        variance = np.var(estimates)
        mse = np.mean((estimates - true_theta)**2)
        mse_decomp = bias**2 + variance
        
        print(f"\n{name}:")
        print(f"  Bias²     = {bias**2:.2f}")
        print(f"  Variance  = {variance:.2f}")
        print(f"  MSE       = {mse:.2f}")
        print(f"  Bias²+Var = {mse_decomp:.2f}")


def plot_heads_tails(py=False):
    # Observed data: 10 coin flips
    observed_flips = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1]  # 1=Heads, 0=Tails
    n_heads = sum(observed_flips)
    n_total = len(observed_flips)
    n_tails = n_total - n_heads
        
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Barplot
    ax1.bar(['Tails', 'Heads'], [n_tails/n_total, n_heads/n_total], color=['silver', 'gold'], edgecolor='black')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Coin Flip Results')
    ax1.set_ylim(0, 1.25)
    for i, (label, count) in enumerate(zip(['Tails', 'Heads'], [n_tails/n_total, n_heads/n_total])):
        ax1.text(i, count + 0.1, str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Plot 2: Scatter plot with vertical stacking
    flip_numbers = np.arange(1, n_total + 1)

    # Count how many of each value appear before each position (for stacking)
    y_positions = []
    x_positions = []
    heads_count = 0
    tails_count = 0

    for flip in observed_flips:
        if flip == 1:  # Heads
            y_positions.append(0 + heads_count * 0.1)  # Stack heads vertically
            x_positions.append(1)
            heads_count += 1
        else:  # Tails
            y_positions.append(0 + tails_count * 0.1)  # Stack tails vertically
            x_positions.append(0)
            tails_count += 1

    # Create scatter plot
    colors = ['silver' if f == 0 else 'gold' for f in observed_flips]
    ax2.scatter(x_positions, y_positions, c=colors, s=200, edgecolors='black', linewidths=2, alpha=0.8)

    ax2.set_xlabel('Outcome')
    ax2.set_yticks([])
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Tails (0)', 'Heads (1)'])
    ax2.set_title('Individual Coin Flips (Stacked)')
    ax2.set_xlim(-0.5, 1.5)
    ax2.set_ylim(-0.25, 1.5)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()

    return fig

def demonstrate_likelihood_concept(py=False):
    """Visual intuition: What coin bias explains our data best?"""
    
    # Observed data: 10 coin flips
    observed_flips = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1]  # 1=Heads, 0=Tails
    n_heads = sum(observed_flips)
    n_total = len(observed_flips)
    
    print(f"Observed data: {n_heads} heads in {n_total} flips")
    str_list = ['H' if x == 1 else 'T' for x in observed_flips]
    str = ' '.join(str_list)
    print(f"{str}")
    print()
    
    # Try different possible values of p (probability of heads)
    p_values = np.linspace(0.01, 0.99, 100)
    likelihoods = []
    
    binomial_coef = comb(n_total, n_heads)
    
    for p in p_values:
        # Likelihood = probability of observing our data given this p
        likelihood = binomial_coef * (p ** n_heads) * ((1-p) ** (n_total - n_heads))
        likelihoods.append(likelihood)
    
    likelihoods = np.array(likelihoods)
    
    # Find MLE
    mle_idx = np.argmax(likelihoods)
    mle_p = p_values[mle_idx]
    
    # Analytical MLE (for comparison)
    analytical_mle = n_heads / n_total
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Left plot: Likelihood function
    ax1.plot(p_values, likelihoods, linewidth=3, color='steelblue', label='Likelihood L(p)')
    ax1.axvline(mle_p, color='red', linewidth=3, linestyle='--', 
                label=f'MLE = {mle_p:.3f}')
    ax1.scatter(mle_p, likelihoods[mle_idx], s=400, color='red', marker='*', 
                edgecolors='darkred', linewidths=2, zorder=5)
    
    # Mark a few other values for comparison
    for test_p, color, label in [(0.3, 'orange', 'p=0.3'), 
                                   (0.5, 'green', 'p=0.5 (fair)'),
                                   (0.9, 'purple', 'p=0.9')]:
        if abs(test_p - mle_p) > 0.05:  # Don't mark if too close to MLE
            test_idx = np.argmin(np.abs(p_values - test_p))
            ax1.scatter(test_p, likelihoods[test_idx], s=200, color=color, 
                       marker='o', edgecolors='black', linewidths=1.5, alpha=0.7,
                       label=label)
    
    ax1.set_xlabel('Parameter p (probability of heads)', fontsize=12)
    ax1.set_ylabel('Likelihood L(p | data)', fontsize=12)
    ax1.set_title('Likelihood Function', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Interpretation
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    interpretation = f"""
    THE LIKELIHOOD PRINCIPLE
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    QUESTION:
    "Which value of p makes our observed data most probable?"
    
    DATA: {n_heads} heads, {n_total-n_heads} tails
    
    ANSWER:
    """+r"$\hat{p}$"+f""" = {analytical_mle:.2f} (= {n_heads}/{n_total})
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    INTERPRETATION:
    
    • p=0.3: "These data are unlikely if coin had p=0.3"
      
    • p=0.5: "Fair coin? Possible, but not the most likely"
      
    • p=0.8: "BEST explanation: Our data are most probable under this parameter"
    
    • p=0.9: "Too extreme, doesn't fit data as well"
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━
    MLE = "Most Likely Explanation"
    """
    
    ax2.text(0.5, 0.5, interpretation, fontsize=11, family='monospace',
             verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    print(f"Numerical MLE: {mle_p:.4f}")
    print(f"Analytical MLE: {analytical_mle:.4f}")
    print(f"They match (as they should)")

    return fig

def plot_binomial(py=False):
    
    # Parameters
    n = 10  # number of trials
    p_values = [0.1, 0.3, 0.5, 0.8, 0.9]
    k = np.arange(0, n + 1)  # possible outcomes (0 to 10)

    # Create plot
    fig = plt.figure(figsize=(10, 6))

    # Plot each binomial distribution
    for p in p_values:
        pmf = stats.binom.pmf(k, n, p)
        plt.plot(k, pmf, marker='o', linestyle='None', markersize=8, label=f'p = {p}', alpha=0.7)

    plt.xlabel('Number of Heads (k)', fontsize=12)
    plt.ylabel('Probability P(X = k)', fontsize=12)
    plt.title(f'Binomial Distribution PMF for n = {n}', fontsize=14, fontweight='bold')
    plt.xticks(k)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()

    return fig

def demonstrate_prior_importance(py=False):
    """Show when prior knowledge helps"""
    
    np.random.seed(42)
    
    # Scenario 1: Small sample
    n_small = 10
    k_small = 9  # 9 spam out of 10
    
    # Scenario 2: Large sample
    n_large = 1000
    k_large = 250  # 250 spam out of 1000
    
    # MLE estimates
    mle_small = k_small / n_small
    mle_large = k_large / n_large
    
    # Known: typical spam rate is around 25%
    prior_belief = 0.25
        
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # For each scenario, show likelihood and prior
    for ax, n, k, title in [(ax1, n_small, k_small, 'Small Sample (n=10)'),
                             (ax2, n_large, k_large, 'Large Sample (n=1000)')]:
        
        p_range = np.linspace(0.01, 0.99, 200)
        
        # Likelihood (normalized for visualization)
        likelihood = p_range**k * (1-p_range)**(n-k)
        likelihood = likelihood / np.max(likelihood)  # Normalize
        
        # Prior: Beta(10, 30) centered around 0.25
        prior = stats.beta.pdf(p_range, 10, 30)
        prior = prior / np.max(prior)  # Normalize for comparison
        
        # Posterior (proportional to likelihood × prior)
        posterior = likelihood * stats.beta.pdf(p_range, 10, 30)
        posterior = posterior / np.max(posterior)  # Normalize
        
        # Plot
        ax.plot(p_range, likelihood, linewidth=3, color='steelblue', 
               label='Likelihood (from data)', alpha=0.7)
        ax.plot(p_range, prior, linewidth=3, color='green', 
               label='Prior (from experience)', alpha=0.7, linestyle='--')
        ax.plot(p_range, posterior, linewidth=4, color='red', 
               label='Posterior (combined)', alpha=0.8)
        
        # Mark estimates
        mle = k / n
        map_estimate = p_range[np.argmax(posterior)]
        
        ax.axvline(mle, color='steelblue', linewidth=2, linestyle=':', 
                  label=f'MLE = {mle:.2f}')
        ax.axvline(map_estimate, color='red', linewidth=2, linestyle=':', 
                  label=f'MAP ≈ {map_estimate:.2f}')
        ax.axvline(prior_belief, color='green', linewidth=2, linestyle=':', 
                  label=f'Prior peak = {prior_belief:.2f}')
        
        ax.set_xlabel('p (spam rate)', fontsize=12)
        ax.set_ylabel('Normalized density', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    return fig

def compare_mle_map(py=False):
    """Detailed comparison of MLE and MAP"""
    
    # Generate data
    np.random.seed(42)
    true_p = 0.3
    n = 20
    data = np.random.binomial(1, true_p, n)
    k = np.sum(data)
    
    print(f"Data: {k} successes in {n} trials (true p = {true_p})")
    print()
    
    # MLE
    mle = k / n
    
    # MAP with different priors
    # Prior 1: Beta(2, 5) - weak prior favoring low p
    # Prior 2: Beta(10, 25) - stronger prior around p=0.28
    
    p_range = np.linspace(0.01, 0.99, 300)
    
    # Likelihood
    likelihood = p_range**k * (1-p_range)**(n-k)
    
    # Priors
    prior_weak = stats.beta.pdf(p_range, 2, 5)
    prior_strong = stats.beta.pdf(p_range, 10, 25)
    
    # Posteriors (unnormalized)
    posterior_weak = likelihood * prior_weak
    posterior_strong = likelihood * prior_strong
    
    # MAP estimates
    map_weak = p_range[np.argmax(posterior_weak)]
    map_strong = p_range[np.argmax(posterior_strong)]
    
    # For Beta prior, MAP has closed form:
    # MAP = (k + α - 1) / (n + α + β - 2)
    map_weak_analytical = (k + 2 - 1) / (n + 2 + 5 - 2)
    map_strong_analytical = (k + 10 - 1) / (n + 10 + 25 - 2)
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Weak prior
    ax = axes[0, 0]
    ax.plot(p_range, likelihood/np.max(likelihood), linewidth=3, color='steelblue', 
           label='Likelihood', alpha=0.7)
    ax.plot(p_range, prior_weak/np.max(prior_weak), linewidth=3, color='green', 
           label='Prior: Beta(2,5)', alpha=0.7, linestyle='--')
    ax.plot(p_range, posterior_weak/np.max(posterior_weak), linewidth=4, color='red', 
           label='Posterior', alpha=0.8)
    ax.axvline(mle, color='steelblue', linewidth=2, linestyle=':', 
              label=f'MLE = {mle:.3f}')
    ax.axvline(map_weak, color='red', linewidth=2, linestyle=':', 
              label=f'MAP = {map_weak:.3f}')
    ax.axvline(true_p, color='gold', linewidth=3, linestyle='--', 
              label=f'True = {true_p}')
    ax.set_xlabel('p', fontsize=12)
    ax.set_ylabel('Normalized density', fontsize=12)
    ax.set_title('Weak Prior: Small Effect on Estimate', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Strong prior
    ax = axes[0, 1]
    ax.plot(p_range, likelihood/np.max(likelihood), linewidth=3, color='steelblue', 
           label='Likelihood', alpha=0.7)
    ax.plot(p_range, prior_strong/np.max(prior_strong), linewidth=3, color='green', 
           label='Prior: Beta(10,25)', alpha=0.7, linestyle='--')
    ax.plot(p_range, posterior_strong/np.max(posterior_strong), linewidth=4, color='red', 
           label='Posterior', alpha=0.8)
    ax.axvline(mle, color='steelblue', linewidth=2, linestyle=':', 
              label=f'MLE = {mle:.3f}')
    ax.axvline(map_strong, color='red', linewidth=2, linestyle=':', 
              label=f'MAP = {map_strong:.3f}')
    ax.axvline(true_p, color='gold', linewidth=3, linestyle='--', 
              label=f'True = {true_p}')
    ax.set_xlabel('p', fontsize=12)
    ax.set_ylabel('Normalized density', fontsize=12)
    ax.set_title('Strong Prior: Pulls Estimate Toward Prior', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Effect of sample size
    ax = axes[1, 0]
    sample_sizes = [5, 10, 20, 50, 100, 200]
    mle_estimates = []
    map_estimates = []
    
    for n_sim in sample_sizes:
        data_sim = np.random.binomial(1, true_p, n_sim)
        k_sim = np.sum(data_sim)
        mle_estimates.append(k_sim / n_sim)
        # MAP with Beta(10, 25) prior
        map_estimates.append((k_sim + 10 - 1) / (n_sim + 10 + 25 - 2))
    
    ax.plot(sample_sizes, mle_estimates, 'o-', linewidth=3, markersize=10, 
           color='steelblue', label='MLE')
    ax.plot(sample_sizes, map_estimates, 's-', linewidth=3, markersize=10, 
           color='red', label='MAP (strong prior)')
    ax.axhline(true_p, color='gold', linewidth=3, linestyle='--', label='True value')
    ax.set_xlabel('Sample size n', fontsize=12)
    ax.set_ylabel('Estimate', fontsize=12)
    ax.set_title('MAP vs MLE: Effect of Sample Size', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    comparison = f"""
    MLE vs MAP COMPARISON
    ═════════════════════════════════════════════
    
    DATA: {k}/{n} successes ({k/n})
    TRUE VALUE: p = {true_p}
    
    ─────────────────────────────────────────────
    
                      MLE        MAP(weak)   MAP(strong)
    Estimate:        {mle:.3f}      {map_weak:.3f}       {map_strong:.3f}
    Error:           {abs(mle-true_p):.3f}      {abs(map_weak-true_p):.3f}       {abs(map_strong-true_p):.3f}
    
    ─────────────────────────────────────────────
    
    """
    
    ax.text(0.5, 0.5, comparison, fontsize=10, family='monospace',
           verticalalignment='center', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
    
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    # Print summary
    print("=" * 70)
    print("RESULTS:")
    print("=" * 70)
    print(f"MLE:              {mle:.4f}  (error: {abs(mle-true_p):.4f})")
    print(f"MAP (weak prior): {map_weak:.4f}  (error: {abs(map_weak-true_p):.4f})")
    print(f"MAP (strong prior): {map_strong:.4f}  (error: {abs(map_strong-true_p):.4f})")
    print(f"True value:       {true_p:.4f}")
    print("=" * 70)
    
    return fig

def demonstrate_map_regularization(py=False):
    """Show MAP-regularization connection"""
    # Generate toy data for linear regression
    np.random.seed(42)
    n = 50
    X = np.random.randn(n, 1)
    true_w = 2.0
    y = true_w * X.ravel() + np.random.randn(n) * 0.5

    # Add X^2 to X^10 features (will cause overfitting without regularization)
    X_poly = np.column_stack([X**i for i in range(1, 11)])

    # Three approaches:
    # 1. MLE (= Ordinary Least Squares, no regularization)
    # 2. MAP with Gaussian prior (= Ridge)
    # 3. MAP with strong Gaussian prior (= Strong Ridge)

    from sklearn.linear_model import Ridge

    model_mle = Ridge(alpha=0.0)  # No regularization = MLE
    model_map_weak = Ridge(alpha=1.0)  # Weak prior
    model_map_strong = Ridge(alpha=100.0)  # Strong prior

    model_mle.fit(X_poly, y)
    model_map_weak.fit(X_poly, y)
    model_map_strong.fit(X_poly, y)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Fitted curves
    ax = axes[0, 0]
    X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
    X_test_poly = np.column_stack([X_test**i for i in range(1, 11)])

    ax.scatter(X, y, alpha=0.6, s=80, color='gray', edgecolors='black', label='Data')
    ax.plot(X_test, model_mle.predict(X_test_poly), linewidth=3, color='steelblue', 
        label='MLE (no regularization)', alpha=0.8)
    ax.plot(X_test, model_map_weak.predict(X_test_poly), linewidth=3, color='orange', 
        label='MAP (weak prior, λ=1)', alpha=0.8)
    ax.plot(X_test, model_map_strong.predict(X_test_poly), linewidth=3, color='red', 
        label='MAP (strong prior, λ=100)', alpha=0.8)
    ax.plot(X_test, true_w * X_test, linewidth=2, color='green', linestyle='--',
        label='True function')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Fitted Models: MAP Prevents Overfitting', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-8, 8)

    # Plot 2: Coefficient magnitudes
    ax = axes[0, 1]
    features = [f'x^{i}' for i in range(1, 11)]
    x_pos = np.arange(len(features))
    width = 0.25

    ax.bar(x_pos - width, model_mle.coef_, width, label='MLE', color='steelblue', alpha=0.7)
    ax.bar(x_pos, model_map_weak.coef_, width, label='MAP (λ=1)', color='orange', alpha=0.7)
    ax.bar(x_pos + width, model_map_strong.coef_, width, label='MAP (λ=100)', color='red', alpha=0.7)

    ax.set_xlabel('Feature', fontsize=12)
    ax.set_ylabel('Coefficient value', fontsize=12)
    ax.set_title('MAP Shrinks Coefficients (Regularization)', fontsize=13, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(features, rotation=45)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(0, color='black', linewidth=0.5)

    # Plot 3: Prior distributions
    ax = axes[1, 0]
    w_range = np.linspace(-5, 5, 200)

    # Gaussian priors with different variances
    sigma_weak = 1/np.sqrt(1.0)  # From λ=1
    sigma_strong = 1/np.sqrt(100.0)  # From λ=100

    prior_flat = np.ones_like(w_range)  # Uniform (MLE)
    prior_weak = stats.norm.pdf(w_range, 0, sigma_weak)
    prior_strong = stats.norm.pdf(w_range, 0, sigma_strong)

    ax.plot(w_range, prior_flat/np.max(prior_flat), linewidth=3, color='steelblue', 
        label='MLE: Uniform (no prior)', linestyle='--')
    ax.plot(w_range, prior_weak, linewidth=3, color='orange', 
        label='MAP: N(0, 1²) weak prior')
    ax.plot(w_range, prior_strong, linewidth=3, color='red', 
        label='MAP: N(0, 0.1²) strong prior')

    ax.set_xlabel('Weight w', fontsize=12)
    ax.set_ylabel('Prior density p(w)', fontsize=12)
    ax.set_title('Prior Distributions on Weights', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Plot 4: Explanation
    ax = axes[1, 1]
    ax.axis('off')

    explanation = f"""
    🎯 MAP = REGULARIZATION
    ══════════════════════════════════════════

    OBJECTIVE FUNCTIONS:

    MLE (no regularization):
    min ∑(yᵢ - wᵀxᵢ)²

    MAP with Gaussian prior:
    min ∑(yᵢ - wᵀxᵢ)² + λ∑wⱼ²
    └─────┬─────┘   └───┬───┘
    likelihood    prior

    ──────────────────────────────────────────

    INTERPRETATION:

    • λ = 0:     No prior, MLE solution
                → Can overfit

    • λ = 1:     Weak prior "weights ≈ 0"
                → Some regularization

    • λ = 100:   Strong prior "weights ≈ 0"
                → Heavy regularization

    ──────────────────────────────────────────

    💡 KEY INSIGHTS:

    1. Regularization = Bayesian prior!
    2. λ ↔ prior strength
    3. Gaussian prior → L2 (Ridge)
    4. Laplace prior → L1 (Lasso)

    ══════════════════════════════════════════
    """

    ax.text(0.5, 0.5, explanation, fontsize=10, family='monospace',
        verticalalignment='center', horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.4))

    plt.tight_layout()
    plt.show()

    # Print coefficient norms
    print("=" * 70)
    print("COEFFICIENT NORMS (||w||₂):")
    print("=" * 70)
    print(f"MLE (no regularization):  {np.linalg.norm(model_mle.coef_):.4f}")
    print(f"MAP (λ=1):                {np.linalg.norm(model_map_weak.coef_):.4f}")
    print(f"MAP (λ=100):              {np.linalg.norm(model_map_strong.coef_):.4f}")
    print()
    print("💡 Stronger prior → smaller coefficients → less overfitting!")
    print("=" * 70)
    
    return fig

if __name__ == "__main__":
    print("Utility Functions for Point Estimation")
    #generate_hook_data(n_experiments=50, py=True)
    #demonstrate_estimator_concept(py=True)
    #explore_sampling_distribution(true_mean=5, true_std=2, sample_size=30, n_experiments=1000)
    #visualize_bias_variance_tradeoff()
    #demonstrate_likelihood_concept(py=True)
    #plot_binomial(py=True)
    #demonstrate_prior_importance(py=True)
    compare_mle_map(py=True)
    