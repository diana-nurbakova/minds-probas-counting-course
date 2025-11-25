import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D

plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji']

def demonstrate_the_crisis(py=False):
        
        """Visualize why point estimates are dangerous"""
        np.random.seed(42)

        
        true_accuracy = 0.91  # Unknown "true" performance
        test_set_size = 500
        
        theoretical_std = np.sqrt(true_accuracy * (1 - true_accuracy) / test_set_size)
        ci_lower = true_accuracy - 1.96 * theoretical_std
        ci_upper = true_accuracy + 1.96 * theoretical_std
        print(f"[{ci_lower}, {ci_upper}")
        
        # Simulation: Train 30 models on different test sets
        n_experiments = 30

        observed_accuracies = []
        for _ in range(n_experiments):
                # Simulate test set results
                correct = np.random.binomial(test_set_size, true_accuracy)
                accuracy = correct / test_set_size
                observed_accuracies.append(accuracy)

        observed_accuracies = np.array(observed_accuracies)

        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Multiple experiments
        ax1.scatter(range(n_experiments), observed_accuracies, s=200, 
                alpha=0.7, c='steelblue', edgecolors='black', linewidth=2)
        ax1.axhline(true_accuracy, color='gold', linewidth=4, linestyle='--', 
                label=f'True accuracy = {true_accuracy:.1%}', zorder=5)

        # Highlight the "your" experiment
        your_experiment = 1
        your_accuracy = observed_accuracies[your_experiment]
        ax1.scatter(your_experiment, your_accuracy, s=500, c='red', marker='*', 
                edgecolors='darkred', linewidths=3, label=f'Your estimate = {your_accuracy:.1%}', 
                zorder=10)

        # Add danger zone
        ax1.axhspan(ci_upper, 0.96, alpha=0.2, color='red', label='Overconfident zone')
        ax1.axhspan(ci_lower, ci_upper, alpha=0.2, color='green', label='Safe zone')

        ax1.set_xlabel('Different Test Sets', fontsize=13)
        ax1.set_ylabel('Observed Accuracy', fontsize=13)
        ax1.set_title('The Danger of Point Estimates\n(20 different test sets, same model)', 
                        fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11, loc='lower right')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.87, 0.96)

        # Right: Distribution
        ax2.hist(observed_accuracies, bins=15, density=True, alpha=0.7, 
                color='steelblue', edgecolor='black', linewidth=2)

        # Overlay theoretical distribution
        x_range = np.linspace(0.87, 0.96, 200)
        
        ax2.plot(x_range, stats.norm.pdf(x_range, true_accuracy, theoretical_std), 
                'g-', linewidth=4, label='Theoretical distribution')

        ax2.axvline(true_accuracy, color='gold', linewidth=4, linestyle='--', 
                label=f'True = {true_accuracy:.1%}')
        ax2.axvline(your_accuracy, color='red', linewidth=4, linestyle=':', 
                label=f'Your estimate = {your_accuracy:.1%}')

        # Show confidence interval
        
        ax2.axvspan(ci_lower, ci_upper, alpha=0.3, color='green', 
                label=f'95% CI: [{ci_lower:.1%}, {ci_upper:.1%}]')

        ax2.set_xlabel('Accuracy', fontsize=13)
        ax2.set_ylabel('Density', fontsize=13)
        ax2.set_title('Sampling Distribution of Accuracy\n(Your estimate is just one draw)', 
                        fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if py:
                plt.show()
        else:
                plt.close()
        
        return fig

def visualize_ci_interpretation(py=False):
    """The definitive visualization of what CI means"""
    
    np.random.seed(42)
    
    # Setup: estimating mean of normal distribution
    true_mean = 10  # The "fish" - fixed location
    true_std = 3
    sample_size = 30
    n_experiments = 100
    confidence_level = 0.95
    
    # Critical value for 95% CI
    z_critical = stats.norm.ppf(1 - (1-confidence_level)/2)
    
    # Run many experiments
    intervals = []
    contains_true = []
    
    for i in range(n_experiments):
        # Draw sample
        sample = np.random.normal(true_mean, true_std, sample_size)
        
        # Compute CI
        sample_mean = np.mean(sample)
        se = true_std / np.sqrt(sample_size)  # Standard error
        margin = z_critical * se
        
        ci_lower = sample_mean - margin
        ci_upper = sample_mean + margin
        
        intervals.append((ci_lower, ci_upper, sample_mean))
        contains_true.append(ci_lower <= true_mean <= ci_upper)
    
    # Count coverage
    coverage = np.mean(contains_true)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Show all intervals
    for i, ((lower, upper, point), contains) in enumerate(zip(intervals[:50], contains_true[:50])):
        color = 'green' if contains else 'red'
        alpha = 0.7 if contains else 1.0
        linewidth = 1.5 if contains else 3
        
        # Draw interval
        ax1.plot([lower, upper], [i, i], color=color, alpha=alpha, linewidth=linewidth)
        # Draw point estimate
        ax1.scatter(point, i, s=50, color=color, alpha=alpha, zorder=5)
    
    # Draw true mean
    ax1.axvline(true_mean, color='gold', linewidth=4, linestyle='--', 
               label=f'True μ = {true_mean}', zorder=10)
    
    ax1.set_xlabel('Value', fontsize=13)
    ax1.set_ylabel('Experiment Number', fontsize=13)
    ax1.set_title(f'95% Confidence Intervals from 50 Experiments\n'
                 f'Green = captured true value | Red = missed', 
                 fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='x')
    ax1.set_ylim(-1, 50)
    
    # Right: Interpretation
    ax2.axis('off')
    
    n_captured = sum(contains_true)
    n_missed = n_experiments - n_captured
    
    interpretation = f"""
    FREQUENTIST INTERPRETATION
    ═══════════════════════════════════════════════════
    
    PROCEDURE: 
    1. Draw random sample of size n={sample_size}
    2. Compute 95% CI: """ + r"$[\bar{X} - 1.96\times SE, \bar{X} + 1.96\times SE]$" + f"""
    3. Repeat {n_experiments} times
    
    ───────────────────────────────────────────────────
    
    RESULTS:
    • Intervals that captured true μ:  {n_captured}/{n_experiments} ({coverage:.1%})
    • Intervals that missed true μ:    {n_missed}/{n_experiments} ({1-coverage:.1%})
    
    ───────────────────────────────────────────────────
    
    CORRECT INTERPRETATION:
    
    "If we repeat this procedure many times,
    about 95% of the intervals will contain the true parameter."
    
    ───────────────────────────────────────────────────
    
    WRONG INTERPRETATION:
    
    "There's a 95% probability that θ is in this specific interval."
    
    (θ is fixed. The interval is random)
        
    ═══════════════════════════════════════════════════
    
    PRACTICAL MEANING:
    
    "I'm 95% confident" means:
    "My procedure is reliable 95% of the time"
    
    NOT: "θ has 95% chance of being here"
    
    ═══════════════════════════════════════════════════
    """
    
    ax2.text(0.5, 0.5, interpretation, fontsize=10, family='monospace',
            verticalalignment='center', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.4))
    
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
        
    return fig

def dissect_confidence_interval(py=False):
    """Break down the anatomy of a CI"""
    
    # Sample data
    np.random.seed(42)
    data = np.random.normal(100, 15, 50)
    
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    
    # For 95% CI
    confidence_level = 0.95
    alpha = 1 - confidence_level
    
    # Using t-distribution (more accurate for small samples)
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Standard error
    se = sample_std / np.sqrt(n)
    
    # Margin of error
    margin_of_error = t_critical * se
    
    # Confidence interval
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: The components
    ax = axes[0, 0]
    ax.barh(['Point Estimate', 'Standard Error', 'Critical Value', 'Margin of Error'], 
           [sample_mean/10, se, t_critical, margin_of_error], 
           color=['steelblue', 'orange', 'green', 'red'], alpha=0.7)
    
    for i, (label, value) in enumerate([('Point Estimate', sample_mean), 
                                         ('Standard Error', se),
                                         ('Critical Value', t_critical),
                                         ('Margin of Error', margin_of_error)]):
        ax.text(value/10 if i==0 else value, i, f'  {value:.2f}', 
               va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Components of Confidence Interval', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Visual representation
    ax = axes[0, 1]
    
    # Draw the interval
    ax.plot([ci_lower, ci_upper], [0.5, 0.5], linewidth=8, color='steelblue', 
           label=f'95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]')
    ax.scatter(sample_mean, 0.5, s=500, color='red', marker='o', 
              edgecolors='darkred', linewidths=3, label=f'Point estimate: {sample_mean:.2f}', 
              zorder=10)
    
    # Show margins
    ax.plot([sample_mean, ci_lower], [0.5, 0.5], linewidth=4, color='orange', 
           linestyle='--', alpha=0.7)
    ax.plot([sample_mean, ci_upper], [0.5, 0.5], linewidth=4, color='orange', 
           linestyle='--', alpha=0.7)
    
    # Annotations
    ax.annotate('', xy=(ci_lower, 0.6), xytext=(sample_mean, 0.6),
               arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    ax.text((ci_lower + sample_mean)/2, 0.65, f'Margin = {margin_of_error:.2f}',
           ha='center', fontsize=11, color='orange', fontweight='bold')
    
    ax.annotate('', xy=(sample_mean, 0.6), xytext=(ci_upper, 0.6),
               arrowprops=dict(arrowstyle='<->', color='orange', lw=2))
    ax.text((ci_upper + sample_mean)/2, 0.65, f'Margin = {margin_of_error:.2f}',
           ha='center', fontsize=11, color='orange', fontweight='bold')
    
    ax.set_ylim(0.3, 0.8)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Confidence Interval Structure', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_yticks([])
    
    # Plot 3: Effect of confidence level
    ax = axes[1, 0]
    
    confidence_levels = [0.80, 0.90, 0.95, 0.99]
    colors = ['lightblue', 'steelblue', 'darkblue', 'navy']
    
    for i, (conf, color) in enumerate(zip(confidence_levels, colors)):
        t_crit = stats.t.ppf(1 - (1-conf)/2, df=n-1)
        margin = t_crit * se
        ci_l = sample_mean - margin
        ci_u = sample_mean + margin
        
        ax.plot([ci_l, ci_u], [i, i], linewidth=6, color=color, 
               label=f'{conf:.0%} CI: width = {ci_u-ci_l:.2f}')
        ax.scatter(sample_mean, i, s=200, color='red', marker='|', linewidths=3, zorder=5)
    
    ax.set_ylim(-0.5, len(confidence_levels)-0.5)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Higher Confidence → Wider Interval', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_yticks(range(len(confidence_levels)))
    ax.set_yticklabels([f'{c:.0%}' for c in confidence_levels])
    ax.set_ylabel('Confidence Level', fontsize=12)
    
    # Plot 4: Effect of sample size
    ax = axes[1, 1]
    
    sample_sizes = [10, 30, 50, 100, 200]
    colors_n = plt.cm.Greens(np.linspace(0.4, 0.9, len(sample_sizes)))
    
    for i, (n_sim, color) in enumerate(zip(sample_sizes, colors_n)):
        se_sim = sample_std / np.sqrt(n_sim)
        t_crit_sim = stats.t.ppf(1 - alpha/2, df=n_sim-1)
        margin_sim = t_crit_sim * se_sim
        ci_l = sample_mean - margin_sim
        ci_u = sample_mean + margin_sim
        
        ax.plot([ci_l, ci_u], [i, i], linewidth=6, color=color, 
               label=f'n={n_sim}: width = {ci_u-ci_l:.2f}')
        ax.scatter(sample_mean, i, s=200, color='red', marker='|', linewidths=3, zorder=5)
    
    ax.set_ylim(-0.5, len(sample_sizes)-0.5)
    ax.set_xlabel('Value', fontsize=12)
    ax.set_title('Larger Sample → Narrower Interval', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_yticks(range(len(sample_sizes)))
    ax.set_yticklabels([f'n={n}' for n in sample_sizes])
    ax.set_ylabel('Sample Size', fontsize=12)
    
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    return fig

def t_distribution_viz(py=False):
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: t vs normal for different df
    ax = axes[0, 0]
    x = np.linspace(-4, 4, 500)
    
    # Standard normal
    ax.plot(x, stats.norm.pdf(x), 'k-', linewidth=3, label='Normal (z)', alpha=0.8)
    
    # t-distributions
    dfs = [1, 2, 5, 10, 30]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(dfs)))
    
    for df, color in zip(dfs, colors):
        ax.plot(x, stats.t.pdf(x, df), linewidth=2.5, color=color,
               label=f't(df={df})', alpha=0.7)
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('t-Distribution vs Normal\n(Heavier tails for small df)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 0.45)
    
    # Plot 2: Critical values vs df
    ax = axes[0, 1]
    
    df_range = np.arange(1, 51)
    t_critical_95 = [stats.t.ppf(0.975, df) for df in df_range]
    z_critical_95 = stats.norm.ppf(0.975)
    
    ax.plot(df_range, t_critical_95, 'o-', linewidth=2.5, markersize=5,
           color='red', label='t critical value (95%)')
    ax.axhline(z_critical_95, color='blue', linewidth=3, linestyle='--',
              label=f'z = {z_critical_95:.3f} (normal)')
    
    # Annotate key points
    ax.annotate(f't(df=5) = {stats.t.ppf(0.975, 5):.3f}',
               xy=(5, stats.t.ppf(0.975, 5)), xytext=(10, 2.8),
               arrowprops=dict(arrowstyle='->', color='red', lw=2),
               fontsize=10, color='red', weight='bold')
    
    ax.annotate(f't(df=30) ≈ z',
               xy=(30, stats.t.ppf(0.975, 30)), xytext=(35, 2.2),
               arrowprops=dict(arrowstyle='->', color='green', lw=2),
               fontsize=10, color='green', weight='bold')
    
    ax.set_xlabel('Degrees of Freedom', fontsize=12)
    ax.set_ylabel('Critical Value (95% CI)', fontsize=12)
    ax.set_title('Critical Values Converge to Normal\n(Rule of thumb: df ≥ 30)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(1.8, 4.5)
    
    # Plot 3: Tail comparison
    ax = axes[1, 0]
    
    x_tail = np.linspace(2, 5, 200)
    
    ax.fill_between(x_tail, 0, stats.norm.pdf(x_tail), 
                    alpha=0.4, color='blue', label='Normal tail')
    ax.fill_between(x_tail, 0, stats.t.pdf(x_tail, df=5),
                    alpha=0.4, color='red', label='t(df=5) tail')
    
    ax.plot(x_tail, stats.norm.pdf(x_tail), 'b-', linewidth=3, alpha=0.8)
    ax.plot(x_tail, stats.t.pdf(x_tail, df=5), 'r-', linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Why t-Distribution Has Heavier Tails\n(More probability in extremes)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = """
    t-DISTRIBUTION SUMMARY
    ═══════════════════════════════════════════════════════
    
    WHEN TO USE:
    ✓ Population standard deviation σ is UNKNOWN
    ✓ Using sample standard deviation s instead
    ✓ Constructing CI for mean
    
    ───────────────────────────────────────────────────────
    
    KEY FORMULA:
    """ + r"$t = (\bar{x} - \mu) / (s/\sqrt{n})  \sim  t(df = n-1)$" + """
    
    CI: """ + r"$\bar{x} \pm t(\alpha/2, n-1) \times s/\sqrt{n}$" + """
    
    ───────────────────────────────────────────────────────
    
    DEGREES OF FREEDOM:
    df = n - 1
    
    • Small df (< 10):  Much wider than normal
    • Medium df (10-30): Noticeably wider
    • Large df (≥ 30):   Nearly identical to normal
    
    ───────────────────────────────────────────────────────
    
    COMPARISON WITH NORMAL:
    
                    Normal (z)    t-distribution
    Tails:          Lighter       Heavier
    Use when:       σ known       σ unknown (usual!)
    Critical val:   Fixed         Depends on df
    Converges to:   -             Normal (as df↑)
    
    ───────────────────────────────────────────────────────
    
    PRACTICAL RULE:
    • Always use t when σ unknown (safe choice)
    • Software defaults to t (R, Python, etc.)
    • For n ≥ 30: t ≈ z (but use t anyway)
    
    ═══════════════════════════════════════════════════════
    """
    
    ax.text(0.5, 0.5, summary, fontsize=9, family='monospace',
           verticalalignment='center', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    return fig

def chi_2_distribution_viz(py=False):
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Chi-square for different df
    ax = axes[0, 0]
    x = np.linspace(0, 30, 500)
    
    dfs = [1, 2, 3, 5, 10, 20]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(dfs)))
    
    for df, color in zip(dfs, colors):
        ax.plot(x, stats.chi2.pdf(x, df), linewidth=2.5, color=color,
               label=f'χ²(df={df})', alpha=0.7)
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('χ² Distribution for Different Degrees of Freedom\n(Right-skewed, becomes more symmetric with df)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 30)
    
    # Plot 2: Mean and shape
    ax = axes[0, 1]
    
    dfs_plot = [2, 5, 10, 20]
    x_max = 50
    x = np.linspace(0, x_max, 500)
    
    for df in dfs_plot:
        pdf = stats.chi2.pdf(x, df)
        ax.plot(x, pdf, linewidth=3, label=f'df={df}', alpha=0.7)
        ax.axvline(df, color='black', linestyle='--', alpha=0.3)
        ax.text(df, np.max(pdf) * 0.5, f'μ={df}', 
               rotation=90, va='bottom', fontsize=9)
    
    ax.set_xlabel('Value', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('χ² Distribution: Mean = df\n(Vertical lines show mean)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, x_max)
    
    # Plot 3: Critical values
    ax = axes[1, 0]
    
    df_range = np.arange(1, 51)
    chi2_lower_95 = [stats.chi2.ppf(0.025, df) for df in df_range]
    chi2_upper_95 = [stats.chi2.ppf(0.975, df) for df in df_range]
    
    ax.plot(df_range, chi2_lower_95, 'o-', linewidth=2.5, markersize=4,
           color='blue', label='Lower critical value (2.5%)')
    ax.plot(df_range, chi2_upper_95, 'o-', linewidth=2.5, markersize=4,
           color='red', label='Upper critical value (97.5%)')
    ax.plot(df_range, df_range, 'g--', linewidth=2, 
           label='Mean (= df)', alpha=0.7)
    
    ax.set_xlabel('Degrees of Freedom', fontsize=12)
    ax.set_ylabel('Critical Value', fontsize=12)
    ax.set_title('χ² Critical Values for 95% CI\n(Asymmetric around mean!)', 
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Summary
    ax = axes[1, 1]
    ax.axis('off')
    
    summary = """
    χ² DISTRIBUTION SUMMARY
    ═══════════════════════════════════════════════════════
    DEFINITION:
    χ²(k) = Z₁² + Z₂² + ... + Z_k²
    (Sum of k squared standard normals)
    ───────────────────────────────────────────────────────    
    KEY PROPERTIES:
    • Domain: [0, ∞) (always positive)
    • Shape: Right-skewed (asymmetric)
    • Parameter: k (degrees of freedom)
    • Mean: k
    • Variance: 2k
    ───────────────────────────────────────────────────────    
    DEGREES OF FREEDOM EFFECT:
    
    Small df (< 5):  Very skewed, peak near 0
    Medium df (5-20): Moderately skewed
    Large df (> 30):  Nearly symmetric
    
    As df → ∞: χ²(df) → Normal
    ───────────────────────────────────────────────────────
    CONNECTION TO SAMPLE VARIANCE:
    
    (n-1)s² / σ²  ~  χ²(n-1)
    
    This is the key for CI of variance    
    ───────────────────────────────────────────────────────    
    USED FOR:
    ✓ Confidence intervals for variance
    ✓ Hypothesis tests for variance
    ✓ Goodness-of-fit tests
    ✓ Independence tests (contingency tables)    
    ═══════════════════════════════════════════════════════
    """
    
    ax.text(0.5, 0.1, summary, fontsize=9, family='monospace',
           verticalalignment='center', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    return fig
    
if __name__ == '__main__':
    print('Confidence Intervals')
    #demonstrate_the_crisis(py=True)
    #visualize_ci_interpretation(py=True)
    #dissect_confidence_interval(py=True)
    #t_distribution_viz(py=True)
    chi_2_distribution_viz(py=True)