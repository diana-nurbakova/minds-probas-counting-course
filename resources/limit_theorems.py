import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
from scipy import stats

# dictionary of some available scipy.stats distributions with their configurations
DISTRIBUTIONS = {
    'uniform': {
        'dist': stats.uniform,
        'params': {'loc': 0, 'scale': 1},
        'display_name': 'Uniform(0, 1)',
        'description': 'Continuous uniform distribution'
    },
    'exponential': {
        'dist': stats.expon,
        'params': {'scale': 1.0},
        'display_name': 'Exponential(λ=1)',
        'description': 'Exponential distribution (memoryless)'
    },
    'normal': {
        'dist': stats.norm,
        'params': {'loc': 0, 'scale': 1},
        'display_name': 'Normal(μ=0, σ=1)',
        'description': 'Standard normal distribution'
    },
    'binomial': {
        'dist': stats.binom,
        'params': {'n': 10, 'p': 0.5},
        'display_name': 'Binomial(n=10, p=0.5)',
        'description': 'Discrete binomial distribution'
    },
    'poisson': {
        'dist': stats.poisson,
        'params': {'mu': 5},
        'display_name': 'Poisson(λ=5)',
        'description': 'Discrete Poisson distribution'
    }
}

    
# Additional visualization: Side-by-side comparison of original vs sample means
def show_transformation(py=False):
    """
    Show the transformation from original distribution to sample means
    to demonstrate the 'magic' of CLT
    """
    
    # Choose one interesting distribution (exponential)
    dist = stats.expon(scale=1)
    dist_name = "Exponential(λ=1)"
    
    sample_size = 30
    n_samples = 2000
    
    np.random.seed(42)
    
    # Generate data
    population = dist.rvs(size=10000)
    sample_means = [dist.rvs(size=sample_size).mean() for _ in range(n_samples)]
    sample_means = np.array(sample_means)
    
    # Get parameters
    true_mean = dist.mean()
    true_std = dist.std()
    theoretical_se = true_std / np.sqrt(sample_size)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Plot 1: Original distribution
    ax1 = axes[0]
    ax1.hist(population, bins=60, density=True, alpha=0.7, 
            color='coral', edgecolor='black', label='Sample data')
    x = np.linspace(0, population.max(), 500)
    ax1.plot(x, dist.pdf(x), 'b-', linewidth=3, label='True PDF')
    ax1.axvline(true_mean, color='green', linestyle='--', linewidth=2.5,
               label=f'μ = {true_mean:.2f}')
    ax1.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title(f'ORIGINAL Distribution\n{dist_name}\n(Right-skewed!)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_xlim(0, 8)
    
    # Plot 2: Arrow showing transformation
    ax2 = axes[1]
    ax2.axis('off')
    ax2.text(0.5, 0.75, '⬇️', fontsize=80, ha='center', va='center', color='#2196f3')
    ax2.text(0.5, 0.55, 'Sample n values\nCalculate mean\nRepeat many times', 
            fontsize=12, ha='center', va='top', 
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8),
            fontweight='bold')
    ax2.text(0.5, 1, 'MAGIC HAPPENS', fontsize=14, ha='center', 
            color='#d63031', fontweight='bold')
    
    # Plot 3: Distribution of sample means
    ax3 = axes[2]
    ax3.hist(sample_means, bins=50, density=True, alpha=0.7,
            color='lightgreen', edgecolor='black', label='Sample means')
    x = np.linspace(sample_means.min(), sample_means.max(), 500)
    normal_pdf = stats.norm.pdf(x, true_mean, theoretical_se)
    ax3.plot(x, normal_pdf, 'b-', linewidth=3, label='Normal fit')
    ax3.axvline(true_mean, color='green', linestyle='--', linewidth=2.5,
               label=f'μ = {true_mean:.2f}')
    ax3.set_xlabel('Sample Mean', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax3.set_title(f'SAMPLE MEANS Distribution\n(n={sample_size})\n(Normal shape!)', 
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    summary = """
Compare the shapes:
   LEFT:  Original """ + f"{dist_name}" + """ - heavily right-skewed
   RIGHT: Sample means - beautiful bell curve (normal)!
The CLT transformed a skewed distribution into a normal one!
   This works for ANY distribution (with finite variance)
"""
    fig.text(0.35, 0.02, summary, ha='left', fontsize=10, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))
    
    plt.suptitle('The Central Limit Theorem Transformation\nOriginal → Sample Means', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    return fig


def demo_clt_cauchy(py=False):
    # Demonstrate when CLT fails - Cauchy distribution
    print("WARNING: When CLT Fails - The Cauchy Distribution\n")

    np.random.seed(42)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Generate Cauchy samples
    n_means = 5000
    sample_sizes = [10, 50, 100, 500]

    for idx, n in enumerate(sample_sizes):
        ax = axes[idx // 2, idx % 2]
        
        # Generate sample means from Cauchy distribution
        sample_means = []
        for _ in range(n_means):
            sample = np.random.standard_cauchy(n)
            sample_means.append(np.mean(sample))
        
        sample_means = np.array(sample_means)
        estimated_std = np.std(sample_means)
        
        # Clip for visualization
        sample_means_display = np.clip(sample_means, -15, 15)
        
        # Plot histogram
        ax.hist(sample_means_display, bins=100, density=True, alpha=0.7, 
                color='darkred', edgecolor='black', range=(-15, 15))
        
        # Try to overlay normal (it won't fit!)
        x = np.linspace(-15, 15, 1000)
        y_normal = stats.norm.pdf(x, 0, estimated_std)
        y_cauchy = stats.cauchy.pdf(x, loc=0, scale=1)
        ax.plot(x, y_normal, 'b--', linewidth=2, label='Normal (bad fit!)', alpha=0.7)
        ax.plot(x, y_cauchy, 'g--', linewidth=2, label='Cauchy(0,1)', alpha=0.7)
        #ax.set_ylim([1e-5, 1])
                
        ax.set_title(f'Sample Size n = {n}\nStill NOT Normal!', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sample Mean')
        ax.set_ylabel('Density')
        ax.set_xlim(-15, 15)
        ax.legend()
        ax.grid(True, alpha=0.3)

    ax.set_yscale('log')
    plt.suptitle('CLT FAILS for Cauchy Distribution (Infinite Variance)', 
                fontsize=14, fontweight='bold', color='darkred')
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    print("\nCauchy Distribution Properties:")
    print("   • Mean: UNDEFINED")
    print("   • Variance: INFINITE")
    print("   • Sample means: Follow Cauchy distribution (same as original!)")
    print("   • CLT: DOES NOT APPLY")
    print("\nMoral: Always check your assumptions! Not all data is CLT-friendly.")
    print("=" * 70)

    return fig

def demo_convergence(py=False):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    n = np.arange(1, 101)

    # Deterministic convergence
    axes[0].plot(n, 1/n, 'b-', linewidth=2, label='1/n')
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5, label='Limit = 0')
    axes[0].fill_between(n, -0.05, 0.05, alpha=0.2, color='green', label='ε = 0.05 band')
    axes[0].set_title('Deterministic Convergence: 1/n → 0', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([-0.1, 1.1])

    # Random convergence (sample means)
    np.random.seed(42)
    sample_means = []
    cumsum = 0
    for i in range(1, 101):
        cumsum += np.random.normal(0, 1)
        sample_means.append(cumsum / i)

    axes[1].plot(n, sample_means, 'b-', linewidth=2, label='Sample mean', alpha=0.7)
    axes[1].axhline(0, color='red', linestyle='--', alpha=0.5, label='True mean = 0')
    axes[1].fill_between(n, -0.2, 0.2, alpha=0.2, color='green', label='ε = 0.2 band')
    axes[1].set_title('Random Convergence: X̄ₙ → μ (wiggly but converging)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Sample Mean')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([-1, 1])

    # Multiple trajectories showing almost sure convergence
    axes[2].set_title('Almost Sure Convergence: Multiple Random Paths', fontsize=14, fontweight='bold')
    for seed in range(10):
        np.random.seed(seed)
        path = []
        cumsum = 0
        for i in range(1, 101):
            cumsum += np.random.normal(0, 1)
            path.append(cumsum / i)
        axes[2].plot(n, path, alpha=0.4, linewidth=1)

    axes[2].axhline(0, color='red', linestyle='--', linewidth=2, label='True mean = 0')
    axes[2].fill_between(n, -0.15, 0.15, alpha=0.2, color='green', label='ε = 0.15 band')
    axes[2].set_ylabel('Sample Mean')
    axes[2].set_xlabel('Sample Size (n)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([-0.8, 0.8])

    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()

    print("\n💡 Key Observation:")
    print("   • Deterministic: Smooth, predictable path")
    print("   • Random: Wiggly but eventually stays close")
    print("   • Almost Sure: ALL paths converge (with probability 1)")

    return fig

def example_no_convergence(py=False):
    """
    Example 1: Sequence with no convergence
    """
    
    print("="*80)
    print("EXAMPLE 1: NO CONVERGENCE")
    print("="*80)
    print("\nSequence: Xₙ = (-1)ⁿ · n")
    print("\nThis sequence oscillates and grows - it does NOT converge!")
    print("="*80)
    
    # Generate sequence
    n_values = np.arange(1, 51)
    X_n = ((-1) ** n_values) * n_values
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Sequence values
    ax1 = axes[0, 0]
    ax1.plot(n_values, X_n, 'bo-', linewidth=2, markersize=6, alpha=0.7)
    ax1.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax1.set_xlabel('n', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Xₙ', fontsize=12, fontweight='bold')
    ax1.set_title('Sequence Values: Xn = (-1)ⁿ · n\nOscillates and Grows', 
                  fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Highlight pattern
    evens = n_values[n_values % 2 == 0]
    odds = n_values[n_values % 2 == 1]
    ax1.plot(evens, X_n[n_values % 2 == 0], 'go', markersize=10, 
            label='Even n (positive)', alpha=0.7)
    ax1.plot(odds, X_n[n_values % 2 == 1], 'ro', markersize=10, 
            label='Odd n (negative)', alpha=0.7)
    ax1.legend(fontsize=10)
    
    # Plot 2: Distance from any proposed limit
    ax2 = axes[0, 1]
    
    # Try several possible "limits"
    proposed_limits = [0, 5, -5, 10]
    for limit in proposed_limits:
        distances = np.abs(X_n - limit)
        ax2.plot(n_values, distances, 'o-', linewidth=2, markersize=4, 
                label=f'|Xn - {limit}|', alpha=0.7)
    
    ax2.set_xlabel('n', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Distance from proposed limit', fontsize=12, fontweight='bold')
    ax2.set_title('Distance Does NOT Go to Zero\n(for any proposed limit)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: "Empirical CDF" at different n (they don't stabilize)
    ax3 = axes[1, 0]
    
    # Show "distributions" at different stages
    stages = [10, 20, 30, 40, 50]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(stages)))
    
    for i, stage in enumerate(stages):
        values = X_n[:stage]
        # Create histogram
        ax3.hist(values, bins=20, alpha=0.3, color=colors[i], 
                edgecolor='black', label=f'First {stage} values')
    
    ax3.set_xlabel('Value', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax3.set_title('"Distribution" Changes - Not Converging!\n(Spreads out more over time)', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Explanation
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    explanation = """
  WHY NO CONVERGENCE? Mathematical Proof:
    
  Claim: Xₙ does not converge to any X
    
  Proof by contradiction:
  Suppose Xₙ → X for some X ∈ ℝ
    
  Then for ε = 1: P(|Xₙ - X| < 1) should → 1
    
  But:
  • For even n: Xₙ = n → +∞
  • For odd n:  Xₙ = -n → -∞
    
  So |Xₙ - X| → ∞ for ANY X!
    
  Therefore: NO convergence (not in distribution, 
  not in probability, not almost surely)
"""
    
    ax4.text(0.05, 1, explanation, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ffebee', alpha=0.9))
    
    plt.suptitle('Example 1: NO Convergence - The Oscillating Sequence', 
                fontsize=16, fontweight='bold', color='#c62828')
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    return fig

def example_convergence_in_distribution_only(py=False):
    """
    Example 2: Convergence in distribution but not in probability
    """
    
    print("="*80)
    print("EXAMPLE 2: CONVERGENCE IN DISTRIBUTION ONLY")
    print("="*80)
    print("\nSequence: Xₙ = Z · (-1)ⁿ, where Z ~ N(0,1)")
    print("\nThe distribution is always N(0,1), but values oscillate!")
    print("="*80)
    
    np.random.seed(42)
    
    # Generate sequence - ONE realization
    Z = np.random.normal(0, 1)  # Draw once
    n_values = np.arange(1, 101)
    X_n = Z * ((-1) ** n_values)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Sequence values (one realization)
    ax1 = axes[0, 0]
    ax1.plot(n_values, X_n, 'bo-', linewidth=2, markersize=4, alpha=0.7)
    ax1.axhline(Z, color='green', linestyle='--', linewidth=2, label=f'Z = {Z:.3f}')
    ax1.axhline(-Z, color='red', linestyle='--', linewidth=2, label=f'-Z = {-Z:.3f}')
    ax1.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax1.set_xlabel('n', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'$X_n$', fontsize=12, fontweight='bold')
    ax1.set_title(f'Single Realization (Z = {Z:.3f})\nValues Keep Flipping', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Multiple realizations
    ax2 = axes[0, 1]
    
    for trial in range(10):
        Z_trial = np.random.normal(0, 1)
        X_n_trial = Z_trial * ((-1) ** n_values)
        ax2.plot(n_values, X_n_trial, alpha=0.5, linewidth=1.5)
    
    ax2.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
    ax2.set_xlabel('n', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'$X_n$', fontsize=12, fontweight='bold')
    ax2.set_title('10 Different Realizations\n All Oscillate, No Convergence to a Value', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: CDFs at different n (they're all the same!)
    ax3 = axes[0, 2]
    
    # Generate many samples for different n
    n_samples = 10000
    x_range = np.linspace(-4, 4, 1000)
    
    for n in [1, 10, 50, 100]:
        # Generate samples
        Z_samples = np.random.normal(0, 1, n_samples)
        X_n_samples = Z_samples * ((-1) ** n)
        
        # Plot empirical CDF
        sorted_samples = np.sort(X_n_samples)
        empirical_cdf = np.arange(1, n_samples + 1) / n_samples
        ax3.plot(sorted_samples, empirical_cdf, linewidth=2, 
                alpha=0.7, label=f'n = {n}')
    
    # Overlay theoretical N(0,1) CDF
    theoretical_cdf = stats.norm.cdf(x_range)
    ax3.plot(x_range, theoretical_cdf, 'k--', linewidth=3, 
            label='N(0,1) CDF', alpha=0.8)
    
    ax3.set_xlabel('x', fontsize=12, fontweight='bold')
    ax3.set_ylabel('CDF(x)', fontsize=12, fontweight='bold')
    ax3.set_title('CDFs at Different n\nAll the Same = N(0,1)', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histograms at different n
    ax4 = axes[1, 0]
    
    for n in [1, 10, 50, 100]:
        Z_samples = np.random.normal(0, 1, n_samples)
        X_n_samples = Z_samples * ((-1) ** n)
        ax4.hist(X_n_samples, bins=50, density=True, alpha=0.3, 
                edgecolor='black', label=f'n = {n}')
    
    # Overlay theoretical PDF
    ax4.plot(x_range, stats.norm.pdf(x_range), 'k-', linewidth=3, 
            label='N(0,1) PDF', alpha=0.8)
    
    ax4.set_xlabel('x', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax4.set_title('Distributions at Different n\nAll N(0,1)', 
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Distance between consecutive values
    ax5 = axes[1, 1]
    
    # Multiple trials
    distances_trials = []
    for trial in range(100):
        Z_trial = np.random.normal(0, 1)
        X_n_trial = Z_trial * ((-1) ** n_values)
        distances = np.abs(np.diff(X_n_trial))
        distances_trials.append(distances)
    
    distances_array = np.array(distances_trials)
    mean_distances = np.mean(distances_array, axis=0)
    
    ax5.plot(n_values[:-1], mean_distances, 'ro-', linewidth=2, markersize=4)
    ax5.axhline(2 * np.sqrt(2/np.pi), color='blue', linestyle='--', 
               linewidth=2, label=r'$E[|X_{n+1} - X_n|] = 2E[|Z|]$')
    ax5.set_xlabel('n', fontsize=12, fontweight='bold')
    ax5.set_ylabel(r'$E[|X_{n+1} - X_n|]$', fontsize=12, fontweight='bold')
    ax5.set_title('Distance Between Consecutive Values\n Does NOT Go to Zero!', 
                  fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Mathematical proof
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    proof = """
  CONVERGENCE IN DISTRIBUTION
    
  Proof:
  For all n: Xₙ = Z · (-1)ⁿ
    
  Since (-1)ⁿ ∈ {-1, 1} and Z ~ N(0,1):
  • Both Z and -Z have same distribution
  • Therefore Xₙ ~ N(0,1) for all n
    
  CDF: Fₙ(x) = P(Xₙ ≤ x) = Φ(x) (standard normal CDF)
    
  Limit: Fₙ(x) = Φ(x) for all n
  So: Xₙ →ᵈ N(0,1) ✓
    
  X NOT CONVERGENCE IN PROBABILITY
    
  Proof:
  Xₙ₊₁ - Xₙ = Z·(-1)ⁿ⁺¹ - Z·(-1)ⁿ = -2Z·(-1)ⁿ
    
  |Xₙ₊₁ - Xₙ| = 2|Z|
    
  P(|Xₙ₊₁ - Xₙ| < ε) = P(|Z| < ε/2) ≠ 1 as n → ∞
    
  Values keep jumping 
    """
    
    ax6.text(0.05, 0.95, proof, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#fff8e1', alpha=0.9))
    
    plt.suptitle('Example 2: Convergence in Distribution ONLY (Not in Probability)', 
                fontsize=16, fontweight='bold', color='#f57c00')
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()   
    
    return fig

def example_almost_sure_convergence(py=False):
    """
    Example 4: Almost sure convergence (strongest)
    """
    
    print("EXAMPLE 4: ALMOST SURE CONVERGENCE (STRONGEST)")
    print("\nSequence: Xₙ = Σᵢ₌₁ⁿ Zᵢ / n², where Zᵢ ~ N(0,1) i.i.d.")
    print("\nEach path actually converges to 0 (with probability 1)")
    print("="*80)
    
    np.random.seed(42)
    
    # Generate sequence
    n_max = 500
    n_values = np.arange(1, n_max + 1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Multiple realizations showing actual convergence
    ax1 = axes[0, 0]
    
    n_paths = 30
    for trial in range(n_paths):
        Z = np.random.normal(0, 1, n_max)
        cumsum_Z = np.cumsum(Z)
        X_n = cumsum_Z / (n_values ** 2)
        ax1.plot(n_values, X_n, alpha=0.6, linewidth=1.5)
    
    ax1.axhline(0, color='red', linestyle='--', linewidth=2.5, label='Limit = 0')
    
    # Add convergence band
    epsilon = 0.01
    ax1.fill_between(n_values, -epsilon, epsilon, alpha=0.2, color='green',
                     label=f'±{epsilon} band')
    
    ax1.set_xlabel('n', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'$X_n$', fontsize=12, fontweight='bold')
    ax1.set_title(f'{n_paths} Realizations\n ALL Paths Converge to 0', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 0.05])
    
    # Plot 2: Log scale to see convergence rate
    ax2 = axes[0, 1]
    
    for trial in range(20):
        Z = np.random.normal(0, 1, n_max)
        cumsum_Z = np.cumsum(Z)
        X_n = cumsum_Z / (n_values ** 2)
        ax2.plot(n_values, np.abs(X_n), alpha=0.5, linewidth=1.5)
    
    # Theoretical upper bound (using Markov/Chebyshev)
    # E[|Xₙ|] ≤ √(Var(Xₙ)) = √(n/n⁴) = 1/n^(3/2)
    theoretical_bound = 2 / (n_values ** 1.5)
    ax2.plot(n_values, theoretical_bound, 'r--', linewidth=3, 
            label='Theoretical: O(1/n^(3/2))', alpha=0.8)
    
    ax2.set_xlabel('n', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'|X_n|', fontsize=12, fontweight='bold')
    ax2.set_title('Absolute Values (Log Scale)\n Fast Convergence', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    
    # Plot 3: Maximum deviation up to time n
    ax3 = axes[0, 2]
    
    max_deviations = []
    for trial in range(100):
        Z = np.random.normal(0, 1, n_max)
        cumsum_Z = np.cumsum(Z)
        X_n = cumsum_Z / (n_values ** 2)
        
        # Maximum absolute value seen so far
        max_so_far = np.maximum.accumulate(np.abs(X_n))
        max_deviations.append(max_so_far)
    
    max_deviations = np.array(max_deviations)
    
    # Plot percentiles
    ax3.fill_between(n_values, 
                     np.percentile(max_deviations, 10, axis=0),
                     np.percentile(max_deviations, 90, axis=0),
                     alpha=0.3, color='blue', label='10th-90th percentile')
    ax3.plot(n_values, np.median(max_deviations, axis=0), 'b-', 
            linewidth=2.5, label='Median')
    ax3.plot(n_values, np.percentile(max_deviations, 95, axis=0), 'r--',
            linewidth=2, label='95th percentile')
    
    ax3.set_xlabel('n', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'$max_{k\leq n} |X_k|$', fontsize=12, fontweight='bold')
    ax3.set_title('Maximum Deviation Up to Time n\n Eventually Stays Small', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Comparing convergence types
    ax4 = axes[1, 0]
    
    # Generate one path for each example type
    Z_single = 2.0
    n_range = np.arange(1, 201)
    
    # Example 2: Convergence in distribution only
    X_dist = Z_single * ((-1) ** n_range)
    
    # Example 3: Convergence in probability
    X_prob = Z_single / n_range
    
    # Example 4: Almost sure convergence
    Z_seq = np.random.normal(0, 1, 200)
    cumsum_Z_ex = np.cumsum(Z_seq)
    X_as = cumsum_Z_ex / (n_range ** 2)
    
    ax4.plot(n_range, X_dist, 'r-', linewidth=2, alpha=0.7, 
            label='Ex 2: Dist only (oscillates)')
    ax4.plot(n_range, X_prob, 'orange', linewidth=2, alpha=0.7,
            label='Ex 3: In probability')
    ax4.plot(n_range, X_as, 'g-', linewidth=2, alpha=0.7,
            label='Ex 4: Almost sure')
    ax4.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax4.set_xlabel('n', fontsize=12, fontweight='bold')
    ax4.set_ylabel(r'$X_n$', fontsize=12, fontweight='bold')
    ax4.set_title('Comparing Three Types of Convergence\n(single realization each)', 
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([-2.5, 2.5])
    
    # Plot 5: Borel-Cantelli illustration
    ax5 = axes[1, 1]
    
    # For almost sure convergence, sum of P(|Xₙ| > ε) should converge
    epsilon = 0.01
    
    # Using Chebyshev: P(|Xₙ| > ε) ≤ Var(Xₙ)/ε² = n/(n⁴·ε²) = 1/(n³·ε²)
    probs = 1 / (n_values ** 3 * epsilon ** 2)
    cumsum_probs = np.cumsum(probs)
    
    ax5.plot(n_values, cumsum_probs, 'b-', linewidth=2.5)
    ax5.set_xlabel('n', fontsize=12, fontweight='bold')
    ax5.set_ylabel(r'\sum_{k=1}^n P(|X_k| > ' + f'{epsilon}' + r'$)$', fontsize=12, fontweight='bold')
    ax5.set_title('Borel-Cantelli Criterion\n Sum Converges → A.S. Convergence', 
                  fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Add annotation
    ax5.text(0.6, 0.3, f'Sum converges!\n(≈ {cumsum_probs[-1]:.4f})',
            transform=ax5.transAxes, fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot 6: Mathematical proof
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    proof = """
  ALMOST SURE CONVERGENCE
    
  Proof outline:
    
  Xₙ = (Z₁+Z₂+...+Zₙ)/n²
    
  E[Xₙ] = 0
    
  Var(Xₙ) = Var(Σᵢ₌₁ⁿ Zᵢ)/n⁴ = n·Var(Z₁)/n⁴  (independence) = n/n⁴ = 1/n³
    
  For ε > 0, by Chebyshev: P(|Xₙ| > ε) ≤ Var(Xₙ)/ε² = 1/(n³ε²)
    
  Key: Σₙ₌₁^∞ P(|Xₙ| > ε) ≤ Σₙ₌₁^∞ 1/(n³ε²) = (1/ε²)·Σₙ₌₁^∞ 1/n³ < ∞  (p-series, p=3>1)
    
  By Borel-Cantelli Lemma: P(|Xₙ| > ε infinitely often) = 0
    
  Therefore: P(Xₙ → 0) = 1 
    
  This implies:
  • Convergence in probability ✓
  • Convergence in distribution ✓
    """
    
    ax6.text(0.05, 0.95, proof, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f5e8', alpha=0.9))
    
    plt.suptitle('Example 4: Almost Sure Convergence (Strongest - implies all other types)', 
                fontsize=16, fontweight='bold', color='#1b5e20')
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    return fig

def example_convergence_in_probability(py=False):
    """
    Example 3: Convergence in probability (and distribution)
    """
    
    print("EXAMPLE 3: CONVERGENCE IN PROBABILITY")
    print("\nSequence: Xₙ = Z/n, where Z ~ N(0,1)")
    print("\nValues get closer and closer to 0 (with high probability)!")
    
    np.random.seed(42)
    
    # Generate sequence
    n_max = 200
    n_values = np.arange(1, n_max + 1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Multiple realizations
    ax1 = axes[0, 0]
    
    for trial in range(20):
        Z = np.random.normal(0, 1)
        X_n = Z / n_values
        ax1.plot(n_values, X_n, alpha=0.5, linewidth=1.5)
    
    ax1.axhline(0, color='red', linestyle='--', linewidth=2.5, label='Limit = 0')
    ax1.set_xlabel('n', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Xₙ = Z/n', fontsize=12, fontweight='bold')
    ax1.set_title('20 Realizations\n All Converge to 0', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.5, 0.5])
    
    # Plot 2: Convergence bands
    ax2 = axes[0, 1]
    
    # For different epsilon, show P(|Xₙ| > ε)
    epsilons = [0.1, 0.05, 0.02, 0.01]
    
    for eps in epsilons:
        # P(|Z/n| > ε) = P(|Z| > nε) = 2*P(Z > nε)
        probs = 2 * (1 - stats.norm.cdf(n_values * eps))
        ax2.plot(n_values, probs, linewidth=2.5, label=f'ε = {eps}')
    
    ax2.set_xlabel('n', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'$P(|X_n| > \varepsilon)$', fontsize=12, fontweight='bold')
    ax2.set_title('Probability of Being Far from 0\n Goes to Zero for Any ' + r'$\varepsilon$', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: CDFs converging to point mass
    ax3 = axes[0, 2]
    
    x_range = np.linspace(-1, 1, 1000)
    
    for n in [1, 5, 10, 50, 200]:
        # Xₙ = Z/n ~ N(0, 1/n²)
        cdf_values = stats.norm.cdf(x_range, 0, 1/n)
        ax3.plot(x_range, cdf_values, linewidth=2.5, label=f'n = {n}', alpha=0.7)
    
    # Limit: point mass at 0 (step function)
    ax3.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax3.axhline(1, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Draw step function at 0
    ax3.plot([-1, 0], [0, 0], 'r-', linewidth=3, label='Limit: δ₀')
    ax3.plot([0, 0], [0, 1], 'r-', linewidth=3)
    ax3.plot([0, 1], [1, 1], 'r-', linewidth=3)
    
    ax3.set_xlabel('x', fontsize=12, fontweight='bold')
    ax3.set_ylabel('CDF(x)', fontsize=12, fontweight='bold')
    ax3.set_title('CDFs Converge to Point Mass at 0\n Convergence in Distribution', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=9, loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-0.05, 1.05])
    
    # Plot 4: Distribution narrowing
    ax4 = axes[1, 0]
    
    n_samples = 10000
    x_range_hist = np.linspace(-0.5, 0.5, 200)
    
    for n in [2, 5, 10, 20, 50]:
        Z_samples = np.random.normal(0, 1, n_samples)
        X_n_samples = Z_samples / n
        ax4.hist(X_n_samples, bins=100, density=True, alpha=0.4, 
                range=(-0.5, 0.5), label=f'n = {n}')
    
    ax4.axvline(0, color='red', linestyle='--', linewidth=2.5)
    ax4.set_xlabel('x', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax4.set_title('Distributions Getting Narrower\n Concentrating at 0', 
                  fontsize=13, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Empirical verification
    ax5 = axes[1, 1]
    
    # For each n, count what fraction of samples are within ε of 0
    epsilon = 0.1
    n_trials = 10000
    n_test = [1, 2, 5, 10, 20, 50, 100, 200]
    
    fractions_inside = []
    for n in n_test:
        Z_samples = np.random.normal(0, 1, n_trials)
        X_n_samples = Z_samples / n
        fraction = np.mean(np.abs(X_n_samples) < epsilon)
        fractions_inside.append(fraction)
    
    ax5.plot(n_test, fractions_inside, 'bo-', linewidth=2.5, markersize=10)
    ax5.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, 
               label='Target = 1.0')
    ax5.set_xlabel('n', fontsize=12, fontweight='bold')
    ax5.set_ylabel(r'$P(|X_n|$ < ' + f' {epsilon}' + r'$)$', fontsize=12, fontweight='bold')
    ax5.set_title(r'Fraction Within $\varepsilon$=' + f'{epsilon} of Zero\n Approaches 1', 
                  fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_ylim([0, 1.05])
    
    # Plot 6: Mathematical proof
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    proof = """
  CONVERGENCE IN PROBABILITY
    
  Proof:
  Xₙ = Z/n where Z ~ N(0,1)
    
  For any ε > 0:
  P(|Xₙ - 0| > ε) = P(|Z/n| > ε) = P(|Z| > nε) = 2·P(Z > nε) = 2·[1 - Φ(nε)] → 0 as n → ∞
    
  Since Φ(nε) → 1 as n → ∞, therefore: Xₙ →ᵖ 0 ✓
   
  CONVERGENCE IN DISTRIBUTION
    
  Proof:
  Xₙ ~ N(0, 1/n²)
    
  CDF: Fₙ(x) = Φ(x·n)
  - For x < 0: Fₙ(x) = Φ(x·n) → 0
  - For x > 0: Fₙ(x) = Φ(x·n) → 1
  - For x = 0: Fₙ(0) = 0.5 for all n
    
  Limit is point mass at 0: δ₀
  Therefore: Xₙ →ᵈ δ₀ ✓
    
  Note: Convergence in probability → Convergence in distribution
    """
    
    ax6.text(0.05, 1, proof, transform=ax6.transAxes,
            fontsize=9.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f5e8', alpha=0.9))
    
    plt.suptitle('Example 3: Convergence in Probability (implies convergence in distribution)', 
                fontsize=16, fontweight='bold', color='#2e7d32')
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    return fig

def casino_simulation_intro(py=False):
    """
    Intuitive demonstration: Casino profit over time
    Shows convergence to expected value
    """
    
    np.random.seed(42)
    
    # Game parameters
    cost_to_play = 1
    payout_on_six = 5
    prob_six = 1/6
    
    # Expected profit per game
    expected_profit = -4 * prob_six + 1 * (1 - prob_six)
    
    print(f"Expected profit per game: ${expected_profit:.4f}")
    
    
    # Simulate games
    n_games = 10000
    profits = []
    
    for _ in range(n_games):
        roll = np.random.randint(1, 7)
        if roll == 6:
            profit = -(payout_on_six - cost_to_play)  # You lose $4
        else:
            profit = cost_to_play  # You keep $1
        profits.append(profit)
    
    profits = np.array(profits)
    
    # Calculate running average
    cumulative_profit = np.cumsum(profits)
    running_average = cumulative_profit / np.arange(1, n_games + 1)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: First 50 games (wild swings)
    ax1 = axes[0, 0]
    games_50 = np.arange(1, 51)
    ax1.plot(games_50, running_average[:50], 'b-', linewidth=2, label="Casino's average profit")
    ax1.axhline(expected_profit, color='red', linestyle='--', linewidth=2, 
                label=f'Expected: ${expected_profit:.2f}')
    ax1.fill_between(games_50, running_average[:50], expected_profit, 
                     alpha=0.3, color='blue')
    ax1.set_xlabel('Number of Games', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Profit per Game ($)', fontsize=12, fontweight='bold')
    ax1.set_title('First 50 Games\n Wild Swings! Should You Worry?', 
                  fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-1, 1.5])
    
    # Add annotation
    ax1.annotate(f'After 50 games:\n${running_average[49]:.3f}/game',
                xy=(50, running_average[49]), xytext=(35, -0.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Plot 2: First 500 games (stabilizing)
    ax2 = axes[0, 1]
    games_500 = np.arange(1, 501)
    ax2.plot(games_500, running_average[:500], 'b-', linewidth=2, label="Casino's average profit")
    ax2.axhline(expected_profit, color='red', linestyle='--', linewidth=2, 
                label=f'Expected: ${expected_profit:.2f}')
    ax2.fill_between(games_500, running_average[:500], expected_profit, 
                     alpha=0.3, color='blue')
    ax2.set_xlabel('Number of Games', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Average Profit per Game ($)', fontsize=12, fontweight='bold')
    ax2.set_title('First 500 Games\n Starting to Stabilize...', 
                  fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.5, 0.8])
    
    # Add annotation
    ax2.annotate(f'After 500 games:\n${running_average[499]:.3f}/game',
                xy=(500, running_average[499]), xytext=(350, -0.15),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    
    # Plot 3: All 10,000 games (converged)
    ax3 = axes[1, 0]
    games_all = np.arange(1, n_games + 1)
    ax3.plot(games_all, running_average, 'b-', linewidth=2, label="Casino's average profit")
    ax3.axhline(expected_profit, color='red', linestyle='--', linewidth=2.5, 
                label=f'Expected: ${expected_profit:.2f}')
    ax3.fill_between(games_all, running_average, expected_profit, 
                     alpha=0.2, color='blue')
    ax3.set_xlabel('Number of Games', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Average Profit per Game ($)', fontsize=12, fontweight='bold')
    ax3.set_title(f'All {n_games:,} Games\n Locked In! The House Always Wins (in the long run)', 
                  fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 0.4])
    
    # Add final annotation
    ax3.annotate(f'After {n_games:,} games:\n${running_average[-1]:.4f}/game\n≈ ${expected_profit:.4f}/game',
                xy=(n_games, running_average[-1]), xytext=(7000, 0.3),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
    
    # Plot 4: Histogram of individual game profits
    ax4 = axes[1, 1]
    ax4.hist(profits, bins=[-4.5, -3.5, 0.5, 1.5], edgecolor='black', 
            alpha=0.7, color='coral', rwidth=0.8)
    #ax4.axvline(expected_profit, color='blue', linestyle='--', linewidth=2.5,
    #           label=f'Average: ${expected_profit:.2f}')
    ax4.set_xlabel('Profit per Game ($)', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax4.set_title('Distribution of Individual Game Profits\n(Each game is random!)', 
                  fontsize=13, fontweight='bold')
    ax4.set_xticks([-4, 1])
    ax4.set_xticklabels(['Lose $4\n(roll 6)', 'Win $1\n(roll 1-5)'])
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add counts
    loses = np.sum(profits == -4)
    wins = np.sum(profits == 1)
    ax4.text(-4, loses + 100, f'{loses} times\n({loses/n_games*100:.2f}%)\nExpected: {prob_six*100:.2f}%', 
            ha='center', fontsize=10, fontweight='bold')
    ax4.text(1, wins + 100, f'{wins} times\n({wins/n_games*100:.2f}%)\nExpected: {(1-prob_six)*100:.2f}%', 
            ha='center', fontsize=10, fontweight='bold')
    
    outcome = f"""
  Results After {n_games:,} Games:
    Expected profit per game: ${expected_profit:.4f}
    Actual average profit:    ${running_average[-1]:.4f}
    Difference:               ${abs(running_average[-1] - expected_profit):.4f}
    Total profit:             ${cumulative_profit[-1]:.2f}
    """
    
    ax4.text(0.3, 0.9, outcome, transform=ax4.transAxes,
            fontsize=9.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f5e8', alpha=0.9))   
    
    plt.suptitle('Law of Large Numbers: Casino Profit Converges to Expected Value', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
      
        
    return running_average, expected_profit, fig

def bridge_to_formal_definition(py=False):
    """
    Visual bridge from intuition to formal mathematical statement
    """
    
    
    # Create a comprehensive visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    np.random.seed(42)
    
    # Use exponential distribution (clearly not normal)
    dist = stats.expon(scale=2)
    true_mean = dist.mean()
    true_std = dist.std()
    
    print(f"\nUsing Exponential(λ=0.5) distribution")
    print(f"   True mean (μ): {true_mean:.4f}")
    print(f"   True std (σ): {true_std:.4f}")
    
    # Generate samples
    max_n = 5000
    all_samples = dist.rvs(size=max_n)
    
    # Calculate running mean
    running_mean = np.cumsum(all_samples) / np.arange(1, max_n + 1)
    
    # Plot 1: Original distribution
    ax1 = axes[0, 0]
    sample_for_hist = dist.rvs(size=10000, random_state=42)
    ax1.hist(sample_for_hist, bins=60, density=True, alpha=0.7, 
            color='steelblue', edgecolor='black')
    x_range = np.linspace(0, 15, 500)
    ax1.plot(x_range, dist.pdf(x_range), 'r-', linewidth=3, label='True PDF')
    ax1.axvline(true_mean, color='green', linestyle='--', linewidth=2.5,
               label=f'μ = {true_mean:.2f}')
    ax1.set_xlabel('Value (X)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Observations\n(Random, unpredictable)', 
                 fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    ax1.text(0.5, 0.5, "Individual observations are random \n(follow the distribution)", transform=ax1.transAxes,
            fontsize=9.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f5e8', alpha=0.9))
    
    # Plot 2: Running mean convergence
    ax2 = axes[0, 1]
    ax2.plot(range(1, max_n + 1), running_mean, 'b-', linewidth=2, alpha=0.8)
    ax2.axhline(true_mean, color='red', linestyle='--', linewidth=2.5,
               label=f'μ = {true_mean:.2f}')
    
    # Add convergence band
    epsilon = 0.1
    ax2.axhline(true_mean + epsilon, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.axhline(true_mean - epsilon, color='orange', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.fill_between(range(1, max_n + 1), true_mean - epsilon, true_mean + epsilon,
                     alpha=0.2, color='orange', label=f'ε = {epsilon} band')
    
    ax2.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Sample Mean (X̄ₙ)', fontsize=12, fontweight='bold')
    ax2.set_title('Sample Mean Converges\n'+r'$\bar{X_n} \rightarrow \mu$', 
                 fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 4])
    
    ax2.text(0.45, 0.2, "The average of individual observations \nconverges to the expected value μ", transform=ax2.transAxes,
            fontsize=9.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f5e8', alpha=0.9))
    
    
    # Plot 3: Probability of being close (convergence in probability)
    ax3 = axes[1, 0]
    
    # For different epsilon values, calculate P(|X̄ₙ - μ| < ε)
    epsilons = [0.5, 0.2, 0.1, 0.05]
    sample_sizes = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
    
    for eps in epsilons:
        probs = []
        for n in sample_sizes:
            # Run many trials
            n_trials = 1000
            close_count = 0
            for _ in range(n_trials):
                sample_mean = np.mean(dist.rvs(size=n))
                if abs(sample_mean - true_mean) < eps:
                    close_count += 1
            probs.append(close_count / n_trials)
        
        ax3.plot(sample_sizes, probs, 'o-', linewidth=2, markersize=8, 
                label=f'ε = {eps}', alpha=0.8)
    
    ax3.axhline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.5)
    ax3.set_xlabel('Sample Size (n)', fontsize=12, fontweight='bold')
    ax3.set_ylabel(r'$P(|\bar{X_n} - \mu| < \varepsilon)$', fontsize=12, fontweight='bold')
    ax3.set_title('Convergence in Probability\n'+r'$P(|\bar{X_n} - \mu| < \varepsilon) \rightarrow 1$', 
                 fontsize=13, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    ax3.set_ylim([0, 1.05])
    
    ax3.text(0.2, 0.1, "Probability of being 'close' → 1 as n increases", transform=ax3.transAxes,
            fontsize=9.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#e8f5e8', alpha=0.9))
    
    
    # Plot 4: Mathematical notation explanation
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    notation_text = """
  Mathematical Formulation:
    
  Let X₁, X₂, X₃, ... be i.i.d. random variables with E[Xᵢ] = μ and Var(Xᵢ) = σ²
    
  Define sample mean:
    """+r"$\bar{X_n} = \frac{X_1 + X_2 + ... + X_n}{n}$"+"""
  
  Law of Large Numbers says:
    """ + r"$\bar{X_n} \rightarrow \mu$ as $n \rightarrow \infty$" + """
  
  More precisely (Weak LLN):
    
  For any ε > 0, however small: """ + r"$P(|\bar{X_n} - \mu| < \varepsilon) \rightarrow 1$ as $n\rightarrow \infty$" + """
    
  In words: The probability that """+ r"$\bar{X_n}$" + """ is close to μ goes to 1
    """
    
    ax4.text(0.02, 1, notation_text, transform=ax4.transAxes,
            fontsize=11, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('From Intuition to Mathematics: The Law of Large Numbers', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    return fig

def polling_simulation(py=False):
    """
    Show how poll estimates converge to true population proportion
    """
    
    np.random.seed(42)
    
    # True population support
    true_support = 0.52
    
    print("ELECTION POLLING: Finding the Truth Through Sampling")
    print(f"\nTrue population support for Candidate A: {true_support*100:.1f}%")
    print("   (But it's unknown)")
    print("\nWe're going to survey random voters and track our estimate...")
    
    
    # Simulate surveys of increasing size
    max_surveys = 5000
    responses = np.random.binomial(1, true_support, max_surveys)
    
    # Running estimate
    cumulative_support = np.cumsum(responses)
    survey_sizes = np.arange(1, max_surveys + 1)
    running_estimate = cumulative_support / survey_sizes
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # Main plot: All surveys
    ax_main = fig.add_subplot(gs[:2, :])
    ax_main.plot(survey_sizes, running_estimate * 100, 'b-', linewidth=2)
    ax_main.axhline(true_support * 100, color='red', linestyle='--', linewidth=2.5,
                   label=f'True support = {true_support*100:.1f}%')
    
    # Add confidence bands
    ax_main.axhline((true_support + 0.03) * 100, color='orange', linestyle=':', 
                   linewidth=1.5, alpha=0.7)
    ax_main.axhline((true_support - 0.03) * 100, color='orange', linestyle=':', 
                   linewidth=1.5, alpha=0.7)
    ax_main.fill_between(survey_sizes, (true_support - 0.03) * 100, 
                         (true_support + 0.03) * 100,
                         alpha=0.2, color='orange', label='±3% margin')
    
    ax_main.set_xlabel('Number of People Surveyed', fontsize=13, fontweight='bold')
    ax_main.set_ylabel('Estimated Support (%)', fontsize=13, fontweight='bold')
    ax_main.set_title(f'Poll Estimate Converges to True Value\n"The more people you ask, the better your estimate"', 
                     fontsize=14, fontweight='bold')
    ax_main.legend(fontsize=11, loc='upper right')
    ax_main.grid(True, alpha=0.3)
    ax_main.set_ylim([40, 65])
    
    key_insights = """
  - Small polls (n=50): Estimates vary widely from poll to poll
  - Large polls (n=1000): Estimates cluster tightly around true value
  - As sample size increases, estimate converges to true population value  
  This is why professional polls typically survey 1000-1500 people.
  You don't need to survey everyone - just a large enough sample.
  """
  
    ax_main.text(0.2, 0.3, key_insights, transform=ax_main.transAxes,
            fontsize=9, verticalalignment='top',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    
    # Add annotations for typical poll sizes
    typical_sizes = [50, 500, 1000]
    for size in typical_sizes:
        if size < max_surveys:
            estimate = running_estimate[size-1] * 100
            ax_main.plot(size, estimate, 'go', markersize=10, zorder=5)
            ax_main.annotate(f'n={size}\n{estimate:.1f}%',
                            xy=(size, estimate), xytext=(size + 400, estimate + 3),
                            arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                            fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Plot: Small sample (n=50) - multiple trials
    ax1 = fig.add_subplot(gs[2, 0])
    small_sample_size = 50
    n_trials = 100
    
    small_estimates = []
    for _ in range(n_trials):
        sample = np.random.binomial(1, true_support, small_sample_size)
        small_estimates.append(np.mean(sample) * 100)
    
    ax1.hist(small_estimates, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax1.axvline(true_support * 100, color='red', linestyle='--', linewidth=2.5,
               label='True value')
    ax1.set_xlabel('Estimated Support (%)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax1.set_title(f'100 Different Polls (n={small_sample_size} each)\n"High variability!"', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot: Large sample (n=1000) - multiple trials
    ax2 = fig.add_subplot(gs[2, 1])
    large_sample_size = 1000
    
    large_estimates = []
    for _ in range(n_trials):
        sample = np.random.binomial(1, true_support, large_sample_size)
        large_estimates.append(np.mean(sample) * 100)
    
    ax2.hist(large_estimates, bins=30, edgecolor='black', alpha=0.7, color='lightgreen')
    ax2.axvline(true_support * 100, color='red', linestyle='--', linewidth=2.5,
               label='True value')
    ax2.set_xlabel('Estimated Support (%)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax2.set_title(f'100 Different Polls (n={large_sample_size} each)\n"Much more consistent!"', 
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Polling Simulation: Law of Large Numbers in Action', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()
    
    return fig

if __name__ == "__main__":
    #demo_estimating_prob_hook()
    #clt_experiment()
    #show_transformation(py=True)
    #demo_clt_cauchy(py=True)
    #demo_convergence(py=True)
    #example_no_convergence(py=True)
    #example_convergence_in_distribution_only(py=True)
    #example_almost_sure_convergence(py=True)
    #example_convergence_in_probability(py=True)
    #running_avg, expected, fig = casino_simulation_intro(py=True)
    #bridge_to_formal_definition(py=True)
    polling_simulation(py=True)