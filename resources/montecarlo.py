import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import seaborn as sns
from matplotlib.patches import Rectangle, FancyBboxPatch

from scipy.stats import norm, expon
from scipy.stats import beta as beta_dist

def demonstrate_sampling_concept(py=False):
    """
    Visual demonstration of what sampling means
    """
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Example 1: Uniform Distribution
    # ================================
        
    ax1 = axes[0, 0]
    # The distribution (theoretical)
    x_uniform = np.linspace(0, 10, 1000)
    y_uniform = np.ones_like(x_uniform) * 0.1  # Height = 1/(10-0) = 0.1
    ax1.fill_between(x_uniform, 0, y_uniform, alpha=0.3, color='blue', 
                     label='Probability Density')
    ax1.plot(x_uniform, y_uniform, 'b-', linewidth=2.5)
    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Probability Density', fontsize=11)
    ax1.set_title('Distribution (The Recipe)\nUniform(0, 10)', 
                  fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 0.15])
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Now let's sample from it
    ax2 = axes[0, 1]
    np.random.seed(42)
    samples_uniform = np.random.uniform(0, 10, size=5)
    
    # print(f"Let's spin it 5 times:")
    for i, sample in enumerate(samples_uniform, 1):
        #print(f"  Spin {i}: {sample:.2f}")
        ax2.scatter([sample], [i], s=200, color='red', zorder=5, edgecolor='black', linewidth=2)
        ax2.text(sample + 0.3, i, f'{sample:.2f}', fontsize=10, va='center')
    
    ax2.set_xlim([0, 10])
    ax2.set_ylim([0, 6])
    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_ylabel('Sample Number', fontsize=11)
    ax2.set_title('Samples (Following the Recipe)\n5 random draws', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # What happens with many samples?
    ax3 = axes[0, 2]
    many_samples_uniform = np.random.uniform(0, 10, size=1000)
    ax3.hist(many_samples_uniform, bins=30, alpha=0.7, edgecolor='black', 
             density=True, color='red', label='1000 samples')
    ax3.plot(x_uniform, y_uniform, 'b-', linewidth=2.5, label='True distribution')
    ax3.set_xlabel('Value', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Many Samples → Reveals Distribution\n1000 draws', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
        
    # Example 2: Normal Distribution
    # ================================
    ax4 = axes[1, 0]
    mu, sigma = 170, 10
    x_normal = np.linspace(140, 200, 1000)
    from scipy.stats import norm
    y_normal = norm.pdf(x_normal, mu, sigma)
    
    ax4.fill_between(x_normal, 0, y_normal, alpha=0.3, color='green')
    ax4.plot(x_normal, y_normal, 'g-', linewidth=2.5)
    ax4.axvline(mu, color='red', linewidth=2, linestyle='--', 
                label=f'Mean: {mu}cm')
    ax4.axvline(mu - sigma, color='orange', linewidth=1.5, linestyle=':', alpha=0.7)
    ax4.axvline(mu + sigma, color='orange', linewidth=1.5, linestyle=':', alpha=0.7)
    ax4.set_xlabel('Height (cm)', fontsize=11)
    ax4.set_ylabel('Probability Density', fontsize=11)
    ax4.set_title('Distribution (The Recipe)\nNormal(μ=170, σ=10)', 
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # Sample from it
    ax5 = axes[1, 1]
    np.random.seed(42)
    samples_normal = np.random.normal(mu, sigma, size=5)
    
    for i, sample in enumerate(samples_normal, 1):
        #print(f"  Student {i}: {sample:.1f} cm")
        ax5.scatter([sample], [i], s=200, color='green', zorder=5, 
                   edgecolor='black', linewidth=2)
        ax5.text(sample + 2, i, f'{sample:.1f}cm', fontsize=10, va='center')
    
    ax5.axvline(mu, color='red', linewidth=2, linestyle='--', alpha=0.5, label='Mean')
    ax5.set_xlim([140, 200])
    ax5.set_ylim([0, 6])
    ax5.set_xlabel('Height (cm)', fontsize=11)
    ax5.set_ylabel('Sample Number', fontsize=11)
    ax5.set_title('Samples (Following the Recipe)\n5 random measurements', 
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='x')
    
    # Many samples
    ax6 = axes[1, 2]
    many_samples_normal = np.random.normal(mu, sigma, size=1000)
    ax6.hist(many_samples_normal, bins=40, alpha=0.7, edgecolor='black', 
             density=True, color='green', label='1000 samples')
    ax6.plot(x_normal, y_normal, 'b-', linewidth=2.5, label='True distribution')
    ax6.set_xlabel('Height (cm)', fontsize=11)
    ax6.set_ylabel('Density', fontsize=11)
    ax6.set_title('Many Samples → Reveals Distribution\n1000 measurements', 
                  fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
        
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    print(f"\n{'='*70}")

    return fig

def complete_sampling_picture(py=False):
    """
    Show how all concepts connect: Population → Sample → Distribution → Sampling
    """
    
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 2, hspace=0.4, wspace=0.3)
    
    
    # Step 1: Real Population
    print("\nSTEP 1: Real-World Population")
    print("-" * 80)
    print("Real scenario: Heights of all university students")
    
    np.random.seed(42)
    true_population = np.random.normal(170, 10, 50000)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(true_population, bins=60, alpha=0.7, edgecolor='black', 
             density=True, color='lightblue')
    ax1.set_xlabel('Height (cm)', fontsize=11)
    ax1.set_ylabel('Density', fontsize=11)
    ax1.set_title('STEP 1: Real Population\n50,000 students (unmeasurable)', 
                  fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    print(f"Population size: 50,000 students")
    print(f"True mean: {np.mean(true_population):.2f} cm")
    print(f"True std: {np.std(true_population):.2f} cm")
    
    # Step 2: Take a Sample
    print("\nSTEP 2: Collect a Sample")
    print("-" * 80)
    print("We randomly measure 200 students")
    
    sample_size = 200
    our_sample = np.random.choice(true_population, size=sample_size, replace=False)
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(our_sample, bins=25, alpha=0.7, edgecolor='black',
             density=True, color='lightcoral')
    ax2.set_xlabel('Height (cm)', fontsize=11)
    ax2.set_ylabel('Density', fontsize=11)
    ax2.set_title(f'STEP 2: Collected Sample\nn = {sample_size} students (measurable)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    sample_mean = np.mean(our_sample)
    sample_std = np.std(our_sample, ddof=1)
    
    print(f"Sample size: {sample_size} students")
    print(f"Sample mean: {sample_mean:.2f} cm")
    print(f"Sample std: {sample_std:.2f} cm")
    print(f"Error in mean: {abs(sample_mean - np.mean(true_population)):.2f} cm")
    
    # Step 3: Infer Distribution
    print("\nSTEP 3: Infer the Distribution")
    print("-" * 80)
    print("Based on our sample, we estimate:")
    print(f"  Heights follow Normal(μ={sample_mean:.1f}, σ={sample_std:.1f})")
    
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Plot sample histogram
    ax3.hist(our_sample, bins=25, alpha=0.5, edgecolor='black',
             density=True, color='lightcoral', label='Our sample')
    
    # Plot inferred distribution
    x_range = np.linspace(130, 210, 1000)
    from scipy.stats import norm
    inferred_dist = norm.pdf(x_range, sample_mean, sample_std)
    ax3.plot(x_range, inferred_dist, 'r-', linewidth=3, 
             label=f'Inferred: N({sample_mean:.1f}, {sample_std:.1f})')
    
    # Plot true distribution for comparison
    true_dist = norm.pdf(x_range, np.mean(true_population), np.std(true_population))
    ax3.plot(x_range, true_dist, 'b--', linewidth=2, alpha=0.7,
             label=f'True: N({np.mean(true_population):.1f}, {np.std(true_population):.1f})')
    
    ax3.set_xlabel('Height (cm)', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('STEP 3: Inferred Distribution\nFrom sample → estimate population distribution', 
                  fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Step 4: Sample from Distribution
    print("\nSTEP 4: Sample from the Distribution (Computational)")
    print("-" * 80)
    print("Now we can GENERATE new samples from the inferred distribution")
    
    # Generate samples from inferred distribution
    generated_sample = np.random.normal(sample_mean, sample_std, sample_size)
    
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(generated_sample, bins=25, alpha=0.7, edgecolor='black',
             density=True, color='lightgreen')
    ax4.plot(x_range, inferred_dist, 'r-', linewidth=3, alpha=0.7,
             label=f'Distribution: N({sample_mean:.1f}, {sample_std:.1f})')
    ax4.set_xlabel('Height (cm)', fontsize=11)
    ax4.set_ylabel('Density', fontsize=11)
    ax4.set_title('STEP 4: Sample from Distribution\nGenerated 200 values from Normal(170.3, 9.8)', 
                  fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    print(f"Generated {sample_size} new values")
    print(f"Mean: {np.mean(generated_sample):.2f} cm")
    print(f"Std: {np.std(generated_sample, ddof=1):.2f} cm")
    
    # Step 5: Multiple Samples (Monte Carlo preview)
    print("\n🎰 STEP 5: Generate MANY Samples (Monte Carlo)")
    print("-" * 80)
    print("Generate 1000 different samples, compute mean of each")
    
    n_simulations = 1000
    sample_means = []
    
    for _ in range(n_simulations):
        simulated_sample = np.random.normal(sample_mean, sample_std, sample_size)
        sample_means.append(np.mean(simulated_sample))
    
    sample_means = np.array(sample_means)
    
    ax5 = fig.add_subplot(gs[2, :])
    ax5.hist(sample_means, bins=50, alpha=0.7, edgecolor='black',
             density=True, color='plum')
    
    # Theoretical distribution of sample means (preview of CLT)
    se_theoretical = sample_std / np.sqrt(sample_size)
    x_means = np.linspace(sample_means.min(), sample_means.max(), 1000)
    theoretical_dist = norm.pdf(x_means, sample_mean, se_theoretical)
    ax5.plot(x_means, theoretical_dist, 'b-', linewidth=3,
             label=f'Theoretical: N({sample_mean:.1f}, {se_theoretical:.2f})')
    
    ax5.axvline(sample_mean, color='red', linewidth=2.5, linestyle='--',
                label=f'Expected mean: {sample_mean:.2f}')
    ax5.set_xlabel('Sample Mean', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('STEP 5: Distribution of Sample Means (1000 simulations)\n'
                  'This is Monte Carlo in action!', 
                  fontsize=12, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    
    print(f"Mean of sample means: {np.mean(sample_means):.2f} cm")
    print(f"Std of sample means (SE): {np.std(sample_means):.2f} cm")
    print(f"Theoretical SE: {se_theoretical:.2f} cm")
    print(f"Match: {abs(np.std(sample_means) - se_theoretical) < 0.1}")
    
    # Summary connections
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')
    
    summary = """
    HOW IT ALL CONNECTS:
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    REAL WORLD                          MATHEMATICAL                       COMPUTATIONAL
    ───────────────────────────────────────────────────────────────────────────────────────
    
    1. Population exists                → Described by distribution       → Can't directly access
       (50k students)                     p(x) = Normal(μ, σ)               (too large/expensive)
                                                ↓
    2. Collect sample                   → Observe realizations             → Measure actual data
       (200 students)                     X₁, X₂, ..., Xₙ                   (real measurements)
                                                ↓
    3. Compute statistics               → Sample mean x̄, std s            → Calculate from data
       (mean = 170.3 cm)                  Estimate μ ≈ x̄                    (Python: np.mean())
                                                ↓
    4. Infer distribution               → Assume p(x) = Normal(x̄, s)      → Model the population
       (Normal(170.3, 9.8))               "Plug-in" estimate                (our best guess)
                                                ↓
    5. Sample from distribution         → Generate Xᵢ ~ Normal(x̄, s)      → Simulate new data
       (generate new data)                Use random number generator       (Python: np.random.normal())
                                                ↓
    6. Repeat many times                → Monte Carlo estimation          → Compute expectations
       (1000 simulations)                 E[f(X)] ≈ (1/n)Σf(Xᵢ)             (average over samples)
    
    ═══════════════════════════════════════════════════════════════════════════════════════
    
    KEY INSIGHT: "Sampling from a distribution" is the computational analog of 
                 "sampling from a population"
    
    THIS IS THE FOUNDATION OF:
    ✓ Monte Carlo methods (today's topic!)
    ✓ Bootstrap sampling (resample from your data)
    ✓ Simulation-based inference
    ✓ All of modern computational statistics
    """
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def interactive_population_sample_demo(py=False):
    """
    Interactive demonstration of population vs sample concept
    """
    
    np.random.seed(42)
    
    # Create a population
    population_size = 10000
    population = np.random.normal(loc=170, scale=10, size=population_size)
    
    # Population parameters (TRUE values we want to know)
    pop_mean = np.mean(population)
    pop_std = np.std(population)
    
    print("="*80)
    print("INTERACTIVE DEMO: Population vs Sample")
    print("="*80)
    print("\n🏫 SCENARIO: Heights of all students at a university")
    print(f"Population: {population_size:,} students")
    print(f"TRUE mean height: {pop_mean:.2f} cm (we don't know this in real life!)")
    print(f"TRUE std deviation: {pop_std:.2f} cm (we don't know this either!)")
    print()
    
    # Take samples of different sizes
    sample_sizes = [10, 50, 100, 500, 1000]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    # Plot population
    ax_pop = axes[0, 0]
    ax_pop.hist(population, bins=50, alpha=0.7, edgecolor='black', 
                density=True, color='lightblue', label='Population')
    ax_pop.axvline(pop_mean, color='red', linewidth=3, linestyle='--',
                   label=f'True μ = {pop_mean:.2f}')
    ax_pop.axvline(pop_mean - pop_std, color='orange', linewidth=2, 
                   linestyle=':', alpha=0.7)
    ax_pop.axvline(pop_mean + pop_std, color='orange', linewidth=2, 
                   linestyle=':', alpha=0.7, label=f'True σ = {pop_std:.2f}')
    ax_pop.set_xlabel('Height (cm)', fontsize=11)
    ax_pop.set_ylabel('Density', fontsize=11)
    ax_pop.set_title(f'POPULATION (N = {population_size:,})\nThe TRUE Distribution', 
                     fontsize=12, fontweight='bold', color='blue')
    ax_pop.legend(fontsize=10)
    ax_pop.grid(True, alpha=0.3)
    
    # Text explanation
    ax_text = axes[0, 1]
    ax_text.axis('off')
    explanation = f"""
    POPULATION (The Truth)
    ══════════════════════
    
    • All {population_size:,} students
    • True mean: {pop_mean:.2f} cm
    • True std: {pop_std:.2f} cm
    
    In reality, we CAN'T measure
    all {population_size:,} students!
    
    Too expensive, time-consuming
    
    Instead: Take a SAMPLE
    """
    ax_text.text(0.1, 0.5, explanation, transform=ax_text.transAxes,
                fontsize=11, verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    # Arrow
    ax_arrow = axes[0, 2]
    ax_arrow.axis('off')
    ax_arrow.annotate('', xy=(0.5, 0.2), xytext=(0.5, 0.8),
                     arrowprops=dict(arrowstyle='->', lw=5, color='green'))
    ax_arrow.text(0.5, 0.5, 'SAMPLE', fontsize=16, fontweight='bold',
                 ha='center', va='center', color='green',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # Plot samples
    results = []
    
    for idx, n in enumerate(sample_sizes):
        row = 1 + idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Take random sample
        sample = np.random.choice(population, size=n, replace=False)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)  # Sample std (with Bessel's correction)
        
        # Calculate error
        mean_error = abs(sample_mean - pop_mean)
        std_error = abs(sample_std - pop_std)
        
        results.append({
            'n': n,
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            'mean_error': mean_error,
            'std_error': std_error
        })
        
        # Plot sample
        ax.hist(sample, bins=min(30, n//2), alpha=0.7, edgecolor='black',
               density=True, color='lightcoral', label=f'Sample (n={n})')
        
        # Overlay population for comparison
        ax.hist(population, bins=50, alpha=0.2, density=True, 
               color='blue', label='Population')
        
        # Sample statistics
        ax.axvline(sample_mean, color='red', linewidth=2.5, linestyle='-',
                  label=f'Sample x̄ = {sample_mean:.2f}')
        
        # True population mean
        ax.axvline(pop_mean, color='blue', linewidth=2, linestyle='--',
                  alpha=0.7, label=f'True μ = {pop_mean:.2f}')
        
        ax.set_xlabel('Height (cm)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'SAMPLE (n = {n})\nError: {mean_error:.2f} cm', 
                    fontsize=11, fontweight='bold',
                    color='green' if mean_error < 1 else 'orange')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        print(f"Sample size n={n:4d}:")
        print(f"  Sample mean: {sample_mean:.2f} cm (error: {mean_error:.2f})")
        print(f"  Sample std:  {sample_std:.2f} cm (error: {std_error:.2f})")
        print(f"  Accuracy: {(1 - mean_error/pop_mean)*100:.1f}%")
        print()
    
    # Summary plot
    ax_summary = axes[2, 2]
    ns = [r['n'] for r in results]
    errors = [r['mean_error'] for r in results]
    
    ax_summary.plot(ns, errors, 'o-', linewidth=2.5, markersize=10, color='purple')
    ax_summary.set_xscale('log')
    ax_summary.set_xlabel('Sample Size (n)', fontsize=11)
    ax_summary.set_ylabel('Error in Mean Estimate (cm)', fontsize=11)
    ax_summary.set_title('Error Decreases as Sample Size Increases', 
                         fontsize=11, fontweight='bold')
    ax_summary.grid(True, alpha=0.3, which='both')
    
    # Add theoretical 1/√n line
    theoretical_error = errors[0] * np.sqrt(ns[0]) / np.sqrt(np.array(ns))
    ax_summary.plot(ns, theoretical_error, '--', linewidth=2, color='red',
                   alpha=0.7, label='Theoretical: 1/√n')
    ax_summary.legend(fontsize=10)
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    print("="*80)
    print("KEY OBSERVATIONS:")
    print("="*80)
    print("1. ✓ Larger samples give better estimates of population parameters")
    print("2. ✓ Even small samples (n=50) can be surprisingly accurate!")
    print("3. ✓ Error decreases as 1/√n (doubling sample size reduces error by ~30%)")
    print("4. ✓ Random sampling ensures sample 'looks like' population")
    print("="*80 + "\n")

    return fig

def sampling_process_diagram(py=False):
    """
    Create a visual diagram of the sampling process
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Population (big circle)
    from matplotlib.patches import Circle, FancyArrowPatch
    
    population_circle = Circle((2.5, 7), 1.5, color='lightblue', alpha=0.5, 
                               edgecolor='blue', linewidth=3)
    ax.add_patch(population_circle)
    ax.text(2.5, 7, 'POPULATION\n\nN = large\nμ = ?\nσ = ?', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(2.5, 5, 'All individuals\nwe want to study\n(usually unmeasurable)',
           ha='center', va='top', fontsize=9, style='italic')
    
    # Random sampling arrow
    arrow1 = FancyArrowPatch((4, 7), (6.5, 7),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='green')
    ax.add_patch(arrow1)
    ax.text(5.25, 7.5, 'Random\nSampling', ha='center', fontsize=10,
           fontweight='bold', color='green',
           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Sample (small circle)
    sample_circle = Circle((7.5, 7), 0.8, color='lightcoral', alpha=0.5,
                           edgecolor='red', linewidth=3)
    ax.add_patch(sample_circle)
    ax.text(7.5, 7, 'SAMPLE\n\nn << N\nx̄\ns', 
           ha='center', va='center', fontsize=11, fontweight='bold')
    ax.text(7.5, 5.7, 'Actually measured\n(observable)',
           ha='center', va='top', fontsize=9, style='italic')
    
    # Inference arrow
    arrow2 = FancyArrowPatch((6.7, 6.5), (4, 6.5),
                            arrowstyle='->', mutation_scale=30, linewidth=3,
                            color='purple', linestyle='--')
    ax.add_patch(arrow2)
    ax.text(5.35, 6, 'Statistical\nInference', ha='center', fontsize=10,
           fontweight='bold', color='purple',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # Key insights boxes
    insight1 = """
    WHAT WE WANT:
    ═══════════════
    • Population mean (μ)
    • Population std (σ)
    • Population proportion (p)
    
    PROBLEM: Can't measure
    entire population!
    """
    
    ax.text(1.5, 3.5, insight1, fontsize=9, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5),
           verticalalignment='top')
    
    insight2 = """
    WHAT WE DO:
    ═══════════════
    • Collect sample
    • Compute statistics:
      - Sample mean ("""+r"$\bar{x}$"+""")
      - Sample std (s)
      - Sample proportion ("""+r"$\hat{p}$"+""")
    
    ✓ These ESTIMATE the
      population parameters!
    """
    
    ax.text(7, 3.5, insight2, fontsize=9, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
           verticalalignment='top')
    
    # Mathematical notation
    notation = """
    NOTATION CHEAT SHEET:
    ══════════════════════════════
    
    Population          Sample
    ──────────────────────────────
    Size:        N                n
    Mean:        μ (mu)           """+r"$\bar{x}$"+""" (x-bar)
    Std Dev:     σ (sigma)        s
    Proportion:  p                """+r"$\hat{p}$"+""" (p-hat)
    Variance:    σ²               s²
    ──────────────────────────────
    
    As n ↑  →  """+r"$\bar{x}$"+""" → μ  and  s → σ
    """
    
    ax.text(4, 0.7, notation, fontsize=9, family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
           verticalalignment='bottom')
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def motivating_examples_visualization(py=False):
    """
    Visual examples showing why sampling is necessary
    """
    
    fig = plt.figure(figsize=(18, 12))
    
    #plt.title("WHY SAMPLING? Three Real-World Scenarios")
    
    # Example 1: Quality Control
    scenario_1 = """SCENARIO 1: Quality Control at Apple
    Apple manufactures 200 million iPhones per year
    Question: What's the defect rate?
    ┌──────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
    │ Option A - Test EVERYTHING (Population):                     │ Option B - Test A SAMPLE:                                        │
    │   • Test all 200 million phones                              │   • Randomly select 10,000 phones                                │
    │   • Cost: Requires destroying each phone to test battery,    │   • Perform destructive testing                                  │
    │     screen durability, etc.                                  │   • Cost: Lose 10,000 phones (~$7M)                              │
    │   • Result: Perfect knowledge... but no phones left to sell! │   • Benefit: Still have 199,990,000 phones to sell!              │
    │                                                              │   • Accuracy: Defect rate ± 0.1% with 95% confidence             │
    └──────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘
    ✓ Sampling is NECESSARY - testing everything would destroy the product!"""
    
    # Visualization for Example 1
    ax1 = plt.subplot(3, 2, 1)
    total_phones = 200_000_000
    sample_phones = 10_000
    
    # Population (grid of all phones)
    ax1.add_patch(Rectangle((0, 0), 10, 10, linewidth=3, edgecolor='blue', 
                            facecolor='lightblue', alpha=0.3))
    ax1.text(5, 11, f'Population: 200 Million iPhones', ha='center', fontsize=10, 
            fontweight='bold')
    
    # Sample (small box)
    ax1.add_patch(Rectangle((2, 2), 0.5, 0.5, linewidth=2, edgecolor='red', 
                            facecolor='red', alpha=0.7))
    ax1.text(1, 1.5, f'Sample\n10,000 phones\n(0.005%)', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
    
    # Arrow
    ax1.annotate('', xy=(2.25, 2), xytext=(2.25, 5),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax1.text(3, 3.5, 'Random\nSelection', fontsize=9, color='red', fontweight='bold')
    
    fig.text(0.05, 0.63, scenario_1, ha='left', fontsize=6, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))
    
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(-1, 13)
    ax1.axis('off')
    ax1.set_title('Scenario 1: Quality Control (Destructive Testing)\n(Must Sample)', fontsize=11, fontweight='bold')
    
    # Example 2: Medical Research
    scenario_2 = """SCENARIO 2: Clinical Drug Trial    
    Pharmaceutical company develops new diabetes medication
    Question: Does it work? What are side effects?    
    ┌────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
    │ Option A - Test on EVERYONE (Population):                  │ Option B - Test on A SAMPLE:                                     │
    │   • Population: 537 million people with diabetes worldwide │   • Recruit 5,000 volunteer patients                             │
    │   • Problem: Can't wait for ALL to participate             │   • Run controlled trial for 2 years                             │
    │   • Ethical issues: Unknown side effects                   │   • Result: Know efficacy and side effects with high confidence  │
    │   • Time: Decades                                          │   • Then: Safely release to entire population                    │
    └────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘    
    ✓ Sampling is NECESSARY - can't wait to test everyone!"""
    
    # Visualization for Example 2
    ax2 = plt.subplot(3, 2, 2)
    
    # Population (many people)
    population_size = 537_000_000
    sample_size = 5_000
    
    # Draw population as dots
    np.random.seed(42)
    x_pop = np.random.uniform(0, 10, 1000)
    y_pop = np.random.uniform(0, 10, 1000)
    ax2.scatter(x_pop, y_pop, s=1, alpha=0.3, color='blue')
    ax2.text(5, 11, f'Population: 537 Million Patients', ha='center', fontsize=10,
            fontweight='bold')
    
    # Sample (highlighted dots)
    x_sample = np.random.uniform(3, 7, 20)
    y_sample = np.random.uniform(3, 7, 20)
    ax2.scatter(x_sample, y_sample, s=100, alpha=0.9, color='red', 
               edgecolor='black', linewidth=1.5, zorder=5)
    
    ax2.add_patch(FancyBboxPatch((2.5, 2.5), 5, 5, boxstyle="round,pad=0.1",
                                 linewidth=2, edgecolor='red', facecolor='none',
                                 linestyle='--'))
    ax2.text(5, 0.5, f'Clinical Trial\n5,000 patients\n(0.0009%)', ha='center', 
            fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    fig.text(0.5, 0.63, scenario_2, ha='left', fontsize=6, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))
    
    ax2.set_xlim(-1, 11)
    ax2.set_ylim(-1, 13)
    ax2.axis('off')
    ax2.set_title('Scenario 2: Medical Research\n(Must Sample)', fontsize=11, fontweight='bold')
    
    # Example 3: Machine Learning
    scenario_3 = """SCENARIO 3: Training GPT-4")    
    OpenAI wants to train a language model
    Question: What's the performance on 'all possible text'?    
    ┌────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────┐
    │ Option A - Test on ALL TEXT (Population):                  │ Option B - Test on A SAMPLE:                                     │
    │   • Population: All possible English sentences             │   • Collect test set: 10,000 diverse text samples                │
    │   • Size: Literally infinite!                              │   • Measure accuracy on this sample                              │
    │   • Problem: Cannot even enumerate, let alone test         │   • Infer: Performance on sample ≈ performance on all text       │
    └────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────┘
    ✓ Sampling is NECESSARY - population is infinite/intractable!"""
    
    # Visualization for Example 3
    ax3 = plt.subplot(3, 2, 3)
    
    # Population (infinite text space)
    theta = np.linspace(0, 2*np.pi, 100)
    for r in range(1, 6):
        x_circle = 5 + r * np.cos(theta)
        y_circle = 5 + r * np.sin(theta)
        ax3.plot(x_circle, y_circle, 'b-', alpha=0.2, linewidth=1)
    
    ax3.text(5, 11.5, 'Population: "All Possible Text" (Infinite!)', ha='center',
            fontsize=10, fontweight='bold')
    ax3.text(5, 5, '∞', fontsize=60, ha='center', va='center', alpha=0.2, color='blue')
    
    # Sample points
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    for angle in angles:
        x = 5 + 3 * np.cos(angle)
        y = 5 + 3 * np.sin(angle)
        ax3.plot(x, y, 'ro', markersize=10, markeredgecolor='black', 
                markeredgewidth=1.5)
    
    ax3.text(5, -0.5, 'Test Set\n10,000 samples', ha='center', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    fig.text(0.05, 0.32, scenario_3, ha='left', fontsize=6, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.7))
    
    ax3.set_xlim(-1, 11)
    ax3.set_ylim(-5, 13)
    ax3.axis('off')
    ax3.set_title('Scenario 3: Testing an LLM\n(Must Sample)', fontsize=11, fontweight='bold')
    
    # Create summary comparison
    ax4 = plt.subplot(3, 1, 2)
    ax4.axis('off')
    
    summary_text = """
    COMMON PATTERN ACROSS ALL SCENARIOS:
    
    ┌─────────────────────┬──────────────────────┬─────────────────────┬──────────────────────┐
    │   Scenario          │   Population         │   Why Can't Test?   │   Solution           │
    ├─────────────────────┼──────────────────────┼─────────────────────┼──────────────────────┤
    │   Quality Control   │   200M iPhones       │   Destructive       │   Sample 10K         │
    │   Medical Research  │   537M patients      │   Time/Ethics       │   Sample 5K          │
    │   ML Testing        │   Infinite texts     │   Impossible        │   Sample 10K         │
    └─────────────────────┴──────────────────────┴─────────────────────┴──────────────────────┘
    
    KEY INSIGHT: We rarely have access to the entire population!
    
    ✓ Sometimes it's too expensive (quality control)
    ✓ Sometimes it's too slow (medical trials)  
    ✓ Sometimes it's physically impossible (infinite populations)
    ✓ Sometimes it's destructive (testing destroys the item)
    
    SOLUTION: Study a carefully chosen SAMPLE and make inferences about the POPULATION
    """
    
    ax4.text(0.5, 0.45, summary_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.subplots_adjust(hspace=0.8, top=0.6, bottom=0.3)
    
    plt.tight_layout()
    
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def interactive_sampling_demo(n_samples=100, distribution='Normal'):
    """
    Interactive widget to explore sampling
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
    # Generate samples based on distribution choice
    if distribution == 'Normal':
        samples = np.random.normal(0, 1, n_samples)
        x_theory = np.linspace(-4, 4, 1000)
        y_theory = norm.pdf(x_theory, 0, 1)
        title = "Normal(μ=0, σ=1)"
        xlim = [-4, 4]
    elif distribution == 'Uniform':
        samples = np.random.uniform(-2, 2, n_samples)
        x_theory = np.linspace(-2, 2, 1000)
        y_theory = np.ones_like(x_theory) * 0.25
        title = "Uniform(-2, 2)"
        xlim = [-3, 3]
    elif distribution == 'Exponential':
        samples = np.random.exponential(1, n_samples)
        x_theory = np.linspace(0, 6, 1000)
        y_theory = expon.pdf(x_theory, scale=1)
        title = "Exponential(λ=1)"
        xlim = [0, 6]
    else:  # Beta
        samples = np.random.beta(2, 5, n_samples)
        x_theory = np.linspace(0, 1, 1000)
        y_theory = beta_dist.pdf(x_theory, 2, 5)
        title = "Beta(α=2, β=5)"
        xlim = [0, 1]
            
    # Left plot: Theoretical distribution
    axes[0].fill_between(x_theory, 0, y_theory, alpha=0.3, color='blue')
    axes[0].plot(x_theory, y_theory, 'b-', linewidth=2.5)
    axes[0].set_xlabel('Value', fontsize=12)
    axes[0].set_ylabel('Probability Density', fontsize=12)
    axes[0].set_title(f'Theoretical Distribution\n{title}', 
                      fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(xlim)
            
    # Right plot: Samples histogram
    axes[1].hist(samples, bins=min(30, n_samples//10 + 1), alpha=0.7, 
                        edgecolor='black', density=True, color='red', label=f'{n_samples} samples')
    axes[1].plot(x_theory, y_theory, 'b-', linewidth=2.5, 
                      label='True distribution', zorder=10)
    axes[1].set_xlabel('Value', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title(f'Samples from Distribution\nn = {n_samples}', 
                            fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(xlim)
            
    plt.tight_layout()
    plt.show() 
            
    # Statistics
    print(f"\nSample Statistics (n={n_samples}):")
    print(f"  Mean: {np.mean(samples):.4f}")
    print(f"  Std:  {np.std(samples):.4f}")
    print(f"  Min:  {np.min(samples):.4f}")
    print(f"  Max:  {np.max(samples):.4f}")

def ml_sampling_example():
    """
    Show where sampling appears in ML workflows
    """
    
    print("="*70)
    print("WHERE SAMPLING APPEARS IN MACHINE LEARNING")
    print("="*70)
    
    # Example 1: Mini-batch training
    print("\n1️⃣  MINI-BATCH GRADIENT DESCENT")
    print("-" * 70)
    print("Instead of using all data, we sample random batches:")
    
    # Simulate a dataset
    dataset_size = 10000
    batch_size = 32
    
    # Sample a mini-batch
    np.random.seed(42)
    batch_indices = np.random.choice(dataset_size, size=batch_size, replace=False)
    
    print(f"  Dataset size: {dataset_size:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  First 10 sampled indices: {batch_indices[:10]}")
    print(f"\n  💡 Each training step samples a DIFFERENT random batch")
    print(f"     This is sampling from a discrete uniform distribution!")
    
    # Example 2: Dropout
    print(f"\n2️⃣  DROPOUT REGULARIZATION")
    print("-" * 70)
    print("During training, we randomly 'drop' neurons:")
    
    n_neurons = 1000
    dropout_rate = 0.5
    
    # Sample dropout mask (Bernoulli distribution)
    dropout_mask = np.random.binomial(1, 1-dropout_rate, n_neurons)
    active_neurons = np.sum(dropout_mask)
    
    print(f"  Total neurons: {n_neurons}")
    print(f"  Dropout rate: {dropout_rate}")
    print(f"  Active neurons this forward pass: {active_neurons}")
    print(f"  Dropout mask sample: {dropout_mask[:20]}...")
    print(f"\n  💡 Each forward pass samples a DIFFERENT mask")
    print(f"     This is sampling from Bernoulli(p={1-dropout_rate})!")
    
    # Example 3: Data augmentation
    print(f"\n3️⃣  DATA AUGMENTATION")
    print("-" * 70)
    print("Randomly transform images with sampled parameters:")
    
    # Sample augmentation parameters
    rotation_angle = np.random.uniform(-15, 15)  # degrees
    brightness_factor = np.random.uniform(0.8, 1.2)
    crop_x = np.random.randint(0, 32)
    crop_y = np.random.randint(0, 32)
    
    print(f"  Rotation: {rotation_angle:.2f}° (sampled from Uniform(-15, 15))")
    print(f"  Brightness: {brightness_factor:.2f}× (sampled from Uniform(0.8, 1.2))")
    print(f"  Crop position: ({crop_x}, {crop_y}) (sampled from discrete Uniform)")
    print(f"\n  💡 Each image gets DIFFERENT random transformations")
    print(f"     This creates infinite variations from finite data!")
    
    # Example 4: Random forest
    print(f"\n4️⃣  RANDOM FOREST")
    print("-" * 70)
    print("Each tree samples both data points and features:")
    
    n_trees = 100
    n_samples_per_tree = int(0.7 * dataset_size)  # Bootstrap sample
    n_features = 50
    n_features_per_split = int(np.sqrt(n_features))
    
    print(f"  Total trees: {n_trees}")
    print(f"  Each tree samples {n_samples_per_tree:,} data points (with replacement)")
    print(f"  Each split considers {n_features_per_split} random features")
    print(f"\n  💡 Every tree uses DIFFERENT random samples")
    print(f"     This is bootstrap sampling + feature sampling!")
    
# Interactive: Estimate π by throwing darts
def estimate_pi_visual(n_darts=1000, py=False):
    """Throw random darts at a square containing a circle"""
    x = np.random.uniform(-1, 1, n_darts)
    y = np.random.uniform(-1, 1, n_darts)
    
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n_darts
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.scatter(x[inside_circle], y[inside_circle], c='red', s=2, alpha=0.5, label='Inside')
    ax.scatter(x[~inside_circle], y[~inside_circle], c='tab:blue', s=2, alpha=0.5, label='Outside')
    circle = Circle((0, 0), 1, color='#EF9A9A', linewidth=2, alpha=0.3)
    ax.add_patch(circle)
    
    # draw lines
    ax.vlines(1, ymin=-1, ymax=1, linestyles="--", color='tab:blue', alpha=0.3)
    ax.hlines(1, xmin=-1, xmax=1, linestyles="--", color='tab:blue', alpha=0.3)
    ax.vlines(-1, ymin=-1, ymax=1, linestyles="--", color='tab:blue', alpha=0.3)
    ax.hlines(-1, xmin=-1, xmax=1, linestyles="--", color='tab:blue', alpha=0.3)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect('equal')
    ax.legend()
    ax.set_title(f'π estimate: {pi_estimate:.4f} (True: {np.pi:.6f})\nSample size: {n_darts}\nError: {abs(pi_estimate - np.pi):.4f}')
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return pi_estimate, fig

def bootstrap_demo(data, statistic_func, n_bootstrap=1000, confidence_level=0.95, py=False):
    """
    Demonstrate bootstrap with visualization
    
    Args:
        data: Original sample
        statistic_func: Function to compute statistic (e.g., np.mean)
        n_bootstrap: Number of bootstrap samples
        confidence_level: For confidence interval
    """
    n = len(data)
    original_stat = statistic_func(data)
    
    # Generate bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    lower = np.percentile(bootstrap_stats, 100 * alpha/2)
    upper = np.percentile(bootstrap_stats, 100 * (1 - alpha/2))
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original data
    axes[0, 0].hist(data, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(original_stat, color='red', linewidth=2, label=f'Statistic: {original_stat:.3f}')
    axes[0, 0].set_title(f'Original Sample, n = {n}')
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    
    # Example bootstrap samples
    for i in range(3):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        axes[0, 1].hist(bootstrap_sample, bins=30, alpha=0.3, label=f'Bootstrap {i+1}')
    axes[0, 1].set_title('Three Bootstrap Samples (with replacement), ' + r"$n_i = $" + f"{n}")
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].legend()
    
    # Bootstrap distribution
    axes[1, 0].hist(bootstrap_stats, bins=50, alpha=0.7, edgecolor='black', density=True)
    axes[1, 0].axvline(original_stat, color='red', linewidth=2, label=f'Original: {original_stat:.3f}')
    axes[1, 0].axvline(lower, color='green', linestyle='--', linewidth=2, label=f'{confidence_level*100}% CI')
    axes[1, 0].axvline(upper, color='green', linestyle='--', linewidth=2)
    axes[1, 0].set_title(f'Bootstrap Distribution ({n_bootstrap} samples)')
    axes[1, 0].set_xlabel('Statistic Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].legend()
    
    # Convergence
    cumulative_mean = np.cumsum(bootstrap_stats) / np.arange(1, len(bootstrap_stats) + 1)
    axes[1, 1].plot(cumulative_mean, alpha=0.7)
    axes[1, 1].axhline(original_stat, color='red', linestyle='--', label='Original Statistic')
    axes[1, 1].set_title('Bootstrap Estimate Convergence')
    axes[1, 1].set_xlabel('Number of Bootstrap Samples')
    axes[1, 1].set_ylabel('Cumulative Mean')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    # Results summary
    print(f"\n{'='*50}")
    print(f"Bootstrap Results ({n_bootstrap} samples)")
    print(f"{'='*50}")
    print(f"Original Statistic: {original_stat:.4f}")
    print(f"Bootstrap Mean: {np.mean(bootstrap_stats):.4f}")
    print(f"Bootstrap Std Error: {np.std(bootstrap_stats):.4f}")
    print(f"{confidence_level*100}% Confidence Interval: [{lower:.4f}, {upper:.4f}]")
    print(f"{'='*50}\n")
    
    return bootstrap_stats, (lower, upper)

def visualize_uncertainty_concept():
    """
    Show what uncertainty means through repeated sampling
    """
    
    print("="*80)
    print("WHAT IS UNCERTAINTY? A Visual Explanation")
    print("="*80)
    
    # Create a "population"
    np.random.seed(42)
    true_mean = 5.0
    true_std = 2.0
    population = np.random.normal(true_mean, true_std, 100000)
    
    print("\nImagine: We have a HUGE population (but we can't measure it all)")
    print(f"True population mean: μ = {true_mean}")
    print(f"True population std: σ = {true_std}")
    
    # Take multiple samples
    n_samples = 30
    n_repetitions = 8
    
    print(f"\nExperiment: Take {n_repetitions} different samples of size {n_samples}")
    print("(In reality, we can only afford ONE sample, but let's see what happens)")
    
    samples = []
    sample_means = []
    
    for i in range(n_repetitions):
        sample = np.random.choice(population, size=n_samples, replace=False)
        samples.append(sample)
        sample_means.append(np.mean(sample))
    
    print(f"\nResults from {n_repetitions} different samples:")
    print(f"{'Sample #':<10} {'Mean':<10} {'Difference from true':<20}")
    print("-" * 45)
    for i, mean in enumerate(sample_means, 1):
        diff = mean - true_mean
        print(f"{i:<10} {mean:<10.3f} {diff:>+7.3f}")
    
    print(f"\n{'='*45}")
    print(f"Range of estimates: [{min(sample_means):.3f}, {max(sample_means):.3f}]")
    print(f"Spread (std dev): {np.std(sample_means):.3f}")
    print(f"{'='*45}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Show different samples
    ax1 = axes[0, 0]
    
    for i in range(min(4, n_repetitions)):
        ax1.hist(samples[i], bins=15, alpha=0.4, label=f'Sample {i+1}', 
                edgecolor='black', linewidth=0.5)
    
    ax1.axvline(true_mean, color='red', linewidth=3, linestyle='--',
               label=f'True mean = {true_mean}')
    ax1.set_xlabel('Value', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Four Different Samples from Same Population\n(Notice they look different!)', 
                 fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Distribution of sample means
    ax2 = axes[0, 1]
    
    ax2.hist(sample_means, bins=6, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.axvline(true_mean, color='red', linewidth=3, linestyle='--',
               label=f'True mean = {true_mean}')
    ax2.axvline(np.mean(sample_means), color='green', linewidth=3,
               label=f'Average of estimates = {np.mean(sample_means):.3f}')
    
    # Show spread
    spread = np.std(sample_means)
    ax2.axvspan(np.mean(sample_means) - spread, np.mean(sample_means) + spread,
               alpha=0.2, color='orange', label=f'±1 SD = {spread:.3f}')
    
    ax2.set_xlabel('Sample Mean', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title(f'Distribution of {n_repetitions} Sample Means\n'
                 'THIS SPREAD IS THE UNCERTAINTY!', 
                 fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Show estimates with uncertainty bars
    ax3 = axes[1, 0]
    
    x_pos = range(1, n_repetitions + 1)
    ax3.scatter(x_pos, sample_means, s=200, c='blue', alpha=0.6, 
               edgecolors='black', linewidth=2, zorder=5)
    
    ax3.axhline(true_mean, color='red', linewidth=3, linestyle='--',
               label=f'True mean = {true_mean}', zorder=4)
    
    # Show spread as shaded region
    mean_of_means = np.mean(sample_means)
    std_of_means = np.std(sample_means)
    ax3.axhspan(mean_of_means - std_of_means, mean_of_means + std_of_means,
               alpha=0.2, color='orange', label=f'Typical variation: ±{std_of_means:.3f}')
    
    ax3.set_xlabel('Sample Number', fontsize=12)
    ax3.set_ylabel('Estimated Mean', fontsize=12)
    ax3.set_title('Each Sample Gives a Different Estimate\n'
                 '(The vertical spread shows UNCERTAINTY)', 
                 fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(x_pos)
    
    # Plot 4: The key insight
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    insight_text = f"""
    THE KEY INSIGHT: UNCERTAINTY
    {'='*50}
    
    What we saw:
    {'─'*50}
    • Collected {n_repetitions} different samples (size {n_samples})
    • Each gave a DIFFERENT estimate
    • Estimates ranged from {min(sample_means):.2f} to {max(sample_means):.2f}
    • Typical variation: ±{std_of_means:.3f}
    
    What this means:
    {'─'*50}
    If we COULD repeat our experiment:
    • We'd get different data each time
    • Our estimate would vary
    • The SPREAD of these estimates is the
      UNCERTAINTY
    
    The Problem:
    {'─'*50}
    In reality, we can only afford ONE sample!
    
    We want to know: "How much would my 
    estimate vary if I could repeat?"
    
    But we CAN'T actually repeat...
    
    The Bootstrap Solution:
    {'─'*50}
    ✓ Can't collect new samples from population
    ✓ BUT can resample from our ONE sample
    ✓ This simulates "collecting new samples"
    ✓ Shows how much estimate would vary
    
    {'='*50}
    
    Next: We'll use BOOTSTRAP to estimate this
    uncertainty from just ONE sample!
    """
    
    ax4.text(0.05, 0.95, insight_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("KEY TAKEAWAY:")
    print("="*80)
    print("UNCERTAINTY = How much our estimate would vary if we repeated sampling")
    print()
    print("We visualized this by actually repeating the sampling.")
    print("But in practice, we only have ONE sample.")
    print()
    print("BOOTSTRAP lets us estimate this uncertainty from just one sample!")
    print("="*80 + "\n")
    
    return std_of_means

def bootstrap_uncertainty_explanation():
    """
    Show how bootstrap estimates uncertainty without collecting new data
    """
    
    print("="*80)
    print("BOOTSTRAP: Estimating Uncertainty from ONE Sample")
    print("="*80)
    
    print("\n" + "="*80)
    print("THE SETUP")
    print("="*80)
    
    # Our ONE sample (in reality, this is all we have)
    np.random.seed(42)
    true_mean = 5.0
    our_sample = np.random.normal(true_mean, 2.0, 30)
    original_estimate = np.mean(our_sample)
    
    print(f"\nWe have ONE sample of size {len(our_sample)}")
    print(f"Our sample: {our_sample[:10]}... (showing first 10)")
    print(f"\nOur estimate from this sample: {original_estimate:.3f}")
    print()
    print("Question: How much would this estimate vary with different samples?")
    print("Problem: We can't collect new samples!")
    print("Solution: BOOTSTRAP!")
    
    print("\n" + "="*80)
    print("BOOTSTRAP PROCEDURE")
    print("="*80)
    
    procedure = """
    Step 1: Start with our ONE sample
    
    Step 2: Create a "new sample" by:
            • Randomly selecting n values FROM our sample
            • Sampling WITH REPLACEMENT
            • This gives us a "bootstrap sample"
    
    Step 3: Compute the statistic on this bootstrap sample
            (e.g., mean, median, whatever we're estimating)
    
    Step 4: Repeat steps 2-3 many times (e.g., 1000 times)
    
    Step 5: Look at the DISTRIBUTION of bootstrap estimates
            • The spread of this distribution = UNCERTAINTY
            • Shows how much estimate varies across "fake samples"
    """
    print(procedure)
    
    print("\n" + "="*80)
    print("BOOTSTRAP IN ACTION")
    print("="*80)
    
    n_bootstrap = 1000
    bootstrap_means = []
    
    print(f"\nGenerating {n_bootstrap} bootstrap samples...")
    print("\nFirst 5 bootstrap samples (showing first 10 values of each):")
    print()
    
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(our_sample, size=len(our_sample), replace=True)
        bootstrap_mean = np.mean(bootstrap_sample)
        bootstrap_means.append(bootstrap_mean)
        
        if i < 5:
            print(f"Bootstrap sample {i+1}: {bootstrap_sample[:10]}...")
            print(f"  → Mean: {bootstrap_mean:.3f}")
            print()
    
    bootstrap_means = np.array(bootstrap_means)
    
    print(f"... (continued for {n_bootstrap} total bootstrap samples)")
    
    # Analyze the bootstrap distribution
    bootstrap_std = np.std(bootstrap_means)
    bootstrap_range = [np.min(bootstrap_means), np.max(bootstrap_means)]
    
    print("\n" + "="*80)
    print("RESULTS: What Bootstrap Tells Us")
    print("="*80)
    
    print(f"\nOriginal estimate: {original_estimate:.3f}")
    print(f"\nBootstrap distribution:")
    print(f"  • Range of estimates: [{bootstrap_range[0]:.3f}, {bootstrap_range[1]:.3f}]")
    print(f"  • Standard deviation: {bootstrap_std:.3f}")
    print(f"  • Middle 95%: [{np.percentile(bootstrap_means, 2.5):.3f}, {np.percentile(bootstrap_means, 97.5):.3f}]")
    
    print("\nINTERPRETATION:")
    print(f"  If we COULD collect new samples of size {len(our_sample)},")
    print(f"  our estimates would typically vary by about ±{bootstrap_std:.3f}")
    print(f"  around the value {original_estimate:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Original sample
    ax1 = axes[0, 0]
    ax1.hist(our_sample, bins=12, alpha=0.7, edgecolor='black', color='lightblue')
    ax1.axvline(original_estimate, color='red', linewidth=3, linestyle='--',
               label=f'Mean = {original_estimate:.3f}')
    ax1.set_xlabel('Value', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title(f'Original Sample (n={len(our_sample)})\nThis is ALL we have!', 
                 fontsize=11, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Example bootstrap samples
    ax2 = axes[0, 1]
    
    for i in range(3):
        bs = np.random.choice(our_sample, size=len(our_sample), replace=True)
        ax2.hist(bs, bins=12, alpha=0.4, label=f'Bootstrap {i+1}', edgecolor='black', linewidth=0.5)
    
    ax2.set_xlabel('Value', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Three Bootstrap Samples\n(Resampled WITH replacement)', 
                 fontsize=11, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Distribution of bootstrap means
    ax3 = axes[0, 2]
    
    ax3.hist(bootstrap_means, bins=40, alpha=0.7, edgecolor='black', color='skyblue')
    ax3.axvline(original_estimate, color='red', linewidth=3, linestyle='--',
               label=f'Original = {original_estimate:.3f}')
    ax3.axvline(np.mean(bootstrap_means), color='green', linewidth=2.5,
               label=f'Bootstrap avg = {np.mean(bootstrap_means):.3f}')
    
    # Show spread
    ax3.axvspan(original_estimate - bootstrap_std, original_estimate + bootstrap_std,
               alpha=0.2, color='orange', label=f'±1 SD = {bootstrap_std:.3f}')
    
    ax3.set_xlabel('Bootstrap Mean', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title(f'Distribution of {n_bootstrap} Bootstrap Means\n'
                 'THE SPREAD = UNCERTAINTY', 
                 fontsize=11, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Show what "sampling with replacement" means
    ax4 = axes[1, 0]
    ax4.axis('off')
    
    # Create visual example
    example_original = [1, 2, 3, 4, 5]
    np.random.seed(123)
    example_bootstrap = np.random.choice(example_original, size=5, replace=True).tolist()
    
    replacement_text = f"""
    SAMPLING WITH REPLACEMENT
    {'='*40}
    
    Original sample: {example_original}
    
    Bootstrap sample: {example_bootstrap}
    
    Notice:
    {'─'*40}
    • Some values appear MULTIPLE times
      (e.g., {max(set(example_bootstrap), key=example_bootstrap.count)} appears {example_bootstrap.count(max(set(example_bootstrap), key=example_bootstrap.count))} times)
    
    • Some values DON'T appear at all
    
    • This is CRUCIAL! Without replacement,
      we'd just get the same sample back!
    
    Why replacement?
    {'─'*40}
    • Mimics getting a "new sample"
    • Each bootstrap sample is DIFFERENT
    • Creates the variability we need to
      estimate uncertainty
    
    On average:
    • ~63.2% of original values appear
      in each bootstrap sample
    • Some appear multiple times
    • Some don't appear at all
    """
    
    ax4.text(0.05, 0.95, replacement_text, transform=ax4.transAxes,
            fontsize=8.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Plot 5: Comparison - what if we had actually repeated sampling?
    ax5 = axes[1, 1]
    
    # Simulate actually collecting new samples (we can't do this in reality!)
    actual_means = []
    for _ in range(1000):
        new_sample = np.random.normal(true_mean, 2.0, len(our_sample))
        actual_means.append(np.mean(new_sample))
    actual_means = np.array(actual_means)
    
    ax5.hist(actual_means, bins=40, alpha=0.5, edgecolor='black', 
            color='lightcoral', label='Actually collecting new samples', density=True)
    ax5.hist(bootstrap_means, bins=40, alpha=0.5, edgecolor='black',
            color='skyblue', label='Bootstrap (from one sample)', density=True)
    
    ax5.set_xlabel('Mean estimate', fontsize=11)
    ax5.set_ylabel('Density', fontsize=11)
    ax5.set_title('Bootstrap vs Reality\n(Bootstrap approximates actual variability!)', 
                 fontsize=11, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    actual_std = np.std(actual_means)
    ax5.text(0.05, 0.95, 
            f'Actual SD: {actual_std:.3f}\nBootstrap SD: {bootstrap_std:.3f}\nDifference: {abs(actual_std - bootstrap_std):.3f}',
            transform=ax5.transAxes, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9)
    
    # Plot 6: Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
    ANSWERING THE QUESTION
    {'='*40}
    
    Original Question:
    {'─'*40}
    "If I collected a different sample,
    would I get a similar answer?"
    
    Bootstrap Answer:
    {'─'*40}
    Your estimate: {original_estimate:.3f}
    
    Typical variation: ±{bootstrap_std:.3f}
    
    Range of plausible values:
    [{np.percentile(bootstrap_means, 2.5):.3f}, {np.percentile(bootstrap_means, 97.5):.3f}]
    
    Interpretation:
    {'─'*40}
    ✓ If you repeated with new samples,
      estimates would typically fall
      within ±{bootstrap_std:.3f} of {original_estimate:.3f}
    
    ✓ Your estimate is {"STABLE" if bootstrap_std < 0.3 else "SOMEWHAT VARIABLE"}
      (small spread = more reliable)
    
    ✓ 95% of repeated samples would give
      estimates between {np.percentile(bootstrap_means, 2.5):.2f} and {np.percentile(bootstrap_means, 97.5):.2f}
    
    What this tells you:
    {'─'*40}
    • How much to trust your estimate
    • Whether you need more data
    • The precision of your measurement
    
    WITHOUT collecting any new data!
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
            fontsize=8.5, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*80)
    print("BOTTOM LINE")
    print("="*80)
    print(f"""
Bootstrap gives us a DISTRIBUTION of estimates.

The SPREAD of this distribution tells us:
  "How different would my estimate be with a different sample?"

For our example:
  • Original estimate: {original_estimate:.3f}
  • Typical variation: ±{bootstrap_std:.3f}
  • This means: if we could repeat, we'd typically get values
    between {original_estimate - bootstrap_std:.3f} and {original_estimate + bootstrap_std:.3f}

The key metric: Standard deviation of bootstrap distribution = {bootstrap_std:.3f}
  • Small SD → stable estimate (low uncertainty)
  • Large SD → unstable estimate (high uncertainty)
    """)
    print("="*80 + "\n")
    
    return bootstrap_means, bootstrap_std

if __name__ == "__main__":
    #estimate_pi_visual(py=True)
    #demonstrate_sampling_concept(py=True)
    #interactive_population_sample_demo(py=True)
    #sampling_process_diagram(py=True)
    #motivating_examples_visualization(py=True)
    #complete_sampling_picture(py=True)
    #interactive_sampling_demo(n_samples=100, distribution='Normal')
    #interactive_sampling_demo(n_samples=100, distribution='Uniform')
    #interactive_sampling_demo(n_samples=100, distribution='Exponential')
    #ml_sampling_example()    
    #np.random.seed(42)
    #sample_data = np.random.exponential(scale=2, size=100)
    #bootstrap_stats, ci = bootstrap_demo(sample_data, np.mean, n_bootstrap=2000)
    #uncertainty = visualize_uncertainty_concept()
    bootstrap_means, bootstrap_std = bootstrap_uncertainty_explanation()
