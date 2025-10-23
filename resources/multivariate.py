import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from ipywidgets import interact, FloatSlider, IntSlider

plt.rcParams['font.family'] = ['DejaVu Sans', 'Segoe UI Emoji']

def generate_movie_data_normal(n_samples = 500, py=False):
    # Generate synthetic movie rating data
    # Create correlated data: longer watch time -> higher ratings (with noise)
    mean = [50, 3.5]  # mean watch time = 50 min, mean rating = 3.5
    cov = [[400, 20],  # covariance matrix
           [20, 0.8]]   # watch_time variance=400, rating variance=0.8, cov=20
    
    # Generate bivariate normal data
    data = np.random.multivariate_normal(mean, cov, n_samples)
    watch_time = np.clip(data[:, 0], 5, 120)  # clip to reasonable range (limits array values within a specified interval)
    rating_continuous = np.clip(data[:, 1], 1, 5)  # clip to 1-5
    rating = np.round(rating_continuous)  # discretize ratings

    # Create DataFrame
    df = pd.DataFrame({
        'watch_time': watch_time,
        'rating': rating
    })
    
    # Visualize the data
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(df['watch_time'], df['rating'], alpha=0.5, s=50)
    axes[0].set_xlabel('Watch Time (minutes)', fontsize=12)
    axes[0].set_ylabel('Rating (1-5 stars)', fontsize=12)
    axes[0].set_title('User Behavior: Watch Time vs Rating', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2D histogram (joint distribution visualization)
    axes[1].hist2d(df['watch_time'], df['rating'], bins=20, cmap='Blues')
    axes[1].set_xlabel('Watch Time (minutes)', fontsize=12)
    axes[1].set_ylabel('Rating (1-5 stars)', fontsize=12)
    axes[1].set_title('Joint Distribution (2D Histogram)', fontsize=14, fontweight='bold')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Frequency')

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def generate_movie_data():
    n_samples = 500
    np.random.seed(42)
    
    # We'll create a mixture model to capture different user behaviors
    # 1. Main group: Positive correlation (watched long, rated high)
    # 2. "Quick lovers": Short watch, high rating
    # 3. "Hate watchers": Long watch, low rating

    # Main group (70% of data) - positive correlation
    n_main = int(0.70 * n_samples)
    mean_main = [55, 4.0] # mean watch time = 55 min, mean rating = 4
    cov_main = [[350, 18], [18, 0.6]] # watch_time variance=350, rating variance=0.6, cov=18
    data_main = np.random.multivariate_normal(mean_main, cov_main, n_main)

    # "Quick lovers" (15% of data) - short time, high ratings
    n_quick = int(0.15 * n_samples)
    watch_quick = np.random.uniform(5, 25, n_quick)  # 5-25 minutes
    rating_quick = np.random.uniform(4, 5, n_quick)  # 4-5 stars

    # "Hate watchers" (15% of data) - long time, low ratings
    n_hate = n_samples - n_main - n_quick  # remaining
    watch_hate = np.random.uniform(60, 110, n_hate)  # 60-110 minutes
    rating_hate = np.random.uniform(1, 2.5, n_hate)  # 1-2.5 stars
    
    # Combine all groups
    watch_time = np.concatenate([
        np.clip(data_main[:, 0], 5, 120),
        watch_quick,
        watch_hate
    ])

    rating_continuous = np.concatenate([
        np.clip(data_main[:, 1], 1, 5),
        rating_quick,
        rating_hate
    ])

    rating = np.round(rating_continuous)  # discretize ratings

    # Create DataFrame with group labels for analysis
    group_labels = ['Main'] * n_main + ['Quick Lover'] * n_quick + ['Hate Watcher'] * n_hate

    df = pd.DataFrame({
        'watch_time': watch_time,
        'rating': rating,
        'rating_continuous': rating_continuous,
        'group': group_labels
    })
    
    return df

def get_vix_movie_data_joint(py=False):
    
    df = generate_movie_data()
    
    # Visualize with highlighting of special cases
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Color by group
    colors = {'Main': 'steelblue', 'Quick Lover': 'green', 'Hate Watcher': 'red'}
    for group, color in colors.items():
        mask = df['group'] == group
        axes[0].scatter(df[mask]['watch_time'], df[mask]['rating'], 
                    alpha=0.6, s=50, c=color, label=group, edgecolors='black', linewidth=0.5)

    axes[0].set_xlabel('Watch Time (minutes)', fontsize=12)
    axes[0].set_ylabel('Rating (1-5 stars)', fontsize=12)
    axes[0].set_title('User Behavior: Three Distinct Groups', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Add annotations for interesting cases
    axes[0].annotate('Quick Lovers!\n(loved it fast)', 
                    xy=(15, 4.5), xytext=(25, 3),
                    arrowprops=dict(arrowstyle='->', color='green', lw=2),
                    fontsize=11, color='darkgreen', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7))

    axes[0].annotate('Hate Watchers!\n(couldn\'t stop)', 
                    xy=(85, 2), xytext=(70, 3.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=11, color='darkred', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.7))

    # Plot 2: 2D histogram showing density
    axes[1].hist2d(df['watch_time'], df['rating'], bins=20, cmap='YlOrRd')
    axes[1].set_xlabel('Watch Time (minutes)', fontsize=12)
    axes[1].set_ylabel('Rating (1-5 stars)', fontsize=12)
    axes[1].set_title('Joint Distribution (showing all patterns)', fontsize=14, fontweight='bold')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Frequency')

    # Highlight regions
    from matplotlib.patches import Rectangle
    # Quick lovers region
    quick_rect = Rectangle((5, 3.5), 20, 1.5, linewidth=2, edgecolor='green', 
                        facecolor='none', linestyle='--', label='Quick Lovers')
    axes[1].add_patch(quick_rect)

    # Hate watchers region
    hate_rect = Rectangle((60, 0.5), 50, 2, linewidth=2, edgecolor='red', 
                          facecolor='none', linestyle='--', label='Hate Watchers')
    axes[1].add_patch(hate_rect)

    axes[1].legend(fontsize=10)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def get_viz_movie_data(py=False):
    
    df = generate_movie_data()
    
    # Analyze each group
    print("\n📊 Group Statistics:")
    print("="*60)

    for group in ['Main', 'Quick Lover', 'Hate Watcher']:
        group_data = df[df['group'] == group]
        print(f"\n{group} ({len(group_data)} users):")
        print(f"  Watch Time: {group_data['watch_time'].mean():.1f} ± {group_data['watch_time'].std():.1f} min")
        print(f"  Rating:     {group_data['rating'].mean():.2f} ± {group_data['rating'].std():.2f} stars")
        
        # Show correlation within group
        corr = np.corrcoef(group_data['watch_time'], group_data['rating'])[0, 1]
        print(f"  Correlation within group: ρ = {corr:.3f}")

    # Find specific examples
    print("\n" + "="*60)
    print("🎯 Specific Examples from Data:")
    print("="*60)

    # Quick lovers: short time, high rating
    quick_examples = df[(df['watch_time'] < 25) & (df['rating'] >= 4)]
    print(f"\n'Quick Lovers' (watch < 25 min, rating ≥ 4): {len(quick_examples)} cases")
    if len(quick_examples) > 0:
        sample = quick_examples.sample(min(3, len(quick_examples)))
        for idx, row in sample.iterrows():
            print(f"  → {row['watch_time']:.1f} min, rating {row['rating']:.0f} ⭐")

    # Hate watchers: long time, low rating  
    hate_examples = df[(df['watch_time'] > 60) & (df['rating'] <= 2)]
    print(f"\n'Hate Watchers' (watch > 60 min, rating ≤ 2): {len(hate_examples)} cases")
    if len(hate_examples) > 0:
        sample = hate_examples.sample(min(3, len(hate_examples)))
        for idx, row in sample.iterrows():
            print(f"  → {row['watch_time']:.1f} min, rating {row['rating']:.0f} ⭐")

    # Overall correlation
    overall_corr = np.corrcoef(df['watch_time'], df['rating'])[0, 1]
    print(f"\n📈 Overall Correlation: ρ = {overall_corr:.3f}")
    print("   (Positive but not perfect - reflects the mixed behaviors!)")
    
    # Enhanced visualization for the hook
    fig2, ax2 = plt.subplots(figsize=(12, 7))

    # Plot with different colors and sizes
    scatter_main = ax2.scatter(df[df['group']=='Main']['watch_time'], 
                            df[df['group']=='Main']['rating'], 
                            alpha=0.4, s=40, c='steelblue', label='Typical users')

    scatter_quick = ax2.scatter(df[df['group']=='Quick Lover']['watch_time'], 
                            df[df['group']=='Quick Lover']['rating'], 
                            alpha=0.7, s=80, c='green', marker='^', 
                            edgecolors='darkgreen', linewidth=1.5,
                            label='❤️ Quick Lovers (short + high)')

    scatter_hate = ax2.scatter(df[df['group']=='Hate Watcher']['watch_time'], 
                            df[df['group']=='Hate Watcher']['rating'], 
                            alpha=0.7, s=80, c='red', marker='v',
                            edgecolors='darkred', linewidth=1.5,
                            label='😤 Hate Watchers (long + low)')

    ax2.set_xlabel('Watch Time (minutes)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Rating (1-5 stars)', fontsize=14, fontweight='bold')
    ax2.set_title('The Recommendation System Challenge: Not All Patterns Are Simple!', 
                fontsize=16, fontweight='bold', pad=20)
    ax2.legend(fontsize=12, loc='center right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 5.5)

    # Add text box with the mystery
    textstr = f'''The Mystery:
    - {len(quick_examples)} users watched <25 min but rated ≥4 stars
    - {len(hate_examples)} users watched >60 min but rated ≤2 stars
    - Overall correlation: ρ = {overall_corr:.2f} (not perfect!)

    Can we predict better than just using watch time alone?'''

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax2.text(0.02, 0.2, textstr, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, family='monospace')

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    print("\n💡 This is why the problem is interesting!")
    print("   The data has MIXED patterns - not just simple correlation.")
    print("   This makes prediction challenging and multivariate modeling essential!")
    
    return fig2

def viz_joint_distr_cont(py=False):
    df = generate_movie_data()
    
    # Estimate parameters from data
    mu = df[['watch_time', 'rating']].mean().values
    cov_matrix = df[['watch_time', 'rating']].cov().values

    # Create grid for plotting
    x = np.linspace(0, 100, 100)
    y = np.linspace(1, 5, 100)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    # Calculate joint PDF values
    rv = multivariate_normal(mu, cov_matrix)
    Z = rv.pdf(pos)

    # 3D surface plot
    fig = plt.figure(figsize=(14, 6))

    # 3D view
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('Watch Time (min)', fontsize=10)
    ax1.set_ylabel('Rating', fontsize=10)
    ax1.set_zlabel('Density', fontsize=10)
    ax1.set_title('Joint PDF: f(watch_time, rating)', fontsize=12, fontweight='bold')
    fig.colorbar(surf, ax=ax1, shrink=0.5)

    # Contour plot (top-down view)
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=15, cmap='viridis')
    ax2.set_xlabel('Watch Time (minutes)', fontsize=10)
    ax2.set_ylabel('Rating', fontsize=10)
    ax2.set_title('Joint PDF: Contour View', fontsize=12, fontweight='bold')
    fig.colorbar(contour, ax=ax2)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    print("\n💡 Key Insight:")
    print("   Higher density (brighter color) = more probable combinations")
    print("   Peak shows most common (watch_time, rating) pairs")
    
    return fig

def covariance_demo(py=False):
    # Visual demonstration of covariance
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Generate examples with different covariances
    n = 200
    x = np.random.randn(n)

    # Positive covariance
    y_pos = x + np.random.randn(n) * 0.5
    axes[0].scatter(x, y_pos, alpha=0.6)
    axes[0].set_title('Positive Covariance (Cov > 0)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    # Negative covariance
    y_neg = -x + np.random.randn(n) * 0.5
    axes[1].scatter(x, y_neg, alpha=0.6, color='coral')
    axes[1].set_title('Negative Covariance (Cov < 0)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    # Zero covariance
    y_zero = np.random.randn(n)
    axes[2].scatter(x, y_zero, alpha=0.6, color='gray')
    axes[2].set_title('Zero Covariance (Cov ≈ 0)', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('X')
    axes[2].set_ylabel('Y')
    axes[2].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].axvline(0, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    print("\n💡 Visual Insight:")
    print("   - Positive: Points trend upward-right")
    print("   - Negative: Points trend downward-right")
    print("   - Zero: No clear linear pattern")

    return fig

def demo_corr_dependence(py=False):
    
    # Demonstrate: Uncorrelated but dependent
    x = np.linspace(-2, 2, 200)
    y = x**2 + np.random.randn(200) * 0.2

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(x, y, alpha=0.6, color='purple')
    axes[0].set_xlabel('X', fontsize=12)
    axes[0].set_ylabel('Y', fontsize=12)
    axes[0].set_title('Y = X² (Uncorrelated but Dependent!)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[0].axvline(0, color='gray', linestyle='--', alpha=0.5)

    # Add text
    corr_quad = np.corrcoef(x, y)[0, 1]
    axes[0].text(0.05, 0.95, f'ρ = {corr_quad:.3f}\n(Close to 0!)', 
                transform=axes[0].transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

    # Compare with linear relationship
    x_lin = np.linspace(-2, 2, 200)
    y_lin = 2*x_lin + np.random.randn(200) * 0.5
    axes[1].scatter(x_lin, y_lin, alpha=0.6, color='green')
    axes[1].set_xlabel('X', fontsize=12)
    axes[1].set_ylabel('Y', fontsize=12)
    axes[1].set_title('Y = 2X + noise (Correlated)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    corr_lin = np.corrcoef(x_lin, y_lin)[0, 1]
    axes[1].text(0.05, 0.95, f'ρ = {corr_lin:.3f}\n(High!)', 
                transform=axes[1].transAxes, fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    print("\n⚠️ Warning:")
    print("   Left: Y depends on X (quadratic), but ρ ≈ 0!")
    print("   Right: Y depends on X (linear), and ρ is high.")
    print("\n   → Correlation ONLY detects LINEAR relationships!")

    return fig

def get_xy_plane(py=False):
    c = np.random.rand(20,2)
    x = c[:,0]
    y = c[:,1]
    col = '#EF9A9A'
    
    fig, ax = plt.subplots(1,1, figsize=(5, 5))
    plt.scatter(x, y, color='tab:blue')
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_zorder(10)  # bring to front
    ax.spines['bottom'].set_zorder(10)  # bring to front
    
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    ax.set_xticks([0.85])
    ax.set_yticks([0.7])

    ax.set_xticklabels(['x'], fontsize=12, horizontalalignment='left')
    ax.set_yticklabels(['y'], fontsize=12, verticalalignment='bottom')

    ax.vlines(0.85, ymin=0, ymax=0.7, linestyles="--", color=col)
    ax.hlines(xmin=0, xmax=0.85, y=0.7, linestyles="--", color=col)

    a = np.linspace(0,0.85)
    b = np.linspace(0,0.7)
    plt.fill_between(a, 0.7, 0, color = col, alpha=0.25)

    ax.annotate("(x,y)",
                xy=(0.85, 0.7), 
                xytext=(0.86, 0.71), fontsize=12)


    ax.annotate(r"$(x_i,y_j)$",
                xy=(0.26, 0.97), 
                xytext=(0.2, 1.01), fontsize=12)

    ax.annotate("", xy=(0.65, 0.2), xytext=(1, 0.3), arrowprops=dict(arrowstyle="->", color='black'))
    ax.annotate(r"$X\leq x \cap Y\leq y$",
                xy=(1, 0.3), 
                xytext=(1, 0.3), fontsize=12, color='black')
    
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def get_all_pdf_cont(show_marginal=True, show_cond=True, py=False):
    mu = np.array([0., 1.])
    sigma = np.array([[ 1. , -0.5], [-0.5,  1.5]])
    F = multivariate_normal(mu, sigma)
    
    N = 60
    X = np.linspace(-3, 3, N)
    Y = np.linspace(-3, 4, N)
    X, Y = np.meshgrid(X, Y)
    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    Z = F.pdf(pos)
    
    fig = plt.figure(figsize=(13, 7))
    ax = plt.axes(projection='3d')

    #verts = [polygon_under_graph(X[0], Z[42])]
    #facecolors = cm.get_cmap('viridis_r')(np.linspace(0, 1, len(verts)))

    #poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
    #ax.add_collection3d(poly, zs=2, zdir='y')

    if show_marginal:
        ax.plot3D(np.linspace(3,3,len(Y[0])), Y[:,0], Z[:,30], color='tab:orange', label=r"$f_Y$", zorder=3, linewidth=3.0)

    w = ax.plot_wireframe(pos[:, :, 0], pos[:, :, 1], Z, zorder=1, label=r"$f_{XY}$", color='tab:blue')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('PDF')
    ax.view_init(55, 20)


    if show_cond:
        ax.plot3D(X[0], np.linspace(2,2,len(X[0])), Z[42], color='red', label=r"$f_{X|Y=y_0}$", zorder=2, linewidth=3.0)

    if show_marginal:
        ax.plot3D(X[0], np.linspace(4,4,len(X[0])), Z[30], color='tab:green', label=r"$f_X$", zorder=3, linewidth=3.0)


    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 4)
    ax.set_zlim(0, 0.15)

    plt.legend()

    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def demo_correlation(py=False):
    # Generate data with different correlations
    np.random.seed(42)
    n_samples = 100
    x = np.random.randn(n_samples)

    # Create different correlation strengths
    correlations = [0.95, 0.7, 0.3, 0.0, -0.3, -0.7, -0.95]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, target_corr in enumerate(correlations):
        # Generate y with specific correlation to x
        # Using Cholesky decomposition
        if target_corr == 0:
            y = np.random.randn(n_samples)
        else:
            y = target_corr * x + np.sqrt(1 - target_corr**2) * np.random.randn(n_samples)
        
        actual_corr = np.corrcoef(x, y)[0, 1]
        
        axes[idx].scatter(x, y, alpha=0.6, s=30)
        axes[idx].set_title(f'ρ = {actual_corr:.2f}', fontsize=13, fontweight='bold')
        axes[idx].set_xlabel('X')
        axes[idx].set_ylabel('Y')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(-3, 3)
        axes[idx].set_ylim(-3, 3)
        
        # Add interpretation
        if actual_corr > 0.7:
            strength = "Strong +"
            color = 'darkgreen'
        elif actual_corr > 0.3:
            strength = "Moderate +"
            color = 'green'
        elif actual_corr > -0.3:
            strength = "Weak/None"
            color = 'gray'
        elif actual_corr > -0.7:
            strength = "Moderate -"
            color = 'orange'
        else:
            strength = "Strong -"
            color = 'darkred'
        
        axes[idx].text(0.05, 0.95, strength, transform=axes[idx].transAxes,
                    fontsize=11, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))

    # Hide last plot
    axes[-1].axis('off')

    plt.suptitle('Correlation Coefficient: Visual Guide', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    print("\nInterpretation Guide:")
    print("  |ρ| = 1.0   → Perfect linear relationship")
    print("  |ρ| = 0.7-1.0 → Strong relationship")
    print("  |ρ| = 0.3-0.7 → Moderate relationship")
    print("  |ρ| = 0.0-0.3 → Weak/no relationship")

    return fig

def simulate_ad_carousel(n_ads=10, p_like=0.5, n_simulations=100000, circular=True):
    """
    Simulate the ad carousel bonus problem to verify our theoretical result.
    
    Parameters:
    -----------
    n_ads : int
        Number of ads in carousel
    p_like : float
        Probability of liking an ad
    n_simulations : int
        Number of simulations to run
    circular : bool
        If True, carousel wraps around (ad 1 and ad n are neighbors)
        If False, linear arrangement (only ads 2 through n-1 can earn bonuses)
        
    Returns:
    --------
    dict : Contains average bonuses and distribution
    """
    bonus_counts = []
    
    for _ in range(n_simulations):
        # Generate random likes/skips (1 = like, 0 = skip)
        reactions = np.random.binomial(1, p_like, n_ads)
        
        # Count bonuses
        bonuses = 0
        
        if circular:
            # ALL ads can earn bonuses in circular arrangement
            for i in range(n_ads):
                # Use modulo for circular indexing
                left_neighbor = reactions[(i - 1) % n_ads]
                right_neighbor = reactions[(i + 1) % n_ads]
                
                # Bonus if ad i differs from both neighbors
                if reactions[i] != left_neighbor and reactions[i] != right_neighbor:
                    bonuses += 1
        else:
            # Linear arrangement: only ads 2 through n-1 can earn bonuses
            for i in range(1, n_ads - 1):
                # Bonus if ad i differs from both neighbors
                if reactions[i] != reactions[i-1] and reactions[i] != reactions[i+1]:
                    bonuses += 1
        
        bonus_counts.append(bonuses)
    
    # Calculate statistics
    avg_bonuses = np.mean(bonus_counts)
    std_bonuses = np.std(bonus_counts)
    
    # Calculate theoretical expectation
    if circular:
        theoretical = n_ads * p_like * (1 - p_like) 
        arrangement = "Circular"
    else:
        theoretical = (n_ads - 2) * p_like * (1 - p_like) 
        arrangement = "Linear"
    
    # Plot distribution
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    unique, counts = np.unique(bonus_counts, return_counts=True)
    plt.bar(unique, counts / n_simulations, alpha=0.7, color='steelblue', edgecolor='black')
    plt.axvline(avg_bonuses, color='red', linestyle='--', linewidth=2, 
                label=f'Simulated = {avg_bonuses:.3f}')
    plt.axvline(theoretical, color='green', linestyle='--', linewidth=2, 
                label=f'Theoretical = {theoretical:.1f}')
    plt.xlabel('Number of Bonuses', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title(f'{arrangement} Carousel: Distribution of Bonuses\n({n_simulations:,} simulations)', fontsize=14)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Example sequences
    plt.subplot(1, 2, 2)
    plt.axis('off')
    
    # Show a few example sequences
    examples_text = "Sample Sequences:\n\n"
    for i in range(5):
        reactions = np.random.binomial(1, p_like, n_ads)
        reaction_symbols = ['L' if r == 1 else 'S' for r in reactions]
        
        bonuses_list = []
        for j in range(1, n_ads - 1):
            if reactions[j] != reactions[j-1] and reactions[j] != reactions[j+1]:
                bonuses_list.append(j)
        
        examples_text += f"{'  '.join(reaction_symbols)}\n"
        examples_text += f"Bonuses at positions: {bonuses_list if bonuses_list else 'None'}\n\n"
    
    plt.text(0.1, 0.5, examples_text, fontsize=10, family='monospace',
             verticalalignment='center')
    
    plt.tight_layout()
    plt.close() # to prevent automatic display
    
    return {
        'average': avg_bonuses,
        'std': std_bonuses,
        'distribution': (unique, counts / n_simulations)
    }

def interactive_carousel_demo():
    """Interactive widget to explore carousel with different parameters."""
    
    def update(n_ads, p_like, circular):
        result = simulate_ad_carousel(n_ads, p_like, n_simulations=50000, circular=circular)
        if circular:
            theoretical = n_ads * p_like * (1 - p_like) 
            arrangement = "Circular"
        else:
            theoretical = (n_ads - 2) * p_like * (1 - p_like) 
            arrangement = "Linear"
        
        print(f"\n{arrangement} Carousel Results:")
        print(f"Simulated average: {result['average']:.3f}")
        print(f"Theoretical expectation: {theoretical:.3f}")
        print(f"Difference: {abs(result['average'] - theoretical):.3f}")
        
        if circular and n_ads > 2:
            linear_theoretical = (n_ads - 2) * p_like * (1 - p_like) 
            improvement = ((theoretical - linear_theoretical) / linear_theoretical) * 100
            print(f"\nCircular vs Linear: +{improvement:.1f}% more bonuses expected!")
    
    interact(update,
             n_ads=IntSlider(min=5, max=20, step=1, value=10, description='# Ads'),
             p_like=FloatSlider(min=0.1, max=0.9, step=0.1, value=0.5, description='P(Like)'),
             circular=True)

def visualize_random_vector_3d(n_samples=500, py=False):
    """
    Visualize a 3D random vector.
    """
    # Generate 3D multivariate normal
    mean = [0, 0, 0]
    cov = [[1, 0.5, 0.3],
           [0.5, 1, 0.4],
           [0.3, 0.4, 1]]
    
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    
    fig = plt.figure(figsize=(12, 5))
    
    # 3D scatter
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(samples[:, 0], samples[:, 1], samples[:, 2], 
                alpha=0.6, s=20, c=samples[:, 2], cmap='viridis')
    ax1.set_xlabel('X₁', fontsize=12)
    ax1.set_ylabel('X₂', fontsize=12)
    ax1.set_zlabel('X₃', fontsize=12)
    ax1.set_title('3D Random Vector', fontsize=14)
    
    # Covariance matrix heatmap
    ax2 = fig.add_subplot(122)
    empirical_cov = np.cov(samples.T)
    im = ax2.imshow(empirical_cov, cmap='RdBu_r', vmin=-1, vmax=1)
    ax2.set_xticks([0, 1, 2])
    ax2.set_yticks([0, 1, 2])
    ax2.set_xticklabels(['X₁', 'X₂', 'X₃'])
    ax2.set_yticklabels(['X₁', 'X₂', 'X₃'])
    ax2.set_title('Empirical Covariance Matrix', fontsize=14)
    
    # Add values to cells
    for i in range(3):
        for j in range(3):
            text = ax2.text(j, i, f'{empirical_cov[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=12)
    
    plt.colorbar(im, ax=ax2)
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def visualize_joint_pdf_3d(py=False):
    """
    Visualize the joint PDF from the lesson example.
    f(x,y,z) = (1/3)(3x + 2y + z) for x,y,z in [0,1]
    """
    # Create grid
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 30)
    X, Y = np.meshgrid(x, y)
    
    # For visualization, fix z at different values
    fig = plt.figure(figsize=(15, 4))
    
    z_values = [0.25, 0.5, 0.75]
    
    for idx, z_val in enumerate(z_values):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Calculate PDF values
        Z = (1/3) * (3*X + 2*Y + z_val)
        
        # Surface plot
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, 
                               edgecolor='none')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
        ax.set_zlabel('f(x,y,z)', fontsize=10)
        ax.set_title(f'Joint PDF slice at z={z_val}', fontsize=12)
        fig.colorbar(surf, ax=ax, shrink=0.5)
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

def demonstrate_pca(py=False):
    """
    Demonstrate PCA as application of covariance matrix.
    """
    # Generate correlated data
    n_samples = 500
    mean = [0, 0]
    cov = [[2, 1.5], [1.5, 1]]
    samples = np.random.multivariate_normal(mean, cov, n_samples)
    
    # Compute covariance matrix
    cov_empirical = np.cov(samples.T)
    
    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_empirical)
    
    # Sort by eigenvalues
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Original data with principal components
    axes[0].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=20, label='Data')
    
    # Draw principal components
    for i in range(2):
        direction = eigenvectors[:, i] * np.sqrt(eigenvalues[i]) * 3
        axes[0].arrow(0, 0, direction[0], direction[1], 
                     head_width=0.2, head_length=0.3, fc=f'C{i+1}', ec=f'C{i+1}',
                     linewidth=3, label=f'PC{i+1} (λ={eigenvalues[i]:.2f})')
    
    axes[0].set_xlabel('X₁', fontsize=12)
    axes[0].set_ylabel('X₂', fontsize=12)
    axes[0].set_title('Principal Component Analysis', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color='k', linewidth=0.5)
    axes[0].axvline(0, color='k', linewidth=0.5)
    axes[0].axis('equal')
    
    # Project onto principal components
    projected = samples @ eigenvectors
    axes[1].scatter(projected[:, 0], projected[:, 1], alpha=0.5, s=20)
    axes[1].set_xlabel('PC1', fontsize=12)
    axes[1].set_ylabel('PC2', fontsize=12)
    axes[1].set_title('Data in Principal Component Space', fontsize=14)
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color='k', linewidth=0.5)
    axes[1].axvline(0, color='k', linewidth=0.5)
    axes[1].axis('equal')
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    print(f"\nVariance explained by PC1: {eigenvalues[0] / eigenvalues.sum() * 100:.1f}%")
    print(f"Variance explained by PC2: {eigenvalues[1] / eigenvalues.sum() * 100:.1f}%")


if __name__ == "__main__":
    #generate_movie_data_normal()
    #get_viz_movie_data(py=True)
    #viz_joint_distr_cont(py=True)
    #covariance_demo(py=True)
    #demo_corr_dependence(py=True)
    #get_xy_plane(py=True)
    #get_all_pdf_cont(py=True)
    #get_all_pdf_cont(show_marginal=True, show_cond=False, py=True)
    #get_all_pdf_cont(show_marginal=False, show_cond=True, py=True)
    #get_all_pdf_cont(show_marginal=False, show_cond=False, py=True)
    #demo_correlation(py=True)
    #visualize_random_vector_3d(n_samples=500, py=True)
    #visualize_joint_pdf_3d(py=True)
    demonstrate_pca(py=True)