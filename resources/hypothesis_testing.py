import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# plt.rcParams["font.family"] = ["DejaVu Sans", "Segoe UI Emoji"]

# import matplotlib
# print(matplotlib.rcsetup.all_backends)
# matplotlib.use('TkAgg')


def visualize_hypothesis_types(test_type="two-tailed", mu0=0, alpha=0.05):
    """
    Interactive visualization of hypothesis test types
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x = np.linspace(-4, 4, 1000)
    y = stats.norm.pdf(x, 0, 1)

    titles = ["Two-Tailed Test", "Right-Tailed Test", "Left-Tailed Test"]
    test_types = ["two-tailed", "right-tailed", "left-tailed"]

    for idx, (ax, title, t_type) in enumerate(zip(axes, titles, test_types)):
        ax.plot(x, y, "b-", linewidth=2, label="Sampling Distribution under H₀")
        ax.axvline(
            mu0, color="green", linestyle="--", linewidth=2, label=f"H₀: μ = {mu0}"
        )

        # Shade rejection regions
        if t_type == "two-tailed":
            critical_left = stats.norm.ppf(alpha / 2)
            critical_right = stats.norm.ppf(1 - alpha / 2)
            ax.fill_between(
                x[x <= critical_left],
                y[x <= critical_left],
                alpha=0.3,
                color="red",
                label="Rejection Region",
            )
            ax.fill_between(
                x[x >= critical_right], y[x >= critical_right], alpha=0.3, color="red"
            )
            ax.text(
                0,
                0.2,
                f"H₁: μ ≠ {mu0}",
                ha="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat"),
            )

        elif t_type == "right-tailed":
            critical_right = stats.norm.ppf(1 - alpha)
            ax.fill_between(
                x[x >= critical_right],
                y[x >= critical_right],
                alpha=0.3,
                color="red",
                label="Rejection Region",
            )
            ax.text(
                2,
                0.2,
                f"H₁: μ > {mu0}",
                ha="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat"),
            )

        else:  # left-tailed
            critical_left = stats.norm.ppf(alpha)
            ax.fill_between(
                x[x <= critical_left],
                y[x <= critical_left],
                alpha=0.3,
                color="red",
                label="Rejection Region",
            )
            ax.text(
                -2,
                0.2,
                f"H₁: μ < {mu0}",
                ha="center",
                fontsize=12,
                bbox=dict(boxstyle="round", facecolor="wheat"),
            )

        # Highlight current selection
        if t_type == test_type:
            ax.set_facecolor("#fffacd")
            for spine in ax.spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Test Statistic")
        ax.set_ylabel("Probability Density")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def create_error_matrix(py=False):
    """Create a visual confusion matrix for hypothesis testing errors"""

    # Create the matrix
    data = {
        "H₀ is True\n(Reality)": [
            "✓ Correct Decision\n(1-α)\nTrue Negative",
            "✗ Type I Error\n(α)\nFalse Positive",
        ],
        "H₀ is False\n(Reality)": [
            "✗ Type II Error\n(β)\nFalse Negative",
            "✓ Correct Decision\n(1-β) = Power\nTrue Positive",
        ],
    }

    df = pd.DataFrame(
        data, index=["Fail to Reject H₀\n(Decision)", "Reject H₀\n(Decision)"]
    )

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    # Color map: green for correct, red for errors
    colors = [["#90EE90", "#FFB6C6"], ["#FFB6C6", "#90EE90"]]

    # Create table
    for i in range(2):
        for j in range(2):
            color = colors[i][j]
            text = df.iloc[i, j]
            ax.add_patch(
                plt.Rectangle(
                    (j, 1 - i), 1, 1, facecolor=color, edgecolor="black", linewidth=2
                )
            )
            ax.text(
                j + 0.5,
                1 - i + 0.5,
                text,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )

    # Add labels
    ax.text(
        -0.2, 1, "Decision\n↓", ha="center", va="center", fontsize=12, fontweight="bold"
    )
    ax.text(
        1, 2.15, "Reality →", ha="center", va="center", fontsize=12, fontweight="bold"
    )

    ax.text(
        0.5,
        2.15,
        df.columns[0],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        1.5,
        2.15,
        df.columns[1],
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
    )

    ax.text(
        -0.2, 1.5, df.index[0], ha="right", va="center", fontsize=11, fontweight="bold"
    )
    ax.text(
        -0.2, 0.5, df.index[1], ha="right", va="center", fontsize=11, fontweight="bold"
    )

    ax.set_xlim(-0.5, 2)
    ax.set_ylim(0, 2.3)
    ax.axis("off")

    plt.title(
        "The Confusion Matrix of Hypothesis Testing",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()

    return fig


def simulate_hypothesis_test_errors(
    true_mean=0, test_mean=0, n_samples=30, n_simulations=1000, alpha=0.05
):
    """
    Simulate hypothesis testing to demonstrate Type I and Type II errors

    Parameters:
    - true_mean: The TRUE population mean
    - test_mean: The mean we're testing against (H₀: μ = test_mean)
    - n_samples: Sample size for each test
    - n_simulations: Number of tests to run
    - alpha: Significance level
    """

    type_1_errors = 0  # Reject H₀ when it's true
    type_2_errors = 0  # Fail to reject H₀ when it's false
    correct_rejections = 0  # Correctly reject false H₀
    correct_non_rejections = 0  # Correctly fail to reject true H₀

    p_values = []

    for _ in range(n_simulations):
        # Generate sample from TRUE distribution
        sample = np.random.normal(true_mean, 1, n_samples)

        # Perform t-test against test_mean (our H₀)
        t_stat, p_value = stats.ttest_1samp(sample, test_mean)
        p_values.append(p_value)

        # Decision: reject if p-value < alpha
        reject_h0 = p_value < alpha

        # Classify the outcome
        if true_mean == test_mean:  # H₀ is actually TRUE
            if reject_h0:
                type_1_errors += 1
            else:
                correct_non_rejections += 1
        else:  # H₀ is actually FALSE
            if reject_h0:
                correct_rejections += 1
            else:
                type_2_errors += 1

    # Calculate rates
    if true_mean == test_mean:
        type_1_rate = type_1_errors / n_simulations
        type_2_rate = None
        power = None
    else:
        type_1_rate = None
        type_2_rate = type_2_errors / n_simulations
        power = correct_rejections / n_simulations

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: P-value distribution
    axes[0].hist(
        p_values, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black"
    )
    axes[0].axvline(
        alpha, color="red", linestyle="--", linewidth=2, label=f"α = {alpha}"
    )
    axes[0].set_xlabel("P-value", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title(
        f"Distribution of P-values\n(True μ={true_mean}, Testing μ={test_mean})",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Error rates
    if true_mean == test_mean:
        labels = ["Correct\nDecisions", "Type I\nErrors"]
        values = [correct_non_rejections, type_1_errors]
        colors = ["#90EE90", "#FFB6C6"]
        title = f"Type I Error Rate: {type_1_rate:.3f}\n(Should be ≈ {alpha})"
    else:
        labels = ["Correct\nRejections\n(Power)", "Type II\nErrors"]
        values = [correct_rejections, type_2_errors]
        colors = ["#90EE90", "#FFB6C6"]
        title = f"Power: {power:.3f}, Type II Error Rate (β): {type_2_rate:.3f}"

    axes[1].bar(labels, values, color=colors, edgecolor="black", linewidth=2)
    axes[1].set_ylabel("Number of Tests", fontsize=12)
    axes[1].set_title(title, fontsize=14, fontweight="bold")
    axes[1].grid(True, alpha=0.3, axis="y")

    for i, v in enumerate(values):
        axes[1].text(i, v + 20, str(v), ha="center", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.show()

    # Print summary
    print("=" * 60)
    print(f"SIMULATION RESULTS ({n_simulations} tests)")
    print("=" * 60)
    print(f"True population mean: {true_mean}")
    print(f"Testing H₀: μ = {test_mean}")
    print(f"Sample size: {n_samples}")
    print(f"Significance level α: {alpha}")
    print("-" * 60)

    if true_mean == test_mean:
        print("H₀ IS TRUE (testing against the correct value)")
        print(f"Type I Error Rate (False Positive): {type_1_rate:.4f}")
        print(f"Expected Type I Error Rate: {alpha}")
        print(f"✓ Correct Non-Rejections: {correct_non_rejections}")
        print(f"✗ Type I Errors: {type_1_errors}")
    else:
        print("H₀ IS FALSE (testing against the wrong value)")
        print(f"Power (1-β): {power:.4f}")
        print(f"Type II Error Rate β: {type_2_rate:.4f}")
        print(f"✓ Correct Rejections: {correct_rejections}")
        print(f"✗ Type II Errors: {type_2_errors}")
    print("=" * 60)


def interactive_pvalue_visualization(
    test_statistic=2.5, test_type="two-tailed", distribution="z"
):
    """
    Interactive visualization showing what a p-value represents
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    x = np.linspace(-4, 4, 1000)

    # Choose distribution
    if distribution == "z":
        y = stats.norm.pdf(x)
        dist_obj = stats.norm
        dist_name = "Standard Normal (z)"
    else:  # t-distribution
        df = 20
        y = stats.t.pdf(x, df)
        dist_obj = stats.t(df)
        dist_name = f"t-distribution (df={df})"

    # Left plot: Show the distribution and test statistic
    axes[0].plot(x, y, "b-", linewidth=2, label=dist_name)
    axes[0].axvline(0, color="green", linestyle="--", linewidth=2, label="H₀: μ = 0")
    axes[0].axvline(
        test_statistic,
        color="red",
        linestyle="-",
        linewidth=3,
        label=f"Observed test statistic = {test_statistic}",
    )

    # Calculate and shade p-value region
    if test_type == "right-tailed":
        if distribution == "z":
            p_value = 1 - stats.norm.cdf(test_statistic)
        else:
            p_value = 1 - dist_obj.cdf(test_statistic)
        axes[0].fill_between(
            x[x >= test_statistic],
            y[x >= test_statistic],
            alpha=0.4,
            color="red",
            label=f"p-value = {p_value:.4f}",
        )

    elif test_type == "left-tailed":
        if distribution == "z":
            p_value = stats.norm.cdf(test_statistic)
        else:
            p_value = dist_obj.cdf(test_statistic)
        axes[0].fill_between(
            x[x <= test_statistic],
            y[x <= test_statistic],
            alpha=0.4,
            color="red",
            label=f"p-value = {p_value:.4f}",
        )

    else:  # two-tailed
        if distribution == "z":
            p_value = 2 * (1 - stats.norm.cdf(abs(test_statistic)))
        else:
            p_value = 2 * (1 - dist_obj.cdf(abs(test_statistic)))
        axes[0].fill_between(
            x[x >= abs(test_statistic)],
            y[x >= abs(test_statistic)],
            alpha=0.4,
            color="red",
        )
        axes[0].fill_between(
            x[x <= -abs(test_statistic)],
            y[x <= -abs(test_statistic)],
            alpha=0.4,
            color="red",
        )
        axes[0].axvline(-abs(test_statistic), color="red", linestyle="-", linewidth=3)
        axes[0].text(
            0,
            max(y) * 0.5,
            f"p-value = {p_value:.4f}",
            ha="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.8),
        )

    axes[0].set_xlabel("Test Statistic", fontsize=12)
    axes[0].set_ylabel("Probability Density", fontsize=12)
    axes[0].set_title(
        f"P-Value Visualization\n({test_type.title()} Test)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10, loc="upper left")
    axes[0].grid(True, alpha=0.3)

    # Right plot: Decision rule with multiple alpha levels
    alphas = [0.01, 0.05, 0.10]
    decisions = [
        "Reject H₀" if p_value < alpha else "Fail to Reject H₀" for alpha in alphas
    ]
    colors_decision = ["#90EE90" if "Reject" in d else "#FFB6C6" for d in decisions]

    y_pos = np.arange(len(alphas))
    axes[1].barh(
        y_pos, [1] * len(alphas), color=colors_decision, edgecolor="black", linewidth=2
    )

    for i, (alpha, decision) in enumerate(zip(alphas, decisions)):
        axes[1].text(
            0.5,
            i,
            f"α = {alpha}\n{decision}",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"α = {alpha}" for alpha in alphas])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Decision", fontsize=12)
    axes[1].set_title(
        f"Decision at Different Significance Levels\n(p-value = {p_value:.4f})",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xticks([])

    plt.tight_layout()
    plt.show()

    # Print interpretation
    print("=" * 70)
    print("P-VALUE INTERPRETATION")
    print("=" * 70)
    print(f"Test Statistic: {test_statistic}")
    print(f"Test Type: {test_type}")
    print(f"Distribution: {dist_name}")
    print(f"P-value: {p_value:.6f}")
    print("-" * 70)
    print("\n📊 What this means:")
    print("If H₀ were true, the probability of observing a test statistic")
    print(
        f"as extreme as {test_statistic} (or more extreme) is {p_value:.4f} ({p_value * 100:.2f}%)."
    )
    print()

    if p_value < 0.01:
        print("🔴 VERY STRONG evidence against H₀ (p < 0.01)")
        print("   This would be extremely surprising if H₀ were true.")
    elif p_value < 0.05:
        print("🟠 STRONG evidence against H₀ (0.01 ≤ p < 0.05)")
        print("   This would be quite surprising if H₀ were true.")
    elif p_value < 0.10:
        print("🟡 MODERATE evidence against H₀ (0.05 ≤ p < 0.10)")
        print("   This would be somewhat surprising if H₀ were true.")
    else:
        print("🟢 WEAK/NO evidence against H₀ (p ≥ 0.10)")
        print("   This would not be surprising at all if H₀ were true.")
    print("=" * 70)


def simulate_pvalue_distribution(
    h0_true=True, effect_size=0, n_simulations=10000, sample_size=50
):
    """
    Simulate many hypothesis tests and show distribution of p-values
    """
    p_values = []

    for _ in range(n_simulations):
        if h0_true:
            # Generate data under H₀ (mean = 0)
            sample = np.random.normal(0, 1, sample_size)
        else:
            # Generate data under H₁ (mean = effect_size)
            sample = np.random.normal(effect_size, 1, sample_size)

        # Perform t-test against H₀: μ = 0
        t_stat, p_value = stats.ttest_1samp(sample, 0)
        p_values.append(p_value)

    p_values = np.array(p_values)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Histogram of p-values
    axes[0].hist(
        p_values, bins=50, density=True, alpha=0.7, color="skyblue", edgecolor="black"
    )

    if h0_true:
        # Overlay uniform distribution
        axes[0].axhline(
            1, color="red", linestyle="--", linewidth=2, label="Expected: Uniform(0,1)"
        )
        title_suffix = "(H₀ is TRUE)"
    else:
        title_suffix = f"(H₀ is FALSE, true effect = {effect_size})"

    axes[0].axvline(0.05, color="orange", linestyle="--", linewidth=2, label="α = 0.05")
    axes[0].set_xlabel("P-value", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].set_title(
        f"Distribution of P-values {title_suffix}\n({n_simulations:,} simulations)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(0, 1)

    # Cumulative distribution
    sorted_p = np.sort(p_values)
    cumulative = np.arange(1, len(sorted_p) + 1) / len(sorted_p)

    axes[1].plot(sorted_p, cumulative, linewidth=2, label="Observed CDF")

    if h0_true:
        axes[1].plot([0, 1], [0, 1], "r--", linewidth=2, label="Expected: Uniform(0,1)")

    axes[1].axvline(0.05, color="orange", linestyle="--", linewidth=2, alpha=0.7)
    axes[1].axhline(
        0.05, color="orange", linestyle="--", linewidth=2, alpha=0.7, label="α = 0.05"
    )

    # Mark the point
    proportion_significant = np.mean(p_values < 0.05)
    axes[1].plot(
        0.05,
        proportion_significant,
        "ro",
        markersize=12,
        label=f"Proportion p<0.05: {proportion_significant:.3f}",
    )

    axes[1].set_xlabel("P-value", fontsize=12)
    axes[1].set_ylabel("Cumulative Probability", fontsize=12)
    axes[1].set_title(
        f"Cumulative Distribution of P-values {title_suffix}",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

    # Statistics
    print("=" * 70)
    print("SIMULATION RESULTS")
    print("=" * 70)
    print(f"Number of simulations: {n_simulations:,}")
    print(f"Sample size per test: {sample_size}")
    if h0_true:
        print("Truth: H₀ is TRUE (population mean = 0)")
    else:
        print(f"Truth: H₀ is FALSE (population mean = {effect_size})")
    print("-" * 70)
    print(
        f"Proportion with p < 0.01: {np.mean(p_values < 0.01):.4f} (expected: 0.0100)"
    )
    print(
        f"Proportion with p < 0.05: {np.mean(p_values < 0.05):.4f} (expected: 0.0500)"
    )
    print(
        f"Proportion with p < 0.10: {np.mean(p_values < 0.10):.4f} (expected: 0.1000)"
    )
    print("-" * 70)

    if h0_true:
        print("\n✓ When H₀ is true, p-values are uniformly distributed")
        print("  This confirms that α directly controls Type I error rate.")
    else:
        power = np.mean(p_values < 0.05)
        print("\n✓ When H₀ is false, p-values concentrate near 0")
        print(f"  Power (proportion rejecting H₀ at α=0.05): {power:.4f}")
    print("=" * 70)


def visualize_test_statistic_concept(
    observed_mean=1.5, h0_mean=0, sample_std=1, sample_size=30
):
    """
    Visualize the concept of a test statistic as 'distance from expected'
    """
    # Calculate standard error and test statistic
    standard_error = sample_std / np.sqrt(sample_size)
    test_statistic = (observed_mean - h0_mean) / standard_error

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Sample distribution
    x = np.linspace(h0_mean - 4 * standard_error, h0_mean + 4 * standard_error, 1000)
    y = stats.norm.pdf(x, h0_mean, standard_error)

    axes[0].plot(
        x,
        y,
        "b-",
        linewidth=2,
        label=f"Sampling Distribution under H₀\n(μ₀={h0_mean}, SE={standard_error:.3f})",
    )
    axes[0].axvline(
        h0_mean,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Expected under H₀ (μ₀={h0_mean})",
    )
    axes[0].axvline(
        observed_mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Observed Sample Mean ({observed_mean})",
    )

    # Shade the distance
    axes[0].fill_between(
        [h0_mean, observed_mean], 0, max(y) * 1.1, alpha=0.3, color="orange"
    )
    axes[0].annotate(
        "",
        xy=(observed_mean, max(y) * 0.7),
        xytext=(h0_mean, max(y) * 0.7),
        arrowprops=dict(arrowstyle="<->", color="orange", lw=3),
    )
    axes[0].text(
        (h0_mean + observed_mean) / 2,
        max(y) * 0.75,
        f"Distance = {abs(observed_mean - h0_mean):.3f}",
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    axes[0].set_xlabel("Sample Mean", fontsize=12)
    axes[0].set_ylabel("Probability Density", fontsize=12)
    axes[0].set_title(
        "Raw Difference: Observed vs Expected", fontsize=14, fontweight="bold"
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Standardized (test statistic)
    z = np.linspace(-4, 4, 1000)
    y_z = stats.norm.pdf(z, 0, 1)

    axes[1].plot(z, y_z, "b-", linewidth=2, label="Standard Normal Distribution")
    axes[1].axvline(
        0, color="green", linestyle="--", linewidth=2, label="Expected (z=0)"
    )
    axes[1].axvline(
        test_statistic,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Test Statistic (z={test_statistic:.3f})",
    )

    # Shade the distance in standard errors
    for i in range(1, int(abs(test_statistic)) + 1):
        if test_statistic > 0:
            axes[1].axvspan(i - 1, i, alpha=0.2, color="orange")
        else:
            axes[1].axvspan(-i, -i + 1, alpha=0.2, color="orange")

    axes[1].text(
        test_statistic / 2,
        max(y_z) * 0.8,
        f"{abs(test_statistic):.2f} standard errors\nfrom expected",
        ha="center",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )

    axes[1].set_xlabel("Standard Errors from Mean", fontsize=12)
    axes[1].set_ylabel("Probability Density", fontsize=12)
    axes[1].set_title(
        "Standardized: Test Statistic (z-score)", fontsize=14, fontweight="bold"
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Summary
    print("=" * 60)
    print("TEST STATISTIC CALCULATION")
    print("=" * 60)
    print(f"H₀: μ = {h0_mean}")
    print(f"Observed sample mean: {observed_mean}")
    print(f"Sample standard deviation: {sample_std}")
    print(f"Sample size: {sample_size}")
    print(f"Standard Error (SE): {standard_error:.4f}")
    print("-" * 60)
    print("Test Statistic = (Observed - Expected) / SE")
    print(f"              = ({observed_mean} - {h0_mean}) / {standard_error:.4f}")
    print(f"              = {test_statistic:.4f}")
    print("=" * 60)
    print(
        f"\n💡 Interpretation: The observed mean is {abs(test_statistic):.2f} standard"
    )
    print("   errors away from what we'd expect under H₀.")
    if abs(test_statistic) < 2:
        print("   This is relatively CLOSE - not very surprising.")
    elif abs(test_statistic) < 3:
        print("   This is getting FAR - somewhat surprising!")
    else:
        print("   This is VERY FAR - highly surprising!")


def compare_z_t_distributions():
    """
    Compare z-distribution and t-distributions with different degrees of freedom
    """
    x = np.linspace(-4, 4, 1000)

    plt.figure(figsize=(14, 7))

    # Standard normal (z)
    plt.plot(
        x, stats.norm.pdf(x), "b-", linewidth=3, label="z (Standard Normal)", alpha=0.8
    )

    # t-distributions
    dfs = [1, 3, 5, 10, 30]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(dfs)))

    for df, color in zip(dfs, colors):
        plt.plot(
            x,
            stats.t.pdf(x, df),
            linewidth=2,
            label=f"t (df={df})",
            color=color,
            alpha=0.7,
        )

    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.title(
        "Comparison of z and t Distributions\n(t-distribution has heavier tails for small df)",
        fontsize=14,
        fontweight="bold",
    )
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(-4, 4)

    # Annotate
    plt.annotate(
        "Heavier tails\n(more uncertainty)",
        xy=(-2.5, 0.05),
        xytext=(-3, 0.15),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
        fontsize=11,
        fontweight="bold",
        color="red",
    )

    plt.tight_layout()
    plt.show()

    print("=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)
    print("1. As df increases, t-distribution approaches z-distribution")
    print("2. At df=30, they're nearly identical (this is why we use n≥30 rule)")
    print("3. Small df → heavier tails → need larger test statistic to reject H₀")
    print("4. This accounts for uncertainty in estimating σ from small samples")
    print("=" * 60)


def demonstrate_pairing_effect(correlation=0.8, true_difference=0.5, n=20):
    """
    Show why paired tests are more powerful when data is correlated
    """
    np.random.seed(42)

    # Generate correlated paired data
    # Subject-specific baseline
    baseline = np.random.normal(5, 2, n)

    # Group 1 (e.g., Model A on each subject)
    group1 = baseline + np.random.normal(0, 0.5, n)

    # Group 2 (e.g., Model B on same subjects) - with true difference
    group2 = baseline + true_difference + np.random.normal(0, 0.5, n)

    # For independent analysis, we'd shuffle group2 to break correlation
    group2_shuffled = np.random.permutation(group2)

    # Calculate differences
    paired_differences = group2 - group1

    # Perform tests
    # Paired t-test
    t_paired, p_paired = stats.ttest_rel(group2, group1)

    # Independent t-test (as if different subjects)
    t_indep, p_indep = stats.ttest_ind(group2_shuffled, group1)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Paired data - connected points
    axes[0, 0].plot(
        [1, 2], [group1, group2], "o-", color="gray", alpha=0.3, linewidth=1
    )
    for i in range(n):
        if group2[i] > group1[i]:
            axes[0, 0].plot(
                [1, 2],
                [group1[i], group2[i]],
                "o-",
                color="green",
                alpha=0.6,
                linewidth=1.5,
            )
        else:
            axes[0, 0].plot(
                [1, 2],
                [group1[i], group2[i]],
                "o-",
                color="red",
                alpha=0.6,
                linewidth=1.5,
            )

    axes[0, 0].boxplot([group1, group2], positions=[1, 2], widths=0.3)
    axes[0, 0].set_xticks([1, 2])
    axes[0, 0].set_xticklabels(["Model A", "Model B"])
    axes[0, 0].set_ylabel("Performance", fontsize=12)
    axes[0, 0].set_title(
        "PAIRED Samples\n(Same subjects tested twice)", fontsize=14, fontweight="bold"
    )
    axes[0, 0].text(
        1.5,
        max(group1.max(), group2.max()) + 0.5,
        f"Mean difference: {paired_differences.mean():.3f}\np-value: {p_paired:.4f}",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Plot 2: Independent data - separate groups
    axes[0, 1].scatter(
        np.ones(n) + np.random.normal(0, 0.03, n),
        group1,
        alpha=0.6,
        s=100,
        label="Model A",
    )
    axes[0, 1].scatter(
        2 * np.ones(n) + np.random.normal(0, 0.03, n),
        group2_shuffled,
        alpha=0.6,
        s=100,
        label="Model B",
    )
    axes[0, 1].boxplot([group1, group2_shuffled], positions=[1, 2], widths=0.3)
    axes[0, 1].set_xticks([1, 2])
    axes[0, 1].set_xticklabels(["Model A", "Model B"])
    axes[0, 1].set_ylabel("Performance", fontsize=12)
    axes[0, 1].set_title(
        "INDEPENDENT Samples\n(Different subjects)", fontsize=14, fontweight="bold"
    )
    axes[0, 1].text(
        1.5,
        max(group1.max(), group2_shuffled.max()) + 0.5,
        f"Mean difference: {(group2_shuffled.mean() - group1.mean()):.3f}\np-value: {p_indep:.4f}",
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat"),
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Plot 3: Distribution of differences (Paired)
    axes[1, 0].hist(
        paired_differences, bins=15, edgecolor="black", alpha=0.7, color="skyblue"
    )
    axes[1, 0].axvline(
        0, color="red", linestyle="--", linewidth=2, label="No difference"
    )
    axes[1, 0].axvline(
        paired_differences.mean(),
        color="green",
        linestyle="-",
        linewidth=2,
        label=f"Mean diff: {paired_differences.mean():.3f}",
    )
    axes[1, 0].set_xlabel("Difference (Model B - Model A)", fontsize=12)
    axes[1, 0].set_ylabel("Frequency", fontsize=12)
    axes[1, 0].set_title(
        f"Distribution of Paired Differences\nSE = {paired_differences.std(ddof=1) / np.sqrt(n):.3f}",
        fontsize=14,
        fontweight="bold",
    )
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Plot 4: Comparison of test power
    effect_sizes = np.linspace(0, 1.5, 50)
    power_paired = []
    power_indep = []

    for es in effect_sizes:
        # Simulate power for paired test
        se_paired = paired_differences.std(ddof=1) / np.sqrt(n)
        ncp_paired = es / se_paired
        power_paired.append(
            1 - stats.t.cdf(stats.t.ppf(0.975, n - 1), n - 1, ncp_paired)
        )

        # Simulate power for independent test
        pooled_std = np.sqrt((group1.var(ddof=1) + group2.var(ddof=1)) / 2)
        se_indep = pooled_std * np.sqrt(2 / n)
        ncp_indep = es / se_indep
        power_indep.append(
            1 - stats.t.cdf(stats.t.ppf(0.975, 2 * n - 2), 2 * n - 2, ncp_indep)
        )

    axes[1, 1].plot(
        effect_sizes, power_paired, "b-", linewidth=3, label="Paired test", alpha=0.8
    )
    axes[1, 1].plot(
        effect_sizes,
        power_indep,
        "r-",
        linewidth=3,
        label="Independent test",
        alpha=0.8,
    )
    axes[1, 1].axhline(
        0.8,
        color="gray",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
        label="Target power = 0.8",
    )
    axes[1, 1].axvline(
        true_difference,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"True difference = {true_difference}",
    )
    axes[1, 1].set_xlabel("Effect Size", fontsize=12)
    axes[1, 1].set_ylabel("Statistical Power", fontsize=12)
    axes[1, 1].set_title(
        "Power Comparison: Paired vs Independent", fontsize=14, fontweight="bold"
    )
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1.5)
    axes[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()


def compare_pooled_welch(mean_diff=1, var1=1, var2=1, n1=30, n2=30):
    """
    Compare Student's (pooled) and Welch's t-tests
    """
    np.random.seed(42)

    # Generate two samples
    sample1 = np.random.normal(0, np.sqrt(var1), n1)
    sample2 = np.random.normal(mean_diff, np.sqrt(var2), n2)

    # Calculate statistics
    x1_bar = sample1.mean()
    x2_bar = sample2.mean()
    s1 = sample1.std(ddof=1)
    s2 = sample2.std(ddof=1)

    # Student's t-test (pooled variance)
    pooled_var = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    pooled_std = np.sqrt(pooled_var)
    se_pooled = pooled_std * np.sqrt(1 / n1 + 1 / n2)
    t_pooled = (x1_bar - x2_bar) / se_pooled
    df_pooled = n1 + n2 - 2
    p_pooled = 2 * (1 - stats.t.cdf(abs(t_pooled), df_pooled))

    # Welch's t-test (separate variances)
    se_welch = np.sqrt(s1**2 / n1 + s2**2 / n2)
    t_welch = (x1_bar - x2_bar) / se_welch
    df_welch = (s1**2 / n1 + s2**2 / n2) ** 2 / (
        (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
    )
    p_welch = 2 * (1 - stats.t.cdf(abs(t_welch), df_welch))

    # scipy verification
    t_scipy_equal, p_scipy_equal = stats.ttest_ind(sample1, sample2, equal_var=True)
    t_scipy_unequal, p_scipy_unequal = stats.ttest_ind(
        sample1, sample2, equal_var=False
    )

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Sample distributions
    axes[0, 0].hist(
        sample1,
        bins=15,
        alpha=0.6,
        label=f"Group 1 (n={n1})",
        edgecolor="black",
        color="skyblue",
    )
    axes[0, 0].hist(
        sample2,
        bins=15,
        alpha=0.6,
        label=f"Group 2 (n={n2})",
        edgecolor="black",
        color="salmon",
    )
    axes[0, 0].axvline(
        x1_bar, color="blue", linestyle="--", linewidth=2, label=f"Mean 1: {x1_bar:.3f}"
    )
    axes[0, 0].axvline(
        x2_bar, color="red", linestyle="--", linewidth=2, label=f"Mean 2: {x2_bar:.3f}"
    )
    axes[0, 0].set_xlabel("Value", fontsize=12)
    axes[0, 0].set_ylabel("Frequency", fontsize=12)
    axes[0, 0].set_title("Sample Distributions", fontsize=14, fontweight="bold")
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis="y")

    # Plot 2: Comparison of standard errors
    methods = ["Pooled\n(Student)", "Welch"]
    se_values = [se_pooled, se_welch]
    colors = ["skyblue", "lightcoral"]

    bars = axes[0, 1].bar(
        methods, se_values, color=colors, edgecolor="black", linewidth=2, alpha=0.7
    )
    for bar, se in zip(bars, se_values):
        height = bar.get_height()
        axes[0, 1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{se:.4f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    axes[0, 1].set_ylabel("Standard Error", fontsize=12)
    axes[0, 1].set_title("Standard Error Comparison", fontsize=14, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3, axis="y")

    # Plot 3: t-distributions
    x_t = np.linspace(-5, 5, 1000)
    y_pooled = stats.t.pdf(x_t, df_pooled)
    y_welch = stats.t.pdf(x_t, df_welch)

    axes[1, 0].plot(
        x_t, y_pooled, "b-", linewidth=2, label=f"Pooled (df={df_pooled})", alpha=0.8
    )
    axes[1, 0].plot(
        x_t, y_welch, "r-", linewidth=2, label=f"Welch (df={df_welch:.1f})", alpha=0.8
    )
    axes[1, 0].axvline(
        t_pooled,
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"t_pooled={t_pooled:.3f}",
    )
    axes[1, 0].axvline(
        t_welch,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"t_welch={t_welch:.3f}",
    )
    axes[1, 0].axvline(0, color="gray", linestyle="-", linewidth=1, alpha=0.5)
    axes[1, 0].set_xlabel("t-value", fontsize=12)
    axes[1, 0].set_ylabel("Probability Density", fontsize=12)
    axes[1, 0].set_title(
        "t-Distributions and Test Statistics", fontsize=14, fontweight="bold"
    )
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: P-value comparison
    p_values = [p_pooled, p_welch]
    colors_p = ["#90EE90" if p < 0.05 else "#FFB6C6" for p in p_values]

    bars = axes[1, 1].bar(
        methods, p_values, color=colors_p, edgecolor="black", linewidth=2, alpha=0.7
    )
    for bar, p in zip(bars, p_values):
        height = bar.get_height()
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f"{p:.4f}",
            ha="center",
            fontsize=12,
            fontweight="bold",
        )

    axes[1, 1].axhline(
        0.05, color="orange", linestyle="--", linewidth=2, label="α = 0.05"
    )
    axes[1, 1].set_ylabel("P-value", fontsize=12)
    axes[1, 1].set_title("P-value Comparison", fontsize=14, fontweight="bold")
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3, axis="y")
    axes[1, 1].set_ylim(0, max(p_values) * 1.3)

    plt.tight_layout()
    plt.show()

    # Summary table
    print("=" * 80)
    print("COMPARISON: POOLED vs WELCH'S t-TEST")
    print("=" * 80)
    print(f"Group 1: n={n1}, mean={x1_bar:.4f}, std={s1:.4f}, var={var1:.4f}")
    print(f"Group 2: n={n2}, mean={x2_bar:.4f}, std={s2:.4f}, var={var2:.4f}")
    print(f"Variance ratio: {max(var1, var2) / min(var1, var2):.2f}")
    print("-" * 80)
    print(
        f"{'Method':<25} {'t-statistic':<15} {'df':<15} {'p-value':<15} {'Reject H₀?':<15}"
    )
    print("-" * 80)
    print(
        f"{'Student (pooled)':<25} {t_pooled:<15.4f} {df_pooled:<15.1f} {p_pooled:<15.6f} {'✓' if p_pooled < 0.05 else '✗':<15}"
    )
    print(
        f"{'Welch (separate var)':<25} {t_welch:<15.4f} {df_welch:<15.1f} {p_welch:<15.6f} {'✓' if p_welch < 0.05 else '✗':<15}"
    )
    print(
        f"{'scipy (equal_var=True)':<25} {t_scipy_equal:<15.4f} {df_pooled:<15.1f} {p_scipy_equal:<15.6f} {'✓' if p_scipy_equal < 0.05 else '✗':<15}"
    )
    print(
        f"{'scipy (equal_var=False)':<25} {t_scipy_unequal:<15.4f} {df_welch:<15.1f} {p_scipy_unequal:<15.6f} {'✓' if p_scipy_unequal < 0.05 else '✗':<15}"
    )
    print("-" * 80)

    if abs(var1 - var2) / max(var1, var2) < 0.2:
        print("\n💡 Variances are similar → Both tests give nearly identical results")
    else:
        print("\n💡 Variances differ substantially → Welch's test is more appropriate")
        if abs(p_pooled - p_welch) > 0.01:
            print("   ⚠️  Notice the p-values differ! Pooled test may be unreliable.")

    print("=" * 80)


def demo_effect_size(py=False):
    # Scenario 1: Large sample, tiny effect
    print("\nScenario 1: Large Sample Size")
    print("-" * 80)

    np.random.seed(42)
    n_large = 10000
    group1_large = np.random.normal(100, 15, n_large)
    group2_large = np.random.normal(101, 15, n_large)  # Tiny 1-point difference

    t_stat_large, p_value_large = stats.ttest_ind(group1_large, group2_large)
    mean_diff_large = np.mean(group2_large) - np.mean(group1_large)

    print(f"  Group 1: mean = {np.mean(group1_large):.2f}, n = {n_large}")
    print(f"  Group 2: mean = {np.mean(group2_large):.2f}, n = {n_large}")
    print(f"  Difference: {mean_diff_large:.2f}")
    print(f"  t-statistic: {t_stat_large:.4f}")
    print(f"  p-value: {p_value_large:.6f}")

    if p_value_large < 0.05:
        print("\n  ✓ STATISTICALLY SIGNIFICANT (p < 0.05)")
        print(f"  ✗ But the difference is only {mean_diff_large:.2f} points!")
        print("  ✗ Is a 1-point difference meaningful? Probably not!")

    # Scenario 2: Small sample, large effect
    print("\n\nScenario 2: Small Sample Size")
    print("-" * 80)

    n_small = 20
    group1_small = np.random.normal(100, 15, n_small)
    group2_small = np.random.normal(115, 15, n_small)  # Large 15-point difference

    t_stat_small, p_value_small = stats.ttest_ind(group1_small, group2_small)
    mean_diff_small = np.mean(group2_small) - np.mean(group1_small)

    print(f"  Group 1: mean = {np.mean(group1_small):.2f}, n = {n_small}")
    print(f"  Group 2: mean = {np.mean(group2_small):.2f}, n = {n_small}")
    print(f"  Difference: {mean_diff_small:.2f}")
    print(f"  t-statistic: {t_stat_small:.4f}")
    print(f"  p-value: {p_value_small:.6f}")

    if p_value_small >= 0.05:
        print("\n  ✗ NOT STATISTICALLY SIGNIFICANT (p ≥ 0.05)")
        print(f"  ✓ But the difference is {mean_diff_small:.2f} points!")
        print("  ✓ This could be a very important difference!")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Large sample, small effect
    axes[0].hist(
        group1_large,
        bins=50,
        alpha=0.6,
        label="Group 1",
        color="blue",
        edgecolor="black",
    )
    axes[0].hist(
        group2_large,
        bins=50,
        alpha=0.6,
        label="Group 2",
        color="red",
        edgecolor="black",
    )
    axes[0].axvline(np.mean(group1_large), color="blue", linestyle="--", linewidth=2)
    axes[0].axvline(np.mean(group2_large), color="red", linestyle="--", linewidth=2)
    axes[0].set_xlabel("Value", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].set_title(
        f"Large Sample (n={n_large})\nDiff={mean_diff_large:.2f}, p={p_value_large:.6f}\n✓ Significant, ✗ Tiny Effect",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Small sample, large effect
    axes[1].hist(
        group1_small,
        bins=10,
        alpha=0.6,
        label="Group 1",
        color="blue",
        edgecolor="black",
    )
    axes[1].hist(
        group2_small,
        bins=10,
        alpha=0.6,
        label="Group 2",
        color="red",
        edgecolor="black",
    )
    axes[1].axvline(np.mean(group1_small), color="blue", linestyle="--", linewidth=2)
    axes[1].axvline(np.mean(group2_small), color="red", linestyle="--", linewidth=2)
    axes[1].set_xlabel("Value", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].set_title(
        f"Small Sample (n={n_small})\nDiff={mean_diff_small:.2f}, p={p_value_small:.6f}\n✗ Not Significant, ✓ Large Effect",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()

    return fig


def viz_f_dist(py=False):
    """
    Docstring for viz_f_dist

    :param py: Description
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Effect of df1 (numerator df)
    x = np.linspace(0, 5, 1000)
    df2_fixed = 10

    axes[0, 0].set_xlim(0, 5)
    for df1 in [1, 2, 5, 10, 20]:
        y = stats.f.pdf(x, df1, df2_fixed)
        axes[0, 0].plot(x, y, linewidth=2, label=f"F({df1}, {df2_fixed})")

    axes[0, 0].set_xlabel("F value", fontsize=12)
    axes[0, 0].set_ylabel("Probability Density", fontsize=12)
    axes[0, 0].set_title(
        f"Effect of Numerator df (df₂ = {df2_fixed} fixed)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(
        3,
        0.7,
        "As df₁ increases:\n→ Peak moves right\n→ Less skewed\n→ More symmetric",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    # Plot 2: Effect of df2 (denominator df)
    df1_fixed = 5

    axes[0, 1].set_xlim(0, 5)
    for df2 in [2, 5, 10, 20, 50]:
        y = stats.f.pdf(x, df1_fixed, df2)
        axes[0, 1].plot(x, y, linewidth=2, label=f"F({df1_fixed}, {df2})")

    axes[0, 1].set_xlabel("F value", fontsize=12)
    axes[0, 1].set_ylabel("Probability Density", fontsize=12)
    axes[0, 1].set_title(
        f"Effect of Denominator df (df₁ = {df1_fixed} fixed)",
        fontsize=14,
        fontweight="bold",
    )
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(
        3,
        0.4,
        "As df₂ increases:\n→ Distribution concentrates\n→ Variance decreases\n→ Mean → 1",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    # Plot 3: Critical values visualization
    df1, df2 = 5, 20
    x = np.linspace(0, 6, 1000)
    y = stats.f.pdf(x, df1, df2)

    axes[1, 0].plot(x, y, "b-", linewidth=2, label=f"F({df1}, {df2})")
    axes[1, 0].fill_between(x, 0, y, alpha=0.3, color="blue")

    # Critical values for different alpha levels
    alphas = [0.10, 0.05, 0.01]
    colors = ["green", "orange", "red"]

    for alpha, color in zip(alphas, colors):
        f_critical = stats.f.ppf(1 - alpha, df1, df2)
        axes[1, 0].axvline(
            f_critical,
            color=color,
            linestyle="--",
            linewidth=2,
            label=f"α = {alpha}: F = {f_critical:.3f}",
        )

        # Shade rejection region
        x_reject = x[x >= f_critical]
        y_reject = stats.f.pdf(x_reject, df1, df2)
        axes[1, 0].fill_between(x_reject, 0, y_reject, alpha=0.2, color=color)

    axes[1, 0].set_xlabel("F value", fontsize=12)
    axes[1, 0].set_ylabel("Probability Density", fontsize=12)
    axes[1, 0].set_title(
        "Critical Values for Different α Levels", fontsize=14, fontweight="bold"
    )
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Comparison with t-distribution relationship
    df = 10
    x_vals = np.linspace(-4, 4, 1000)
    t_vals = stats.t.pdf(x_vals, df)

    x_f_vals = np.linspace(0, 20, 1000)
    f_vals = stats.f.pdf(x_f_vals, 1, df)

    # t-distribution
    ax_t = axes[1, 1]
    ax_t.plot(x_vals, t_vals, "b-", linewidth=2, label=f"t({df})")
    ax_t.set_xlabel("t value", fontsize=12)
    ax_t.set_ylabel("Probability Density", fontsize=12, color="b")
    ax_t.tick_params(axis="y", labelcolor="b")
    ax_t.grid(True, alpha=0.3)

    # F-distribution on same plot (secondary axis)
    ax_f = ax_t.twiny()
    ax_f.plot(x_f_vals, f_vals, "r-", linewidth=2, label=f"F(1, {df})")
    ax_f.set_xlabel("F value (= t²)", fontsize=12, color="r")
    ax_f.tick_params(axis="x", labelcolor="r")

    axes[1, 1].set_title(
        f"Relationship: t²({df}) = F(1, {df})", fontsize=14, fontweight="bold"
    )
    axes[1, 1].legend(loc="upper left", fontsize=10)
    ax_f.legend(loc="upper right", fontsize=10)

    # Add annotation
    axes[1, 1].text(
        0.5,
        0.5,
        f"If t ~ t({df})\nthen t² ~ F(1, {df})",
        transform=axes[1, 1].transAxes,
        ha="center",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
    )

    plt.tight_layout()
    if py:
        plt.show()
    else:
        plt.close()

    return fig


if __name__ == "__main__":
    print("Hypothesis Testing Module")
    # visualize_hypothesis_types()
    # create_error_matrix(py=True)
    # simulate_hypothesis_test_errors()
    # interactive_pvalue_visualization()
    # simulate_pvalue_distribution()
    # visualize_test_statistic_concept()
    # compare_z_t_distributions()
    # demo_effect_size(py=True)
    viz_f_dist(py=True)
