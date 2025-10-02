import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import pandas as pd
from itertools import product, combinations, permutations
import warnings
warnings.filterwarnings('ignore')
from IPython.display import HTML, display
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed


def create_probability_scale(py=False):
    # Visual representation of probability scale
    fig, ax = plt.subplots(1, 1, figsize=(12, 3))
    x = np.linspace(0, 1, 100)
    y = np.ones_like(x)

    # Create gradient effect
    for i, val in enumerate(x):
        color_intensity = val
        ax.scatter(val, 1, c=plt.cm.RdYlBu_r(color_intensity), s=30, alpha=0.7)

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(0.5, 1.5)
    ax.set_xlabel('Probability Value', fontsize=12)
    ax.set_title('The Probability Scale: From Impossible to Certain', fontsize=14, fontweight='bold')

    # Add labels
    ax.text(0, 0.8, 'Impossible\nP = 0', ha='center', fontsize=10, fontweight='bold')
    ax.text(0.5, 0.8, 'Uncertain\nP = 0.5', ha='center', fontsize=10, fontweight='bold')
    ax.text(1, 0.8, 'Certain\nP = 1', ha='center', fontsize=10, fontweight='bold')

    ax.set_yticks([])
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def rolling_dice(py=False):
    print("📍 EXAMPLE 1: Rolling a Six-Sided Die")
    dice_outcomes = {1, 2, 3, 4, 5, 6}
    print(f"Sample Space Ω = {dice_outcomes}")
    print(f"Size of sample space: |Ω| = {len(dice_outcomes)}")

    # Define events
    even_event = {2, 4, 6}
    odd_event = {1, 3, 5}
    high_event = {4, 5, 6}

    print(f"\nEvent A = 'rolling even number' = {even_event}")
    print(f"Event B = 'rolling odd number' = {odd_event}")
    print(f"Event C = 'rolling ≥ 4' = {high_event}")

    # Visualize with Venn diagram style
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Create visual representation of events
    for idx, (event_name, event_set, color) in enumerate([
        ('Even Numbers', even_event, 'lightblue'),
        ('Odd Numbers', odd_event, 'lightcoral'), 
        ('High Numbers (≥4)', high_event, 'lightgreen')
    ]):
        ax = axes[idx]
        
        # Draw all outcomes
        positions = [(i%3, i//3) for i in range(6)]
        for i, outcome in enumerate(dice_outcomes):
            x, y = positions[i]
            if outcome in event_set:
                ax.scatter(x, y, s=800, c=color, alpha=0.7, edgecolor='black', linewidth=2)
            else:
                ax.scatter(x, y, s=800, c='lightgray', alpha=0.5, edgecolor='black', linewidth=1)
            ax.text(x, y, str(outcome), ha='center', va='center', fontsize=14, fontweight='bold')
        
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.5, 1.5)
        ax.set_title(event_name, fontsize=12, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def print_classif_example():
    print("\n📍 EXAMPLE 2: AI Image Classification System")
    ai_outcomes = {'cat', 'dog', 'bird', 'car', 'tree', 'person', 'unknown'}
    print(f"Sample Space Ω = {ai_outcomes}")

    animal_event = {'cat', 'dog', 'bird'}
    object_event = {'car', 'tree'}
    error_event = {'unknown'}

    print(f"\nEvent A = 'animal detected' = {animal_event}")
    print(f"Event B = 'object detected' = {object_event}")  
    print(f"Event C = 'classification error' = {error_event}")

def prob_properties_viz(py=False):
    # Visualization of probability properties
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Property 1: Non-negativity
    probs = [0.1, 0.3, 0.6, 0.8, 0.2]
    events = ['A', 'B', 'C', 'D', 'E'] 
    ax1.bar(events, probs, color='skyblue', alpha=0.7)
    ax1.set_ylim(0, 1)
    ax1.set_title('Property 1: Non-negativity\nP(A) ≥ 0', fontweight='bold')
    ax1.set_ylabel('Probability')

    # Property 2: Normalization
    ax2.bar(['Sample Space Ω'], [1.0], color='lightcoral', alpha=0.7)
    ax2.set_ylim(0, 1.2)
    ax2.set_title('Property 2: Normalization\nP(Ω) = 1', fontweight='bold')
    ax2.set_ylabel('Probability')

    # Property 3: Additivity (disjoint events)
    disjoint_A = 0.3
    disjoint_B = 0.4
    union_AB = disjoint_A + disjoint_B
    ax3.bar(['P(A)', 'P(B)', 'P(A ∪ B)'], [disjoint_A, disjoint_B, union_AB], 
            color=['lightblue', 'lightgreen', 'orange'], alpha=0.7)
    ax3.set_ylim(0, 1)
    ax3.set_title('Property 3: Additivity\nP(A ∪ B) = P(A) + P(B) for disjoint A, B', fontweight='bold')
    ax3.set_ylabel('Probability')

    # Example: ML model confidence distribution
    confidence_scores = np.random.beta(8, 2, 1000)  # Skewed toward high confidence
    ax4.hist(confidence_scores, bins=30, density=True, alpha=0.7, color='purple')
    ax4.set_xlabel('Confidence Score')
    ax4.set_ylabel('Density')
    ax4.set_title('ML Model Confidence Distribution\n(Should integrate to 1)', fontweight='bold')
    ax4.axvline(confidence_scores.mean(), color='red', linestyle='--', 
            label=f'Mean = {confidence_scores.mean():.3f}')
    ax4.legend()

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    return fig

def examples_elementary_prob():
    # Example 1: Fair die
    print("\n1. Fair Six-Sided Die:")
    dice_outcomes = {1, 2, 3, 4, 5, 6}
    omega_size = len(dice_outcomes)
    
    # Define events
    even_event = {2, 4, 6}
    odd_event = {1, 3, 5}
    high_event = {4, 5, 6}
    
    for event_name, event_set in [('Even', even_event), ('Odd', odd_event), ('High (≥4)', high_event)]:
        prob = len(event_set) / omega_size
        print(f"   P({event_name}) = |{event_set}| / |Ω| = {len(event_set)}/{omega_size} = {prob:.3f}")

    # Verify properties
    print(f"\n   Verification:")
    print(f"   P(Even) + P(Odd) = {len(even_event)/omega_size:.3f} + {len(odd_event)/omega_size:.3f} = {(len(even_event) + len(odd_event))/omega_size:.3f}")
    print(f"   P(Ω) = {omega_size}/{omega_size} = 1.000 ✓")

    # Example 2: AI System Performance
    print("\n2. AI Classification System:")
    ai_performance = {
        'correct_classification': 0.92,
        'misclassification': 0.07, 
        'system_error': 0.01
    }

    print("   Performance probabilities:")
    for outcome, prob in ai_performance.items():
        print(f"   P({outcome}) = {prob:.3f}")

    total_prob = sum(ai_performance.values())
    print(f"\n   Total probability: {total_prob:.3f} {'✓' if abs(total_prob - 1.0) < 1e-6 else '✗'}")

def example_hospitals(py=False):
    # Create the sample space for federated learning scenario
    states = ['A', 'P', 'O']  # Active, Partial, Offline
    hospitals = ['H1', 'H2', 'H3', 'H4']

    # Generate all possible combinations
    sample_space = list(product(states, repeat=4))
    print(f"Sample Space Size: |Ω| = {len(sample_space)} possible configurations")
    print(f"First few outcomes: {sample_space[:5]}...")

    # Define events of interest
    def count_active(outcome):
        return sum(1 for state in outcome if state == 'A')

    def count_offline(outcome):
        return sum(1 for state in outcome if state == 'O')

    # Event A: At least 3 hospitals are active
    event_A = [outcome for outcome in sample_space if count_active(outcome) >= 3]
    prob_A = len(event_A) / len(sample_space)

    # Event B: No hospital is offline  
    event_B = [outcome for outcome in sample_space if count_offline(outcome) == 0]
    prob_B = len(event_B) / len(sample_space)

    # Event C: Exactly 2 hospitals are active
    event_C = [outcome for outcome in sample_space if count_active(outcome) == 2]
    prob_C = len(event_C) / len(sample_space)

    print(f"\nEvent Analysis:")
    print(f"A = 'At least 3 hospitals active': P(A) = {prob_A:.3f}")
    print(f"B = 'No hospital offline': P(B) = {prob_B:.3f}") 
    print(f"C = 'Exactly 2 hospitals active': P(C) = {prob_C:.3f}")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Event probabilities
    events = ['≥3 Active', 'No Offline', 'Exactly 2 Active']
    probabilities = [prob_A, prob_B, prob_C]
    colors = ['green', 'blue', 'orange']

    bars = ax1.bar(events, probabilities, color=colors, alpha=0.7)
    ax1.set_ylabel('Probability')
    ax1.set_title('Federated Learning System Events', fontweight='bold')
    ax1.set_ylim(0, max(probabilities) * 1.1)

    # Add probability values on bars
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')

    # Distribution of number of active hospitals
    active_counts = [count_active(outcome) for outcome in sample_space]
    active_distribution = Counter(active_counts)

    x_vals = list(active_distribution.keys())
    y_vals = [count/len(sample_space) for count in active_distribution.values()]

    ax2.bar(x_vals, y_vals, color='skyblue', alpha=0.7)
    ax2.set_xlabel('Number of Active Hospitals')
    ax2.set_ylabel('Probability') 
    ax2.set_title('Distribution of Active Hospitals', fontweight='bold')
    ax2.set_xticks(x_vals)

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

    # Verification of probability axioms
    print(f"\n🔍 Verification of Probability Axioms:")
    print(f"1. Non-negativity: All probabilities ≥ 0? {all(p >= 0 for p in [prob_A, prob_B, prob_C])}")
    print(f"2. Normalization: P(Ω) = 1? {len(sample_space)/len(sample_space) == 1}")
    print(f"3. Total probability of all active counts: {sum(y_vals):.3f}")
    
    return fig

def ai_reliability_example(py=False):
    class AIReliabilityCalculator:
        """Calculate reliability metrics for AI systems"""
        
        def __init__(self, component_reliabilities):
            """
            component_reliabilities: dict mapping component names to reliability scores
            """
            self.components = component_reliabilities
            self.sample_space = list(product([0, 1], repeat=len(component_reliabilities)))
        
        def system_failure_probability(self, failure_mode='any'):
            """Calculate probability of system failure"""
            if failure_mode == 'any':
                # System fails if ANY component fails
                prob_all_work = 1.0
                for reliability in self.components.values():
                    prob_all_work *= reliability
                return 1 - prob_all_work
            elif failure_mode == 'all':
                # System fails only if ALL components fail
                prob_any_work = 1.0
                for reliability in self.components.values():
                    prob_any_work *= (1 - reliability)
                return prob_any_work
        
        def critical_component_analysis(self):
            """Identify most critical components"""
            impacts = {}
            baseline = self.system_failure_probability()
            
            for comp_name, reliability in self.components.items():
                # Temporarily set component to perfect reliability
                temp_components = self.components.copy()
                temp_components[comp_name] = 1.0
                temp_calc = AIReliabilityCalculator(temp_components)
                improvement = baseline - temp_calc.system_failure_probability()
                impacts[comp_name] = improvement
                
            return impacts

    # Example: Autonomous Vehicle AI System
    av_components = {
        'camera_system': 0.995,
        'lidar_sensor': 0.998, 
        'radar_system': 0.997,
        'ai_processor': 0.999,
        'decision_engine': 0.993,
        'communication': 0.990
    }

    print("🚗 AUTONOMOUS VEHICLE RELIABILITY ANALYSIS")
    av_calculator = AIReliabilityCalculator(av_components)

    failure_prob = av_calculator.system_failure_probability('any')
    reliability_prob = 1 - failure_prob
    print(f"\nSystem failure probability: {failure_prob:.6f} ({failure_prob*100:.4f}%)")
    print(f"System reliability: {reliability_prob:.6f} ({(reliability_prob)*100:.4f}%)")

    # Critical component analysis
    impacts = av_calculator.critical_component_analysis()
    print(f"\n🔍 Critical Component Analysis:")
    sorted_impacts = sorted(impacts.items(), key=lambda x: x[1], reverse=True)
    for component, impact in sorted_impacts:
        print(f"   {component}: {impact:.6f} improvement if perfected")
    
    max_impact = sorted_impacts[0][1]
    # print(max_impact)
    max_impact_key = [key for key, val in sorted_impacts if val == max_impact]
    print(f"The most critical component: {max_impact_key}")
    
    # expected failure hours per year
    print(f"\nFailure Time:")
    total_hours = 10000
    expected_failure_hours = total_hours * failure_prob
    expected_failure_hours_per_month = expected_failure_hours / 12
    expected_failure_hours_day = expected_failure_hours / 365
    print(f"Expected failure time: {expected_failure_hours:.2f}h per year ({expected_failure_hours_per_month:.2f}h per month or {expected_failure_hours_day:.2f}h per day)")
    
    # safety assessment
    print(f"\nSafety Assessment:")
    required_reliability_level = 0.999
    gap = required_reliability_level - reliability_prob
    if gap > 0:
        conclusion = 'NO, the system does NOT meet safety standards'
    else:
        conclusion = "YES, the system meets safety standards"
    print(f"{conclusion}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Component reliabilities
    components = list(av_components.keys())
    reliabilities = list(av_components.values())

    bars1 = ax1.barh(components, reliabilities, color='lightblue', alpha=0.7)
    ax1.set_xlabel('Reliability')
    ax1.set_title('Component Reliabilities', fontweight='bold')
    ax1.set_xlim(0.98, 1.0)

    # Add values on bars
    for bar, rel in zip(bars1, reliabilities):
        width = bar.get_width()
        ax1.text(width - 0.0005, bar.get_y() + bar.get_height()/2,
                f'{rel:.3f}', ha='right', va='center', fontweight='bold')

    # Critical component impacts
    impact_components = [item[0] for item in sorted_impacts]
    impact_values = [item[1] for item in sorted_impacts]

    bars2 = ax2.barh(impact_components, impact_values, color='lightcoral', alpha=0.7)
    ax2.set_xlabel('Failure Probability Reduction')
    ax2.set_title('Critical Component Analysis', fontweight='bold')

    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig


def solve_ai_doctor_paradox(py=False):
    """Solve the opening challenge using Bayes' theorem"""
    
    print("🎯 AI Doctor Paradox - SOLUTION")
    print("="*40)
    
    # Given information
    disease_rate = 0.01        # 1% of population has disease
    sensitivity = 0.95         # 95% accuracy detecting disease
    specificity = 0.90         # 90% accuracy identifying healthy patients
    
    no_disease_rate = 1 - disease_rate  # 99% of population is heathy

    print("Given:")
    print(f"- Disease prevalence: {disease_rate:.1%}")
    print(f"- Healthy population rate: {no_disease_rate}")
    print(f"- AI sensitivity (accuracy detecting disease, True Positive Rate): {sensitivity:.0%}")
    print(f"- AI specificity (accuracy identifying healthy patients, True Negative Rate): {specificity:.0%}")

    # false positive rate (if a patient is HEALTHY but the test says positive)
    false_positive_rate = 1 - specificity
    print(f"- False Positive Rate): {false_positive_rate:.0%}")
    
    # P(positive test)
    p_positive = (sensitivity * disease_rate + 
                  false_positive_rate * (1 - disease_rate))
    
    # P(disease | positive test)
    posterior = (sensitivity * disease_rate) / p_positive
    
    print(f"\nSolution using Bayes' Theorem:")
    print(f"P(Disease | Positive Test) = {posterior:.3f} = {posterior:.1%}")
    
    print(f"\n🤯 SHOCKING RESULT:")
    print(f"Even with 95% accuracy, a positive test only indicates")
    print(f"an {posterior:.1%} chance the patient has the disease!")
    
    print(f"\nWhy intuition fails:")
    print(f"- Base rate is very low (1%)")
    print(f"- Many false positives from healthy patients")
    print(f"- False positives outnumber true positives!")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Population breakdown
    population = 10000
    diseased = int(population * disease_rate)
    healthy = population - diseased
    
    true_positives = int(diseased * sensitivity)
    false_negatives = diseased - true_positives
    false_positives = int(healthy * false_positive_rate)
    true_negatives = healthy - false_positives
    
    categories = ['True\nPositives', 'False\nPositives', 'True\nNegatives', 'False\nNegatives']
    counts = [true_positives, false_positives, true_negatives, false_negatives]
    colors = ['green', 'red', 'lightgreen', 'orange']
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.7)
    ax1.set_ylabel('Number of People')
    ax1.set_title(f'Population Breakdown (n={population:,})')
    
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Positive test breakdown
    total_positives = true_positives + false_positives
    positive_categories = ['Actually\nDiseased', 'Actually\nHealthy']
    positive_counts = [true_positives, false_positives]
    positive_colors = ['green', 'red']
    
    bars2 = ax2.bar(positive_categories, positive_counts, color=positive_colors, alpha=0.7)
    ax2.set_ylabel('Number of People')
    ax2.set_title(f'Among {total_positives} Positive Tests')
    
    for bar, count in zip(bars2, positive_counts):
        percentage = count / total_positives * 100
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return posterior   

def netflix_conditional_probability(py=False):
    """Visualize conditional probability with Netflix data"""
    
    # Simulate 10,000 Netflix users
    n_users = 10000
    
    # 40% watch comedies
    watch_comedy = np.random.binomial(1, 0.4, n_users)
    
    # Conditional probabilities for romantic movies
    watch_romantic = np.zeros(n_users)
    for i in range(n_users):
        if watch_comedy[i] == 1:
            # 60% of comedy watchers also watch romantic
            watch_romantic[i] = np.random.binomial(1, 0.6)
        else:
            # 20% of non-comedy watchers watch romantic  
            watch_romantic[i] = np.random.binomial(1, 0.2)
    
    # Create contingency table
    contingency = pd.crosstab(
        pd.Series(watch_comedy, name='Watches Comedy'),
        pd.Series(watch_romantic, name='Watches Romantic'),
        margins=True
    )
    
    print("Netflix User Analysis (10,000 users):")
    print("="*40)
    print(contingency)
    
    # Calculate conditional probabilities
    comedy_watchers = np.sum(watch_comedy)
    romantic_watchers = np.sum(watch_romantic)
    both_watchers = np.sum((watch_comedy == 1) & (watch_romantic == 1))
    
    # P(Romantic | Comedy) - given
    p_romantic_given_comedy = both_watchers / comedy_watchers
    
    # P(Comedy | Romantic) - what we want to find
    p_comedy_given_romantic = both_watchers / romantic_watchers
    
    print(f"\nConditional Probabilities:")
    print(f"P(Romantic | Comedy) = {p_romantic_given_comedy:.3f} (≈ 0.600)")
    print(f"P(Comedy | Romantic) = {p_comedy_given_romantic:.3f}")
    print(f"P(Comedy) = {comedy_watchers/n_users:.3f} (≈ 0.400)")
    print(f"P(Romantic) = {romantic_watchers/n_users:.3f}")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Venn diagram style visualization
    comedy_only = comedy_watchers - both_watchers
    romantic_only = romantic_watchers - both_watchers
    neither = n_users - comedy_only - romantic_only - both_watchers
    
    labels = ['Comedy Only', 'Both', 'Romantic Only', 'Neither']
    sizes = [comedy_only, both_watchers, romantic_only, neither]
    colors = ['lightblue', 'red', 'lightgreen', 'lightgray']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Netflix User Preferences Distribution')
    
    # Conditional probability bar chart
    scenarios = ['P(R|C)', 'P(C|R)', 'P(C)', 'P(R)']
    probabilities = [
        p_romantic_given_comedy,
        p_comedy_given_romantic, 
        comedy_watchers/n_users,
        romantic_watchers/n_users
    ]
    
    bars = ax2.bar(scenarios, probabilities, color=['orange', 'red', 'blue', 'green'])
    ax2.set_ylabel('Probability')
    ax2.set_title('Conditional vs Marginal Probabilities')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, prob in zip(bars, probabilities):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return p_comedy_given_romantic

def demonstrate_independence(py=False):
    """Show difference between independent and dependent events"""
    
    print("Independence Analysis:")
    print("="*40)
    
    # Independent events: Two fair coin flips
    n_trials = 10000
    coin1 = np.random.binomial(1, 0.5, n_trials)  
    coin2 = np.random.binomial(1, 0.5, n_trials)
    
    # Dependent events: Weather and umbrella
    rain_prob = 0.3
    rain = np.random.binomial(1, rain_prob, n_trials)
    
    # P(umbrella | rain) = 0.8, P(umbrella | no rain) = 0.2
    umbrella = np.zeros(n_trials)
    for i in range(n_trials):
        if rain[i] == 1:
            umbrella[i] = np.random.binomial(1, 0.8)
        else:
            umbrella[i] = np.random.binomial(1, 0.2)
    
    # Calculate probabilities
    # Coins (should be independent)
    p_coin1 = np.mean(coin1)
    p_coin1_given_coin2 = np.mean(coin1[coin2 == 1]) if np.sum(coin2) > 0 else 0
    
    print(f"Coin Flips (Independent Events):")
    print(f"  P(Coin1 = H) = {p_coin1:.4f}")
    print(f"  P(Coin1 = H | Coin2 = H) = {p_coin1_given_coin2:.4f}")
    print(f"  Difference: {abs(p_coin1 - p_coin1_given_coin2):.4f} ≈ 0")
    
    # Rain and umbrella (should be dependent)
    p_umbrella = np.mean(umbrella)
    p_umbrella_given_rain = np.mean(umbrella[rain == 1]) if np.sum(rain) > 0 else 0
    p_umbrella_given_no_rain = np.mean(umbrella[rain == 0]) if np.sum(rain == 0) > 0 else 0
    
    print(f"\nRain and Umbrella (Dependent Events):")
    print(f"  P(Umbrella) = {p_umbrella:.4f}")
    print(f"  P(Umbrella | Rain) = {p_umbrella_given_rain:.4f}")
    print(f"  P(Umbrella | No Rain) = {p_umbrella_given_no_rain:.4f}")
    print(f"  Big difference! → Dependence")
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Independence visualization
    contingency_coins = pd.crosstab(
        pd.Series(coin1, name='Coin 1'),
        pd.Series(coin2, name='Coin 2'),
        normalize=True
    )
    
    sns.heatmap(contingency_coins, annot=True, cmap='Blues', ax=ax1, 
                cbar_kws={'label': 'Probability'})
    ax1.set_title('Independent Events: Coin Flips\n(All cells ≈ 0.25)')
    
    # Dependence visualization  
    contingency_weather = pd.crosstab(
        pd.Series(rain, name='Rain'),
        pd.Series(umbrella, name='Umbrella'),
        normalize=True
    )
    
    sns.heatmap(contingency_weather, annot=True, cmap='Reds', ax=ax2,
                cbar_kws={'label': 'Probability'})
    ax2.set_title('Dependent Events: Rain & Umbrella\n(Unequal probabilities)')
    
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display

def interactive_bayes_explorer():
    """Interactive widget for exploring Bayes' theorem"""
    
    def bayes_calculator(prior, likelihood, evidence):
        posterior = (likelihood * prior) / evidence
        return posterior
    
    def update_bayes(prior_disease=0.01, sensitivity=0.95, specificity=0.90):
        """Medical diagnosis example with interactive parameters"""
        
        # Calculate P(test positive)
        false_positive_rate = 1 - specificity
        p_test_positive = (sensitivity * prior_disease + 
                          false_positive_rate * (1 - prior_disease))
        
        # Apply Bayes' theorem
        posterior = (sensitivity * prior_disease) / p_test_positive
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Tree diagram
        ax1.text(0.1, 0.8, f'Disease\n{prior_disease:.3f}', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        ax1.text(0.1, 0.2, f'No Disease\n{1-prior_disease:.3f}', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        
        ax1.text(0.5, 0.9, f'Test +\n{sensitivity:.3f}', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange"))
        ax1.text(0.5, 0.7, f'Test -\n{1-sensitivity:.3f}', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax1.text(0.5, 0.3, f'Test +\n{false_positive_rate:.3f}', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="orange"))
        ax1.text(0.5, 0.1, f'Test -\n{specificity:.3f}', ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        # Arrows
        ax1.arrow(0.15, 0.8, 0.3, 0.08, head_width=0.02, head_length=0.02, fc='black')
        ax1.arrow(0.15, 0.8, 0.3, -0.08, head_width=0.02, head_length=0.02, fc='black')
        ax1.arrow(0.15, 0.2, 0.3, 0.08, head_width=0.02, head_length=0.02, fc='black')
        ax1.arrow(0.15, 0.2, 0.3, -0.08, head_width=0.02, head_length=0.02, fc='black')
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_title('Probability Tree Diagram')
        ax1.axis('off')
        
        # Results bar chart
        categories = ['Prior\nP(Disease)', 'Posterior\nP(Disease|Test+)']
        probabilities = [prior_disease, posterior]
        colors = ['blue', 'red']
        
        bars = ax2.bar(categories, probabilities, color=colors, alpha=0.7)
        ax2.set_ylabel('Probability')
        ax2.set_title('Before vs After Test')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for bar, prob in zip(bars, probabilities):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Medical Diagnosis Analysis:")
        print(f"="*40)
        print(f"Prior probability of disease: {prior_disease:.3f}")
        print(f"Test sensitivity: {sensitivity:.3f}")
        print(f"Test specificity: {specificity:.3f}")
        print(f"Probability of positive test: {p_test_positive:.3f}")
        print(f"Posterior probability (Disease | Test+): {posterior:.3f}")
        print(f"\nSurprising insight: Even with {sensitivity:.0%} accuracy,")
        print(f"a positive test only indicates {posterior:.1%} chance of disease!")
        
        return posterior
    
    # Create interactive widget
    interactive_plot = interactive(
        update_bayes,
        prior_disease=widgets.FloatSlider(value=0.01, min=0.001, max=0.1, step=0.001, 
                                         description='Disease Rate'),
        sensitivity=widgets.FloatSlider(value=0.95, min=0.5, max=1.0, step=0.01,
                                       description='Sensitivity'),
        specificity=widgets.FloatSlider(value=0.90, min=0.5, max=1.0, step=0.01,
                                       description='Specificity')
    )
    
    display(interactive_plot)

    print("🎮 Interactive Bayes' Theorem Explorer:")
    print("Try different parameter values to see how they affect the posterior probability!")

if __name__ == "__main__":
   #create_probability_scale(py=True) 
   #rolling_dice(py=True)
   #prob_properties_viz(py=True)
   #examples_elementary_prob()
   #example_hospitals(py=True)
   #ai_reliability_example(py=True)
   #solve_ai_doctor_paradox(py=True)
   #p_comedy_given_romantic = netflix_conditional_probability(py=True)
   demonstrate_independence(py=True)