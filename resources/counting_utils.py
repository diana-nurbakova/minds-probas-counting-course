import networkx as nx
import matplotlib.pyplot as plt
from itertools import permutations
from collections import defaultdict
import pandas as pd
from IPython.display import HTML, display
from itertools import combinations_with_replacement

def create_permutation_graph(elements, k=None):
    """Create a directed graph showing permutation/arrangement generation
    
    Parameters:
    elements: list of elements to permute
    k: number of elements to select (if None, uses all elements for full permutations)
    """
    G = nx.DiGraph()
    
    if k is None:
        k = len(elements)
    
    def add_permutation_nodes(current_perm, remaining, level):
        node_id = ''.join(current_perm) if current_perm else 'root'
        G.add_node(node_id, level=level, label=''.join(current_perm))
        
        # Stop when we've selected k elements
        if len(current_perm) < k and remaining:
            for elem in remaining:
                next_perm = current_perm + [elem]
                next_remaining = [e for e in remaining if e != elem]
                child_id = ''.join(next_perm)
                
                G.add_edge(node_id, child_id)
                add_permutation_nodes(next_perm, next_remaining, level + 1)
    
    add_permutation_nodes([], elements, 0)
    return G

def visualize_permutation_graph(elements, k=None, py=False):
    G = create_permutation_graph(elements, k=k)
        
    pos = {}
    levels = nx.get_node_attributes(G, 'level')
    
    # Group nodes by level
    level_nodes = defaultdict(list)
    for node, level in levels.items():
        level_nodes[level].append(node)
    
    # Adaptive spacing based on label length
    max_label_len = max(len(elem) for elem in elements)
    spacing_factor = max(1, max_label_len / 3)
    
    # Calculate positions
    y_spacing = 1.5
    for level, nodes in level_nodes.items():
        n_nodes = len(nodes)
        # Calculate x spacing based on level
        x_spacing = max(3.0 / (level + 1), 1.0) * spacing_factor
        total_width = (n_nodes - 1) * x_spacing
        
        for i, node in enumerate(sorted(nodes)):
            x = i * x_spacing - total_width / 2
            y = -level * y_spacing
            pos[node] = (x, y)
    
    # Find the actual leftmost position
    min_x = min(p[0] for p in pos.values()) if pos else 0
    max_x = max(p[0] for p in pos.values()) if pos else 0
    
    # Adaptive figure size
    fig_width = max(16, (max_x - min_x) * 1.3)
    if k is None:
        fig_height = max(10,  len(elements)* 3)
    else:
        fig_height = max(10,  k * 3)
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, 
                          edge_color='#FF6B6B',
                          arrows=True,
                          arrowsize=15,
                          width=2,
                          alpha=0.6,
                          arrowstyle='->')
    
    # Draw nodes by level with different colors
    colors = ['#FFB6C1', '#FFE4E1', '#FFF0F5']  # Light coral shades
    for level in level_nodes:
        color_idx = min(level - 1, len(colors) - 1)
        nx.draw_networkx_nodes(G, pos,
                              nodelist=level_nodes[level],
                              node_color=colors[color_idx],
                              node_size=1200,
                              edgecolors='#FF6B6B',
                              linewidths=2)
    
    # Draw labels
    labels = nx.get_node_attributes(G, 'label')
    font_size = max(9, min(14, 50 / max_label_len))
    nx.draw_networkx_labels(G, pos, labels,
                           font_size=font_size,
                           font_family='sans-serif')
    
    # Add level indicators
    if k is None:
        level_num = len(elements) + 1
    else:
        level_num = k + 1
    
    # Position level indicators OUTSIDE the leftmost nodes
    # Add extra padding to account for node size
    level_x_position = min_x - 1.5
    
    for level in range(1, level_num):
        plt.text(level_x_position - 1.5,
                -level * y_spacing,
                f'{level}: # options for fixed lvl-1: {int(len(level_nodes[level])/len(level_nodes.get(level-1, level_nodes[1])))}',
                fontsize=18,
                color='#FF6B6B',
                fontweight='bold',
                ha='center',
                va='center')
    
    # Add title with element labels at top
    #plt.text(0, 0.5, '  '.join(elements),
    #        fontsize=16,
    #        style='italic',
    #        ha='center')
    
    plt.axis('off')
    plt.title(f'Permutation Tree for {elements}', 
             fontsize=18, fontweight='bold', pad=30)
    plt.tight_layout()
    if py:
        plt.show() 
    plt.close() # to prevent automatic display
    
    return fig

def generate_stars_and_bars_table(n=4, k=3, show_all=False):
    """
    Generate stars and bars visualization for combinations with repetition
    
    n: number of types/categories
    k: number of items to select
    """
    # Generate all multisets
    types = list(range(1, n+1))
    multisets = list(combinations_with_replacement(types, k))
    
    # Prepare data for table
    data = []
    
    for idx, mset in enumerate(multisets):
        # Count occurrences of each type
        counts = [mset.count(i) for i in types]
        
        # Create stars and bars representation
        stars_bars = ""
        for i, count in enumerate(counts):
            stars_bars += "★" * count
            if i < len(counts) - 1:
                stars_bars += "|"
        
        # Multiset notation
        multiset_str = "{" + ",".join(map(str, sorted(mset))) + "}"
        
        # Equation solution
        eq_solution = "[" + ",".join(map(str, counts)) + "]"
        
        data.append({
            'No.': idx + 1,
            'Multiset': multiset_str,
            'Eq. solution': eq_solution,
            'Stars and Bars': stars_bars
        })
        
        # Limit display if requested
        if not show_all and idx >= 9:  # Show first 10
            break
    
    df = pd.DataFrame(data)
    
    # Create HTML table with styling
    html = f"""
    <div style='font-family: Arial, sans-serif;'>
    <h3>Stars and Bars Method: Combinations with Repetition</h3>
    <p><strong>Problem:</strong> Select {k} items from {n} types (repetition allowed)</p>
    <p><strong>Solution:</strong> C({n}+{k}-1, {k}) = C({n+k-1}, {k}) = {len(multisets)}</p>
    <br>
    <div style='background-color: #f8f9ff; padding: 15px; border-radius: 8px; margin-bottom: 15px;'>
        <strong>Interpretation:</strong><br>
        • <strong>★ (stars)</strong> represent the {k} items to select<br>
        • <strong>| (bars)</strong> separate the {n} categories (need {n-1} bars)<br>
        • Total positions: {k} stars + {n-1} bars = {n+k-1}<br>
        • Choose {k} positions for stars from {n+k-1} total positions
    </div>
    """
    
    # Style the dataframe
    html += df.to_html(index=False, escape=False)
    
    if not show_all and len(multisets) > 10:
        html += f"<p style='color: #666; font-style: italic;'>Showing first 10 of {len(multisets)} total combinations...</p>"
    
    html += "</div>"
    
    # Add CSS for table styling
    styled_html = """
    <style>
    .stars-bars-table {
        border-collapse: collapse;
        width: 100%;
        margin-top: 10px;
    }
    .stars-bars-table th {
        background-color: #667eea;
        color: white;
        padding: 12px;
        text-align: center;
        font-weight: bold;
    }
    .stars-bars-table td {
        padding: 10px;
        border-bottom: 1px solid #ddd;
        text-align: center;
    }
    .stars-bars-table tr:hover {
        background-color: #f5f5f5;
    }
    .stars-bars-table td:last-child {
        font-family: monospace;
        font-size: 16px;
        color: #e74c3c;
    }
    </style>
    """ + html
    
    # Replace the table tag to include the class
    styled_html = styled_html.replace('<table', '<table class="stars-bars-table"')
    
    return styled_html
    
def main():
    elements = ['a', 'b', 'c']
    visualize_permutation_graph(elements)
    
if __name__ == "__main__":
    #elements = ['a', 'b', 'c']
    elements = ['Karine', 'Michel', 'Kevin', 'Amélie']
    visualize_permutation_graph(elements, k=2, py=True)
    