### 2D complexity plot
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

import seaborn as sns

def set_color_map(plt, palette: str = "colorblind") -> bool:
    """
    Sets the color cycle for the specified plot using a Seaborn palette.
    """
    try:
        colors = sns.color_palette(palette)
        plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colors)
        return True
    except Exception as e:
        print(f"Error setting color map: {e}")
        return False

set_color_map(plt, "colorblind")

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "font.serif": "Times",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "font.size": 22,
    }
)
use_pgf = True
plt.rcParams["text.usetex"] = True
plt.rcParams["figure.figsize"] = (8,5)

colors = sns.color_palette("colorblind")

def calculate_entropy(probs):
    """Calculates Shannon entropy (log base 2) for a probability distribution."""
    p_filt = probs[probs > 0]
    if len(p_filt) == 0:
        return 0.0
    return -np.sum(p_filt * np.log(p_filt))

def calculate_disequilibrium(probs):
    """Calculates disequilibrium for a probability distribution."""
    n_states = len(probs)
    if n_states == 0:
        return 0.0
    uniform_prob = 1.0 / n_states

    dis = np.sum((probs - uniform_prob)**2)
    return dis

def calculate_complexity(probs):
    """Calculates complexity as Entropy * Disequilibrium."""
    n_states = len(probs)
    ent = calculate_entropy(probs)
    dis = calculate_disequilibrium(probs)
    return ent * dis

# Generate data for a 2-state system
# We'll use a parameter 'x_param' to define the probability distribution P = [p1, p2].
# Let p1 = x_param, p2 = 1 - x_param.
# - When x_param is near 0, P approaches [0, 1].
# - When x_param = 0.5, P becomes [0.5, 0.5] (uniform distribution).
# - When x_param is near 1, P approaches [1, 0].

x_param_values_2state = np.linspace(0.001, 0.999, 300)  # Avoid exact 0 and 1 for p1
entropies_2state_list = []
disequilibria_2state_list = []
complexities_2state_list = []

for x_param in x_param_values_2state:
    # Define the probability distribution for 2 states
    current_probs_2state = np.array([x_param, 1 - x_param])
    # Normalize to ensure sum is exactly 1, handling potential floating point inaccuracies
    current_probs_2state /= np.sum(current_probs_2state)
    
    entropies_2state_list.append(calculate_entropy(current_probs_2state))
    disequilibria_2state_list.append(calculate_disequilibrium(current_probs_2state))
    complexities_2state_list.append(calculate_complexity(current_probs_2state))

# Find maxima for entropy and complexity
entropies_np = np.array(entropies_2state_list)
complexities_np = np.array(complexities_2state_list)
x_vals = x_param_values_2state

idx_ent_max = int(np.argmax(entropies_np))
idx_comp_max = int(np.argmax(complexities_np))
#find the second max of complexity
idx_comp_max_2 = int(np.argsort(complexities_np)[-2])

x_ent_max, y_ent_max = x_vals[idx_ent_max], entropies_np[idx_ent_max]
x_comp_max, y_comp_max = x_vals[idx_comp_max], complexities_np[idx_comp_max]
x_comp_max_2, y_comp_max_2 = x_vals[idx_comp_max_2], complexities_np[idx_comp_max_2]

# Plotting the results for the 2-state system (all in one figure)
fig2_combined, ax_combined = plt.subplots(1, 1, figsize=(7, 7))
ax_combined.set_xlim(0, 1)
ax_combined.set_ylim(0, max(entropies_2state_list)*1.1)

#don't show y-axis
ax_combined.yaxis.set_visible(False)
# don't show x-axis
ax_combined.xaxis.set_visible(False)

# Plot Entropy, Disequilibrium, and Complexity for 2-state system on the same axes
ax_combined.plot(x_param_values_2state, disequilibria_2state_list, label='Disequilibrium', color = colors[-2], linewidth=5)
ax_combined.plot(x_param_values_2state, entropies_2state_list, label='Entropy', color = colors[0], linewidth=5)
ax_combined.plot(x_param_values_2state, complexities_2state_list, label='Complexity', color = colors[3], linewidth=5)

# Highlight the maxima points for Entropy and Complexity
ax_combined.scatter([x_ent_max], [y_ent_max], color=colors[0], edgecolors='black', s=160, zorder=6)
ax_combined.scatter([x_comp_max], [y_comp_max], color=colors[3], edgecolors='black', s=160, zorder=6)
ax_combined.scatter([x_comp_max_2], [y_comp_max_2], color=colors[3], edgecolors='black', s=160, zorder=6)


#set legen on top
# ax_combined.legend(loc = "upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fontsize=22)
ax_combined.legend(fontsize=22)

#write text as labels in the x axis
ax_combined.text(0.5, -0.05, r'Randomness'+'\n'+r'$[p_1 = 0.5, p_2 = 0.5]$', ha='center', va='center', fontsize=22)
ax_combined.text(0.0, -0.05, r'Determinism'+'\n'+r'$[p_1 = 1, p_2 = 0]$', ha='center', va='center', fontsize=22)
ax_combined.text(1.0, -0.05, r'Determinism'+'\n'+r'$[p_1 = 0, p_2 = 1]$', ha='center', va='center', fontsize=22)


# plt.tight_layout()
plt.savefig("figures/entropy_disequilibrium_complexity_2state.pdf", bbox_inches='tight')
