import os
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.ticker import MaxNLocator

import utils as ut

cwd_path = os.getcwd()
lower_path = 'data/'

# Define countries and states.
countries_states = [
    ('Spain', 'Spain'),
    ('United States', 'Massachusetts'),
    ('India', 'India'),
    ('Japan', 'Tokyo')
]

file_name_list = [
    'Spain_country_level_M_overall_contact_matrix_18.csv',
    'United_States_subnational_Massachusetts_M_overall_contact_matrix_18.csv',
    'India_country_level_M_overall_contact_matrix_18.csv',
    'Japan_subnational_Tokyo-to_M_overall_contact_matrix_18.csv',
]

full_name_list = [os.path.join(cwd_path, lower_path, file_name) for file_name in file_name_list]

contact_matrices = [
    pd.read_csv(full_name, header=None).values
    for full_name in full_name_list
]

vmin = min([matrix.min() for matrix in contact_matrices])
vmax = max([matrix.max() for matrix in contact_matrices])

# Create a 2x2 figure and axes.
fig, ax = plt.subplots(2, 2, figsize=(16, 12))  # Adjusted for a 2x2 grid.

# Iterate over the contact_matrices and plot them.
for i, (contact_matrix, (country, state)) in enumerate(zip(contact_matrices, countries_states)):
    # Find the current axis by indexing into ax, which is now a 2x2 array.
    ax_current = ax[i // 2, i % 2]

    # Plot the current contact matrix.
    im_current = ax_current.imshow(contact_matrix, cmap='viridis', aspect='auto', vmin=vmin, vmax=vmax, origin='lower')
    ax_current.set_title(f'{state}', fontsize=30)
    #ax_current.set_xlabel('Age Group', fontsize=25)
    #ax_current.set_ylabel('Age Group', fontsize=25)
    ax_current.tick_params(axis='both', labelsize=17)

    ax_current.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_current.yaxis.set_major_locator(MaxNLocator(integer=True))

    cbar = plt.colorbar(im_current, ax=ax_current)
    cbar.ax.tick_params(labelsize=17)

ax[0, 0].set_ylabel('age group', fontsize=25)
ax[1, 0].set_ylabel('age group', fontsize=25)
ax[1, 0].set_xlabel('age group', fontsize=25)
ax[1, 1].set_xlabel('age group', fontsize=25)

ax[0, 0].text(0.04, 0.9, r"A", transform=ax[0, 0].transAxes, fontsize=30, color='white', weight="bold")
ax[0, 1].text(0.04, 0.9, r"B", transform=ax[0, 1].transAxes, fontsize=30, color='white', weight="bold")
ax[1, 0].text(0.04, 0.9, r"C", transform=ax[1, 0].transAxes, fontsize=30, color='white', weight="bold")
ax[1, 1].text(0.04, 0.9, r"D", transform=ax[1, 1].transAxes, fontsize=30, color='white', weight="bold")

plt.rcParams.update({'font.size': 15})
plt.rc('axes', labelsize=20)
plt.rcParams['xtick.labelsize'] = 20
plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rcParams['pdf.fonttype'] = 42
plt.tight_layout()

# Save plot
full_path = os.path.join(cwd_path)
base_name = 'contact_'
extension_list = ['pdf', 'png']
if not os.path.exists(full_path):
    os.makedirs(full_path)
for ext in extension_list:
    full_name = os.path.join(full_path, base_name + '.' + ext)
    plt.savefig(full_name, format=ext, bbox_inches='tight')
plt.clf()