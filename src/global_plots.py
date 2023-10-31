import os
import json
import pickle as pk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MultipleLocator
import seaborn as sns
from sklearn.linear_model import LinearRegression

import utils as ut
import analysis as an

cwd_path = os.getcwd()


def plot_hom1_heatmap_panel_threshold_active(fixed_dict):
    """ 4x3 figure panel.
    Fixed <k>.
    - row 0: vaccination rate 0.001
    - row 1: vaccination rate 0.005
    - row 2: vaccination rate 0.01
    - row 3: vaccination rate 0.05
    - columns: (0) prevalence, (1) active, (2) vaccinated
    """

    # Build path
    lower_path = 'results/sim_output'
    fullpath = os.path.join(cwd_path, lower_path)
    header = 'global_1'
    
    segment_list = ut.convert_dict_to_seamless_strings(fixed_dict)
    control_strings = ['acf', 'thr', 'var']

    filenames = ut.collect_pickle_filenames_by_inclusion(
        fullpath, 
        header, 
        segment_list,
        )
    
    results_dict = {}

    # Collect results
    for filename in filenames:

        control_dict = ut.extract_parameters_from_string(filename, control_strings)
        threshold = float(control_dict['thr'])
        active = float(control_dict['acf'])
        vac_rate = float(control_dict['var'])

        fullname = os.path.join(cwd_path, lower_path, filename)

        with open(fullname, 'rb') as input_data:
            sim_out_dict = pk.load(input_data)
        
        # This contains global results for all the simulations under this parameter configuration
        print("Hold it!")

        # Extract the specific results we need for this plot

        # Store them into the results dict

    # Prepare panel & plot


def plot_hom2_curve_panel_active():
    """2x3 figure panel.
    Fixed theta.
    - row 0: vaccination rate 0.001
    - row 1: vaccination rate 0.05
    - columns: (0) prevalence, (1) active, (2) vaccinated
    """

    pass


def plot_zea01_heatmap_panel_threshold_active():
    """ 4x4 figure panel.
    Fixed <k>.
    - column 0: vaccination rate 0.001
    - column 1: vaccination rate 0.005
    - column 2: vaccination rate 0.01
    - column 3: vaccination rate 0.05
    - row 0: zealot fraction 0.1
    - row 1: zealot fraction 0.25
    - row 2: zealot fraction 0.5
    - row 3: zealot fraction 0.75
    """

    pass


def main():

    plot_hom1_heatmap_panel_threshold_active()
    plot_hom2_curve_panel_active()
    plot_zea01_heatmap_panel_threshold_active()



if __name__ == "__main__":
    main()

