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

import utils_old as ut
import analysis as an

path = os.getcwd()

def choose_rk4_plot(dynmod_id, pars):
    if dynmod_id == 'rk4_cghmf':
        plot_rk4_cghmf_time_series(pars)
    elif dynmod_id == 'rk4_hmf':
        plot_rk4_hmf_time_series(pars)

def plot_rk4_cghmf_time_series(pars):
    # Load the results dictionary from a pickle file
    lower_path = 'results'
    file_name = ut.build_file_name(pars)
    pickle_name = file_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    results_dict = pk.load(open(full_path, "rb"))

    # Extract the data from the dictionary
    t = results_dict['time']
    sus = results_dict['sus']
    inf = results_dict['inf']
    rem = results_dict['rem']
    vac = results_dict['vac']
    act = results_dict['act']

    # Create a plot of the data
    fig, ax = plt.subplots(5, 1, sharex=True, figsize=(8, 12))
    labels = ['S', 'I', 'R', 'V', 'A']
    ax[0].set_title('Simulation Results')
    ax[0].plot(t, sus, label='Susceptible')
    ax[1].plot(t, inf, label='Infected')
    ax[2].plot(t, rem, label='Removed')
    ax[3].plot(t, vac, label='Vaccinated')
    ax[4].plot(t, act, label='Active')
    for i in range(5):
        ax[i].set_ylabel(labels[i])
        ax[i].legend()
    ax[-1].set_xlabel('time')

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    file_name = ut.build_file_name(pars)
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    base_name = 'rk4_tseries_oh_' + file_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.show()

def plot_rk4_hmf_time_series(pars):
    # Load the results dictionary from a pickle file
    lower_path = 'results'
    file_name = ut.build_file_name(pars)
    pickle_name = file_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    results_dict = pk.load(open(full_path, "rb"))

    # Extract the data from the dictionary
    t = results_dict['time']
    as_pop_t = results_dict['AS']
    hs_pop_t = results_dict['HS']
    ai_pop_t = results_dict['AI']
    hi_pop_t = results_dict['HI']
    ar_pop_t = results_dict['AR']
    hr_pop_t = results_dict['HR']
    av_pop_t = results_dict['AV']

    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(12, 4))

    labels = ['AS', 'HS']
    ax[0].plot(t, as_pop_t)
    ax[0].plot(t, hs_pop_t)
    ax[0].set_ylabel('Number of Individuals')
    ax[0].legend(labels, loc='best')

    labels = ['AI', 'HI']
    ax[1].plot(t, ai_pop_t)
    ax[1].plot(t, hi_pop_t)
    ax[1].set_ylabel('Number of Individuals')
    ax[1].legend(labels, loc='best')

    labels = ['AR', 'HR', 'AV']
    ax[1].plot(t, ar_pop_t)
    ax[1].plot(t, hr_pop_t)
    ax[1].plot(t, av_pop_t)
    ax[1].set_ylabel('Number of Individuals')
    ax[1].legend(labels, loc='best')

    # Set x label for all subplots
    fig.text(0.5, 0.04, 'Time', ha='center', fontsize=14)

    # Set global font size
    plt.rcParams.update({'font.size': 12})

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    file_name = ut.build_file_name(pars)
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    base_name = 'rk4_tseries_oh_' + file_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.show()

def plot_rk4_heatmap(pars, exclude_keys=['acf', 'thr']):
    # Load the results dictionary from a pickle file
    lower_path = 'results'
    base_name = ut.build_file_name(pars, exclude_keys, collection=True)
    pickle_name = base_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    results_dict = pk.load(open(full_path, "rb"))

    obs_dict = results_dict['final']
    if pars['header']['model'] == 'rk4_hmf':
        obs_dict = results_dict['global']

    con1_id = pars['plot']['control1']
    con2_id = pars['plot']['control2']
    observable = pars['plot']['observable']
    con1_array = results_dict[con1_id]
    con2_array = results_dict[con2_id]

    # Generate the heatmap using results_dict
    X, Y = np.meshgrid(con1_array, con2_array)
    Z = np.zeros((len(con2_array), len(con1_array)))
    for i, t in enumerate(con2_array):
        for j, a in enumerate(con1_array):
            if (t, a) in obs_dict:
                Z[i][j] = obs_dict[(t, a)][observable]
            else:
                Z[i][j] = np.nan
    
    # Obtain homogeneous R0
    R0 = pars['epidemic']['r0']
    alpha = pars['epidemic']['var']
    # Compute HMF opinion-epidemic model R0
    R0_oe = (1.0 - alpha * Y) * R0
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = plt.cm.jet
    cmap.set_bad(color='white')
    im = ax.pcolormesh(Y, X, Z, shading='auto', cmap=cmap, vmin=0, vmax=1)
    ax.set_xlabel(con1_id)
    ax.set_ylabel(con2_id)
    fig.colorbar(im)
    # Draw implicit curve
    #cs = ax.contour(Y, X, R0_oe, levels=[1.0], colors='white')
    #ax.clabel(cs, inline=1, fontsize=10)
    
    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = ut.build_file_name(
        pars, 
        exclude_keys, 
        collection=True, 
        plot=True
    )
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

def plot_outbreak_distribution(pars):
    # Load results
    lower_path = 'results'
    file_name = ut.build_file_name(pars)
    pickle_name = file_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    results_dict = pk.load(open(full_path, "rb"))

    # Obtain N, nsims & tmax
    N = results_dict['pars']['network']['n']
    final_size_array = np.asarray(results_dict['global']['prevalence'])
    final_act_array = np.asarray(results_dict['global']['convinced'])
    final_vac_array = np.asarray(results_dict['global']['vaccinated'])

    # Prepare figure 1: Final size distribution
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 8))

    # Plot curves
    bins = 25
    density = True
    axs[0].hist(
        final_size_array / N, 
        bins=bins, 
        density=density, 
        color='slateblue', 
        label='R'
    )
    axs[1].hist(
        final_act_array / N, 
        bins=bins, 
        density=density, 
        color='cornflowerblue', 
        label='A'
    )
    axs[2].hist(
        final_vac_array / N, 
        bins=bins, 
        density=density, 
        color='mediumseagreen', 
        label='V'
    )

    # Compute SIR analytical solution
    R0 = results_dict['pars']['epidemic']['r0']
    anal_final_size = an.sir_prevalence(R0)
    axs[0].vlines(
        x=anal_final_size, 
        ymin=0, 
        ymax=np.max(np.histogram(final_size_array, bins=bins)[0]), 
        linestyle='dashed', 
        color='black'
    )
    
    # Settings
    fig.suptitle("outbreak distribution")
    axs[0].set_ylabel(r'counts')
    axs[0].set_xlabel(r'value')
    axs[1].set_xlabel(r'value')
    axs[2].set_xlabel(r'value')
    axs[0].legend()
    axs[1].legend()
    axs[2].legend()
    
    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()

    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = 'outbreak_dist_' + file_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.show()

def plot_outbreak_distribution_histogram(pars, exclude_keys=['acf', 'thr']):
    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, exclude_keys, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_mc_heatmap(pars)

    # Load the results into the results_dict variable
    results_dict = pk.load(open(full_name, "rb"))

    # Get control parameter values separatedly
    par1_array = np.array([0.0, 0.2, 0.5, 0.8, 1.0]) #sorted(list(set([key[0] for key in results_dict.keys()])))
    par2_array = np.array([0.0, 0.2, 0.5, 0.8, 1.0]) #sorted(list(set([key[1] for key in results_dict.keys()])))
    
    observable = pars['plot']['observable']

    # Prepare figure 1: Final size distribution
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(14, 8))

    bins = 25
    density = True

    # Generate the heatmap using results_dict
    n = pars['network']['n']
    for i, a in enumerate(par1_array):
        for j, t in enumerate(par2_array):
            if (t, a) in results_dict:
                h = len(par1_array) - (i+1)
                obs_dist = np.array(results_dict[(t, a)][observable]) / n
                axs[h, j].hist(
                    obs_dist, 
                    bins=bins, 
                    density=density, 
                    color='slateblue', 
                    label='R'
                )
                #axs[i, j].set_xlim(0.0, 1.0)
    
    # Settings
    for i in range(5):
        axs[i, 0].set_ylabel("counts", fontsize=20)
        axs[-1, i].set_xlabel(r"$r(\infty)$", fontsize=20)
    
    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = 'dists_' + base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.show()

def plot_outbreak_distribution_histogram_theta_active(
        pars, 
        exclude_keys=['thr', 'acf']
    ):
    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, exclude_keys, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_mc_heatmap(pars)

    # Load the results into the results_dict variable
    results_dict = pk.load(open(full_name, "rb"))

    # Get control parameter values separatedly
    par1_array = np.array([0.0, 0.2, 0.5, 0.8, 1.0]) #sorted(list(set([key[0] for key in results_dict.keys()])))
    par2_array = np.array([0.0, 0.2, 0.5, 0.8, 1.0]) #sorted(list(set([key[1] for key in results_dict.keys()])))
    
    # Prepare figure 1: Final size distribution
    fig1, axs = plt.subplots(nrows=5, ncols=5, figsize=(14, 8))

    r_cutoff = 0.005

    bins = 25
    density = True

    # Generate the heatmap using results_dict
    n = pars['network']['n']
    for i, a in enumerate(par1_array):
        for j, t in enumerate(par2_array):
            if (t, a) in results_dict:
                h = len(par1_array) - (i+1)
                obs_dist = np.array(results_dict[(t, a)]['prevalence']) / n

                mean_obs = np.mean(obs_dist[obs_dist > r_cutoff])

                axs[h, j].axvline(
                    mean_obs, 
                    linestyle='dashed', 
                    color='crimson'
                    )

                axs[h, j].hist(
                    obs_dist, 
                    bins=bins, 
                    density=density, 
                    color='slateblue'
                )
                axs[h, j].legend(
                    [r"$\theta$={0}, $n_A(0)$={1}".format(t, a)], 
                    loc='best', 
                    fontsize=7
                ) 
                axs[h, j].tick_params(axis='x', labelsize=12)  # set font size of x label ticks
                axs[h, j].tick_params(axis='y', labelsize=12)
                #axs[h, j].annotate(f"theta={t}\nalpha={a}", xy=(0.5, 0.9), xycoords='axes fraction', ha='center')
                #axs[i, j].set_xlim(0.0, 1.0)
    
    # Settings
    alpha = pars['epidemic']['var']
    n_Z = pars['opinion']['zef']
    fig1.suptitle(
        r'Prevalence distribution with $\alpha={0}$, $n_Z={1}$'.format(
        alpha, 
        n_Z
        ), 
        fontsize=22,
        y=0.95,
    )

    axs[2, 0].set_ylabel(r"$P(r(\infty))$", fontsize=20)
    axs[-1, 2].set_xlabel(r"$r(\infty)$", fontsize=20)

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    #plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()

    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = 'dists_pre_' + base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

    # Prepare figure 2: Vaccinated distribution
    fig2, axs = plt.subplots(nrows=5, ncols=5, figsize=(14, 8))

    bins = 25
    density = True

    # Generate the heatmap using results_dict
    n = pars['network']['n']
    for i, a in enumerate(par1_array):
        for j, t in enumerate(par2_array):
            if (t, a) in results_dict:
                h = len(par1_array) - (i+1)
                obs_dist = np.array(results_dict[(t, a)]['vaccinated']) / n
                axs[h, j].hist(
                    obs_dist, 
                    bins=bins, 
                    density=density, 
                    color='slateblue'
                )
                axs[h, j].legend(
                    [r"$\theta$={0}, $n_A(0)$={1}".format(t, a)], 
                    loc='best', 
                    fontsize=7
                ) 
                axs[h, j].tick_params(axis='x', labelsize=10)  # set font size of x label ticks
                axs[h, j].tick_params(axis='y', labelsize=10)
                #axs[h, j].annotate(f"theta={t}\nalpha={a}", xy=(0.5, 0.9), xycoords='axes fraction', ha='center')
                #axs[i, j].set_xlim(0.0, 1.0)
    
    # Settings
    fig2.suptitle(
        r'Vaccination coverage distribution with $\alpha={0}$, $n_Z={1}$'.format(
        alpha, 
        n_Z,
        ), 
        fontsize=22,
        y=0.95,
    )
    axs[2, 0].set_ylabel(r"$P(v(\infty))$", fontsize=20)
    axs[-1, 2].set_xlabel(r"$v(\infty)$", fontsize=20)

    plt.tight_layout()

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    #plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = 'dists_vac_' + base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

def plot_outbreak_distribution_histogram_active_zealot(
        pars, 
        exclude_keys=['acf', 'zef']
    ):
    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, exclude_keys, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_mc_heatmap(pars)

    # Load the results into the results_dict variable
    results_dict = pk.load(open(full_name, "rb"))

    # Get control parameter values separatedly
    par1_array = np.array([0.0, 0.2, 0.3, 0.4, 0.5])
    par2_array = np.array([0.0, 0.2, 0.3, 0.4, 0.5])

    # Prepare figure 1: Final size distribution
    fig1, axs = plt.subplots(nrows=5, ncols=5, figsize=(14, 8))

    bins = 25
    density = True

    # Generate the heatmap using results_dict
    n = pars['network']['n']
    for i, z in enumerate(par1_array):
        for j, a in enumerate(par2_array):
            if (a, z) in results_dict:
                h = len(par1_array) - (i+1)
                obs_dist = np.array(results_dict[(a, z)]['prevalence']) / n
                axs[h, j].hist(
                    obs_dist, 
                    bins=bins, 
                    density=density, 
                    color='slateblue'
                )
                axs[h, j].legend(
                    [r"$n_A(0)$={0}, $n_Z$={1}".format(a, z)], 
                    loc='best', 
                    fontsize=7
                )  # add legend
                axs[h, j].tick_params(axis='x', labelsize=12)  # set font size of x label ticks
                axs[h, j].tick_params(axis='y', labelsize=12)
                #axs[h, j].annotate(f"theta={t}\nalpha={a}", xy=(0.5, 0.9), xycoords='axes fraction', ha='center')
                #axs[i, j].set_xlim(0.0, 1.0)
    
    # Settings
    alpha = pars['epidemic']['var']
    theta = pars['opinion']['thr']
    fig1.suptitle(
        r'Prevalence distribution with $\alpha={0}$, $\theta={1}$'.format(
        alpha, 
        theta,
        ), 
        fontsize=22,
        y=0.95,
    )

    axs[2, 0].set_ylabel(r"$P(r(\infty))$", fontsize=20)
    axs[-1, 2].set_xlabel(r"$r(\infty)$", fontsize=20)

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    #plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()

    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = 'dists_pre_' + base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

    # Prepare figure 2: Vaccinated distribution
    fig2, axs = plt.subplots(nrows=5, ncols=5, figsize=(14, 8))

    bins = 25
    density = True

    # Generate the heatmap using results_dict
    n = pars['network']['n']
    for i, z in enumerate(par1_array):
        for j, a in enumerate(par2_array):
            if (a, z) in results_dict:
                h = len(par1_array) - (i+1)
                obs_dist = np.array(results_dict[(a, z)]['vaccinated']) / n
                axs[h, j].hist(
                    obs_dist, 
                    bins=bins, 
                    density=density, 
                    color='slateblue'
                )
                axs[h, j].legend(
                    [r"$n_A(0)$={0}, $n_Z$={1}".format(a, z)], 
                    loc='best', 
                    fontsize=7
                ) 
                axs[h, j].tick_params(axis='x', labelsize=10) # set font size of x label ticks
                axs[h, j].tick_params(axis='y', labelsize=10)
                #axs[h, j].annotate(f"theta={t}\nalpha={a}", xy=(0.5, 0.9), xycoords='axes fraction', ha='center')
                #axs[i, j].set_xlim(0.0, 1.0)
    
    # Settings
    fig2.suptitle(
        r'Vaccination coverage distribution with $\alpha={0}$, $\theta={1}$'.format(
        alpha, 
        theta,
        ),
        fontsize=22,
        y=0.95,
    )
    axs[2, 0].set_ylabel(r"$P(v(\infty))$", fontsize=20)
    axs[-1, 2].set_xlabel(r"$v(\infty)$", fontsize=20)

    plt.tight_layout()

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    #plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = 'dists_vac_' + base_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

def plot_time_series(pars, cutoff_fraction):
    # Load results
    lower_path = 'results'
    file_name = ut.build_file_name(pars)
    pickle_name = file_name + '.pickle'
    full_name = os.path.join(path, lower_path, pickle_name)
    input_data = open(full_name, 'rb')
    results_dict = pk.load(input_data)

    # Obtain N, nsims & tmax
    N = results_dict['as_pop_st'][0][0] \
        + results_dict['hs_pop_st'][0][0] \
            + results_dict['ai_pop_st'][0][0] \
                + results_dict['hi_pop_st'][0][0] \
                    + results_dict['ar_pop_st'][0][0] \
                        + results_dict['hr_pop_st'][0][0] \
                            + results_dict['av_pop_st'][0][0]
    t_array = np.array(results_dict['t_array'][0])
    raw_nsims = len(results_dict['t_array'])
    tmax = len(t_array)
    
    # Rearrange data: get arrays of populations by opinion-health status for every sim & time step  
    raw_actsus_st = results_dict['as_pop_st']
    raw_hessus_st = results_dict['hs_pop_st']
    raw_actinf_st = results_dict['ai_pop_st']
    raw_hesinf_st = results_dict['hi_pop_st']
    raw_actrem_st = results_dict['ar_pop_st']
    raw_hesrem_st = results_dict['hr_pop_st']
    raw_actvac_st = results_dict['av_pop_st']

    # Filter failed outbreaks
    actsus_st = []
    hessus_st = []
    actinf_st = []
    hesinf_st = []
    actrem_st = []
    hesrem_st = []
    actvac_st = []
    prevalence_cutoff = cutoff_fraction * N
    for s in range(raw_nsims):
        total_prevalence = raw_actrem_st[s][-1] + raw_actrem_st[s][-1]
        if total_prevalence >= prevalence_cutoff:
            actsus_st.append(raw_actsus_st[s])
            hessus_st.append(raw_hessus_st[s])
            actinf_st.append(raw_actinf_st[s])
            hesinf_st.append(raw_hesinf_st[s])
            actrem_st.append(raw_actrem_st[s])
            hesrem_st.append(raw_hesrem_st[s])
            actvac_st.append(raw_actvac_st[s])
    np.array(actsus_st)
    np.array(hessus_st)
    np.array(actinf_st)
    np.array(hesinf_st)
    np.array(actrem_st)
    np.array(hesrem_st)
    np.array(actvac_st)

    nsims = len(actsus_st)

    # Obtain stats over simulations for every time step
    z = 1.96
    mean_actsus_t = np.mean(actsus_st, axis=0)
    std_actsus_t = np.std(actsus_st, axis=0)
    l95_actsus_t =  mean_actsus_t - (z * std_actsus_t / np.sqrt(nsims))
    u95_actsus_t = mean_actsus_t + (z * std_actsus_t / np.sqrt(nsims))

    mean_hessus_t = np.mean(hessus_st, axis=0)
    std_hessus_t = np.std(hessus_st, axis=0)
    l95_hessus_t =  mean_hessus_t - (z * std_hessus_t / np.sqrt(nsims))
    u95_hessus_t = mean_hessus_t + (z * std_hessus_t / np.sqrt(nsims))

    mean_actinf_t = np.mean(actinf_st, axis=0)
    std_actinf_t = np.std(actinf_st, axis=0)
    l95_actinf_t =  mean_actinf_t - (z * std_actinf_t / np.sqrt(nsims))
    u95_actinf_t = mean_actinf_t + (z * std_actinf_t / np.sqrt(nsims))

    mean_hesinf_t = np.mean(hesinf_st, axis=0)
    std_hesinf_t = np.std(hesinf_st, axis=0)
    l95_hesinf_t =  mean_hesinf_t - (z * std_hesinf_t / np.sqrt(nsims))
    u95_hesinf_t = mean_hesinf_t + (z * std_hesinf_t / np.sqrt(nsims))

    mean_actrem_t = np.mean(actrem_st, axis=0)
    std_actrem_t = np.std(actrem_st, axis=0)
    l95_actrem_t =  mean_actrem_t - (z * std_actrem_t / np.sqrt(nsims))
    u95_actrem_t = mean_actrem_t + (z * std_actrem_t / np.sqrt(nsims))

    mean_hesrem_t = np.mean(hesrem_st, axis=0)
    std_hesrem_t = np.std(hesrem_st, axis=0)
    l95_hesrem_t =  mean_hesrem_t - (z * std_hesrem_t / np.sqrt(nsims))
    u95_hesrem_t = mean_hesrem_t + (z * std_hesrem_t / np.sqrt(nsims))

    mean_actvac_t = np.mean(actvac_st, axis=0)
    std_actvac_t = np.std(actvac_st, axis=0)
    l95_actvac_t =  mean_actvac_t - (z * std_actvac_t / np.sqrt(nsims))
    u95_actvac_t = mean_actvac_t + (z * std_actvac_t / np.sqrt(nsims))

    # Coarse times series by opinion and health separatedly
    mean_act_t = mean_actsus_t + mean_actinf_t + mean_actrem_t + mean_actvac_t
    mean_hes_t = mean_hessus_t + mean_hesinf_t + mean_hesrem_t
    mean_sus_t = mean_actsus_t + mean_hessus_t
    mean_inf_t = mean_actinf_t + mean_hesinf_t
    mean_rem_t = mean_actrem_t + mean_hesrem_t

    # Prepare figure 1: All (opinion, health) status space time series
    fig = plt.figure(facecolor='w')
    ax = fig.add_subplot(111, axisbelow=True)

    # Plot curves
    ax.plot(
        t_array, 
        mean_actsus_t / N, 
        linestyle='-', 
        color='cornflowerblue', 
        label='AS'
    )
    ax.plot(
        t_array, 
        mean_hessus_t / N, 
        linestyle='-', 
        color='royalblue', 
        label='HS'
    )
    ax.plot(
        t_array, 
        mean_actinf_t / N, 
        linestyle='-', 
        color='lightcoral', 
        label='AI'
    )
    ax.plot(
        t_array, 
        mean_hesinf_t / N, 
        linestyle='-', 
        color='firebrick', 
        label='HI'
    )
    ax.plot(
        t_array, 
        mean_actrem_t / N, 
        linestyle='-', 
        color='slateblue', 
        label='AR'
    )
    ax.plot(
        t_array, 
        mean_hesrem_t / N, 
        linestyle='-', 
        color='indigo', 
        label='HR'
    )
    ax.plot(
        t_array, 
        mean_actvac_t / N, 
        linestyle='-', 
        color='mediumseagreen', 
        label='AV'
    )

    # Settings
    ax.set_ylim(0.0, 1.0)
    ax.set_title("opinion-health status density time series")
    ax.set_ylabel(r'$x(t)$')
    ax.set_xlabel("unit time steps")
    ax.legend()
    
    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = 'tseries_oh_' + file_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.show()

    # Prepare figure 1: All opinion and health space statuses time series
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 8))

    # Plot curves
    axs[0].plot(
        t_array, 
        mean_act_t / N, 
        linestyle='-', 
        color='cornflowerblue', 
        label='A'
    )
    axs[0].plot(
        t_array, 
        mean_hes_t / N, 
        linestyle='-', 
        color='indigo', 
        label='H'
    )
    axs[1].plot(
        t_array, 
        mean_sus_t / N, 
        linestyle='-', 
        color='cornflowerblue', 
        label='S'
    )
    axs[1].plot(
        t_array, 
        mean_inf_t / N, 
        linestyle='-', 
        color='firebrick', 
        label='I'
    )
    axs[1].plot(
        t_array, 
        mean_rem_t / N, 
        linestyle='-', 
        color='slateblue', 
        label='R'
    )
    axs[1].plot(
        t_array, 
        mean_actvac_t / N, 
        linestyle='-', 
        color='mediumseagreen', 
        label='V'
    )

    # Settings
    fig.suptitle("aggregated opinion and health statuses density time series")
    axs[0].set_ylabel(r'$x(t)$')
    axs[0].set_xlabel("unit time steps")
    axs[1].set_xlabel("unit time steps")
    axs[0].legend()
    axs[1].legend()
    
    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = 'tseries_oah_' + file_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.show()

def plot_mc_heatmap(pars, exclude_keys=['acf', 'thr']):
    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, exclude_keys, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_mc_heatmap(pars)

    # Load the results into the results_dict variable
    results_dict = pk.load(open(full_name, "rb"))

    # Get control parameter values separatedly
    par1_array = sorted(list(set([key[0] for key in results_dict.keys()])))
    par2_array = sorted(list(set([key[1] for key in results_dict.keys()])))
    
    observable = pars['plot']['observable']

    # Generate the heatmap using results_dict
    n = pars['network']['n']
    X, Y = np.meshgrid(par1_array, par2_array)
    Z = np.zeros((len(par2_array), len(par1_array)))
    for i, t in enumerate(par1_array):
        for j, a in enumerate(par2_array):
            if (t, a) in results_dict:
                obs_dist = np.array(results_dict[(t, a)][observable]) / n
                filtered_dist = an.filter_outbreaks(obs_dist, pars)
                if len(filtered_dist) == 0:
                    filtered_dist = obs_dist
                stat_dict = an.compute_distribution_statistics(filtered_dist)
                Z[j][i] = stat_dict['avg']
            else:
                Z[j][i] = np.nan
    
    fig, ax = plt.subplots(figsize=(12, 8))
    cmap = 'viridis' #plt.cm.jet
    im = ax.pcolormesh(
        X, 
        Y, 
        Z, 
        shading='auto', 
        cmap=cmap, 
        vmin=0, 
        vmax=np.max(Z)
    )
    con1_id = pars['plot']['control1']
    con2_id = pars['plot']['control2']
    con1_label = ut.get_full_label(con1_id)
    con2_label = ut.get_full_label(con2_id)
    ax.set_xlabel(con1_label, fontsize=20)
    ax.set_ylabel(con2_label, fontsize=20)

    # Set ticks
    ax.tick_params(axis='both', labelsize=12)
    
    ax.set_aspect('equal')

    fig.colorbar(im)
    
    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = ut.build_file_name(pars, 
                                   exclude_keys, 
                                   collection=True, 
                                   plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

def plot_mc_heatmap_theta_active(pars, exclude_keys=['thr', 'acf']):
    """ Heatmap 1x2 for prevalence and vaccinated in (theta, active)-space
    for fixed vaccination rate, fixed zealot, given network model. 
    """
    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, exclude_keys, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_mc_heatmap(pars)

    # Load the results into the results_dict variable
    results_dict = pk.load(open(full_name, "rb"))

    # Get control parameter values separatedly
    par1_array = sorted(list(set([key[0] for key in results_dict.keys()])))
    par2_array = sorted(list(set([key[1] for key in results_dict.keys()])))

    # Generate the heatmap using results_dict
    n = pars['network']['n']
    X, Y = np.meshgrid(par1_array, par2_array)
    R = np.zeros((len(par2_array), len(par1_array)))
    V = np.zeros((len(par2_array), len(par1_array)))
    for i, t in enumerate(par1_array):
        for j, a in enumerate(par2_array):
            if (t, a) in results_dict:
                pre_dist = np.array(results_dict[(t, a)]['prevalence']) / n
                vac_dist = np.array(results_dict[(t, a)]['vaccinated']) / n
                filtered_pre_dist = an.filter_outbreaks(pre_dist, pars)
                filtered_vac_dist = an.filter_outbreaks(vac_dist, pars)
                if len(filtered_pre_dist) == 0:
                    filtered_pre_dist = pre_dist
                pre_stat_dict = \
                    an.compute_distribution_statistics(filtered_pre_dist)
                if len(filtered_vac_dist) == 0:
                    filtered_vac_dist = vac_dist
                vac_stat_dict = \
                    an.compute_distribution_statistics(filtered_vac_dist)
                R[j][i] = pre_stat_dict['avg']
                V[j][i] = vac_stat_dict['avg']
            else:
                R[j][i] = np.nan
                V[j][i] = np.nan

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    cmap = 'viridis' #plt.cm.jet
    im0 = ax[0].pcolormesh(
        X, 
        Y, 
        R, 
        shading='auto', 
        cmap=cmap, 
        vmin=0, 
        vmax=np.max(R)
    )
    im1 = ax[1].pcolormesh(
        X, 
        Y, 
        V, 
        shading='auto', 
        cmap=cmap, 
        vmin=0, 
        vmax=np.max(V)
    )

    fig.colorbar(im0, ax=ax[0], shrink=0.55)
    fig.colorbar(im1, ax=ax[1], shrink=0.55)

    alpha = pars['epidemic']['var']
    n_Z = pars['opinion']['zef']
    fig.suptitle(
        r'Epidemic impact with $\alpha={0}$, $n_Z={1}$'.format(alpha, n_Z), 
        fontsize=22,
        y=0.8,
    )
    ax[0].set_title(r'$r(\infty)$', fontsize=20)
    ax[0].set_xlabel(r'$\theta$', fontsize=20)
    ax[0].set_ylabel(r'$n_A(0)$', fontsize=20)
    ax[1].set_title(r'$v(\infty)$', fontsize=20)
    ax[1].set_xlabel(r'$\theta$', fontsize=20)
    ax[1].set_ylabel(r'$n_A(0)$', fontsize=20)

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = ut.build_file_name(pars, 
                                   exclude_keys, 
                                   collection=True, 
                                   plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

def plot_mc_heatmap_active_zealot(pars, exclude_keys=['acf', 'zef']):
    """ Heatmap 1x2 for prevalence and vaccinated in (active, zealot)-space
    for fixed vaccination rate, fixed threshold, given network model. 
    """
    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, exclude_keys, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_mc_heatmap(pars)

    # Load the results into the results_dict variable
    results_dict = pk.load(open(full_name, "rb"))

    # Get control parameter values separatedly
    par1_array = sorted(list(set([key[0] for key in results_dict.keys()])))
    par2_array = sorted(list(set([key[1] for key in results_dict.keys()])))

    # Generate the heatmap using results_dict
    n = pars['network']['n']
    X, Y = np.meshgrid(par1_array, par2_array)
    R = np.zeros((len(par2_array), len(par1_array)))
    V = np.zeros((len(par2_array), len(par1_array)))
    for i, a in enumerate(par1_array):
        for j, z in enumerate(par2_array):
            if (a, z) in results_dict:
                pre_dist = np.array(results_dict[(a, z)]['prevalence']) / n
                vac_dist = np.array(results_dict[(a, z)]['vaccinated']) / n
                filtered_pre_dist = an.filter_outbreaks(pre_dist, pars)
                filtered_vac_dist = an.filter_outbreaks(vac_dist, pars)
                if len(filtered_pre_dist) == 0:
                    filtered_pre_dist = pre_dist
                pre_stat_dict = \
                    an.compute_distribution_statistics(filtered_pre_dist)
                if len(filtered_vac_dist) == 0:
                    filtered_vac_dist = vac_dist
                vac_stat_dict = \
                    an.compute_distribution_statistics(filtered_vac_dist)
                R[j][i] = pre_stat_dict['avg']
                V[j][i] = vac_stat_dict['avg']
            else:
                R[j][i] = np.nan
                V[j][i] = np.nan
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))
    cmap = 'viridis' #plt.cm.jet
    im0 = ax[0].pcolormesh(
        X, 
        Y, 
        R, 
        shading='auto', 
        cmap=cmap, 
        vmin=0.0, 
        vmax=np.nanmax(R)
    )
    im1 = ax[1].pcolormesh(
        X, 
        Y, 
        V, 
        shading='auto', 
        cmap=cmap, 
        vmin=0.0, 
        vmax=np.nanmax(V)
    )

    fig.colorbar(im0, ax=ax[0], shrink=0.65)
    fig.colorbar(im1, ax=ax[1], shrink=0.65)

    alpha = pars['epidemic']['var']
    theta = pars['opinion']['thr']
    fig.suptitle(
        r'Epidemic impact with $\alpha={0}$, $\theta={1}$'.format(
        alpha, theta), 
        fontsize=22,
        y=0.8,
    )
    ax[0].set_title(r'$r(\infty)$', fontsize=20)
    ax[0].set_xlabel(r'$n_A(0)$', fontsize=20)
    ax[0].set_ylabel(r'$n_Z$', fontsize=20)
    ax[1].set_title(r'$v(\infty)$', fontsize=20)
    ax[1].set_xlabel(r'$n_A(0)$', fontsize=20)
    ax[1].set_ylabel(r'$n_Z$', fontsize=20)

    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = ut.build_file_name(pars, 
                                   exclude_keys, 
                                   collection=True, 
                                   plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

def plot_mc_section_curves(pars, exclude_keys = ['acf', 'zef']):
    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, exclude_keys, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_mc_heatmap(pars)

    # Load the results into the results_dict variable
    results_dict = pk.load(open(full_name, "rb"))

    # Get control parameter values separatedly
    n = pars['network']['n']
    par1_array = sorted(list(set([key[0] for key in results_dict.keys()])))
    par2_array = sorted(list(set([key[1] for key in results_dict.keys()])))
    
    observable = pars['plot']['observable']

    # Prepare figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Build curves
    for par2 in par2_array:
        main_value = []
        for par1 in par1_array:
            if (par1, par2) in results_dict:
                obs_dist = \
                    np.array(results_dict[(par1, par2)][observable]) / n
                filtered_dist = an.filter_outbreaks(obs_dist, pars)
                if len(filtered_dist) == 0:
                    filtered_dist = obs_dist
                stat_dict = an.compute_distribution_statistics(filtered_dist)
            main_value.append(stat_dict['avg'])
        ax.plot(par1_array, main_value, label='{0}'.format(par2))

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = 'curve_' + ut.build_file_name(pars, 
                                              exclude_keys, 
                                              collection=True, 
                                              plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

def plot_agent_statuses(pars):
    # Load agent data
    pass

def plot_degree_distribution(pars):
    """ What is the degree distribution of active agents at t=0?
        What is the degree distribution of zealots?
    """
    # Load agent data
    
    pass

def plot_zealot_distribution(pars):
    """ How many zealot neighbors (density) do hesitant susceptible agents have at t=0?
        How many active agents (density) do hesitant susceptible agents have at t=0? 
        How many vaccinated (density) do hesitant susceptible agents have at t=0?
    """
    # Load results
    lower_path = 'results'
    file_name = ut.build_file_name(pars)
    pickle_name = file_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    results_dict = pk.load(open(full_path, "rb"))

    # Obtain N, nsims & tmax
    N = results_dict['pars']['network']['n']
    degree_sa = np.asarray(results_dict['agent_w1']['degree'])
    initial_as_w1_sa = \
        np.asarray(results_dict['agent_w1']['initial_active_susceptible'])
    initial_av_w1_sa = \
        np.asarray(results_dict['agent_w1']['initial_vaccinated'])
    zealots_w1_sa = np.asarray(results_dict['agent_w1']['zealots'])
    final_as_w1_sa = \
        np.asarray(results_dict['agent_w1']['final_active_susceptible'])
    final_av_w1_sa = np.asarray(results_dict['agent_w1']['final_vaccinated'])
    final_prev_w1_sa = \
        np.asarray(results_dict['agent_w1']['final_prevalence'])

    initial_as_w1_density_sa = initial_as_w1_sa / degree_sa
    initial_av_w1_density_sa = initial_av_w1_sa / degree_sa
    zealot_density_sa = zealots_w1_sa / degree_sa
    final_as_w1_density_sa = final_as_w1_sa / degree_sa
    final_av_w1_density_sa = final_av_w1_sa / degree_sa
    final_prev_w1_density_sa = final_prev_w1_sa / degree_sa

    # Flatten the matrix into a 1D array
    initial_w1_as_density_flat = initial_as_w1_density_sa.flatten()
    initial_w1_av_density_flat = initial_av_w1_density_sa.flatten()
    zealot_density_flat = zealot_density_sa.flatten()
    final_as_w1_density_flat = final_as_w1_density_sa.flatten()
    final_av_w1_density_flat = final_av_w1_density_sa.flatten()
    final_prev_w1_density_flat = final_prev_w1_density_sa.flatten()

    # Create a new figure and axis objects
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # Specify the number of bins and density parameter
    num_bins = 30
    density = True

    # Plot the histograms on the axis objects
    axs[0, 0].hist(initial_w1_as_density_flat, bins=num_bins, density=density)
    axs[0, 1].hist(initial_w1_av_density_flat, bins=num_bins, density=density)
    axs[0, 2].hist(zealot_density_flat, bins=num_bins, density=density)
    axs[1, 0].hist(final_as_w1_density_flat, bins=num_bins, density=density)
    axs[1, 1].hist(final_av_w1_density_flat, bins=num_bins, density=density)
    axs[1, 2].hist(final_prev_w1_density_flat, bins=num_bins, density=density)

    # Add labels and titles
    axs[0, 0].set_xlabel('Initial AS Density')
    axs[0, 0].set_ylabel('Count')
    #axs[0, 0].set_title('Histogram of Initial AS Density')
    axs[0, 1].set_xlabel('Initial AV Density')
    #axs[0, 1].set_title('Histogram of Initial AV Density')
    axs[0, 2].set_xlabel('Zealot Density')
    #axs[0, 2].set_title('Histogram of Zealot Density')
    axs[1, 0].set_xlabel('Final AS Density')
    axs[1, 0].set_ylabel('Count')
    #axs[1, 0].set_title('Histogram of Final AS Density')
    axs[1, 1].set_xlabel('Final AV Density')
    #axs[1, 1].set_title('Histogram of Final AV Density')
    axs[1, 2].set_xlabel('Final Prev Density')
    #axs[1, 2].set_title('Histogram of Final Prev Density')

    # Adjust the spacing between the plots
    fig.tight_layout()

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = \
        'neigh_dist_' + ut.build_file_name(pars, collection=True, plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    
    plt.show()

def plot_zealot_cluster(pars):
    """ What is the size distribution of zealot clusters?
        What is the size distribution of active clusters?
        What is the size distribution of vaccinated clusters?
    """
    # Load results
    lower_path = 'results'
    file_name = ut.build_file_name(pars)
    pickle_name = file_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    results_dict = pk.load(open(full_path, "rb"))

    # Obtain N, nsims & tmax
    N = results_dict['pars']['network']['n']
    cluster_as_w1_sa = np.asarray(results_dict['cluster_w1']['as_cluster'])
    cluster_hs_w1_sa = np.asarray(results_dict['cluster_w1']['hs_cluster'])
    cluster_ai_w1_sa = np.asarray(results_dict['cluster_w1']['ai_cluster'])
    cluster_hi_w1_sa = np.asarray(results_dict['cluster_w1']['hi_cluster'])
    cluster_ar_w1_sa = np.asarray(results_dict['cluster_w1']['ar_cluster'])
    cluster_hr_w1_sa = np.asarray(results_dict['cluster_w1']['hr_cluster'])
    cluster_av_w1_sa = np.asarray(results_dict['cluster_w1']['av_cluster'])
    cluster_ze_w1_sa = np.asarray(results_dict['cluster_w1']['ze_cluster'])

    cluster_as_w1_flat = cluster_as_w1_sa.flatten()
    cluster_hs_w1_flat = cluster_hs_w1_sa.flatten()
    cluster_ai_w1_flat = cluster_ai_w1_sa.flatten()
    cluster_hi_w1_flat = cluster_hi_w1_sa.flatten()
    cluster_ar_w1_flat = cluster_ar_w1_sa.flatten()
    cluster_hr_w1_flat = cluster_hr_w1_sa.flatten()
    cluster_av_w1_flat = cluster_av_w1_sa.flatten()
    cluster_ze_w1_flat = cluster_ze_w1_sa.flatten()

    # Specify the number of bins and density parameter
    num_bins = 30
    density = True

    # Create a new figure and axis objects
    fig, axs = plt.subplots(4, 2, figsize=(12, 8))

    # Plot the histograms on the axis objects
    axs[0, 0].hist(cluster_as_w1_flat, bins=num_bins, density=density)
    axs[0, 1].hist(cluster_hs_w1_flat, bins=num_bins, density=density)
    axs[1, 0].hist(cluster_ai_w1_flat, bins=num_bins, density=density)
    axs[1, 1].hist(cluster_hi_w1_flat, bins=num_bins, density=density)
    axs[2, 0].hist(cluster_ar_w1_flat, bins=num_bins, density=density)
    axs[2, 1].hist(cluster_hr_w1_flat, bins=num_bins, density=density)
    axs[3, 0].hist(cluster_av_w1_flat, bins=num_bins, density=density)
    axs[3, 1].hist(cluster_ze_w1_flat, bins=num_bins, density=density)

    # Add labels and titles
    axs[0, 0].set_ylabel('count', fontsize=20)
    axs[1, 0].set_ylabel('count', fontsize=20)
    axs[2, 0].set_ylabel('count', fontsize=20)
    axs[3, 0].set_ylabel('count', fontsize=20)
    axs[3, 0].set_xlabel('size', fontsize=20)
    axs[3, 1].set_xlabel('size', fontsize=20)

    # Add legends
    for ax in axs.flatten():
        ax.legend()

    # Adjust the spacing between the plots
    fig.tight_layout()

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = 'cluster_' + ut.build_file_name(pars, 
                                                collection=True, 
                                                plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

    plt.show()

def plot_us_state_outbreak_distribution(pars):
    # Load results
    lower_path = 'results'
    file_name = ut.build_file_name(pars)
    pickle_name = file_name + '.pickle'
    full_path = os.path.join(path, lower_path, pickle_name)
    results_dict = pk.load(open(full_path, "rb"))

    # Load results
    N = results_dict['pars']['network']['n']
    pre_w1 = np.asarray(results_dict['global_w1']['prevalence']) / N
    con_w1 = np.asarray(results_dict['global_w1']['convinced']) / N
    vac_w1 = np.asarray(results_dict['global_w1']['vaccinated']) / N
    pre_w2 = np.asarray(results_dict['global_w2']['prevalence']) / N - pre_w1
    con_w2 = np.asarray(results_dict['global_w2']['convinced']) / N - con_w1
    vac_w2 = np.asarray(results_dict['global_w2']['vaccinated']) / N - vac_w1
    
    # Prepare figure: Observable distributions
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(16, 12))

    # Plot curves for the 1st wave
    bins = 50
    density = True
    axs[0,0].hist(
        pre_w1, 
        bins=bins, 
        density=density, 
        color='lightcoral', 
    )
    axs[1,0].hist(
        vac_w1, 
        bins=bins, 
        density=density, 
        color='mediumseagreen', 
    )
    axs[2,0].hist(
        pre_w1 + vac_w1, 
        bins=bins, 
        density=density, 
        color='cornflowerblue', 
    )
    axs[3,0].hist(
        con_w1, 
        bins=bins, 
        density=density, 
        color='slateblue', 
    )

    # Plot curves for the 1st wave
    bins = 50
    density = True
    axs[0,1].hist(
        pre_w2, 
        bins=bins, 
        density=density, 
        color='lightcoral', 
    )
    axs[1,1].hist(
        vac_w2, 
        bins=bins, 
        density=density, 
        color='mediumseagreen', 
    )
    axs[2,1].hist(
        pre_w1 + pre_w2 + + vac_w1 + vac_w2, 
        bins=bins, 
        density=density, 
        color='cornflowerblue', 
    )
    axs[3,1].hist(
        con_w2, 
        bins=bins, 
        density=density, 
        color='slateblue', 
    )

    # Compute SIR analytical solution
    R0 = results_dict['pars']['epidemic']['r0']
    R02 = 2.0 * R0
    anal_final_size = an.sir_prevalence(R0)
    axs[0,0].vlines(
        x=anal_final_size, 
        ymin=0, 
        ymax=np.max(np.histogram(pre_w1, bins=bins, density=density)[0]), 
        linestyle='dashed', 
        color='crimson',
        label='classical SIR',
    )
    HIT = 1.0 - 1.0 / R0
    axs[2,0].vlines(
        x=HIT,
        ymin=0,
        ymax=np.max(np.histogram(pre_w1, bins=bins, density=density)[0]),
        linestyle='dashed',
        color='limegreen',
        label=r'HIT for $R_0=${0}'.format(R0),
    )
    axs[2,1].vlines(
        x=1.0-1.0/R02,
        ymin=0,
        ymax=np.max(np.histogram(pre_w1, bins=bins, density=density)[0]),
        linestyle='dashed',
        color='limegreen',
        label=r'HIT for $R_0=${0}'.format(R02),
    )

    # Settings
    state_id = pars['header']['usa_id']
    fig.suptitle(
        "outbreak distribution for {0}".format(state_id), 
        fontsize=30
    )
    axs[0,0].set_title(r'1st wave, $R_0$={0}'.format(R0), fontsize=25)
    axs[0,1].set_title(r'2nd wave, $R_0$={0}'.format(R02), fontsize=25)
    axs[0,0].set_ylabel(r'$P(r(\infty))$', fontsize=25)
    axs[1,0].set_ylabel(r'$P(v(\infty))$', fontsize=25)
    axs[2,0].set_ylabel(r'$P(r(\infty)+v(\infty))$', fontsize=25)
    axs[3,0].set_ylabel(r'$P(n_A(\infty))$', fontsize=25)
    axs[0,0].set_xlabel(r'$r(\infty)$', fontsize=25)
    axs[0,1].set_xlabel(r'$r(\infty)$', fontsize=25)
    axs[1,0].set_xlabel(r'$v(\infty)$', fontsize=25)
    axs[1,1].set_xlabel(r'$v(\infty)$', fontsize=25)
    axs[2,0].set_xlabel(r'$r(\infty)+v(\infty)$', fontsize=25)
    axs[2,1].set_xlabel(r'$r(\infty)_{tot}+v(\infty)_{tot}$', fontsize=25)
    axs[3,0].set_xlabel(r'$n_A(\infty)$', fontsize=25)
    axs[3,1].set_xlabel(r'$n_A(\infty)$', fontsize=25)

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()

    # Save plot
    full_path = os.path.join(path, lower_path)
    base_name = 'outbreak_dist_' + file_name
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.show()


def plot_bar_us_states_immunized(pars):
    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_us_states(pars)

    # Load results
    state_raw_dict = pk.load(open(full_name, "rb"))

    # Collect filtered results
    N = pars['network']['n']
    prevalence_cutoff = 0.025
    state_stats_dict = {}
    avg_pre_w1_list = []
    l95_pre_w1_list = []
    u95_pre_w1_list = []
    avg_vac_w1_list = []
    l95_vac_w1_list = []
    u95_vac_w1_list = []
    avg_pre_w2_list = []
    l95_pre_w2_list = []
    u95_pre_w2_list = []
    avg_vac_w2_list = []
    l95_vac_w2_list = []
    u95_vac_w2_list = []
    state_code_list = []

    for state in state_raw_dict.keys():
        print("{0}".format(state))
        state_code_list.append(ut.get_state_code(state))

        pre_w1_dist = np.array(state_raw_dict[state]['w1']['prevalence']) / N
        vac_w1_dist = np.array(state_raw_dict[state]['w1']['vaccinated']) / N

        fil_pre_w1_dist = pre_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_pre_w1_dist.size == 0:
            fil_pre_w1_dist = pre_w1_dist
        fil_vac_w1_dist = vac_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_vac_w1_dist.size == 0:
            fil_vac_w1_dist = vac_w1_dist

        pre_w1_stat_dict = an.compute_distribution_statistics(fil_pre_w1_dist)
        vac_w1_stat_dict = an.compute_distribution_statistics(fil_vac_w1_dist)

        state_stats_dict = {state: {'w1': {}, 'w2': {}}}
        state_stats_dict[state]['w1']['prevalence'] = pre_w1_stat_dict
        state_stats_dict[state]['w1']['vaccinated'] = vac_w1_stat_dict

        avg_pre_w1_list.append(pre_w1_stat_dict['avg'])
        l95_pre_w1_list.append(pre_w1_stat_dict['l95'])
        u95_pre_w1_list.append(pre_w1_stat_dict['u95'])
        avg_vac_w1_list.append(vac_w1_stat_dict['avg'])
        l95_vac_w1_list.append(vac_w1_stat_dict['l95'])
        u95_vac_w1_list.append(vac_w1_stat_dict['u95'])

        pre_w2_dist = np.array(state_raw_dict[state]['w2']['prevalence']) / N
        vac_w2_dist = np.array(state_raw_dict[state]['w2']['vaccinated']) / N

        fil_pre_w2_dist = pre_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_pre_w2_dist.size == 0:
            fil_pre_w2_dist = pre_w2_dist
        fil_vac_w2_dist = vac_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_vac_w2_dist.size == 0:
            fil_vac_w2_dist = vac_w2_dist
    
        pre_w2_stat_dict = an.compute_distribution_statistics(fil_pre_w2_dist)
        vac_w2_stat_dict = an.compute_distribution_statistics(fil_vac_w2_dist)
    
        state_stats_dict[state]['w2']['prevalence'] = pre_w2_stat_dict
        state_stats_dict[state]['w2']['vaccinated'] = vac_w2_stat_dict
    
        avg_pre_w2_list.append(pre_w2_stat_dict['avg'])
        l95_pre_w2_list.append(pre_w2_stat_dict['l95'])
        u95_pre_w2_list.append(pre_w2_stat_dict['u95'])
        avg_vac_w2_list.append(vac_w2_stat_dict['avg'])
        l95_vac_w2_list.append(vac_w2_stat_dict['l95'])
        u95_vac_w2_list.append(vac_w2_stat_dict['u95'])

    # create the figure and axis objects
    fig, ax = plt.subplots(figsize=(17, 6))

    # create the bar plot
    width = 0.4
    ax.bar(np.arange(len(state_code_list)), avg_vac_w1_list, width, color='limegreen')
    ax.bar(np.arange(len(state_code_list)), avg_pre_w1_list, width, color='lightcoral', bottom=avg_vac_w1_list)

    R0 = pars['epidemic']['r0']
    R02 = 2.0 * R0
    HIT = 1.0 - 1.0 / R0
    ax.axhline(HIT, linestyle='dashed', color='navy', label=r'$R_0$={0} HIT'.format(R0))
    HIT2 = 1.0 - 1.0 / R02
    ax.axhline(HIT2, linestyle='dashed', color='navy', label=r'$R_0$={0} HIT'.format(R02))

    # set the axis and title labels
    ax.set_title('Immunized after the 1st wave', fontsize=30)
    ax.set_ylabel(r'$v(\infty)+r(\infty)$', fontsize=25)
    ax.set_xticks(np.arange(len(state_code_list)))
    ax.set_xticklabels(state_code_list, rotation=90) # rotate state labels
    ax.legend()
   
    # adjust the spacing between the subplots
    fig.subplots_adjust(wspace=0.4)

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = 'bar_' + ut.build_file_name(pars, 
                                                collection=True, 
                                                plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

    plt.show()

def plot_bar_us_states_thresholds(pars):
    # Load data
    vaccination_results = ut.read_vaccination_data()

    nentries = len(vaccination_results.keys())
    state_code = []
    already = np.zeros(nentries)
    soon = np.zeros(nentries)
    someone = np.zeros(nentries)
    majority = np.zeros(nentries)
    never = np.zeros(nentries)
    avg_thresholds = np.zeros(nentries)

    for state, i in zip(vaccination_results.keys(), range(nentries)):
        print("{0}".format(state))
        state_code.append(ut.get_state_code(state))
        already[i] = vaccination_results[state]['already']
        soon[i] = vaccination_results[state]['soon']
        someone[i] = vaccination_results[state]['someone']
        majority[i] = vaccination_results[state]['majority']
        never[i] = vaccination_results[state]['never']
        avg_thresholds[i] = \
            ut.compute_average_theshold(vaccination_results[state], pars)

    # create the figure and axis objects
    fig, ax = plt.subplots(figsize=(17, 6))

    # create the bar plot
    width = 0.4
    bar1 = ax.bar(
        np.arange(len(state_code)), 
        already, 
        width, 
        color='limegreen'
        )
    ax.bar(
        np.arange(len(state_code)), 
        soon, 
        width, 
        color='turquoise', 
        bottom=already
    )
    ax.bar(np.arange(
        len(state_code)), 
        someone, 
        width, 
        color='lightseagreen', 
        bottom=np.add(soon, already)
    )
    ax.bar(
        np.arange(len(state_code)), 
        majority, 
        width, 
        color='slateblue', 
        bottom=np.add(someone, np.add(soon, already))
    )
    ax.bar(
        np.arange(len(state_code)), 
        never, 
        width, 
        color='darkblue', 
        bottom=np.add(
        majority, 
        np.add(someone, np.add(soon, already))
        )
    )

    # create the right axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())

    x_start = np.array([plt.getp(item, 'x') for item in bar1])
    x_end = x_start+[plt.getp(item, 'width') for item in bar1]

    ax2.hlines(avg_thresholds, x_start, x_end, linestyles='dashed', label=r'$\langle\theta\rangle$')

    R0 = pars['epidemic']['r0']
    HIT = 1.0 - 1.0 / R0
    ax2.axhline(
        HIT, 
        color='crimson', 
        linestyle='dashed', 
        linewidth=1
    )

    # set the axis and title labels
    ax.set_title('Vaccination thresholds distribution by state', fontsize=30)
    ax.set_ylabel(r'population fraction', fontsize=25)
    ax.set_xticks(np.arange(len(state_code)))
    ax.set_xticklabels(state_code, rotation=90) # rotate state labels
    ax2.set_ylabel(r'$\langle\theta\rangle$', fontsize=25)
    ax2.legend()
   
    # adjust the spacing between the subplots
    fig.subplots_adjust(wspace=0.4)

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = 'bar_vaccination_thresholds'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

    plt.show()


def plot_scatter_us_states_outbreaks(pars):
    # Load data
    vaccination_results = ut.read_vaccination_data()

    nentries = len(vaccination_results.keys())
    state_code = []
    already = np.zeros(nentries)
    soon = np.zeros(nentries)
    someone = np.zeros(nentries)
    majority = np.zeros(nentries)
    never = np.zeros(nentries)
    avg_thresholds = np.zeros(nentries)

    for state, i in zip(vaccination_results.keys(), range(nentries)):
        print("{0}".format(state))
        state_code.append(ut.get_state_code(state))
        already[i] = vaccination_results[state]['already']
        soon[i] = vaccination_results[state]['soon']
        someone[i] = vaccination_results[state]['someone']
        majority[i] = vaccination_results[state]['majority']
        never[i] = vaccination_results[state]['never']
        avg_thresholds[i] = \
            ut.compute_average_theshold(vaccination_results[state], pars)

    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_us_states_scatter(pars)

    # Load results
    state_raw_dict = pk.load(open(full_name, "rb"))

    # Read vaccination data
    vac_file_name = os.path.join(path, 'data', 'vaccination_data.json')
    with open(vac_file_name, 'r') as f:
        vac_data = json.load(f)

    # Collect filtered results
    N = pars['network']['n']
    prevalence_cutoff = 0.025
    state_stats_dict = {}
    avg_pre_w1_list = []
    l95_pre_w1_list = []
    u95_pre_w1_list = []
    avg_con_w1_list = []
    l95_con_w1_list = []
    u95_con_w1_list = []
    avg_vac_w1_list = []
    l95_vac_w1_list = []
    u95_vac_w1_list = []
    avg_ent_w1_list = []
    l95_ent_w1_list = []
    u95_ent_w1_list = []
    avg_pre_w2_list = []
    l95_pre_w2_list = []
    u95_pre_w2_list = []
    avg_con_w2_list = []
    l95_con_w2_list = []
    u95_con_w2_list = []
    avg_vac_w2_list = []
    l95_vac_w2_list = []
    u95_vac_w2_list = []
    avg_ent_w2_list = []
    l95_ent_w2_list = []
    u95_ent_w2_list = []
    state_code_list = []

    for state in vaccination_results.keys():
        print("{0}".format(state))

        state_code_list.append(ut.get_state_code(state))
        pre_w1_dist = np.array(state_raw_dict[state]['w1']['prevalence']) / N
        con_w1_dist = np.array(state_raw_dict[state]['w1']['convinced']) / N
        vac_w1_dist = np.array(state_raw_dict[state]['w1']['vaccinated']) / N
        ent_w1_dist = np.array(state_raw_dict[state]['w1']['end_time'])

        fil_pre_w1_dist = pre_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_pre_w1_dist.size == 0:
            fil_pre_w1_dist = pre_w1_dist
        fil_con_w1_dist = con_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_con_w1_dist.size == 0:
            fil_con_w1_dist = con_w1_dist
        fil_vac_w1_dist = vac_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_vac_w1_dist.size == 0:
            fil_vac_w1_dist = vac_w1_dist
        fil_ent_w1_dist = ent_w1_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_ent_w1_dist.size == 0:
            fil_ent_w1_dist = ent_w1_dist

        pre_w1_stat_dict = an.compute_distribution_statistics(fil_pre_w1_dist)
        con_w1_stat_dict = an.compute_distribution_statistics(fil_con_w1_dist)
        vac_w1_stat_dict = an.compute_distribution_statistics(fil_vac_w1_dist)
        ent_w1_stat_dict = an.compute_distribution_statistics(fil_ent_w1_dist)
        
        state_stats_dict = {state: {'w1': {}, 'w2': {}}}
        state_stats_dict[state]['w1']['prevalence'] = pre_w1_stat_dict
        state_stats_dict[state]['w1']['convinced'] = con_w1_stat_dict
        state_stats_dict[state]['w1']['vaccinated'] = vac_w1_stat_dict
        state_stats_dict[state]['w1']['end_time'] = ent_w1_stat_dict

        avg_pre_w1_list.append(pre_w1_stat_dict['avg'])
        l95_pre_w1_list.append(pre_w1_stat_dict['l95'])
        u95_pre_w1_list.append(pre_w1_stat_dict['u95'])
        avg_con_w1_list.append(con_w1_stat_dict['avg'])
        l95_con_w1_list.append(con_w1_stat_dict['l95'])
        u95_con_w1_list.append(con_w1_stat_dict['u95'])
        avg_vac_w1_list.append(vac_w1_stat_dict['avg'])
        l95_vac_w1_list.append(vac_w1_stat_dict['l95'])
        u95_vac_w1_list.append(vac_w1_stat_dict['u95'])
        avg_ent_w1_list.append(ent_w1_stat_dict['avg'])
        l95_ent_w1_list.append(ent_w1_stat_dict['l95'])
        u95_ent_w1_list.append(ent_w1_stat_dict['u95'])

        pre_w2_dist = np.array(state_raw_dict[state]['w2']['prevalence']) / N
        con_w2_dist = np.array(state_raw_dict[state]['w2']['convinced']) / N
        vac_w2_dist = np.array(state_raw_dict[state]['w2']['vaccinated']) / N
        ent_w2_dist = np.array(state_raw_dict[state]['w2']['end_time'])

        fil_pre_w2_dist = pre_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_pre_w2_dist.size == 0:
            fil_pre_w2_dist = pre_w2_dist
        fil_con_w2_dist = con_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_con_w2_dist.size == 0:
            fil_con_w2_dist = con_w2_dist
        fil_vac_w2_dist = vac_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_vac_w2_dist.size == 0:
            fil_vac_w2_dist = vac_w2_dist
        fil_ent_w2_dist = ent_w2_dist[pre_w1_dist>=prevalence_cutoff]
        if fil_ent_w2_dist.size == 0:
            fil_ent_w2_dist = ent_w2_dist

        pre_w2_stat_dict = an.compute_distribution_statistics(fil_pre_w2_dist)
        con_w2_stat_dict = an.compute_distribution_statistics(fil_con_w2_dist)
        vac_w2_stat_dict = an.compute_distribution_statistics(fil_vac_w2_dist)
        ent_w2_stat_dict = an.compute_distribution_statistics(fil_ent_w2_dist)

        state_stats_dict[state]['w2']['prevalence'] = pre_w2_stat_dict
        state_stats_dict[state]['w2']['convinced'] = con_w2_stat_dict
        state_stats_dict[state]['w2']['vaccinated'] = vac_w2_stat_dict
        state_stats_dict[state]['w2']['end_time'] = ent_w2_stat_dict

        avg_pre_w2_list.append(pre_w2_stat_dict['avg'])
        l95_pre_w2_list.append(pre_w2_stat_dict['l95'])
        u95_pre_w2_list.append(pre_w2_stat_dict['u95'])
        avg_con_w2_list.append(con_w2_stat_dict['avg'])
        l95_con_w2_list.append(con_w2_stat_dict['l95'])
        u95_con_w2_list.append(con_w2_stat_dict['u95'])
        avg_vac_w2_list.append(vac_w2_stat_dict['avg'])
        l95_vac_w2_list.append(vac_w2_stat_dict['l95'])
        u95_vac_w2_list.append(vac_w2_stat_dict['u95'])
        avg_ent_w2_list.append(ent_w2_stat_dict['avg'])
        l95_ent_w2_list.append(ent_w2_stat_dict['l95'])
        u95_ent_w2_list.append(ent_w2_stat_dict['u95'])

    # Rearrange some data
    avg_pre_diff_array = np.asarray(avg_pre_w2_list) \
        - np.asarray(avg_pre_w1_list)
    
    # Prepare figure
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(15, 12))

    # Create scatter plots for the 1st Wave
    axs[0,0].scatter(already+soon, 
               avg_pre_w1_list, 
               color='royalblue')
    axs[1,0].scatter(never, 
               avg_pre_w1_list, 
               color='royalblue')
    axs[2,0].scatter(soon, 
               avg_pre_w1_list, 
               color='royalblue')
    axs[3,0].scatter(someone, 
               avg_pre_w1_list, 
               color='royalblue')
    axs[4,0].scatter(majority+never, 
               avg_pre_w1_list, 
               color='royalblue')
    
    # Create scatter plots for the 2nd Wave
    axs[0,1].scatter(already+soon, 
               avg_pre_diff_array, 
               color='royalblue')
    axs[1,1].scatter(never, 
               avg_pre_diff_array, 
               color='royalblue')
    axs[2,1].scatter(soon, 
               avg_pre_diff_array, 
               color='royalblue')
    axs[3,1].scatter(someone, 
               avg_pre_diff_array, 
               color='royalblue')
    axs[4,1].scatter(majority+never, 
               avg_pre_diff_array, 
               color='royalblue')
    
    # Perform linear regression
    X00 = np.asarray(already+soon).reshape(-1, 1)
    y00 = avg_pre_w1_list
    reg00 = LinearRegression().fit(X00, y00)
    r2_00 = reg00.score(X00, y00)
    y00_pred = reg00.predict(X00)

    # Perform linear regression
    X10 = np.asarray(never).reshape(-1, 1)
    y10 = avg_pre_w1_list
    reg10 = LinearRegression().fit(X10, y10)
    r2_10 = reg10.score(X10, y10)
    y10_pred = reg10.predict(X10)

    # Perform linear regression
    X20 = np.asarray(soon).reshape(-1, 1)
    y20 = avg_pre_w1_list
    reg20 = LinearRegression().fit(X20, y20)
    r2_20 = reg20.score(X20, y20)
    y20_pred = reg20.predict(X20)

    # Perform linear regression
    X30 = np.asarray(someone).reshape(-1, 1)
    y30 = avg_pre_w1_list
    reg30 = LinearRegression().fit(X30, y30)
    r2_30 = reg30.score(X30, y30)
    y30_pred = reg30.predict(X30)

    # Perform linear regression
    X40 = np.asarray(majority+never).reshape(-1, 1)
    y40 = avg_pre_w1_list
    reg40 = LinearRegression().fit(X40, y40)
    r2_40 = reg40.score(X40, y40)
    y40_pred = reg40.predict(X40)

    # Perform linear regression
    X01 = np.asarray(already+soon).reshape(-1, 1)
    y01 = avg_pre_diff_array
    reg01 = LinearRegression().fit(X01, y01)
    r2_01 = reg01.score(X01, y01)
    y01_pred = reg01.predict(X01)

    # Perform linear regression
    X11 = np.asarray(never).reshape(-1, 1)
    y11 = avg_pre_diff_array
    reg11 = LinearRegression().fit(X11, y11)
    r2_11 = reg11.score(X11, y11)
    y11_pred = reg11.predict(X11)

    # Perform linear regression
    X21 = np.asarray(soon).reshape(-1, 1)
    y21 = avg_pre_diff_array
    reg21 = LinearRegression().fit(X21, y21)
    r2_21 = reg21.score(X21, y21)
    y21_pred = reg21.predict(X21)

    # Perform linear regression
    X31 = np.asarray(someone).reshape(-1, 1)
    y31 = avg_pre_diff_array
    reg31 = LinearRegression().fit(X31, y31)
    r2_31 = reg31.score(X31, y31)
    y31_pred = reg31.predict(X31)

    # Perform linear regression
    X41 = np.asarray(majority+never).reshape(-1, 1)
    y41 = avg_pre_diff_array
    reg41 = LinearRegression().fit(X41, y41)
    r2_41 = reg41.score(X41, y41)
    y41_pred = reg41.predict(X41)

    # Plot the fitting lines for the 1st Wave
    axs[0,0].plot(already+soon, y00_pred, 
            color='black', label=f'R^2 = {r2_00:.2f}')
    axs[1,0].plot(never, y10_pred, 
            color='black', label=f'R^2 = {r2_10:.2f}')
    axs[2,0].plot(soon, y20_pred, 
            color='black', label=f'R^2 = {r2_20:.2f}')
    axs[3,0].plot(someone, y30_pred, 
            color='black', label=f'R^2 = {r2_30:.2f}')
    axs[4,0].plot(majority+never, y40_pred, 
            color='black', label=f'R^2 = {r2_40:.2f}')
    # Plot the fitting lines for the 2nd Wace
    axs[0,1].plot(already+soon, y01_pred, 
            color='black', label=f'R^2 = {r2_01:.2f}')
    axs[1,1].plot(never, y11_pred, 
            color='black', label=f'R^2 = {r2_11:.2f}')
    axs[2,1].plot(soon, y21_pred, 
            color='black', label=f'R^2 = {r2_21:.2f}')
    axs[3,1].plot(someone, y31_pred, 
            color='black', label=f'R^2 = {r2_31:.2f}')
    axs[4,1].plot(majority+never, y41_pred, 
            color='black', label=f'R^2 = {r2_41:.2f}')

    # Add US state code annotations
    for i, state_code in enumerate(state_code_list):
        axs[0,0].annotate(
            state_code, 
            (already[i]+soon[i], 
             avg_pre_w1_list[i])
        )
        axs[1,0].annotate(
            state_code, 
            (never[i], 
             avg_pre_w1_list[i])
        )
        axs[2,0].annotate(
            state_code, 
            (soon[i], 
            avg_pre_w1_list[i])
        )
        axs[3,0].annotate(
            state_code, 
            (someone[i], 
            avg_pre_w1_list[i])
        )
        axs[4,0].annotate(
            state_code, 
            (majority[i]+never[i], 
             avg_pre_w1_list[i])
        )
        axs[0,1].annotate(
            state_code, 
            (already[i]+soon[i], 
             avg_pre_diff_array[i])
        )
        axs[1,1].annotate(
            state_code, 
            (never[i], 
             avg_pre_diff_array[i])
        )
        axs[2,1].annotate(
            state_code, 
            (soon[i], 
            avg_pre_diff_array[i])
        )
        axs[3,1].annotate(
            state_code, 
            (someone[i], 
            avg_pre_diff_array[i])
        )
        axs[4,1].annotate(
            state_code, 
            (majority[i]+never[i], 
             avg_pre_diff_array[i])
        )

    # Settings
    axs[0,0].set_title("1st wave", fontsize=25)
    axs[0,1].set_title("2nd wave", fontsize=25)
    
    axs[0,0].tick_params(axis='both', which='major', labelsize=15)
    axs[1,0].tick_params(axis='both', which='major', labelsize=15)
    axs[2,0].tick_params(axis='both', which='major', labelsize=15)
    axs[3,0].tick_params(axis='both', which='major', labelsize=15)
    axs[4,0].tick_params(axis='both', which='major', labelsize=15)
    axs[0,1].tick_params(axis='both', which='major', labelsize=15)
    axs[1,1].tick_params(axis='both', which='major', labelsize=15)
    axs[2,1].tick_params(axis='both', which='major', labelsize=15)
    axs[3,1].tick_params(axis='both', which='major', labelsize=15)
    axs[4,1].tick_params(axis='both', which='major', labelsize=15)

    axs[0,0].set_xlabel('already fraction', fontsize=20)
    axs[1,0].set_xlabel('never fraction', fontsize=20)
    axs[2,0].set_xlabel('soon fraction', fontsize=20)
    axs[3,0].set_xlabel('someone fraction', fontsize=20)
    axs[4,0].set_xlabel('majority fraction', fontsize=20)
    axs[0,1].set_xlabel('already fraction', fontsize=20)
    axs[1,1].set_xlabel('never fraction', fontsize=20)
    axs[2,1].set_xlabel('soon fraction', fontsize=20)
    axs[3,1].set_xlabel('someone fraction', fontsize=20)
    axs[4,1].set_xlabel('majority fraction', fontsize=20)
    
    axs[0,0].set_ylabel('prevalence', fontsize=20)
    axs[1,0].set_ylabel('prevalence', fontsize=20)
    axs[2,0].set_ylabel('prevalence', fontsize=20)
    axs[3,0].set_ylabel('prevalence', fontsize=20)
    axs[4,0].set_ylabel('prevalence', fontsize=20)

    axs[0,0].legend()
    axs[1,0].legend()
    axs[2,0].legend()
    axs[3,0].legend()
    axs[4,0].legend()
    axs[0,1].legend()
    axs[1,1].legend()
    axs[2,1].legend()
    axs[3,1].legend()
    axs[4,1].legend()
    
    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = 'scatter_' + ut.build_file_name(pars, 
                                                collection=True, 
                                                plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    
    plt.show()

def plot_heatmap_correlation(pars):
    # Load the data into a pandas DataFrame
    data = pd.read_csv('your_data_file.csv')

    # Calculate the correlation coefficients between the prevalence and each of the vaccination categories
    corr = data[['already', 'soon', 'someone', 'most', 'never', 'prevalence']].corr(method='pearson')

    # Create a heatmap of the correlation coefficients
    sns.heatmap(corr, annot=True, cmap='coolwarm')

def plot_us_states_time_series(pars):
    # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_us_states(pars)

    # Load results
    state_raw_dict = pk.load(open(full_name, "rb"))

    N = pars['network']['n']
    nsims = pars['algorithm']['nsn'] * pars['algorithm']['nsd']
    t_max = pars['algorithm']['tmax']
    prevalence_cutoff = 0.025
    state_code_list = []
    state_time_dict = {}
    filtered_state_time_dict = {}
    filtered_state_stat_dict = {}

    for state in state_raw_dict.keys():
        print("{0}".format(state))
        state_code_list.append(ut.get_state_code(state))

        state_time_dict[state] = {'w1': {}, 'w2': {}}

        for wave in ['w1', 'w2']:
            state_time_dict[state][wave]['time'] \
                = state_raw_dict[state][wave]['time']
            state_time_dict[state][wave]['as'] \
                = state_raw_dict[state][wave]['as']
            state_time_dict[state][wave]['hs'] \
                = state_raw_dict[state][wave]['hs']
            state_time_dict[state][wave]['ai'] \
                = state_raw_dict[state][wave]['ai']
            state_time_dict[state][wave]['hi'] \
                = state_raw_dict[state][wave]['hi']
            state_time_dict[state][wave]['ar'] \
                = state_raw_dict[state][wave]['ar']
            state_time_dict[state][wave]['hr'] \
                = state_raw_dict[state][wave]['hr']
            state_time_dict[state][wave]['av'] \
                = state_raw_dict[state][wave]['av']
       
        filtered_state_time_dict[state] = {'w1': {}, 'w2': {}}
        for wave in ['w1', 'w2']:
            ar_hr_sum = state_time_dict[state][wave]['ar'][:, -1] \
                + state_time_dict[state][wave]['hr'][:, -1]
            mask = ar_hr_sum / N >= prevalence_cutoff
            for key in state_time_dict[state][wave]:
                arr = state_time_dict[state][wave][key]
                filtered_arr = arr[mask]
                filtered_state_time_dict[state][wave][key] = filtered_arr

        filtered_state_stat_dict[state] = {'w1': 
                                           {'t': {}, 
                                            'as': {}, 
                                            'hs': {}, 
                                            'ai': {}, 
                                            'hi': {}, 
                                            'ar': {}, 
                                            'hr': {}, 
                                            'av': {}}, 
                                            'w2': 
                                            {'t': {}, 
                                            'as': {}, 
                                            'hs': {}, 
                                            'ai': {}, 
                                            'hi': {}, 
                                            'ar': {}, 
                                            'hr': {}, 
                                            'av': {}}}
        
        for wave in ['w1', 'w2']:
            for status in filtered_state_time_dict[state][wave]:
                dist_st = filtered_state_time_dict[state][wave][status]
                stat_dict = an.compute_distribution_statistics(dist_st[:,:]) # TO FIX
                filtered_state_stat_dict[state][wave][status] = stat_dict
    
    # Prepare figure
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))

    # Upper left plot
    axs[0, 0].plot(
        filtered_state_stat_dict[state]['w1']['ar']['avg'], 
        label='ar'
    )
    axs[0, 0].set_xlabel('Time')
    axs[0, 0].set_ylabel('Average')
    axs[0, 0].legend()

    # Upper right plot
    axs[0, 1].plot(
        filtered_state_stat_dict[state]['w1']['av']['avg'], 
        label='av'
    )
    axs[0, 1].set_xlabel('Time')
    axs[0, 1].set_ylabel('Average')
    axs[0, 1].legend()

    # Lower left plot
    axs[1, 0].plot(
        filtered_state_stat_dict[state]['w2']['ar']['avg'], 
        label='ar'
    )
    axs[1, 0].set_xlabel('Time')
    axs[1, 0].set_ylabel('Average')
    axs[1, 0].legend()

    # Lower right plot
    axs[1, 1].plot(
        filtered_state_stat_dict[state]['w2']['av']['avg'], 
        label='av'
    )
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Average')
    axs[1, 1].legend()

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = \
        'tseries_' + ut.build_file_name(pars, collection=True, plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    
    plt.show()

def plot_us_states_control_vaccination(pars):
        # Check if the unified pickle exists, and generate it if not
    base_name = ut.build_file_name(pars, exclude_keys=['var'], collection=True)
    pickle_name = base_name + '.pickle'
    lower_path = 'results'
    full_name = os.path.join(path, lower_path, pickle_name)
    if not os.path.isfile(full_name):
        ut.collect_pickles_for_us_states_vaccination(pars)

    # Load results
    results_dict = pk.load(open(full_name, "rb"))

    # Collect filtered results
    N = pars['network']['n']
    prevalence_cutoff = 0.025
    state_code_list = []

    # Prepare final figure
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    
    # Prepare results dictionary
    stats_state_alpha_dict = {}

    for state in results_dict.keys():
        print("{0}".format(state))
        state_code_list.append(ut.get_state_code(state))

        if state not in stats_state_alpha_dict:
            stats_state_alpha_dict[state] = {}

        for alpha in sorted(results_dict[state].keys()):

            if alpha not in stats_state_alpha_dict[state]:
                    stats_state_alpha_dict[state][alpha] = {'w1': {}, 'w2': {}}
           
            res_alpha = results_dict[state][alpha]
            # Get relevant data for 1st wave
            pre_w1_dist = np.array(res_alpha['w1']['prevalence']) / N
            con_w1_dist = np.array(res_alpha['w1']['convinced']) / N
            vac_w1_dist = np.array(res_alpha['w1']['vaccinated']) / N

            # Filter failed outbreaks
            fil_pre_w1_dist = pre_w1_dist[pre_w1_dist>=prevalence_cutoff]
            if fil_pre_w1_dist.size == 0:
                fil_pre_w1_dist = pre_w1_dist
            fil_con_w1_dist = con_w1_dist[pre_w1_dist>=prevalence_cutoff]
            if fil_con_w1_dist.size == 0:
                fil_con_w1_dist = con_w1_dist
            fil_vac_w1_dist = vac_w1_dist[pre_w1_dist>=prevalence_cutoff]
            if fil_vac_w1_dist.size == 0:
                fil_vac_w1_dist = vac_w1_dist

            # Compute statistics
            pre_w1_stat_dict = an.compute_distribution_statistics(fil_pre_w1_dist)
            con_w1_stat_dict = an.compute_distribution_statistics(fil_con_w1_dist)
            vac_w1_stat_dict = an.compute_distribution_statistics(fil_vac_w1_dist)

            # Store data
            stats_state_alpha_dict[state][alpha]['w1'] \
                = {'prevalence': {}, 'convinced': {}, 'vaccinated': {}}
            stats_state_alpha_dict[state][alpha]['w1']['prevalence'] = pre_w1_stat_dict
            stats_state_alpha_dict[state][alpha]['w1']['convinced'] = con_w1_stat_dict
            stats_state_alpha_dict[state][alpha]['w1']['vaccinated'] = vac_w1_stat_dict

            # Get relevant data for 2nd wave
            pre_w2_dist = np.array(res_alpha['w2']['prevalence']) / N
            con_w2_dist = np.array(res_alpha['w2']['convinced']) / N
            vac_w2_dist = np.array(res_alpha['w2']['vaccinated']) / N

            # Filter outbreaks
            fil_pre_w2_dist = pre_w2_dist[pre_w2_dist>=prevalence_cutoff]
            if fil_pre_w2_dist.size == 0:
                fil_pre_w2_dist = pre_w2_dist
            fil_con_w2_dist = con_w2_dist[pre_w2_dist>=prevalence_cutoff]
            if fil_con_w2_dist.size == 0:
                fil_con_w2_dist = con_w2_dist
            fil_vac_w2_dist = vac_w2_dist[pre_w2_dist>=prevalence_cutoff]
            if fil_vac_w2_dist.size == 0:
                fil_vac_w2_dist = vac_w2_dist

            # Compute statistics
            pre_w2_stat_dict = an.compute_distribution_statistics(fil_pre_w2_dist)
            con_w2_stat_dict = an.compute_distribution_statistics(fil_con_w2_dist)
            vac_w2_stat_dict = an.compute_distribution_statistics(fil_vac_w2_dist)

            # Store data
            stats_state_alpha_dict[state][alpha]['w2'] \
                = {'prevalence': {}, 'convinced': {}, 'vaccinated': {}}
            stats_state_alpha_dict[state][alpha]['w2']['prevalence'] = pre_w2_stat_dict
            stats_state_alpha_dict[state][alpha]['w2']['convinced'] = con_w2_stat_dict
            stats_state_alpha_dict[state][alpha]['w2']['vaccinated'] = vac_w2_stat_dict
                
        # Plot vaccination rate curves for the state
        alpha_array = sorted(stats_state_alpha_dict[state].keys())
        pre_w1_list = [stats_state_alpha_dict[state][alpha]['w1']['prevalence']['avg'] for alpha in alpha_array]
        pre_w1_array = np.array(pre_w1_list)
        con_w1_list = [stats_state_alpha_dict[state][alpha]['w1']['convinced']['avg'] for alpha in alpha_array]
        con_w1_array = np.array(con_w1_list)
        pre_w2_list = [stats_state_alpha_dict[state][alpha]['w2']['prevalence']['avg'] for alpha in alpha_array]
        pre_w2_array = np.array(pre_w2_list)
        axs[0].plot(alpha_array, pre_w1_array, marker='o', markersize=2, color='black', linestyle=':')
        axs[1].plot(alpha_array, con_w1_array, marker='o', markersize=2, color='black', linestyle=':')

   
    # Settings
    R0 = pars['epidemic']['r0']
    R02 = 2.0 * R0
    #axs[0].set_title(r"1st wave, $R_0={0}$".format(R0), fontsize=25)
    axs[0].tick_params(axis='both', which='major', labelsize=15)
    axs[0].set_xscale('log')
    axs[0].set_xlabel(r'vaccination rate $\alpha$', fontsize=20)
    axs[0].set_ylabel(r'$r(\infty)$', fontsize=20)
    axs[0].legend()
    #axs[1].set_title(r"2nd wave, $R_0={0}$".format(R02), fontsize=25)
    axs[1].tick_params(axis='both', which='major', labelsize=15)
    axs[1].set_xscale('log')
    axs[1].set_xlabel(r'vaccination rate $\alpha$', fontsize=20)
    axs[1].legend()

    # Font & font sizes
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20) 
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42

    plt.tight_layout()

    # Save plot
    lower_path = 'results'
    full_path = os.path.join(path, lower_path)
    plot_name = 'control_' + ut.build_file_name(pars, 
                                                collection=True, 
                                                plot=True)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    
    plt.show()

def plot_us_states_control_r0(pars):
    pass


def main():

    # Input parameters for the model
    project_id = 'thr'
    dynmod_id = 'mc'
    attribute_id = 'global'

    model = 'ErdosRenyi'
    n = 10000
    p = 0.001
    m = 5
 
    r0 = 1.5
    gamma = 0.2
    beta = 0.46
    
    seeds = 10
    t_max = 500
    nsims_net = 35
    nsims_dyn = 35

    # EXP1: (THETA, ACTIVE) + VACRATE, NO ZEALOTS
    # Build full parameter dictionary
    alpha = 0.001

    pars = ut.build_parameter_dictionary(
        project_id=project_id, 
        dynmod_id=dynmod_id, 
        attribute_id=attribute_id, 
        exp_id=1, 
        model=model, 
        n=n, 
        p=p, 
        m=m,
        n_Z=0, 
        seeds=seeds, 
        alpha=alpha, 
        beta=beta, 
        gamma=gamma, 
        r0=r0, 
        nsims_net=nsims_net, 
        nsims_dyn=nsims_dyn, 
        t_max=t_max, 
        control1='thr', 
        control2='acf',
        observable='vaccinated',
    )

    #plot_mc_heatmap_theta_active(pars)
    #plot_outbreak_distribution_histogram_theta_active(pars)

    # EXP1: (THETA, ACTIVE) + VACRATE + ZEALOT
    # Build full parameter dictionary
    alpha = 0.05
    n_Z = 0.1

    pars = ut.build_parameter_dictionary(
        project_id=project_id, 
        dynmod_id=dynmod_id, 
        attribute_id=attribute_id, 
        exp_id=1, 
        model=model, 
        n=n, 
        p=p,
        m=m,
        n_Z=n_Z, 
        seeds=seeds, 
        alpha=alpha, 
        beta=beta, 
        gamma=gamma, 
        r0=r0, 
        nsims_net=nsims_net, 
        nsims_dyn=nsims_dyn, 
        t_max=t_max, 
        control1='thr', 
        control2='acf',
        observable='vaccinated',
    )

    #plot_zealot_distribution(pars)
    #plot_mc_heatmap_theta_active(pars)
    #plot_outbreak_distribution_histogram_theta_active(pars)

    # EXP1: (ACTIVE,ZEALOT) + VACRATE + THETA
    alpha = 0.05
    theta = 0.3
    
    pars = ut.build_parameter_dictionary(
        project_id=project_id, 
        dynmod_id=dynmod_id, 
        attribute_id=attribute_id, 
        exp_id=1, 
        model=model, 
        n=n, 
        p=p, 
        m=m,
        theta=theta,
        seeds=seeds, 
        alpha=alpha, 
        beta=beta, 
        gamma=gamma, 
        r0=r0, 
        nsims_net=nsims_net, 
        nsims_dyn=nsims_dyn, 
        t_max=t_max, 
        control1='acf', 
        control2='zef',
        observable='vaccinated',
    )

    #plot_mc_heatmap_active_zealot(pars)
    #plot_outbreak_distribution_histogram_active_zealot(pars)

    # EXP2: US SCATTER & BARS
    nsims_net = 50
    nsims_dyn = 50
    alpha = 0.01
    
    pars = ut.build_parameter_dictionary(
        project_id=project_id, 
        dynmod_id=dynmod_id, 
        attribute_id=attribute_id, 
        exp_id=2, 
        model=model, 
        n=n, 
        p=p, 
        m=m,
        seeds=seeds, 
        alpha=alpha, 
        beta=beta, 
        gamma=gamma, 
        r0=r0, 
        nsims_net=nsims_net, 
        nsims_dyn=nsims_dyn, 
        t_max=t_max,
    )

    #plot_scatter_us_states_outbreaks(pars)
    #plot_bar_us_states_immunized(pars)
    #plot_bar_us_states_thresholds(pars)

    # EXP2: STATE OUTBREAK
    nsims_net = 50
    nsims_dyn = 50
    usa_id = 'Oklahoma'
    alpha = 0.001
    
    pars = ut.build_parameter_dictionary(
        project_id=project_id, 
        dynmod_id=dynmod_id, 
        attribute_id=attribute_id, 
        usa_id=usa_id,
        exp_id=2, 
        model=model, 
        n=n, 
        p=p, 
        m=m,
        theta=0.9,
        n_A=0.2,
        n_Z=0,
        seeds=seeds, 
        alpha=alpha, 
        beta=beta, 
        gamma=gamma, 
        r0=r0, 
        nsims_net=nsims_net, 
        nsims_dyn=nsims_dyn, 
        t_max=t_max,
    )

    #plot_us_state_outbreak_distribution(pars)

    # EXP2: VACCINATION RATE
    nsims_net = 50
    nsims_dyn = 50
    
    pars = ut.build_parameter_dictionary(
        project_id=project_id, 
        dynmod_id=dynmod_id, 
        attribute_id=attribute_id, 
        exp_id=2, 
        model=model, 
        n=n, 
        p=p, 
        m=m,
        seeds=seeds, 
        beta=beta, 
        gamma=gamma, 
        r0=r0, 
        nsims_net=nsims_net, 
        nsims_dyn=nsims_dyn, 
        t_max=t_max,
    )

    #plot_us_states_control_vaccination(pars)

    


if __name__ == "__main__":
    main()
