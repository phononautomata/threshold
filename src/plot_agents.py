import os
import collections
import itertools
import numpy as np
import pickle as pk
import ruptures as rpt
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from collections import Counter, OrderedDict, defaultdict
from itertools import chain
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from statistics import mode
from sklearn.linear_model import LinearRegression
from scipy.stats import gaussian_kde

import analysis as an
import utils as ut

cwd_path = os.getcwd()


def plot_agents_one(pars):

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_agents_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'results/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        net_flag = epi_filename.split('_net')[1].split('_')[0]
        k_flag = int(epi_filename.split('_k')[1].split('_')[0])
        acf_flag = float(epi_filename.split('_acf')[1].split('_')[0])
        thr_flag = float(epi_filename.split('_thr')[1].split('_')[0])
        var_flag = float(epi_filename.split('_var')[1].split('_')[0])
        zef_flag = float(epi_filename.split('_zef')[1].split('_')[0])

        tavz_tuple = (thr_flag, acf_flag, var_flag, zef_flag)

        if net_flag in pars['net'] and k_flag == pars['k'] and tavz_tuple == pars['tavz']:

            agent_output = ut.load_raw_agent_output(epi_fullname)

    # Unpack agent output
    ids = agent_output['id']
    degree = agent_output['degree']
    convinced_when = agent_output['convinced_when']
    infected_when = agent_output['infected_when']
    removed_when = agent_output['removed_when']
    vaccinated_when = agent_output['vaccinated_when']

    times_k_dict = ut.collect_agent_times_by_degree(
        ids=ids, 
        degree=degree, 
        convinced_when=convinced_when, 
        infected_when=infected_when, 
        removed_when=removed_when, 
        vaccinated_when=vaccinated_when,
        )
    
    stat_times_event_degree_dict = ut.stat_agent_times_by_event_degree(times_k_dict)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 14))

    degrees = []
    avg_values = []

    convinced_event = 'convinced'
    vaccinated_event = 'vaccinated'
    infected_event = 'infected'

    for i, event_type in enumerate([convinced_event, vaccinated_event, infected_event]):
        for degree, stats in stat_times_event_degree_dict[event_type].items():
            degrees.append(degree)
            avg_values.append(stats['avg'])

            ax[i].scatter(degrees, avg_values, color='slateblue')

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'fig1_age_times_'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_agents_two(pars):

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_agents_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'results/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        net_flag = epi_filename.split('_net')[1].split('_')[0]
        k_flag = int(epi_filename.split('_k')[1].split('_')[0])
        acf_flag = float(epi_filename.split('_acf')[1].split('_')[0])
        thr_flag = float(epi_filename.split('_thr')[1].split('_')[0])
        var_flag = float(epi_filename.split('_var')[1].split('_')[0])
        zef_flag = float(epi_filename.split('_zef')[1].split('_')[0])

        tavz_tuple = (thr_flag, acf_flag, var_flag, zef_flag)

        if net_flag in pars['net'] and k_flag == pars['k'] and tavz_tuple == pars['tavz']:

            agent_output = ut.load_raw_agent_output(epi_fullname)

    # Unpack agent output
    ids = agent_output['id']
    degree = agent_output['degree']
    status = agent_output['status']
    final_act_sus = agent_output['final_act_sus']
    final_prevalence = agent_output['final_prevalence']
    final_vaccinated = agent_output['final_vaccinated']
    initial_active_sus = agent_output['initial_active_susceptible']
    zealots = agent_output['zealots']

    neigh_k_dict = ut.collect_agent_neighborhood_by_degree(
        ids, 
        status=status, 
        degree=degree, 
        final_act_sus=final_act_sus, 
        final_prevalence=final_prevalence, 
        final_vaccinated=final_vaccinated,
        initial_active_susceptible=initial_active_sus,
        zealots=zealots,
        )
    
    stat_neigh_obs_degree_dict = ut.stat_agent_neighborhood_by_obs_degree(neigh_k_dict)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 14))

    fp = 'final_prevalence'
    fv = 'final_vaccinated'
    ias = 'initial_active_sus'

    for i, event_type in enumerate([fp, fv, ias]):
        degrees = []
        avg_values = []

        for degree, stats in stat_neigh_obs_degree_dict[event_type].items():
            degrees.append(degree)
            avg_values.append(stats['avg'])

            ax[i].scatter(degrees, avg_values, color='slateblue')
    
    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'fig2_age_neigh_'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_agents_three(pars):

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_agents_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'results/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        net_flag = epi_filename.split('_net')[1].split('_')[0]
        k_flag = int(epi_filename.split('_k')[1].split('_')[0])
        acf_flag = float(epi_filename.split('_acf')[1].split('_')[0])
        thr_flag = float(epi_filename.split('_thr')[1].split('_')[0])
        var_flag = float(epi_filename.split('_var')[1].split('_')[0])
        zef_flag = float(epi_filename.split('_zef')[1].split('_')[0])

        tavz_tuple = (thr_flag, acf_flag, var_flag, zef_flag)

        if net_flag in pars['net'] and k_flag == pars['k'] and tavz_tuple == pars['tavz']:

            agent_output = ut.load_raw_agent_output(epi_fullname)

            break

    # Unpack agent output
    ids = agent_output['id']
    degree = agent_output['degree']
    status = agent_output['status']
    final_act_sus = agent_output['final_act_sus']
    final_prevalence = agent_output['final_prevalence']
    final_vaccinated = agent_output['final_vaccinated']
    initial_active_sus = agent_output['initial_active_susceptible']
    zealots = agent_output['zealots']

    agent_df = ut.build_agent_dataframe(
        ids=ids, 
        status=status, 
        degree=degree, 
        final_act_sus=final_act_sus, 
        final_prevalence=final_prevalence, 
        final_vaccinated=final_vaccinated, 
        initial_active_sus=initial_active_sus, 
        zealots=zealots,
        )

    #fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 14))

    status_categories = agent_df['status'].unique()
    color_dict = {0: 'springgreen', 1: 'springgreen', 2: 'firebrick', 3: 'indigo'}

    # Create a scatter plot
    fig, ax = plt.subplots()

    # Iterate through status categories and plot points with the corresponding color
    for i, status_category in enumerate(status_categories):
        subset = agent_df[agent_df['status'] == status_category]
        ax.scatter(subset['initial_active_susceptible'], subset['final_vaccinated'], label=status_category, c=color_dict[status_category], alpha=0.7)

    # Set labels for the axes
    ax.set_xlabel(r'$n_A^{\ell}(0)$', fontsize=30)
    ax.set_ylabel(r'$v^{\ell}(\infty)$', fontsize=30)

    ax.tick_params(axis='both', labelsize=18)

    ax.legend(loc='center right', fontsize=15)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'fig3_age_neigh_' + str(tavz_tuple)
    extension_list = ['png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_clusters_one(pars):

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_clusters_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'results/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        net_flag = epi_filename.split('_net')[1].split('_')[0]
        k_flag = int(epi_filename.split('_k')[1].split('_')[0])
        acf_flag = float(epi_filename.split('_acf')[1].split('_')[0])
        thr_flag = float(epi_filename.split('_thr')[1].split('_')[0])
        var_flag = float(epi_filename.split('_var')[1].split('_')[0])
        zef_flag = float(epi_filename.split('_zef')[1].split('_')[0])

        tavz_tuple = (thr_flag, acf_flag, var_flag, zef_flag)

        if net_flag in pars['net'] and k_flag == pars['k'] and tavz_tuple == pars['tavz']:

            cluster_output = ut.load_raw_cluster_output(epi_fullname)

            break

    # Load clusters
    ar_clusters = cluster_output['ar']
    as_clusters = cluster_output['as']
    av_clusters = cluster_output['av']
    hr_clusters = cluster_output['hr']
    hs_clusters = cluster_output['hs']
    hv_clusters = cluster_output['hv']
    ze_clusters = cluster_output['ze']

    # Filter clusters
    #prevalence_cutoff = 0.005
    #failed_outbreaks = []
    #for s, hrc in enumerate(hr_clusters):
    #    prevalence = (sum(hr_clusters[s]) + sum(ar_clusters[s]))
    #    if prevalence <= prevalence_cutoff:
    #        failed_outbreaks.append(s)

    ## Filter clusters based on failed_outbreaks_indices
    #ar_clusters = [ar_clusters[i] for i in range(len(ar_clusters)) if i not in failed_outbreaks]
    #as_clusters = [as_clusters[i] for i in range(len(as_clusters)) if i not in failed_outbreaks]
    #av_clusters = [av_clusters[i] for i in range(len(av_clusters)) if i not in failed_outbreaks]
    #hr_clusters = [hr_clusters[i] for i in range(len(hr_clusters)) if i not in failed_outbreaks]
    #hs_clusters = [hs_clusters[i] for i in range(len(hs_clusters)) if i not in failed_outbreaks]
    #hv_clusters = [hv_clusters[i] for i in range(len(hv_clusters)) if i not in failed_outbreaks]
    #ze_clusters = [ze_clusters[i] for i in range(len(ze_clusters)) if i not in failed_outbreaks]

    ar_clusters = [item for sublist in ar_clusters for item in sublist]
    as_clusters = [item for sublist in as_clusters for item in sublist]
    av_clusters = [item for sublist in av_clusters for item in sublist]
    hr_clusters = [item for sublist in hr_clusters for item in sublist]
    hs_clusters = [item for sublist in hs_clusters for item in sublist]
    hv_clusters = [item for sublist in hv_clusters for item in sublist]
    ze_clusters = [item for sublist in ze_clusters for item in sublist]

    # Create subplots with 3 rows and 1 column
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(10, 14))

    bins = 30
    density = True

    # Plot distribution of av_clusters
    av_clusters = np.array(av_clusters)
    norm_av_clusters = av_clusters[av_clusters > 10] / n
    ax[0].hist(norm_av_clusters, bins=bins, density=density, color='slateblue', alpha=0.7)
    ax[0].set_title('vaccinated clusters', fontsize=30)
    ax[0].set_xlabel('normalized size', fontsize=30)
    ax[0].set_ylabel('frequency', fontsize=30)

    # Merge ar_clusters and hr_clusters into one list
    r_clusters = np.array(ar_clusters + hr_clusters)
    norm_r_clusters = r_clusters[r_clusters > 10] / n

    # Plot distribution of merged_ar_hr_clusters
    ax[1].hist(norm_r_clusters, bins=bins, density=density, color='slateblue', alpha=0.7)
    ax[1].set_title('removed clusters', fontsize=30)
    ax[1].set_xlabel('normalized size', fontsize=30)
    ax[1].set_ylabel('frequency', fontsize=30)

    # Plot distribution of hs_clusters
    hs_clusters = np.array(hs_clusters)
    norm_hs_clusters = hs_clusters[hs_clusters > 10] / n
    ax[2].hist(norm_hs_clusters, bins=bins, density=density, color='slateblue', alpha=0.7)
    ax[2].set_title('hesitant clusters', fontsize=30)
    ax[2].set_xlabel('normalized size', fontsize=30)
    ax[2].set_ylabel('frequency', fontsize=30)

    ax[0].set_xlim(-0.01, 1.01)
    ax[1].set_xlim(-0.01, 1.01)
    ax[2].set_xlim(-0.01, 1.01)

    for axis in ax.flatten():
        axis.tick_params(axis='both', labelsize=18)

    plt.rcParams.update({'font.size': 15})
    plt.rc('axes', labelsize=20)
    plt.rcParams['xtick.labelsize'] = 20
    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    plt.rcParams['pdf.fonttype'] = 42
    plt.tight_layout()

    # Save plot
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    epi_filename = ut.trim_file_extension(epi_filename)
    base_name = 'fig1_clu_' + str(tavz_tuple)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def main():

    net_id = 'er'
    target_k = 10
    thr = 0.75
    acf = 0.5
    var = 1
    zef = 0

    pars = {}
    pars['net'] = net_id
    pars['k'] = target_k
    pars['tavz'] = (thr, acf, var, zef)

    plot_agents_three(pars)

if __name__ == '__main__':
    main()