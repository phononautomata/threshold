import os
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap, Normalize
from collections import Counter, OrderedDict, defaultdict

import analysis as an
import utils as ut

cwd_path = os.getcwd()


def plot_threshold_hom_f1(target_net='er', target_k=10, target_zef=0.0, target_var=None):

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_global_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

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

        if net_flag == target_net and k_flag == target_k and zef_flag == target_zef and var_flag in target_var:

            global_output = ut.load_global_output(epi_fullname)

            filtered_output = ut.filter_global_output(global_output, n, prevalence_cutoff)

            stat_output = ut.stat_global_output(filtered_output)

            if net_flag not in go_dict:
                go_dict[net_flag] = {}

            if k_flag not in go_dict[net_flag]:
                go_dict[net_flag][k_flag] = {}
            
            if var_flag not in go_dict[net_flag][k_flag]:
                go_dict[net_flag][k_flag][var_flag] = {}

            key_tuple = (thr_flag, acf_flag)
            go_dict[net_flag][k_flag][var_flag][key_tuple] = stat_output

    for net_flag, net_data in go_dict.items():
        for k_flag, k_data in net_data.items():
            sorted_keys = sorted(k_data.keys())
            sorted_items = [(key, k_data[key]) for key in sorted_keys]
            go_dict[net_flag][k_flag] = dict(sorted_items)

    fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(18, 20))
    cmap = 'viridis'
    
    for net_flag in go_dict.keys():

        for v, var_flag in enumerate(go_dict[net_flag][k_flag].keys()):

            go_net_k_var_dict = go_dict[net_flag][k_flag][var_flag]

            par1_array = sorted(list(set([key[0] for key in go_net_k_var_dict.keys()]))) # THRESHOLD
            par2_array = sorted(list(set([key[1] for key in go_net_k_var_dict.keys()]))) # ACTIVE FRACTION

            par1_len = len(par1_array)
            par2_len = len(par2_array)

            X, Y = np.meshgrid(par1_array, par2_array)
            R = np.zeros((par2_len, par1_len))
            V = np.zeros((par2_len, par1_len))
            C = np.zeros((par2_len, par1_len))

            for i, p1 in enumerate(par1_array):
                for j, p2 in enumerate(par2_array):
                    if (p1, p2) in go_net_k_var_dict:

                        R[j][i] = go_net_k_var_dict[(p1, p2)]['prevalence']['avg']
                        V[j][i] = go_net_k_var_dict[(p1, p2)]['vaccinated']['avg']

                        max_change = 1.0 - p2
                        norm_change = (go_net_k_var_dict[(p1, p2)]['convinced']['avg'] - p2) / max_change
                        if p2 == 1.0:
                            norm_change = 0.0
                        C[j][i] = norm_change
                    else:
                        R[j][i] = np.nan
                        V[j][i] = np.nan
                        C[j][i] = np.nan

            if net_flag == 'er':
                im2 = ax[v, 2].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=0.6)
                im1 = ax[v, 1].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
                im0 = ax[v, 0].pcolormesh(X, Y, C, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)

                contour_levels = [0.0195]  # Adjust these levels as needed
                if v == 2:
                    contour_levels = [0.07]
                if v == 3:
                    contour_levels = [0.05]
                ax[v, 2].contour(X, Y, R, contour_levels, colors='white', linestyles='dashed')
                #ax[v, 1].contour(X, Y, V, contour_levels, colors='white', linestyles='dashed')
                contour_levels = [0.1]
                ax[v, 0].contour(X, Y, C, levels=contour_levels, colors='white', linestyles='dashed')

                cb2 = plt.colorbar(im2, ax=ax[v, 2])
                cb2.ax.tick_params(labelsize=18)
                cb1 = plt.colorbar(im1, ax=ax[v, 1])
                cb1.ax.tick_params(labelsize=18)
                cb0 = plt.colorbar(im0, ax=ax[v, 0])
                cb0.ax.tick_params(labelsize=18)

            elif net_flag == 'ba':
                im2 = ax[v, 2].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
                im1 = ax[v, 1].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
                im0 = ax[v, 0].pcolormesh(X, Y, C, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)

                contour_levels = [0.02]  # Adjust these levels as needed
                if v == 4:
                    contour_levels = [0.1]
                ax[v, 2].contour(X, Y, R, contour_levels, colors='white', linestyles='dashed')
                #ax[v, 1].contour(X, Y, V, contour_levels, colors='white', linestyles='dashed')
                contour_levels = [0.07]
                ax[v, 0].contour(X, Y, C, levels=contour_levels, colors='white', linestyles='dashed')

                cb2 = plt.colorbar(im0, ax=ax[v, 0])
                cb2.ax.tick_params(labelsize=18)
                cb1 = plt.colorbar(im1, ax=ax[v, 1])
                cb1.ax.tick_params(labelsize=18)
                cb0 = plt.colorbar(im2, ax=ax[v, 2])
                cb0.ax.tick_params(labelsize=18)

    ax[0, 2].set_title(r"$r(\infty)$", fontsize=35)
    ax[0, 1].set_title(r"$v(\infty)$", fontsize=35)
    ax[0, 0].set_title(r"$\Delta n_A(\infty)$", fontsize=35)

    ax[0, 0].text(-0.35, 0.15, r'$\alpha=0.001$', fontsize=30, transform=ax[0, 0].transAxes, rotation='vertical')
    ax[1, 0].text(-0.35, 0.12, r'$\alpha=0.0025$', fontsize=30, transform=ax[1, 0].transAxes, rotation='vertical')
    ax[2, 0].text(-0.35, 0.15, r'$\alpha=0.005$', fontsize=30, transform=ax[2, 0].transAxes, rotation='vertical')
    ax[3, 0].text(-0.35, 0.2, r'$\alpha=0.01$', fontsize=30, transform=ax[3, 0].transAxes, rotation='vertical')
    ax[4, 0].text(-0.35, 0.2, r'$\alpha=0.05$', fontsize=30, transform=ax[4, 0].transAxes, rotation='vertical')
    ax[5, 0].text(-0.35, 0.25, r'$\alpha=1.0$', fontsize=30, transform=ax[5, 0].transAxes, rotation='vertical')
    #ax[6, 0].text(-0.32, 0.25, r'$\alpha=0.05$', fontsize=30, transform=ax[6, 0].transAxes, rotation='vertical')
    #ax[7, 0].text(-0.32, 0.25, r'$\alpha=1.0$', fontsize=30, transform=ax[7, 0].transAxes, rotation='vertical')
    
    ax[5, 0].set_xlabel(r'$\theta$', fontsize=30)
    ax[5, 1].set_xlabel(r'$\theta$', fontsize=30)
    ax[5, 2].set_xlabel(r'$\theta$', fontsize=30)

    ax[0, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[1, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[2, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[3, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[4, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[5, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    #ax[6, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    #ax[7, 0].set_ylabel(r'$n_A(0)$', fontsize=30)

    ax[0, 0].text(0.04, 0.8, r"A1", transform=ax[0, 0].transAxes, fontsize=30, color='white', weight="bold")
    ax[1, 0].text(0.04, 0.8, r"A2", transform=ax[1, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 0].text(0.04, 0.8, r"A3", transform=ax[2, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[3, 0].text(0.04, 0.8, r"A4", transform=ax[3, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[4, 0].text(0.04, 0.8, r"A5", transform=ax[4, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[5, 0].text(0.04, 0.8, r"A6", transform=ax[5, 0].transAxes, fontsize=30, color='black', weight="bold")
    #ax[6, 0].text(0.04, 0.8, r"A7", transform=ax[6, 0].transAxes, fontsize=30, color='black', weight="bold")
    #ax[7, 0].text(0.04, 0.8, r"A8", transform=ax[7, 0].transAxes, fontsize=30, color='black', weight="bold")

    ax[0, 1].text(0.04, 0.8, r"B1", transform=ax[0, 1].transAxes, fontsize=30, color='white', weight="bold")
    ax[1, 1].text(0.04, 0.8, r"B2", transform=ax[1, 1].transAxes, fontsize=30, color='white', weight="bold")
    ax[2, 1].text(0.04, 0.8, r"B3", transform=ax[2, 1].transAxes, fontsize=30, color='white', weight="bold")
    ax[3, 1].text(0.04, 0.8, r"B4", transform=ax[3, 1].transAxes, fontsize=30, color='white', weight="bold")
    ax[4, 1].text(0.04, 0.8, r"B5", transform=ax[4, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[5, 1].text(0.04, 0.8, r"B6", transform=ax[5, 1].transAxes, fontsize=30, color='black', weight="bold")
    #ax[6, 1].text(0.04, 0.8, r"B7", transform=ax[6, 1].transAxes, fontsize=30, color='black', weight="bold")
    #ax[7, 1].text(0.04, 0.8, r"B8", transform=ax[7, 1].transAxes, fontsize=30, color='black', weight="bold")

    ax[0, 2].text(0.04, 0.8, r"C1", transform=ax[0, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[1, 2].text(0.04, 0.8, r"C2", transform=ax[1, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[2, 2].text(0.04, 0.8, r"C3", transform=ax[2, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[3, 2].text(0.04, 0.8, r"C4", transform=ax[3, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[4, 2].text(0.04, 0.8, r"C5", transform=ax[4, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[5, 2].text(0.04, 0.8, r"C6", transform=ax[5, 2].transAxes, fontsize=30, color='white', weight="bold")
    #ax[6, 2].text(0.04, 0.8, r"C7", transform=ax[6, 2].transAxes, fontsize=30, color='white', weight="bold")
    #ax[7, 2].text(0.04, 0.8, r"C8", transform=ax[7, 2].transAxes, fontsize=30, color='white', weight="bold")

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
    base_name = 'thrf1_' + target_net + '_' + str(target_k)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_threshold_hom_f1b(target_net='er', target_k=10, target_zef=0.0, target_var=None):

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_global_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

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

        if net_flag == target_net and k_flag == target_k and zef_flag == target_zef and var_flag in target_var:

            global_output = ut.load_global_output(epi_fullname)

            filtered_output = ut.filter_global_output(global_output, n, prevalence_cutoff)

            stat_output = ut.stat_global_output(filtered_output)

            if net_flag not in go_dict:
                go_dict[net_flag] = {}

            if k_flag not in go_dict[net_flag]:
                go_dict[net_flag][k_flag] = {}
            
            if var_flag not in go_dict[net_flag][k_flag]:
                go_dict[net_flag][k_flag][var_flag] = {}

            key_tuple = (thr_flag, acf_flag)
            go_dict[net_flag][k_flag][var_flag][key_tuple] = stat_output

    for net_flag, net_data in go_dict.items():
        for k_flag, k_data in net_data.items():
            sorted_keys = sorted(k_data.keys())
            sorted_items = [(key, k_data[key]) for key in sorted_keys]
            go_dict[net_flag][k_flag] = dict(sorted_items)

    fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(18, 20))
    cmap = 'viridis'
    
    for net_flag in go_dict.keys():

        for v, var_flag in enumerate(go_dict[net_flag][k_flag].keys()):

            go_net_k_var_dict = go_dict[net_flag][k_flag][var_flag]

            par1_array = sorted(list(set([key[0] for key in go_net_k_var_dict.keys()]))) # THRESHOLD
            par2_array = sorted(list(set([key[1] for key in go_net_k_var_dict.keys()]))) # ACTIVE FRACTION

            par1_len = len(par1_array)
            par2_len = len(par2_array)

            X, Y = np.meshgrid(par1_array, par2_array)
            TTP = np.zeros((par2_len, par1_len))
            CAP = np.zeros((par2_len, par1_len))
            VAP = np.zeros((par2_len, par1_len))

            for i, p1 in enumerate(par1_array):
                for j, p2 in enumerate(par2_array):
                    if (p1, p2) in go_net_k_var_dict:

                        TTP[j][i] = go_net_k_var_dict[(p1, p2)]['time_to_peak']['avg']

                        max_change = 1.0 - p2
                        norm_change = (go_net_k_var_dict[(p1, p2)]['convinced_at_peak']['avg'] - p2) / max_change
                        if p2 == 1.0:
                            norm_change = 0.0
                        CAP[j][i] = norm_change
                        VAP[j][i] = go_net_k_var_dict[(p1, p2)]['vaccinated_at_peak']['avg']

                    else:
                        TTP[j][i] = np.nan
                        CAP[j][i] = np.nan
                        VAP[j][i] = np.nan

            if net_flag == 'er':
                im2 = ax[v, 2].pcolormesh(X, Y, VAP, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
                im1 = ax[v, 1].pcolormesh(X, Y, CAP, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
                im0 = ax[v, 0].pcolormesh(X, Y, TTP, shading='auto', cmap=cmap, vmin=0.0, vmax=100)

                cb2 = plt.colorbar(im2, ax=ax[v, 2])
                cb2.ax.tick_params(labelsize=18)
                cb1 = plt.colorbar(im1, ax=ax[v, 1])
                cb1.ax.tick_params(labelsize=18)
                cb0 = plt.colorbar(im0, ax=ax[v, 0])
                cb0.ax.tick_params(labelsize=18)

            elif net_flag == 'ba':
                im2 = ax[v, 2].pcolormesh(X, Y, VAP, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
                im1 = ax[v, 1].pcolormesh(X, Y, CAP, shading='auto', cmap=cmap, vmin=0.0, vmax=1.0)
                im0 = ax[v, 0].pcolormesh(X, Y, TTP, shading='auto', cmap=cmap, vmin=0.0, vmax=100)

                cb2 = plt.colorbar(im2, ax=ax[v, 2])
                cb2.ax.tick_params(labelsize=18)
                cb1 = plt.colorbar(im1, ax=ax[v, 1])
                cb1.ax.tick_params(labelsize=18)
                cb0 = plt.colorbar(im0, ax=ax[v, 0])
                cb0.ax.tick_params(labelsize=18)

    ax[0, 0].set_title(r"$t_{{peak}}$", fontsize=35)
    ax[0, 1].set_title(r"$\Delta n_A(t_{{peak}})$", fontsize=35)
    ax[0, 2].set_title(r"$v(t_{{peak}})$", fontsize=35)

    ax[0, 0].text(-0.35, 0.15, r'$\alpha=0.001$', fontsize=30, transform=ax[0, 0].transAxes, rotation='vertical')
    ax[1, 0].text(-0.35, 0.12, r'$\alpha=0.0025$', fontsize=30, transform=ax[1, 0].transAxes, rotation='vertical')
    ax[2, 0].text(-0.35, 0.15, r'$\alpha=0.005$', fontsize=30, transform=ax[2, 0].transAxes, rotation='vertical')
    ax[3, 0].text(-0.35, 0.2, r'$\alpha=0.01$', fontsize=30, transform=ax[3, 0].transAxes, rotation='vertical')
    ax[4, 0].text(-0.35, 0.2, r'$\alpha=0.05$', fontsize=30, transform=ax[4, 0].transAxes, rotation='vertical')
    ax[5, 0].text(-0.35, 0.25, r'$\alpha=1.0$', fontsize=30, transform=ax[5, 0].transAxes, rotation='vertical')
    #ax[6, 0].text(-0.32, 0.25, r'$\alpha=0.05$', fontsize=30, transform=ax[6, 0].transAxes, rotation='vertical')
    #ax[7, 0].text(-0.32, 0.25, r'$\alpha=1.0$', fontsize=30, transform=ax[7, 0].transAxes, rotation='vertical')
    
    ax[5, 0].set_xlabel(r'$\theta$', fontsize=30)
    ax[5, 1].set_xlabel(r'$\theta$', fontsize=30)
    ax[5, 2].set_xlabel(r'$\theta$', fontsize=30)

    ax[0, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[1, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[2, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[3, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[4, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[5, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    #ax[6, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    #ax[7, 0].set_ylabel(r'$n_A(0)$', fontsize=30)

    ax[0, 0].text(0.04, 0.8, r"A1", transform=ax[0, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 0].text(0.04, 0.8, r"A2", transform=ax[1, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 0].text(0.04, 0.8, r"A3", transform=ax[2, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[3, 0].text(0.04, 0.8, r"A4", transform=ax[3, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[4, 0].text(0.04, 0.8, r"A5", transform=ax[4, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[5, 0].text(0.04, 0.8, r"A6", transform=ax[5, 0].transAxes, fontsize=30, color='white', weight="bold")
    #ax[6, 0].text(0.04, 0.8, r"A7", transform=ax[6, 0].transAxes, fontsize=30, color='black', weight="bold")
    #ax[7, 0].text(0.04, 0.8, r"A8", transform=ax[7, 0].transAxes, fontsize=30, color='black', weight="bold")

    ax[0, 1].text(0.04, 0.8, r"B1", transform=ax[0, 1].transAxes, fontsize=30, color='white', weight="bold")
    ax[1, 1].text(0.04, 0.8, r"B2", transform=ax[1, 1].transAxes, fontsize=30, color='white', weight="bold")
    ax[2, 1].text(0.04, 0.8, r"B3", transform=ax[2, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[3, 1].text(0.04, 0.8, r"B4", transform=ax[3, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[4, 1].text(0.04, 0.8, r"B5", transform=ax[4, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[5, 1].text(0.04, 0.8, r"B6", transform=ax[5, 1].transAxes, fontsize=30, color='black', weight="bold")
    #ax[6, 1].text(0.04, 0.8, r"B7", transform=ax[6, 1].transAxes, fontsize=30, color='black', weight="bold")
    #ax[7, 1].text(0.04, 0.8, r"B8", transform=ax[7, 1].transAxes, fontsize=30, color='black', weight="bold")

    ax[0, 2].text(0.04, 0.8, r"C1", transform=ax[0, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[1, 2].text(0.04, 0.8, r"C2", transform=ax[1, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[2, 2].text(0.04, 0.8, r"C3", transform=ax[2, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[3, 2].text(0.04, 0.8, r"C4", transform=ax[3, 2].transAxes, fontsize=30, color='white', weight="bold")
    ax[4, 2].text(0.04, 0.8, r"C5", transform=ax[4, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[5, 2].text(0.04, 0.8, r"C6", transform=ax[5, 2].transAxes, fontsize=30, color='black', weight="bold")
    #ax[6, 2].text(0.04, 0.8, r"C7", transform=ax[6, 2].transAxes, fontsize=30, color='white', weight="bold")
    #ax[7, 2].text(0.04, 0.8, r"C8", transform=ax[7, 2].transAxes, fontsize=30, color='white', weight="bold")

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
    base_name = 'thrf1b_hom_' + target_net + '_' + str(target_k)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_threshold_zea_f1(target_net='er', target_k=10, target_var=0.001):

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_global_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

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

        if net_flag == target_net and k_flag == target_k and var_flag == target_var:

            global_output = ut.load_global_output(epi_fullname)

            filtered_output = ut.filter_global_output(global_output, n, prevalence_cutoff)

            stat_output = ut.stat_global_output(filtered_output)

            if net_flag not in go_dict:
                go_dict[net_flag] = {}

            if k_flag not in go_dict[net_flag]:
                go_dict[net_flag][k_flag] = {}
            
            if zef_flag not in go_dict[net_flag][k_flag]:
                go_dict[net_flag][k_flag][zef_flag] = {}

            key_tuple = (thr_flag, acf_flag)
            go_dict[net_flag][k_flag][zef_flag][key_tuple] = stat_output

    for net_flag, net_data in go_dict.items():
        for k_flag, k_data in net_data.items():
            sorted_keys = sorted(k_data.keys())
            sorted_items = [(key, k_data[key]) for key in sorted_keys]
            go_dict[net_flag][k_flag] = dict(sorted_items)

    fig, ax = plt.subplots(nrows=6, ncols=3, figsize=(18, 16))
    cmap = 'viridis'
    
    for net_flag in go_dict.keys():

        for z, zef_flag in enumerate(go_dict[net_flag][k_flag].keys()):

            go_net_k_zef_dict = go_dict[net_flag][k_flag][zef_flag]

            par1_array = sorted(list(set([key[0] for key in go_net_k_zef_dict.keys()])))
            par2_array = sorted(list(set([key[1] for key in go_net_k_zef_dict.keys()])))

            par1_len = len(par1_array)
            par2_len = len(par2_array)

            X, Y = np.meshgrid(par1_array, par2_array)
            R = np.zeros((par2_len, par1_len))
            V = np.zeros((par2_len, par1_len))
            C = np.zeros((par2_len, par1_len))

            for i, p1 in enumerate(par1_array):
                for j, p2 in enumerate(par2_array):
                    if (p1, p2) in go_net_k_zef_dict:

                        R[j][i] = go_net_k_zef_dict[(p1, p2)]['prevalence']['avg']
                        V[j][i] = go_net_k_zef_dict[(p1, p2)]['vaccinated']['avg']
                        
                        max_change = 1.0 - p2
                        norm_change = (go_net_k_zef_dict[(p1, p2)]['convinced']['avg'] - p2) / max_change
                        if p2 == 1.0:
                            norm_change = 0.0
                        C[j][i] = norm_change
                    else:
                        R[j][i] = np.nan
                        V[j][i] = np.nan
                        C[j][i] = np.nan

            if net_flag == 'er':
                im0 = ax[z, 0].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=np.max(R))
                im1 = ax[z, 1].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=np.max(V))
                im2 = ax[z, 2].pcolormesh(X, Y, C, shading='auto', cmap=cmap, vmin=0.0, vmax=np.max(C))

                cb0 = plt.colorbar(im0, ax=ax[z, 0])
                cb0.ax.tick_params(labelsize=18)
                cb1 = plt.colorbar(im1, ax=ax[z, 1])
                cb1.ax.tick_params(labelsize=18)
                cb2 = plt.colorbar(im2, ax=ax[z, 2])
                cb2.ax.tick_params(labelsize=18)
    
            elif net_flag == 'ba':
                im0 = ax[z, 0].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=np.max(R))
                im1 = ax[z, 1].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=np.max(V))
                im2 = ax[z, 2].pcolormesh(X, Y, C, shading='auto', cmap=cmap, vmin=0.0, vmax=np.max(C))

                cb0 = plt.colorbar(im0, ax=ax[z, 0])
                cb0.ax.tick_params(labelsize=18)
                cb1 = plt.colorbar(im1, ax=ax[z, 1])
                cb1.ax.tick_params(labelsize=18)
                cb2 = plt.colorbar(im2, ax=ax[z, 2])
                cb2.ax.tick_params(labelsize=18)

    ax[0, 0].set_title(r"$r(\infty)$", fontsize=35)
    ax[0, 1].set_title(r"$v(\infty)$", fontsize=35)
    ax[0, 2].set_title(r"$n_A(\infty)$", fontsize=35)

    ax[0, 0].text(-0.35, 0.25, r'$n_Z=0$', fontsize=30, transform=ax[0, 0].transAxes, rotation='vertical')
    ax[1, 0].text(-0.35, 0.25, r'$n_Z=0.01$', fontsize=30, transform=ax[1, 0].transAxes, rotation='vertical')
    ax[2, 0].text(-0.35, 0.25, r'$n_Z=0.05$', fontsize=30, transform=ax[2, 0].transAxes, rotation='vertical')
    ax[3, 0].text(-0.35, 0.25, r'$n_Z=0.1$', fontsize=30, transform=ax[3, 0].transAxes, rotation='vertical')
    ax[4, 0].text(-0.35, 0.25, r'$n_Z=0.25$', fontsize=30, transform=ax[4, 0].transAxes, rotation='vertical')
    ax[5, 0].text(-0.35, 0.25, r'$n_Z=0.5$', fontsize=30, transform=ax[5, 0].transAxes, rotation='vertical')
    
    ax[5, 0].set_xlabel(r'$\theta$', fontsize=30)
    ax[5, 1].set_xlabel(r'$\theta$', fontsize=30)
    ax[5, 2].set_xlabel(r'$\theta$', fontsize=30)

    ax[0, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[1, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[2, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[3, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[4, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[5, 0].set_ylabel(r'$n_A(0)$', fontsize=30)

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
    base_name = 'fig1_zea_' + target_net + '_' + str(target_k) + '_' + str(target_var)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_threshold_zea_f1b(target_net='er', target_k=10, target_var=None, target_zef=None):

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_global_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

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

        if net_flag == target_net and k_flag == target_k and var_flag in target_var and zef_flag in target_zef:

            global_output = ut.load_global_output(epi_fullname)

            filtered_output = ut.filter_global_output(global_output, n, prevalence_cutoff)

            stat_output = ut.stat_global_output(filtered_output)

            if net_flag not in go_dict:
                go_dict[net_flag] = {}

            if k_flag not in go_dict[net_flag]:
                go_dict[net_flag][k_flag] = {}

            if var_flag not in go_dict[net_flag][k_flag]:
                go_dict[net_flag][k_flag][var_flag] = {}

            if zef_flag not in go_dict[net_flag][k_flag][var_flag]:
                go_dict[net_flag][k_flag][var_flag][zef_flag] = {}

            if acf_flag <= 0.5:
                key_tuple = (thr_flag, acf_flag)
                go_dict[net_flag][k_flag][var_flag][zef_flag][key_tuple] = stat_output

    for net_flag, net_data in go_dict.items():
        for k_flag, k_data in net_data.items():
            sorted_keys = sorted(k_data.keys())
            sorted_items = [(key, k_data[key]) for key in sorted_keys]
            go_dict[net_flag][k_flag] = dict(sorted_items)
            for z_flag, z_data in k_data.items():
                sorted_keys = sorted(z_data.keys())
                sorted_items = [(key, z_data[key]) for key in sorted_keys]
                go_dict[net_flag][k_flag][z_flag] = dict(sorted_items)

    fig, ax = plt.subplots(nrows=3, ncols=5, figsize=(22, 12))
    cmap = 'viridis'
    
    for net_flag in go_dict.keys():

        for v, var_flag in enumerate(go_dict[net_flag][k_flag].keys()):

            for z, zef_flag in enumerate(go_dict[net_flag][k_flag][var_flag].keys()):

                go_net_k_var_zef_dict = go_dict[net_flag][k_flag][var_flag][zef_flag]

                par1_array = sorted(list(set([key[0] for key in go_net_k_var_zef_dict.keys()])))
                par2_array = sorted(list(set([key[1] for key in go_net_k_var_zef_dict.keys()])))

                par1_len = len(par1_array)
                par2_len = len(par2_array)

                X, Y = np.meshgrid(par1_array, par2_array)
                R = np.zeros((par2_len, par1_len))
                V = np.zeros((par2_len, par1_len))
                C = np.zeros((par2_len, par1_len))

                for i, p1 in enumerate(par1_array):
                    for j, p2 in enumerate(par2_array):
                        if (p1, p2) in go_net_k_var_zef_dict:
                            R[j][i] = go_net_k_var_zef_dict[(p1, p2)]['prevalence']['avg']
                            V[j][i] = go_net_k_var_zef_dict[(p1, p2)]['vaccinated']['avg']

                            max_change = 1.0 - p2
                            norm_change = (go_net_k_var_zef_dict[(p1, p2)]['convinced']['avg'] - p2) / max_change
                            if p2 == 1.0:
                                norm_change = 0.0
                            C[j][i] = norm_change

                        else:
                            R[j][i] = np.nan
                            V[j][i] = np.nan
                            C[j][i] = np.nan

                if net_flag == 'er':

                    if v == 0:
                        c_vmax = 1.0
                        v_vmax = 0.2
                        r_vmax = 0.5
                    elif v == 1:
                        c_vmax = 1.0
                        v_vmax = 0.5
                        r_vmax = 0.5
                    elif v == 2:
                        c_vmax = 1.0
                        v_vmax = 1.0
                        r_vmax = 0.5

                    if z == 0:
                        im0 = ax[v, z].pcolormesh(X, Y, C, shading='auto', cmap=cmap, vmin=0.0, vmax=c_vmax)
                        cb0 = plt.colorbar(im0, ax=ax[v, z])
                        cb0.ax.tick_params(labelsize=18)

                        contour_levels = [0.1]  # Adjust these levels as needed
                        ax[v, 0].contour(X, Y, C, levels=contour_levels, colors='white', linestyles='dashed')
                    
                    im1 = ax[v, z + 1].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=v_vmax)
                    im2 = ax[v, z + 3].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=r_vmax) 

                    contour_levels = [0.0195]  # Adjust these levels as needed
                    if v == 1:
                        contour_levels = [0.07]
                    ax[v, z + 3].contour(X, Y, R, levels=contour_levels, colors='white', linestyles='dashed')

                    cb1 = plt.colorbar(im1, ax=ax[v, z + 1])
                    cb1.ax.tick_params(labelsize=18)
                    cb2 = plt.colorbar(im2, ax=ax[v, z + 3])
                    cb2.ax.tick_params(labelsize=18)

                    if v == 1:
                        p0 = (0.1, 0.1)
                        p1 = (0.1, 0.25)
                        p2 = (0.1, 0.5)
                        p3 = (0.25, 0.25)
                        p4 = (0.25, 0.5)
                        p5 = (0.5, 0.5)

                        if z == 0:
                            ax[v, z].scatter(p0[0], p0[1], color='red')
                            ax[v, z].scatter(p1[0], p1[1], color='red')
                            ax[v, z].scatter(p2[0], p2[1], color='red')
                            ax[v, z].scatter(p3[0], p3[1], color='red')
                            ax[v, z].scatter(p4[0], p4[1], color='red')
                            ax[v, z].scatter(p5[0], p5[1], color='red')
                        
                        ax[v, z + 1].scatter(p0[0], p0[1], color='red')
                        ax[v, z + 3].scatter(p0[0], p0[1], color='red')

                        ax[v, z + 1].scatter(p1[0], p1[1], color='red')
                        ax[v, z + 3].scatter(p1[0], p1[1], color='red')
 
                        ax[v, z + 1].scatter(p2[0], p2[1], color='red')
                        ax[v, z + 3].scatter(p2[0], p2[1], color='red')

                        ax[v, z + 1].scatter(p3[0], p3[1], color='red')
                        ax[v, z + 3].scatter(p3[0], p3[1], color='red')

                        ax[v, z + 1].scatter(p4[0], p4[1], color='red')
                        ax[v, z + 3].scatter(p4[0], p4[1], color='red')

                        ax[v, z + 1].scatter(p5[0], p5[1], color='red')
                        ax[v, z + 3].scatter(p5[0], p5[1], color='red')

                elif net_flag == 'ba':

                    if v == 0:
                        c_vmax = 1.0
                        v_vmax = 1.0
                        r_vmax = 1.0
                    elif v == 1:
                        c_vmax = 1.0
                        v_vmax = 1.0
                        r_vmax = 1.0
                    elif v == 2:
                        c_vmax = 1.0
                        v_vmax = 1.0
                        r_vmax = 1.0
            
                    if z == 0:
                        im0 = ax[v, z].pcolormesh(X, Y, C, shading='auto', cmap=cmap, vmin=0.0, vmax=c_vmax)
                        cb0 = plt.colorbar(im0, ax=ax[v, z])
                        cb0.ax.tick_params(labelsize=18)
                    
                    im1 = ax[v, z + 1].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=v_vmax)
                    im2 = ax[v, z + 3].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=r_vmax) 

                    cb1 = plt.colorbar(im1, ax=ax[v, z + 1])
                    cb1.ax.tick_params(labelsize=18)
                    cb2 = plt.colorbar(im2, ax=ax[v, z + 3])
                    cb2.ax.tick_params(labelsize=18)

                    if v == 2:
                        p0 = (0.1, 0.1)
                        p1 = (0.1, 0.2)
                        p2 = (0.1, 0.5)
                        p3 = (0.5, 0.1)
                        p4 = (0.5, 0.2)
                        p5 = (0.5, 0.5)

                        if z == 0:
                            ax[v, z].scatter(p0[0], p0[1], color='red')
                            ax[v, z].scatter(p1[0], p1[1], color='red')
                            ax[v, z].scatter(p2[0], p2[1], color='red')
                            ax[v, z].scatter(p3[0], p3[1], color='red')
                            ax[v, z].scatter(p4[0], p4[1], color='red')
                            ax[v, z].scatter(p5[0], p5[1], color='red')
                        
                        ax[v, z + 1].scatter(p0[0], p0[1], color='red')
                        ax[v, z + 3].scatter(p0[0], p0[1], color='red')

                        ax[v, z + 1].scatter(p1[0], p1[1], color='red')
                        ax[v, z + 3].scatter(p1[0], p1[1], color='red')
 
                        ax[v, z + 1].scatter(p2[0], p2[1], color='red')
                        ax[v, z + 3].scatter(p2[0], p2[1], color='red')

                        ax[v, z + 1].scatter(p3[0], p3[1], color='red')
                        ax[v, z + 3].scatter(p3[0], p3[1], color='red')

                        ax[v, z + 1].scatter(p4[0], p4[1], color='red')
                        ax[v, z + 3].scatter(p4[0], p4[1], color='red')

                        ax[v, z + 1].scatter(p5[0], p5[1], color='red')
                        ax[v, z + 3].scatter(p5[0], p5[1], color='red')

    ax[0, 0].set_title(r"$\Delta n_A(\infty)$ at $n_Z={0}$".format(target_zef[0]), fontsize=30)
    ax[0, 1].set_title(r"$v(\infty)$ at $n_Z={0}$".format(target_zef[0]), fontsize=30)
    ax[0, 2].set_title(r"$v(\infty)$ at $n_Z={0}$".format(target_zef[1]), fontsize=30)
    ax[0, 3].set_title(r"$r(\infty)$ at $n_Z={0}$".format(target_zef[0]), fontsize=30)
    ax[0, 4].set_title(r"$r(\infty)$ at $n_Z={0}$".format(target_zef[1]), fontsize=30)

    ax[0, 0].text(0.8, 0.85, r"A1", transform=ax[0, 0].transAxes, fontsize=25, color='white', weight="bold")
    ax[1, 0].text(0.8, 0.85, r"A2", transform=ax[1, 0].transAxes, fontsize=25, color='white', weight="bold")
    ax[2, 0].text(0.8, 0.85, r"A3", transform=ax[2, 0].transAxes, fontsize=25, color='white', weight="bold")
    ax[0, 1].text(0.8, 0.85, r"B1", transform=ax[0, 1].transAxes, fontsize=25, color='white', weight="bold")
    ax[1, 1].text(0.8, 0.85, r"B2", transform=ax[1, 1].transAxes, fontsize=25, color='white', weight="bold")
    ax[2, 1].text(0.8, 0.85, r"B3", transform=ax[2, 1].transAxes, fontsize=25, color='white', weight="bold")
    ax[0, 2].text(0.8, 0.85, r"C1", transform=ax[0, 2].transAxes, fontsize=25, color='white', weight="bold")
    ax[1, 2].text(0.8, 0.85, r"C2", transform=ax[1, 2].transAxes, fontsize=25, color='white', weight="bold")
    ax[2, 2].text(0.8, 0.85, r"C3", transform=ax[2, 2].transAxes, fontsize=25, color='white', weight="bold")
    ax[0, 3].text(0.8, 0.85, r"D1", transform=ax[0, 3].transAxes, fontsize=25, color='white', weight="bold")
    ax[1, 3].text(0.8, 0.85, r"D2", transform=ax[1, 3].transAxes, fontsize=25, color='white', weight="bold")
    ax[2, 3].text(0.8, 0.85, r"D3", transform=ax[2, 3].transAxes, fontsize=25, color='white', weight="bold")
    ax[0, 4].text(0.8, 0.85, r"E1", transform=ax[0, 4].transAxes, fontsize=25, color='white', weight="bold")
    ax[1, 4].text(0.8, 0.85, r"E2", transform=ax[1, 4].transAxes, fontsize=25, color='white', weight="bold")
    ax[2, 4].text(0.8, 0.85, r"E3", transform=ax[2, 4].transAxes, fontsize=25, color='white', weight="bold")

    ax[0, 0].text(-0.47, 0.22, r'$\alpha={0}$'.format(target_var[0]), fontsize=30, transform=ax[0, 0].transAxes, rotation='vertical')
    ax[1, 0].text(-0.47, 0.22, r'$\alpha={0}$'.format(target_var[1]), fontsize=30, transform=ax[1, 0].transAxes, rotation='vertical')
    ax[2, 0].text(-0.47, 0.25, r'$\alpha={0}$'.format(target_var[2]), fontsize=30, transform=ax[2, 0].transAxes, rotation='vertical')

    ax[2, 0].set_xlabel(r'$\theta$', fontsize=30)
    ax[2, 1].set_xlabel(r'$\theta$', fontsize=30)
    ax[2, 2].set_xlabel(r'$\theta$', fontsize=30)
    ax[2, 3].set_xlabel(r'$\theta$', fontsize=30)
    ax[2, 4].set_xlabel(r'$\theta$', fontsize=30)

    ax[0, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[1, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[2, 0].set_ylabel(r'$n_A(0)$', fontsize=30)

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
    base_name = 'thr2_' + target_net + '_' + str(target_k)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_threshold_zea_f1c(target_net='er', target_k=10, target_var=None, target_zef=None):

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_global_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

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

        if net_flag == target_net and k_flag == target_k and var_flag in target_var and zef_flag in target_zef:

            global_output = ut.load_global_output(epi_fullname)

            filtered_output = ut.filter_global_output(global_output, n, prevalence_cutoff)

            stat_output = ut.stat_global_output(filtered_output)

            if net_flag not in go_dict:
                go_dict[net_flag] = {}

            if k_flag not in go_dict[net_flag]:
                go_dict[net_flag][k_flag] = {}

            if var_flag not in go_dict[net_flag][k_flag]:
                go_dict[net_flag][k_flag][var_flag] = {}

            if zef_flag not in go_dict[net_flag][k_flag][var_flag]:
                go_dict[net_flag][k_flag][var_flag][zef_flag] = {}

            if acf_flag <= 0.5:
                key_tuple = (thr_flag, acf_flag)
                go_dict[net_flag][k_flag][var_flag][zef_flag][key_tuple] = stat_output

    for net_flag, net_data in go_dict.items():
        for k_flag, k_data in net_data.items():
            sorted_keys = sorted(k_data.keys())
            sorted_items = [(key, k_data[key]) for key in sorted_keys]
            go_dict[net_flag][k_flag] = dict(sorted_items)
            for z_flag, z_data in k_data.items():
                sorted_keys = sorted(z_data.keys())
                sorted_items = [(key, z_data[key]) for key in sorted_keys]
                go_dict[net_flag][k_flag][z_flag] = dict(sorted_items)

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(18, 12))
    cmap = 'viridis'
    
    for net_flag in go_dict.keys():

        for v, var_flag in enumerate(go_dict[net_flag][k_flag].keys()):

            for z, zef_flag in enumerate(go_dict[net_flag][k_flag][var_flag].keys()):

                go_net_k_var_zef_dict = go_dict[net_flag][k_flag][var_flag][zef_flag]

                par1_array = sorted(list(set([key[0] for key in go_net_k_var_zef_dict.keys()])))
                par2_array = sorted(list(set([key[1] for key in go_net_k_var_zef_dict.keys()])))

                par1_len = len(par1_array)
                par2_len = len(par2_array)

                X, Y = np.meshgrid(par1_array, par2_array)
                R = np.zeros((par2_len, par1_len))
                V = np.zeros((par2_len, par1_len))

                for i, p1 in enumerate(par1_array):
                    for j, p2 in enumerate(par2_array):
                        if (p1, p2) in go_net_k_var_zef_dict:
                            R[j][i] = go_net_k_var_zef_dict[(p1, p2)]['prevalence']['avg']
                            V[j][i] = go_net_k_var_zef_dict[(p1, p2)]['vaccinated']['avg']

                        else:
                            R[j][i] = np.nan
                            V[j][i] = np.nan

                if v == 0:
                    v_vmax = 0.2
                    r_vmax = 0.5
                elif v == 1:
                    v_vmax = 0.5
                    r_vmax = 0.5
    
                elif v == 2:
                    v_vmax = 1.0
                    r_vmax = 0.5

                if net_flag == 'er':

                    im0 = ax[v, z].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=v_vmax)
                    im1 = ax[v, z + 2].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=r_vmax) 

                    cb0 = plt.colorbar(im0, ax=ax[v, z])
                    cb0.ax.tick_params(labelsize=18)
                    cb1 = plt.colorbar(im1, ax=ax[v, z + 2])
                    cb1.ax.tick_params(labelsize=18)

                    if v == 1:
                        p0 = (0.1, 0.1)
                        p1 = (0.1, 0.25)
                        p2 = (0.1, 0.5)
                        p3 = (0.25, 0.25)
                        p4 = (0.25, 0.5)
                        p5 = (0.5, 0.5)

                        ax[v, z].scatter(p0[0], p0[1], color='goldenrod')
                        ax[v, z + 2].scatter(p0[0], p0[1], color='goldenrod')

                        ax[v, z].scatter(p1[0], p1[1], color='crimson')
                        ax[v, z + 2].scatter(p1[0], p1[1], color='crimson')

                        ax[v, z].scatter(p2[0], p2[1], color='deeppink')
                        ax[v, z + 2].scatter(p2[0], p2[1], color='deeppink')

                        ax[v, z].scatter(p3[0], p3[1], color='lightskyblue')
                        ax[v, z + 2].scatter(p3[0], p3[1], color='lightskyblue')

                        ax[v, z].scatter(p4[0], p4[1], color='firebrick')
                        ax[v, z + 2].scatter(p4[0], p4[1], color='firebrick')

                        ax[v, z].scatter(p5[0], p5[1], color='gray')
                        ax[v, z + 2].scatter(p5[0], p5[1], color='gray')

                elif net_flag == 'ba':
                    im0 = ax[v, z].pcolormesh(X, Y, V, shading='auto', cmap=cmap, vmin=0.0, vmax=v_vmax)
                    im1 = ax[v, z + 2].pcolormesh(X, Y, R, shading='auto', cmap=cmap, vmin=0.0, vmax=r_vmax)

                    cb0 = plt.colorbar(im0, ax=ax[v, z])
                    cb0.ax.tick_params(labelsize=18)
                    cb1 = plt.colorbar(im1, ax=ax[v, z + 2])
                    cb1.ax.tick_params(labelsize=18)

    ax[0, 0].set_title(r"$v(\infty)$ at $n_Z={0}$".format(target_zef[0]), fontsize=30)
    ax[0, 1].set_title(r"$v(\infty)$ at $n_Z={0}$".format(target_zef[1]), fontsize=30)
    ax[0, 2].set_title(r"$r(\infty)$ at $n_Z={0}$".format(target_zef[0]), fontsize=30)
    ax[0, 3].set_title(r"$r(\infty)$ at $n_Z={0}$".format(target_zef[1]), fontsize=30)

    ax[0, 0].text(0.8, 0.85, r"A1", transform=ax[0, 0].transAxes, fontsize=25, color='white', weight="bold")
    ax[1, 0].text(0.8, 0.85, r"A2", transform=ax[1, 0].transAxes, fontsize=25, color='white', weight="bold")
    ax[2, 0].text(0.8, 0.85, r"A3", transform=ax[2, 0].transAxes, fontsize=25, color='white', weight="bold")
    ax[0, 1].text(0.8, 0.85, r"B1", transform=ax[0, 1].transAxes, fontsize=25, color='white', weight="bold")
    ax[1, 1].text(0.8, 0.85, r"B2", transform=ax[1, 1].transAxes, fontsize=25, color='white', weight="bold")
    ax[2, 1].text(0.8, 0.85, r"B3", transform=ax[2, 1].transAxes, fontsize=25, color='white', weight="bold")
    ax[0, 2].text(0.8, 0.85, r"C1", transform=ax[0, 2].transAxes, fontsize=25, color='white', weight="bold")
    ax[1, 2].text(0.8, 0.85, r"C2", transform=ax[1, 2].transAxes, fontsize=25, color='white', weight="bold")
    ax[2, 2].text(0.8, 0.85, r"C3", transform=ax[2, 2].transAxes, fontsize=25, color='white', weight="bold")
    ax[0, 3].text(0.8, 0.85, r"D1", transform=ax[0, 3].transAxes, fontsize=25, color='white', weight="bold")
    ax[1, 3].text(0.8, 0.85, r"D2", transform=ax[1, 3].transAxes, fontsize=25, color='white', weight="bold")
    ax[2, 3].text(0.8, 0.85, r"D3", transform=ax[2, 3].transAxes, fontsize=25, color='white', weight="bold")

    ax[0, 0].text(-0.47, 0.22, r'$\alpha=0.001$', fontsize=30, transform=ax[0, 0].transAxes, rotation='vertical')
    ax[1, 0].text(-0.47, 0.22, r'$\alpha=0.005$', fontsize=30, transform=ax[1, 0].transAxes, rotation='vertical')
    ax[2, 0].text(-0.47, 0.25, r'$\alpha=1.0$', fontsize=30, transform=ax[2, 0].transAxes, rotation='vertical')
    
    ax[2, 0].set_xlabel(r'$\theta$', fontsize=30)
    ax[2, 1].set_xlabel(r'$\theta$', fontsize=30)
    ax[2, 2].set_xlabel(r'$\theta$', fontsize=30)
    ax[2, 3].set_xlabel(r'$\theta$', fontsize=30)

    ax[0, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[1, 0].set_ylabel(r'$n_A(0)$', fontsize=30)
    ax[2, 0].set_ylabel(r'$n_A(0)$', fontsize=30)

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
    base_name = 'fig1c_zea_' + target_net + '_' + str(target_k)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_threshold_zea_f2(target_net='er', target_k=10, tav_list=None):
    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_global_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

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

        tav_tuple = (thr_flag, acf_flag, var_flag)

        if net_flag in target_net and k_flag == target_k and tav_tuple in tav_list:

            global_output = ut.load_global_output(epi_fullname)

            filtered_output = ut.filter_global_output(global_output, n, prevalence_cutoff)

            stat_output = ut.stat_global_output(filtered_output)

            if net_flag not in go_dict:
                go_dict[net_flag] = {}

            if k_flag not in go_dict[net_flag]:
                go_dict[net_flag][k_flag] = {}

            if tav_tuple not in go_dict[net_flag][k_flag]:
                go_dict[net_flag][k_flag][tav_tuple] = {}

            if zef_flag not in go_dict[net_flag][k_flag][tav_tuple]:
                go_dict[net_flag][k_flag][tav_tuple][zef_flag] = stat_output

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

    er_color_tav_dict = {
        (0.1, 0.1, 0.005): 'dodgerblue',
        (0.1, 0.25, 0.005): 'crimson',
        (0.1, 0.5, 0.005): 'deeppink',
        (0.25, 0.25, 0.005): 'lightskyblue',
        (0.25, 0.5, 0.005): 'crimson',
        (0.5, 0.5, 0.005): 'firebrick',
        (0.1, 0.1, 1.0): 'deepskyblue',
        (0.1, 0.2, 1.0): 'dodgerblue',
        (0.1, 0.5, 1.0): 'steelblue',
        (0.25, 0.1, 1.0): 'crimson',
        (0.25, 0.2, 1.0): 'crimson',
        (0.25, 0.5, 1.0): 'crimson',
        (0.3, 0.1, 1.0): 'crimson',
        (0.5, 0.1, 1.0): 'deeppink',
        (0.5, 0.2, 1.0): 'firebrick',
        (0.5, 0.5, 1.0): 'crimson',
    }

    for net_flag in go_dict.keys():

        for k_flag in go_dict[net_flag].keys():

            for tav_idx, tav_tuple in enumerate(go_dict[net_flag][k_flag].keys()):

                go_net_k_tav_dict = go_dict[net_flag][k_flag][tav_tuple]

                n_z_values = sorted(go_net_k_tav_dict.keys())
                stats_values = [go_net_k_tav_dict[key] for key in n_z_values]

                prev_avg = [obs['prevalence']['avg'] for obs in list(stats_values)]
                prev_l95 = [obs['prevalence']['l95'] for obs in list(stats_values)]
                prev_u95 = [obs['prevalence']['u95'] for obs in list(stats_values)]

                vacc_avg = [obs['vaccinated']['avg'] for obs in list(stats_values)]
                vacc_l95 = [obs['vaccinated']['l95'] for obs in list(stats_values)]
                vacc_u95 = [obs['vaccinated']['u95'] for obs in list(stats_values)]

                label = r'$(\theta,n_A(0))=({0},{1})$'.format(tav_tuple[0], tav_tuple[1], tav_tuple[2])
        
                if net_flag == 'er':

                    if tav_tuple in er_color_tav_dict.keys():
                        color = er_color_tav_dict[tav_tuple]
                    else:
                        color = 'dodgerblue'

                    ax[0].scatter(n_z_values, vacc_avg, color=color, label=label)
                    ax[0].fill_between(n_z_values, vacc_l95, vacc_u95, color=color, alpha=0.2)
                    
                    ax[1].scatter(n_z_values, prev_avg, color=color, label=label)
                    ax[1].fill_between(n_z_values, prev_l95, prev_u95, color=color, alpha=0.2)

                elif net_flag == 'ba':

                    if tav_tuple in er_color_tav_dict.keys():
                        color = er_color_tav_dict[tav_tuple]
                    else:
                        color = 'dodgerblue'

                    ax[0].scatter(n_z_values, vacc_avg, color=color, label=label)
                    ax[0].fill_between(n_z_values, vacc_l95, vacc_u95, color=color, alpha=0.2)
                    
                    ax[1].scatter(n_z_values, prev_avg, color=color, label=label)
                    ax[1].fill_between(n_z_values, prev_l95, prev_u95, color=color, alpha=0.2)

    ax[0].legend(loc='upper right', fontsize=15)
    ax[1].legend(loc='lower right', fontsize=15)

    ax[0].text(0.04, 0.9, r"A", transform=ax[0].transAxes, fontsize=30, color='black', weight="bold")
    ax[1].text(0.9, 0.9, r"B", transform=ax[1].transAxes, fontsize=30, color='black', weight="bold")

    ax[0].set_xlim(0.0, 1.001)
    ax[1].set_xlim(0.0, 1.001)

    ax[0].set_ylim(0.0, 0.5)
    ax[1].set_ylim(0.0, 0.5)

    ax[0].set_xlabel(r'$n_Z$', fontsize=30)
    ax[1].set_xlabel(r'$n_Z$', fontsize=30)
   
    ax[0].set_ylabel(r'$v(\infty)$', fontsize=30)
    ax[1].set_ylabel(r'$r(\infty)$', fontsize=30)

    ax[0].tick_params(axis='both', labelsize=18)
    ax[1].tick_params(axis='both', labelsize=18)

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
    base_name = 'thrf3_' + target_net + '_' + str(target_k)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_bar_us_states_thresholds(k_avg):
    # Load data
    vaccination_results = ut.read_vaccination_data(cwd_path)

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
            ut.compute_average_theshold(vaccination_results[state], k_avg)

    # create the figure and axis objects
    fig, ax = plt.subplots(figsize=(17, 6))

    ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)

    # create the bar plot
    width = 0.4
    bar1 = ax.bar(
        np.arange(len(state_code)), 
        already, 
        width, 
        color='deepskyblue',
        zorder=2,
        )
    ax.bar(
        np.arange(len(state_code)), 
        soon, 
        width, 
        color='dodgerblue', 
        bottom=already,
        zorder=2,
    )
    ax.bar(np.arange(
        len(state_code)), 
        someone, 
        width, 
        color='royalblue', 
        bottom=np.add(soon, already),
        zorder=2,
    )
    ax.bar(
        np.arange(len(state_code)), 
        majority, 
        width, 
        color='slateblue', 
        bottom=np.add(someone, np.add(soon, already)),
        zorder=2,
    )
    ax.bar(
        np.arange(len(state_code)), 
        never, 
        width, 
        color='indigo', 
        bottom=np.add(
        majority, 
        np.add(someone, np.add(soon, already)),
        ), 
        zorder=2,
    )

    # create the right axis
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())

    x_start = np.array([plt.getp(item, 'x') - 0.3 for item in bar1])
    x_end = x_start + [plt.getp(item, 'width') + 0.6 for item in bar1]

    ax2.hlines(avg_thresholds, x_start, x_end, linestyles='dashed', linewidth=1.0, label=r'$\langle\theta\rangle$', color='crimson')

    # set the axis and title labels
    ax.set_title(r'$\theta$ distribution by state', fontsize=30)
    ax.set_ylabel(r'population fraction', fontsize=25)
    ax.set_xticks(np.arange(len(state_code)))
    ax.set_xticklabels(state_code, rotation=90) # rotate state labels
    ax2.set_ylabel(r'$\langle\theta\rangle$', fontsize=25)
    ax.tick_params(axis='both', labelsize=17)
    ax2.tick_params(axis='both', labelsize=17)
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
    lower_path = 'results/'
    full_path = os.path.join(cwd_path, lower_path)
    plot_name = 'bar_vaccination_thresholds'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, plot_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')

    #plt.show()

def plot_thr_us_vaccination(net_ids):

    vaccination_results = ut.read_vaccination_data(cwd_path)

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_global_2_')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'results/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        net_flag = epi_filename.split('_net')[1].split('_')[0]
        state_flag = epi_filename.split('_2_')[1].split('_')[0]
        var_flag = float(epi_filename.split('_var')[1].split('_')[0])
       
        if net_flag in net_ids:

            us_output = ut.load_us_output(epi_fullname)

            us_output_w1 = us_output['w1']
            us_output_w2 = us_output['w2']

            filtered_output_w1 = ut.filter_global_output(us_output_w1, n, prevalence_cutoff)
            filtered_output_w2 = ut.filter_global_output(us_output_w2, n, prevalence_cutoff)

            stat_output_w1 = ut.stat_global_output(filtered_output_w1)
            stat_output_w2 = ut.stat_global_output(filtered_output_w2)

            if net_flag not in go_dict:
                go_dict[net_flag] = {}

            if state_flag not in go_dict[net_flag]:
                go_dict[net_flag][state_flag] = {}

            if var_flag not in go_dict[net_flag][state_flag]:
                go_dict[net_flag][state_flag][var_flag] = {'w1': stat_output_w1, 'w2': stat_output_w2}

    vmin = 0.0
    vmax = 0.75
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))

    for net_flag in go_dict.keys():

        for state_idx, state_flag in enumerate(go_dict[net_flag].keys()):
            go_net_state_dict = go_dict[net_flag][state_flag]
            
            var_values = sorted(go_net_state_dict.keys())
            
            stats_values = [go_net_state_dict[key] for key in var_values]
            
            prev_avg_w1 = np.array([obs['w1']['prevalence']['avg'] for obs in list(stats_values)])
            prev_l95_w1 = np.array([obs['w1']['prevalence']['l95'] for obs in list(stats_values)])
            prev_u95_w1 = np.array([obs['w1']['prevalence']['u95'] for obs in list(stats_values)])
            
            vacc_avg = [obs['w1']['vaccinated']['avg'] for obs in list(stats_values)]
            vacc_l95 = [obs['w1']['vaccinated']['l95'] for obs in list(stats_values)]
            vacc_u95 = [obs['w1']['vaccinated']['u95'] for obs in list(stats_values)]
            
            prev_avg_w2 = np.array([obs['w2']['prevalence']['avg'] for obs in list(stats_values)])
            prev_l95_w2 = np.array([obs['w2']['prevalence']['l95'] for obs in list(stats_values)])
            prev_u95_w2 = np.array([obs['w2']['prevalence']['u95'] for obs in list(stats_values)])
            
            vac_shares = vaccination_results[state_flag]
            initial_support = vac_shares['already'] + vac_shares['soon']
            zealots = vac_shares['never']
            
            color = plt.cm.viridis(norm(initial_support))
            
            if net_flag == 'er':

                ax[0, 0].scatter(var_values, vacc_avg, color=color)
                ax[0, 0].fill_between(var_values, vacc_l95, vacc_u95, color=color, alpha=0.2)
                
                ax[0, 1].scatter(var_values, prev_avg_w1, color=color)
                ax[0, 1].fill_between(var_values, prev_l95_w1, prev_u95_w1, color=color, alpha=0.2)
                ax[0, 2].scatter(var_values, prev_avg_w2, color=color)
                ax[0, 2].fill_between(var_values, prev_l95_w2, prev_u95_w2, color=color, alpha=0.2)
            elif net_flag == 'ba':
                ax[1, 0].scatter(var_values, vacc_avg, color=color)
                ax[1, 0].fill_between(var_values, vacc_l95, vacc_u95, color=color, alpha=0.2)
                
                ax[1, 1].scatter(var_values, prev_avg_w1, color=color)
                ax[1, 1].fill_between(var_values, prev_l95_w1, prev_u95_w1, color=color, alpha=0.2)

                ax[1, 2].scatter(var_values, prev_avg_w2, color=color)
                ax[1, 2].fill_between(var_values, prev_l95_w2, prev_u95_w2, color=color, alpha=0.2)

    cbar_er = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax[0, 2], orientation='vertical')
    cbar_er.set_label(r'$v(0)+n_A(0)$', size=25)
    cbar_er.ax.tick_params(labelsize=15)

    cbar_ba = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax[1, 2], orientation='vertical')
    cbar_ba.set_label(r'$v(0)+n_A(0)$', size=25)
    cbar_ba.ax.tick_params(labelsize=15)

    #ax[0].legend(loc='upper right', fontsize=15)
    #ax[1].legend(loc='lower right', fontsize=15)

    ax[0, 0].set_title("vaccination campaign", fontsize=30)
    ax[0, 1].set_title("wave 1 ($R_0=3$)", fontsize=30)
    ax[0, 2].set_title("wave 2 ($R_0=6$)", fontsize=30)

    ax[0, 0].text(-0.35, 0.25, r'ER networks', fontsize=30, transform=ax[0, 0].transAxes, rotation='vertical')
    ax[1, 0].text(-0.35, 0.25, r'BA networks', fontsize=30, transform=ax[1, 0].transAxes, rotation='vertical')

    ax[0, 0].text(0.04, 0.9, r"A1", transform=ax[0, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 1].text(0.04, 0.9, r"B1", transform=ax[0, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 2].text(0.04, 0.9, r"C1", transform=ax[0, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 0].text(0.04, 0.9, r"A2", transform=ax[1, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 1].text(0.04, 0.9, r"B2", transform=ax[1, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 2].text(0.04, 0.9, r"C2", transform=ax[1, 2].transAxes, fontsize=30, color='black', weight="bold")

    ax[1, 0].set_xlabel(r'$\alpha$', fontsize=30)
    ax[1, 1].set_xlabel(r'$\alpha$', fontsize=30)
    ax[1, 2].set_xlabel(r'$\alpha$', fontsize=30)

    ax[0, 0].set_ylabel(r'$v(\infty)$', fontsize=30)
    ax[0, 1].set_ylabel(r'$r(\infty)$', fontsize=30)
    #ax[0, 2].set_ylabel(r'$r(\infty)$', fontsize=30)
    ax[1, 0].set_ylabel(r'$v(\infty)$', fontsize=30)
    ax[1, 1].set_ylabel(r'$r(\infty)$', fontsize=30)
    #ax[1, 2].set_ylabel(r'$r(\infty)$', fontsize=30)

    for axs in ax.flatten():
        axs.set_ylim(0.0, 1.0)
        #axs.set_xlim(0.0, 1.0)
        axs.tick_params(axis='both', which='major', labelsize=15)
        axs.set_xscale('log')
        axs.tick_params(axis='both', labelsize=18)
    
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
    base_name = 'thr_us_vaccination_curves'
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_thr_us_scatter(net_ids, target_var):
    vaccination_results = ut.read_vaccination_data(cwd_path)

    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_global_2_')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'results/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        net_flag = epi_filename.split('_net')[1].split('_')[0]
        state_flag = epi_filename.split('_2_')[1].split('_')[0]
        k_flag = epi_filename.split('_k')[1].split('_')[0]
        var_flag = float(epi_filename.split('_var')[1].split('_')[0])
       
        if net_flag in net_ids and var_flag in target_var:

            us_output = ut.load_us_output(epi_fullname)

            us_output_w1 = us_output['w1']
            us_output_w2 = us_output['w2']

            filtered_output_w1 = ut.filter_global_output(us_output_w1, n, prevalence_cutoff)
            filtered_output_w2 = ut.filter_global_output(us_output_w2, n, prevalence_cutoff)

            stat_output_w1 = ut.stat_global_output(filtered_output_w1)
            stat_output_w2 = ut.stat_global_output(filtered_output_w2)

            if net_flag not in go_dict:
                go_dict[net_flag] = {}

            if var_flag not in go_dict[net_flag]:
                go_dict[net_flag][var_flag] = {}

            if state_flag not in go_dict[net_flag][var_flag]:
                go_dict[net_flag][var_flag][state_flag] = {'w1': stat_output_w1, 'w2': stat_output_w2, 'k_avg': k_flag}

    for net_flag, net_data in go_dict.items():
        for v_flag, v_data in net_data.items():
            sorted_keys = sorted(v_data.keys())
            sorted_items = [(key, v_data[key]) for key in sorted_keys]
            go_dict[net_flag][v_flag] = dict(sorted_items)

    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

    for net_flag in go_dict.keys():

        for v, var_flag in enumerate(go_dict[net_flag].keys()):
        
            initial_support_list = []
            zealot_list = []
            prw1_avg_list = []
            prw2_avg_list = []
            vacc_avg_list = []
            effective_theta_list = []
            
            for state_idx, state_flag in enumerate(go_dict[net_flag][var_flag].keys()):
        
                outcome = go_dict[net_flag][var_flag][state_flag]
                prw1_avg = outcome['w1']['prevalence']['avg']
                vacc_avg = outcome['w1']['vaccinated']['avg']
                prw2_avg = outcome['w2']['prevalence']['avg']
                k_avg = outcome['k_avg']
                
                vac_shares = vaccination_results[state_flag]
                initial_support = vac_shares['already'] + vac_shares['soon']
                zealots = vac_shares['never']
                
                initial_support_list.append(initial_support)
                zealot_list.append(zealots)
                
                prw1_avg_list.append(prw1_avg)
                vacc_avg_list.append(vacc_avg)
                prw2_avg_list.append(prw2_avg)
        
                theta_eff = ut.compute_average_theshold(vaccination_results[state_flag], k_avg=k_avg)
                effective_theta_list.append(theta_eff)
            
            if net_flag == 'er':
        
                if v == 0:
                    vmin = 0.5
                    vmax = 0.8
                elif v == 1:
                    vmin = 0.3
                    vmax = 0.7
                elif v == 2:
                    vmin = 0.0
                    vmax = 0.3
        
                sc_er = ax[0, v].scatter(initial_support_list, effective_theta_list, c=prw1_avg_list, vmin=vmin, vmax=vmax)
                cb_er = plt.colorbar(sc_er, ax=ax[0, v])
                cb_er.ax.tick_params(labelsize=18)
                
                if v == 2:  # only for the last column
                    cb_er.set_label(r'$r(\infty)$', size=30)

            elif net_flag == 'ba':
                if v == 0:
                    vmin = 0.5
                    vmax = 0.7
                elif v == 1:
                    vmin = 0.4
                    vmax = 0.7
                elif v == 2:
                    vmin = 0.0
                    vmax = 0.5
                
                sc_ba = ax[1, v].scatter(initial_support_list, effective_theta_list, c=prw1_avg_list, vmin=vmin, vmax=vmax)
                cb_ba = plt.colorbar(sc_ba, ax=ax[1, v])
                cb_ba.ax.tick_params(labelsize=18)
                
                if v == 2:  # only for the last column
                    cb_ba.set_label(r'$r(\infty)$', size=30)
    
    #ax[0].legend(loc='upper right', fontsize=15)
    #ax[1].legend(loc='lower right', fontsize=15)    

    ax[0, 0].set_title(r"$\alpha=${0}".format(target_var[0]), fontsize=30)
    ax[0, 1].set_title(r"$\alpha=${0}".format(target_var[1]), fontsize=30)
    ax[0, 2].set_title(r"$\alpha=${0}".format(target_var[2]), fontsize=30)

    ax[0, 0].text(0.8, 0.8, r"A1", transform=ax[0, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 1].text(0.8, 0.8, r"B1", transform=ax[0, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 2].text(0.8, 0.8, r"C1", transform=ax[0, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 0].text(0.8, 0.8, r"A2", transform=ax[1, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 1].text(0.8, 0.8, r"B2", transform=ax[1, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 2].text(0.8, 0.8, r"C2", transform=ax[1, 2].transAxes, fontsize=30, color='black', weight="bold")

    ax[1, 0].set_xlabel(r'$v(0)+n_A(0)$', fontsize=30)
    ax[1, 1].set_xlabel(r'$v(0)+n_A(0)$', fontsize=30)
    ax[1, 2].set_xlabel(r'$v(0)+n_A(0)$', fontsize=30)

    ax[0, 0].set_ylabel(r'$n_Z$', fontsize=30)
    ax[1, 0].set_ylabel(r'$n_Z$', fontsize=30)

    for axs in ax.flatten():
        axs.set_xlim(0.3, 0.7)
        axs.set_ylim(0.3, 0.7)
        axs.tick_params(axis='both', labelsize=17)
    
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
    base_name = 'us_scatter_' + str(target_var)
    extension_list = ['pdf', 'png']
    if not os.path.exists(full_path):
        os.makedirs(full_path)
    for ext in extension_list:
        full_name = os.path.join(full_path, base_name + '.' + ext)
        plt.savefig(full_name, format=ext, bbox_inches='tight')
    plt.clf()

def plot_thr_us_zealots(target_net, target_var):
    lower_path = 'results/'
    fullpath = os.path.join(cwd_path, lower_path)
    epi_filenames = ut.collect_pickle_filenames(fullpath=fullpath, header='thr_mc_global_1_net')

    n = 20000
    prevalence_cutoff = 0.05

    go_dict = {}

    for i, epi_filename in enumerate(epi_filenames):
        print("Loop {0}, filename: {1}".format(i + 1, epi_filename))

        lower_path = 'results/'
        epi_fullname = os.path.join(cwd_path, lower_path, epi_filename)

        net_flag = epi_filename.split('_net')[1].split('_')[0]
        state_flag = epi_filename.split('_2_')[1].split('_')[0]
        var_flag = float(epi_filename.split('_var')[1].split('_')[0])
        zef_flag = float(epi_filename.split('_zef')[1].split('_')[0])

        if net_flag in target_net and var_flag in target_var:

            us_output = ut.load_us_output(epi_fullname)

            us_output_w1 = us_output['w1']
            us_output_w2 = us_output['w2']

            filtered_output_w1 = ut.filter_global_output(us_output_w1, n, prevalence_cutoff)
            filtered_output_w2 = ut.filter_global_output(us_output_w2, n, prevalence_cutoff)

            stat_output_w1 = ut.stat_global_output(filtered_output_w1)
            stat_output_w2 = ut.stat_global_output(filtered_output_w2)

            if net_flag not in go_dict:
                go_dict[net_flag] = {}

            if var_flag not in go_dict[net_flag]:
                go_dict[net_flag][var_flag] = {}

            if state_flag not in go_dict[net_flag][var_flag]:
                go_dict[net_flag][var_flag][state_flag] = {}

            if zef_flag not in go_dict[net_flag][var_flag][state_flag]:
                go_dict[net_flag][var_flag][state_flag][zef_flag] = {'w1': stat_output_w1, 'w2': stat_output_w2}

    for net_flag, net_data in go_dict.items():
        for v_flag, v_data in net_data.items():
            sorted_keys = sorted(v_data.keys())
            sorted_items = [(key, v_data[key]) for key in sorted_keys]
            go_dict[net_flag][v_flag] = dict(sorted_items)

    fig, ax = plt.subplots(nrows=3, ncols=4, figsize=(20, 14))

    for net_flag in go_dict.keys():

        for v, var_flag in enumerate(go_dict[net_flag].keys()):

            for state in go_dict[net_flag][var_flag].keys():

                go_net_var_dict = go_dict[net_flag][var_flag][state]

                n_z_values = sorted(go_net_var_dict.keys())
                stats_values = [go_net_var_dict[key] for key in n_z_values]

                prev_avg = [obs['w1']['prevalence']['avg'] for obs in list(stats_values)]
                prev_l95 = [obs['w1']['prevalence']['l95'] for obs in list(stats_values)]
                prev_u95 = [obs['w1']['prevalence']['u95'] for obs in list(stats_values)]
                vacc_avg = [obs['w1']['vaccinated']['avg'] for obs in list(stats_values)]
                vacc_l95 = [obs['w1']['vaccinated']['l95'] for obs in list(stats_values)]
                vacc_u95 = [obs['w1']['vaccinated']['u95'] for obs in list(stats_values)]
            
                if net_flag == 'er':
                    ax[v, 0].scatter(n_z_values, vacc_avg)
                    ax[v, 0].fill_between(n_z_values, vacc_l95, vacc_u95, alpha=0.2)

                    ax[v, 1].scatter(n_z_values, prev_avg)
                    ax[v, 1].fill_between(n_z_values, prev_l95, prev_u95, alpha=0.2)

                elif net_flag == 'ba':
                    ax[v, 2].scatter(n_z_values, vacc_avg)
                    ax[v, 2].fill_between(n_z_values, vacc_l95, vacc_u95, alpha=0.2)

                    ax[v, 3].scatter(n_z_values, prev_avg)
                    ax[v, 3].fill_between(n_z_values, prev_l95, prev_u95, alpha=0.2)

    ax[0, 0].text(0.04, 0.9, r"A1", transform=ax[0, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 1].text(0.9, 0.9, r"B1", transform=ax[0, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 2].text(0.9, 0.9, r"C1", transform=ax[0, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[0, 3].text(0.9, 0.9, r"D1", transform=ax[0, 3].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 0].text(0.04, 0.9, r"A2", transform=ax[1, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 1].text(0.9, 0.9, r"B2", transform=ax[1, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 2].text(0.9, 0.9, r"C2", transform=ax[1, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[1, 3].text(0.9, 0.9, r"D2", transform=ax[1, 3].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 0].text(0.04, 0.9, r"A3", transform=ax[2, 0].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 1].text(0.9, 0.9, r"B3", transform=ax[2, 1].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 2].text(0.9, 0.9, r"C3", transform=ax[2, 2].transAxes, fontsize=30, color='black', weight="bold")
    ax[2, 3].text(0.9, 0.9, r"D3", transform=ax[2, 3].transAxes, fontsize=30, color='black', weight="bold")

    for axs in ax.flatten():
        axs.set_xlim(0.0, 1.001)
        axs.set_ylom(0.0, 1.001)
        axs.tick_params(axis='both', labelsize=18)
    
    ax[2, 0].set_xlabel(r'$n_Z$', fontsize=30)
    ax[2, 1].set_xlabel(r'$n_Z$', fontsize=30)
    ax[2, 2].set_xlabel(r'$n_Z$', fontsize=30)
    ax[2, 3].set_xlabel(r'$n_Z$', fontsize=30)

    ax[0, 0].set_ylabel(r'$v(\infty)$', fontsize=30)
    ax[0, 1].set_ylabel(r'$r(\infty)$', fontsize=30)
    ax[0, 2].set_ylabel(r'$v(\infty)$', fontsize=30)
    ax[0, 3].set_ylabel(r'$r(\infty)$', fontsize=30)
    ax[1, 0].set_ylabel(r'$v(\infty)$', fontsize=30)
    ax[1, 1].set_ylabel(r'$r(\infty)$', fontsize=30)
    ax[1, 2].set_ylabel(r'$v(\infty)$', fontsize=30)
    ax[1, 3].set_ylabel(r'$r(\infty)$', fontsize=30)
    ax[2, 0].set_ylabel(r'$v(\infty)$', fontsize=30)
    ax[2, 1].set_ylabel(r'$r(\infty)$', fontsize=30)
    ax[2, 2].set_ylabel(r'$v(\infty)$', fontsize=30)
    ax[2, 3].set_ylabel(r'$r(\infty)$', fontsize=30)

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
    base_name = 'thr_us_zealots'
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
    
    target_var = [0.001, 0.0025, 0.005, 0.01, 0.05, 1.0]
    #plot_threshold_hom_f1(target_net=net_id, target_k=target_k, target_var=target_var)

    #plot_threshold_hom_f1b(target_net=net_id, target_k=10, target_var=target_var)

    target_var = [0.001, 0.005, 1]
    target_zef = [0, 0.5]
    #plot_threshold_zea_f1b(target_net=net_id, target_k=10, target_var=target_var, target_zef=target_zef)

    tav_list = [
        (0.1, 0.1, 0.005),
        #(0.1, 0.25, 0.005), 
        #(0.1, 0.25, 0.01), 
        #(0.1, 0.25, 0.05),
        (0.1, 0.5, 0.005), 
        #(0.1, 0.5, 0.01), 
        #(0.1, 0.5, 0.05),
        #(0.25, 0.5, 0.005), 
        (0.25, 0.25, 0.005), 
        #(0.25, 0.25, 0.01), 
        #(0.25, 0.25, 0.05),
        (0.25, 0.5, 0.005), 
        #(0.25, 0.5, 0.01), 
        #(0.25, 0.5, 0.05),
        #(0.25, 0.75, 0.005), 
        #(0.25, 0.75, 0.01), 
        #(0.25, 0.75, 0.05),
        #(0.5, 0.25, 0.005), 
        #(0.5, 0.25, 0.01), 
        #(0.5, 0.25, 0.05),
        #(0.5, 0.5, 0.005), 
        #(0.5, 0.5, 0.01),
        (0.5, 0.5, 0.005),
        #(0.5, 0.75, 0.005), 
        #(0.5, 0.75, 0.01), 
        #(0.5, 0.75, 0.05),
        #(0.75, 0.5, 0.005),
        #(0.1, 0.1, 1.0),
        #(0.1, 0.2, 1.0),
        #(0.1, 0.5, 1.0),
        #(0.1, 0.2, 1.0),
        #(0.25, 0.2, 1.0),
        #(0.5, 0.2, 1.0),
        #(0.7, 0.1, 1.0),
        #(0.1, 0.5, 1.0),
        #(0.25, 0.5, 1.0),
        #(0.5, 0.1, 1.0),
        #(0.5, 0.2, 1.0),
        #(0.5, 0.5, 1.0),
        ]

    plot_threshold_zea_f2(target_net=net_id, target_k=target_k, tav_list=tav_list)

    #plot_bar_us_states_thresholds(k_avg=10)

    #plot_thr_us_vaccination(net_ids=['er', 'ba'])

    #plot_thr_us_scatter(net_ids=['er', 'ba'], target_var=[0.001, 0.01, 0.050119])

    

if __name__ == '__main__':
    main()