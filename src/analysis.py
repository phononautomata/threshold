import os
import numpy as np
import pickle as pk

import utils as ut
import plots as pt

path = os.getcwd()


def sir_prevalence(r0):
    # Initialize r_inf
    r_inf = 0.0
    # Self-consistent solver for r_inf
    guess = 0.8
    escape = 0
    condition = True
    while condition:
        r_inf = 1.0 - np.exp(-(r0 * guess))
        if r_inf == guess:
            condition = False
        guess = r_inf
        escape += 1
        if escape > 10000:
            r_inf = 0.0
            condition = False
    return r_inf

def sir_prevalence_array(control_array):
    # Initialize r_inf
    r_inf = np.zeros(len(control_array))
    # Self-consistent solver for r_inf
    for i in range(len(control_array)):
        guess = 0.8
        escape = 0
        condition = True
        while condition:
            r_inf[i] = 1.0 - np.exp(-(control_array[i] * guess))
            if r_inf[i] == guess:
                condition = False
            guess = r_inf[i]
            escape += 1
            if escape > 10000:
                r_inf[i] = 0.0
                condition = False
    return r_inf

def compute_distribution_statistics(dist):
    dist_ = dist.copy()
    dist_array = np.array(dist_)
    dist = dist_array[~np.isnan(dist_array)]
    #dist = dist_[~np.isnan(dist)]
    
    if dist.size == False:
        dist = dist_.copy()

    # Compute average value of the distribution
    dist_avg = np.mean(dist)
    # Compute standard deviation
    dist_std = np.std(dist)
    # Compute 95% confidence interval
    z = 1.96
    nsims = len(dist)
    dist_l95 = dist_avg - (z * dist_std / np.sqrt(nsims))
    dist_u95 = dist_avg + (z * dist_std / np.sqrt(nsims))
    # Compute median
    dist_med = np.median(dist)
    # Compute 5th-percentile
    dist_p05 = np.percentile(dist, 5)
    # Compute 95th-percentiñe
    dist_p95 = np.percentile(dist, 95)
    # Prepare output dictionary & store results
    dist_dict = {}
    dist_dict['avg'] = dist_avg
    dist_dict['std'] = dist_std
    dist_dict['l95'] = dist_l95
    dist_dict['u95'] = dist_u95
    dist_dict['med'] = dist_med
    dist_dict['p05'] = dist_p05
    dist_dict['p95'] = dist_p95
    dist_dict['nsims'] = nsims
    
    return dist_dict

def filter_outbreaks(outbreak_distribution, pars, prevalence_cutoff=0.005):
    filtered_distribution = outbreak_distribution[outbreak_distribution>=prevalence_cutoff]
    return filtered_distribution

def choose_rk4_model(dynmod_id):
    if dynmod_id == 'rk4_hmf':
        return homogeneous_mean_field_equations
    elif dynmod_id == 'rk4_cghmf':
        return coarsed_grained_homogeneous_mean_field_equations

def initialize_model(pars):
    dynmod_id = pars['header']['model']
    if dynmod_id == 'rk4_cghmf':
        return initialize_cghmf(n_inf=pars['epidemic']['seeds'], 
                                n_rem=0, 
                                n_vac=0, 
                                n_act=pars['opinion']['acf'])
    elif dynmod_id == 'rk4_hmf':
        return initialize_hmf(n_as=pars['opinion']['acf'],
                              n_ai=pars['epidemic']['seeds'],
                              n_hi=pars['epidemic']['seeds'])

def initialize_cghmf(n_inf, n_rem=0, n_vac=0, n_act=0):
    y0 = np.zeros(5, dtype=np.float64)
    y0[0] = 1.0 - n_inf - n_rem - n_vac
    y0[1] = n_inf
    y0[2] = n_rem
    y0[3] = n_vac
    y0[4] = n_act

    return y0

def initialize_hmf(n_as, n_ai, n_hi, n_ar=0, n_hr=0, n_av=0):
    y0 = np.zeros(7, dtype=np.float64)
    y0[0] = n_as
    y0[1] = 1.0 - n_as - n_ai - n_hi - n_ar - n_hr - n_av
    y0[2] = n_as * n_ai
    y0[3] = (1.0 - n_as) * n_hi
    y0[4] = n_ar
    y0[5] = n_hr
    y0[6] = n_av

    return y0

def coarsed_grained_homogeneous_mean_field_equations(t, y, pars):
    # Unpack parameters
    N = 1.0
    k_avg = pars['network']['k_avg']
    alpha = pars['epidemic']['var']
    r0 = pars['epidemic']['r0']
    beta = pars['epidemic']['ifr']
    gamma = pars['epidemic']['rer']
    theta = pars['opinion']['thr']

    # Build terms
    infected_density = y[1] / N
    vaccinated_density = y[3] / N
    active_density = y[4] / N
    vaccination = alpha * active_density * y[0]
    active_infection = (1.0 - alpha) * beta * k_avg * active_density * y[0] * infected_density
    hesitant_infection = beta * k_avg * (1.0 - active_density) * y[0] * infected_density
    removal = gamma * y[1]
    if vaccinated_density >= theta:
        convinced = (1.0 - active_density) * vaccinated_density
    else:
        convinced = 0.0

    rates_of_change = np.zeros_like(y)
    rates_of_change[0] = -vaccination - active_infection - hesitant_infection
    rates_of_change[1] = active_infection + hesitant_infection - removal
    rates_of_change[2] = removal
    rates_of_change[3] = vaccination
    rates_of_change[4] = convinced
    
    return rates_of_change

def homogeneous_mean_field_equations(t, y, pars):
    # Unpack parameters
    N = 1.0
    k_avg = pars['network']['k_avg']
    alpha = pars['epidemic']['var']
    r0 = pars['epidemic']['r0']
    beta = pars['epidemic']['ifr']
    gamma = pars['epidemic']['rer']
    theta = pars['opinion']['thr']

    # Build terms
    infected_density = (y[2] + y[3]) / N
    vaccinated_density = y[6] / N
    if vaccinated_density >= theta:
        convinced_rate = 1.0
    else:
        convinced_rate = 0.0
    vaccination = alpha * y[0]
    active_infection = (1.0 - alpha) * beta * k_avg * y[0] * infected_density
    convinced = convinced_rate * y[1]
    hesitant_infection = (1.0 - convinced_rate) * beta * k_avg * y[1] * infected_density
    active_removal = gamma * y[2]
    hesitant_removal = gamma * y[3]

    # Build rates of change
    rates_of_change = np.zeros_like(y)
    rates_of_change[0] = - vaccination - active_infection + convinced
    rates_of_change[1] = - convinced - hesitant_infection
    rates_of_change[2] = active_infection - active_removal
    rates_of_change[3] = hesitant_infection - hesitant_removal
    rates_of_change[4] = active_removal
    rates_of_change[5] = hesitant_removal
    rates_of_change[6] = vaccination

    return rates_of_change

def integrate_runge_kutta_4(f, tspan, y0, nsteps, pars):
    """
    4th order Runge-Kutta method for solving a system of ODEs.

    Parameters:
    f : function
        A function that takes in two arguments: t, y and returns the derivative of y with respect to t.
    tspan : tuple
        A tuple containing the start and end time of the simulation, e.g. (0, 10).
    y0 : array-like
        An array of initial values for the variables in the system.
    N : int
        The number of time steps to take.

    Returns:
    t : array-like
        An array of time values corresponding to the solution.
    y : array-like
        An array of solution values.
    """
    t0, tf = tspan
    h = (tf - t0) / nsteps
    t_array = np.linspace(t0, tf, nsteps+1)
    y = np.zeros((nsteps+1, len(y0)))
    y[0] = y0
    
    for i in range(nsteps):
        k1 = h * f(t_array[i], y[i], pars)
        k2 = h * f(t_array[i] + h / 2.0, y[i] + k1 / 2.0, pars)
        k3 = h * f(t_array[i] + h / 2.0, y[i] + k2 / 2.0, pars)
        k4 = h * f(t_array[i] + h, y[i] + k3, pars)
        y[i+1] = y[i] + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    
    return t_array, y

def invoke_runge_kutta(equations, pars, save=False):
    tspan = (0, pars['algorithm']['tmax'])
    nsteps = 50000
    y0 = initialize_model(pars)
    t_array, y = integrate_runge_kutta_4(equations, tspan, y0, nsteps, pars)

    # Save results into pickle
    dynmod_id = pars['header']['model']
    if save:
        if dynmod_id == 'rk4_cghmf':
            results_dict = get_cg_hmf_time_series(t_array, y)
        elif dynmod_id == 'rk4_hmf':
            results_dict = get_hmf_time_series(t_array, y)
        # Build output file name
        lower_path = 'results'
        file_name = ut.build_file_name(pars) + '.pickle'
        full_path = os.path.join(path, lower_path, file_name)
        # Dump the results to a file
        pk.dump(results_dict, open(full_path, "wb"))

    return t_array, y

def get_cg_hmf_time_series(t, y):
    tseries_dict = {}
    tseries_dict['time'] = t
    tseries_dict['sus'] = y[:,0]
    tseries_dict['inf'] = y[:,1]
    tseries_dict['rem'] = y[:,2]
    tseries_dict['vac'] = y[:,3]
    tseries_dict['act'] = y[:,4]
    return tseries_dict

def get_hmf_time_series(t, y):
    tseries_dict = {}
    tseries_dict['time'] = t
    tseries_dict['AS'] = y[:,0]
    tseries_dict['HS'] = y[:,1]
    tseries_dict['AI'] = y[:,2]
    tseries_dict['HI'] = y[:,3]
    tseries_dict['AR'] = y[:,4]
    tseries_dict['HR'] = y[:,5]
    tseries_dict['AV'] = y[:,6]
    return tseries_dict

def get_cg_hmf_final_states(y):
    final_states_dict = {}
    final_states_dict['sus'] = y[-1][0]
    final_states_dict['inf'] = y[-1][1]
    final_states_dict['rem'] = y[-1][2]
    final_states_dict['vac'] = y[-1][3]
    final_states_dict['act'] = y[-1][4]
    return final_states_dict
  
def get_hmf_final_states(y):
    final_states_dict = {}
    final_states_dict['AS'] = y[-1][0]
    final_states_dict['HS'] = y[-1][1]
    final_states_dict['AI'] = y[-1][2]
    final_states_dict['HI'] = y[-1][3]
    final_states_dict['AR'] = y[-1][4]
    final_states_dict['HR'] = y[-1][5]
    final_states_dict['AV'] = y[-1][6]
    return final_states_dict

def get_hmf_global_states(final_states_dict):
    global_dict = {}
    global_dict['prevalence'] = final_states_dict['AR'] \
        + final_states_dict['HR']
    global_dict['vaccinated'] = final_states_dict['AV']
    global_dict['convinced'] = final_states_dict['AS'] \
        + final_states_dict['AI'] \
            + final_states_dict['AR'] \
                + final_states_dict['AV']
    return global_dict

def collect_rk4_final_states(
        model,
        control1, 
        control2,
        pars, 
):  
    dynmod_id = pars['header']['model']
    con1_id = pars['plot']['control1']
    con2_id = pars['plot']['control2']
    results_dict = {'final': {}, 'global': {}}

    for con1 in control1:
        print("Control {0}: {1}".format(con1_id, con1))
        for con2 in control2:
            print("Control {0}: {1}".format(con2_id, con2))
            # Load parametes
            pars['opinion'][con1_id] = con1
            pars['opinion'][con2_id] = con2
            # Integrate model
            t_array, y = invoke_runge_kutta(model, pars)

            if dynmod_id == 'rk4_cghmf':
                fs_dict = get_cg_hmf_final_states(y)
            elif dynmod_id == 'rk4_hmf':
                fs_dict = get_hmf_final_states(y)
                gs_dict = get_hmf_global_states(fs_dict)
                results_dict['global'][(con1, con2)] = gs_dict
            results_dict['final'][(con1, con2)] = fs_dict
    
    # Save control parameter array
    results_dict[con1_id] = control1
    results_dict[con2_id] = control2

    # Build output file name
    lower_path = 'results'
    file_name = ut.build_file_name(pars, 
                                   exclude_keys=[con1_id, con2_id], 
                                   collection=True) + '.pickle'
    full_path = os.path.join(path, lower_path, file_name)
    # Dump the results to a file
    pk.dump(results_dict, open(full_path, "wb"))

def main():

    # Input parameters for the model
    project_id = 'thr'
    dynmod_id = 'rk4_hmf'
    attribute_id = 'tseries'
    exp_id = '1'
    model = 'Complete'
    equations = homogeneous_mean_field_equations
    N = 10000
    p = 0.001
    k_avg = N - 1
    theta = 0.2
    n_A = 0.25
    alpha = 0.005
    r0 = 1.25
    gamma = 0.2
    beta = r0 * gamma / k_avg 
    seeds = 1.0 / N
    t_max = 500
    con1_id = 'thr'
    con2_id = 'acf'
    observable = 'prevalence'

    # Build full parameter dictionary
    pars = {}
    pars['header'] = ut.build_header_parameters_dictionary(project_id, dynmod_id, attribute_id, exp_id)
    pars['network'] = ut.build_network_parameters_dictionary(model, n=N, p=p)
    pars['network']['k_avg'] = k_avg
    pars['opinion'] = ut.build_opinion_parameters_dictionary(n_A, theta)
    pars['epidemic'] = ut.build_epidemic_parameters_dictionary(seeds, alpha, beta, gamma, r0)
    pars['epidemic']['seeds'] = seeds
    pars['epidemic']['ifr'] = beta
    pars['algorithm'] = ut.build_algorithm_parameters_dictionary(dynmod_id, t_max=t_max)
    pars['plot'] = ut.build_plot_parameters_dictionary(con1_id, con2_id, observable)

    equations = choose_rk4_model(dynmod_id)
    #t_array, y = invoke_runge_kutta(equations, pars, save=True)
    #pt.choose_rk4_plot(dynmod_id, pars)
    control1 = np.linspace(0.0, 1.0, 21)
    control2 = np.linspace(0.0, 1.0, 21)
    #collect_rk4_final_states(equations, control1, control2, pars)
    pt.plot_rk4_heatmap_active_threshold_space(pars)


if __name__ == '__main__':

    main()