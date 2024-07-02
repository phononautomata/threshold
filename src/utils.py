import json
import numpy as np
import os
import pandas as pd
import pickle as pk
import requests
import time
import urllib.request

from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression

cwd_path = os.getcwd()

default_country = 'United_States'
default_year = 2020

def age_group_contact_number_distribution(contact):
    return np.sum(contact, axis=1) #* pop_a / np.sum(pop_a)

def check_contact_reciprocity(contact_matrix, population_array):
    contact_matrix = np.asarray(contact_matrix)
    population_array = np.asarray(population_array)

    left_side = contact_matrix * population_array[:, np.newaxis]
    right_side = contact_matrix.T * population_array

    reciprocity = np.isclose(left_side, right_side, atol=1e-04)

    if not np.all(reciprocity):
        print("Reciprocity does not hold for the matrix.")
        return False
    else:
        print("Reciprocity holds for the matrix.")
        return True

def coarse_grain_population(pop_a, ngroups):
    if ngroups <= 0:
        raise ValueError("Number of groups must be a positive integer.")
    if ngroups > len(pop_a):
        raise ValueError("Number of groups cannot exceed the length of the original population vector.")

    n = len(pop_a)
    group_size = n // ngroups
    remainder = n % ngroups

    new_pop_a = []
    sum_group = 0
    counter = 0

    for age, count in enumerate(pop_a):
        sum_group += count
        counter += 1

        # Check if we have reached the end of a group
        if counter == group_size + (1 if remainder > 0 else 0):
            new_pop_a.append(sum_group)
            sum_group = 0
            counter = 0
            remainder -= 1

    return new_pop_a

def coarse_grain_contact(contact, ngroups):
    if ngroups <= 0:
        raise ValueError("Number of groups must be a positive integer.")
    if ngroups > contact.shape[0] or ngroups > contact.shape[1]:
        raise ValueError("Number of groups cannot exceed the dimensions of the original matrix.")

    rows, cols = contact.shape
    row_group_size = rows // ngroups
    col_group_size = cols // ngroups
    row_remainder = rows % ngroups
    col_remainder = cols % ngroups

    # Initialize the new matrix
    new_contact = np.zeros((ngroups, ngroups))

    # Aggregate matrix elements
    for i in range(ngroups):
        for j in range(ngroups):
            row_start = i * row_group_size + min(i, row_remainder)
            col_start = j * col_group_size + min(j, col_remainder)
            row_end = (i + 1) * row_group_size + min(i + 1, row_remainder)
            col_end = (j + 1) * col_group_size + min(j + 1, col_remainder)

            new_contact[i, j] = contact[row_start:row_end, col_start:col_end].sum()

    return new_contact

def collect_age_structure_data(path=cwd_path, lower_path='data'):
    state_list = get_state_list()

    population_dict = {}
    contact_dict = {}
    degree_distribution_dict = {}
    degree_distribution_age_dict = {}

    for state in state_list:
        # Reference data
        pop_a = import_age_distribution(state=state)
        contact = import_contact_matrix(state=state)

        # Updated data
        new_pop_a = import_age_distribution(state=state, reference=False, year=2019)
        new_contact = update_contact_matrix(contact, old_pop_a=pop_a, new_pop_a=new_pop_a)

        # Age-group degree distribution
        degree_pdf = average_degree_distribution(new_contact, new_pop_a)

        # Intralayer degree distribution

        #population_dict[state] = new_pop_a
        #contact_dict[state] = new_contact
        #degree_distribution_dict[state] = degree_distribution

        if np.sum(new_pop_a) != 1.0:
            new_pop_a /= np.sum(new_pop_a)
        if np.sum(degree_pdf) != 1.0:
            degree_pdf / np.sum(degree_pdf)

        population_dict[state] = new_pop_a.tolist() if isinstance(new_pop_a, np.ndarray) else new_pop_a
        contact_dict[state] = new_contact.tolist() if isinstance(new_contact, np.ndarray) else new_contact
        degree_distribution_dict[state] = degree_pdf.tolist() if isinstance(degree_pdf, np.ndarray) else degree_pdf

    file_name = 'state_population_age.json'
    full_name = os.path.join(path, lower_path, file_name)
    with open(full_name, 'w') as file:
        json.dump(population_dict, file)

    file_name = 'state_contact_matrix.json'
    full_name = os.path.join(path, lower_path, file_name)
    with open(full_name, 'w') as file:
        json.dump(contact_dict, file)

    file_name = 'state_degree.json'
    full_name = os.path.join(path, lower_path, file_name)
    with open(full_name, 'w') as file:
        json.dump(degree_distribution_dict, file)

def collect_filenames(path_search, header=None, string_segments=None, extension='.json'):
    file_list = os.listdir(path_search)

    result = []
    for file_name in file_list:
        if file_name.endswith(extension) and file_name.startswith(header) and (string_segments is None or all(segment in file_name for segment in string_segments)):
            result.append(file_name)

    return result

def compute_distribution_statistics(dist):
    dist_ = dist.copy()
    dist_array = np.array(dist_)
    dist = dist_array[~np.isnan(dist_array)]
    #dist = dist_[~np.isnan(dist)]

    if dist.size == False:
        dist = dist_.copy()

    dist_avg = np.mean(dist)
    dist_std = np.std(dist)
    z = 1.96
    nsims = len(dist)
    dist_l95 = dist_avg - (z * dist_std / np.sqrt(nsims))
    dist_u95 = dist_avg + (z * dist_std / np.sqrt(nsims))
    dist_med = np.median(dist)
    dist_p05 = np.percentile(dist, 5)
    dist_p95 = np.percentile(dist, 95)

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

def convert_string_state_to_rust_enum_format(state):
    words = state.replace('_', ' ').split()
    formatted_state = ''.join(word.capitalize() for word in words)
    return formatted_state

def count_fraction_underage(array_population):
    return np.sum(array_population[0:18])

def average_degree_distribution(contact, pop_a, norm_flag=False):
    degree_a = np.sum(contact, axis=1)
    pdf = np.zeros(len(degree_a))

    for a in range(len(degree_a)):
        k = round(degree_a[a])
        pdf[k] += pop_a[a]

    if norm_flag:
        pdf /= np.sum(pdf)

    return pdf

def download_us_census_data(
        path=cwd_path, 
        lower_path='data', 
        year=default_year):
    state_list = get_state_list()

    if 2010 <= year < 2020:
        year_substring = '2010s'
    elif 2030 > year >= 2020:
        year_substring = '2020s'

    # URL to webscrape from
    url = 'https://www.census.gov/data/tables/time-series/demo/popest/' + year_substring + '-state-detail.html'

    # Connect to the URL
    response = requests.get(url)
    # Parse HTML and save to BeautifulSoup object
    soup = BeautifulSoup(response.text, "html.parser")

    # Download full dataset
    to_be_downloaded = soup.findAll('a')[288:339]
    for one_a_tag, state in zip(to_be_downloaded, state_list):

        link = one_a_tag['href']
        download_url = 'https:' + link

        filename = str(year) + '_United_States_subnational_' + state + '_age_distribution_85' + '.xlsx'
        stored_data = os.path.join(path, lower_path, filename)

        urllib.request.urlretrieve(download_url, stored_data)

        print("Data from {0} {1} successfully downloaded".format(state, year))

        time.sleep(1)

def extract_code_from_state(state):
    state_code_dict = {
        'Alaska': 'AK', 
        'Alabama': 'AL', 
        'Arkansas': 'AR', 
        'Arizona': 'AZ', 
        'California': 'CA', 
        'Colorado': 'CO',
        'Connecticut': 'CT', 
        'District_of_Columbia': 'DC',
        'DistrictOfColumbia': 'DC',
        'Delaware': 'DE', 
        'Florida': 'FL',
        'Georgia': 'GA',
        'Hawaii': 'HI',
        'Iowa': 'IA',
        'Idaho': 'ID',
        'Illinois': 'IL',
        'Indiana': 'IN',
        'Kansas': 'KS',
        'Kentucky': 'KY',
        'Louisiana': 'LA',
        'Massachusetts': 'MA', 
        'Maryland': 'MD',
        'Maine': 'ME',
        'Michigan': 'MI',
        'Minnesota': 'MN', 
        'Missouri': 'MO', 
        'Mississippi': 'MS', 
        'Montana': 'MT',
        'North_Carolina': 'NC', 
        'NorthCarolina': 'NC',
        'North_Dakota': 'ND', 
        'NorthDakota': 'ND',
        'Nebraska': 'NE', 
        'New_Hampshire': 'NH',
        'NewHampshire': 'NH', 
        'New_Jersey': 'NJ',
        'NewJersey': 'NJ',
        'New_Mexico': 'NM',
        'NewMexico': 'NM', 
        'Nevada': 'NV',
        'New_York': 'NY', 
        'NewYork': 'NY',
        'Ohio': 'OH', 
        'Oklahoma': 'OK', 
        'Oregon': 'OR', 
        'Pennsylvania': 'PA', 
        'Rhode_Island': 'RI', 
        'RhodeIsland': 'RI',
        'South_Carolina': 'SC',
        'SouthCarolina': 'SC', 
        'South_Dakota': 'SD',
        'SouthDakota': 'SD',
        'Tennessee': 'TN', 
        'Texas': 'TX',
        'Utah': 'UT', 
        'Virginia': 'VA', 
        'Vermont': 'VT', 
        'Washington': 'WA', 
        'Wisconsin': 'WI', 
        'West_Virginia': 'WV',
        'WestVirginia': 'WV',
        'Wyoming': 'WY',
        'National': 'US',
        }

    return state_code_dict[state]

def extract_failed_outbreaks(
        prevalence_distribution, 
        prevalence_cutoff=0.0, 
        n=None,
        ):
    if np.any(prevalence_distribution > 1.0):
        prevalence_distribution /= n
    failed_outbreaks = np.where(prevalence_distribution < prevalence_cutoff)[0]
    return failed_outbreaks

def extract_state_from_code(code):
    code_state_dict = {
        'AK': 'Alaska', 
        'AL': 'Alabama', 
        'AR': 'Arkansas', 
        'AZ': 'Arizona',
        'CA': 'California', 
        'CO': 'Colorado',
        'CT': 'Connecticut', 
        'DC': 'District_of_Columbia', 
        'DE': 'Delaware', 
        'FL': 'Florida', 
        'GA': 'Georgia', 
        'HI': 'Hawaii', 
        'IA': 'Iowa', 
        'ID': 'Idaho',
        'IL': 'Illinois', 
        'IN': 'Indiana', 
        'KS': 'Kansas', 
        'KY': 'Kentucky', 
        'LA': 'Louisiana', 
        'MA': 'Massachusetts', 
        'MD': 'Maryland', 
        'ME': 'Maine', 
        'MI': 'Michigan', 
        'MN': 'Minnesota', 
        'MO': 'Missouri', 
        'MS': 'Mississippi', 
        'MT': 'Montana', 
        'NC': 'North_Carolina', 
        'ND': 'North_Dakota', 
        'NE': 'Nebraska', 
        'NH': 'New_Hampshire', 
        'NJ': 'New_Jersey', 
        'NM': 'New_Mexico', 
        'NV': 'Nevada',
        'NY': 'New_York', 
        'OH': 'Ohio', 
        'OK': 'Oklahoma', 
        'OR': 'Oregon', 
        'PA': 'Pennsylvania', 
        'RI': 'Rhode_Island', 
        'SC': 'South_Carolina', 
        'SD': 'South_Dakota', 
        'TN': 'Tennessee', 
        'TX': 'Texas',
        'UT': 'Utah',  
        'VA': 'Virginia', 
        'VT': 'Vermont', 
        'WA': 'Washington', 
        'WI': 'Wisconsin', 
        'WV': 'West_Virginia', 
        'WY': 'Wyoming',
        'US': 'National',
        }
    
    return code_state_dict[code]

def filter_global_output(output_dict, n, prevalence_cutoff):
    norm_dist = {key: (np.array(val) / n if 'time' not in key else np.array(val)) 
                 for key, val in output_dict.items()}

    failed_outbreaks = np.where(norm_dist['prevalence'] < prevalence_cutoff)[0]

    filtered_dist = {}
    for key, val in norm_dist.items():
        filt_val = np.delete(val, failed_outbreaks)
        if len(filt_val) == 0:
            filt_val = val
        filtered_dist[key] = filt_val

    return filtered_dist

def filter_stratified_attribute_output(
        output_dict, 
        n, 
        prevalence_cutoff, 
        norm_age_flag=False,
        ):
    norm_dist = {key: (np.array(val) / n if ('time' not in key and 'when' not in key) else np.array(val)) 
                 for key, val in output_dict.items()}

    failed_outbreaks = np.where(np.sum(norm_dist['prevalence']) < prevalence_cutoff)[0]

    if norm_age_flag:
        norm_dist = {key: (np.array(val) / output_dict['age'] if ('time' not in key and 'when' not in key) else np.array(val)) 
                     for key, val in output_dict.items()}

    filtered_dist = {}
    for key, val in norm_dist.items():
        filt_val = np.delete(val, failed_outbreaks)
        if len(filt_val) == 0:
            filt_val = val
        filtered_dist[key] = filt_val

    return filtered_dist

def filter_observable_array(observable_distribution, failed_outbreaks):
    observable_distribution = np.array(observable_distribution)
    filtered_observable = np.delete(observable_distribution, failed_outbreaks, axis=0)
    return filtered_observable

def filter_observable_list(observable_distribution_list, failed_outbreaks):
    filt_obs_dist_list = [sim for i, sim in enumerate(observable_distribution_list) if i not in failed_outbreaks]
    return filt_obs_dist_list

def find_nth_occurrence(string, substring, n):
    parts = string.split(substring, n)
    if len(parts) <= n:
        return None
    return substring.join(parts[:-1]), parts[-1]

def get_state_list():
    return ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 
              'Colorado', 'Connecticut', 'Delaware', 'District_of_Columbia', 
              'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana',
              'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland',
              'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', 
              'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New_Hampshire',  
              'New_Jersey', 'New_Mexico', 'New_York', 'North_Carolina', 
              'North_Dakota', 'Ohio', 'Oklahoma', 'Oregon', 
              'Pennsylvania', 'Rhode_Island', 'South_Carolina', 'South_Dakota', 
              'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 
              'Washington', 'West_Virginia', 'Wisconsin', 'Wyoming', 'National']

def import_age_distribution(
        state, 
        path=cwd_path, 
        lower_path='data/raw', 
        country=default_country, 
        reference=True, 
        year=default_year,
        norm_flag=False,
        ):
    if reference == True:
        if state == 'National':
            file_name = country + '_country_level_' + 'age_distribution_85' 
        else:
            file_name = country + '_subnational_' + state + '_age_distribution_85'
        extension = '.csv'
        full_name = os.path.join(path, lower_path, file_name + extension)

        age_df = pd.read_csv(full_name, header=None)

        pop_a = age_df.values.T[1]
    else:
        if state == 'National':
            file_name = str(year) + '_' + country + '_country_level_' + 'age_distribution_85'
            extension = '.xlsx'
            full_name = os.path.join(path, lower_path, file_name + extension)

            full_age_df = pd.read_excel(full_name)

            age_df = full_age_df['Unnamed: 12'][4:89].values
            merge_older = np.sum(full_age_df['Unnamed: 12'][88:105].values)
            age_df[-1] = merge_older

            pop_a = np.zeros(len(age_df), dtype=float)
            for a in range(len(age_df)):
                pop_a[a] = age_df[a]

        else:
            file_name = str(year) + '_' + country + '_subnational_' + state + '_age_distribution_85'
            extension = '.xlsx'
            full_name = os.path.join(path, lower_path, file_name + extension)

            full_age_df = pd.read_excel(full_name)

            age_df = full_age_df['Unnamed: 34'][5:90].values
            merge_older = full_age_df['Unnamed: 34'][89] + full_age_df['Unnamed: 34'][90]
            age_df[-1] = merge_older

            pop_a = np.zeros(len(age_df), dtype=float)
            for a in range(len(age_df)):
                pop_a[a] = age_df[a]

    if norm_flag:
        pop_a /= np.sum(pop_a)

    return pop_a

def import_age_vaccination_attitudes(
        path=cwd_path, 
        lower_path='data/raw',
        ):
    file_name = 'vaccination_attitude_fraction_by_age'
    extension = '.csv'
    full_name = os.path.join(path, lower_path, file_name + extension)

    attitudes_df = pd.read_csv(full_name, sep=",", skiprows=1, names=range(4))

    return attitudes_df

def import_contact_matrix(
        state, 
        path=cwd_path, 
        lower_path='data/raw', 
        country=default_country
        ):
    if state == 'National':
        file_name = country + '_country_level_' + 'M_overall_contact_matrix_85'
    else:
        file_name = country + '_subnational_' + state + '_M_overall_contact_matrix_85'
    extension = '.csv'
    full_name = os.path.join(path, lower_path, file_name + extension)

    contact_df = pd.read_csv(full_name, header=None)

    return contact_df.values

def interlayer_probability(contact_matrix):
    contact_matrix = np.asarray(contact_matrix)

    interlayer_probability = contact_matrix / np.sum(contact_matrix, axis=1)[:, np.newaxis]

    if np.all(np.isclose(np.sum(interlayer_probability, axis=1), 1, atol=1e-06)):
        print("All rows sum to 1 within the tolerance.")
    else:
        print("Not all rows sum to 1 within the tolerance.")
    
    return interlayer_probability

def layer_average_degree(contact_matrix):
    contact_matrix = np.asarray(contact_matrix)

    return np.sum(contact_matrix, axis=1)

def load_output(fullname, main_keys, observable_keys):
    if fullname.endswith('.pickle'):
        with open(fullname, 'rb') as input_data:
            results_dict = pk.load(input_data)
    elif fullname.endswith('.json'):
        with open(fullname, 'w') as file:
            results_dict = json.dump(input_data, file)

    if 'prevalence' not in observable_keys:
        observable_keys.append('prevalence')

    output_dict = {}

    for main_key in main_keys:
        if main_key:
            for obs_key in observable_keys:
                if obs_key in results_dict[main_key]:
                    output_dict[obs_key] = results_dict[main_key][obs_key]
                else:
                    print(f"Warning: Key '{obs_key}' not found in {main_key} data.")
        else:
            print(f"Warning: Main key '{main_key}' not found in results dict.")

    return output_dict

def load_vaccination_data(
        path_cwd, 
        path_relative_source='data/curated', 
        file_name='vaccination_attitude_data.json'
        ):
    full_path = os.path.join(path_cwd, path_relative_source, file_name)
    with open(full_path, 'r') as f:
        data = json.load(f)

    dict_state_attitudes = {}
    for state, values in data.items():
        dict_state_attitudes[state] = {
            "already": values[0],
            "soon": values[1],
            "someone": values[2],
            "majority": values[3],
            "never": values[4]
        }

    return dict_state_attitudes

def plot_regression(ax, x, y, v, i):
    x_array = np.array(x).reshape(-1, 1)
    y_array = np.array(y)

    reg = LinearRegression().fit(x_array, y_array)
    
    y_pred = reg.predict(x_array)
    
    corr_coef = np.corrcoef(x, y)[0, 1]
    
    scatter = ax[v, i].scatter(x, y, color='slateblue')
    
    ax[v, i].plot(x, y_pred, color='crimson', linewidth=2)

    if i == 0 or i == 1:
        ax[v, i].text(0.05, 0.15, r'$\rho$={0}'.format(np.round(corr_coef, 2)), transform=ax[v, i].transAxes, 
                      fontsize=25, verticalalignment='top')
    else:
        ax[v, i].text(0.05, 0.95, r'$\rho$={0}'.format(np.round(corr_coef, 2)), transform=ax[v, i].transAxes, 
                  fontsize=25, verticalalignment='top')
    
    return scatter, reg

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

def stat_global_results(model_region, value_af, value_th, value_vr, path_cwd):
    pass

def stat_global_output(output_dict):
    stat_dict = {}
    for key, val in output_dict.items():
        stat_dict[key] = compute_distribution_statistics(val)

    return stat_dict

def stat_observable(observable_dist):
    return compute_distribution_statistics(observable_dist)

def stat_stratified_attribute_observable_array(
        observable_distribution, 
        attribute='age',
        ):
    stat_dict = {}
    for age, value_a in enumerate(observable_distribution.T):
            stat_dict[age] = compute_distribution_statistics(value_a)
    return stat_dict

def stat_stratified_attribute_observable_list(
        observable_distribution_list, 
        attribute='age',
        ):
    z = 1.96
    nan_threshold = 10000000

    nbins = len(observable_distribution_list[0])
    
    dist_per_att = [[] for _ in range(nbins)]
    
    for sim_idx in range(len(observable_distribution_list)):
        for att_idx in range(nbins):
            dist_values = observable_distribution_list[sim_idx][att_idx]
            cleaned_dist_values = [num if num < nan_threshold else np.nan for num in dist_values]
            dist_per_att[att_idx].extend(cleaned_dist_values)
    dist_avg_per_att = np.array([np.nanmean(sublist) for sublist in dist_per_att])
    dist_std_per_att = np.array([np.nanstd(sublist) for sublist in dist_per_att])

    dist_moe = z * (dist_std_per_att / np.sqrt([len(sublist) for sublist in dist_per_att]))
    dist_u95_per_att = dist_avg_per_att + dist_moe
    dist_l95_per_att = dist_avg_per_att - dist_moe

    flattened_list = [num if num < nan_threshold else np.nan
                  for sublist1 in observable_distribution_list
                  for sublist2 in sublist1
                  for num in sublist2]
    dist_avg_global = np.nanmean(flattened_list)
    dist_std_global = np.nanstd(flattened_list)
    dist_moe = z * dist_std_global / np.sqrt(len(flattened_list))
    dist_u95_global = dist_avg_global + dist_moe
    dist_l95_global = dist_avg_global - dist_moe

    stat_dict = {}
    stat_dict['dist_avg_per_' + attribute] = dist_avg_per_att
    stat_dict['dist_l95_per_' + attribute] = dist_l95_per_att
    stat_dict['dist_u95_per_' + attribute] = dist_u95_per_att
    stat_dict['dist_avg_global'] = dist_avg_global
    stat_dict['dist_l95_global'] = dist_l95_global
    stat_dict['dist_u95_global'] = dist_u95_global
    
    return stat_dict

def stat_stratified_attribute_output(output_dict):
    stat_dict = {}
    for key, val in output_dict.items():
        stat_dict[key] = {}
        for age in range(len(output_dict[key])):
            stat_dict[key][age] = compute_distribution_statistics(val[age])

    return stat_dict

def trim_file_extension(file_string):
    extensions = ['.pickle', '.pdf', '.png', '.txt', '.csv']

    for ext in extensions:
        if file_string.endswith(ext):
            file_string = file_string[:-(len(ext))]
            break

    return file_string

def update_contact_matrix(contact, old_pop_a, new_pop_a):
    N_old = np.sum(old_pop_a)
    N_new = np.sum(new_pop_a)
    A = len(old_pop_a)
    new_contact = np.zeros((A, A), dtype=float)

    for i in range(A):
        for j in range(A):
            old_fraction = N_old / old_pop_a[j]
            new_fraction = new_pop_a[j] / N_new
            factor = old_fraction * new_fraction
            new_contact[i][j] = contact[i][j] * factor

    return new_contact

def write_annotations_vaccination_curves(
        state_abb, 
        ax, 
        var_values, 
        vacc_avg, 
        prev_avg,
        ):
    if state_abb == 'AK':
        ax[1].annotate(
            state_abb, 
            (var_values[0], vacc_avg[0]), 
            fontsize=10, 
            xytext=(var_values[0], vacc_avg[0] + 0.4), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[0], prev_avg[0]), 
            fontsize=10, 
            xytext=(var_values[0], prev_avg[0] + 0.01), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'AR':
        ax[1].annotate(
            state_abb, 
            (var_values[15], vacc_avg[15]), 
            fontsize=10, 
            xytext=(var_values[15], vacc_avg[15] - 0.1), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[15], prev_avg[15]), 
            fontsize=10, 
            xytext=(var_values[15], prev_avg[15] + 0.1), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'CA':
        ax[1].annotate(
            state_abb, 
            (var_values[25], vacc_avg[25]), 
            fontsize=10, 
            xytext=(var_values[25], vacc_avg[25] + 0.25), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[7], prev_avg[7]), 
            fontsize=10, 
            xytext=(var_values[7], prev_avg[7] + 0.1), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        
    elif state_abb == 'DE':
        ax[1].annotate(
            state_abb, 
            (var_values[-5], vacc_avg[-5]), 
            fontsize=10, 
            xytext=(var_values[-5], vacc_avg[-5] + 0.1), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[6], prev_avg[6]), 
            fontsize=10, 
            xytext=(var_values[6], prev_avg[6] - 0.05), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'DC':
        ax[1].annotate(
            state_abb, 
            (var_values[15], vacc_avg[15]), 
            fontsize=10, 
            xytext=(var_values[15], vacc_avg[15] + 0.1), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[0], prev_avg[0]), 
            fontsize=10, 
            xytext=(var_values[3], prev_avg[0] - 0.04), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'HI':
        ax[1].annotate(
            state_abb, 
            (var_values[2], vacc_avg[2]), 
            fontsize=10, 
            xytext=(var_values[2], vacc_avg[2] + 0.25), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[2], prev_avg[2]), 
            fontsize=10, 
            xytext=(var_values[4], prev_avg[2] - 0.02), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'IL':
        ax[1].annotate(
            state_abb, 
            (var_values[20], vacc_avg[20]), 
            fontsize=10, 
            xytext=(var_values[20], vacc_avg[20] + 0.15), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[4], prev_avg[4]), 
            fontsize=10, 
            xytext=(var_values[4], prev_avg[4] + 0.15), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'KY':
        ax[1].annotate(
            state_abb, 
            (var_values[-5], vacc_avg[-5]), 
            fontsize=10, 
            xytext=(var_values[-5], vacc_avg[-5] - 0.2), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[5], prev_avg[5]), 
            fontsize=10, 
            xytext=(var_values[5], prev_avg[5] + 0.05), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'MA':
        ax[1].annotate(
            state_abb, 
            (var_values[0], vacc_avg[0]), 
            fontsize=10, 
            xytext=(var_values[0], vacc_avg[0] - 0.15), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[1].annotate(
            state_abb, 
            (var_values[-1], vacc_avg[-1]), 
            fontsize=10, 
            xytext=(var_values[-1], vacc_avg[-1] + 0.03), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[0], prev_avg[0]), 
            fontsize=10, 
            xytext=(var_values[0], prev_avg[0] + 0.075), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[15], prev_avg[15]), 
            fontsize=10, 
            xytext=(var_values[10], prev_avg[15] - 0.01), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'ND':
        ax[1].annotate(
            state_abb, 
            (var_values[10], vacc_avg[10]), 
            fontsize=10, 
            xytext=(var_values[10], vacc_avg[10] - 0.15), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[10], prev_avg[10]), 
            fontsize=10, 
            xytext=(var_values[10], prev_avg[10] + 0.1), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'OK':
        ax[1].annotate(
            state_abb, 
            (var_values[40], vacc_avg[40]), 
            fontsize=10, 
            xytext=(var_values[40], vacc_avg[40] - 0.15), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[0], prev_avg[0]), 
            fontsize=10, 
            xytext=(var_values[0] - 0.000025, prev_avg[0] - 0.12), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )

    elif state_abb == 'WY':
        ax[1].annotate(
            state_abb, 
            (var_values[20], vacc_avg[20]), 
            fontsize=10, 
            xytext=(var_values[20], vacc_avg[20] - 0.15), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
        ax[2].annotate(
            state_abb, 
            (var_values[12], prev_avg[12]), 
            fontsize=10, 
            xytext=(var_values[12], prev_avg[12] + 0.15), 
            arrowprops=dict(arrowstyle='-', lw=1),
            )
