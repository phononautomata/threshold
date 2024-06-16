import os
import sys
import numpy as np
import json

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))

try:
    import utils as ut
except ModuleNotFoundError:
    print("Module 'utils' not found. Ensure 'utils.py' is in the 'src' directory and you're running this script from the correct directory.")
    sys.exit(1)

path_base = project_root

def collect_age_structure_data():
    state_list = ut.get_state_list()

    population_dict = {}
    contact_dict = {}
    degree_distribution_dict = {}

    for state in state_list:

        if state == 'National':
            print('Hold it!')
        pop_a = ut.import_age_distribution(state=state)
        contact = ut.import_contact_matrix(state=state)

        new_pop_a = ut.import_age_distribution(state=state, reference=False, year=2019)
        new_contact = ut.update_contact_matrix(contact, old_pop_a=pop_a, new_pop_a=new_pop_a)
    
        degree_pdf = ut.average_degree_distribution(new_contact, new_pop_a)

        if np.sum(new_pop_a) != 1.0:
            new_pop_a /= np.sum(new_pop_a)
        if np.sum(degree_pdf) != 1.0:
            degree_pdf / np.sum(degree_pdf)

        population_dict[state] = new_pop_a.tolist() if isinstance(new_pop_a, np.ndarray) else new_pop_a
        contact_dict[state] = new_contact.tolist() if isinstance(new_contact, np.ndarray) else new_contact
        degree_distribution_dict[state] = degree_pdf.tolist() if isinstance(degree_pdf, np.ndarray) else degree_pdf

    file_name = 'state_population_age.json'
    path_full_target = os.path.join(path_base, 'data', 'curated', file_name)
    with open(path_full_target, 'w') as file:
        json.dump(population_dict, file)

    file_name = 'state_contact_matrix.json'
    path_full_target = os.path.join(path_base, 'data', 'curated', file_name)
    with open(path_full_target, 'w') as file:
        json.dump(contact_dict, file)

    file_name = 'state_degree.json'
    path_full_target = os.path.join(path_base, 'data', 'curated', file_name)
    with open(path_full_target, 'w') as file:
        json.dump(degree_distribution_dict, file)

if __name__ == "__main__":
    collect_age_structure_data()

    print("Age and contact structure ready to rust!")

    