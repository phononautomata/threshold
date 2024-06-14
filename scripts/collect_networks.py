import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))

try:
    import utils as ut
except ModuleNotFoundError:
    print("Module 'utils' not found. Ensure 'utils.py' is in the 'src' directory and you're running this script from the correct directory.")
    sys.exit(1)

path_base = '/data/alfonso/threshold/data/curated/networks'

def collect_filenames(id_region, size, path_storage=path_base):
    path_search_source = os.path.join(path_storage, id_region)
    
    if not os.path.isdir(path_search_source):
        print(f"Directory {path_search_source} does not exist.")
        return []

    filenames = [filename for filename in os.listdir(path_search_source)
                 if filename.startswith(f'ml{id_region}_n{size}') and filename.endswith('.pickle')]
    
    return filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect network UUIDs for simulations.')
    parser.add_argument('--id_region', type=str, required=True, help='Regional ID to load the corresponding multilayer network')
    parser.add_argument('--size', type=str, required=True, help='Number of agents involved in the network')
    parser.add_argument('--path_storage', type=str, default=path_base, help='Base path where the network is stored')
    args = parser.parse_args()

    id_region = args.id_region
    size = args.size
    path_storage = args.path_storage

    filenames = collect_filenames(id_region, size, path_storage)
    for filename in filenames:
        print(filename)