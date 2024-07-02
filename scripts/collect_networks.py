import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(project_root, 'src'))

path_base = '/data/alfonso/threshold/data/curated/networks'

def collect_filenames(list_id_region, size, path_storage=path_base):
    all_filenames = []
    
    for id_region in list_id_region:
        path_search_source = os.path.join(path_storage, id_region)
        
        if not os.path.isdir(path_search_source):
            print(f"Directory {path_search_source} does not exist.")
            continue
        
        filenames = [os.path.splitext(filename)[0] for filename in os.listdir(path_search_source)
                     if filename.startswith(f'ml{id_region}_n{size}') and filename.endswith('.pickle')]
        all_filenames.extend(filenames)
    
    return all_filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Collect network UUIDs for simulations.')
    parser.add_argument('--id_region', type=str, nargs='+', required=True, help='Regional ID to load the corresponding multilayer network')
    parser.add_argument('--size', type=str, required=True, help='Number of agents involved in the network')
    parser.add_argument('--path_storage', type=str, default=path_base, help='Base path where the network is stored')
    args = parser.parse_args()

    list_id_region = args.id_region
    size = args.size
    path_storage = args.path_storage

    filenames = collect_filenames(list_id_region, size, path_storage)
    for filename in filenames:
        print(filename)