import os
import json
import sys
import torch
import time
import multiprocessing as multiproc
from pymatgen.core import Structure
from chgnet.graph import CrystalGraphConverter
# Adjust the line below to point to where you are storing your source code
sys.path.append(os.path.expanduser("~/bin/CHGNet-finetuning"))
from data_preprocessing import Filter, DataExtracter, read_json, write_json

# === CONFIG ===
JSON_PATH = "Fe-O.json"
GRAPH_DIR = os.path.join(os.getcwd(),"Fe-O-graphs")
REMAKE = False
VERBOSE = True
num_graphs = 0
ATOM_GRAPH_CUTOFF = 7
BOND_GRAPH_CUTOFF = 3

# === PARSE STRUCTURES ===
def get_structure_dicts(json_path):
    structure_dicts = []
    all_data = read_json(json_path)
    for entry in all_data:
        struct = Structure.from_dict(entry["structure"])
        # n_atoms = entry['nsites']
        # label_data = {
        #     "energy_per_atom": entry.get("energy") / n_atoms,
        #     "force": entry.get("forces"),
        #     "stress": entry.get("stress"),
        #     "magmom": [
        #         s.get("properties", {}).get("magmom", 0.0)
        #         for s in entry["structure"].get("sites", [])
        #     ],
        # }
        
        ex = DataExtracter([entry])

        label_data = {
            "energy_per_atom": ex.get_energies_per_atom()[0],
            "force": ex.get_forces(),
            "stress": ex.get_stresses(),
            "magmom": ex.get_magmoms(),
        }

        mp_id = entry.get("provenance", {}).get("original_mp_id", "unknown")
        graph_id = entry.get("matpes_id", f"{mp_id}-0")
        structure_dicts.append((struct, label_data, mp_id, graph_id))

    return structure_dicts

# === GRAPH CREATION ===
def create_graph(structure_data, verbose=VERBOSE):
    converter = CrystalGraphConverter(atom_graph_cutoff=ATOM_GRAPH_CUTOFF, bond_graph_cutoff=BOND_GRAPH_CUTOFF)
    struct, label_data, mp_id, graph_id = structure_data
    global num_graphs
    num_graphs += 1

    if verbose and num_graphs % 500 == 0:
        print(f"Processed {num_graphs} graphs")

    label = {mp_id: {graph_id: label_data}}
    failed_graphs = []
    isolated_atom_graphs = []

    graph_path = os.path.join(GRAPH_DIR, f"{graph_id}.pt")
    if not REMAKE and os.path.exists(graph_path):
        return (label, failed_graphs, isolated_atom_graphs)

    try:
        graph = converter(struct, graph_id=graph_id, mp_id=mp_id)
        torch.save(graph, graph_path)

    except Exception as e:
        isolated_atom_graphs.append((mp_id, graph_id))
        print(f"Graph failed for {mp_id}-{graph_id}: {e}")
        sys.stdout.flush()

    return (label, failed_graphs, isolated_atom_graphs)

# === MAIN MULTIPROCESSING DRIVER ===
def run_with_pool(n_procs=16):
    start_time = time.time()

    structure_dicts = get_structure_dicts(JSON_PATH)
    print(f"Loaded {len(structure_dicts)} structures")

    pool = multiproc.Pool(processes=n_procs)
    results = pool.map(create_graph, structure_dicts)
    pool.close()
    pool.join()

    labels = {}
    failed_graphs = []
    isolated_atom_graphs = []

    for result in results:
        for key in result[0]:
            labels.setdefault(key, {}).update(result[0][key])
        failed_graphs.extend(result[1])
        isolated_atom_graphs.extend(result[2])

    os.makedirs(GRAPH_DIR, exist_ok=True)
    with open(os.path.join(GRAPH_DIR, "labels.json"), "w") as f:
        json.dump(labels, f, indent=4)
    with open(os.path.join(GRAPH_DIR, "failed_graphs.txt"), "w") as f:
        f.write(str(failed_graphs))
    with open(os.path.join(GRAPH_DIR, "isolated_atom_graphs.txt"), "w") as f:
        f.write(str(isolated_atom_graphs))

    print(f"✅ Done in {time.time() - start_time:.2f} seconds.")

# === RUN ===
if __name__ == "__main__":
    run_with_pool()

