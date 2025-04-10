import os
import subprocess
import sys
sys.path.append(os.path.expanduser("~/bin/CHGNet-finetuning"))
from data_preprocessing import Filter, DataExtracter, read_json, write_json, extract_json_from_gzip
from chgnet.model import CHGNet
from chgnet.trainer import Trainer
from chgnet.data.dataset import StructureData,GraphData, get_train_val_test_loader

HOME = os.path.expanduser("~")
CURR_DIR = os.getcwd()

# ======== CONFIGS to be set by user ========
DATA_PATH = os.path.join('..','data')

MATERIAL_TYPE = 'transition-metal-oxide'

SRC_CODE_PATH = os.path.join(HOME, 'bin','CHGNet-finetuning')

LEARNING_RATE = 1e-2

EPOCHS = 5

TRAIN_COMPOSITION_MODEL = False

GRAPHS = True

ATOM_GRAPH_CUTOFF = 7

BOND_GRAPH_CUTOFF = 3

READY_TO_TRAIN = False
# =========================================

def main():

    ready_to_train = READY_TO_TRAIN

    # Checks inside DATA_PATH for .json or .gz file
    # If both types are present simply read the .json file
    # If only .gz file is present, extract it and read the .json file

    for file in os.listdir(DATA_PATH):
        if file.endswith('.json'):
            all_data = read_json(os.path.join(DATA_PATH, file))
            break
        elif file.endswith('.gz'):
            extract_json_from_gzip(os.path.join(DATA_PATH, file), os.path.join(DATA_PATH, 'MatPES-PBE-2025.1.json'))
            all_data = read_json(os.path.join(DATA_PATH, 'MatPES-PBE-2025.1.json'))
            break
        else:
            raise ValueError(f"No data files found in {DATA_PATH}")

    # Filter the dataset
    filter = Filter(all_data)
    filtered_data = filter.filter(material_type=MATERIAL_TYPE)
    if GRAPHS:
        # Save the filtered data to a new JSON file
        write_json(os.path.join(DATA_PATH, MATERIAL_TYPE+'.json'), filtered_data)

        json_path = os.path.join(DATA_PATH, MATERIAL_TYPE+'.json')
        graph_dir = os.path.join(DATA_PATH, MATERIAL_TYPE+'-graphs')
        remake = False
        verbose = True
        num_graphs = 0
        atom_graph_cutoff = ATOM_GRAPH_CUTOFF
        bond_graph_cutoff = BOND_GRAPH_CUTOFF

        # Make the graph directory if it doesn't exist
        if not os.path.exists(graph_dir):
            os.makedirs(graph_dir)

        # Copying the generate_graphs.py file from the source code directory to the current working directory
        generate_graphs_path = os.path.join(SRC_CODE_PATH, 'generate_graphs.py')
        dest_generate_graphs_path = os.path.join(CURR_DIR, 'generate_graphs.py')
        subprocess.run(['cp', generate_graphs_path, dest_generate_graphs_path])

        # Editing the generate_graphs.py file to set the JSON_PATH and GRAPH_DIR variables
        with open(dest_generate_graphs_path, 'r') as f:
            lines = f.readlines()
        with open(dest_generate_graphs_path, 'w') as f:
            for line in lines:
                if line.startswith('JSON_PATH ='):
                    f.write(f'JSON_PATH = "{json_path}"\n')
                elif line.startswith('GRAPH_DIR ='):
                    f.write(f'GRAPH_DIR = "{graph_dir}"\n')
                elif line.startswith('REMAKE ='):
                    f.write(f'REMAKE = {remake}\n')
                elif line.startswith('VERBOSE ='):
                    f.write(f'VERBOSE = {verbose}\n')
                elif line.startswith('num_graphs ='):
                    f.write(f'num_graphs = {num_graphs}\n')
                elif line.startswith('ATOM_GRAPH_CUTOFF ='):
                    f.write(f'ATOM_GRAPH_CUTOFF = {atom_graph_cutoff}\n')
                elif line.startswith('BOND_GRAPH_CUTOFF ='):
                    f.write(f'BOND_GRAPH_CUTOFF = {bond_graph_cutoff}\n')
                else:
                    f.write(line)
        
        # Running the generate_graphs.py script
        subprocess.run(['python3', dest_generate_graphs_path])
        if ready_to_train:
            
            dataset = GraphData(
                        graph_path = os.path.join(os.getcwd(),'tmo-graphs'),
                        labels = os.path.join(os.getcwd(),'tmo-graphs/labels.json')
                        )

            train_loader, val_loader, test_loader = dataset.get_train_val_test_loader(
                batch_size=8, train_ratio=0.9, val_ratio=0.05
            )
            # Load pretrained CHGNet
            chgnet = CHGNet.load()

            # Optionally fix the weights of some layers
            for layer in [
                chgnet.atom_embedding,
                chgnet.bond_embedding,
                chgnet.angle_embedding,
                chgnet.bond_basis_expansion,
                chgnet.angle_basis_expansion,
                chgnet.atom_conv_layers[:-1],
                chgnet.bond_conv_layers,
                chgnet.angle_layers,
            ]:
                for param in layer.parameters():
                    param.requires_grad = False

            # Define Trainer
            trainer = Trainer(
                model=chgnet,
                targets="ef",
                optimizer="Adam",
                scheduler="CosLR",
                criterion="MSE",
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                use_device="cpu",
                print_freq=6,)
            
            save_dir = os.path.join('..','fine_tuned_models',)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_name = MATERIAL_TYPE + 'LR_' + str(LEARNING_RATE) + 'E_' + str(EPOCHS) + str(TRAIN_COMPOSITION_MODEL) + 'AC_' + str(ATOM_GRAPH_CUTOFF) + 'BC' + str(BOND_GRAPH_CUTOFF)

            trainer.train(train_loader, 
                        val_loader, 
                        test_loader, 
                        train_composition_model = TRAIN_COMPOSITION_MODEL,
                        save_dir = os.path.join(save_dir, save_name),)

    else:
        if ready_to_train:
            ex = DataExtracter(filtered_data)
            
            structures = ex.get_strucs()
            energies = ex.get_energies_per_atom()
            forces = ex.get_forces()
            stresses = ex.get_stresses()
            magmoms = ex.get_magmoms()

            dataset = StructureData(
                structures=structures,
                energies=energies,
                forces=forces,
                stresses=None,  
                magmoms=None,) 

            train_loader, val_loader, test_loader = get_train_val_test_loader(
                batch_size=8, train_ratio=0.9, val_ratio=0.05
            )
            # Load pretrained CHGNet
            chgnet = CHGNet.load()

            # Optionally fix the weights of some layers
            for layer in [
                chgnet.atom_embedding,
                chgnet.bond_embedding,
                chgnet.angle_embedding,
                chgnet.bond_basis_expansion,
                chgnet.angle_basis_expansion,
                chgnet.atom_conv_layers[:-1],
                chgnet.bond_conv_layers,
                chgnet.angle_layers,
            ]:
                for param in layer.parameters():
                    param.requires_grad = False

            # Define Trainer
            trainer = Trainer(
                model=chgnet,
                targets="ef",
                optimizer="Adam",
                scheduler="CosLR",
                criterion="MSE",
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                use_device="cpu",
                print_freq=6,)
            
            save_dir = os.path.join('..','fine_tuned_models',)
            
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            save_name = MATERIAL_TYPE + 'LR_' + str(LEARNING_RATE) + 'E_' + str(EPOCHS) + str(TRAIN_COMPOSITION_MODEL)

            trainer.train(train_loader, 
                        val_loader, 
                        test_loader, 
                        train_composition_model = TRAIN_COMPOSITION_MODEL,
                        save_dir = os.path.join(save_dir, save_name),)

    return

if __name__ == "__main__":
    main()







