import re
from pymatgen.core.structure import Structure
import os
import sys
from chgnet.model import CHGNet
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
from sklearn.linear_model import LinearRegression

SRC_CODE_DIR = '/Users/christopherakiki/CHEN5802_ChE-ML/src'
FINE_TUNED_MODEL_PATH = '/Users/christopherakiki/bin/CHEN5802_ChE-ML/model-tarball/bestE_epoch3_e98_f193_sNA_mNA.pth.tar'

sys.path.append(os.path.expanduser(SRC_CODE_DIR))
from data_preprocessing import Filter, DataExtracter, EnergyCorrector, read_json, write_json, extract_json_from_gzip

# Replace these with where your data is stored
EQUILIBRIUM_DFT_DATA = '/Users/christopherakiki/CHEN5802_ChE-ML/DFT_data/equilibrium_data/results.json'
PERTURBED_DFT_DATA = '/Users/christopherakiki/CHEN5802_ChE-ML/DFT_data/perturbed_data/results.json'

def read_results(results_file):
    return read_json(results_file)

def evaluate_and_plot(dft_results,
                      parity_plot_savename,
                      parity_plot_with_syst_softening_corr_savename,):
    """
    This function evaluates the DFT results and plots the results.
    :param dft_results: The DFT results to evaluate.
    :return: None
    """
    # Extracting the energies, compounds, and structures from the DFT results

    energies = []
    compounds = []
    structures = []

    for entry in dft_results:
        energy = dft_results[entry]['results']['E_per_at']
        energies.append(energy)
        split = re.split(r'--', entry)
        compound = split[0]
        compounds.append(compound)
        if dft_results[entry]['structure']:
            structure = Structure.from_dict(dft_results[entry]['structure'])
            structures.append(structure)
        else:
            structures.append(None)

    # zipping the three lists together
    df = pd.DataFrame(list(zip(compounds, energies, structures)), columns=['compound', 'dft_energy', 'structure'])

    # Dropping the rows that have Nan values in them
    df = df.dropna()
    df = df.reset_index(drop=True)


    # Loading the fine-tuned model
    chgnet = CHGNet.from_file(FINE_TUNED_MODEL_PATH)

    # Evaluating energies on the fine-tuned model
    finetuned_energies = chgnet.predict_structure(df['structure'],task='e')

    # Loading the pretrained model
    chgnet_pretrained = CHGNet.load(model_name='0.3.0')

    # Evaluating energies on the pretrained model
    pretrained_energies = chgnet_pretrained.predict_structure(df['structure'],task='e')

    # creating emtpy columns for finetuned and pretrained energies
    df['finetuned_energy'] = None
    df['pretrained_energy'] = None

    for i in range(len(df)):
        df['finetuned_energy'][i] = finetuned_energies[i]['e']
        df['pretrained_energy'][i] = pretrained_energies[i]['e']

    # Parity plot of finetuned energies vs dft energies with R^2 value

    # Calculate R^2 value
    r2 = r2_score(df['dft_energy'], df['finetuned_energy'])
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['dft_energy'], df['finetuned_energy'], color='red', label='Finetuned Energy')
    plt.plot([min(df['dft_energy']), max(df['dft_energy'])], [min(df['dft_energy']), max(df['dft_energy'])], color='black', linestyle='--')
    plt.xlabel('DFT Energy (eV)')
    plt.ylabel('Predicted Energy (eV/atom)')
    plt.title(f'Parity Plot of DFT vs Predicted Energies (R^2 = {r2:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(parity_plot_savename)
    plt.show()

    E_pred = np.array(df['finetuned_energy'])   # 50 CHGNet energies (same units & per‑atom basis)
    E_DFT  = np.array(df['dft_energy'])   # 50 reference DFT energies

    lr = LinearRegression(fit_intercept=True).fit(E_pred.reshape(-1,1), E_DFT)
    a, c = lr.intercept_, lr.coef_[0]          # E_DFT ≈ a + c·E_pred
    print(f"a = {a:.3f}  eV,   c = {c:.3f}")

    E_corr = c * E_pred + a

    df['finetuned_systematically_corrected_energy'] = E_corr
    
    # Parity plot of systematically corrected finetuned energies vs dft energies with R^2 value
    
    # Calculate R^2 value
    r2 = r2_score(df['dft_energy'], df['finetuned_systematically_corrected_energy'])
    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['dft_energy'], df['finetuned_systematically_corrected_energy'], color='red', label='Finetuned Energy')
    plt.plot([min(df['dft_energy']), max(df['dft_energy'])], [min(df['dft_energy']), max(df['dft_energy'])], color='black', linestyle='--')
    plt.xlabel('DFT Energy (eV)')
    plt.ylabel('Predicted Energy (eV/atom)')
    plt.title(f'Parity Plot of DFT vs Predicted Energies (R^2 = {r2:.2f})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(parity_plot_with_syst_softening_corr_savename)
    plt.show()
    
    finetuned_mae = mean_absolute_error(df['dft_energy'], df['finetuned_energy'])
    pretrained_mae = mean_absolute_error(df['dft_energy'], df['pretrained_energy'])
    systematically_corrected_finetuned_mae = mean_absolute_error(df['dft_energy'], df['finetuned_systematically_corrected_energy'])
    finetuned_mae = finetuned_mae * 1000
    pretrained_mae = pretrained_mae * 1000
    systematically_corrected_finetuned_mae = systematically_corrected_finetuned_mae * 1000
    print(f"Finetuned MAE: {finetuned_mae} meV/atom")
    print(f"Pretrained MAE: {pretrained_mae} meV/atom")
    print(f"Systematically Corrected Finetuned MAE: {systematically_corrected_finetuned_mae} meV/atom")

    metrics = {
        'finetuned_mae': finetuned_mae,
        'pretrained_mae': pretrained_mae,
        'systematically_corrected_finetuned_mae': systematically_corrected_finetuned_mae,
    }

    return metrics

def main():
    # Read the DFT results
    dft_results = read_results(EQUILIBRIUM_DFT_DATA)

    # Evaluate and plot the DFT results
    metrics = evaluate_and_plot(dft_results,
                                 parity_plot_savename='parity_plot_finetuned.png',
                                 parity_plot_with_syst_softening_corr_savename='parity_plot_finetuned_with_syst_softening_corr.png',)

    # Save the metrics to a json file
    write_json('metrics.json', metrics)

    return metrics

if __name__ == '__main__':
    main()



