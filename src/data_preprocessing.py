import numpy as np
import matplotlib.pyplot as plt
import json 
import os
import sys
from pymatgen.core.structure import Structure
from pymatgen.core import Composition
from typing import List, Optional
import gzip 

# ========= Utils =========
def read_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def write_json(json_file, data):
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def extract_json_from_gzip(gzip_file, output_json):
    with gzip.open(gzip_file, 'rb') as f_in:
        with open(output_json, 'wb') as f_out:
            f_out.write(f_in.read())

class Filter:
    """
    A class to filter a dataset of materials based on various criteria.
    Attributes:
        data (List[dict]): A list of dictionaries containing material data.
    
    Methods:
        filter: Filters the dataset based on specified criteria.
        _filter_by_material_type: Filters the dataset by material type.
    
    Example:
        data = [
            {"elements": ["Fe", "O"], "nelements": 2, "bandgap": 1.5, "symmetry": {"crystal_system": "cubic"}},
            {"elements": ["Si"], "nelements": 1, "bandgap": 1.1, "symmetry": {"crystal_system": "cubic"}},
            {"elements": ["Na", "Cl"], "nelements": 2, "bandgap": 0.0, "symmetry": {"crystal_system": "cubic"}},
        ]
        filter = Filter(data)
        filtered_data = filter.filter(n_elements=2, contains_elements=["Fe"], bandgap_min=1.0)
    """

    def __init__(self, data: List[dict]):
        self.data = data

    def filter(
        self,
        n_elements: Optional[int] = None,
        contains_elements: Optional[List[str]] = None,
        excludes_elements: Optional[List[str]] = None,
        crystal_system: Optional[str] = None,
        bandgap_min: Optional[float] = None,
        bandgap_max: Optional[float] = None,
        material_type: Optional[str] = None  # e.g., 'oxide', 'intermetallic', etc.
    ) -> List[dict]:
        results = self.data

        if n_elements is not None:
            results = [d for d in results if d.get("nelements") == n_elements]

        if contains_elements is not None:
            contains_set = set(contains_elements)
            results = [d for d in results if contains_set.issubset(set(d.get("elements", [])))]

        if excludes_elements is not None:
            excludes_set = set(excludes_elements)
            results = [d for d in results if not excludes_set.intersection(set(d.get("elements", [])))]

        if crystal_system is not None:
            results = [d for d in results if d.get("symmetry", {}).get("crystal_system") == crystal_system.lower()]

        if bandgap_min is not None or bandgap_max is not None:
            bandgap_min = bandgap_min if bandgap_min is not None else 0.0
            bandgap_max = bandgap_max if bandgap_max is not None else float("inf")
            results = [d for d in results if bandgap_min <= d.get("bandgap", 0.0) <= bandgap_max]

        if material_type is not None:
            results = self._filter_by_material_type(results, material_type.lower())

        return results

    def _filter_by_material_type(self, dataset: List[dict], mtype: str) -> List[dict]:
        if mtype == "oxide":
            return [d for d in dataset if "O" in d.get("elements", [])]
        elif mtype == "transition-metal-oxide":
            tmos = {
                    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg'
                    }
            return [d for d in dataset if "O" in d.get("elements", []) and any(tm in tmos for tm in d.get("elements", []))]
        elif mtype == "nitride":
            return [d for d in dataset if "N" in d.get("elements", [])]
        elif mtype == "sulfide":
            return [d for d in dataset if "S" in d.get("elements", [])]
        elif mtype == "halide":
            halogens = {"F", "Cl", "Br", "I"}
            return [d for d in dataset if halogens.intersection(d.get("elements", []))]
        elif mtype == "intermetallic":
            non_metals = {"H", "B", "C", "N", "O", "F", "P", "S", "Cl", "Se", "Br", "I", "At", "Ts"}
            return [d for d in dataset if not non_metals.intersection(d.get("elements", []))]
        elif mtype == "metal":
            return [d for d in dataset if d.get("bandgap", None) == 0.0]
        elif mtype == 'all':
            return dataset
        else:
            raise ValueError(f"Unknown material type: {mtype}")
        
class DataExtracter:
    """
    A class to extract all the relevant information you need to fine-tune CHGNet (note that magmoms and stresses are optional but forces, energies and structures are required).
    Attributes:
        data (List[dict]): A list of dictionaries containing material data.
    Methods:
        get_energies_per_atom: Returns a list of energies per atom.
        get_forces: Returns a list of forces.
        get_strucs: Returns a list of structures.
        get_magmoms: Returns a list of magnetic moments.
        get_stresses: Returns a list of stresses.
    """
    def __init__(self, data: List[dict]):
        self.data = data

    def get_energies_per_atom(self) -> List[float]:
        total_energies =  [d.get("energy", 0.0) for d in self.data]
        number_atoms = [d.get("nsites", 0) for d in self.data]
        per_atom_energies = [e / n if n > 0 else 0.0 for e, n in zip(total_energies, number_atoms)]
        return per_atom_energies
    
    def get_forces(self) -> List[List[float]]:
        return [d.get("forces", []) for d in self.data]
    
    def get_strucs(self) -> List[Structure]:
        return [Structure.from_dict(d.get("structure", {})) for d in self.data]
    
    def get_magmoms(self) -> List[float]:
        magmoms = []
        for entry in self.data:
            structure = entry.get("structure", {})
            sites = structure.get("sites", [])
            magmom = [
                s.get("properties", {}).get("magmom", 0.0)
                for s in sites
            ]
            magmoms.append(magmom)
        return magmoms
    
    def get_stresses(self) -> List[List[float]]:
        return [d.get("stress", []) for d in self.data]

class EnergyCorrector:
    corrections = {
        "O": -0.687,  # assume all O is oxide (safe for most of MatPES)
        "S": -0.503,
        "F": -0.462,
        "Cl": -0.614,
        "Br": -0.534,
        "I": -0.379,
        "N": -0.361,
        "Se": -0.472,
        "Si": 0.071,
        "Sb": -0.192,
        "Te": -0.422,
        "H": -0.179,
        # Skip peroxide/superoxide/ozonide for speed
    }

    def __init__(self, data):
        self.data = data

    def apply_corrections(self):
        corrected_entries = []
        for idx, entry in enumerate(self.data):
            corrected_energy = self.correct_entry_energy(entry)
            entry["corrected_energy"] = corrected_energy
            corrected_entries.append(entry)
            if (idx + 1) % 2000 == 0:
                print(f"Processed {idx + 1} entries.")
        return corrected_entries

    def correct_entry_energy(self, entry):
        comp = Composition(entry["composition"])
        correction = 0.0
        for el, amt in comp.items():
            if str(el) in self.corrections:
                correction += amt * self.corrections[str(el)]
        return entry["energy"] + correction