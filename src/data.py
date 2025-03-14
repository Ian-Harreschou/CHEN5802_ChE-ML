import json
import os
import numpy as np
from pymatgen.core import Structure
from dpdata import LabeledSystem

class DeePMDDatasetProcessor:
    """Processes a large JSON dataset, extracts structures, and converts to DeePMD format."""

    def __init__(self, input_file, output_dir, deepmd_dir, batch_size=10000):
        """
        Initializes the dataset processor.
        :param input_file: Path to the large JSON dataset.
        :param output_dir: Directory to store extracted JSON batches.
        :param deepmd_dir: Directory to store DeePMD-ready data.
        :param batch_size: Number of structures per batch.
        """
        self.input_file = input_file
        self.output_dir = output_dir
        self.deepmd_dir = deepmd_dir
        self.batch_size = batch_size

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(deepmd_dir, exist_ok=True)

    def process_json_and_convert(self):
        """Extracts data from JSON, splits it into batches, and converts to DeePMD format."""
        with open(self.input_file, "r") as f:
            data = json.load(f)

        batch, batch_count = [], 0

        for mp_id, mp_data in data.items():
            for frame_id, frame_data in mp_data.items():
                if "structure" not in frame_data:
                    continue  # Skip if no structure

                # Extract structure and convert to pymatgen object
                structure = Structure.from_dict(frame_data["structure"])
                positions = np.array([site.coords for site in structure.sites])  # Atomic positions
                species = [site.species_string for site in structure.sites]  # Element types

                # Extract labels
                energy = frame_data.get("energy_per_atom", None)
                forces = np.array(frame_data.get("force", []))
                stress = np.array(frame_data.get("stress", []))

                # Store in batch
                batch.append({
                    "atoms": species,
                    "positions": positions.tolist(),
                    "energy": energy,
                    "forces": forces.tolist(),
                    "stress": stress.tolist()
                })

                # Save batch and convert when reaching batch_size
                if len(batch) >= self.batch_size:
                    batch_path = self._save_batch(batch, batch_count)
                    self._convert_to_deepmd(batch_path, batch_count)
                    batch, batch_count = [], batch_count + 1

        # Save remaining data
        if batch:
            batch_path = self._save_batch(batch, batch_count)
            self._convert_to_deepmd(batch_path, batch_count)

    def _save_batch(self, batch, batch_count):
        """Helper function to save a batch to a JSON file."""
        batch_file = os.path.join(self.output_dir, f"batch_{batch_count}.json")
        with open(batch_file, "w") as f_out:
            json.dump(batch, f_out, indent=4)
        print(f"Saved {batch_file} with {len(batch)} entries.")
        return batch_file

    def _convert_to_deepmd(self, batch_file, batch_count):
        """Converts a batch JSON file to DeePMD format."""
        dp_output_path = os.path.join(self.deepmd_dir, f"batch_{batch_count}")
        ls = LabeledSystem(batch_file, fmt="deepmd/npy")
        ls.to_deepmd_npy(dp_output_path)
        print(f"Converted {batch_file} to DeePMD format at {dp_output_path}")
