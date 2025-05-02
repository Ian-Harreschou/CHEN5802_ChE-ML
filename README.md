# CHEN5802
CHEN5802 (UofM) - Machine Learning Applications in ChemE &amp; MatSci

## Overview

This repository represents a collaborative project for a course offered at the University of Minnesota on the application of machine learning (ML) techniques to chemical engineering and materials science problems. The project explores how ML techniques can be used to approximate interatomic forces in crystals, aiming to enhance computational efficiency in molecular simulations.

## Dependencies
If you have an NVIDIA GPU the following applies to you:
To build the dependencies you should use the .yml file in the repo. Here is how you do it

```
conda env create -f 250310-ml-cuda-clean.yml
```

and then to activate it

```
conda activate ml-cuda
```

## How to use 

To run a fine tuning routine, only the launcher script needs to be modified. Best practice is to copy over the launcher script into whichever directory you are operating in then change the user configurations at the top of the script to fit the needs of your run. The launcher will take care of the rest. 

For running the RandomForestRegressor on the CHGNet embeddings, you do the same by copying the script into your working directory and editing the relevant parameters before launching. 

For users of the minnesota supercomputing institute, to succesfully run the code, you must create the anaconda environment using the project.yml file in the github repo. Follow the same steps outlined in the GPU environment creation above

## Data

https://matpes.ai/

## MLPs:

CHGNet: https://github.com/CederGroupHub/chgnet/blob/main/examples/fine_tuning.ipynb

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Disclaimer

This repository is provided "as is," without warranty of any kind, express or implied, including but not limited to the warranties of merchantability, fitness for a particular purpose, and noninfringement. In no event shall the authors or contributors be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

By using this repository, you acknowledge that the authors bear no responsibility for any consequences, intended or unintended, resulting from its use.
