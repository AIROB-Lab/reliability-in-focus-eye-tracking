# Trust in Prosthesis - Analysis

## Installation and Setup

Clone this project to your local machine and make sure python and poetry are installed 
(otherwise: see [this guide for a full multiple project python setup](https://github.com/mad-lab-fau/mad-cookiecutter/blob/main/python-setup-tips.md)
or [this guide for poetry](https://python-poetry.org/docs/#installing-with-pipx)).

Then, in your project folder run the following command to install the dependencies:

```bash
poetry install --all-extras
```

Next, you should copy the [ `default_config.json` ](default_config.json) file, rename it to `config.json` and edit it 
to point to the data folder that contains all of the participant data.

If you want to work with the jupyter notebooks, you can register a new kernel with the following command:

```bash
poe register_ipykernel
```

## Running the analysis
The full feature calculation is done in the 
1. [`feature_calculations.py`](trust_in_prosthesis_analysis/eye_tracking/feature_calculations.py) file.

Not needed anymore:
Basically the same calculation can also be found in the 
[`Feature_Calculation.ipynb`](notebooks/Feature_Calculation.ipynb) notebook (!without the EAL/ELL features though).
[`SavePastaBoxResults.ipynb`](notebooks/study_feature_analysis/SavePastaboxResults.ipynb) Calculates and saves PastaBoxResults.csv
[`ExclusionAndCompleteness`](notebooks/study_feature_analysis/ExclusionsAndCompleteness.ipynb) Performs completenes checks,Calculate the eye tracking validity per round, performs exclusions and saves clnd_PastaBoxResults.csv


Some example code on how to plot the features can be found in the [`Plot_PhasesMovsFixations.ipynb`](notebooks/Plot_PhasesMovsFixations.ipynb) notebook.

Most of the python files have a somewhat sane `__main__` block that can be used to test out the code.
