# Trust in Prosthesis - Analysis

## Installation and Setup

Clone this project to your local machine and make sure python and poetry are installed 
(otherwise: see [this guide for a full multiple project python setup](https://github.com/mad-lab-fau/mad-cookiecutter/blob/main/python-setup-tips.md)
or [this guide for poetry](https://python-poetry.org/docs/#installing-with-pipx)).

Then, in your project folder run the following command to install the dependencies:

```bash
poetry install --all-extras
```
Next, you should edit the  `config.json` to point to the data folder that contains all of the participant data.

## Running the analysis
The full feature calculation is done in the 
* [`feature_calculations.py`](trust_in_prosthesis_analysis/eye_tracking/feature_calculations.py) file.
* For single calculations and exploration call TPCP dataset function (see Dataset)
