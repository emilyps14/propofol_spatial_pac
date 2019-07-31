# propofol_spatial_pac
[![Build Status](https://travis-ci.org/emilyps14/propofol_spatial_pac.svg?branch=master)](https://travis-ci.org/emilyps14/propofol_spatial_pac)

This package contains code for running the analyses described in:
>Stephen, E.P., Hotan, G.C., Pierce, E.T., Harrell, P.G., Walsh, J.L., Brown, 
>E.N., Purdon, P.L. (in submission). Broadband slow-wave modulation in 
>posterior and anterior cortex tracks distinct states of 
>propofol-induced unconsciousness. [doi.org/10.1101/712604](https://www.biorxiv.org/content/10.1101/712604v1)

To set up the code, run:
```
git clone https://github.com/emilyps14/propofol_spatial_pac.git
conda env update --file environment.yml
```

Note that the requirements are driven by MNE-Python, so if you run into 
installation issues, see their [installation instructions](https://martinos.org/mne/stable/install_mne_python.html#installing-python).

The scripts `1_single_subject_pipeline.py` and `2_across_subjects_pipeline.py` 
regenerate Figures 1-4 in the manuscript. The required data will be made 
available online when the manuscript is published. The variable `subjects_dir`
in each of these scripts should point to the folder containing the data, 
creating the following folder structure:
```
<subjects_dir>/eeganes07/input/
<subjects_dir>/input/
```

The code should work as-is if the repository is stored in the same folder, i.e.:
```
<subjects_dir>/propofol_spatial_pac/
```
