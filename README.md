# endomysial analysis

## Dependencies
To set up dependencies, run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

First navigate to this folder at the terminal, then run
```bash
source .venv/bin/activate
```

Image data (.czi files) should be placed in the `Data` folder. The analysis scripts will automatically look for files in that folder.

To look at the annotated images with highlighted line segments, run for each file:

```bash
python czi_annotation_analysis.py --file Data/<path_to_czi_file>
```

To run the analysis on all files and output tabulated results in a csv file, run:

```bash
python czi_batch_analysis.py
```

The script
```bash
python swarm_plots.py
```
then reads that csv file to create the swarm plots (Figs 4 and 5 in the paper) which provide a visual overview of the results. 
N.B. The warnings that are output about points not fitting on swarm plots are for when the script is showing a plot on your screen at lower resolution than the saved figures, and can be ignored.

```bash
python plot_group_means.py
```
plots the group means with their 95% confidence intervals (Fig 6 in the paper).