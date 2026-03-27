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

To look at the annotated images with highlighted line segments, run for each file:

```bash
python czi_annotation_analysis.py --file Data/<path_to_czi_file>
```

To run the analysis on all files and output tabulated results in a csv file, run:

```bash
python czi_batch_analysis.py
```
