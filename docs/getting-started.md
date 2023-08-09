## Install

The following will install COMPASS into a conda environment.

```bash
conda install -c conda-forge compass
```


Alternatively, you can install in development mode:

```bash
git clone https://github.com/opera-adt/COMPASS.git && cd COMPASS
conda env create --file environment.yml
conda activate compass
python -m pip install -e .
```

## Usage

The following commands generate coregistered SLC in radar or geo-coordinates from terminal:

```bash
s1_cslc.py --grid geo   <path to s1_cslc_geo   yaml file>
s1_cslc.py --grid radar <path to s1_cslc_radar yaml file for reference burst>
s1_cslc.py --grid radar <path to s1_cslc_radar yaml file for secondary burst>
```

## Creating Documentation


We use [MKDocs](https://www.mkdocs.org/) to generate the documentation.
The reference documentation is generated from the code docstrings using [mkdocstrings](mkdocstrings.github.io/).

When adding new documentation, you can build and serve the documentation locally using:

```
mkdocs serve
```
then open http://localhost:8000 in your browser.
Creating new files or updating existing files will automatically trigger a rebuild of the documentation while `mkdocs serve` is running.
