# Enron Network Analysis

This repository contains the code, dataset, and exported outputs for the Enron email network analysis project.

The main analysis is in [notebooks/enron_network_project_final_v2.ipynb](./notebooks/enron_network_project_final_v2.ipynb). The notebook now prefers the local dataset in `data/raw/` and writes generated files into `data/processed/` and `outputs/`.

## Repository Structure

- `notebooks/`: final Jupyter notebook used for the Enron network analysis
- `data/raw/`: raw SNAP Enron email dataset
- `data/processed/`: exported CSV and JSON summary files
- `outputs/figures/`: figures and PDFs used for the report
- `outputs/gephi/`: GEXF files for Gephi visualization
- `scripts/`: helper script for downloading the raw dataset again if needed

## Dataset

- Included raw dataset: `data/raw/email-Enron.txt.gz`
- Original source: Stanford SNAP Enron email network
- Source page: https://snap.stanford.edu/data/email-Enron.html
- Direct download used by the project: https://snap.stanford.edu/data/email-Enron.txt.gz

## Environment Setup

Create a virtual environment and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How To Run

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Then open:

```text
notebooks/enron_network_project_final_v2.ipynb
```

Run all cells. The notebook will:

- load the Enron network from `data/raw/email-Enron.txt.gz`
- compute centrality, clustering, diameter/path-length approximations, and comparison metrics
- export summary tables to `data/processed/`
- export Gephi-ready graph files to `outputs/gephi/`
- save figures to `outputs/figures/`