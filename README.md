# Alternative Feature Selection

This repository contains the code for

- two papers,
- (parts of) a dissertation,
- and the Python package [`alfese`](https://pypi.org/project/alfese/).

This document provides:

- An overview of the [related publications](#publications).
- An outline of the [repo structure](#repo-structure).
- Steps for [setting up](#setup) a virtual environment and [reproducing](#reproducing-the-experiments) the experiments.

## Publications

> Bach, Jakob, and Klemens Böhm (2024): "Alternative feature selection with user control"

is published in the [International Journal of Data Science and Analytics](https://link.springer.com/journal/41060).
You can find the paper [here](https://doi.org/10.1007/s41060-024-00527-8).
You can find the corresponding complete experimental data (inputs as well as results) on [*RADAR4KIT*](https://doi.org/10.35097/1975).
Use the tags `run-2023-06-23` and `evaluation-2024-03-19` for reproducing the experiments.

> Bach, Jakob (2023): "Finding Optimal Diverse Feature Sets with Alternative Feature Selection"

is published on [*arXiv*](https://arxiv.org/).
You can find the paper [here](https://doi.org/10.48550/arXiv.2307.11607).
You can find the corresponding complete experimental data (inputs as well as results) on *RADAR4KIT*.
Use the tags `run-2023-06-23` and `evaluation-2023-07-04` for reproducing the [experimental data for v1](https://doi.org/10.35097/1623) of the paper.
Use the tags `run-2024-01-23` and `evaluation-2024-02-01` for reproducing the [experimental data for v2](https://doi.org/10.35097/1920) of the paper.
Use the tags `run-2024-09-28-arXiv-v3` and `evaluation-2024-12-08-arXiv-v3` for reproducing the [experimental data for v3](https://doi.org/10.35097/4ttgrpx92p30jwww) of the paper.

> Bach, Jakob (2025): "Leveraging Constraints for User-Centric Feature Selection"

is a dissertation at the [Department of Informatics](https://www.informatik.kit.edu/english/index.php) of the [Karlsruhe Institute of Technology](https://www.kit.edu/english/).
You can find the dissertation [here](https://doi.org/10.5445/IR/1000178649).
You can find the corresponding complete experimental data (inputs as well as results) on [*RADAR4KIT*](https://doi.org/10.35097/4kjyeg0z2bxmr6eh).
Use the tags `run-2024-09-28-dissertation` and `evaluation-2024-11-02-dissertation` for reproducing the experiments.

## Repo Structure

Currently, the repository contains six Python files and four non-code files.
The non-code files are:

- `.gitignore`: For Python development.
- `LICENSE`: The software is MIT-licensed, so feel free to use the code.
- `README.md`: You are here :upside_down_face:
- `requirements.txt`: To set up an environment with all necessary dependencies; see below for details.

The code files comprise our experimental pipeline (see below for details):

- `prepare_datasets.py`: First stage of the experiments
  (download prediction datasets).
- `run_experiments.py`: Second stage of the experiments
  (run feature selection, search for alternatives, and make predictions).
- `run_evaluation_(arxiv|dissertation|journal).py`: Third stage of the experiments
  (compute statistics and create plots).
- `data_handling.py`: Functions for working with prediction datasets and experimental data.

Additionally, we have organized the (alternative) feature-selection methods for our experiments
as the standalone Python package `alfese`, located in the directory `alfese_package/`.
See the corresponding [README](alfese_package/README.md) for more information.

## Setup

Before running the scripts to reproduce the experiments, you should

1) Set up an environment (optional but recommended).
2) Install all necessary dependencies.

Our code is implemented in Python (version 3.8; other versions, including lower ones, might work as well).

### Option 1: `conda` Environment

If you use `conda`, you can directly install the correct Python version into a new `conda` environment
and activate the environment as follows:

```bash
conda create --name <conda-env-name> python=3.8
conda activate <conda-env-name>
```

Choose `<conda-env-name>` as you like.

To leave the environment, run

```bash
conda deactivate
```

### Option 2: `virtualenv` Environment

We used [`virtualenv`](https://virtualenv.pypa.io/) (version 20.4.7; other versions might work as well)
to create an environment for our experiments.
First, you need to install the correct Python version yourself.
Let's assume the Python executable is located at `<path/to/python>`.
Next, you install `virtualenv` with

```bash
python -m pip install virtualenv==20.4.7
```

To set up an environment with `virtualenv`, run

```bash
python -m virtualenv -p <path/to/python> <path/to/env/destination>
```

Choose `<path/to/env/destination>` as you like.

Activate the environment in Linux with

```bash
source <path/to/env/destination>/bin/activate
```

Activate the environment in Windows (note the back-slashes) with

```cmd
<path\to\env\destination>\Scripts\activate
```

To leave the environment, run

```bash
deactivate
```

### Dependency Management

After activating the environment, you can use `python` and `pip` as usual.
To install all necessary dependencies for this repo, run

```bash
python -m pip install -r requirements.txt
```

If you make changes to the environment and you want to persist them, run

```bash
python -m pip freeze > requirements.txt
```

## Reproducing the Experiments

After setting up and activating an environment, you are ready to run the code.
Run

```bash
python -m prepare_datasets
```

to download and pre-process the input data for the experiments (prediction datasets from PMLB).

Next, start the experimental pipeline with

```bash
python -m run_experiments
```

Depending on your hardware, this might take several days.
For the last pipeline run, we had a runtime of 141 hours on a server with an `AMD EPYC 7551`
[CPU](https://www.amd.com/en/products/cpu/amd-epyc-7551) (32 physical cores, base clock of 2.0 GHz).
In case the pipeline is nearly finished but doesn't make progress anymore,
the solver might have silently crashed (which happened in the past with `Cbc` as the solver, though
we didn't encounter the phenomenon with the current solver `SCIP`).
In this case, or if you had to abort the experimental run for other reasons, you could re-start the
experimental pipeline by calling the same script again; it automatically detects existing results
and only runs the remaining tasks.

To print statistics and create the plots, run

```bash
python -m run_evaluation_<<version>>
```

with `<<version>>` being one of `arxiv`, `dissertation`, or `journal`.

(The evaluation length differs between versions, as does the plot formatting.
The arXiv version has the longest and most detailed evaluation.)

All scripts have a few command-line options, which you can see by running the scripts like

```bash
python -m prepare_datasets --help
```
