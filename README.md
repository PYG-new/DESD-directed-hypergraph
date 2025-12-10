# Dirac-Equation Synchronization Dynamics (DESD) on Directed Hypergraphs

This repository contains the official implementation of **Dirac-Equation Synchronization Dynamics (DESD)** on directed hypergraphs, supporting analytical and experimental study of higher-order network synchronization. The framework integrates **Dirac operators**, **isolated eigenmodes**, and **Runge–Kutta numerical integration**, and is validated on both **high-school social contact hypergraphs** and **resting-state fMRI brain networks**.

### 📂 Project Structure
* DESD/
    * Dataset/
        * high_school/       # Empirical high-school hypergraph dataset
            * Brain_dataset.py
            * High_dataset.py
    * figure4.py
    * figure5.py
    * DESD_brain.py
    * DESD_High.py

### Requirements
All dependencies required to reproduce the Dirac-equation synchronization experiments are listed in requirements.txt.
You can install them using:
```
pip install -r requirements.txt
```
This will automatically install all necessary Python packages 
(e.g., NumPy, SciPy, Matplotlib, NetworkX, etc.) and ensure a consistent 
environment for running every experiment in the repository.

### Data generation
All datasets used in this project can be fully reproduced. 
We provide complete data-generation scripts for both the high-school hypergraph and the brain functional hypergraph. 
```
cd Dataset

python Brain_dataset.py     # Brain functional hypergraph

python High_dataset.py      # High-school hypergraph
```
### Run experiments

To run the Dirac-equation synchronization experiments, simply use:
```
cd DESD

python figure2_3.py

```

For the real-world hypergraph experiments (high-school social hypergraph and brain functional hypergraph), simply run:
```
cd DESD

python DESD_brain.py

python DESD_High.py
```
This script automatically loads the empirical datasets and executes all Dirac-equation synchronization analyses used in the paper.



