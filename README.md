# Interplay of epidemic spreading and vaccine uptake under complex social contagion

This repository hosts the code for the work *Interplay of epidemic spreading and vaccine uptake under complex social contagion* by de Miguel-Arribas et al. The work is currently a pre-print on [https://arxiv.org/abs/2412.11766](ArXiV).

## Abstract

Modeling human behavior is essential to accurately predict epidemic spread, with behaviors like vaccine hesitancy complicating control efforts. While epidemic spread is often treated as a simple contagion, vaccine uptake may follow complex contagion dynamics, where individuals' decisions depend on multiple social contacts. Recently, the concept of complex contagion has received strong theoretical underpinnings thanks to the generalization of spreading phenomena from pairwise to higher-order interactions. Although several potential applications have been suggested, examples of complex contagions motivated by real data remain scarce. Surveys on COVID-19 vaccine hesitancy in the US suggest that vaccination attitudes may indeed depend on the vaccination status of social peers, aligning with complex contagion principles. In this work, we examine the interactions between epidemic spread, vaccination, and vaccine uptake attitudes under complex contagion. Using the SIR model with a dynamic, threshold-based vaccination campaign, we simulate scenarios on an age-structured multilayer network informed by US contact data. Our results offer insights into the role of social dynamics in shaping vaccination behavior and epidemic outcomes.

---

## Prerequisites

- **Rust**: Ensure you have the Rust programming language installed. You can install it using:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **Python 3.8+**: Install dependencies using pip:
  ```bash
  pip install -r requirements.txt
  ```

## Installation
 ```bash
 git clone https://github.com/phononautomata/threshold.git
 cd threshold
 ```

## Project Structure

This section provides an overview of the main directories and files in the project, explaining their purpose and how they fit into the overall project.

- **/config/**: Configuration files for running simulations.
- **/data/**:
  - **/curated/**: Processed data for the model input (contact structure, population, vaccination).
- **/src/**: Source code for the simulations and analysis, including both Rust (core) and Python code (results analysis and plotting).
- **/scripts/**: Bash scripts for automation tasks such as setup, build, and deployment.
- **/notebooks/**: Jupyter notebooks for data analysis and visualization.
- **/results/**:
  - **/temp/**: Temporary results from simulations and analyses.
  - **/plots/**: Final results used in the plotting functions for the draft.
- **/figures/**: Figures obtained from the plotting functions for intermediate reports or final draft.
- **README.md**: You are reading me, yes!
- **LICENSE.md**: License information.
- **requirements.txt**: Python dependencies.

## Usage examples
This project can be utilized in two primary ways: running stochastic simulations through Rust code and reproducing curated results and figures using Jupyter notebooks.

To run the simulations take into account that you'll need a network first for the dynamical process. The network can also be generated from the Rust code. The current version of the code only supports the data-driven age-multilayer structures of the work's main text. To generate the contact structure you can run:

### Generating multilayer structure 
To generate the contact structure you can run:

```bash
cargo run -r -- --id-experiment 1 --model-region "national" --nagents 100000
```

This creates an instance of a multilayer network of size $10^5$ following national US contact patterns.


### Running Stochastic Simulations with Rust

The core model's simulations are performed by executing Rust code with specific flags. Here’s how to run a simulation:

1. **Run simulations under config file default**:
   ```bash
   cargo run --r --id-experiment 2
   ```

2. **Run simulations with input parameters from command line**
   ```bash
   cargo run -r --id-experiment 2 --fraction-active 0.1 --threshold-opinion 0.25 --rate-vaccination 0.001
   ```
   This command runs an ensemble of simulations (assuming a network id is passed by default) where the initial fraction of pro-active individuals is $n_A(0)=0.1$, the (homogeneous) activation threshold is $\theta=0.25$, and the vaccination rate is set to $\alpha=0.001$.

### Running Jupyter notebooks
The main text results can be obtained (provided the required data from the simulations is at hand) running analysis_paper.ipynb

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
