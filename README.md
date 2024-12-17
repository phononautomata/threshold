
# Interplay of epidemic spreading and vaccine uptake under complex social contagion

This repository hosts the code for the work *Interplay of epidemic spreading and vaccine uptake under complex social contagion* by de Miguel-Arribas et al. The work is currently a pre-print on [https://arxiv.org/abs/2412.11766](ArXiv).

Abstract:

Modeling human behavior is essential to accurately predict epidemic spread, with behaviors like vaccine hesitancy complicating control efforts. While epidemic spread is often treated as a simple contagion, vaccine uptake may follow complex contagion dynamics, where individuals' decisions depend on multiple social contacts. Recently, the concept of complex contagion has received strong theoretical underpinnings thanks to the generalization of spreading phenomena from pairwise to higher-order interactions. Although several potential applications have been suggested, examples of complex contagions motivated by real data remain scarce. Surveys on COVID-19 vaccine hesitancy in the US suggest that vaccination attitudes may indeed depend on the vaccination status of social peers, aligning with complex contagion principles. In this work, we examine the interactions between epidemic spread, vaccination, and vaccine uptake attitudes under complex contagion. Using the SIR model with a dynamic, threshold-based vaccination campaign, we simulate scenarios on an age-structured multilayer network informed by US contact data. Our results offer insights into the role of social dynamics in shaping vaccination behavior and epidemic outcomes.

### Prerequisites

...

## Installation

...

## Project Structure

This section provides an overview of the main directories and files in the project, explaining their purpose and how they fit into the overall project.

- **/config/**: Configuration files and environment variables necessary for the project.
- **/data/**:
  - **/raw/**: Raw data files used in simulations and analysis.
  - **/curated/**: Processed data that is ready for analysis or model feeding.
- **/src/**: Source code for the project, including both Rust and Python code.
- **/scripts/**: Bash scripts for automation tasks such as setup, build, and deployment.
- **/notebooks/**: Jupyter notebooks for interactive data analysis, results reproduction, and figures generation.
- **/results/**:
  - **/temp/**: Temporary results from simulations and analyses.
  - **/plots/**: Final results used in the plotting functions for reports and papers.
- **/figures/**: Result .png or vectorial files for reports and papers.
- **Dockerfile**: Containerization of the project.
- **Makefile**: Contains commands for common tasks such as setup, build, test, and clean.
- **README.md**: You are reading me, yes!
- **LICENSE.md**: The license file specifying the terms under which the project is made available.

## Usage examples
This project can be utilized in two primary ways: running stochastic simulations through Rust code and reproducing curated results and figures using Jupyter notebooks. Below are instructions for both methods.

### Running Stochastic Simulations with Rust

The core model's simulations are performed by executing Rust code with specific flags. Here’s how to run a simulation:

1. **Run simulations under config file default**:
   ```bash
   cargo run
   ````

2. **Run simulations with input parameters from command line**
   ```bash
   cargo run -r --
   ```

   TODO

### Running Jupyter notebooks
TODO

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
