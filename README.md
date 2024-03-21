
# Project: THRESHOLD

This repository hosts the code for the work developed in QUOTE:THRESHOLD.

In the aforementioned reference we aimed to study the coupling of stochastic dynamical processes of different nature on a networked population. The spreading of epidemics is typically modeled through pairwise interactions thus constituting what is know as a simple contagion process. Vaccination dynamics on the other hand is quite simple, being typically modeled as a spontaneous (automatic) transition. By 'automatic' here I refer to the lack of agency on the vaccination process itself of the agent to be vaccinated. However, and here comes the catch, to be vaccinated or not is a behavioral attitude that may vary from individual to individual. Thus, what makes an individual to adopt a pro-vaccine stance? Whereas some individuals may assess more rationally or not pros and cons of being vaccinated to adopt a decision, others may look at their social peers to make up their minds. Therefore, at least for some, adopting pro-vaccine opinion could be understood as a social contagion process. Moreover, a social contagion process that depends not on pairwise interactions but on the overall status of a group of individuals. This leads us to the concept of complex contagion and higher-order interactions.

In brief, we aimed to assess the coupled dynamics of an epidemic spreading and a vaccination campaign where individuals' pro-vaccine stance is subjected to a threshold-based (Watts-Granovetter) opinion dynamics. Far from a fabricated thought experiment, posing vaccination-opinion dynamics as a higher-order threshold process has been inspired by some surveys on vaccine attitudes within the COVID-19 vaccination campaign that started in late 2020.

## Getting Started

These instructions will guide you through setting up the project on your local machine. This project is structured to use Python for scripting and automation, Rust for performance-critical components, and Docker for creating a consistent development and deployment environment.

### Prerequisites

Before you start, ensure you have the following installed on your system:

- **Git**: For cloning the repository.
- **Python**: This project uses Python for some scripting and analysis utilities. [Download and install Python](https://www.python.org/downloads/). This project has been tested with Python 3.9.
- **Jupyter Notebooks**: For reproducing curated results and generating figures, Jupyter notebooks are used. After installing Python, Jupyter can be installed via pip.
- **Rust**: Model's core is written in Rust. [Install Rust](https://www.rust-lang.org/tools/install) by following the official instructions.
- **Docker**: For ease of setup and deployment, this project is containerized with Docker. [Install Docker Desktop](https://docs.docker.com/desktop/) for your operating system.

### Installation

Follow these steps to get your development environment running:

1. **Clone the Repository**:
   Start by cloning the project repository to your local machine.
   ```bash
   git clone https://github.com/phononautomata/threshold.git
   ```

2. **Run the Makefile**:
   From your terminal, navigate to the project directory:
   ```bash
   cd threshold
   ```
   And execute:
   ```bash
   make setup
   ```
   This command automatically sets up the project environment. Specifically, it creates a Python virtual environment, installs required Python packages, and compiles Rust components.

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
This project is licensed under the MIT License - see the [LICENSE](../LICENSE.md) file for details.
