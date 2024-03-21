
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
- **Rust**: Model's core is written in Rust. [Install Rust](https://www.rust-lang.org/tools/install) by following the official instructions.
- **Docker**: For ease of setup and deployment, this project is containerized with Docker. [Install Docker Desktop](https://docs.docker.com/desktop/) for your operating system.

### Installation

Follow these steps to get your development environment running:

1. **Clone the Repository**:
   Start by cloning the project repository to your local machine.
   ```bash
   git clone https://github.com/phononautomata/threshold.git

2. **Run the Makefile**:
   From your terminal, navigate to the project directory:
   ```cd threshold```
   And execute:
   ```make setup```
   This command automatically sets up the project environment. Specifically, it creates a Python virtual environment, installs required Python packages, and compiles Rust components.




## License
This project is licensed under the MIT License - see the `LICENSE.md` file for details.
 
   
  



1. **Project Details**: Can you provide a brief description of your project, including its purpose and key features?
2. **Installation Steps**: Are there specific installation steps or prerequisites, especially given the mix of Python, Rust, and Docker in your project?
3. **Testing**: How are tests run in your project?
4. **Usage**: Are there specific examples of how to use your project that should be highlighted?
5. **Contributing**: Do you have a contributing guide, or are there specific guidelines for contributors?
6. **Versioning and Authors**: How do you handle versioning? Who are the main authors and contributors?
7. **Acknowledgments**: Are there any acknowledgments or credits you'd like to include?

Feel free to answer any of these questions or provide additional details, and we can iterate on crafting a README that best represents your project.
