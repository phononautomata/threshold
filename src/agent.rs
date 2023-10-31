use netrust::network::Network;
use rand::prelude::*;
use serde::Serialize;
use strum::Display;

use crate::utils::{Input, VaccinationPars};

#[derive(Serialize, PartialEq, Eq, Clone, Copy, Debug)]
pub enum Health {
    Infected,
    Removed,
    Susceptible,
    Vaccinated,
}

#[derive(Serialize, PartialEq, Eq, Clone, Copy, Debug)]
pub enum Opinion {
    Active,
    Hesitant,
}

#[derive(Serialize, PartialEq, Eq, Clone, Copy, Debug, Display)]
pub enum Status {
    ActInf,
    ActRem,
    ActSus,
    ActVac,
    HesInf,
    HesRem,
    HesSus,
    HesVac,
}

#[derive(Serialize, PartialEq, Eq, Clone, Copy, Debug, Display, clap::ValueEnum)]
pub enum SeedModel {
    BottomDegreeNeighborhood,
    BottomMultiLocus,
    RandomMultiLocus,
    RandomNeighborhood,
    TopDegreeMultiLocus,
    TopDegreeNeighborhood,
}

#[derive(Serialize, Clone)]
pub struct Agent {
    pub degree: usize,
    pub health: Health,
    pub id: usize,
    pub opinion: Opinion,
    pub status: Status,
    pub threshold: f64,
    pub convinced_when: Option<usize>,
    pub final_active_susceptible: Option<usize>,
    pub final_prevalence: Option<usize>,
    pub final_vaccinated: Option<usize>,
    pub infected_by: Option<usize>,
    pub infected_when: Option<usize>,
    pub initial_active_susceptible: Option<usize>,
    pub initial_vaccinated: Option<usize>,
    pub neighbors: Option<Vec<usize>>,
    pub removed_when: Option<usize>,
    pub vaccinated_when: Option<usize>,
    pub zealots: Option<usize>,
}

impl Agent {
    pub fn new(agent_id: usize) -> Self {
        Self {
            degree: 0,
            health: Health::Susceptible,
            id: agent_id,
            status: Status::HesSus,
            opinion: Opinion::Hesitant,
            threshold: 0.0,
            final_active_susceptible: None,
            final_prevalence: None,
            final_vaccinated: None,
            infected_by: None,
            infected_when: None,
            convinced_when: None,
            initial_active_susceptible: None,
            initial_vaccinated: None,
            neighbors: None,
            removed_when: None,
            vaccinated_when: None,
            zealots: None,
        }
    }

    pub fn get_neighbors(&mut self, graph: &Network) {
        self.neighbors = Some(graph.inner()[self.id].neighbors.clone());
        self.degree = self.neighbors.as_ref().unwrap().len();
    }

    pub fn number_of_neighbors(&self) -> usize {
        self.neighbors.as_ref().unwrap().len()
    }

    pub fn set_threshold(&mut self, theta: f64) {
        self.threshold = theta
    }

    pub fn get_vaccinated_neighbor_fraction(
        &mut self, 
        neighbors: &Vec<Agent>
    ) -> f64 {
        let k = self.number_of_neighbors();
        let mut vaccinated_neighbors = 0;
        for neighbor_agent in neighbors {
            if neighbor_agent.status == Status::ActVac {
                vaccinated_neighbors += 1;
            }
        }
        vaccinated_neighbors as f64 / k as f64
    }

    pub fn watts_threshold_step(&mut self, neighbors: &Vec<Agent>) -> bool {

        let vaccinated_fraction = 
        self.get_vaccinated_neighbor_fraction(neighbors);

        if vaccinated_fraction >= self.threshold {
            match self.status {
                Status::HesSus => {
                    self.status = Status::ActSus;
                    true
                },
                Status::HesInf => {
                    self.status = Status::ActInf;
                    true
                },
                Status::HesRem => {
                    self.status = Status::ActRem;
                    true
                },
                _ => false,
            } 
        } else {
            false
        }
    }

    pub fn watts_threshold_checked_step(
        &mut self, 
        agent_ensemble: &AgentEnsemble
    ) {
        if self.opinion == Opinion::Hesitant {
            let mut vaccinated_neighbors = 0;
            let k = self.number_of_neighbors();
            for neighbor in self.neighbors.as_ref().unwrap() {
                let neighbor_agent = &agent_ensemble.inner()[*neighbor];
                if neighbor_agent.health == Health::Vaccinated {
                    vaccinated_neighbors += 1;
                }
            }
            let vaccinated_fraction = vaccinated_neighbors as f64 / k as f64;
            if vaccinated_fraction >= self.threshold {
                self.opinion = Opinion::Active;
            }
        }
    }

    pub fn vaccination_step(&mut self, pars: &Input) -> bool {
        let mut rng = rand::thread_rng();
        let trial: f64 = rng.gen();
        if trial < pars.epidemic.vaccination_rate {
            self.status = Status::ActVac;
            true
        } else {
            false
        }
    }

    pub fn vaccination_checked_step(&mut self, pars: &Input) {
        if self.health == Health::Susceptible && self.opinion == Opinion::Active {
            let mut rng = rand::thread_rng();
            let trial: f64 = rng.gen();
            if trial < pars.epidemic.vaccination_rate {
                self.health = Health::Vaccinated;
            }
        }
    }

    pub fn infection_step(
        &mut self, 
        agent_ensemble: & AgentEnsemble, 
        pars: &Input
    ) -> (Vec<usize>, Vec<usize>) {
        let mut hes_inf_a = Vec::new();
        let mut act_inf_a = Vec::new();
        let mut rng = rand::thread_rng();
        
        for neighbor in self.neighbors.as_ref().unwrap() {
            let neighbor_agent = &agent_ensemble.inner()[*neighbor];
            if neighbor_agent.status == Status::HesSus {
                let trial: f64 = rng.gen();
                if trial < pars.epidemic.infection_rate {
                    hes_inf_a.push(neighbor_agent.id);
                }
            } else if neighbor_agent.status == Status::ActSus {
                let trial: f64 = rng.gen();
                if trial < pars.epidemic.infection_rate {
                    act_inf_a.push(neighbor_agent.id);
                }
            }
        }
        
        (hes_inf_a, act_inf_a)
    }
    
    pub fn infection_checked_step(
        &mut self, 
        agent_ensemble: &AgentEnsemble, 
        pars: &Input
    ) -> Vec<usize> {
        let mut infected_a = Vec::new();
        if self.health == Health::Infected {
            let mut rng = rand::thread_rng();
            for neighbor in self.neighbors.as_ref().unwrap() {
                let neighbor_agent = &agent_ensemble.inner()[*neighbor];
                if neighbor_agent.health == Health::Susceptible {
                    let trial: f64 = rng.gen();
                    if trial < pars.epidemic.infection_rate {
                        infected_a.push(neighbor_agent.id);
                    }
                }
            }
        }
        infected_a
    }

    pub fn removal_step(&mut self, pars: &Input) -> bool {
        let mut rng = rand::thread_rng();
        let trial: f64 = rng.gen();
        if trial < pars.epidemic.infection_decay {
            if self.status == Status::HesInf {
                self.status = Status::HesRem;
            } else if self.status == Status::ActInf {
                self.status = Status::ActRem;
            }
            true
        } else {
            false
        }
    }

    pub fn removal_checked_step(&mut self, pars: &Input) {
        if self.health == Health::Infected {
            let mut rng = rand::thread_rng();
            let trial: f64 = rng.gen();
            if trial < pars.epidemic.infection_decay {
                self.health = Health::Removed;
            }
        }
    }

    pub fn measure_neighborhood(&mut self, agent_ensemble: &mut AgentEnsemble, t: usize) {
        let neighbors = self.neighbors.as_ref().unwrap();
        let mut active_susceptible = 0;
        let mut vaccinated = 0;
        let mut zealots = 0;
        let mut prevalence = 0;

        for neigh in neighbors {
            let status = agent_ensemble.inner()[*neigh].status;
            let threshold = agent_ensemble.inner()[*neigh].threshold;
            if threshold >= 1.0 {
                zealots += 1;
            } else {
                if status == Status::ActSus {
                    active_susceptible += 1;
                } else if status == Status::ActVac {
                    vaccinated += 1;
                } else if status == Status::ActRem {
                    prevalence += 1;
                } else if status == Status::HesRem {
                    prevalence += 1;
                }
            }
        }

        if t == 0 {
            self.initial_active_susceptible = Some(active_susceptible);
            self.initial_vaccinated = Some(vaccinated);
            self.zealots = Some(zealots);
        } else {
            self.final_active_susceptible = Some(active_susceptible);
            self.final_vaccinated = Some(vaccinated);
            self.zealots = Some(zealots);
            self.final_prevalence = Some(prevalence);
        }
    }
}

pub struct AgentEnsemble {
    inner: Vec<Agent>,
}

impl AgentEnsemble {
    pub fn new(number_of_agents: usize) -> Self {
        let mut agent_ensemble = AgentEnsemble { inner: Vec::new() };
        for (count, _) in (0..number_of_agents).enumerate() {
            let agent = Agent::new(count);
            agent_ensemble.inner.push(agent);
        }
        agent_ensemble
    }

    pub fn inner(&self) -> &Vec<Agent> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Vec<Agent> {
        &mut self.inner
    }

    pub fn into_inner(self) -> Vec<Agent> {
        self.inner
    }

    pub fn get_neighbors(&mut self, graph: &Network) {
        for agent in self.inner_mut() {
            agent.get_neighbors(graph);
        }
    }

    pub fn find_bottom_degree_agent(&self) -> Option<usize> {
        let mut min_degree = 0;
        let mut bottom_idx = None;

        for (idx, agent) in self.inner().iter().enumerate() {
            let degree = agent.degree;
            if degree > min_degree {
                min_degree = degree;
                bottom_idx = Some(idx);
            }
        }
        bottom_idx
    }

    pub fn find_top_degree_agent(&self) -> Option<usize> {
        let mut max_degree = 0;
        let mut top_idx = None;

        for (idx, agent) in self.inner().iter().enumerate() {
            let degree = agent.degree;
            if degree > max_degree {
                max_degree = degree;
                top_idx = Some(idx);
            }
        }
        top_idx
    }

    pub fn get_subensemble(&self, agents: &Vec<usize>) -> Vec<Agent> {
        let mut agent_subensemble = Vec::new();
        for id in agents {
            let agent = self.inner()[*id].clone();
            agent_subensemble.push(agent);
        }
        agent_subensemble
    }

    pub fn introduce_infections(&mut self, model: SeedModel, nseeds: usize) {
        match model {
            SeedModel::BottomDegreeNeighborhood => {
                let hub = self.find_bottom_degree_agent().unwrap();
                let hub_neighs = self.inner()[hub].neighbors.as_ref().unwrap().clone();
                let mut exhausted_seeds = nseeds - 1;

                for idx in hub_neighs {
                    if exhausted_seeds == 0 {
                        break; // Stop infecting if no more seeds available
                    }

                    match self.inner_mut()[idx].status {
                        Status::HesSus | Status::ActSus => {
                            self.inner_mut()[idx].status = match self.inner_mut()[idx].status {
                                Status::HesSus => Status::HesInf,
                                Status::ActSus => Status::ActInf,
                                _ => unreachable!(), // This line will never be reached
                            };
                            self.inner_mut()[idx].health = Health::Infected;
                            exhausted_seeds -= 1;
                        }
                        _ => (),
                    }
                }
                let effective_infected = nseeds - exhausted_seeds;
                println!("Effective infected individuals = {effective_infected}");
            },
            SeedModel::BottomMultiLocus => {
                todo!()
            },
            SeedModel::RandomMultiLocus => {
                let mut rng = rand::thread_rng();
                let nagents = self.number_of_agents();
                let agent_indices: Vec<usize> = (0..nagents).collect();
                let selected_indices = 
                agent_indices.choose_multiple(&mut rng, nseeds);

                for idx in selected_indices {
                    match self.inner_mut()[*idx].status {
                        Status::HesSus => {
                            self.inner_mut()[*idx].status = Status::HesInf;
                            self.inner_mut()[*idx].health = Health::Infected;
                        }
                        Status::ActSus =>  {
                            self.inner_mut()[*idx].status = Status::ActInf;
                            self.inner_mut()[*idx].health = Health::Infected;
                        }
                        _ => (),
                    }
                }
            },
            SeedModel::RandomNeighborhood => {
                let mut rng = rand::thread_rng();
                let nagents = self.number_of_agents();
                let zero_patient = rng.gen_range(0..nagents);
                let zp_neighs = self.inner()[zero_patient].neighbors.as_ref().unwrap().clone();
                let mut exhausted_seeds = nseeds - 1;

                for idx in zp_neighs {
                    if exhausted_seeds == 0 {
                        break; // Stop infecting if no more seeds available
                    }

                    match self.inner_mut()[idx].status {
                        Status::HesSus | Status::ActSus => {
                            self.inner_mut()[idx].status = match self.inner_mut()[idx].status {
                                Status::HesSus => Status::HesInf,
                                Status::ActSus => Status::ActInf,
                                _ => unreachable!(), // This line will never be reached
                            };
                            self.inner_mut()[idx].health = Health::Infected;
                            exhausted_seeds -= 1;
                        }
                        _ => (),
                    }
                }
                let effective_infected = nseeds - exhausted_seeds;
                println!("Effective infected individuals = {effective_infected}");
            },
            SeedModel::TopDegreeMultiLocus => {
                todo!()
            },
            SeedModel::TopDegreeNeighborhood => {
                let hub = self.find_top_degree_agent().unwrap();
                let hub_neighs = self.inner()[hub].neighbors.as_ref().unwrap().clone();
                let mut exhausted_seeds = nseeds - 1;

                for idx in hub_neighs {
                    if exhausted_seeds == 0 {
                        break; // Stop infecting if no more seeds available
                    }

                    match self.inner_mut()[idx].status {
                        Status::HesSus | Status::ActSus => {
                            self.inner_mut()[idx].status = match self.inner_mut()[idx].status {
                                Status::HesSus => Status::HesInf,
                                Status::ActSus => Status::ActInf,
                                _ => unreachable!(), // This line will never be reached
                            };
                            self.inner_mut()[idx].health = Health::Infected;
                            exhausted_seeds -= 1;
                        }
                        _ => (),
                    }
                }
                let effective_infected = nseeds - exhausted_seeds;
                println!("Effective infected individuals = {effective_infected}");
            },
        }
    }

    pub fn introduce_infections_dd(&mut self, nseeds: usize) {
        let mut rng = rand::thread_rng();
        let nagents = self.number_of_agents();
        let agent_indices: Vec<usize> = (0..nagents).collect();
        let selected_indices = 
        agent_indices.choose_multiple(&mut rng, nseeds);
        
        for idx in selected_indices {
            self.inner_mut()[*idx].status = Status::HesInf;
        }
    }

    pub fn introduce_infections_and_vaccines(
        &mut self, 
        nseeds: usize, 
        nvaxx: usize
    ) {
        let mut rng = rand::thread_rng();
        let nagents = self.number_of_agents();
        let agent_indices: Vec<usize> = (0..nagents).collect();
        let selected_indices = 
        agent_indices.choose_multiple(&mut rng, nseeds);
        
        for idx in selected_indices {
            match self.inner_mut()[*idx].status {
                Status::HesSus => 
                self.inner_mut()[*idx].status = Status::HesInf,
                Status::ActSus => 
                self.inner_mut()[*idx].status = Status::ActInf,
                _ => (),
            }
        }

        let mut rng = rand::thread_rng();
        let agent_indices: Vec<usize> = (0..self.number_of_agents()).collect();
        let selected_indices = 
        agent_indices.choose_multiple(&mut rng, nseeds + nvaxx);
        
        for (count, &idx) in selected_indices.enumerate() {
            if count < nseeds {
                match self.inner_mut()[idx].status {
                    Status::HesSus => 
                    self.inner_mut()[idx].status = Status::HesInf,
                    Status::ActSus => 
                    self.inner_mut()[idx].status = Status::ActInf,
                    _ => (),
                }
            }
            else {
                self.inner_mut()[idx].status = Status::ActVac;
            }
        }         
    }

    pub fn reintroduce_infections(&mut self, nseeds: usize) {
        let mut rng = rand::thread_rng();
        let nagents = self.number_of_agents();
        let agent_indices: Vec<usize> = (0..nagents).collect();
        let mut selected_indices: Vec<usize> = vec![];
        
        while selected_indices.len() < nseeds {
            let idx = agent_indices.choose(&mut rng).unwrap();
            if let Status::HesSus | Status::ActSus = self.inner_mut()[*idx].status {
                self.inner_mut()[*idx].status = match self.inner_mut()[*idx].status {
                    Status::HesSus => Status::HesInf,
                    Status::ActSus => Status::ActInf,
                    _ => unreachable!(),
                };
                selected_indices.push(*idx);
            }
        }
    }

    pub fn set_opinion_threshold(&mut self, threshold: f64) {
        for agent in self.inner_mut() {
            agent.threshold = threshold;
        }
    }

    pub fn introduce_vaccination_thresholds(
        &mut self, 
        vpars: &VaccinationPars
    ) {
        let nagents = self.number_of_agents();
        let mut indices: Vec<usize> = (0..nagents).collect();
        indices.shuffle(&mut rand::thread_rng());

        let already_count = (vpars.already * nagents as f64) as i32;
        let soon_count = (vpars.soon * nagents as f64) as i32;
        let someone_count = (vpars.someone * nagents as f64) as i32;
        let majority_count = (vpars.majority * nagents as f64) as i32;
        let never_count = (vpars.never * nagents as f64) as i32;
        
        let mut already_assigned = 0;
        let mut soon_assigned = 0;
        let mut someone_assigned = 0;
        let mut majority_assigned = 0;
        let mut never_assigned = 0;
        
        for i in indices {
            if already_assigned < already_count {
                self.inner_mut()[i].status = Status::ActVac;
                self.inner_mut()[i].threshold = 0.0;
                already_assigned += 1;
            } else if soon_assigned < soon_count {
                self.inner_mut()[i].status = Status::ActSus;
                self.inner_mut()[i].threshold = 0.0;
                soon_assigned += 1;
            } else if someone_assigned < someone_count {
                self.inner_mut()[i].status = Status::HesSus;
                let num_neighbors = self.inner()[i].number_of_neighbors();
                self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                someone_assigned += 1;
            } else if majority_assigned < majority_count {
                self.inner_mut()[i].status = Status::HesSus;
                self.inner_mut()[i].threshold = 0.5;
                majority_assigned += 1;
            } else if never_assigned < never_count {
                self.inner_mut()[i].status = Status::HesSus;
                self.inner_mut()[i].threshold = 1.1;
                never_assigned += 1;
            }
        }
    }

    pub fn introduce_opinions(
        &mut self, 
        active_fraction: f64, 
        zealot_fraction: f64
    ) {
        let total_fraction = active_fraction + zealot_fraction;

        // Check if total fraction exceeds 1
        if total_fraction > 1.0 {
            // Adjust zealot_fraction to make total_fraction equal to 1
            let adjusted_zealot_fraction = 1.0 - active_fraction;
            println!("Total fraction is higher than 1! Readjusting zealot fraction...");

            let mut rng = rand::thread_rng();
            let n_active = (active_fraction * self.number_of_agents() as f64) as usize;
            let n_zealot = (adjusted_zealot_fraction * self.number_of_agents() as f64) as usize;
            let agent_indices: Vec<usize> = (0..self.number_of_agents()).collect();
            let selected_indices = 
            agent_indices.choose_multiple(&mut rng, n_active + n_zealot);
        
            for (count, &idx) in selected_indices.enumerate() {
                if count < n_active {
                    match self.inner_mut()[idx].status {
                        Status::HesSus => {
                            self.inner_mut()[idx].status = Status::ActSus;
                            self.inner_mut()[idx].opinion = Opinion::Active;
                        },
                        Status::HesInf => {
                            self.inner_mut()[idx].status = Status::ActInf;
                            self.inner_mut()[idx].opinion = Opinion::Active;
                        },
                        Status::HesRem => {
                            self.inner_mut()[idx].status = Status::ActRem;
                            self.inner_mut()[idx].opinion = Opinion::Active;
                        }
                        _ => (),
                    }
                }
                else {
                    self.inner_mut()[idx].threshold = 1.1;
                }
            }
            
        } else {
            let mut rng = rand::thread_rng();
            let n_active = (active_fraction * self.number_of_agents() as f64) as usize;
            let n_zealot = (zealot_fraction * self.number_of_agents() as f64) as usize;
            let agent_indices: Vec<usize> = (0..self.number_of_agents()).collect();
            let selected_indices = 
            agent_indices.choose_multiple(&mut rng, n_active + n_zealot);
            
            for (count, &idx) in selected_indices.enumerate() {
                if count < n_active {
                    match self.inner_mut()[idx].status {
                        Status::HesSus => {
                            self.inner_mut()[idx].status = Status::ActSus;
                            self.inner_mut()[idx].opinion = Opinion::Active;
                        },
                        Status::HesInf => {
                            self.inner_mut()[idx].status = Status::ActInf;
                            self.inner_mut()[idx].opinion = Opinion::Active;
                        },
                        Status::HesRem => {
                            self.inner_mut()[idx].status = Status::ActRem;
                            self.inner_mut()[idx].opinion = Opinion::Active;
                        }
                        _ => (),
                    }
                }
                else {
                    self.inner_mut()[idx].threshold = 1.1;
                }
            }
        }
    }
    
    pub fn number_of_agents(&self) -> usize {
        self.inner.len()
    }

    pub fn total_hesitant(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.opinion == Opinion::Hesitant {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_active(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.opinion == Opinion::Active {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_zealot(&self) -> usize {
        let mut summa = 0;
        for agent in self.inner()  {
            if agent.threshold > 1.0 {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_susceptible(&self) -> usize {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.health == Health::Susceptible {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_infected(&self) -> usize {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.health == Health::Infected {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_removed(&self) -> usize {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.health == Health::Removed {
                summa += 1;
            }
        }
        summa
    }

    pub fn total_vaccinated(&self) -> usize {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.health == Health::Vaccinated {
                summa += 1;
            }
        }
        summa
    }

    pub fn gather_hesitant(&self) -> Vec<usize> {
        let mut hesitant_a = Vec::new();
        for agent in self.inner() {
            if agent.opinion == Opinion::Hesitant {
                hesitant_a.push(agent.id);
            }
        }
        hesitant_a
    }

    pub fn gather_active(&self) -> Vec<usize> {
        let mut active_a = Vec::new();
        for agent in self.inner() {
            if agent.opinion == Opinion::Active {
                active_a.push(agent.id);
            }
        }
        active_a
    }

    pub fn gather_susceptible(&self) -> Vec<usize> {
        let mut susceptible_a = Vec::new();
        for agent in self.inner() {
            if agent.health == Health::Susceptible {
                susceptible_a.push(agent.id);
            }
        }
        susceptible_a
    }

    pub fn gather_infected(&self) -> Vec<usize> {
        let mut infected_a = Vec::new();
        for agent in self.inner() {
            if agent.health == Health::Infected {
                infected_a.push(agent.id);
            }
        }
        infected_a
    }

    pub fn gather_removed(&self) -> Vec<usize> {
        let mut removed_a = Vec::new();
        for agent in self.inner() {
            if agent.health == Health::Removed {
                removed_a.push(agent.id);
            }
        }
        removed_a
    }

    pub fn gather_vaccinated(&self) -> Vec<usize> {
        let mut vaccinated_a = Vec::new();
        for agent in self.inner() {
            if agent.health == Health::Vaccinated {
                vaccinated_a.push(agent.id);
            }
        }
        vaccinated_a
    }

    pub fn gather_hesitant_susceptible(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::HesSus)
            .map(|agent| agent.id)
            .collect()
    }

    pub fn gather_active_susceptible(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::ActSus)
            .map(|agent| agent.id)
            .collect()
    }
    
    pub fn gather_hesitant_infected(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::HesInf)
            .map(|agent| agent.id)
            .collect()
    }
    
    pub fn gather_active_infected(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::ActInf)
            .map(|agent| agent.id)
            .collect()
    }
    
    pub fn gather_hesitant_removed(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::HesRem)
            .map(|agent| agent.id)
            .collect()
    }
    
    pub fn gather_active_removed(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::ActRem)
            .map(|agent| agent.id)
            .collect()
    }

    pub fn gather_hesitant_vaccinated(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::HesVac)
            .map(|agent| agent.id)
            .collect()
    }
    
    pub fn gather_active_vaccinated(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::ActVac)
            .map(|agent| agent.id)
            .collect()
    }
    
    pub fn gather_hesitant_active_links(&self) -> Vec<(usize, usize)> {
        todo!()
    }

    pub fn gather_susceptible_infected_links(&self) -> Vec<(usize, usize)> {
        todo!()
    }

    pub fn build_status_array(&self) -> Vec<Status> {
        self.inner().iter().map(|agent| agent.status).collect()
    }

    pub fn update_list(
        &self, 
        list_to_update: &mut [usize], 
        status: Status
    ) -> Vec<usize> {
        let mut new_list = Vec::new();
        match status {
            Status::HesSus => {
                for a in list_to_update.iter() {
                    let agent_id = *a;
                    if self.inner()[agent_id].status == Status::HesSus {
                        new_list.push(agent_id);
                    }
                }
            },
            Status::ActSus => {
                for a in list_to_update.iter() {
                    let agent_id = *a;
                    if self.inner()[agent_id].status == Status::ActSus {
                        new_list.push(agent_id);
                    }
                }
            },
            Status::HesInf => {
                for a in list_to_update.iter() {
                    let agent_id = *a;
                    if self.inner()[agent_id].status == Status::HesInf {
                        new_list.push(agent_id);
                    }
                }
            },
            Status::ActInf => {
                for a in list_to_update.iter() {
                    let agent_id = *a;
                    if self.inner()[agent_id].status == Status::ActInf {
                        new_list.push(agent_id);
                    }
                }
            },
            _ => {
                /* do nothing */
            }
        }
        new_list
    }
}
