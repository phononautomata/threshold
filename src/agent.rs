use std::collections::{HashMap, HashSet};

use rand::Rng;
use rand::{distributions::WeightedIndex, prelude::*};
use rv::traits::Rv;
use rv::dist::NegBinomial;
use serde::{Serialize, Deserialize};
use strum::Display;

use crate::cons::{CONST_ELDER_THRESHOLD, CONST_MIDDLEAGE_THRESHOLD, CONST_UNDERAGE_THRESHOLD};
use crate::{
    utils::{VaccinationPars, sample_from_cdf}, 
    cons::{
        FLAG_VERBOSE, 
        CONST_ZEALOT_THRESHOLD, 
        CONST_ALREADY_THRESHOLD, 
        CONST_SOON_THRESHOLD, 
        CONST_MAJORITY_THRESHOLD,
    }
};

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
pub enum Attitude {
    Vaccinated,
    Soon,
    Someone,
    Most,
    Never,
}

#[derive(Clone, Copy, Serialize, Deserialize, Display, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum HesitancyAttributionModel {
    #[serde(rename = "adult")]
    Adult,
    #[serde(rename = "elder")]
    Elder,
    #[serde(rename = "elder-to-young")]
    ElderToYoung,
    #[serde(rename = "data-driven")]
    DataDriven,
    #[serde(rename = "middle-age")]
    Middleage,
    #[serde(rename = "random")]
    Random,
    #[serde(rename = "underage")]
    Underage,
    #[serde(rename = "young")]
    Young,
    #[serde(rename = "young-to-elder")]
    YoungToElder,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
pub enum Health {
    Infected,
    Removed,
    Susceptible,
    Vaccinated,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug)]
pub enum Opinion {
    Active,
    Hesitant,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug, Display, clap::ValueEnum)]
pub enum OpinionModel {
    ElderCare,
    DataDrivenThresholds,
    HomogeneousThresholds,
    HomogeneousWithZealots,
    Majority,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug, Display, clap::ValueEnum)]
pub enum SeedModel {
    #[serde(rename = "bottom-degree-neighborhood")]
    BottomDegreeNeighborhood,
    #[serde(rename = "bottom-multi-locus")]
    BottomMultiLocus,
    #[serde(rename = "random-multi-locus")]
    RandomMultiLocus,
    #[serde(rename = "random-neighborhood")]
    RandomNeighborhood,
    #[serde(rename = "top-degree-multi-locus")]
    TopDegreeMultiLocus,
    #[serde(rename = "top-degree-neighborhood")]
    TopDegreeNeighborhood,
}

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug, Display)]
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

#[derive(Serialize, Deserialize, PartialEq, Eq, Clone, Copy, Debug, Display, clap::ValueEnum)]
pub enum VaccinationPolicy {
    #[serde(rename = "age-adult")]
    AgeAdult,
    #[serde(rename = "age-elder")]
    AgeElder,
    #[serde(rename = "age-elder-to-young")]
    AgeTop,
    #[serde(rename = "age-middle-age")]
    AgeMiddleage,
    #[serde(rename = "age-underage")]
    AgeUnderage,
    #[serde(rename = "age-young")]
    AgeYoung,
    #[serde(rename = "young-to-elder")]
    AgeYoungToElder,
    #[serde(rename = "automatic")]
    Automatic,
    #[serde(rename = "combo-elder-hub")]
    ComboElderTop,
    #[serde(rename = "combo-young-hub")]
    ComboYoungTop,
    #[serde(rename = "data-driven")]
    DataDriven,
    #[serde(rename = "degree-bottom")]
    DegreeBottom,
    #[serde(rename = "degree-top")]
    DegreeTop,
    #[serde(rename = "degree-random")]
    DegreeRandom,
    #[serde(rename = "random")]
    Random,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct Agent {
    pub age: usize,
    pub degree: usize,
    pub health: Health,
    pub id: usize,
    pub neighbors: Vec<usize>,
    pub opinion: Opinion,
    pub status: Status,
    pub threshold: f64,
    pub vaccination_target: bool,
    pub activation_potential: Option<usize>,
    pub attitude: Option<Attitude>,
    pub cascading_threshold: Option<usize>,
    pub convinced_when: Option<usize>,
    pub effective_threshold: Option<f64>,
    pub final_active_susceptible: Option<usize>,
    pub final_prevalence: Option<usize>,
    pub final_vaccinated: Option<usize>,
    pub infected_by: Option<usize>,
    pub infected_when: Option<usize>,
    pub initial_active_susceptible: Option<usize>,
    pub initial_vaccinated: Option<usize>,
    pub removed_when: Option<usize>,
    pub vaccinated_when: Option<usize>,
    pub zealots: Option<usize>,
}

#[allow(dead_code)]
impl Agent {
    pub fn new(agent_id: usize) -> Self {
        Self {
            age: 0,
            degree: 0,
            health: Health::Susceptible,
            id: agent_id,
            neighbors: Vec::new(),
            opinion: Opinion::Hesitant,
            status: Status::HesSus,
            threshold: 0.0,
            vaccination_target: true,
            activation_potential: Some(1),
            attitude: Some(Attitude::Never),
            cascading_threshold: Some(1),
            effective_threshold: Some(1.0),
            final_active_susceptible: None,
            final_prevalence: None,
            final_vaccinated: None,
            infected_by: None,
            infected_when: None,
            convinced_when: None,
            initial_active_susceptible: None,
            initial_vaccinated: None,
            removed_when: None,
            vaccinated_when: None,
            zealots: None,
        }
    }

    fn measure_activation_potential(
        &mut self, 
        agent_ensemble: &mut AgentEnsemble,
    ) {
        let neighbors = self.neighbors.clone();
        let threshold = self.threshold;
        let mut cascading_count = 0;
        let degree = self.degree;

        for neigh in neighbors {
            let neigh_threshold = agent_ensemble.inner()[neigh].threshold;
            if neigh_threshold < threshold {
                cascading_count += 1;
            }
        }

        if (cascading_count / degree) as f64 >= threshold {
            self.activation_potential = Some(1);
        } else {
            self.activation_potential = Some(0);
        }
    }

    fn measure_average_neighbor_threshold(
        &mut self, 
        agent_ensemble: &mut AgentEnsemble,
    ) -> f64 {
        let neighbors = self.neighbors.clone();
        let mut cumulative_neighbor_threshold = 0.0;
        let degree = self.degree;

        for neigh in neighbors {
            let neigh_threshold = agent_ensemble.inner()[neigh].threshold;
            cumulative_neighbor_threshold += neigh_threshold;
        }

        let average_neighbor_threshold = cumulative_neighbor_threshold / (degree as f64);
        average_neighbor_threshold
    }

    pub fn measure_cascading_threshold(
        &mut self, 
        agent_ensemble: &mut AgentEnsemble,
    ) -> usize {
        let neighbors = self.neighbors.clone();
        let threshold = self.threshold;
        let mut cumulative_neighbor_threshold = 0.0;
        let degree = self.degree;

        for neigh in neighbors {
            let neigh_threshold = agent_ensemble.inner()[neigh].threshold;
            cumulative_neighbor_threshold += neigh_threshold;
        }
        let average_neighbor_threshold = cumulative_neighbor_threshold / (degree as f64);

        match self.attitude.unwrap() {
            Attitude::Vaccinated => {
                0
            },
            Attitude::Soon => {
                0
            },
            Attitude::Someone => {
                if threshold >= average_neighbor_threshold {
                    0
                } else {
                    1
                }
            },
            Attitude::Most => {
                if threshold >= average_neighbor_threshold {
                    0
                } else {
                    1
                }
            },
            Attitude::Never => {
                1
            },
        }
    }

    fn measure_neighborhood(
        &mut self, 
        agent_ensemble: &mut AgentEnsemble, 
        t: usize,
    ) {
        let neighbors = self.neighbors.clone();
        let threshold = self.threshold;
        let mut active_susceptible = 0;
        let mut cascading_count = 0;
        let degree = self.degree;
        let mut vaccinated = 0;
        let mut zealots = 0;
        let mut prevalence = 0;

        for neigh in neighbors {
            let status = agent_ensemble.inner()[neigh].status;
            let neigh_threshold = agent_ensemble.inner()[neigh].threshold;

            if neigh_threshold < threshold {
                cascading_count += 1;
            }

            if neigh_threshold >= 1.0 {
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

        if (cascading_count / degree) as f64 >= threshold {
            self.activation_potential = Some(1);
        } else {
            self.activation_potential = Some(0);
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

    fn number_of_neighbors(&self) -> usize {
        self.neighbors.len()
    }

    fn set_threshold(&mut self, theta: f64) {
        self.threshold = theta
    }

    fn vaccinated_neighbor_fraction(
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

    fn sample_age(&mut self, cdf: &Vec<f64>) {
        self.age = sample_from_cdf(cdf)
    }

    fn sample_degree(&mut self, cdf: &Vec<f64>) {
        self.degree = sample_from_cdf(cdf)
    }

    fn sample_degree_from_negative_binomial(
        &mut self, 
        mean_value: f64, 
        _standard_deviation: f64,
    ) {
        let variance = mean_value + 13.0;
        let r = mean_value.powi(2) / (variance - mean_value);
        let p = r / (r + mean_value);

        let mut rng = rand::thread_rng();
        let neg_binom = NegBinomial::new(r, p).unwrap();

        let degree = Rv::<u16>::draw(&neg_binom, &mut rng) as usize; 
        self.degree = degree; 
    }
}

pub struct AgentEnsemble {
    inner: Vec<Agent>,
}

#[allow(dead_code)]
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

    fn into_inner(self) -> Vec<Agent> {
        self.inner
    }

    pub fn arrange_nodes_to_age_layers(
        &self, 
        age_groups: usize,
    ) -> Vec<Vec<usize>> {
        let mut nodes_to_layers = vec![Vec::new(); age_groups];

        for (_, agent) in self.inner().iter().enumerate() {
            let id = agent.id;
            let age = agent.age;
            nodes_to_layers[age].push(id);
        }

        nodes_to_layers
    }

    pub fn average_degree(&self) -> f64 {
        let mut average_degree = 0.0;

        for agent in self.inner() {
            let degree = agent.number_of_neighbors() as f64;
            average_degree += degree;
        }

        average_degree / self.number_of_agents() as f64
    }

    fn build_status_array(&self) -> Vec<Status> {
        self.inner().iter().map(|agent| agent.status).collect()
    }

    pub fn cascading_threshold(&mut self) -> Vec<usize> {
        let nagents = self.number_of_agents();
        let mut cascading_threshold_vec = vec![0; nagents];

        let attitudes: Vec<Attitude> = self.inner().iter().filter_map(|agent| agent.attitude).collect();
    
        for (agent_idx, agent) in self.inner_mut().iter_mut().enumerate() {
            let attitude = agent.attitude.unwrap();
            let degree = agent.degree;

            let cascading_threshold = match attitude {
                Attitude::Vaccinated => {1},
                Attitude::Soon => {1},
                Attitude::Someone => {
                    let neighbors = agent.neighbors.clone();

                    let mut cascade_assistant = 0;

                    for &neigh_idx in &neighbors {   
                        match attitudes[neigh_idx] {
                            Attitude::Vaccinated => {
                                cascade_assistant += 1
                            },
                            Attitude::Soon => {
                                cascade_assistant += 1
                            },
                            _ => {},        
                        }
                    }

                    if cascade_assistant >= 1 {
                        1
                    } else {
                        0
                    }
                },
                Attitude::Most => {
                    let neighbors = agent.neighbors.clone();

                    let mut cascade_assistant = 0;

                    for &neigh_idx in &neighbors {   
                        match attitudes[neigh_idx] {
                            Attitude::Vaccinated => {
                                cascade_assistant += 1
                            },
                            Attitude::Soon => {
                                cascade_assistant += 1
                            },
                            _ => {},        
                        }
                    }

                    if cascade_assistant as f64 / degree as f64  >= 0.5 {
                        1
                    } else {
                        0
                    }
                },
                Attitude::Never => {0},
            };

            agent.cascading_threshold = Some(cascading_threshold);

            cascading_threshold_vec[agent_idx] = cascading_threshold;
        }

        cascading_threshold_vec
    }

    pub fn clear_epidemic_consequences(&mut self) {
        for agent in self.inner_mut() {
            agent.health = Health::Susceptible;
            agent.opinion = Opinion::Hesitant;
            agent.status = Status::HesSus;
            agent.attitude = Some(Attitude::Never);
            agent.final_active_susceptible = None;
            agent.final_prevalence = None;
            agent.final_vaccinated = None;
            agent.infected_by = None;
            agent.infected_when = None;
            agent.convinced_when = None;
            agent.initial_active_susceptible = None;
            agent.initial_vaccinated = None;
            agent.removed_when = None;
            agent.vaccinated_when = None;
            agent.zealots = None;
        }
    }

    pub fn clone_subensemble(&self, agents: &Vec<usize>) -> Vec<Agent> {
        let mut agent_subensemble = Vec::new();
        for id in agents {
            let agent = self.inner()[*id].clone();
            agent_subensemble.push(agent);
        }
        agent_subensemble
    }

    pub fn collect_intralayer_stubs(
        &self, 
        age_groups: usize,
    ) -> Vec<Vec<usize>> {
        let mut intralayer_stubs = vec![Vec::new(); age_groups];

        for (_, agent) in self.inner().iter().enumerate()  {
            let id = agent.id;
            let age = agent.age;
            let degree = agent.degree;
            for _ in 0..degree  {
                intralayer_stubs[age].push(id)
            }
        }

        intralayer_stubs
    }

    pub fn compute_layer_total_degree(&self, age_groups: usize) -> Vec<f64> {
        let mut total_degree = vec![0.0; age_groups]; 

        for (_, agent) in self.inner().iter().enumerate() {
            let age = agent.age;
            let degree = agent.degree;
            total_degree[age] += degree as f64;
        }
        total_degree
    }

    fn find_bottom_degree_agent(&self) -> Option<usize> {
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

    fn find_top_degree_agent(&self) -> Option<usize> {
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

    pub fn gather_active(&self) -> Vec<usize> {
        let mut active_a = Vec::new();
        for agent in self.inner() {
            if agent.opinion == Opinion::Active {
                active_a.push(agent.id);
            }
        }
        active_a
    }

    pub fn gather_active_infected(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::ActInf)
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

    pub fn gather_active_removed(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::ActRem)
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

    pub fn gather_degree_bottom_to_top(&self) -> Vec<usize> {
        let mut agents_with_degrees: Vec<(usize, usize)> = self.inner().iter()
        .enumerate()
        .map(|(id, agent)| (id, agent.neighbors.len()))
        .collect();

        agents_with_degrees.sort_by(|a, b| a.1.cmp(&b.1));

        // Extract the agent IDs, discarding the degrees
        let sorted_agent_ids: Vec<usize> = agents_with_degrees.into_iter()
            .map(|(id, _)| id)
            .collect();

        sorted_agent_ids
    }

    pub fn gather_degree_top_to_bottom(&self) -> Vec<usize> {
        let mut agents_with_degrees: Vec<(usize, usize)> = self.inner().iter()
        .enumerate()
        .map(|(id, agent)| (id, agent.neighbors.len()))
        .collect();

        // Sort by degree in decreasing order. If you want to sort by increasing order, use `.sort_by_key(|&(_, degree)| degree);`
        agents_with_degrees.sort_by(|a, b| b.1.cmp(&a.1));

        // Extract the agent IDs, discarding the degrees
        let sorted_agent_ids: Vec<usize> = agents_with_degrees.into_iter()
            .map(|(id, _)| id)
            .collect();

        sorted_agent_ids
    }

    pub fn gather_elders(&self) -> Vec<usize> {
        let mut elders_a = Vec::new();
        for agent in self.inner() {
            if agent.age >= CONST_ELDER_THRESHOLD {
                elders_a.push(agent.id);
            }
        }
        elders_a
    }

    pub fn gather_from_elder_to_younger(&self) -> Vec<usize> {
        let mut agents: Vec<_> = self.inner().iter() // Use .iter() to get an iterator over the agents
        .filter(|agent| agent.age >= CONST_UNDERAGE_THRESHOLD)
        .collect::<Vec<&Agent>>(); // Collect into a vector of references to Agent
        
        // Now sort this vector of references by age in descending order
        agents.sort_by(|a, b| b.age.cmp(&a.age));

        // Map the sorted agent references to their IDs and collect into a vector
        agents.iter().map(|agent| agent.id).collect()
    }

    pub fn gather_from_younger_to_elder(&self) -> Vec<usize> {
        let mut agents: Vec<_> = self.inner().iter() // Use .iter() to get an iterator over the agents
        .filter(|agent| agent.age >= CONST_UNDERAGE_THRESHOLD)
        .collect::<Vec<&Agent>>(); // Collect into a vector of references to Agent
        
        // Now sort this vector of references by age in descending order
        agents.sort_by(|a, b| a.age.cmp(&b.age));
        
        // Map the sorted agent references to their IDs and collect into a vector
        agents.iter().map(|agent| agent.id).collect()
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

    pub fn gather_hesitant_infected(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::HesInf)
            .map(|agent| agent.id)
            .collect()
    }

    pub fn gather_hesitant_susceptible(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::HesSus)
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

    pub fn gather_hesitant_vaccinated(&self) -> Vec<usize> {
        self.inner()
            .iter()
            .filter(|agent| agent.status == Status::HesVac)
            .map(|agent| agent.id)
            .collect()
    }

    pub fn gather_middleage(&self) -> Vec<usize> {
        let mut middleage_a = Vec::new();
        for agent in self.inner() {
            if CONST_MIDDLEAGE_THRESHOLD <= agent.age && CONST_ELDER_THRESHOLD > agent.age {
                middleage_a.push(agent.id);
            }
        }
        middleage_a
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

    pub fn gather_susceptible(&self) -> Vec<usize> {
        let mut susceptible_a = Vec::new();
        for agent in self.inner() {
            if agent.health == Health::Susceptible {
                susceptible_a.push(agent.id);
            }
        }
        susceptible_a
    }

    pub fn gather_underage(&self) -> Vec<usize> {
        let mut underage_a = Vec::new();
        for agent in self.inner() {
            if agent.age < CONST_UNDERAGE_THRESHOLD {
                underage_a.push(agent.id);
            }
        }
        underage_a
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

    pub fn gather_young_adults(&self) -> Vec<usize> {
        let mut young_adults_a = Vec::new();
        for agent in self.inner() {
            if agent.age >= CONST_UNDERAGE_THRESHOLD && agent.age < CONST_MIDDLEAGE_THRESHOLD {
                young_adults_a.push(agent.id);
            }
        }
        young_adults_a
    }

    pub fn generate_age_multilayer_network(
        &mut self, 
        layer_probability: &Vec<Vec<f64>>, 
        intralayer_stubs: &mut Vec<Vec<usize>>,
    ) {
        let mut rng = rand::thread_rng();
        let mut connection_pairs = HashSet::new();
        let mut node_connections: HashMap<usize, usize> = HashMap::new();
        let max_attempts = 100;

        for (_, focal_agent) in self.inner().iter().enumerate() {
            //println!("Agent {} connecting to the multilayer...", idx);
            let focal_id = focal_agent.id;
            let focal_layer = focal_agent.age;
            let target_layer_probs = &layer_probability[focal_layer];
            let layer_dist = WeightedIndex::new(target_layer_probs).unwrap();
    
            let mut attempts = 0;
    
            while attempts < max_attempts {
                if !intralayer_stubs[focal_layer].contains(&focal_id) ||
                   *node_connections.get(&focal_id).unwrap_or(&0) >= focal_agent.degree {
                    break; // Focal node has reached its connection capacity or is no longer eligible
                }
    
                let target_layer = layer_dist.sample(&mut rng);
    
                if intralayer_stubs[target_layer].is_empty() {
                    attempts += 1;
                    continue; // Skip if the target layer is empty
                }
    
                let target_index = rng.gen_range(0..intralayer_stubs[target_layer].len());
                let target_id = intralayer_stubs[target_layer][target_index];
    
                if target_id != focal_id && !connection_pairs.contains(&(focal_id, target_id)) && !connection_pairs.contains(&(target_id, focal_id)) {
                    connection_pairs.insert((focal_id, target_id));
    
                    intralayer_stubs[target_layer].remove(target_index);

                    if let Some(focal_index) = intralayer_stubs[focal_layer].iter().position(|&id| id == focal_id) {
                        intralayer_stubs[focal_layer].remove(focal_index);
                    } else {
                        println!("Tried to remove focal_id: {} but it was not found in its layer", focal_id);
                    }

                    // Update connection count for both focal and target nodes
                    *node_connections.entry(focal_id).or_insert(0) += 1;
                    *node_connections.entry(target_id).or_insert(0) += 1;
                } else {
                    attempts += 1;
                }
            }
        }
    
        // Establish connections using IDs, considering each pair as a bidirectional connection.
        for &(focal_id, target_id) in &connection_pairs {
            if let Some(agent) = self.inner_mut().get_mut(focal_id) {
                if !agent.neighbors.contains(&target_id) {
                    agent.neighbors.push(target_id);
                }
            }
            if let Some(agent) = self.inner_mut().get_mut(target_id) {
                if !agent.neighbors.contains(&focal_id) {
                    agent.neighbors.push(focal_id);
                }
            }
        }

        //let mut unequalized_agents = 0;
        // Remove duplicate and self-connections
        //for agent in self.inner_mut() {
        //    agent.neighbors.sort_unstable();
        //    agent.neighbors.dedup();
        //    agent.neighbors.retain(|&x| x != agent.id);
        //    if agent.degree != agent.neighbors.len() {
        //        println!("Agent's degree is {}, whereas agent's neighbor number is {}", agent.degree, agent.neighbors.len());
        //        unequalized_agents += 1;
        //    }
        //}
        //println!("Number of unequalized agents is {}", unequalized_agents);
    }

    pub fn introduce_infections(&mut self, model: SeedModel, nseeds: usize) {
        match model {
            SeedModel::BottomDegreeNeighborhood => {
                let hub = self.find_bottom_degree_agent().unwrap();
                let hub_neighs = self.inner()[hub].neighbors.clone();
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
                if FLAG_VERBOSE {
                    println!("Effective infected individuals = {effective_infected}");
                }
            },
            SeedModel::BottomMultiLocus => {
                if FLAG_VERBOSE {
                    println!("Sorry! Not implemented yet!");
                }
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
                let zp_neighs = self.inner()[zero_patient].neighbors.clone();
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
                if FLAG_VERBOSE {
                    println!("Effective infected individuals = {effective_infected}");
                }
            },
            SeedModel::TopDegreeMultiLocus => {
                if FLAG_VERBOSE {
                    println!("Sorry! Not implemented yet!")
                }
                todo!()
            },
            SeedModel::TopDegreeNeighborhood => {
                let hub = self.find_top_degree_agent().unwrap();
                let hub_neighs = self.inner()[hub].neighbors.clone();
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
                if FLAG_VERBOSE {
                    println!("Effective infected individuals = {effective_infected}");
                }
            },
        }
    }

    fn introduce_infections_and_vaccines(
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

    pub fn introduce_infections_dd(
        &mut self, model: SeedModel, 
        nseeds: usize,
    ) {
        match model {
            SeedModel::BottomDegreeNeighborhood => {
                let hub = self.find_bottom_degree_agent().unwrap();
                let hub_neighs = self.inner()[hub].neighbors.clone();
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
                if FLAG_VERBOSE {
                    println!("Effective infected individuals = {effective_infected}");
                }
            },
            SeedModel::BottomMultiLocus => {
                if FLAG_VERBOSE {
                    println!("Sorry! Not implemented yet!")
                }
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
                let zp_neighs = self.inner()[zero_patient].neighbors.clone();
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
                if FLAG_VERBOSE {
                    println!("Effective infected individuals = {effective_infected}");
                }
            },
            SeedModel::TopDegreeMultiLocus => {
                if FLAG_VERBOSE {
                    println!("Sorry! Not implemented yet!")
                }
                todo!()
            },
            SeedModel::TopDegreeNeighborhood => {
                let hub = self.find_top_degree_agent().unwrap();
                let hub_neighs = self.inner()[hub].neighbors.clone();
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
                if FLAG_VERBOSE {
                    println!("Effective infected individuals = {effective_infected}");
                }
            },
        }
    }

    pub fn introduce_opinions(
        &mut self, 
        active_fraction: f64, 
        zealot_fraction: f64,
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
                    self.inner_mut()[idx].threshold = CONST_ZEALOT_THRESHOLD;
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
                    self.inner_mut()[idx].threshold = CONST_ZEALOT_THRESHOLD;
                }
            }
        }
    }

    pub fn introduce_vaccination_attitudes(
        &mut self,
        vpars: &VaccinationPars,
    ) {
        let hesitancy_attribution_model = vpars.hesitancy_attribution;
        let nagents = self.number_of_agents();

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

        let age_threshold = vpars.age_threshold;

        match hesitancy_attribution_model {
            HesitancyAttributionModel::Adult => {
                todo!()
            },
            HesitancyAttributionModel::DataDriven => {
                todo!()
            },
            HesitancyAttributionModel::Elder => {
                let mut all_indices: Vec<usize> = (0..nagents).collect();
                all_indices.shuffle(&mut rand::thread_rng());

                let elder_indices: HashSet<usize> = self.gather_elders().into_iter().collect();
                
                all_indices.retain(|x| !elder_indices.contains(x));

                for i in elder_indices {
                    if never_assigned < never_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Never);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                        never_assigned += 1;
                    } else if majority_assigned < majority_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Most);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                        majority_assigned += 1;
                    } else if someone_assigned < someone_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Someone);
                        self.inner_mut()[i].status = Status::HesSus;
                        let num_neighbors = self.inner()[i].number_of_neighbors();
                        self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                        someone_assigned += 1;
                    } else if soon_assigned < soon_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Soon);
                        self.inner_mut()[i].status = Status::ActSus;
                        self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                        soon_assigned += 1;
                    } else if already_assigned < already_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                        self.inner_mut()[i].status = Status::ActVac;
                        self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                        already_assigned += 1;
                    }
                }

                for i in all_indices {
                    if already_assigned < already_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                            self.inner_mut()[i].status = Status::ActVac;
                            self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                            already_assigned += 1;
                        }
                    } else if soon_assigned < soon_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Soon);
                            self.inner_mut()[i].status = Status::ActSus;
                            self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                            soon_assigned += 1;
                        }
                    } else if someone_assigned < someone_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Someone);
                            self.inner_mut()[i].status = Status::HesSus;
                            let num_neighbors = self.inner()[i].number_of_neighbors();
                            self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                            someone_assigned += 1;
                        }
                    } else if majority_assigned < majority_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Most);
                            self.inner_mut()[i].status = Status::HesSus;
                            self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                            majority_assigned += 1;
                        }
                    } else if never_assigned < never_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Never);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                        never_assigned += 1;
                    }
                }        
            },
            HesitancyAttributionModel::ElderToYoung => {
                let indices: Vec<usize> = self.gather_from_elder_to_younger();

                for i in indices {
                    if never_assigned < never_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Never);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                        never_assigned += 1;
                    } else if majority_assigned < majority_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Most);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                        majority_assigned += 1;
                    } else if someone_assigned < someone_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Someone);
                        self.inner_mut()[i].status = Status::HesSus;
                        let num_neighbors = self.inner()[i].number_of_neighbors();
                        self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                        someone_assigned += 1;
                    } else if soon_assigned < soon_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Soon);
                        self.inner_mut()[i].status = Status::ActSus;
                        self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                        soon_assigned += 1;
                    } else if already_assigned < already_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                        self.inner_mut()[i].status = Status::ActVac;
                        self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                        already_assigned += 1;
                    }
                }
            },
            HesitancyAttributionModel::Middleage => {
                let mut all_indices: Vec<usize> = (0..nagents).collect();
                all_indices.shuffle(&mut rand::thread_rng());

                let middleage_indices: HashSet<usize> = self.gather_middleage().into_iter().collect();
                
                all_indices.retain(|x| !middleage_indices.contains(x));

                for i in middleage_indices {
                    if never_assigned < never_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Never);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                        never_assigned += 1;
                    } else if majority_assigned < majority_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Most);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                        majority_assigned += 1;
                    } else if someone_assigned < someone_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Someone);
                        self.inner_mut()[i].status = Status::HesSus;
                        let num_neighbors = self.inner()[i].number_of_neighbors();
                        self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                        someone_assigned += 1;
                    } else if soon_assigned < soon_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Soon);
                        self.inner_mut()[i].status = Status::ActSus;
                        self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                        soon_assigned += 1;
                    } else if already_assigned < already_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                        self.inner_mut()[i].status = Status::ActVac;
                        self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                        already_assigned += 1;
                    }
                }

                for i in all_indices {
                    if already_assigned < already_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                            self.inner_mut()[i].status = Status::ActVac;
                            self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                            already_assigned += 1;
                        }
                    } else if soon_assigned < soon_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Soon);
                            self.inner_mut()[i].status = Status::ActSus;
                            self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                            soon_assigned += 1;
                        }
                    } else if someone_assigned < someone_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Someone);
                            self.inner_mut()[i].status = Status::HesSus;
                            let num_neighbors = self.inner()[i].number_of_neighbors();
                            self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                            someone_assigned += 1;
                        }
                    } else if majority_assigned < majority_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Most);
                            self.inner_mut()[i].status = Status::HesSus;
                            self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                            majority_assigned += 1;
                        }
                    } else if never_assigned < never_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Never);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                        never_assigned += 1;
                    }
                }  
            },
            HesitancyAttributionModel::Random => {
                let mut indices: Vec<usize> = (0..nagents).collect();
                indices.shuffle(&mut rand::thread_rng());

                for i in indices {
                    if already_assigned < already_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                            self.inner_mut()[i].status = Status::ActVac;
                            self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                            already_assigned += 1;
                        }
                    } else if soon_assigned < soon_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Soon);
                            self.inner_mut()[i].status = Status::ActSus;
                            self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                            soon_assigned += 1;
                        }
                    } else if someone_assigned < someone_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Someone);
                            self.inner_mut()[i].status = Status::HesSus;
                            let num_neighbors = self.inner()[i].number_of_neighbors();
                            self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                            someone_assigned += 1;
                        }
                    } else if majority_assigned < majority_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Most);
                            self.inner_mut()[i].status = Status::HesSus;
                            self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                            majority_assigned += 1;
                        }
                    } else if never_assigned < never_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Never);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                        never_assigned += 1;
                    }
                }
            },
            HesitancyAttributionModel::Underage => {
                todo!()
            },
            HesitancyAttributionModel::Young => {
                let mut all_indices: Vec<usize> = (0..nagents).collect();
                all_indices.shuffle(&mut rand::thread_rng());

                let young_adult_indices: HashSet<usize> = self.gather_young_adults().into_iter().collect();
                
                all_indices.retain(|x| !young_adult_indices.contains(x));

                for i in young_adult_indices {
                    if never_assigned < never_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Never);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                        never_assigned += 1;
                    } else if majority_assigned < majority_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Most);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                        majority_assigned += 1;
                    } else if someone_assigned < someone_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Someone);
                        self.inner_mut()[i].status = Status::HesSus;
                        let num_neighbors = self.inner()[i].number_of_neighbors();
                        self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                        someone_assigned += 1;
                    } else if soon_assigned < soon_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Soon);
                        self.inner_mut()[i].status = Status::ActSus;
                        self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                        soon_assigned += 1;
                    } else if already_assigned < already_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                        self.inner_mut()[i].status = Status::ActVac;
                        self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                        already_assigned += 1;
                    }
                }

                for i in all_indices {
                    if already_assigned < already_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                            self.inner_mut()[i].status = Status::ActVac;
                            self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                            already_assigned += 1;
                        }
                    } else if soon_assigned < soon_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Soon);
                            self.inner_mut()[i].status = Status::ActSus;
                            self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                            soon_assigned += 1;
                        }
                    } else if someone_assigned < someone_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Someone);
                            self.inner_mut()[i].status = Status::HesSus;
                            let num_neighbors = self.inner()[i].number_of_neighbors();
                            self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                            someone_assigned += 1;
                        }
                    } else if majority_assigned < majority_count {
                        if self.inner()[i].age >= age_threshold {
                            self.inner_mut()[i].attitude = Some(Attitude::Most);
                            self.inner_mut()[i].status = Status::HesSus;
                            self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                            majority_assigned += 1;
                        }
                    } else if never_assigned < never_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Never);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                        never_assigned += 1;
                    }
                }
            },
            HesitancyAttributionModel::YoungToElder => {
                let indices: Vec<usize> = self.gather_from_younger_to_elder();

                for i in indices {
                    if never_assigned < never_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Never);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                        never_assigned += 1;
                    } else if majority_assigned < majority_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Most);
                        self.inner_mut()[i].status = Status::HesSus;
                        self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                        majority_assigned += 1;
                    } else if someone_assigned < someone_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Someone);
                        self.inner_mut()[i].status = Status::HesSus;
                        let num_neighbors = self.inner()[i].number_of_neighbors();
                        self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                        someone_assigned += 1;
                    } else if soon_assigned < soon_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Soon);
                        self.inner_mut()[i].status = Status::ActSus;
                        self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                        soon_assigned += 1;
                    } else if already_assigned < already_count {
                        self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                        self.inner_mut()[i].status = Status::ActVac;
                        self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                        already_assigned += 1;
                    }
                }
            },
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
                self.inner_mut()[i].attitude = Some(Attitude::Vaccinated);
                self.inner_mut()[i].status = Status::ActVac;
                self.inner_mut()[i].threshold = CONST_ALREADY_THRESHOLD;
                already_assigned += 1;
            } else if soon_assigned < soon_count {
                self.inner_mut()[i].attitude = Some(Attitude::Soon);
                self.inner_mut()[i].status = Status::ActSus;
                self.inner_mut()[i].threshold = CONST_SOON_THRESHOLD;
                soon_assigned += 1;
            } else if someone_assigned < someone_count {
                self.inner_mut()[i].attitude = Some(Attitude::Someone);
                self.inner_mut()[i].status = Status::HesSus;
                let num_neighbors = self.inner()[i].number_of_neighbors();
                self.inner_mut()[i].threshold = 1.0 / num_neighbors as f64;
                someone_assigned += 1;
            } else if majority_assigned < majority_count {
                self.inner_mut()[i].attitude = Some(Attitude::Most);
                self.inner_mut()[i].status = Status::HesSus;
                self.inner_mut()[i].threshold = CONST_MAJORITY_THRESHOLD;
                majority_assigned += 1;
            } else if never_assigned < never_count {
                self.inner_mut()[i].attitude = Some(Attitude::Never);
                self.inner_mut()[i].status = Status::HesSus;
                self.inner_mut()[i].threshold = CONST_ZEALOT_THRESHOLD;
                never_assigned += 1;
            }
        }
    }

    pub fn number_of_agents(&self) -> usize {
        self.inner.len()
    }

    fn reintroduce_infections(&mut self, nseeds: usize) {
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

    pub fn sample_age(&mut self, cdf: &Vec<f64>) {
        for agent in self.inner_mut() {
            agent.sample_age(cdf);
        }
    }

    pub fn sample_degree(&mut self, cdf: &Vec<f64>) {
        for agent in self.inner_mut() {
            agent.sample_degree(cdf);
        }
    }

    pub fn sample_degree_conditioned_to_age(
        &mut self, 
        intralayer_average_degree: &Vec<f64>,
    ) {
        for agent in self.inner_mut() {
            let age = agent.age;
            let average_degree = intralayer_average_degree[age];
            let standard_deviation = 4.0;
            agent.sample_degree_from_negative_binomial(average_degree, standard_deviation);
        }
    } 

    pub fn set_opinion_threshold(&mut self, threshold: f64) {
        for agent in self.inner_mut() {
            agent.threshold = threshold;
        }
    }

    pub fn set_vaccination_policy_model(
        &mut self, vaccination_policy: VaccinationPolicy, 
        vaccination_quota: f64,
    ) {
        match vaccination_policy {
            VaccinationPolicy::AgeAdult => {
                todo!()
            },
            VaccinationPolicy::AgeElder => {
                self.target_age_elders(vaccination_quota)
            },
            VaccinationPolicy::AgeMiddleage => {
                self.target_age_middleage(vaccination_quota)
            },
            VaccinationPolicy::AgeTop => {
                self.target_age_top(vaccination_quota)
            },
            VaccinationPolicy::AgeUnderage => {
                self.target_age_underage(vaccination_quota)
            },
            VaccinationPolicy::AgeYoung => {
                self.target_age_young_adult(vaccination_quota)
            },
            VaccinationPolicy::AgeYoungToElder => {
                self.target_age_young_to_elder(vaccination_quota)
            },
            VaccinationPolicy::Automatic => {
                self.target_automatic(vaccination_quota)
            },
            VaccinationPolicy::ComboElderTop => {
                todo!()
            },
            VaccinationPolicy::ComboYoungTop => {
                todo!()
            },
            VaccinationPolicy::DataDriven => {
                todo!()
            },
            VaccinationPolicy::DegreeBottom => {
                self.target_degree_bottom_to_top(vaccination_quota)
            },
            VaccinationPolicy::DegreeRandom => {
                self.target_random(vaccination_quota)
            },
            VaccinationPolicy::DegreeTop => {
                self.target_degree_top_to_bottom(vaccination_quota)
            },
            VaccinationPolicy::Random => {
                self.target_random(vaccination_quota)
            },
        }
    }

    pub fn target_age_elders(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents = self.number_of_agents();

        for agent in self.inner_mut() {
            if target_quota >= vaccination_quota {
                break;
            }
            if agent.age < CONST_UNDERAGE_THRESHOLD {
                agent.vaccination_target = true;
                target_quota += 1.0 / nagents as f64;   
            }
        }
    }

    pub fn target_age_middleage(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents =self.number_of_agents();

        for agent in self.inner_mut() {
            if target_quota >= vaccination_quota {
                break;
            }
            if CONST_MIDDLEAGE_THRESHOLD <= agent.age && CONST_ELDER_THRESHOLD > agent.age {
                agent.vaccination_target = true;
                target_quota += 1.0 / nagents as f64;   
            }
        }
    }

    pub fn target_age_top(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents = self.number_of_agents();

        let mut agents: Vec<_> = self.inner().iter() // Use .iter() to get an iterator over the agents
        .filter(|agent| agent.age >= CONST_UNDERAGE_THRESHOLD)
        .collect::<Vec<&Agent>>(); // Collect into a vector of references to Agent
        
        // Now sort this vector of references by age in descending order
        agents.sort_by(|a, b| b.age.cmp(&a.age));

        // Map the sorted agent references to their IDs and collect into a vector
        let older_to_younger: Vec<usize> = agents.iter().map(|agent| agent.id).collect();

        for id in older_to_younger {
            if target_quota >= vaccination_quota {
                break;
            }
            if self.inner()[id].age < CONST_UNDERAGE_THRESHOLD {
                self.inner_mut()[id].vaccination_target = true;
                target_quota += 1.0 / nagents as f64;   
            }
        }
    }

    pub fn target_age_underage(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents = self.number_of_agents();

        for agent in self.inner_mut() {
            if target_quota >= vaccination_quota {
                break;
            }
            if agent.age >= CONST_ELDER_THRESHOLD {
                agent.vaccination_target = true;
                target_quota += 1.0 / nagents as f64;   
            }
        }
    }

    pub fn target_age_young_adult(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents = self.number_of_agents();

        for agent in self.inner_mut() {
            if target_quota >= vaccination_quota {
                break;
            }
            if agent.age >= CONST_UNDERAGE_THRESHOLD && agent.age < CONST_MIDDLEAGE_THRESHOLD {
                agent.vaccination_target = true;
                target_quota += 1.0 / nagents as f64;   
            }
        }
    }

    pub fn target_age_young_to_elder(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents = self.number_of_agents();

        let mut agents: Vec<_> = self.inner().iter() // Use .iter() to get an iterator over the agents
        .filter(|agent| agent.age >= CONST_UNDERAGE_THRESHOLD)
        .collect::<Vec<&Agent>>(); // Collect into a vector of references to Agent
        
        // Now sort this vector of references by age in descending order
        agents.sort_by(|a, b| a.age.cmp(&b.age));
        
        // Map the sorted agent references to their IDs and collect into a vector
        let young_to_elder: Vec<usize> = agents.iter().map(|agent| agent.id).collect();

        for id in young_to_elder {
            if target_quota >= vaccination_quota {
                break;
            }
            if self.inner()[id].age < CONST_UNDERAGE_THRESHOLD {
                self.inner_mut()[id].vaccination_target = true;
                target_quota += 1.0 / nagents as f64;   
            }
        }
    }

    pub fn target_automatic(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents = self.number_of_agents();

        for agent in self.inner_mut() {
            if target_quota >= vaccination_quota {
                break;
            }
            agent.vaccination_target = true;
            target_quota += 1.0 / nagents as f64;
        }
    }

    pub fn target_degree_bottom_to_top(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents = self.number_of_agents();

        let mut agents_with_degrees: Vec<(usize, usize)> = self.inner().iter()
        .enumerate()
        .map(|(id, agent)| (id, agent.neighbors.len()))
        .collect();

        agents_with_degrees.sort_by(|a, b| a.1.cmp(&b.1));

        let bottom_to_top_ids: Vec<usize> = agents_with_degrees.into_iter()
            .map(|(id, _)| id)
            .collect();

        for id in bottom_to_top_ids {
            if target_quota >= vaccination_quota {
                break;
            }
            if self.inner()[id].age < CONST_UNDERAGE_THRESHOLD {
                self.inner_mut()[id].vaccination_target = true;
                target_quota += 1.0 / nagents as f64;   
            }
        }
    }

    pub fn target_degree_top_to_bottom(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents = self.number_of_agents();
        
        let mut agents_with_degrees: Vec<(usize, usize)> = self.inner().iter()
        .enumerate()
        .map(|(id, agent)| (id, agent.neighbors.len()))
        .collect();

        agents_with_degrees.sort_by(|a, b| b.1.cmp(&a.1));

        let top_to_bottom_ids: Vec<usize> = agents_with_degrees.into_iter()
            .map(|(id, _)| id)
            .collect();

        for id in top_to_bottom_ids {
            if target_quota >= vaccination_quota {
                break;
            }
            if self.inner()[id].age < CONST_UNDERAGE_THRESHOLD {
                self.inner_mut()[id].vaccination_target = true;
                target_quota += 1.0 / nagents as f64;   
            }
        }
    }

    pub fn target_random(&mut self, vaccination_quota: f64) {
        let mut target_quota = 0.0;
        let nagents = self.number_of_agents();
        let mut shuffled_indices: Vec<usize> = (0..nagents).collect();
        shuffled_indices.shuffle(&mut rand::thread_rng());

        for id in shuffled_indices {
            if target_quota >= vaccination_quota {
                break;
            }
            self.inner_mut()[id].vaccination_target = true;
            target_quota += 1.0 / nagents as f64;
        }
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

    pub fn total_hesitant(&self) -> u32 {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.opinion == Opinion::Hesitant {
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

    pub fn total_susceptible(&self) -> usize {
        let mut summa = 0;
        for agent in self.inner() {
            if agent.health == Health::Susceptible {
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

    pub fn total_zealot(&self) -> usize {
        let mut summa = 0;
        for agent in self.inner()  {
            if agent.threshold > 1.0 {
                summa += 1;
            }
        }
        summa
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