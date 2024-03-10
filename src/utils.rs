use rand::Rng;
use serde_pickle::SerOptions;
use std::cmp::Ordering;
use std::fs::{File, self};
use std::io::Read;
use std::path::PathBuf;
use std::collections::{HashSet, HashMap};
use std::{vec, env};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use strum::Display;

use netrust::network::Network;
use netrust::utils::{NetworkPars, NetworkModel};
use crate::agent::{AgentEnsemble, Attitude, HesitancyAttributionModel, SeedModel, Status, VaccinationPolicy};
use crate::cons::{CONST_EPIDEMIC_THRESHOLD, EXTENSION_RESULTS, FOLDER_RESULTS, HEADER_AGE, HEADER_AGENT, HEADER_AGENT_DISTRIBUTION, HEADER_AGENT_STATS, HEADER_CLUSTER, HEADER_CLUSTER_DISTRIBUTION, HEADER_CLUSTER_STATS, HEADER_GLOBAL, HEADER_PROJECT, HEADER_REBUILD, HEADER_REBUILD_STATS, HEADER_TIME, INIT_ATTITUDE, INIT_STATUS, INIT_USIZE, PAR_AGE_GROUPS, PAR_NBINS, PAR_OUTBREAK_PREVALENCE_FRACTION_CUTOFF};

pub fn build_normalized_cdf(values: &mut [f64]) -> Vec<f64> {
    let sum: f64 = values.iter().sum();

    let is_normalized = (sum - 1.0).abs() < 1e-6;

    if !is_normalized {
        values.iter_mut().for_each(|v| *v /= sum);
    }

    let mut cdf = Vec::new();
    let mut cum_sum = 0.0;
    for &mut value in values {
        cum_sum += value;
        cdf.push(cum_sum);
    }

    cdf
}

pub fn build_normalized_layered_cdf(values: &mut Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    values.iter_mut().map(|layer| {
        // Calculate the sum of the current layer
        let sum: f64 = layer.iter().sum();

        // Check if the layer is already normalized
        let is_normalized = (sum - 1.0).abs() < 1e-6;

        // Normalize the layer if it is not already normalized
        if !is_normalized {
            layer.iter_mut().for_each(|v| *v /= sum);
        }

        // Build the CDF for the current layer
        let mut cdf = Vec::new();
        let mut cum_sum = 0.0;
        for &value in layer.iter() {
            cum_sum += value;
            cdf.push(cum_sum);
        }

        cdf
    }).collect()
}

pub fn compute_beta_from_r0(
    r0: f64, 
    removal_rate: f64, 
    npars: &NetworkPars, 
    graph: &Network,
) -> f64 {
    match npars.model {
        NetworkModel::BarabasiAlbert => {
            let k_avg = graph.average_degree();
            //let k2_avg = graph.second_moment();
            r0 * removal_rate / k_avg // (k2_avg - k_avg))
        },
        NetworkModel::Complete => {
            r0 * removal_rate / (npars.size - 1) as f64
        },
        NetworkModel::ErdosRenyi => {
            let k_avg = graph.average_degree();
            r0 * removal_rate / k_avg
        },
        NetworkModel::Regular => {
            let k_avg = graph.average_degree();
            r0 * removal_rate / k_avg
        },
        NetworkModel::ScaleFree => {
            let k_avg = graph.average_degree();
            //let k2_avg = graph.second_moment();
            r0 * removal_rate / k_avg // (k2_avg - k_avg))
        },
        NetworkModel::WattsStrogatz => {
            let k_avg = graph.average_degree();
            r0 * removal_rate / k_avg
        },
    }
}

pub fn compute_interlayer_probability_matrix(
    contact_matrix: &Vec<Vec<f64>>
) -> Vec<Vec<f64>> {
     let mut probabilities = vec![vec![0.0; contact_matrix.len()]; contact_matrix.len()];
     for (alpha, row) in contact_matrix.iter().enumerate() {
         let sum: f64 = row.iter().sum();
         for (beta, &value) in row.iter().enumerate() {
             probabilities[alpha][beta] = value / sum;
         }
     }
     probabilities
}

pub fn compute_intralayer_average_degree(contact_matrix: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut intralayer_average_degree = vec![0.0; contact_matrix.len()];

    for (alpha, row) in contact_matrix.into_iter().enumerate()  {
        let sum: f64 = row.iter().sum();
        intralayer_average_degree[alpha] = sum;
    }

    intralayer_average_degree
}

pub fn convert_hm_value_to_bool(
    hash_map: HashMap<String, Value>,
) -> HashMap<String, bool> {
    hash_map
    .into_iter()
    .map(|(key, value)| {
        let bool_value = match value.as_bool() {
            Some(v) => v,
            None => panic!("Value conversion error for key: {}", key),
        };
        (key, bool_value)
    })
    .collect()
}

pub fn convert_hm_value_to_f64(
    hash_map: HashMap<String, Value>,
) -> HashMap<String, f64> {
    hash_map
    .into_iter()
    .map(|(key, value)| {
        let f64_value = match value.as_f64() {
            Some(v) => v,
            None => panic!("Value conversion error for key: {}", key),
        };
        (key, f64_value)
    })
    .collect()
}

pub fn count_underaged(population_vector: &[f64]) -> f64 {    
    population_vector
        .iter()
        .take(18)
        .sum()
}

pub fn load_json_config(
    filename: &str,
    subfolder: Option<&str>,
) -> Result<Input, Box<dyn std::error::Error>> {
    let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
    
    let subfolder = subfolder.unwrap_or("config");
    path.push(subfolder);
    if !path.exists() {
        fs::create_dir(&path)?;
    }

    path.push(format!("{}.json", filename));

    if !path.exists() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound, 
            format!("File not found: {:?}", path)
        )));
    }

    let mut file = File::open(&path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let config: Input = serde_json::from_str(&contents)?;

    Ok(config)
}

pub fn load_json_data(filename: &str) -> HashMap<String, Value> {
    let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
    path.push("config");
    path.push(format!("{}.json", filename));

    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let data: HashMap<String, Value> = 
    serde_json::from_str(&contents).unwrap();

    data
}

pub fn read_key_and_f64_from_json(state: USState, filename: &str) -> f64 {
    let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
    path.push("data");
    path.push(format!("{}.json", filename));

    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let data: Value = serde_json::from_str(&contents).unwrap();

    let state_str = state.to_string();
    let state_data = &data[state_str];
    let data_f64: f64 = serde_json::from_value(state_data.clone())
        .expect("Failed to parse state data");

    data_f64
}

pub fn read_key_and_matrixf64_from_json(
    state: USState, 
    filename: &str,
) -> Vec<Vec<f64>> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");
    path.push(format!("{}.json", filename));

    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let data: Value = serde_json::from_str(&contents).unwrap();

    let state_str = state.to_string();
    let state_data = &data[state_str];
    let data_matrixf64: Vec<Vec<f64>> = serde_json::from_value(state_data.clone())
        .expect("Failed to parse state data");

    data_matrixf64
}

pub fn read_key_and_vecf64_from_json(
    state: USState, 
    filename: &str,
) -> Vec<f64> {
    let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
    path.push("data");
    path.push(format!("{}.json", filename));

    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let data: Value = serde_json::from_str(&contents).unwrap();

    let state_str = state.to_string();
    let state_data = &data[state_str];
    let data_vecf64: Vec<f64> = serde_json::from_value(state_data.clone())
        .expect("Failed to parse state data");

    data_vecf64
}

pub fn remove_duplicates(vec: Vec<usize>) -> Vec<usize> {
    let set: HashSet<_> = vec.into_iter().collect();
    set.into_iter().collect()
}

pub fn sample_from_cdf(cdf: &[f64]) -> usize {
    let mut rng = rand::thread_rng();
    let u: f64 = rng.gen();

    // Find the index where CDF[i] >= u
    cdf.iter().position(|&value| value >= u).unwrap_or(0)
}

pub fn select_network_model(
    net_id: NetworkModel,
) -> HashMap<String, Value> {
    match net_id {
        NetworkModel::BarabasiAlbert => {
            let filename = "config_network_model_barabasialbert";
            load_json_data(filename)
        },
        NetworkModel::Complete => {
            let filename = "config_network_model_complete";
            load_json_data(filename)
        },
        NetworkModel::ErdosRenyi => {
            let filename = "config_network_model_erdosrenyi";
            load_json_data(filename)
        },
        NetworkModel::Regular => {
            let filename = "config_network_model_regular";
            load_json_data(filename)
        },
        NetworkModel::ScaleFree => {
            let filename = "config_network_model_scalefree";
            load_json_data(filename)
        },
        NetworkModel::WattsStrogatz => {
            let filename = "config_network_model_wattsstrogatz";
            load_json_data(filename)
        },
    }
}

pub fn sir_prevalence(r0: f64, sus0: f64) -> f64 {
    const MAX_ITERATIONS: usize = 10000;
    const TOLERANCE: f64 = 1e-6;

    let mut r_inf = 0.0;
    let mut guess = 0.8;

    for _ in 0..MAX_ITERATIONS {
        r_inf = 1.0 - sus0 * (-r0 * guess).exp();

        if (r_inf - guess).abs() < TOLERANCE {
            return r_inf;
        }

        guess = r_inf;
    }

    r_inf
}

fn write_algorithm_string(apars: &AlgorithmPars) -> String {
    format!(
        "_nsd{0}_nsn{1}_tmax{2}",
        apars.nsims_dyn,
        apars.nsims_net, 
        apars.t_max,
    )
}

fn write_epidemic_string(epars: &EpidemicPars) -> String {
    format!(
        "_r0{0}_rer{1}",
        epars.r0,
        epars.infection_decay, 
    )
}

pub fn write_file_name(
    pars: &Input, 
    exp_id: String,
    ml_flag: bool,
) -> String {
    let head = HEADER_PROJECT.to_string();
    let npars_chain = if ml_flag {
        write_multilayer_string(&pars.network.unwrap())
    } else {
        write_network_string(&pars.network.unwrap())
    };
    let opars_chain = write_opinion_string(&pars.opinion.unwrap());
    let epars_chain = write_epidemic_string(&pars.epidemic);
    let vpars_chain = write_vaccination_string(&pars.vaccination.unwrap());
    let apars_chain = write_algorithm_string(&pars.algorithm.unwrap());
    head + &exp_id + &npars_chain + &opars_chain + &epars_chain + &vpars_chain + &apars_chain
}

fn write_multilayer_string(npars: &NetworkPars) -> String {    
    format!("_nml_n{}", npars.size)
}

fn write_network_string(npars: &NetworkPars) -> String {    
    match npars.model {
        NetworkModel::BarabasiAlbert => {
            format!("_netba_n{}_k{}", npars.size, npars.average_degree.unwrap())
        },
        NetworkModel::Complete => {
            format!("_netco_n{}", npars.size)
        },
        NetworkModel::ErdosRenyi => {
            format!("_neter_n{}_k{}", npars.size, npars.average_degree.unwrap())
        },
        NetworkModel::Regular => {
            format!("_netre_n{}_k{}", npars.size, npars.average_degree.unwrap())
        },
        NetworkModel::ScaleFree => {
            format!("_netsf_n{}_kmin{}_kmax{}_gamma{}", npars.size, npars.minimum_degree.unwrap(), npars.maximum_degree.unwrap(), npars.powerlaw_exponent.unwrap())
        },
        NetworkModel::WattsStrogatz => {
            format!("_netws_n{}_k{}_p{}", npars.size, npars.average_degree.unwrap(), npars.rewiring_probability.unwrap())
        },
    }
}

fn write_opinion_string(opars: &OpinionPars) -> String {
    format!(
        "_acf{0}_thr{1}_zef{2}",
        opars.active_fraction,
        opars.threshold,
        opars.zealot_fraction,
    )
}

fn write_vaccination_string(vpars: &VaccinationPars) -> String {
    format!(
        "_ath{0}_hem{1}_vpo{2}_vqu{3}_var{4}",
        vpars.age_threshold,
        vpars.hesitancy_attribution,
        vpars.vaccination_policy,
        vpars.vaccination_quota, 
        vpars.vaccination_rate,
    )
}

#[derive(Serialize)]
struct AgentOutput {
    pub activation_potential: Option<usize>,
    pub age: Option<usize>,
    pub attitude: Option<Attitude>,
    pub convinced_when: Option<usize>,
    pub degree: Option<usize>,
    pub final_active_susceptible: Option<usize>,
    pub final_prevalence: Option<usize>,
    pub final_vaccinated: Option<usize>,
    pub id: Option<usize>,
    pub infected_by: Option<usize>, 
    pub infected_when: Option<usize>,
    pub initial_active_susceptible: Option<usize>,
    pub initial_vaccinated: Option<usize>,
    pub neighbors: Option<Vec<usize>>,
    pub removed_when: Option<usize>,
    pub status: Option<Status>,
    pub threshold: Option<f64>,
    pub vaccinated_when: Option<usize>,
    pub zealots: Option<usize>,
}

impl AgentOutput {
    fn new(
        activation_potential: Option<usize>,
        age: Option<usize>,
        attitude: Option<Attitude>,
        convinced_when: Option<usize>,
        degree: Option<usize>,
        final_active_susceptible: Option<usize>,
        final_prevalence: Option<usize>,
        final_vaccinated: Option<usize>,
        id: Option<usize>,
        infected_by: Option<usize>, 
        infected_when: Option<usize>,
        initial_active_susceptible: Option<usize>,
        initial_vaccinated: Option<usize>,
        neighbors: Option<Vec<usize>>,
        removed_when: Option<usize>,
        status: Option<Status>,
        threshold: Option<f64>,
        vaccinated_when: Option<usize>,
        zealots: Option<usize>,
    ) -> Self {
        Self {
            activation_potential,
            age,
            attitude,
            convinced_when,
            degree,
            final_active_susceptible,
            final_prevalence,
            final_vaccinated,
            id,
            infected_by, 
            infected_when,
            initial_active_susceptible,
            initial_vaccinated,
            neighbors,
            removed_when,
            status,
            threshold,
            vaccinated_when,
            zealots,
        }
    }
}

#[derive(Serialize)]
pub struct AgentEnsembleOutput {
    inner: Vec<AgentOutput>,
}

impl AgentEnsembleOutput {
    pub fn new(agent_ensemble: &AgentEnsemble) -> Self {
        let mut agent_ensemble_output = Self { inner: Vec::new() };
        
        for agent in agent_ensemble.inner() {
            let activation_potential = agent.activation_potential.unwrap_or(INIT_USIZE);
            let age = agent.age;
            let attitude = agent.attitude.unwrap_or(INIT_ATTITUDE);
            let convinced_when = agent.convinced_when.unwrap_or(INIT_USIZE);
            let degree = agent.neighbors.len();
            let final_active_susceptible = agent.final_active_susceptible.unwrap_or(INIT_USIZE);
            let final_vaccinated = agent.final_vaccinated.unwrap_or(INIT_USIZE);
            let final_prevalence = agent.final_prevalence.unwrap_or(INIT_USIZE);
            let id = agent.id;
            let infected_by = agent.infected_by.unwrap_or(INIT_USIZE);
            let infected_when = agent.infected_when.unwrap_or(INIT_USIZE);
            let initial_active_susceptible = agent.initial_active_susceptible.unwrap_or(INIT_USIZE);
            let initial_vaccinated = agent.initial_vaccinated.unwrap_or(INIT_USIZE);
            let neighbors = agent.neighbors.clone();
            let removed_when = agent.removed_when.unwrap_or(INIT_USIZE);
            let status = agent.status;
            let threshold = agent.threshold;
            let vaccinated_when = agent.vaccinated_when.unwrap_or(INIT_USIZE);
            let zealots = agent.zealots.unwrap_or(INIT_USIZE);
    
            let agent_output = AgentOutput::new(
                Some(activation_potential),
                Some(age),
                Some(attitude),
                Some(convinced_when),
                Some(degree),
                Some(final_active_susceptible),
                Some(final_prevalence),
                Some(final_vaccinated),
                Some(id),
                Some(infected_by), 
                Some(infected_when), 
                Some(initial_active_susceptible),
                Some(initial_vaccinated),
                Some(neighbors),
                Some(removed_when),
                Some(status),
                Some(threshold),
                Some(vaccinated_when), 
                Some(zealots),
            );
            agent_ensemble_output.inner.push(agent_output);
        }
        agent_ensemble_output
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct AlgorithmPars {
    pub experiment_id: usize,
    pub nsims_dyn: usize,
    pub nsims_net: usize,
    pub t_max: usize,
}

impl AlgorithmPars {
    pub fn new(
        experiment_id: usize,
        nsims_dyn: usize,
        nsims_net: usize,
        t_max: usize,
    ) -> Self {
        Self {
            experiment_id,
            nsims_dyn,
            nsims_net,
            t_max, 
        }
    }
}

#[derive(Serialize)]
pub struct AssembledAgeOutput {
    activation_potential: Vec<Vec<Vec<usize>>>,
    active: Vec<Vec<usize>>,
    age: Vec<Vec<usize>>,
    convinced_when: Vec<Vec<Vec<usize>>>,
    degree: Vec<Vec<Vec<usize>>>,
    infected_when: Vec<Vec<Vec<usize>>>,
    prevalence: Vec<Vec<usize>>,
    removed_when: Vec<Vec<Vec<usize>>>,
    vaccinated: Vec<Vec<usize>>,
    vaccinated_when: Vec<Vec<Vec<usize>>>,
    zealots: Vec<Vec<usize>>,
}

impl AssembledAgeOutput {
    pub fn new(
        activation_potential: Vec<Vec<Vec<usize>>>,
        active: Vec<Vec<usize>>,
        age: Vec<Vec<usize>>,
        convinced_when: Vec<Vec<Vec<usize>>>,
        degree: Vec<Vec<Vec<usize>>>,
        infected_when: Vec<Vec<Vec<usize>>>,
        prevalence: Vec<Vec<usize>>,
        removed_when: Vec<Vec<Vec<usize>>>,
        vaccinated: Vec<Vec<usize>>,
        vaccinated_when: Vec<Vec<Vec<usize>>>,
        zealots: Vec<Vec<usize>>,
    ) -> Self {
        Self { 
            activation_potential,
            active, 
            age,
            convinced_when, 
            degree,
            infected_when, 
            prevalence, 
            removed_when, 
            vaccinated, 
            vaccinated_when, 
            zealots,
        }
    }
}

#[derive(Serialize)]
pub struct AssembledAgentOutput {
    pub activation_potential: Vec<Vec<usize>>,
    pub age: Vec<Vec<usize>>,
    pub attitude: Vec<Vec<Attitude>>,
    pub convinced_when: Vec<Vec<usize>>,
    pub degree: Vec<Vec<usize>>,
    pub final_active_susceptible: Vec<Vec<usize>>,
    pub final_prevalence: Vec<Vec<usize>>,
    pub final_vaccinated: Vec<Vec<usize>>,
    pub id: Vec<Vec<usize>>,
    pub infected_by: Vec<Vec<usize>>,
    pub infected_when: Vec<Vec<usize>>,
    pub initial_active_susceptible: Vec<Vec<usize>>,
    pub initial_vaccinated: Vec<Vec<usize>>,
    pub removed_when: Vec<Vec<usize>>,
    pub status: Vec<Vec<Status>>,
    pub threshold: Vec<Vec<f64>>,
    pub vaccinated_when: Vec<Vec<usize>>,
    pub zealots: Vec<Vec<usize>>,
}

impl AssembledAgentOutput {
    pub fn new(
        activation_potential: Vec<Vec<usize>>,
        age: Vec<Vec<usize>>,
        attitude: Vec<Vec<Attitude>>,
        convinced_when: Vec<Vec<usize>>,
        degree: Vec<Vec<usize>>,
        final_active_susceptible: Vec<Vec<usize>>,
        final_prevalence: Vec<Vec<usize>>,
        final_vaccinated: Vec<Vec<usize>>,
        id: Vec<Vec<usize>>,
        infected_by: Vec<Vec<usize>>,
        infected_when: Vec<Vec<usize>>,
        initial_active_susceptible: Vec<Vec<usize>>,
        initial_vaccinated: Vec<Vec<usize>>,
        removed_when: Vec<Vec<usize>>,
        status: Vec<Vec<Status>>,
        threshold: Vec<Vec<f64>>,
        vaccinated_when: Vec<Vec<usize>>,
        zealots: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            activation_potential,
            age,
            attitude,
            convinced_when,
            degree,
            final_active_susceptible,
            final_prevalence,
            final_vaccinated,
            id,
            infected_by,
            infected_when,
            initial_active_susceptible,
            initial_vaccinated,
            removed_when,
            status,
            threshold,
            vaccinated_when,
            zealots,
        }
    }
}


#[derive(Serialize)]
pub struct AssembledClusterAttitudeOutput {
    pub already_cluster: Vec<Vec<usize>>,
    pub soon_cluster: Vec<Vec<usize>>,
    pub someone_cluster: Vec<Vec<usize>>,
    pub majority_cluster: Vec<Vec<usize>>,
    pub never_cluster: Vec<Vec<usize>>,
}

impl AssembledClusterAttitudeOutput {
    pub fn new(
        already_cluster: Vec<Vec<usize>>, 
        soon_cluster: Vec<Vec<usize>>,
        someone_cluster: Vec<Vec<usize>>,
        majority_cluster: Vec<Vec<usize>>,
        never_cluster: Vec<Vec<usize>>
    ) -> Self {
        Self { 
            already_cluster, 
            soon_cluster, 
            someone_cluster, 
            majority_cluster, 
            never_cluster 
        }
    }
}

#[derive(Serialize)]
pub struct AssembledClusterCascadingOutput {
    pub cascading_cluster: Vec<Vec<usize>>,
    pub nonzealot_cluster: Vec<Vec<usize>>,
}

impl AssembledClusterCascadingOutput {
    pub fn new(
        cascading_cluster: Vec<Vec<usize>>, 
        nonzealot_cluster: Vec<Vec<usize>>,
    ) -> Self {
        Self { 
            cascading_cluster, 
            nonzealot_cluster,
        }
    }
}

#[derive(Serialize)]
pub struct AssembledClusterOpinionHealthOutput {
    pub ai_cluster: Vec<Vec<usize>>,
    pub ar_cluster: Vec<Vec<usize>>,
    pub as_cluster: Vec<Vec<usize>>,
    pub av_cluster: Vec<Vec<usize>>,
    pub hi_cluster: Vec<Vec<usize>>,
    pub hr_cluster: Vec<Vec<usize>>,
    pub hs_cluster: Vec<Vec<usize>>,
    pub hv_cluster: Vec<Vec<usize>>,
    pub ze_cluster: Vec<Vec<usize>>,
}

impl AssembledClusterOpinionHealthOutput {
    pub fn new(
        ai_cluster: Vec<Vec<usize>>,
        ar_cluster: Vec<Vec<usize>>,
        as_cluster: Vec<Vec<usize>>,
        av_cluster: Vec<Vec<usize>>,
        hi_cluster: Vec<Vec<usize>>,
        hr_cluster: Vec<Vec<usize>>,
        hs_cluster: Vec<Vec<usize>>,
        hv_cluster: Vec<Vec<usize>>,
        ze_cluster: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            ai_cluster,
            ar_cluster,
            as_cluster,
            av_cluster,
            hi_cluster,
            hr_cluster,
            hs_cluster,
            hv_cluster,
            ze_cluster,
        }
    }
}

#[derive(Serialize)]
pub struct AssembledDegreeOutput {
    active: Vec<Vec<usize>>,
    age: Vec<Vec<Vec<usize>>>,
    convinced_when: Vec<Vec<Vec<usize>>>,
    degree: Vec<Vec<usize>>,
    infected_when: Vec<Vec<Vec<usize>>>,
    prevalence: Vec<Vec<usize>>,
    removed_when: Vec<Vec<Vec<usize>>>,
    vaccinated: Vec<Vec<usize>>,
    vaccinated_when: Vec<Vec<Vec<usize>>>,
    zealots: Vec<Vec<usize>>,
}

impl AssembledDegreeOutput {
    pub fn new(
        active: Vec<Vec<usize>>,
        age: Vec<Vec<Vec<usize>>>,
        convinced_when: Vec<Vec<Vec<usize>>>,
        degree: Vec<Vec<usize>>,
        infected_when: Vec<Vec<Vec<usize>>>,
        prevalence: Vec<Vec<usize>>,
        removed_when: Vec<Vec<Vec<usize>>>,
        vaccinated: Vec<Vec<usize>>,
        vaccinated_when: Vec<Vec<Vec<usize>>>,
        zealots: Vec<Vec<usize>>,
    ) -> Self {
        Self { 
            age, 
            active, 
            convinced_when, 
            degree,
            infected_when, 
            prevalence, 
            removed_when, 
            vaccinated, 
            vaccinated_when, 
            zealots,
        }
    }
}

#[derive(Serialize)]
pub struct AssembledGlobalOutput {
    pub convinced: Vec<usize>,
    pub convinced_at_peak: Vec<usize>,
    pub peak_incidence: Vec<usize>,
    pub prevalence: Vec<usize>,
    pub time_to_end: Vec<usize>,
    pub time_to_peak: Vec<usize>,
    pub vaccinated: Vec<usize>,
    pub vaccinated_at_peak: Vec<usize>,
}

impl AssembledGlobalOutput {
    pub fn new(
        convinced: Vec<usize>,
        convinced_at_peak: Vec<usize>,
        peak_incidence: Vec<usize>,
        prevalence: Vec<usize>,
        time_to_end: Vec<usize>,
        time_to_peak: Vec<usize>,
        vaccinated: Vec<usize>,
        vaccinated_at_peak: Vec<usize>,
    ) -> Self {
        Self {
            convinced,
            convinced_at_peak,
            peak_incidence,
            prevalence,
            time_to_end,
            time_to_peak,
            vaccinated,
            vaccinated_at_peak,
        }
    }
}

#[derive(Serialize)]
pub struct AssembledRebuildOutput {
    pub age_distribution: Vec<Vec<f64>>,
    pub contact_matrix: Vec<Vec<Vec<f64>>>,
    pub degree_distribution: Vec<Vec<f64>>,
}

impl AssembledRebuildOutput {
    pub fn new(
        age_distribution: Vec<Vec<f64>>, 
        contact_matrix: Vec<Vec<Vec<f64>>>, 
        degree_distribution: Vec<Vec<f64>>,
    ) -> Self {
        Self { 
            age_distribution, 
            contact_matrix, 
            degree_distribution,
         }
    }
}

#[derive(Serialize)]
pub struct AssembledTimeSeriesOutput {
    pub ai_pop_st: Vec<Vec<usize>>,
    pub ar_pop_st: Vec<Vec<usize>>,
    pub as_pop_st: Vec<Vec<usize>>,
    pub av_pop_st: Vec<Vec<usize>>,
    pub hi_pop_st: Vec<Vec<usize>>,
    pub hr_pop_st: Vec<Vec<usize>>,
    pub hs_pop_st: Vec<Vec<usize>>,
    pub hv_pop_st: Vec<Vec<usize>>,
    pub t_array: Vec<Vec<usize>>,
}

impl AssembledTimeSeriesOutput {
    pub fn new(
        ai_pop_st: Vec<Vec<usize>>,
        ar_pop_st: Vec<Vec<usize>>,
        as_pop_st: Vec<Vec<usize>>,
        av_pop_st: Vec<Vec<usize>>,
        hi_pop_st: Vec<Vec<usize>>,
        hr_pop_st: Vec<Vec<usize>>,
        hs_pop_st: Vec<Vec<usize>>,
        hv_pop_st: Vec<Vec<usize>>,    
        t_array: Vec<Vec<usize>>,
    ) -> Self {
        Self {    
            ai_pop_st,
            ar_pop_st,   
            as_pop_st,
            av_pop_st, 
            hi_pop_st,
            hr_pop_st,
            hs_pop_st,
            hv_pop_st,
            t_array, 
        }
    }
}

#[derive(Serialize)]
pub struct ClusterAttitudeOutput {
    pub already_cluster: Vec<usize>,
    pub soon_cluster: Vec<usize>,
    pub someone_cluster: Vec<usize>,
    pub majority_cluster: Vec<usize>,
    pub never_cluster: Vec<usize>,
}

impl ClusterAttitudeOutput {
    pub fn new(
        already_cluster: Vec<usize>,
        soon_cluster: Vec<usize>,
        someone_cluster: Vec<usize>,
        majority_cluster: Vec<usize>,
        never_cluster: Vec<usize>,
     ) -> Self {
        Self { 
            already_cluster, 
            soon_cluster, 
            someone_cluster, 
            majority_cluster, 
            never_cluster,
        }
     }
}

#[derive(Serialize)]
pub struct ClusterCascadingOutput {
    pub cascading_cluster: Vec<usize>,
    pub nonzealot_cluster: Vec<usize>,
}

impl ClusterCascadingOutput {
    pub fn new(
        cascading_cluster: Vec<usize>,
        nonzealot_cluster: Vec<usize>,
    ) -> Self {
        Self { 
            cascading_cluster, 
            nonzealot_cluster,
        }
    }
}

#[derive(Serialize)]
pub struct ClusterOpinionHealthOutput {
    pub ai_cluster: Vec<usize>,
    pub ar_cluster: Vec<usize>,
    pub as_cluster: Vec<usize>,
    pub av_cluster: Vec<usize>,
    pub hi_cluster: Vec<usize>,
    pub hr_cluster: Vec<usize>,
    pub hs_cluster: Vec<usize>,
    pub hv_cluster: Vec<usize>,
    pub ze_cluster: Vec<usize>,
}

impl ClusterOpinionHealthOutput {
    pub fn new(
        ai_cluster: Vec<usize>,
        ar_cluster: Vec<usize>,
        as_cluster: Vec<usize>,
        av_cluster: Vec<usize>,
        hi_cluster: Vec<usize>,
        hr_cluster: Vec<usize>,
        hs_cluster: Vec<usize>,
        hv_cluster: Vec<usize>,
        ze_cluster: Vec<usize>,
    ) -> Self {
        Self {
            ai_cluster,
            ar_cluster,
            as_cluster,
            av_cluster,
            hi_cluster,
            hr_cluster,
            hs_cluster,
            hv_cluster,
            ze_cluster,
        }
    }
}

#[derive(Serialize)]
pub struct ClusterOutput {
    pub attitude: Option<ClusterAttitudeOutput>,
    pub cascading: Option<ClusterCascadingOutput>,
    pub opinion_health: Option<ClusterOpinionHealthOutput>,
}

impl ClusterOutput {
    pub fn new(attitude: Option<ClusterAttitudeOutput>, cascading: Option<ClusterCascadingOutput>, opinion_health: Option<ClusterOpinionHealthOutput>) -> Self {
        Self { 
            attitude, 
            cascading, 
            opinion_health, 
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct EpidemicPars {
    pub immunity_decay: f64,
    pub infection_decay: f64,
    pub infection_rate: f64,
    pub nseeds: usize,
    pub r0: f64,
    pub r0_w2: f64,
    pub secondary_outbreak: bool,
    pub seed_model: SeedModel,
    pub vaccination_decay: f64,
    pub vaccination_rate: f64,
}

impl EpidemicPars {
    pub fn new(
        immunity_decay: f64,
        infection_decay: f64,
        infection_rate: f64,
        nseeds: usize,
        r0: f64,
        r0_w2: f64,
        secondary_outbreak: bool,
        seed_model: SeedModel,
        vaccination_decay: f64,
        vaccination_rate: f64,
    ) -> Self {
        Self {
            immunity_decay,
            infection_rate,
            infection_decay,
            nseeds,
            r0,
            r0_w2,
            secondary_outbreak,
            seed_model,
            vaccination_decay,
            vaccination_rate,
        }
    }
}

#[derive(Serialize)]
pub struct GlobalOutput {
    pub active: usize,
    pub convinced_at_peak: usize,
    pub peak_incidence: usize,
    pub prevalence: usize,
    pub time_to_end: usize,
    pub time_to_peak: usize,
    pub vaccinated: usize,
    pub vaccinated_at_peak: usize,
}

impl GlobalOutput {
    pub fn new(
        active: usize,
        convinced_at_peak: usize,
        peak_incidence: usize,
        prevalence: usize,
        time_to_end: usize,
        time_to_peak: usize,
        vaccinated: usize,
        vaccinated_at_peak: usize,
    ) -> Self {
        Self {
            active,
            convinced_at_peak,
            peak_incidence,
            prevalence,
            time_to_end,
            time_to_peak,
            vaccinated,
            vaccinated_at_peak,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct Input {
    pub algorithm: Option<AlgorithmPars>,
    pub epidemic: EpidemicPars,
    pub network: Option<NetworkPars>,
    pub opinion: Option<OpinionPars>,
    pub output: Option<OutputPars>,
    pub vaccination: Option<VaccinationPars>,
}

impl Input {
    pub fn new(
        algorithm: Option<AlgorithmPars>,
        epidemic: EpidemicPars,
        network: Option<NetworkPars>,
        opinion: Option<OpinionPars>,
        output: Option<OutputPars>,
        vaccination: Option<VaccinationPars>,
    ) -> Self {
        Self {
            algorithm,
            epidemic,
            network,
            opinion,
            output,
            vaccination,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct OpinionPars {
    pub active_fraction: f64,
    pub attitude_attribution_model: Option<HesitancyAttributionModel>,
    pub threshold: f64,
    pub zealot_fraction: f64,
}

impl OpinionPars {
    pub fn new(
        active_fraction: f64,
        attitude_attribution_model: Option<HesitancyAttributionModel>,
        threshold: f64,
        zealot_fraction: f64,
    ) -> Self {
        Self {
            active_fraction,
            attitude_attribution_model,
            threshold,
            zealot_fraction,
        }
    }
}

#[derive(Serialize)]
pub struct OutputResults {
    pub agent_ensemble: Option<AgentEnsembleOutput>,
    pub cluster: Option<ClusterOutput>,
    pub global: GlobalOutput,
    pub rebuild: Option<RebuildOutput>,
    pub time: Option<TimeOutput>,
}

impl OutputResults {
    pub fn new(
        agent_ensemble: AgentEnsembleOutput,
        cluster: ClusterOutput,
        global: GlobalOutput,
        rebuild: RebuildOutput,
        time: TimeOutput,
    ) -> Self {
        Self {
            agent_ensemble: Some(agent_ensemble),
            cluster: Some(cluster),
            global,
            rebuild: Some(rebuild),
            time: Some(time),
        }
    }
}

#[derive(Serialize)]
pub struct OutputEnsemble {
    inner: Vec<OutputResults>,
}

impl Default for OutputEnsemble {
    fn default() -> Self {
        Self::new()
    }
}

impl OutputEnsemble {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
        }
    }

    pub fn inner(&self) -> &Vec<OutputResults> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Vec<OutputResults> {
        &mut self.inner
    }

    pub fn number_of_simulations(&self) -> usize {
        self.inner.len()
    }

    pub fn add_outbreak(&mut self, output: OutputResults, n: usize, r0: f64) {
        let cutoff_fraction = PAR_OUTBREAK_PREVALENCE_FRACTION_CUTOFF; 
        let global_prevalence = output.global.prevalence;
        if (r0 > CONST_EPIDEMIC_THRESHOLD) && ((global_prevalence as f64) > cutoff_fraction * (n as f64)) {
            self.inner_mut().push(output)
        }
    }

    pub fn filter_outbreaks(&mut self, input: &Input) {
        let n = input.network.unwrap().size;
        let r0 = input.epidemic.r0;
        let cutoff_fraction = PAR_OUTBREAK_PREVALENCE_FRACTION_CUTOFF;
        let mut s: usize = 0;
        let nsims = self.number_of_simulations() as usize;
        while s < nsims {
            let global_prevalence = self.inner()[s].global.prevalence;
            if (r0 > CONST_EPIDEMIC_THRESHOLD) && ((global_prevalence as f64) < cutoff_fraction * (n as f64)) {
                self.inner_mut().remove(s);
            } else {
                s += 1;
            }
        }
    }

    pub fn assemble_age_observables(
        &mut self, 
        input: &Input,
    ) -> AssembledAgeOutput {
        let nsims = self.number_of_simulations();
        let nagents = input.network.unwrap().size;

        let mut activation_potential = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut active = vec![vec![0; PAR_AGE_GROUPS]; nsims];
        let mut age = vec![vec![0; PAR_AGE_GROUPS]; nsims];
        let mut convinced_when = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut degree = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut infected_when = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut prevalence = vec![vec![0; PAR_AGE_GROUPS]; nsims];
        let mut removed_when = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut vaccinated = vec![vec![0; PAR_AGE_GROUPS]; nsims];
        let mut vaccinated_when = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut zealots = vec![vec![0; PAR_AGE_GROUPS]; nsims];

        for s in 0..nsims {
            for i in 0..nagents {
                let a = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].age.unwrap();
                let status = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].status.unwrap();
                let threshold = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].threshold.unwrap();

                age[s][a] += 1;
                degree[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].degree.unwrap());
                
                activation_potential[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].activation_potential.unwrap());

                if threshold > 1.0 {
                    zealots[s][a] += 1;
                }

                match status {
                    Status::ActRem => {
                        active[s][a] += 1;
                        convinced_when[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].convinced_when.unwrap());
                        infected_when[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].infected_when.unwrap());
                        prevalence[s][a] += 1;
                        removed_when[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].removed_when.unwrap());
                    },
                    Status::ActSus => {
                        active[s][a] += 1;
                        convinced_when[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].convinced_when.unwrap());
                    },
                    Status::ActVac => {
                        active[s][a] += 1;
                        convinced_when[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].convinced_when.unwrap());
                        vaccinated[s][a] += 1;
                        vaccinated_when[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].vaccinated_when.unwrap());
                    },
                    Status::HesRem => {
                        infected_when[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].infected_when.unwrap());
                        prevalence[s][a] += 1;
                        removed_when[s][a].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].removed_when.unwrap());
                    },
                    _ => {},
                };
            }
        }

        AssembledAgeOutput::new(
            activation_potential,
            active, 
            age, 
            convinced_when, 
            degree, 
            infected_when, 
            prevalence, 
            removed_when, 
            vaccinated, 
            vaccinated_when, 
            zealots,
        )
    }

    pub fn assemble_agent_observables(
        &mut self, 
        input: &Input
    ) -> AssembledAgentOutput {
        let nsims = self.number_of_simulations();
        let n = input.network.unwrap().size;
        let mut age = vec![vec![INIT_USIZE; n]; nsims];
        let mut convinced_when = vec![vec![INIT_USIZE; n]; nsims];
        let mut degree = vec![vec![INIT_USIZE; n]; nsims];
        let mut activation_potential = vec![vec![INIT_USIZE; n]; nsims];
        let mut attitude = vec![vec![Attitude::Never; n]; nsims];
        let mut final_active_susceptible = vec![vec![INIT_USIZE; n]; nsims];
        let mut final_prevalence = vec![vec![INIT_USIZE; n]; nsims];
        let mut final_vaccinated = vec![vec![INIT_USIZE; n]; nsims];
        let mut id = vec![vec![INIT_USIZE; n]; nsims];
        let mut infected_by = vec![vec![INIT_USIZE; n]; nsims];
        let mut infected_when = vec![vec![INIT_USIZE; n]; nsims];
        let mut initial_active_susceptible = vec![vec![INIT_USIZE; n]; nsims];
        let mut initial_vaccinated = vec![vec![INIT_USIZE; n]; nsims];
        let mut removed_when = vec![vec![INIT_USIZE; n]; nsims];
        let mut status = vec![vec![INIT_STATUS; n]; nsims];
        let mut threshold = vec![vec![0.0; n]; nsims];
        let mut vaccinated_when = vec![vec![INIT_USIZE; n]; nsims];
        let mut zealots = vec![vec![INIT_USIZE; n]; nsims];
    
        for s in 0..nsims {
            for i in 0..n {
                activation_potential[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].activation_potential.unwrap();
                age[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].age.unwrap();
                attitude[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].attitude.unwrap();
                convinced_when[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].convinced_when.unwrap();
                degree[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].degree.unwrap();
                final_active_susceptible[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].final_active_susceptible.unwrap();
                final_prevalence[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].final_prevalence.unwrap();
                final_vaccinated[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].final_vaccinated.unwrap();
                id[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].id.unwrap();
                infected_by[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].infected_by.unwrap();
                infected_when[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].infected_when.unwrap();
                initial_active_susceptible[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].initial_active_susceptible.unwrap();
                initial_vaccinated[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].initial_vaccinated.unwrap();
                removed_when[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].removed_when.unwrap();
                status[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].status.unwrap();
                threshold[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].threshold.unwrap();
                vaccinated_when[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].vaccinated_when.unwrap();
                zealots[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].zealots.unwrap();
            }
        }

        AssembledAgentOutput::new(
            activation_potential,
            age,
            attitude,
            convinced_when,
            degree, 
            final_active_susceptible,
            final_prevalence,
            final_vaccinated,
            id,
            infected_by, 
            infected_when, 
            initial_active_susceptible,
            initial_vaccinated,
            removed_when, 
            status, 
            threshold, 
            vaccinated_when,
            zealots,
        )
    }

    pub fn assemble_cluster_observables(&mut self) -> AssembledClusterOpinionHealthOutput {
        let nsims = self.number_of_simulations() as usize;
        let mut as_clusters = vec![vec![]; nsims];
        let mut hs_clusters = vec![vec![]; nsims];
        let mut ai_clusters = vec![vec![]; nsims];
        let mut hi_clusters = vec![vec![]; nsims];
        let mut ar_clusters = vec![vec![]; nsims];
        let mut hr_clusters = vec![vec![]; nsims];
        let mut av_clusters = vec![vec![]; nsims];
        let mut hv_clusters = vec![vec![]; nsims];
        let mut ze_clusters = vec![vec![]; nsims];

        for s in 0..nsims {
            as_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().opinion_health.as_ref().unwrap().as_cluster.clone();
            hs_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().opinion_health.as_ref().unwrap().hs_cluster.clone();
            ai_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().opinion_health.as_ref().unwrap().ai_cluster.clone();
            hi_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().opinion_health.as_ref().unwrap().hi_cluster.clone();
            ar_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().opinion_health.as_ref().unwrap().ar_cluster.clone();
            hr_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().opinion_health.as_ref().unwrap().hr_cluster.clone();
            av_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().opinion_health.as_ref().unwrap().av_cluster.clone();
            hv_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().opinion_health.as_ref().unwrap().hv_cluster.clone();
            ze_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().opinion_health.as_ref().unwrap().ze_cluster.clone();
        }

        AssembledClusterOpinionHealthOutput::new(
            ai_clusters, 
            ar_clusters, 
            as_clusters, 
            av_clusters, 
            hi_clusters, 
            hr_clusters, 
            hs_clusters, 
            hv_clusters,
            ze_clusters,
        )
    }

    pub fn assemble_degree_observables(
        &mut self, 
        input: &Input,
    ) -> AssembledDegreeOutput {
        let nsims = self.number_of_simulations();
        let nagents = input.network.unwrap().size;
       
        let mut active = vec![vec![0; nagents - 1]; nsims];
        let mut age = vec![vec![Vec::new(); nagents - 1]; nsims];
        let mut convinced_when = vec![vec![Vec::new(); nagents - 1]; nsims];
        let mut degree = vec![vec![0; nagents - 1]; nsims];
        let mut infected_when = vec![vec![Vec::new(); nagents - 1]; nsims];
        let mut prevalence = vec![vec![0; nagents - 1]; nsims];
        let mut removed_when = vec![vec![Vec::new(); nagents - 1]; nsims];
        let mut vaccinated = vec![vec![0; nagents - 1]; nsims];
        let mut vaccinated_when = vec![vec![Vec::new(); nagents - 1]; nsims];
        let mut zealots = vec![vec![0; nagents - 1]; nsims];

        for s in 0..nsims {
            for i in 0..nagents {
                let a = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].age.unwrap();
                let k = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].degree.unwrap();
                let status = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].status.unwrap();
                let threshold = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].threshold.unwrap();

                age[s][k].push(a);
                degree[s][k] += 1;

                if threshold > 1.0 {
                    zealots[s][k] += 1;
                }

                match status {
                    Status::ActRem => {
                        active[s][k] += 1;
                        convinced_when[s][k].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].convinced_when.unwrap());
                        infected_when[s][k].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].infected_when.unwrap());
                        prevalence[s][k] += 1;
                        removed_when[s][k].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].removed_when.unwrap());
                    },
                    Status::ActSus => {
                        active[s][k] += 1;
                        convinced_when[s][k].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].convinced_when.unwrap());
                    },
                    Status::ActVac => {
                        active[s][k] += 1;
                        convinced_when[s][k].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].convinced_when.unwrap());
                        vaccinated[s][k] += 1;
                        vaccinated_when[s][k].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].vaccinated_when.unwrap());
                    },
                    Status::HesRem => {
                        infected_when[s][k].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].infected_when.unwrap());
                        prevalence[s][k] += 1;
                        removed_when[s][k].push(self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].removed_when.unwrap());
                    },
                    _ => {},
                };
            }
        }

        AssembledDegreeOutput::new(
            active, 
            age, 
            convinced_when, 
            degree, 
            infected_when, 
            prevalence, 
            removed_when, 
            vaccinated, 
            vaccinated_when, 
            zealots
        )
    }

    pub fn assemble_global_observables(&mut self) -> AssembledGlobalOutput {
        let nsims = self.number_of_simulations() as usize;
        let mut convinced = vec![0; nsims];
        let mut convinced_at_peak = vec![0; nsims];
        let mut peak_incidence = vec![0; nsims];
        let mut prevalence = vec![0; nsims];
        let mut time_to_end = vec![0; nsims];
        let mut time_to_peak = vec![0; nsims];
        let mut vaccinated = vec![0; nsims];
        let mut vaccinated_at_peak = vec![0; nsims];

        for s in 0..nsims {
            convinced[s] = self.inner()[s].global.active;
            convinced_at_peak[s] = self.inner()[s].global.convinced_at_peak;
            peak_incidence[s] = self.inner()[s].global.peak_incidence;
            prevalence[s] = self.inner()[s].global.prevalence;
            time_to_end[s] = self.inner()[s].global.time_to_end;
            time_to_peak[s] = self.inner()[s].global.time_to_peak;
            vaccinated[s] = self.inner()[s].global.vaccinated;
            vaccinated_at_peak[s] = self.inner()[s].global.vaccinated_at_peak;
        }

        AssembledGlobalOutput::new(
            convinced,
            convinced_at_peak,
            peak_incidence,
            prevalence,
            time_to_end,
            time_to_peak,
            vaccinated,
            vaccinated_at_peak,
        )
    }

    pub fn assemble_rebuild_observables(
        &mut self, 
        input: &Input,
    ) -> AssembledRebuildOutput {

        let age_distribution = self.rebuild_age_distribution(input);
        let contact_matrix = self.rebuild_contact_matrix(input);
        let degree_distribution = self.rebuild_degree_distribution(input);

        AssembledRebuildOutput::new(
            age_distribution, 
            contact_matrix,
            degree_distribution,
        )
    }

    pub fn assemble_time_series(
        &mut self,
        t_max: usize,
    ) -> AssembledTimeSeriesOutput {
        let nsims = self.number_of_simulations() as usize;
        let mut ai_pop_st = vec![vec![0; t_max]; nsims];
        let mut ar_pop_st = vec![vec![0; t_max]; nsims];
        let mut as_pop_st = vec![vec![0; t_max]; nsims];
        let mut av_pop_st = vec![vec![0; t_max]; nsims];
        let mut hi_pop_st = vec![vec![0; t_max]; nsims];
        let mut hr_pop_st = vec![vec![0; t_max]; nsims];
        let mut hs_pop_st = vec![vec![0; t_max]; nsims];
        let mut hv_pop_st = vec![vec![0; t_max]; nsims];
        let mut t_array_st = vec![vec![0; t_max]; nsims];

        for s in 0..nsims {
            for t in 0..t_max as usize {
                t_array_st[s][t] = self.inner()[s].time.as_ref().unwrap().t_array[t];
                as_pop_st[s][t] = self.inner()[s].time.as_ref().unwrap().as_pop_t[t];
                hs_pop_st[s][t] = self.inner()[s].time.as_ref().unwrap().hs_pop_t[t];
                ai_pop_st[s][t] = self.inner()[s].time.as_ref().unwrap().ai_pop_t[t];
                hi_pop_st[s][t] = self.inner()[s].time.as_ref().unwrap().hi_pop_t[t];
                ar_pop_st[s][t] = self.inner()[s].time.as_ref().unwrap().ar_pop_t[t];
                hr_pop_st[s][t] = self.inner()[s].time.as_ref().unwrap().hr_pop_t[t];
                av_pop_st[s][t] = self.inner()[s].time.as_ref().unwrap().av_pop_t[t];
                hv_pop_st[s][t] = self.inner()[s].time.as_ref().unwrap().hv_pop_t[t];
            }
        }
        
        AssembledTimeSeriesOutput::new(
            t_array_st, 
            ai_pop_st, 
            ar_pop_st,
            as_pop_st,
            av_pop_st,
            hi_pop_st,  
            hr_pop_st, 
            hs_pop_st, 
            hv_pop_st,
        )
    }

    pub fn rebuild_age_distribution(&mut self, input: &Input) -> Vec<Vec<f64>> {
        let nsims = self.number_of_simulations();
        let n = input.network.unwrap().size;
        let mut age_distribution = vec![vec![0.0; PAR_AGE_GROUPS]; nsims];

        for s in 0..nsims {
            for i in 0..n {
                let age_i = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].age.unwrap();
                age_distribution[s][age_i] += 1.0 / n as f64;
            }
        }

        age_distribution
    }

    pub fn rebuild_contact_matrix(&mut self, input: &Input) -> Vec<Vec<Vec<f64>>> {
        let nsims = self.number_of_simulations();
        let n = input.network.unwrap().size;
        let mut age_distribution = vec![vec![0.0; PAR_AGE_GROUPS]; nsims];
        let mut contact_matrix = vec![vec![vec![0.0; PAR_AGE_GROUPS]; PAR_AGE_GROUPS]; nsims];
    
        for s in 0..nsims {
            for i in 0..n {
                let a_i = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].age.unwrap();
                let neighbors = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].neighbors.as_ref().unwrap().clone();
                for j in neighbors {
                    let a_j = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[j].age.unwrap();
                    contact_matrix[s][a_i][a_j] += 1.0;
                }
                age_distribution[s][a_i] += 1.0;
            }

            for a_i in 0..PAR_AGE_GROUPS {
                for a_j in 0..PAR_AGE_GROUPS {
                    contact_matrix[s][a_i][a_j] /= age_distribution[s][a_i];
                }
            }
        }

        contact_matrix
    }

    pub fn rebuild_degree_distribution(&mut self, input: &Input) -> Vec<Vec<f64>> {
        let nsims = self.number_of_simulations();
        let n = input.network.unwrap().size;
        //let k_max = n / 100;
        let mut degree_distribution = vec![vec![0.0; PAR_AGE_GROUPS]; nsims];

        for s in 0..nsims {
            for i in 0..n {
                let k_i = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i].degree.unwrap();
                degree_distribution[s][k_i] += 1.0 / n as f64;
            }
        }

        degree_distribution
    }

    pub fn save_to_pickle(&mut self, pars: &Input) {
        if pars.output.unwrap().age {
            let pars_replica = Input::new(pars.algorithm, pars.epidemic, pars.network, pars.opinion, pars.output, pars.vaccination);
    
            let assembled_age_output = self.assemble_age_observables(&pars_replica);
    
            let output_to_serialize = SerializedAgeAssembly {
                age: assembled_age_output,
                pars: pars_replica,
            };
    
            let serialized = serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();
    
            let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
            let age_string = HEADER_AGE.to_owned() + &exp_string.to_string();
            let file_name = write_file_name(&pars, age_string, true);

            let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
            path.push(FOLDER_RESULTS);
            path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
            std::fs::write(path, serialized).unwrap();
        }
        
        let agent_flag = false;
        if agent_flag {
            let assembled_agent_output = self.assemble_agent_observables(&pars);
    
            if pars.output.unwrap().agent_raw {
                let serialized = serde_pickle::to_vec(
                    &assembled_agent_output, 
                    SerOptions::new(),
                ).unwrap();
                let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
                let agents_string = HEADER_AGENT.to_owned() + &exp_string.to_string();
                let file_name = write_file_name(&pars, agents_string, true);
                
                let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
                path.push(FOLDER_RESULTS);
                path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
                std::fs::write(path, serialized).unwrap();
            } else {
                let agent_stat_package = compute_agent_stats(&assembled_agent_output);
                let asp_serialized = serde_pickle::to_vec(&agent_stat_package, SerOptions::new(),).unwrap();
                let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
                let asp_string = HEADER_AGENT_STATS.to_owned() + &exp_string.to_string();
                let file_name = write_file_name(&pars, asp_string, true);
                
                let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
                path.push(FOLDER_RESULTS);
                path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
                std::fs::write(path, asp_serialized).unwrap();
    
                let agent_distribution = compute_agent_distribution(&assembled_agent_output);
                let ad_serialized = serde_pickle::to_vec(&agent_distribution, SerOptions::new(),).unwrap();
                let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
                let ad_string = HEADER_AGENT_DISTRIBUTION.to_owned() + &exp_string.to_string();
                let file_name = write_file_name(&pars, ad_string, true);
                
                let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
                path.push(FOLDER_RESULTS);
                path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
                std::fs::write(path, ad_serialized).unwrap();
            }
        }
    
        if pars.output.unwrap().cluster {
            let assembled_cluster_output = 
            self.assemble_cluster_observables();
            if pars.output.unwrap().cluster_raw {
                let serialized = serde_pickle::to_vec(
                    &assembled_cluster_output, 
                    SerOptions::new(),
                ).unwrap();
                let cluster_string = HEADER_CLUSTER.to_owned() + &pars.algorithm.unwrap().experiment_id.to_string();
                let file_name = write_file_name(&pars, cluster_string, true);
                
                let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
                path.push(FOLDER_RESULTS);
                path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
                std::fs::write(path, serialized).unwrap();
            } else {
                let cluster_stat_package = compute_cluster_stats(&assembled_cluster_output);
                let csp_serialized = serde_pickle::to_vec(&cluster_stat_package, SerOptions::new(),).unwrap();
                let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
                let csp_string = HEADER_CLUSTER_STATS.to_owned() + &exp_string.to_string();
                let file_name = write_file_name(&pars, csp_string, true);
                let path = FOLDER_RESULTS.to_owned() + &file_name + ".pickle";
                std::fs::write(path, csp_serialized).unwrap();
            
                let cluster_distribution = compute_cluster_distribution(&assembled_cluster_output);
                let cd_serialized = serde_pickle::to_vec(&cluster_distribution, SerOptions::new(),).unwrap();
                let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
                let cd_string = HEADER_CLUSTER_DISTRIBUTION.to_owned() + &exp_string.to_string();
                let file_name = write_file_name(&pars, cd_string, true);
                
                let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
                path.push(FOLDER_RESULTS);
                path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
                std::fs::write(path, cd_serialized).unwrap();
            }
        }

        if pars.output.unwrap().degree {
            let pars_replica = Input::new(pars.algorithm, pars.epidemic, pars.network, pars.opinion, pars.output, pars.vaccination);
    
            let assembled_degree_output = self.assemble_degree_observables(&pars_replica);
    
            let output_to_serialize = SerializedDegreeAssembly {
                degree: assembled_degree_output,
                pars: pars_replica,
            };

            let serialized = serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();
    
            let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
            let global_string = "degree_".to_owned() + &exp_string.to_string();
            let file_name = write_file_name(&pars, global_string, true);
            
            let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
            path.push(FOLDER_RESULTS);
            path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
            std::fs::write(path, serialized).unwrap();
        }
    
        if pars.output.unwrap().global {
            let pars_replica = 
            Input::new(
                pars.algorithm, 
                pars.epidemic, 
                pars.network, 
                pars.opinion, 
                pars.output, 
                pars.vaccination,
            );

            let assembled_global_output = 
            self.assemble_global_observables();
        
            let output_to_serialize = SerializeGlobalAssembly {
                global: assembled_global_output,
                pars: pars_replica,
            };

            let serialized = serde_pickle::to_vec(
                &output_to_serialize, 
                SerOptions::new()
            ).unwrap();
        
            let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
            let global_string = HEADER_GLOBAL.to_owned() + &exp_string.to_string();
            let file_name = write_file_name(&pars, global_string, true);
            
            let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
            path.push(FOLDER_RESULTS);
            path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
            std::fs::write(path, serialized).unwrap();
        }

        if pars.output.unwrap().rebuild {
            let pars_replica = 
            Input::new(
                pars.algorithm, 
                pars.epidemic, 
                pars.network, 
                pars.opinion, 
                pars.output, 
                pars.vaccination,
            );

            if pars.output.unwrap().rebuild_raw {
                let assembled_rebuild_output = self.assemble_rebuild_observables(&pars_replica);
    
                let output_to_serialize = SerializedRebuildAssembly {
                    rebuild: assembled_rebuild_output,
                    pars: pars_replica,
                };
    
                let serialized = serde_pickle::to_vec(
                    &output_to_serialize, 
                    SerOptions::new()
                ).unwrap();
    
                let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
                let global_string = HEADER_REBUILD.to_owned() + &exp_string.to_string();
                let file_name = write_file_name(&pars, global_string, true);

                let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
                path.push(FOLDER_RESULTS);
                path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
                std::fs::write(path, serialized).unwrap();
            } else {
                let stat_rebuild_output = self.stat_rebuild_observables(&pars_replica);
    
                let output_to_serialize = SerializedRebuildStats {
                    rebuild: stat_rebuild_output,
                    pars: pars_replica,
                };
    
                let serialized = serde_pickle::to_vec(
                    &output_to_serialize, 
                    SerOptions::new()
                ).unwrap();
    
                let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
                let global_string = HEADER_REBUILD_STATS.to_owned() + &exp_string.to_string();
                let file_name = write_file_name(&pars, global_string, true);
                
                let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
                path.push(FOLDER_RESULTS);
                path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
                std::fs::write(path, serialized).unwrap();
            }
        }

        if pars.output.unwrap().time {
            let assembled_time_series = self.assemble_time_series(pars.algorithm.unwrap().t_max);

            if pars.output.unwrap().time_raw {
                let serialized = serde_pickle::to_vec(
                    &assembled_time_series, 
                    SerOptions::new()
                ).unwrap();
                let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, pars.vaccination.unwrap().us_state.unwrap());
                let time_string = HEADER_TIME.to_owned() + &exp_string.to_string();
                let file_name = write_file_name(&pars, time_string, true);
                
                let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
                path.push(FOLDER_RESULTS);
                path.push(format!("{}{}", file_name, EXTENSION_RESULTS));
                std::fs::write(path, serialized).unwrap();
            } else {
                todo!()
            }
        }
    }

    pub fn stat_rebuild_observables(&mut self, input: &Input) -> StatRebuildOutput {
        
        let age_distribution = self.rebuild_age_distribution(input);
        let contact_matrix = self.rebuild_contact_matrix(input);
        let degree_distribution = self.rebuild_degree_distribution(input);

        let nsims = age_distribution.len();

        let mut mean_age_distribution = vec![0.0; PAR_AGE_GROUPS];
        let mut mean_contact_matrix = vec![vec![0.0; PAR_AGE_GROUPS]; PAR_AGE_GROUPS];
        let mut mean_degree_distribution = vec![0.0; PAR_AGE_GROUPS];

        for s in 0..nsims {
            for a_i in 0..PAR_AGE_GROUPS {
                for a_j in 0..PAR_AGE_GROUPS {
                    mean_contact_matrix[a_i][a_j] += contact_matrix[s][a_i][a_j] as f64;
                }
                mean_age_distribution[a_i] += age_distribution[s][a_i] as f64;
                mean_degree_distribution[a_i] += degree_distribution[s][a_i] as f64;
            }
        }

        for a_i in 0..PAR_AGE_GROUPS {
            for a_j in 0..PAR_AGE_GROUPS {
                mean_contact_matrix[a_i][a_j] /= nsims as f64;
            }
            mean_age_distribution[a_i] /= nsims as f64;
            mean_degree_distribution[a_i] /= nsims as f64;
        }

        StatRebuildOutput::new(mean_age_distribution, mean_contact_matrix, mean_degree_distribution)
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct OutputPars {
    pub age: bool,
    pub agent: bool,
    pub agent_raw: bool,
    pub cluster: bool,
    pub cluster_raw: bool,
    pub degree: bool,
    pub global: bool,
    pub rebuild: bool,
    pub rebuild_raw: bool,
    pub time: bool,
    pub time_raw: bool,
}

impl OutputPars {
    pub fn new(
        age: bool,
        agent: bool,
        agent_raw: bool,
        cluster: bool,
        cluster_raw: bool,
        degree: bool,
        global: bool,
        rebuild: bool,
        rebuild_raw: bool,
        time: bool,
        time_raw: bool,
    ) -> Self {
        Self {
            age,
            agent,
            agent_raw,
            cluster,
            cluster_raw,
            degree,
            global,
            rebuild,
            rebuild_raw,
            time,
            time_raw,
        }
    }
}

#[derive(Serialize)]
pub struct RebuildOutput {
    age_distribution: Option<Vec<usize>>,
    contact_matrix: Option<Vec<Vec<usize>>>,
    degree_distribution: Option<Vec<usize>>,
}

impl  RebuildOutput {
    pub fn new(
        age_distribution: Vec<usize>, 
        contact_matrix: Vec<Vec<usize>>, 
        degree_distribution: Vec<usize>,
    ) -> Self {
        Self { 
            age_distribution: Some(age_distribution), 
            contact_matrix: Some(contact_matrix), 
            degree_distribution: Some(degree_distribution), 
        }
    }
    
}

#[derive(Serialize)]
pub struct SerializedAgeAssembly {
    pub age: AssembledAgeOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedAgeAssemblyTwoWaves {
    pub age_w1: AssembledAgeOutput,
    pub age_w2: AssembledAgeOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedAgentAssembly {
    pub agent: AssembledAgentOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedAgentAssemblyTwoWaves {
    pub agent_w1: AssembledAgentOutput,
    pub agent_w2: AssembledAgentOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedClusterAssembly {
    pub attitude: Option<AssembledClusterAttitudeOutput>,
    pub cascading: Option<AssembledClusterCascadingOutput>,
    pub opinion_health: Option<AssembledClusterOpinionHealthOutput>,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializeClusterAssemblyTwoWaves {
    pub attitude_w1: Option<AssembledClusterAttitudeOutput>,
    pub attitude_w2: Option<AssembledClusterAttitudeOutput>,
    pub cascading_w1: Option<AssembledClusterCascadingOutput>,
    pub cascading_w2: Option<AssembledClusterCascadingOutput>,
    pub opinion_health_w1: Option<AssembledClusterOpinionHealthOutput>,
    pub opinion_health_w2: Option<AssembledClusterOpinionHealthOutput>,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedDegreeAssembly {
    pub degree: AssembledDegreeOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedDegreeAssemblyTwoWaves {
    pub degree_w1: AssembledDegreeOutput,
    pub degree_w2: AssembledDegreeOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializeGlobalAssembly {
    pub global: AssembledGlobalOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializeGlobalAssemblyTwoWaves {
    pub global_w1: AssembledGlobalOutput,
    pub global_w2: AssembledGlobalOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedRebuildAssembly {
    pub rebuild: AssembledRebuildOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedRebuildStats {
    pub rebuild: StatRebuildOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedTimeSeriesAssembly {
    pub time: AssembledTimeSeriesOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedTimeSeriesAssemblyTwoWaves {
    pub time_w1: AssembledTimeSeriesOutput,
    pub time_w2: AssembledTimeSeriesOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct StatRebuildOutput {
    pub mean_age_distribution: Vec<f64>,
    pub mean_contact_matrix: Vec<Vec<f64>>,
    pub mean_degree_distribution: Vec<f64>,
}

impl StatRebuildOutput {
    pub fn new(
        mean_age_distribution: Vec<f64>, 
        mean_contact_matrix: Vec<Vec<f64>>, 
        mean_degree_distribution: Vec<f64>,
    ) -> Self {
        Self { 
            mean_age_distribution, 
            mean_contact_matrix, 
            mean_degree_distribution 
        }
    }
}

#[derive(Serialize)]
pub struct TimeOutput {
    pub ai_pop_t: Vec<usize>,
    pub ar_pop_t: Vec<usize>,
    pub as_pop_t: Vec<usize>,
    pub av_pop_t: Vec<usize>,
    pub hi_pop_t: Vec<usize>,
    pub hr_pop_t: Vec<usize>,
    pub hs_pop_t: Vec<usize>,
    pub hv_pop_t: Vec<usize>,
    pub t_array: Vec<usize>,
}

impl Default for TimeOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeOutput {
    pub fn new() -> Self {
        let ai_pop_t = Vec::new();
        let ar_pop_t = Vec::new();
        let as_pop_t = Vec::new();
        let av_pop_t = Vec::new();
        let hi_pop_t = Vec::new();
        let hr_pop_t = Vec::new();
        let hs_pop_t = Vec::new();
        let hv_pop_t = Vec::new();
        let t_array = Vec::new();

        Self {
            t_array,
            ai_pop_t,
            ar_pop_t,
            as_pop_t,
            av_pop_t,
            hi_pop_t,
            hr_pop_t,
            hs_pop_t,    
            hv_pop_t,
        }
    }

    pub fn update_time_series(&mut self, t: usize, pop_t: &TimeUnitPop) {
        self.ai_pop_t.push(pop_t.ai_pop);
        self.ar_pop_t.push(pop_t.ar_pop);
        self.as_pop_t.push(pop_t.as_pop);
        self.av_pop_t.push(pop_t.av_pop);
        self.hi_pop_t.push(pop_t.hi_pop);
        self.hr_pop_t.push(pop_t.hr_pop);
        self.hs_pop_t.push(pop_t.hs_pop);
        self.hv_pop_t.push(pop_t.hv_pop);
        self.t_array.push(t);
    }
}

pub struct TimeUnitPop {
    pub ai_pop: usize,
    pub ar_pop: usize,
    pub as_pop: usize,
    pub av_pop: usize,
    pub hi_pop: usize,
    pub hr_pop: usize,
    pub hs_pop: usize,
    pub hv_pop: usize,
}

impl TimeUnitPop {
    pub fn new(
        ai_pop: usize,
        ar_pop: usize,
        as_pop: usize,
        av_pop: usize,
        hi_pop: usize,
        hr_pop: usize,
        hs_pop: usize,
        hv_pop: usize,
    ) -> Self {
        Self {
            ai_pop,
            ar_pop, 
            as_pop, 
            av_pop,
            hi_pop,
            hr_pop,
            hs_pop, 
            hv_pop,
        }
    }
}

#[derive(
    Clone, 
    Copy, 
    Serialize, 
    Deserialize, 
    Display, 
    Debug, 
    clap::ValueEnum, 
    PartialEq, 
    Eq,
)]
pub enum USState {
    #[serde(rename = "alabama")]
    Alabama,
    #[serde(rename = "alaska")]
    Alaska,
    #[serde(rename = "arizona")]
    Arizona,
    #[serde(rename = "arkansas")]
    Arkansas,
    #[serde(rename = "california")]
    California,
    #[serde(rename = "colorado")]
    Colorado,
    #[serde(rename = "connecticut")]
    Connecticut,
    #[serde(rename = "delaware")]
    Delaware,
    #[serde(rename = "district-of-columbia")]
    DistrictOfColumbia,
    #[serde(rename = "florida")]
    Florida,
    #[serde(rename = "georgia")]
    Georgia,
    #[serde(rename = "hawaii")]
    Hawaii,
    #[serde(rename = "idaho")]
    Idaho,
    #[serde(rename = "illinois")]
    Illinois,
    #[serde(rename = "indiana")]
    Indiana,
    #[serde(rename = "iowa")]
    Iowa,
    #[serde(rename = "kansas")]
    Kansas,
    #[serde(rename = "kentucky")]
    Kentucky,
    #[serde(rename = "louisiana")]
    Louisiana,
    #[serde(rename = "maine")]
    Maine,
    #[serde(rename = "maryland")]
    Maryland,
    #[serde(rename = "massachusetts")]
    Massachusetts,
    #[serde(rename = "michigan")]
    Michigan,
    #[serde(rename = "minnesota")]
    Minnesota,
    #[serde(rename = "mississippi")]
    Mississippi,
    #[serde(rename = "missouri")]
    Missouri,
    #[serde(rename = "montana")]
    Montana,
    #[serde(rename = "nebraska")]
    Nebraska,
    #[serde(rename = "nevada")]
    Nevada,
    #[serde(rename = "new-hampshire")]
    NewHampshire,
    #[serde(rename = "new-jersey")]
    NewJersey,
    #[serde(rename = "new-mexico")]
    NewMexico,
    #[serde(rename = "new-york")]
    NewYork,
    #[serde(rename = "north-carolina")]
    NorthCarolina,
    #[serde(rename = "north-dakota")]
    NorthDakota,
    #[serde(rename = "ohio")]
    Ohio,
    #[serde(rename = "oklahoma")]
    Oklahoma,
    #[serde(rename = "oregon")]
    Oregon,
    #[serde(rename = "pennsylvania")]
    Pennsylvania,
    #[serde(rename = "rhode-island")]
    RhodeIsland,
    #[serde(rename = "south-carolina")]
    SouthCarolina,
    #[serde(rename = "south-dakota")]
    SouthDakota,
    #[serde(rename = "tennessee")]
    Tennessee,
    #[serde(rename = "texas")]
    Texas,
    #[serde(rename = "utah")]
    Utah,
    #[serde(rename = "vermont")]
    Vermont,
    #[serde(rename = "virginia")]
    Virginia,
    #[serde(rename = "washington")]
    Washington,
    #[serde(rename = "west-virginia")]
    WestVirginia,
    #[serde(rename = "wisconsin")]
    Wisconsin,
    #[serde(rename = "wyoming")]
    Wyoming,
    #[serde(rename = "national")]
    National,
}

#[derive(Debug, Deserialize, Serialize)]
struct VaccinationData {
    #[serde(rename = "state")]
    state_name: String,
    fractions: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct VaccinationPars {
    pub age_threshold: usize,
    pub already: f64,
    pub hesitancy_attribution: HesitancyAttributionModel,
    pub majority: f64,
    pub never: f64,
    pub someone: f64,
    pub soon: f64,
    pub underage_correction: bool,
    pub us_state: Option<USState>,
    pub vaccination_decay: f64,
    pub vaccination_policy: VaccinationPolicy,
    pub vaccination_quota: f64,
    pub vaccination_rate: f64,
} 

impl VaccinationPars {
    pub fn new(
        age_threshold: usize,
        already: f64,
        hesitancy_attribution: HesitancyAttributionModel,
        majority: f64,
        never: f64,
        someone: f64,
        soon: f64,
        underage_correction: bool,
        us_state: Option<USState>,
        vaccination_decay: f64,
        vaccination_policy: VaccinationPolicy,
        vaccination_quota: f64,
        vaccination_rate: f64,
    ) -> Self {
        Self {
            age_threshold,
            already,
            hesitancy_attribution,
            majority,
            never,
            someone,
            soon,
            underage_correction,
            us_state,
            vaccination_decay,
            vaccination_policy,
            vaccination_quota,
            vaccination_rate,
        }
    }
}

///////////////////

#[derive(Debug, Clone, Serialize)]
pub struct AgentDistributionPacker {
    convinced_when_dist: AgentDistribution,
    degree_dist: AgentDistribution,
    frac_final_active_dist: AgentDistribution,
    frac_final_prevalence_dist: AgentDistribution,
    frac_final_vaccinated_dist: AgentDistribution,
    frac_initial_active_susceptible_dist: AgentDistribution,
    frac_initial_vaccinated_dist: AgentDistribution,
    frac_zealots_dist: AgentDistribution,
    infected_when_dist: AgentDistribution,
    removed_when_dist: AgentDistribution,
    vaccinated_when_dist: AgentDistribution,
}

#[derive(Debug, Clone, Serialize)]
struct AgentDistribution {
    pub bin_counts: HashMap<usize, usize>,
    pub bin_edges: Vec<usize>,
}

impl AgentDistribution {
    pub fn new(bin_edges: Vec<usize>) -> Self {
        let mut bin_counts = HashMap::new();
        for &edge in &bin_edges {
            bin_counts.insert(edge, 0);
        }
        Self { bin_counts, bin_edges }
    }

    pub fn add_value(&mut self, value: f64) {
        let bin_index = self.find_bin_index(value);
        if let Some(bin_edge) = self.bin_edges.get(bin_index) {
            let count = self.bin_counts.get_mut(bin_edge).unwrap();
            *count += 1;
        }
    }

    fn find_bin_index(&self, value: f64) -> usize {
        for (i, &edge) in self.bin_edges.iter().enumerate() {
            if value <= edge as f64 {
                return i;
            }
        }
        self.bin_edges.len() - 1
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentStatPacker {
    convinced_when_stats: StatPacker,
    degree_stats: StatPacker,
    final_active_stats: StatPacker,
    final_prevalence_stats: StatPacker,
    final_vaccinated_stats: StatPacker,
    infected_when_stats: StatPacker,
    initial_active_susceptible_stats: StatPacker,
    initial_vaccinated_stats: StatPacker,
    removed_when_stats: StatPacker,
    vaccinated_when_stats: StatPacker,
    zealots_stats: StatPacker,
}

#[derive(Debug, Clone, Serialize)]
pub struct ClusterDistribution {
    pub count_by_size: HashMap<usize, usize>,
}

impl ClusterDistribution {
    pub fn new() -> Self {
        Self {
            count_by_size: HashMap::new(),
        }
    }

    pub fn add_cluster(&mut self, size: usize) {
        let count = self.count_by_size.entry(size).or_insert(0);
        *count += 1;
    }
}

#[derive(Debug, Serialize)]
pub struct ClusterDistributionPacker {
    pub ai_cluster_dist: ClusterDistribution,
    pub ar_cluster_dist: ClusterDistribution,
    pub as_cluster_dist: ClusterDistribution,
    pub av_cluster_dist: ClusterDistribution,
    pub hi_cluster_dist: ClusterDistribution,
    pub hr_cluster_dist: ClusterDistribution,
    pub hs_cluster_dist: ClusterDistribution,
    pub hv_cluster_dist: ClusterDistribution,
}

#[derive(Clone, Copy, Serialize)]
struct ClusterStatsOverSims {
    pub average_size: StatPacker,
    pub max_size: StatPacker,
    pub number: StatPacker,
}

impl ClusterStatsOverSims {
    pub fn new(
        average_size: StatPacker, 
        max_size: StatPacker, 
        number: StatPacker,
    ) -> Self {
        Self {
            average_size,
            max_size,
            number,
        }
    }
}

#[derive(Clone, Copy, Serialize)]
pub struct ClusterStatPacker {
    ai_avg_size_stats: StatPacker,
    ai_max_size_stats: StatPacker,
    ai_number_stats: StatPacker,
    ar_avg_size_stats: StatPacker,
    ar_max_size_stats: StatPacker,
    ar_number_stats: StatPacker,
    as_avg_size_stats: StatPacker,
    as_max_size_stats: StatPacker,
    as_number_stats: StatPacker,
    av_avg_size_stats: StatPacker,
    av_max_size_stats: StatPacker,
    av_number_stats: StatPacker,
    hi_avg_size_stats: StatPacker,
    hi_max_size_stats: StatPacker,
    hi_number_stats: StatPacker,
    hr_avg_size_stats: StatPacker,
    hr_max_size_stats: StatPacker,
    hr_number_stats: StatPacker,
    hs_avg_size_stats: StatPacker,
    hs_max_size_stats: StatPacker,
    hs_number_stats: StatPacker,
    hv_avg_size_stats: StatPacker,
    hv_max_size_stats: StatPacker,
    hv_number_stats: StatPacker,
}

#[derive(Debug, Clone, Copy, Serialize)]
struct StatPacker {
    pub mean: f64,
    pub std: f64,
    pub l95: f64,
    pub u95: f64,
    pub min: f64,
    pub max: f64,
}

impl StatPacker {
    pub fn new(
        mean: f64,
        std: f64,
        l95: f64,
        u95: f64,
        min: f64,
        max: f64,
    ) -> Self {
        Self {
            mean,
            std,
            l95,
            u95,
            min,
            max,
        }
    }
}

fn calculate_cluster_sim_stats(
    simulations: &[Vec<usize>],
) -> ClusterStatsOverSims {
    let mut avg_sizes = Vec::new();
    let mut max_sizes = Vec::new();
    let mut numbers = Vec::new();

    for sim_vector in simulations {
        let avg_size = sim_vector.iter().sum::<usize>() as f64 / sim_vector.len() as f64;
        let max_size = *sim_vector.iter().max().unwrap_or(&0) as f64;
        let number = sim_vector.len() as f64;

        avg_sizes.push(avg_size);
        max_sizes.push(max_size);
        numbers.push(number);
    }

    let avg_stats = calculate_stat_pack(&avg_sizes);
    let max_stats = calculate_stat_pack(&max_sizes);
    let number_stats = calculate_stat_pack(&numbers);

    ClusterStatsOverSims::new(avg_stats, max_stats, number_stats)
}

fn calculate_stat_pack(values: &[f64]) -> StatPacker {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    let std = variance.sqrt();
    let z = 1.96; // 95% confidence interval

    let l95 = mean - z * (std / (values.len() as f64).sqrt());
    let u95 = mean + z * (std / (values.len() as f64).sqrt());

    let min = *values.iter().min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal)).unwrap();
    let max = *values.iter().max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal)).unwrap();

    StatPacker::new(mean, std, l95, u95, min, max)
}

pub fn compute_agent_distribution(
    assembled_agent_output: &AssembledAgentOutput,
) -> AgentDistributionPacker {
    let convinced_when_seq_s = &assembled_agent_output.convinced_when;
    let degree_seq_s = &assembled_agent_output.degree;
    let final_active_seq_s = &assembled_agent_output.final_active_susceptible;
    let final_prevalence_seq_s = &assembled_agent_output.final_prevalence;
    let final_vaccinated_seq_s = &assembled_agent_output.final_vaccinated;
    let infected_when_seq_s = &assembled_agent_output.infected_when;
    let initial_active_susceptible_seq_s = &assembled_agent_output.initial_active_susceptible;
    let initial_vaccinated_seq_s = &assembled_agent_output.initial_vaccinated;
    let removed_when_seq_s = &assembled_agent_output.removed_when;
    let vaccinated_when_seq_s = &assembled_agent_output.vaccinated_when;
    let zealots_seq_s = &assembled_agent_output.zealots;

    let frac_final_active = compute_fractions(final_active_seq_s, degree_seq_s);
    let frac_final_prevalence = compute_fractions(final_prevalence_seq_s, degree_seq_s);
    let frac_final_vaccinated = compute_fractions(final_vaccinated_seq_s, degree_seq_s);
    let frac_initial_active_susceptible = compute_fractions(initial_active_susceptible_seq_s, degree_seq_s);
    let frac_initial_vaccinated = compute_fractions(initial_vaccinated_seq_s, degree_seq_s);
    let frac_zealots = compute_fractions(zealots_seq_s, degree_seq_s);

    let convinced_when_dist = compute_agent_sim_distribution(&convinced_when_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let degree_dist = compute_agent_sim_distribution(&degree_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let frac_final_active_dist = compute_agent_sim_distribution(&frac_final_active);
    let frac_final_prevalence_dist = compute_agent_sim_distribution(&frac_final_prevalence);
    let frac_final_vaccinated_dist = compute_agent_sim_distribution(&frac_final_vaccinated);
    let infected_when_dist = compute_agent_sim_distribution(&infected_when_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let frac_initial_active_susceptible_dist = compute_agent_sim_distribution(&frac_initial_active_susceptible);
    let frac_initial_vaccinated_dist = compute_agent_sim_distribution(&frac_initial_vaccinated);
    let removed_when_dist = compute_agent_sim_distribution(&removed_when_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let vaccinated_when_dist = compute_agent_sim_distribution(&vaccinated_when_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let frac_zealots_dist = compute_agent_sim_distribution(&frac_zealots);

    AgentDistributionPacker {
        convinced_when_dist,
        degree_dist,
        frac_final_active_dist,
        frac_final_prevalence_dist,
        frac_final_vaccinated_dist,
        infected_when_dist,
        frac_initial_active_susceptible_dist ,
        frac_initial_vaccinated_dist,
        removed_when_dist,
        vaccinated_when_dist,
        frac_zealots_dist,
    }
}

fn compute_agent_sim_distribution(
    simulations: &Vec<Vec<f64>>,
) -> AgentDistribution {
    let mut all_values = Vec::new();
    for sim_vector in simulations {
        all_values.extend_from_slice(sim_vector);
    }

    let min_value = *all_values.iter().min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)).unwrap() as usize;
    let max_value = *all_values.iter().max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)).unwrap() as usize;

    let bin_size = (max_value - min_value) / PAR_NBINS as usize;
    let bin_edges: Vec<usize> = (0..=PAR_NBINS).map(|i| min_value + i * bin_size).collect();

    let mut distribution = AgentDistribution::new(bin_edges.clone());

    for sim_vector in simulations {
        for &value in sim_vector {
            distribution.add_value(value);
        }
    }

    distribution
}

pub fn compute_agent_stats(
    assembled_agent_output: &AssembledAgentOutput,
) -> AgentStatPacker {
    let convinced_when_seq_s = &assembled_agent_output.convinced_when;
    let degree_seq_s = &assembled_agent_output.degree;
    let final_active_seq_s = &assembled_agent_output.final_active_susceptible;
    let final_prevalence_seq_s = &assembled_agent_output.final_prevalence;
    let final_vaccinated_seq_s = &assembled_agent_output.final_vaccinated;
    let infected_when_seq_s = &assembled_agent_output.infected_when;
    let initial_active_susceptible_seq_s = &assembled_agent_output.initial_active_susceptible;
    let initial_vaccinated_seq_s = &assembled_agent_output.initial_vaccinated;
    let removed_when_seq_s = &assembled_agent_output.removed_when;
    let vaccinated_when_seq_s = &assembled_agent_output.vaccinated_when;
    let zealots_seq_s = &assembled_agent_output.zealots;

    let frac_final_active = compute_fractions(final_active_seq_s, degree_seq_s);
    let frac_final_prevalence = compute_fractions(final_prevalence_seq_s, degree_seq_s);
    let frac_final_vaccinated = compute_fractions(final_vaccinated_seq_s, degree_seq_s);
    let frac_initial_active_susceptible = compute_fractions(initial_active_susceptible_seq_s, degree_seq_s);
    let frac_initial_vaccinated = compute_fractions(initial_vaccinated_seq_s, degree_seq_s);
    let frac_zealots = compute_fractions(zealots_seq_s, degree_seq_s);

    let convinced_when_stats = compute_stats(&convinced_when_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let degree_stats = compute_stats(&degree_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let final_active_stats = compute_stats(&frac_final_active);
    let final_prevalence_stats = compute_stats(&frac_final_prevalence);
    let final_vaccinated_stats = compute_stats(&frac_final_vaccinated);
    let infected_when_stats = compute_stats(&infected_when_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let initial_active_susceptible_stats = compute_stats(&frac_initial_active_susceptible);
    let initial_vaccinated_stats = compute_stats(&frac_initial_vaccinated);
    let removed_when_stats = compute_stats(&removed_when_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let vaccinated_when_stats = compute_stats(&vaccinated_when_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let zealots_stats = compute_stats(&frac_zealots);

    AgentStatPacker {
        convinced_when_stats,
        degree_stats,
        final_active_stats,
        final_prevalence_stats,
        final_vaccinated_stats,
        infected_when_stats,
        initial_active_susceptible_stats,
        initial_vaccinated_stats,
        removed_when_stats,
        vaccinated_when_stats,
        zealots_stats,
    }
}

pub fn compute_cluster_distribution(
    assembled_cluster_opinion_health_output: &AssembledClusterOpinionHealthOutput,
) -> ClusterDistributionPacker {
    let ai_cluster_s = &assembled_cluster_opinion_health_output.ai_cluster;
    let ar_cluster_s = &assembled_cluster_opinion_health_output.ar_cluster;
    let as_cluster_s = &assembled_cluster_opinion_health_output.as_cluster;
    let av_cluster_s = &assembled_cluster_opinion_health_output.av_cluster;
    let hi_cluster_s = &assembled_cluster_opinion_health_output.hi_cluster;
    let hr_cluster_s = &assembled_cluster_opinion_health_output.hr_cluster;
    let hs_cluster_s = &assembled_cluster_opinion_health_output.hs_cluster;
    let hv_cluster_s = &assembled_cluster_opinion_health_output.hv_cluster;

    let ai_cluster_dist = compute_cluster_sim_distribution(ai_cluster_s);
    let ar_cluster_dist = compute_cluster_sim_distribution(ar_cluster_s);
    let as_cluster_dist = compute_cluster_sim_distribution(as_cluster_s);
    let av_cluster_dist = compute_cluster_sim_distribution(av_cluster_s);
    let hi_cluster_dist = compute_cluster_sim_distribution(hi_cluster_s);
    let hr_cluster_dist = compute_cluster_sim_distribution(hr_cluster_s);
    let hs_cluster_dist = compute_cluster_sim_distribution(hs_cluster_s);
    let hv_cluster_dist = compute_cluster_sim_distribution(hv_cluster_s);

    ClusterDistributionPacker {
        ai_cluster_dist,
        ar_cluster_dist,
        as_cluster_dist,
        av_cluster_dist,
        hi_cluster_dist,
        hr_cluster_dist,
        hs_cluster_dist,
        hv_cluster_dist,
    }
}

fn compute_cluster_sim_distribution(
    simulations: &[Vec<usize>],
) -> ClusterDistribution {
    let mut distribution = ClusterDistribution::new();

    for sim_vector in simulations {
        for &cluster_size in sim_vector {
            distribution.add_cluster(cluster_size);
        }
    }

    distribution
}

pub fn compute_cluster_stats(
    assembled_cluster_opinion_health_output: &AssembledClusterOpinionHealthOutput,
) -> ClusterStatPacker {
    let ai_cluster_s = &assembled_cluster_opinion_health_output.ai_cluster;
    let ar_cluster_s = &assembled_cluster_opinion_health_output.ar_cluster;
    let as_cluster_s = &assembled_cluster_opinion_health_output.as_cluster;
    let av_cluster_s = &assembled_cluster_opinion_health_output.av_cluster;
    let hi_cluster_s = &assembled_cluster_opinion_health_output.hi_cluster;
    let hr_cluster_s = &assembled_cluster_opinion_health_output.hr_cluster;
    let hs_cluster_s = &assembled_cluster_opinion_health_output.hs_cluster;
    let hv_cluster_s = &assembled_cluster_opinion_health_output.hv_cluster;

    let ai_stats = calculate_cluster_sim_stats(ai_cluster_s);
    let ar_stats = calculate_cluster_sim_stats(ar_cluster_s);
    let as_stats = calculate_cluster_sim_stats(as_cluster_s);
    let av_stats = calculate_cluster_sim_stats(av_cluster_s);
    let hi_stats = calculate_cluster_sim_stats(hi_cluster_s);
    let hr_stats = calculate_cluster_sim_stats(hr_cluster_s);
    let hs_stats = calculate_cluster_sim_stats(hs_cluster_s);
    let hv_stats = calculate_cluster_sim_stats(hv_cluster_s);

    ClusterStatPacker {
        ai_avg_size_stats: ai_stats.average_size,
        ar_avg_size_stats: ar_stats.average_size,
        as_avg_size_stats: as_stats.average_size,
        av_avg_size_stats: av_stats.average_size,
        hi_avg_size_stats: hi_stats.average_size,
        hr_avg_size_stats: hr_stats.average_size,
        hs_avg_size_stats: hs_stats.average_size,
        hv_avg_size_stats: hv_stats.average_size,
        ai_max_size_stats: ai_stats.max_size,
        ar_max_size_stats: ar_stats.max_size,
        as_max_size_stats: as_stats.max_size,
        av_max_size_stats: av_stats.max_size,
        hi_max_size_stats: hi_stats.max_size,
        hr_max_size_stats: hr_stats.max_size,
        hs_max_size_stats: hs_stats.max_size,
        hv_max_size_stats: hv_stats.max_size,
        ai_number_stats: ai_stats.number,
        ar_number_stats: ar_stats.number,
        as_number_stats: as_stats.number,
        av_number_stats: av_stats.number,
        hi_number_stats: hi_stats.number,
        hr_number_stats: hr_stats.number,
        hs_number_stats: hs_stats.number,
        hv_number_stats: hv_stats.number,
    }
}

fn compute_fractions(
    numerators: &Vec<Vec<usize>>,
    denominators: &Vec<Vec<usize>>,
) -> Vec<Vec<f64>> {
    numerators
        .iter()
        .zip(denominators.iter())
        .map(|(numerator_vec, denominator_vec)| {
            numerator_vec
                .iter()
                .zip(denominator_vec.iter())
                .map(|(numerator, denominator)| {
                    if *denominator == 0 {
                        0.0
                    } else {
                        *numerator as f64 / *denominator as f64
                    }
                })
                .collect()
        })
        .collect()
}

fn compute_stats(values: &Vec<Vec<f64>>) -> StatPacker {
    let mut flat_values = Vec::new();

    for inner_vec in values {
        for &value in inner_vec {
            flat_values.push(value);
        }
    }

    let mean = flat_values.iter().sum::<f64>() / flat_values.len() as f64;
    let variance = flat_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (flat_values.len() - 1) as f64;
    let std = variance.sqrt();
    let z = 1.96; // 95% confidence interval

    let l95 = mean - z * (std / (flat_values.len() as f64).sqrt());
    let u95 = mean + z * (std / (flat_values.len() as f64).sqrt());

    let min = *flat_values.iter().min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal)).unwrap();
    let max = *flat_values.iter().max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal)).unwrap();

    StatPacker::new(mean, std, l95, u95, min, max)
}

fn convert_to_f64(vec: &Vec<usize>) -> Vec<f64> {
    vec.iter().map(|&value| value as f64).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalization() {
        let mut values = vec![2.0, 3.0, 5.0]; // Using non-normalized values for clear testing
        let cdf = build_normalized_cdf(&mut values);

        // Check if the values are normalized
        let sum: f64 = values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check the correctness of the cdf calculation
        // Expected cdf: [2/10, 5/10, 10/10] = [0.2, 0.5, 1.0]
        assert_eq!(cdf, vec![0.2, 0.5, 1.0]);
    }

    #[test]
    fn test_already_normalized() {
        let mut values = vec![0.25, 0.25, 0.25, 0.25];
        let cdf = build_normalized_cdf(&mut values);

        // Check if the values remain the same after the function call
        let expected_values = vec![0.25, 0.25, 0.25, 0.25];
        assert_eq!(values, expected_values);

        // Check the correctness of the cdf calculation
        // Expected cdf: [0.25, 0.5, 0.75, 1.0]
        assert_eq!(cdf, vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_empty_input() {
        let mut values: Vec<f64> = vec![];
        let cdf = build_normalized_cdf(&mut values);
        assert!(cdf.is_empty());
    }

    #[test]
    fn test_cdf_output() {
        let mut values = vec![0.2, 0.3, 0.5];
        let cdf = build_normalized_cdf(&mut values);
        assert_eq!(cdf, vec![0.2, 0.5, 1.0]);
    }

    #[test]
    fn test_standard_case() {
        let contact_matrix = vec![vec![1.0, 1.0], vec![2.0, 2.0]];
        let probabilities = compute_interlayer_probability_matrix(&contact_matrix);
        assert_eq!(probabilities, vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
    }

    #[test]
    fn test_empty_matrix() {
        let contact_matrix: Vec<Vec<f64>> = vec![];
        let probabilities = compute_interlayer_probability_matrix(&contact_matrix);
        assert!(probabilities.is_empty());
    }

    #[test]
    fn test_uniform_matrix() {
        let contact_matrix = vec![vec![2.0, 2.0], vec![2.0, 2.0]];
        let probabilities = compute_interlayer_probability_matrix(&contact_matrix);
        assert_eq!(probabilities, vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
    }

    #[test]
    fn test_zero_row_sum() {
        let contact_matrix = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let probabilities = compute_interlayer_probability_matrix(&contact_matrix);
        // This test depends on how you want to handle the division by zero case.
        // Adjust the expected result based on your specific implementation needs.
        assert_eq!(probabilities, vec![vec![0.0, 0.0], vec![0.5, 0.5]]);
    }

    #[test]
    fn test_all_zeros_cdf() {
        let cdf = vec![0.0, 0.0, 1.0];
        let index = sample_from_cdf(&cdf);
        assert_eq!(index, 2);
    }

    #[test]
    fn test_linear_cdf() {
        let cdf = vec![0.0, 0.5, 1.0];
        // This test might not always pass due to randomness, but it should mostly return 1 or 2.
        let index = sample_from_cdf(&cdf);
        assert!(index >= 1 && index <= 2);
    }

    #[test]
    fn test_single_step_cdf() {
        let cdf = vec![0.0, 0.0, 0.0, 1.0];
        let index = sample_from_cdf(&cdf);
        assert_eq!(index, 3);
    }

    #[test]
    fn test_r0_less_than_one() {
        let r0 = 0.5;
        let sus0 = 1.0;
        let r_inf = sir_prevalence(r0, sus0);
        assert!(r_inf < 0.01); // Expect a low final size
    }

    #[test]
    fn test_r0_greater_than_one() {
        let r0 = 2.0;
        let sus0 = 1.0;
        let r_inf = sir_prevalence(r0, sus0);
        assert!(r_inf > 0.5); // Expect a high final size
    }

    #[test]
    fn test_full_susceptibility() {
        let r0 = 1.5;
        let sus0 = 1.0;
        let r_inf = sir_prevalence(r0, sus0);
        // The expected value should be based on known results or more detailed calculations
        assert!(r_inf > 0.7 && r_inf < 1.0); // Adjust the range based on expected outcomes
    }
}
