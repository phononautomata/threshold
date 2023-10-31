use std::cmp::Ordering;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::collections::{HashSet, HashMap};
use std::{vec, env};
use serde::{Serialize, Deserialize};
use serde_json::Value;
use strum::Display;

use netrust::utils::{NetworkPars, NetworkModel, NetworkParsEnum};
use crate::agent::{Status, AgentEnsemble, SeedModel};

const DEFAULT_VALUE: usize = 9999999;

pub fn convert_hm_value_to_bool(
    hash_map: HashMap<String, Value>
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
    hash_map: HashMap<String, Value>
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

pub fn load_json_data(filename: &str) -> HashMap<String, Value> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("config");
    path.push(format!("{}.json", filename));

    // Open the file
    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    // Parse the JSON data into a HashMap
    let data: HashMap<String, Value> = 
    serde_json::from_str(&contents).unwrap();

    data
}

pub fn remove_duplicates(vec: Vec<usize>) -> Vec<usize> {
    let set: HashSet<_> = vec.into_iter().collect();
    set.into_iter().collect()
}

pub fn select_network_model(
    net_id: NetworkModel
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

// Input-related structs & implementations
#[derive(Clone, Copy, Serialize)]
pub struct OpinionPars {
    pub active_fraction: f64,
    pub threshold: f64,
    pub zealot_fraction: f64,
}

impl OpinionPars {
    pub fn new(
        active_fraction: f64,
        threshold: f64,
        zealot_fraction: f64,
    ) -> Self {
        Self {
            active_fraction,
            threshold,
            zealot_fraction,
        }
    }
}

#[derive(Clone, Copy, Serialize)]
pub struct EpidemicPars {
    pub immunity_decay: f64,
    pub infection_rate: f64,
    pub infection_decay: f64,
    pub vaccination_decay: f64,
    pub vaccination_rate: f64,
    pub secondary_outbreak: bool,
    pub seed_model: SeedModel,
    pub seeds: usize,
    pub r0: f64,
    pub r0_w2: f64,
}

impl EpidemicPars {
    pub fn new(
        immunity_decay: f64,
        infection_rate: f64,
        infection_decay: f64,
        r0: f64,
        r0_w2: f64,
        secondary_outbreak: bool,
        seed_model: SeedModel,
        seeds: usize,
        vaccination_decay: f64,
        vaccination_rate: f64,
    ) -> Self {
        Self {
            immunity_decay,
            infection_rate,
            infection_decay,
            secondary_outbreak,
            seed_model,
            seeds,
            r0,
            r0_w2,
            vaccination_decay,
            vaccination_rate,
        }
    }
}

#[derive(Serialize, Clone, Copy)]
pub struct VaccinationPars {
    pub state: USState,
    pub vaccination_decay: f64,
    pub vaccination_rate: f64,
    pub already: f64,
    pub soon: f64,
    pub someone: f64,
    pub majority: f64,
    pub never: f64,
} 

impl VaccinationPars {
    pub fn new(
        vaccination_rate: f64,
        vaccination_decay: f64,
        state: USState,
        already: f64,
        soon: f64,
        someone: f64,
        majority: f64,
        never: f64,
    ) -> Self {
        Self {
            state,
            vaccination_decay,
            vaccination_rate,
            already,
            soon,
            someone,
            majority,
            never,
        }
    }
}

#[derive(Clone, Copy, Serialize)]
pub struct AlgorithmPars {
    pub nsims_dyn: usize,
    pub nsims_net: usize,
    pub t_max: usize,
}

impl AlgorithmPars {
    pub fn new(
        nsims_dyn: usize,
        nsims_net: usize,
        t_max: usize,
    ) -> Self {
        Self {
            nsims_dyn,
            nsims_net,
            t_max, 
        }
    }
}

#[derive(Clone, Copy, Serialize)]
pub struct OutputPars {
    pub agent: bool,
    pub agent_raw: bool,
    pub cluster: bool,
    pub cluster_raw: bool,
    pub time: bool,
    pub time_raw: bool,
}

impl OutputPars {
    pub fn new(
        agent: bool,
        agent_raw: bool,
        cluster: bool,
        cluster_raw: bool,
        time: bool,
        time_raw: bool,
    ) -> Self {
        Self {
            agent,
            agent_raw,
            cluster,
            cluster_raw,
            time,
            time_raw,
        }
    }
}

#[derive(Serialize)]
pub struct Input {
    pub network: NetworkPars,
    pub opinion: OpinionPars,
    pub epidemic: EpidemicPars,
    pub vaccination: Option<VaccinationPars>,
    pub algorithm: AlgorithmPars,
    pub output: OutputPars,
}

impl Input {
    pub fn new(
        network: NetworkPars,
        opinion: OpinionPars,
        epidemic: EpidemicPars,
        vaccination: Option<VaccinationPars>,
        algorithm: AlgorithmPars,
        output: OutputPars,
    ) -> Self {
        Self {
            network,
            opinion,
            epidemic,
            vaccination,
            algorithm,
            output,
        }
    }
}

// Output-related structs & implementations

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

#[derive(Serialize)]
pub struct ClusterOutput {
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

impl ClusterOutput {
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
pub struct AgentOutput {
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
    pub removed_when: Option<usize>,
    pub status: Option<Status>,
    pub threshold: Option<f64>,
    pub vaccinated_when: Option<usize>,
    pub zealots: Option<usize>,
}

impl AgentOutput {
    pub fn new(
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
        removed_when: Option<usize>,
        status: Option<Status>,
        threshold: Option<f64>,
        vaccinated_when: Option<usize>,
        zealots: Option<usize>,
    ) -> Self {
        Self { 
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
pub struct AgentEnsembleOutput {
    inner: Vec<AgentOutput>,
}

impl AgentEnsembleOutput {
    pub fn new(agent_ensemble: &AgentEnsemble) -> Self {
        let mut agent_ensemble_output = Self { inner: Vec::new() };
        
        for agent in agent_ensemble.inner() {
            let convinced_when = agent.convinced_when.unwrap_or(DEFAULT_VALUE);
            let degree = agent.neighbors.as_ref().unwrap().len();
            let final_active_susceptible = agent.final_active_susceptible.unwrap_or(DEFAULT_VALUE);
            let final_vaccinated = agent.final_vaccinated.unwrap_or(DEFAULT_VALUE);
            let final_prevalence = agent.final_prevalence.unwrap_or(DEFAULT_VALUE);
            let id = agent.id;
            let infected_by = agent.infected_by.unwrap_or(DEFAULT_VALUE);
            let infected_when = agent.infected_when.unwrap_or(DEFAULT_VALUE);
            let initial_active_susceptible = agent.initial_active_susceptible.unwrap_or(DEFAULT_VALUE);
            let initial_vaccinated = agent.initial_vaccinated.unwrap_or(DEFAULT_VALUE);
            let removed_when = agent.removed_when.unwrap_or(DEFAULT_VALUE);
            let status = agent.status;
            let threshold = agent.threshold;
            let vaccinated_when = agent.vaccinated_when.unwrap_or(DEFAULT_VALUE);
            let zealots = agent.zealots.unwrap_or(DEFAULT_VALUE);
    
            let agent_output = AgentOutput::new(
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

pub struct TimeUnitPop {
    pub as_pop: usize,
    pub hs_pop: usize,
    pub ai_pop: usize,
    pub hi_pop: usize,
    pub ar_pop: usize,
    pub hr_pop: usize,
    pub av_pop: usize,
    pub hv_pop: usize,
}

impl TimeUnitPop {
    pub fn new(
        as_pop: usize,
        hs_pop: usize,
        ai_pop: usize,
        hi_pop: usize,
        ar_pop: usize,
        hr_pop: usize,
        av_pop: usize,
        hv_pop: usize,
    ) -> Self {
        Self { 
            as_pop, 
            hs_pop, 
            ai_pop, 
            hi_pop, 
            ar_pop, 
            hr_pop, 
            av_pop,
            hv_pop,
        }
    }
}

#[derive(Serialize)]
pub struct TimeOutput {
    pub t_array: Vec<usize>,
    pub as_pop_t: Vec<usize>,
    pub hs_pop_t: Vec<usize>,
    pub ai_pop_t: Vec<usize>,
    pub hi_pop_t: Vec<usize>,
    pub ar_pop_t: Vec<usize>,
    pub hr_pop_t: Vec<usize>,
    pub av_pop_t: Vec<usize>,
    pub hv_pop_t: Vec<usize>,
}

impl Default for TimeOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl TimeOutput {
    pub fn new() -> Self {
        let t_array = Vec::new();
        let as_pop_t = Vec::new();
        let hs_pop_t = Vec::new();
        let ai_pop_t = Vec::new();
        let hi_pop_t = Vec::new();
        let ar_pop_t = Vec::new();
        let hr_pop_t = Vec::new();
        let av_pop_t = Vec::new();
        let hv_pop_t = Vec::new();

        Self {
            t_array,
            as_pop_t,
            hs_pop_t,
            ai_pop_t,
            hi_pop_t,
            ar_pop_t,
            hr_pop_t,
            av_pop_t,
            hv_pop_t,
        }
    }

    pub fn update_time_series(&mut self, t: usize, pop_t: &TimeUnitPop) {
        self.t_array.push(t);
        self.as_pop_t.push(pop_t.as_pop);
        self.hs_pop_t.push(pop_t.hs_pop);
        self.ai_pop_t.push(pop_t.ai_pop);
        self.hi_pop_t.push(pop_t.hi_pop);
        self.ar_pop_t.push(pop_t.ar_pop);
        self.hr_pop_t.push(pop_t.hr_pop);
        self.av_pop_t.push(pop_t.av_pop);
        self.hv_pop_t.push(pop_t.hv_pop);
    }
}

#[derive(Serialize)]
pub struct OutputResults {
    pub agent_ensemble: Option<AgentEnsembleOutput>,
    pub cluster: Option<ClusterOutput>,
    pub global: GlobalOutput,
    pub time: Option<TimeOutput>,
}

impl OutputResults {
    pub fn new(
        agent_ensemble: AgentEnsembleOutput,
        cluster: ClusterOutput,
        global: GlobalOutput,
        time: TimeOutput,
    ) -> Self {
        Self {
            agent_ensemble: Some(agent_ensemble),
            cluster: Some(cluster),
            global,
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

    pub fn add_outbreak(&mut self, output: OutputResults, input: &Input) {
        let n = input.network.n;
        let r0 = input.epidemic.r0;
        let cutoff_fraction = 0.0001; 
        let global_prevalence = output.global.prevalence;
        if (r0 > 1.0) && ((global_prevalence as f64) > cutoff_fraction * (n as f64)) {
            self.inner_mut().push(output)
        }
    }

    pub fn filter_outbreaks(&mut self, input: &Input) {
        let n = input.network.n;
        let r0 = input.epidemic.r0;
        let cutoff_fraction = 0.005; //0.005
        let mut s: usize = 0;
        let nsims = self.number_of_simulations() as usize;
        while s < nsims {
            let global_prevalence = self.inner()[s].global.prevalence;
            if (r0 > 1.0) && ((global_prevalence as f64) < cutoff_fraction * (n as f64)) {
                self.inner_mut().remove(s);
            } else {
                s += 1;
            }
        }
    }

    pub fn assemble_global_observables(&mut self) -> GlobalAssembledVectors {
        let nsims = self.number_of_simulations() as usize;
        let mut assembled_convinced = vec![0; nsims];
        let mut assembled_convinced_at_peak = vec![0; nsims];
        let mut assembled_peak_incidence = vec![0; nsims];
        let mut assembled_prevalence = vec![0; nsims];
        let mut assembled_time_to_end = vec![0; nsims];
        let mut assembled_time_to_peak = vec![0; nsims];
        let mut assembled_vaccinated = vec![0; nsims];
        let mut assembled_vaccinated_at_peak = vec![0; nsims];

        for s in 0..nsims {
            assembled_convinced[s] = self.inner()[s].global.active;
            assembled_convinced_at_peak[s] = self.inner()[s].global.convinced_at_peak;
            assembled_peak_incidence[s] = self.inner()[s].global.peak_incidence;
            assembled_prevalence[s] = self.inner()[s].global.prevalence;
            assembled_time_to_end[s] = self.inner()[s].global.time_to_end;
            assembled_time_to_peak[s] = self.inner()[s].global.time_to_peak;
            assembled_vaccinated[s] = self.inner()[s].global.vaccinated;
            assembled_vaccinated_at_peak[s] = self.inner()[s].global.vaccinated_at_peak;
        }

        GlobalAssembledVectors::new(
            assembled_convinced,
            assembled_convinced_at_peak,
            assembled_peak_incidence,
            assembled_prevalence,
            assembled_time_to_end,
            assembled_time_to_peak,
            assembled_vaccinated,
            assembled_vaccinated_at_peak,
        )
    }

    pub fn assemble_cluster_observables(&mut self) -> ClusterAssembledVectors {
        let nsims = self.number_of_simulations() as usize;
        let mut assembled_as_clusters = vec![vec![]; nsims];
        let mut assembled_hs_clusters = vec![vec![]; nsims];
        let mut assembled_ai_clusters = vec![vec![]; nsims];
        let mut assembled_hi_clusters = vec![vec![]; nsims];
        let mut assembled_ar_clusters = vec![vec![]; nsims];
        let mut assembled_hr_clusters = vec![vec![]; nsims];
        let mut assembled_av_clusters = vec![vec![]; nsims];
        let mut assembled_hv_clusters = vec![vec![]; nsims];
        let mut assembled_ze_clusters = vec![vec![]; nsims];

        for s in 0..nsims {
            assembled_as_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().as_cluster.clone();
            assembled_hs_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().hs_cluster.clone();
            assembled_ai_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().ai_cluster.clone();
            assembled_hi_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().hi_cluster.clone();
            assembled_ar_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().ar_cluster.clone();
            assembled_hr_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().hr_cluster.clone();
            assembled_av_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().av_cluster.clone();
            assembled_hv_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().hv_cluster.clone();
            assembled_ze_clusters[s] = self.inner()[s].cluster.as_ref().unwrap().ze_cluster.clone();
        }

        ClusterAssembledVectors::new(
            assembled_ai_clusters, 
            assembled_ar_clusters, 
            assembled_as_clusters, 
            assembled_av_clusters, 
            assembled_hi_clusters, 
            assembled_hr_clusters, 
            assembled_hs_clusters, 
            assembled_hv_clusters,
            assembled_ze_clusters,
        )
    }

    pub fn assemble_agent_observables(
        &mut self, 
        input: &Input
    ) -> AssembledAgentOutput {
        let nsims = self.number_of_simulations() as usize;
        let n = input.network.n;
        let mut assembled_convinced_when = vec![vec![9999999; n]; nsims];
        let mut assembled_degree = vec![vec![9999999; n]; nsims];
        let mut assembled_final_active_susceptible = vec![vec![999999; n]; nsims];
        let mut assembled_final_prevalence = vec![vec![999999; n]; nsims];
        let mut assembled_final_vaccinated = vec![vec![999999; n]; nsims];
        let mut assembled_id = vec![vec![9999999; n]; nsims];
        let mut assembled_infected_by = vec![vec![9999999; n]; nsims];
        let mut assembled_infected_when = vec![vec![9999999; n]; nsims];
        let mut assembled_initial_active_susceptible = vec![vec![999999; n]; nsims];
        let mut assembled_initial_vaccinated = vec![vec![999999; n]; nsims];
        let mut assembled_removed_when = vec![vec![999999; n]; nsims];
        let mut assembled_status = vec![vec![Status::HesSus; n]; nsims];
        let mut assembled_threshold = vec![vec![0.0; n]; nsims];
        let mut assembled_vaccinated_when = vec![vec![9999999; n]; nsims];
        let mut assembled_zealots = vec![vec![999999; n]; nsims];
    
        for s in 0..nsims {
            for a in 0..n {
                assembled_convinced_when[s][a] = 
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].convinced_when.unwrap();
                assembled_degree[s][a] = 
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].degree.unwrap();
                assembled_final_active_susceptible[s][a] =
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].final_active_susceptible.unwrap();
                assembled_final_prevalence[s][a] =
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].final_prevalence.unwrap();
                assembled_final_vaccinated[s][a] =
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].final_vaccinated.unwrap();
                assembled_id[s][a] = 
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].id.unwrap();
                assembled_infected_by[s][a] = 
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].infected_by.unwrap();
                assembled_infected_when[s][a] = 
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].infected_when.unwrap();
                assembled_initial_active_susceptible[s][a] =
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].initial_active_susceptible.unwrap();
                assembled_initial_vaccinated[s][a] =
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].initial_vaccinated.unwrap();
                assembled_removed_when[s][a] = 
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].removed_when.unwrap();
                assembled_status[s][a] = 
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].status.unwrap();
                assembled_threshold[s][a] = 
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].threshold.unwrap();
                assembled_vaccinated_when[s][a] = 
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].vaccinated_when.unwrap();
                assembled_zealots[s][a] =
                self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[a].zealots.unwrap();
            }
        }

        AssembledAgentOutput::new(
            assembled_convinced_when,
            assembled_degree, 
            assembled_final_active_susceptible,
            assembled_final_prevalence,
            assembled_final_vaccinated,
            assembled_id,
            assembled_infected_by, 
            assembled_infected_when, 
            assembled_initial_active_susceptible,
            assembled_initial_vaccinated,
            assembled_removed_when, 
            assembled_status, 
            assembled_threshold, 
            assembled_vaccinated_when,
            assembled_zealots,
        )
    }

    pub fn assemble_time_series(
        &mut self,
        t_max: usize,
    ) -> AssembledTimeSeries {
        let nsims = self.number_of_simulations() as usize;
        let mut assembled_t_array_st = vec![vec![9999999; t_max]; nsims];
        let mut assembled_as_pop_st = vec![vec![9999999; t_max]; nsims];
        let mut assembled_hs_pop_st = vec![vec![9999999; t_max]; nsims];
        let mut assembled_ai_pop_st = vec![vec![9999999; t_max]; nsims];
        let mut assembled_hi_pop_st = vec![vec![9999999; t_max]; nsims];
        let mut assembled_ar_pop_st = vec![vec![9999999; t_max]; nsims];
        let mut assembled_hr_pop_st = vec![vec![9999999; t_max]; nsims];
        let mut assembled_av_pop_st = vec![vec![9999999; t_max]; nsims];
        let mut assembled_hv_pop_st = vec![vec![9999999; t_max]; nsims];

        for s in 0..nsims {
            for t in 0..t_max as usize {
                assembled_t_array_st[s][t] = 
                self.inner()[s].time.as_ref().unwrap().t_array[t];
                assembled_as_pop_st[s][t] = 
                self.inner()[s].time.as_ref().unwrap().as_pop_t[t];
                assembled_hs_pop_st[s][t] = 
                self.inner()[s].time.as_ref().unwrap().hs_pop_t[t];
                assembled_ai_pop_st[s][t] = 
                self.inner()[s].time.as_ref().unwrap().ai_pop_t[t];
                assembled_hi_pop_st[s][t] = 
                self.inner()[s].time.as_ref().unwrap().hi_pop_t[t];
                assembled_ar_pop_st[s][t] = 
                self.inner()[s].time.as_ref().unwrap().ar_pop_t[t];
                assembled_hr_pop_st[s][t] = 
                self.inner()[s].time.as_ref().unwrap().hr_pop_t[t];
                assembled_av_pop_st[s][t] = 
                self.inner()[s].time.as_ref().unwrap().av_pop_t[t];
                assembled_hv_pop_st[s][t] =
                self.inner()[s].time.as_ref().unwrap().hv_pop_t[t];
            }
        }
        
        AssembledTimeSeries::new(
            assembled_t_array_st, 
            assembled_as_pop_st, 
            assembled_hs_pop_st, 
            assembled_ai_pop_st, 
            assembled_hi_pop_st, 
            assembled_ar_pop_st, 
            assembled_hr_pop_st, 
            assembled_av_pop_st,
            assembled_hv_pop_st,
        )
    }
}

#[derive(Serialize)]
pub struct GlobalAssembledVectors {
    pub convinced: Vec<usize>,
    pub convinced_at_peak: Vec<usize>,
    pub peak_incidence: Vec<usize>,
    pub prevalence: Vec<usize>,
    pub time_to_end: Vec<usize>,
    pub time_to_peak: Vec<usize>,
    pub vaccinated: Vec<usize>,
    pub vaccinated_at_peak: Vec<usize>,
}

impl GlobalAssembledVectors {
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
pub struct ClusterAssembledVectors {
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

impl ClusterAssembledVectors {
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
pub struct SerializeGlobalAssembly {
    pub global: GlobalAssembledVectors,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedClusterAssembly {
    pub cluster: ClusterAssembledVectors,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct AssembledAgentOutput {
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
pub struct AssembledTimeSeries {
    pub t_array: Vec<Vec<usize>>,
    pub as_pop_st: Vec<Vec<usize>>,
    pub hs_pop_st: Vec<Vec<usize>>,
    pub ai_pop_st: Vec<Vec<usize>>,
    pub hi_pop_st: Vec<Vec<usize>>,
    pub ar_pop_st: Vec<Vec<usize>>,
    pub hr_pop_st: Vec<Vec<usize>>,
    pub av_pop_st: Vec<Vec<usize>>,
    pub hv_pop_st: Vec<Vec<usize>>,
}

impl AssembledTimeSeries {
    pub fn new(
        t_array: Vec<Vec<usize>>,
        as_pop_st: Vec<Vec<usize>>,
        hs_pop_st: Vec<Vec<usize>>,
        ai_pop_st: Vec<Vec<usize>>,
        hi_pop_st: Vec<Vec<usize>>,
        ar_pop_st: Vec<Vec<usize>>,
        hr_pop_st: Vec<Vec<usize>>,
        av_pop_st: Vec<Vec<usize>>,
        hv_pop_st: Vec<Vec<usize>>,
    ) -> Self {
        Self {
            t_array,            
            as_pop_st,
            hs_pop_st,
            ai_pop_st,
            hi_pop_st,
            ar_pop_st,
            hr_pop_st,
            av_pop_st, 
            hv_pop_st,
        }
    }
}

#[derive(Serialize)]
pub struct SerializeGlobalAssemblyTwoWaves {
    pub global_w1: GlobalAssembledVectors,
    pub global_w2: GlobalAssembledVectors,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializeClusterAssemblyTwoWaves {
    pub cluster_w1: ClusterAssembledVectors,
    pub cluster_w2: ClusterAssembledVectors,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedAgentAssemblyTwoWaves {
    pub agent_w1: AssembledAgentOutput,
    pub agent_w2: AssembledAgentOutput,
    pub pars: Input,
}

#[derive(Serialize)]
pub struct SerializedTimeSeriesAssemblyTwoWaves {
    pub time_w1: AssembledTimeSeries,
    pub time_w2: AssembledTimeSeries,
    pub pars: Input,
}

// Output file name writing-related functions

pub fn write_network_string(npars: &NetworkPars) -> String {
    let pars = &npars.pars;
    
    match pars {
        NetworkParsEnum::BarabasiAlbert { n, k_avg, .. } => {
            format!("_netba_n{}_k{}", n, k_avg)
        },
        NetworkParsEnum::Complete { n } => {
            format!("_netco_n{}", n)
        },
        NetworkParsEnum::ErdosRenyi { n, k_avg, .. } => {
            format!("_neter_n{}_k{}", n, k_avg)
        },
        NetworkParsEnum::Regular { n, k } => {
            format!("_netre_n{}_k{}", n, k)
        },
        NetworkParsEnum::ScaleFree { n, k_min, k_max, gamma, .. } => {
            format!("_netsf_n{}_kmin{}_kmax{}_gamma{}", n, k_min, k_max, gamma)
        },
        NetworkParsEnum::WattsStrogatz { n, k, p } => {
            format!("_netws_n{}_k{}_p{}", n, k, p)
        },
    }
}

pub fn write_opinion_string(opars: &OpinionPars) -> String {
    format!(
        "_acf{0}_thr{1}_zef{2}",
        opars.active_fraction,
        opars.threshold,
        opars.zealot_fraction,
    )
}

pub fn write_epidemic_string(epars: &EpidemicPars) -> String {
    format!(
        "_r0{0}_rer{1}_var{2}",
        epars.r0,
        epars.infection_decay, 
        epars.vaccination_rate,
    )
}

pub fn write_algorithm_string(apars: &AlgorithmPars) -> String {
    format!(
        "_nsd{0}_nsn{1}_tmax{2}",
        apars.nsims_dyn,
        apars.nsims_net, 
        apars.t_max,
    )
}

pub fn write_file_name(
    pars: &Input, 
    exp_id: String
) -> String {
    let head = "thr_mc_".to_string(); // THRESHOLD PROJECT, MONTE CARLO SIMULATIONS
    let npars_chain = write_network_string(&pars.network);
    let opars_chain = write_opinion_string(&pars.opinion);
    let epars_chain = write_epidemic_string(&pars.epidemic);
    let apars_chain = write_algorithm_string(&pars.algorithm);
    head + &exp_id + &npars_chain + &opars_chain + &epars_chain + &apars_chain
}

// Data-driven experiments related stuff

#[derive(Clone, Copy, Serialize, Display, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum USState {
    Alabama,
    Alaska,
    Arizona,
    Arkansas,
    California,
    Colorado,
    Connecticut,
    Delaware,
    DistrictOfColumbia,
    Florida,
    Georgia,
    Hawaii,
    Idaho,
    Illinois,
    Indiana,
    Iowa,
    Kansas,
    Kentucky,
    Louisiana,
    Maine,
    Maryland,
    Massachusetts,
    Michigan,
    Minnesota,
    Mississippi,
    Missouri,
    Montana,
    Nebraska,
    Nevada,
    NewHampshire,
    NewJersey,
    NewMexico,
    NewYork,
    NorthCarolina,
    NorthDakota,
    Ohio,
    Oklahoma,
    Oregon,
    Pennsylvania,
    RhodeIsland,
    SouthCarolina,
    SouthDakota,
    Tennessee,
    Texas,
    Utah,
    Vermont,
    Virginia,
    Washington,
    WestVirginia,
    Wisconsin,
    Wyoming,
    National,
}

#[derive(Debug, Deserialize, Serialize)]
struct VaccinationData {
    #[serde(rename = "state")]
    state_name: String,
    fractions: Vec<f64>,
}

pub fn _read_categories(state: USState) -> Vec<f64> {
    // Load the JSON file into a vector of VaccinationData
    let mut file_path = PathBuf::new();
    file_path.push("/results/vaccination_data.json");

    let file_contents = std::fs::read_to_string(file_path)
        .expect("Failed to read file");
    let data: Vec<VaccinationData> = serde_json::from_str(&file_contents)
        .expect("Failed to parse JSON data");

    // Find the VaccinationData that matches the state
    let vaccination_data = data.into_iter()
        .find(|v| v.state_name == state.to_string())
        .unwrap_or_else(|| panic!("No data for state: {}", state));

    vaccination_data.fractions
}

pub fn read_categories(state: USState, filename: &str) -> Vec<f64> {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");
    path.push(format!("{}.json", filename));

    // Open the file
    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    //let file_contents = std::fs::read_to_string(file_path).expect("Failed to read file");
    let data: Value = serde_json::from_str(&contents).unwrap();

    let state_str = state.to_string();
    let state_data = &data[state_str];
    let fractions: Vec<f64> = serde_json::from_value(state_data.clone())
        .expect("Failed to parse state data");

    fractions
}

pub fn read_degree(state: USState, filename: &str) -> f64 {
    let mut path = 
    PathBuf::from(env::current_dir()
    .expect("Failed to get current directory"));
    path.push("data");
    path.push(format!("{}.json", filename));

    // Open the file
    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    //let file_contents = std::fs::read_to_string(file_path).expect("Failed to read file");
    let data: Value = serde_json::from_str(&contents).unwrap();

    let state_str = state.to_string();
    let state_data = &data[state_str];
    let degree: f64 = serde_json::from_value(state_data.clone())
        .expect("Failed to parse state data");

    degree
}

// Input-related structs & implementations
#[derive(Debug, Clone, Copy, Serialize)]
pub struct StatPacker {
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

#[derive(Clone, Copy, Serialize)]
pub struct ClusterStatsOverSims {
    pub average_size: StatPacker,
    pub max_size: StatPacker,
    pub number: StatPacker,
}

impl ClusterStatsOverSims {
    pub fn new(average_size: StatPacker, max_size: StatPacker, number: StatPacker) -> Self {
        Self {
            average_size,
            max_size,
            number,
        }
    }
}

#[derive(Clone, Copy, Serialize)]
pub struct ClusterStatPacker {
    pub ai_avg_size_stats: StatPacker,
    pub ar_avg_size_stats: StatPacker,
    pub as_avg_size_stats: StatPacker,
    pub av_avg_size_stats: StatPacker,
    pub hi_avg_size_stats: StatPacker,
    pub hr_avg_size_stats: StatPacker,
    pub hs_avg_size_stats: StatPacker,
    pub hv_avg_size_stats: StatPacker,
    pub ai_max_size_stats: StatPacker,
    pub ar_max_size_stats: StatPacker,
    pub as_max_size_stats: StatPacker,
    pub av_max_size_stats: StatPacker,
    pub hi_max_size_stats: StatPacker,
    pub hr_max_size_stats: StatPacker,
    pub hs_max_size_stats: StatPacker,
    pub hv_max_size_stats: StatPacker,
    pub ai_number_stats: StatPacker,
    pub ar_number_stats: StatPacker,
    pub as_number_stats: StatPacker,
    pub av_number_stats: StatPacker,
    pub hi_number_stats: StatPacker,
    pub hr_number_stats: StatPacker,
    pub hs_number_stats: StatPacker,
    pub hv_number_stats: StatPacker,
}

pub fn calculate_cluster_sim_stats(simulations: &[Vec<usize>]) -> ClusterStatsOverSims {
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

pub fn calculate_stat_pack(values: &[f64]) -> StatPacker {
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

#[derive(Debug, Clone, Serialize)]
pub struct AgentStatPacker {
    pub convinced_when_stats: StatPacker,
    pub degree_stats: StatPacker,
    pub final_active_stats: StatPacker,
    pub final_prevalence_stats: StatPacker,
    pub final_vaccinated_stats: StatPacker,
    pub infected_when_stats: StatPacker,
    pub initial_active_susceptible_stats: StatPacker,
    pub initial_vaccinated_stats: StatPacker,
    pub removed_when_stats: StatPacker,
    pub vaccinated_when_stats: StatPacker,
    pub zealots_stats: StatPacker,
}

pub fn compute_stats(values: &Vec<Vec<f64>>) -> StatPacker {
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

pub fn convert_to_f64(vec: &Vec<usize>) -> Vec<f64> {
    vec.iter().map(|&value| value as f64).collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentDistributionPacker {
    pub convinced_when_dist: AgentDistribution,
    pub degree_dist: AgentDistribution,
    pub frac_final_active_dist: AgentDistribution,
    pub frac_final_prevalence_dist: AgentDistribution,
    pub frac_final_vaccinated_dist: AgentDistribution,
    pub infected_when_dist: AgentDistribution,
    pub frac_initial_active_susceptible_dist: AgentDistribution,
    pub frac_initial_vaccinated_dist: AgentDistribution,
    pub removed_when_dist: AgentDistribution,
    pub vaccinated_when_dist: AgentDistribution,
    pub frac_zealots_dist: AgentDistribution,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentDistribution {
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