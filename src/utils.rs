use rand::rngs::ThreadRng;
use rand::Rng;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use serde_pickle::{DeOptions, SerOptions};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::{env, vec};
use strum::Display;

use crate::agent::{
    AgentEnsemble, Attitude, HesitancyModel, Multilayer, OpinionModel, SeedModel, Status,
    VaccinationPolicy,
};
use crate::cons::{
    CONST_EPIDEMIC_THRESHOLD, EXTENSION_RESULTS_PICKLE, FOLDER_DATA_CURATED, FOLDER_RESULTS, HEADER_AGE, HEADER_AGENT_DISTRIBUTION, HEADER_AGENT_STATS, HEADER_ATTITUDE, HEADER_CLUSTER_DISTRIBUTION, HEADER_CLUSTER_STATS, HEADER_DEGREE, HEADER_GLOBAL, HEADER_PROJECT, HEADER_TIME, HEADER_TIME_STATS, INIT_ATTITUDE, INIT_STATUS, INIT_USIZE, PAR_AGE_GROUPS, PAR_ATTITUDE_GROUPS, PAR_NBINS, PAR_OUTBREAK_PREVALENCE_FRACTION_CUTOFF, PATH_RESULTS_CURATED_LOCAL
};

#[derive(Clone, Copy, Serialize, Deserialize, Display, Debug, clap::ValueEnum, PartialEq, Eq)]
pub enum Region {
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
    #[serde(rename = "none")]
    None,
}

#[derive(Serialize)]
struct AgentOutput {
    pub activation_potential: Option<usize>,
    pub age: Option<usize>,
    pub attitude: Option<Attitude>,
    pub cascading_threshold: Option<usize>,
    pub convinced_when: Option<usize>,
    pub degree: Option<usize>,
    pub effective_threshold: Option<f64>,
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
        cascading_threshold: Option<usize>,
        convinced_when: Option<usize>,
        degree: Option<usize>,
        effective_threshold: Option<f64>,
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
            cascading_threshold,
            convinced_when,
            degree,
            effective_threshold,
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
            let cascading_potential = agent.cascading_threshold.unwrap();
            let convinced_when = agent.convinced_when.unwrap_or(INIT_USIZE);
            let degree = agent.neighbors.len();
            let effective_threshold = agent.effective_threshold.unwrap();
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
                Some(cascading_potential),
                Some(convinced_when),
                Some(degree),
                Some(effective_threshold),
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

#[derive(Serialize)]
pub struct AssembledAgeOutput {
    activation_potential: Option<Vec<Vec<Vec<usize>>>>,
    active: Option<Vec<Vec<usize>>>,
    age: Option<Vec<Vec<usize>>>,
    cascading_threshold: Option<Vec<Vec<Vec<usize>>>>,
    convinced_when: Option<Vec<Vec<Vec<usize>>>>,
    degree: Option<Vec<Vec<Vec<usize>>>>,
    effective_threshold: Option<Vec<Vec<Vec<f64>>>>,
    final_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
    final_prevalence: Option<Vec<Vec<Vec<usize>>>>,
    final_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
    infected_when: Option<Vec<Vec<Vec<usize>>>>,
    initial_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
    initial_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
    prevalence: Option<Vec<Vec<usize>>>,
    removed_when: Option<Vec<Vec<Vec<usize>>>>,
    vaccinated: Option<Vec<Vec<usize>>>,
    vaccinated_when: Option<Vec<Vec<Vec<usize>>>>,
    zealots: Option<Vec<Vec<Vec<usize>>>>,
}

impl AssembledAgeOutput {
    pub fn new(
        activation_potential: Option<Vec<Vec<Vec<usize>>>>,
        active: Option<Vec<Vec<usize>>>,
        age: Option<Vec<Vec<usize>>>,
        cascading_threshold: Option<Vec<Vec<Vec<usize>>>>,
        convinced_when: Option<Vec<Vec<Vec<usize>>>>,
        degree: Option<Vec<Vec<Vec<usize>>>>,
        effective_threshold: Option<Vec<Vec<Vec<f64>>>>,
        final_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
        final_prevalence: Option<Vec<Vec<Vec<usize>>>>,
        final_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
        infected_when: Option<Vec<Vec<Vec<usize>>>>,
        initial_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
        initial_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
        prevalence: Option<Vec<Vec<usize>>>,
        removed_when: Option<Vec<Vec<Vec<usize>>>>,
        vaccinated: Option<Vec<Vec<usize>>>,
        vaccinated_when: Option<Vec<Vec<Vec<usize>>>>,
        zealots: Option<Vec<Vec<Vec<usize>>>>,
    ) -> Self {
        Self {
            activation_potential,
            active,
            age,
            cascading_threshold,
            convinced_when,
            degree,
            effective_threshold,
            final_active_susceptible,
            final_prevalence,
            final_vaccinated,
            infected_when,
            initial_active_susceptible,
            initial_vaccinated,
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
    pub activation_potential: Option<Vec<Vec<usize>>>,
    pub age: Option<Vec<Vec<usize>>>,
    pub attitude: Option<Vec<Vec<Attitude>>>,
    pub convinced_when: Option<Vec<Vec<usize>>>,
    pub degree: Option<Vec<Vec<usize>>>,
    pub final_active_susceptible: Option<Vec<Vec<usize>>>,
    pub final_prevalence: Option<Vec<Vec<usize>>>,
    pub final_vaccinated: Option<Vec<Vec<usize>>>,
    pub id: Option<Vec<Vec<usize>>>,
    pub infected_by: Option<Vec<Vec<usize>>>,
    pub infected_when: Option<Vec<Vec<usize>>>,
    pub initial_active_susceptible: Option<Vec<Vec<usize>>>,
    pub initial_vaccinated: Option<Vec<Vec<usize>>>,
    pub removed_when: Option<Vec<Vec<usize>>>,
    pub status: Option<Vec<Vec<Status>>>,
    pub threshold: Option<Vec<Vec<f64>>>,
    pub vaccinated_when: Option<Vec<Vec<usize>>>,
    pub zealots: Option<Vec<Vec<usize>>>,
}

impl AssembledAgentOutput {
    pub fn new(
        activation_potential: Option<Vec<Vec<usize>>>,
        age: Option<Vec<Vec<usize>>>,
        attitude: Option<Vec<Vec<Attitude>>>,
        convinced_when: Option<Vec<Vec<usize>>>,
        degree: Option<Vec<Vec<usize>>>,
        final_active_susceptible: Option<Vec<Vec<usize>>>,
        final_prevalence: Option<Vec<Vec<usize>>>,
        final_vaccinated: Option<Vec<Vec<usize>>>,
        id: Option<Vec<Vec<usize>>>,
        infected_by: Option<Vec<Vec<usize>>>,
        infected_when: Option<Vec<Vec<usize>>>,
        initial_active_susceptible: Option<Vec<Vec<usize>>>,
        initial_vaccinated: Option<Vec<Vec<usize>>>,
        removed_when: Option<Vec<Vec<usize>>>,
        status: Option<Vec<Vec<Status>>>,
        threshold: Option<Vec<Vec<f64>>>,
        vaccinated_when: Option<Vec<Vec<usize>>>,
        zealots: Option<Vec<Vec<usize>>>,
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
pub struct AssembledAttitudeOutput {
    activation_potential: Option<Vec<Vec<Vec<usize>>>>,
    active: Option<Vec<Vec<usize>>>,
    age: Option<Vec<Vec<Vec<usize>>>>,
    cascading_threshold: Option<Vec<Vec<Vec<usize>>>>,
    convinced_when: Option<Vec<Vec<Vec<usize>>>>,
    degree: Option<Vec<Vec<Vec<usize>>>>,
    effective_threshold: Option<Vec<Vec<Vec<f64>>>>,
    final_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
    final_prevalence: Option<Vec<Vec<Vec<usize>>>>,
    final_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
    infected_when: Option<Vec<Vec<Vec<usize>>>>,
    initial_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
    initial_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
    prevalence: Option<Vec<Vec<usize>>>,
    removed_when: Option<Vec<Vec<Vec<usize>>>>,
    vaccinated: Option<Vec<Vec<usize>>>,
    vaccinated_when: Option<Vec<Vec<Vec<usize>>>>,
    zealots: Option<Vec<Vec<Vec<usize>>>>,
}

impl AssembledAttitudeOutput {
    pub fn new(
        activation_potential: Option<Vec<Vec<Vec<usize>>>>,
        active: Option<Vec<Vec<usize>>>,
        age: Option<Vec<Vec<Vec<usize>>>>,
        cascading_threshold: Option<Vec<Vec<Vec<usize>>>>,
        convinced_when: Option<Vec<Vec<Vec<usize>>>>,
        degree: Option<Vec<Vec<Vec<usize>>>>,
        effective_threshold: Option<Vec<Vec<Vec<f64>>>>,
        final_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
        final_prevalence: Option<Vec<Vec<Vec<usize>>>>,
        final_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
        infected_when: Option<Vec<Vec<Vec<usize>>>>,
        initial_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
        initial_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
        prevalence: Option<Vec<Vec<usize>>>,
        removed_when: Option<Vec<Vec<Vec<usize>>>>,
        vaccinated: Option<Vec<Vec<usize>>>,
        vaccinated_when: Option<Vec<Vec<Vec<usize>>>>,
        zealots: Option<Vec<Vec<Vec<usize>>>>,
    ) -> Self {
        Self {
            activation_potential,
            active,
            age,
            cascading_threshold,
            convinced_when,
            degree,
            effective_threshold,
            final_active_susceptible,
            final_prevalence,
            final_vaccinated,
            infected_when,
            initial_active_susceptible,
            initial_vaccinated,
            prevalence,
            removed_when,
            vaccinated,
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
        never_cluster: Vec<Vec<usize>>,
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
pub struct AssembledClusterCascadingOutput {
    pub cascading_cluster: Vec<Vec<usize>>,
    pub nonzealot_cluster: Vec<Vec<usize>>,
}

impl AssembledClusterCascadingOutput {
    pub fn new(cascading_cluster: Vec<Vec<usize>>, nonzealot_cluster: Vec<Vec<usize>>) -> Self {
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
    activation_potential: Option<Vec<Vec<Vec<usize>>>>,
    active: Option<Vec<Vec<usize>>>,
    age: Option<Vec<Vec<Vec<usize>>>>,
    cascading_threshold: Option<Vec<Vec<Vec<usize>>>>,
    convinced_when: Option<Vec<Vec<Vec<usize>>>>,
    degree: Option<Vec<Vec<usize>>>,
    effective_threshold: Option<Vec<Vec<Vec<f64>>>>,
    final_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
    final_prevalence: Option<Vec<Vec<Vec<usize>>>>,
    final_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
    infected_when: Option<Vec<Vec<Vec<usize>>>>,
    initial_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
    initial_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
    prevalence: Option<Vec<Vec<usize>>>,
    removed_when: Option<Vec<Vec<Vec<usize>>>>,
    vaccinated: Option<Vec<Vec<usize>>>,
    vaccinated_when: Option<Vec<Vec<Vec<usize>>>>,
    zealots: Option<Vec<Vec<Vec<usize>>>>,
}

impl AssembledDegreeOutput {
    pub fn new(
        activation_potential: Option<Vec<Vec<Vec<usize>>>>,
        active: Option<Vec<Vec<usize>>>,
        age: Option<Vec<Vec<Vec<usize>>>>,
        cascading_threshold: Option<Vec<Vec<Vec<usize>>>>,
        convinced_when: Option<Vec<Vec<Vec<usize>>>>,
        degree: Option<Vec<Vec<usize>>>,
        effective_threshold: Option<Vec<Vec<Vec<f64>>>>,
        final_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
        final_prevalence: Option<Vec<Vec<Vec<usize>>>>,
        final_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
        infected_when: Option<Vec<Vec<Vec<usize>>>>,
        initial_active_susceptible: Option<Vec<Vec<Vec<usize>>>>,
        initial_vaccinated: Option<Vec<Vec<Vec<usize>>>>,
        prevalence: Option<Vec<Vec<usize>>>,
        removed_when: Option<Vec<Vec<Vec<usize>>>>,
        vaccinated: Option<Vec<Vec<usize>>>,
        vaccinated_when: Option<Vec<Vec<Vec<usize>>>>,
        zealots: Option<Vec<Vec<Vec<usize>>>>,
    ) -> Self {
        Self {
            activation_potential,
            active,
            age,
            cascading_threshold,
            convinced_when,
            degree,
            effective_threshold,
            final_active_susceptible,
            final_prevalence,
            final_vaccinated,
            infected_when,
            initial_active_susceptible,
            initial_vaccinated,
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
    pub fn new(cascading_cluster: Vec<usize>, nonzealot_cluster: Vec<usize>) -> Self {
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
    pub fn new(
        attitude: Option<ClusterAttitudeOutput>,
        cascading: Option<ClusterCascadingOutput>,
        opinion_health: Option<ClusterOpinionHealthOutput>,
    ) -> Self {
        Self {
            attitude,
            cascading,
            opinion_health,
        }
    }
}

#[derive(Serialize)]
pub struct ContactOutput {
    pub age_distribution: Vec<f64>,
    pub contact_matrix: Vec<Vec<f64>>,
    pub degree_distribution: Vec<f64>,
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
pub struct InputMultilayer {
    pub flag_underage: bool,
    pub fraction_active: f64,
    pub fraction_majority: f64,
    pub fraction_someone: f64,
    pub fraction_soon: f64,
    pub fraction_vaccinated: f64,
    pub fraction_zealot: f64,
    pub id_experiment: usize,
    pub model_hesitancy: HesitancyModel,
    pub model_opinion: OpinionModel,
    pub model_region: Region,
    pub model_seed: SeedModel,
    pub nseeds: usize,
    pub nsims: usize,
    pub policy_vaccination: VaccinationPolicy,
    pub quota_vaccination: f64,
    pub r0: f64,
    pub rate_infection: f64,
    pub rate_removal: f64,
    pub rate_vaccination: f64,
    pub t_max: usize,
    pub threshold_age: usize,
    pub threshold_opinion: f64,
}

impl InputMultilayer {
    pub fn new(
        flag_underage: bool,
        fraction_active: f64,
        fraction_majority: f64,
        fraction_someone: f64,
        fraction_soon: f64,
        fraction_vaccinated: f64,
        fraction_zealot: f64,
        id_experiment: usize,
        model_hesitancy: HesitancyModel,
        model_opinion: OpinionModel,
        model_region: Region,
        model_seed: SeedModel,
        nseeds: usize,
        nsims: usize,
        policy_vaccination: VaccinationPolicy,
        quota_vaccination: f64,
        r0: f64,
        rate_infection: f64,
        rate_removal: f64,
        rate_vaccination: f64,
        t_max: usize,
        threshold_age: usize,
        threshold_opinion: f64,
    ) -> Self {
        Self {
            flag_underage,
            fraction_active,
            fraction_majority,
            fraction_someone,
            fraction_soon,
            fraction_vaccinated,
            fraction_zealot,
            id_experiment,
            model_hesitancy,
            model_opinion,
            model_region,
            model_seed,
            nseeds,
            nsims,
            policy_vaccination,
            quota_vaccination,
            r0,
            rate_infection,
            rate_removal,
            rate_vaccination,
            t_max,
            threshold_age,
            threshold_opinion,
        }
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
        Self { inner: Vec::new() }
    }

    pub fn inner(&self) -> &Vec<OutputResults> {
        &self.inner
    }

    pub fn inner_mut(&mut self) -> &mut Vec<OutputResults> {
        &mut self.inner
    }

    pub fn clear(&mut self) {
        self.inner.clear();
    }

    pub fn number_of_simulations(&self) -> usize {
        self.inner.len()
    }

    pub fn add_outbreak(&mut self, output: OutputResults, n: usize, r0: f64) {
        let cutoff_fraction = PAR_OUTBREAK_PREVALENCE_FRACTION_CUTOFF;
        let global_prevalence = output.global.prevalence;
        if (r0 > CONST_EPIDEMIC_THRESHOLD)
            && ((global_prevalence as f64) > cutoff_fraction * (n as f64))
        {
            self.inner_mut().push(output)
        }
    }

    pub fn filter_outbreaks(&mut self, nagents: usize, r0: f64) {
        let cutoff_fraction = PAR_OUTBREAK_PREVALENCE_FRACTION_CUTOFF;
        let mut s: usize = 0;
        let nsims = self.number_of_simulations();
        while s < nsims {
            let global_prevalence = self.inner()[s].global.prevalence;
            if (r0 > CONST_EPIDEMIC_THRESHOLD)
                && ((global_prevalence as f64) < cutoff_fraction * (nagents as f64))
            {
                self.inner_mut().remove(s);
            } else {
                s += 1;
            }
        }
    }

    pub fn assemble_age_observables(&mut self) -> AssembledAgeOutput {
        let nagents = self.inner()[0].agent_ensemble.as_ref().unwrap().inner.len();
        let nsims = self.number_of_simulations();

        let mut activation_potential = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut active = vec![vec![0; PAR_AGE_GROUPS]; nsims];
        let mut age = vec![vec![0; PAR_AGE_GROUPS]; nsims];
        let mut cascading_threshold = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut convinced_when = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut degree = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut effective_threshold = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut final_active_susceptible = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut final_prevalence = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut final_vaccinated = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut infected_when = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut initial_active_susceptible = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut initial_vaccinated = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut prevalence = vec![vec![0; PAR_AGE_GROUPS]; nsims];
        let mut removed_when = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut vaccinated = vec![vec![0; PAR_AGE_GROUPS]; nsims];
        let mut vaccinated_when = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];
        let mut zealots = vec![vec![Vec::new(); PAR_AGE_GROUPS]; nsims];

        for s in 0..nsims {
            for i in 0..nagents {
                let a = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .age
                    .unwrap();
                let status = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .status
                    .unwrap();

                age[s][a] += 1;
                degree[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .degree
                        .unwrap(),
                );

                activation_potential[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .activation_potential
                        .unwrap(),
                );
                cascading_threshold[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .cascading_threshold
                        .unwrap(),
                );
                effective_threshold[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .effective_threshold
                        .unwrap(),
                );
                final_active_susceptible[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_active_susceptible
                        .unwrap(),
                );
                final_prevalence[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_prevalence
                        .unwrap(),
                );
                final_vaccinated[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_vaccinated
                        .unwrap(),
                );
                initial_active_susceptible[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .initial_active_susceptible
                        .unwrap(),
                );
                initial_vaccinated[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .initial_vaccinated
                        .unwrap(),
                );
                zealots[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .zealots
                        .unwrap(),
                );

                match status {
                    Status::ActRem => {
                        active[s][a] += 1;
                        convinced_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .convinced_when
                                .unwrap(),
                        );
                        infected_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .infected_when
                                .unwrap(),
                        );
                        prevalence[s][a] += 1;
                        removed_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .removed_when
                                .unwrap(),
                        );
                    }
                    Status::ActSus => {
                        active[s][a] += 1;
                        convinced_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .convinced_when
                                .unwrap(),
                        );
                    }
                    Status::ActVac => {
                        active[s][a] += 1;
                        convinced_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .convinced_when
                                .unwrap(),
                        );
                        vaccinated[s][a] += 1;
                        vaccinated_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .vaccinated_when
                                .unwrap(),
                        );
                    }
                    Status::HesRem => {
                        infected_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .infected_when
                                .unwrap(),
                        );
                        prevalence[s][a] += 1;
                        removed_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .removed_when
                                .unwrap(),
                        );
                    }
                    _ => {}
                };
            }
        }

        AssembledAgeOutput::new(
            Some(activation_potential),
            Some(active),
            Some(age),
            Some(cascading_threshold),
            Some(convinced_when),
            Some(degree),
            Some(effective_threshold),
            Some(final_active_susceptible),
            Some(final_prevalence),
            Some(final_vaccinated),
            Some(infected_when),
            Some(initial_active_susceptible),
            Some(initial_vaccinated),
            Some(prevalence),
            Some(removed_when),
            Some(vaccinated),
            Some(vaccinated_when),
            Some(zealots),
        )
    }

    pub fn assemble_agent_observables(&mut self) -> AssembledAgentOutput {
        let nsims = self.number_of_simulations();
        let nagents = self.inner()[0].agent_ensemble.as_ref().unwrap().inner.len();

        let mut age = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut convinced_when = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut degree = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut activation_potential = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut attitude = vec![vec![Attitude::Never; nagents]; nsims];
        let mut final_active_susceptible = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut final_prevalence = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut final_vaccinated = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut id = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut infected_by = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut infected_when = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut initial_active_susceptible = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut initial_vaccinated = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut removed_when = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut status = vec![vec![INIT_STATUS; nagents]; nsims];
        let mut threshold = vec![vec![0.0; nagents]; nsims];
        let mut vaccinated_when = vec![vec![INIT_USIZE; nagents]; nsims];
        let mut zealots = vec![vec![INIT_USIZE; nagents]; nsims];

        for s in 0..nsims {
            for i in 0..nagents {
                activation_potential[s][i] =
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .activation_potential
                        .unwrap();
                age[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .age
                    .unwrap();
                attitude[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .attitude
                    .unwrap();
                convinced_when[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner
                    [i]
                    .convinced_when
                    .unwrap();
                degree[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .degree
                    .unwrap();
                final_active_susceptible[s][i] =
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_active_susceptible
                        .unwrap();
                final_prevalence[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner
                    [i]
                    .final_prevalence
                    .unwrap();
                final_vaccinated[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner
                    [i]
                    .final_vaccinated
                    .unwrap();
                id[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .id
                    .unwrap();
                infected_by[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .infected_by
                    .unwrap();
                infected_when[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .infected_when
                    .unwrap();
                initial_active_susceptible[s][i] =
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .initial_active_susceptible
                        .unwrap();
                initial_vaccinated[s][i] =
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .initial_vaccinated
                        .unwrap();
                removed_when[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .removed_when
                    .unwrap();
                status[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .status
                    .unwrap();
                threshold[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .threshold
                    .unwrap();
                vaccinated_when[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner
                    [i]
                    .vaccinated_when
                    .unwrap();
                zealots[s][i] = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .zealots
                    .unwrap();
            }
        }

        AssembledAgentOutput::new(
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
            Some(removed_when),
            Some(status),
            Some(threshold),
            Some(vaccinated_when),
            Some(zealots),
        )
    }

    pub fn assemble_attitude_observables(&mut self) -> AssembledAttitudeOutput {
        let nsims = self.number_of_simulations();
        let nagents = self.inner()[0].agent_ensemble.as_ref().unwrap().inner.len();

        let mut activation_potential = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut active = vec![vec![0; PAR_ATTITUDE_GROUPS]; nsims];
        let mut age = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut cascading_threshold = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut convinced_when = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut degree = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut effective_threshold = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut final_active_susceptible = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut final_prevalence = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut final_vaccinated = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut infected_when = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut initial_active_susceptible = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut initial_vaccinated = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut prevalence = vec![vec![0; PAR_ATTITUDE_GROUPS]; nsims];
        let mut removed_when = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut vaccinated = vec![vec![0; PAR_ATTITUDE_GROUPS]; nsims];
        let mut vaccinated_when = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];
        let mut zealots = vec![vec![Vec::new(); PAR_ATTITUDE_GROUPS]; nsims];

        for s in 0..nsims {
            for i in 0..nagents {
                let attitude = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .attitude
                    .unwrap();
                let status = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .status
                    .unwrap();

                let a = match attitude {
                    Attitude::Vaccinated => 0,
                    Attitude::Soon => 1,
                    Attitude::Someone => 2,
                    Attitude::Most => 3,
                    Attitude::Never => 4,
                };

                age[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .age
                        .unwrap(),
                );
                degree[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .degree
                        .unwrap(),
                );

                activation_potential[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .activation_potential
                        .unwrap(),
                );
                cascading_threshold[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .cascading_threshold
                        .unwrap(),
                );
                effective_threshold[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .effective_threshold
                        .unwrap(),
                );
                final_active_susceptible[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_active_susceptible
                        .unwrap(),
                );
                final_prevalence[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_prevalence
                        .unwrap(),
                );
                final_vaccinated[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_vaccinated
                        .unwrap(),
                );
                initial_active_susceptible[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .initial_active_susceptible
                        .unwrap(),
                );
                initial_vaccinated[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .initial_vaccinated
                        .unwrap(),
                );
                zealots[s][a].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .zealots
                        .unwrap(),
                );

                match status {
                    Status::ActRem => {
                        active[s][a] += 1;
                        convinced_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .convinced_when
                                .unwrap(),
                        );
                        infected_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .infected_when
                                .unwrap(),
                        );
                        prevalence[s][a] += 1;
                        removed_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .removed_when
                                .unwrap(),
                        );
                    }
                    Status::ActSus => {
                        active[s][a] += 1;
                        convinced_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .convinced_when
                                .unwrap(),
                        );
                    }
                    Status::ActVac => {
                        active[s][a] += 1;
                        convinced_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .convinced_when
                                .unwrap(),
                        );
                        vaccinated[s][a] += 1;
                        vaccinated_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .vaccinated_when
                                .unwrap(),
                        );
                    }
                    Status::HesRem => {
                        infected_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .infected_when
                                .unwrap(),
                        );
                        prevalence[s][a] += 1;
                        removed_when[s][a].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .removed_when
                                .unwrap(),
                        );
                    }
                    _ => {}
                };
            }
        }

        AssembledAttitudeOutput::new(
            Some(activation_potential),
            Some(active),
            Some(age),
            Some(cascading_threshold),
            Some(convinced_when),
            Some(degree),
            Some(effective_threshold),
            Some(final_active_susceptible),
            Some(final_prevalence),
            Some(final_vaccinated),
            Some(infected_when),
            Some(initial_active_susceptible),
            Some(initial_vaccinated),
            Some(prevalence),
            Some(removed_when),
            Some(vaccinated),
            Some(vaccinated_when),
            Some(zealots),
        )
    }

    pub fn assemble_cluster_observables(&mut self) -> AssembledClusterOpinionHealthOutput {
        let nsims = self.number_of_simulations();

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
            as_clusters[s] = self.inner()[s]
                .cluster
                .as_ref()
                .unwrap()
                .opinion_health
                .as_ref()
                .unwrap()
                .as_cluster
                .clone();
            hs_clusters[s] = self.inner()[s]
                .cluster
                .as_ref()
                .unwrap()
                .opinion_health
                .as_ref()
                .unwrap()
                .hs_cluster
                .clone();
            ai_clusters[s] = self.inner()[s]
                .cluster
                .as_ref()
                .unwrap()
                .opinion_health
                .as_ref()
                .unwrap()
                .ai_cluster
                .clone();
            hi_clusters[s] = self.inner()[s]
                .cluster
                .as_ref()
                .unwrap()
                .opinion_health
                .as_ref()
                .unwrap()
                .hi_cluster
                .clone();
            ar_clusters[s] = self.inner()[s]
                .cluster
                .as_ref()
                .unwrap()
                .opinion_health
                .as_ref()
                .unwrap()
                .ar_cluster
                .clone();
            hr_clusters[s] = self.inner()[s]
                .cluster
                .as_ref()
                .unwrap()
                .opinion_health
                .as_ref()
                .unwrap()
                .hr_cluster
                .clone();
            av_clusters[s] = self.inner()[s]
                .cluster
                .as_ref()
                .unwrap()
                .opinion_health
                .as_ref()
                .unwrap()
                .av_cluster
                .clone();
            hv_clusters[s] = self.inner()[s]
                .cluster
                .as_ref()
                .unwrap()
                .opinion_health
                .as_ref()
                .unwrap()
                .hv_cluster
                .clone();
            ze_clusters[s] = self.inner()[s]
                .cluster
                .as_ref()
                .unwrap()
                .opinion_health
                .as_ref()
                .unwrap()
                .ze_cluster
                .clone();
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

    pub fn assemble_degree_observables(&mut self) -> AssembledDegreeOutput {
        let nsims = self.number_of_simulations();
        let nagents = self.inner()[0].agent_ensemble.as_ref().unwrap().inner.len();
        let eff_max_degree = (nagents as f64 / 10.0) as usize;

        let mut activation_potential = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut active = vec![vec![0; eff_max_degree]; nsims];
        let mut age = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut cascading_threshold = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut convinced_when = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut degree = vec![vec![0; eff_max_degree]; nsims];
        let mut effective_threshold = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut final_active_susceptible = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut final_prevalence = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut final_vaccinated = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut infected_when = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut initial_active_susceptible = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut initial_vaccinated = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut prevalence = vec![vec![0; eff_max_degree]; nsims];
        let mut removed_when = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut vaccinated = vec![vec![0; eff_max_degree]; nsims];
        let mut vaccinated_when = vec![vec![Vec::new(); eff_max_degree]; nsims];
        let mut zealots = vec![vec![Vec::new(); eff_max_degree]; nsims];

        for s in 0..nsims {
            for i in 0..nagents {
                let a = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .age
                    .unwrap();
                let k = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .degree
                    .unwrap();
                let status = self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                    .status
                    .unwrap();

                age[s][k].push(a);
                degree[s][k] += 1;

                activation_potential[s][k].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .activation_potential
                        .unwrap(),
                );
                cascading_threshold[s][k].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .cascading_threshold
                        .unwrap(),
                );
                effective_threshold[s][k].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .effective_threshold
                        .unwrap(),
                );
                final_active_susceptible[s][k].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_active_susceptible
                        .unwrap(),
                );
                final_prevalence[s][k].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_prevalence
                        .unwrap(),
                );
                final_vaccinated[s][k].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .final_vaccinated
                        .unwrap(),
                );
                initial_active_susceptible[s][k].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .initial_active_susceptible
                        .unwrap(),
                );
                initial_vaccinated[s][k].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .initial_vaccinated
                        .unwrap(),
                );
                zealots[s][k].push(
                    self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                        .zealots
                        .unwrap(),
                );

                match status {
                    Status::ActRem => {
                        active[s][k] += 1;
                        convinced_when[s][k].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .convinced_when
                                .unwrap(),
                        );
                        infected_when[s][k].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .infected_when
                                .unwrap(),
                        );
                        prevalence[s][k] += 1;
                        removed_when[s][k].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .removed_when
                                .unwrap(),
                        );
                    }
                    Status::ActSus => {
                        active[s][k] += 1;
                        convinced_when[s][k].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .convinced_when
                                .unwrap(),
                        );
                    }
                    Status::ActVac => {
                        active[s][k] += 1;
                        convinced_when[s][k].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .convinced_when
                                .unwrap(),
                        );
                        vaccinated[s][k] += 1;
                        vaccinated_when[s][k].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .vaccinated_when
                                .unwrap(),
                        );
                    }
                    Status::HesRem => {
                        infected_when[s][k].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .infected_when
                                .unwrap(),
                        );
                        prevalence[s][k] += 1;
                        removed_when[s][k].push(
                            self.inner_mut()[s].agent_ensemble.as_ref().unwrap().inner[i]
                                .removed_when
                                .unwrap(),
                        );
                    }
                    _ => {}
                };
            }
        }

        AssembledDegreeOutput::new(
            Some(activation_potential),
            Some(active),
            Some(age),
            Some(cascading_threshold),
            Some(convinced_when),
            Some(degree),
            Some(effective_threshold),
            Some(final_active_susceptible),
            Some(final_prevalence),
            Some(final_vaccinated),
            Some(infected_when),
            Some(initial_active_susceptible),
            Some(initial_vaccinated),
            Some(prevalence),
            Some(removed_when),
            Some(vaccinated),
            Some(vaccinated_when),
            Some(zealots),
        )
    }

    pub fn assemble_global_observables(&mut self) -> AssembledGlobalOutput {
        let nsims = self.number_of_simulations();
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

    pub fn assemble_time_series(&mut self, t_max: usize) -> AssembledTimeSeriesOutput {
        let nsims = self.number_of_simulations();
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
            for t in 0..t_max {
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
            t_array_st, ai_pop_st, ar_pop_st, as_pop_st, av_pop_st, hi_pop_st, hr_pop_st,
            hs_pop_st, hv_pop_st,
        )
    }

    pub fn to_pickle(
        &mut self,
        pars_model: &InputMultilayer,
        flags_output: &OutputFlags,
        string_epidemic: &str,
        string_multilayer: &str,
    ) {
        if flags_output.age {
            let assembled_age_output = self.assemble_age_observables();

            let output_to_serialize = SerializedAgeAssembly {
                age: assembled_age_output,
                pars: *pars_model,
            };

            let serialized = serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();

            let string_result = format!(
                "{}_{}_{}",
                HEADER_AGE, string_multilayer, string_epidemic
            );

            let mut path = PathBuf::from(PATH_RESULTS_CURATED_LOCAL);
            path.push(format!("{}{}", string_result, EXTENSION_RESULTS_PICKLE));
            std::fs::write(path, serialized).unwrap();
        }

        if flags_output.agent {
            let assembled_agent_output = self.assemble_agent_observables();

            let agent_stat_package = compute_agent_stats(&assembled_agent_output);

            let asp_serialized =
                serde_pickle::to_vec(&agent_stat_package, SerOptions::new()).unwrap();

            let string_result = format!(
                "{}_{}_{}",
                HEADER_AGENT_STATS, string_multilayer, string_epidemic,
            );

            let mut path = PathBuf::from(PATH_RESULTS_CURATED_LOCAL);
            path.push(format!("{}{}", string_result, EXTENSION_RESULTS_PICKLE));
            std::fs::write(path, asp_serialized).unwrap();

            let agent_distribution = compute_agent_distribution(&assembled_agent_output);

            let ad_serialized =
                serde_pickle::to_vec(&agent_distribution, SerOptions::new()).unwrap();

            let string_result = format!(
                "{}_{}_{}",
                HEADER_AGENT_DISTRIBUTION, string_multilayer, string_epidemic
            );

            let mut path = PathBuf::from(PATH_RESULTS_CURATED_LOCAL);
            path.push(format!("{}{}", string_result, EXTENSION_RESULTS_PICKLE));
            std::fs::write(path, ad_serialized).unwrap();
        }

        if flags_output.attitude {
            let assembled_attitude_output = self.assemble_attitude_observables();

            let output_to_serialize = SerializedAttitudeAssembly {
                attitude: assembled_attitude_output,
                pars: *pars_model,
            };

            let serialized = serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();

            let string_result = format!(
                "{}_{}_{}",
                HEADER_ATTITUDE, string_multilayer, string_epidemic
            );

            let mut path = PathBuf::from(PATH_RESULTS_CURATED_LOCAL);
            path.push(format!("{}{}", string_result, EXTENSION_RESULTS_PICKLE));
            std::fs::write(path, serialized).unwrap();
        }

        if flags_output.cluster {
            let assembled_cluster_output = self.assemble_cluster_observables();

            let cluster_stat_package = compute_cluster_stats(&assembled_cluster_output);
            let csp_serialized =
                serde_pickle::to_vec(&cluster_stat_package, SerOptions::new()).unwrap();

            let string_result = format!(
                "{}_{}_{}",
                HEADER_CLUSTER_STATS, string_multilayer, string_epidemic,
            );

            let mut path = PathBuf::from(PATH_RESULTS_CURATED_LOCAL);
            path.push(format!("{}{}", string_result, EXTENSION_RESULTS_PICKLE));
            std::fs::write(path, csp_serialized).unwrap();

            let cluster_distribution = compute_cluster_distribution(&assembled_cluster_output);

            let cd_serialized =
                serde_pickle::to_vec(&cluster_distribution, SerOptions::new()).unwrap();

            let string_result = format!(
                "{}_{}_{}",
                HEADER_CLUSTER_DISTRIBUTION, string_multilayer, string_epidemic
            );

            let mut path = PathBuf::from(PATH_RESULTS_CURATED_LOCAL);
            path.push(format!("{}{}", string_result, EXTENSION_RESULTS_PICKLE));
            std::fs::write(path, cd_serialized).unwrap();
        }

        if flags_output.degree {
            let assembled_degree_output = self.assemble_degree_observables();

            let output_to_serialize = SerializedDegreeAssembly {
                degree: assembled_degree_output,
                pars: *pars_model,
            };

            let serialized = serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();

            let string_result = format!(
                "degree_{}_{}",
                string_multilayer, string_epidemic
            );

            let mut path = PathBuf::from(PATH_RESULTS_CURATED_LOCAL);
            path.push(format!("{}{}", string_result, EXTENSION_RESULTS_PICKLE));
            std::fs::write(path, serialized).unwrap();
        }

        if flags_output.global {
            let assembled_global_output = self.assemble_global_observables();

            let output_to_serialize = SerializeGlobalAssembly {
                global: assembled_global_output,
                pars: *pars_model,
            };

            let serialized = serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();

            let string_result = format!(
                "{}_{}_{}",
                HEADER_GLOBAL, string_multilayer, string_epidemic
            );

            let mut path = PathBuf::from(PATH_RESULTS_CURATED_LOCAL);
            path.push(format!("{}{}", string_result, EXTENSION_RESULTS_PICKLE));
            std::fs::write(path, serialized).unwrap();
        }

        if flags_output.time {
            let assembled_time_series = self.assemble_time_series(pars_model.t_max);

            let serialized =
                serde_pickle::to_vec(&assembled_time_series, SerOptions::new()).unwrap();

            let string_result = format!(
                "{}_{}_{}",
                HEADER_TIME, string_multilayer, string_epidemic
            );

            let mut path = PathBuf::from(PATH_RESULTS_CURATED_LOCAL);
            path.push(format!("{}{}", string_result, EXTENSION_RESULTS_PICKLE));
            std::fs::write(path, serialized).unwrap();
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct OutputFlags {
    pub age: bool,
    pub agent: bool,
    pub attitude: bool,
    pub cluster: bool,
    pub degree: bool,
    pub global: bool,
    pub time: bool,
}

impl OutputFlags {
    pub fn new(
        age: bool,
        agent: bool,
        attitude: bool,
        cluster: bool,
        degree: bool,
        global: bool,
        time: bool,
    ) -> Self {
        Self {
            age,
            agent,
            attitude,
            cluster,
            degree,
            global,
            time,
        }
    }
}

#[derive(Serialize)]
pub struct SerializedAgeAssembly {
    pub age: AssembledAgeOutput,
    pub pars: InputMultilayer,
}

#[derive(Serialize)]
pub struct SerializedAgentAssembly {
    pub agent: AssembledAgentOutput,
    pub pars: InputMultilayer,
}

#[derive(Serialize)]
pub struct SerializedAttitudeAssembly {
    pub attitude: AssembledAttitudeOutput,
    pub pars: InputMultilayer,
}

#[derive(Serialize)]
pub struct SerializedClusterAssembly {
    pub attitude: Option<AssembledClusterAttitudeOutput>,
    pub cascading: Option<AssembledClusterCascadingOutput>,
    pub opinion_health: Option<AssembledClusterOpinionHealthOutput>,
    pub pars: InputMultilayer,
}

#[derive(Serialize)]
pub struct SerializedDegreeAssembly {
    pub degree: AssembledDegreeOutput,
    pub pars: InputMultilayer,
}

#[derive(Serialize)]
pub struct SerializeGlobalAssembly {
    pub global: AssembledGlobalOutput,
    pub pars: InputMultilayer,
}

#[derive(Serialize)]
pub struct SerializedTimeSeriesAssembly {
    pub time: AssembledTimeSeriesOutput,
    pub pars: InputMultilayer,
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

#[derive(Debug, Deserialize, Serialize)]
struct VaccinationData {
    #[serde(rename = "state")]
    state_name: String,
    fractions: Vec<f64>,
}

#[derive(Serialize, Deserialize, Clone, Copy)]
pub struct VaccinationPars {
    pub flag_underage: bool,
    pub fraction_majority: f64,
    pub fraction_someone: f64,
    pub fraction_soon: f64,
    pub fraction_vaccinated: f64,
    pub fraction_zealot: f64,
    pub model_hesitancy: HesitancyModel,
    pub model_region: Region,
    pub policy_vaccination: VaccinationPolicy,
    pub quota_vaccination: f64,
    pub rate_vaccination: f64,
    pub threshold_age: usize,
}

impl VaccinationPars {
    pub fn new(
        flag_underage: bool,
        fraction_majority: f64,
        fraction_someone: f64,
        fraction_soon: f64,
        fraction_vaccinated: f64,
        fraction_zealot: f64,
        model_hesitancy: HesitancyModel,
        model_region: Region,
        policy_vaccination: VaccinationPolicy,
        quota_vaccination: f64,
        rate_vaccination: f64,
        threshold_age: usize,
    ) -> Self {
        Self {
            flag_underage,
            fraction_majority,
            fraction_someone,
            fraction_soon,
            fraction_vaccinated,
            fraction_zealot,
            model_hesitancy,
            model_region,
            policy_vaccination,
            quota_vaccination,
            rate_vaccination,
            threshold_age,
        }
    }
}

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
        Self {
            bin_counts,
            bin_edges,
        }
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

impl Default for ClusterDistribution {
    fn default() -> Self {
        Self::new()
    }
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
    pub fn new(mean: f64, std: f64, l95: f64, u95: f64, min: f64, max: f64) -> Self {
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

pub fn build_normalized_layered_cdf(values: &mut [Vec<f64>]) -> Vec<Vec<f64>> {
    values
        .iter_mut()
        .map(|layer| {
            let sum: f64 = layer.iter().sum();

            let is_normalized = (sum - 1.0).abs() < 1e-6;

            if !is_normalized {
                layer.iter_mut().for_each(|v| *v /= sum);
            }

            let mut cdf = Vec::new();
            let mut cum_sum = 0.0;
            for &value in layer.iter() {
                cum_sum += value;
                cdf.push(cum_sum);
            }

            cdf
        })
        .collect()
}

fn calculate_cluster_sim_stats(simulations: &[Vec<usize>]) -> ClusterStatsOverSims {
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
    let variance =
        values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    let std = variance.sqrt();
    let z = 1.96;

    let l95 = mean - z * (std / (values.len() as f64).sqrt());
    let u95 = mean + z * (std / (values.len() as f64).sqrt());

    let min = *values
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .unwrap();
    let max = *values
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .unwrap();

    StatPacker::new(mean, std, l95, u95, min, max)
}

fn compute_agent_distribution(
    assembled_agent_output: &AssembledAgentOutput,
) -> AgentDistributionPacker {
    let convinced_when_seq_s = &assembled_agent_output.convinced_when.as_ref().unwrap();
    let degree_seq_s = &assembled_agent_output.degree.as_ref().unwrap();
    let final_active_seq_s = &assembled_agent_output
        .final_active_susceptible
        .as_ref()
        .unwrap();
    let final_prevalence_seq_s = &assembled_agent_output.final_prevalence.as_ref().unwrap();
    let final_vaccinated_seq_s = &assembled_agent_output.final_vaccinated.as_ref().unwrap();
    let infected_when_seq_s = &assembled_agent_output.infected_when.as_ref().unwrap();
    let initial_active_susceptible_seq_s = &assembled_agent_output
        .initial_active_susceptible
        .as_ref()
        .unwrap();
    let initial_vaccinated_seq_s = &assembled_agent_output.initial_vaccinated.as_ref().unwrap();
    let removed_when_seq_s = &assembled_agent_output.removed_when.as_ref().unwrap();
    let vaccinated_when_seq_s = &assembled_agent_output.vaccinated_when.as_ref().unwrap();
    let zealots_seq_s = &assembled_agent_output.zealots.as_ref().unwrap();

    let frac_final_active = compute_fractions(final_active_seq_s, degree_seq_s);
    let frac_final_prevalence = compute_fractions(final_prevalence_seq_s, degree_seq_s);
    let frac_final_vaccinated = compute_fractions(final_vaccinated_seq_s, degree_seq_s);
    let frac_initial_active_susceptible =
        compute_fractions(initial_active_susceptible_seq_s, degree_seq_s);
    let frac_initial_vaccinated = compute_fractions(initial_vaccinated_seq_s, degree_seq_s);
    let frac_zealots = compute_fractions(zealots_seq_s, degree_seq_s);

    let convinced_when_dist = compute_agent_sim_distribution(
        &convinced_when_seq_s
            .iter()
            .map(|vec| convert_to_f64(vec))
            .collect(),
    );
    let degree_dist = compute_agent_sim_distribution(
        &degree_seq_s.iter().map(|vec| convert_to_f64(vec)).collect(),
    );
    let frac_final_active_dist = compute_agent_sim_distribution(&frac_final_active);
    let frac_final_prevalence_dist = compute_agent_sim_distribution(&frac_final_prevalence);
    let frac_final_vaccinated_dist = compute_agent_sim_distribution(&frac_final_vaccinated);
    let infected_when_dist = compute_agent_sim_distribution(
        &infected_when_seq_s
            .iter()
            .map(|vec| convert_to_f64(vec))
            .collect(),
    );
    let frac_initial_active_susceptible_dist =
        compute_agent_sim_distribution(&frac_initial_active_susceptible);
    let frac_initial_vaccinated_dist = compute_agent_sim_distribution(&frac_initial_vaccinated);
    let removed_when_dist = compute_agent_sim_distribution(
        &removed_when_seq_s
            .iter()
            .map(|vec| convert_to_f64(vec))
            .collect(),
    );
    let vaccinated_when_dist = compute_agent_sim_distribution(
        &vaccinated_when_seq_s
            .iter()
            .map(|vec| convert_to_f64(vec))
            .collect(),
    );
    let frac_zealots_dist = compute_agent_sim_distribution(&frac_zealots);

    AgentDistributionPacker {
        convinced_when_dist,
        degree_dist,
        frac_final_active_dist,
        frac_final_prevalence_dist,
        frac_final_vaccinated_dist,
        infected_when_dist,
        frac_initial_active_susceptible_dist,
        frac_initial_vaccinated_dist,
        removed_when_dist,
        vaccinated_when_dist,
        frac_zealots_dist,
    }
}

fn compute_agent_sim_distribution(simulations: &Vec<Vec<f64>>) -> AgentDistribution {
    let mut all_values = Vec::new();
    for sim_vector in simulations {
        all_values.extend_from_slice(sim_vector);
    }

    let min_value = *all_values
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as usize;
    let max_value = *all_values
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap() as usize;

    let bin_size = (max_value - min_value) / PAR_NBINS;
    let bin_edges: Vec<usize> = (0..=PAR_NBINS).map(|i| min_value + i * bin_size).collect();

    let mut distribution = AgentDistribution::new(bin_edges.clone());

    for sim_vector in simulations {
        for &value in sim_vector {
            distribution.add_value(value);
        }
    }

    distribution
}

fn compute_agent_stats(assembled_agent_output: &AssembledAgentOutput) -> AgentStatPacker {
    let convinced_when_seq_s = &assembled_agent_output.convinced_when.as_ref().unwrap();
    let degree_seq_s = &assembled_agent_output.degree.as_ref().unwrap();
    let final_active_seq_s = &assembled_agent_output
        .final_active_susceptible
        .as_ref()
        .unwrap();
    let final_prevalence_seq_s = &assembled_agent_output.final_prevalence.as_ref().unwrap();
    let final_vaccinated_seq_s = &assembled_agent_output.final_vaccinated.as_ref().unwrap();
    let infected_when_seq_s = &assembled_agent_output.infected_when.as_ref().unwrap();
    let initial_active_susceptible_seq_s = &assembled_agent_output
        .initial_active_susceptible
        .as_ref()
        .unwrap();
    let initial_vaccinated_seq_s = &assembled_agent_output.initial_vaccinated.as_ref().unwrap();
    let removed_when_seq_s = &assembled_agent_output.removed_when.as_ref().unwrap();
    let vaccinated_when_seq_s = &assembled_agent_output.vaccinated_when.as_ref().unwrap();
    let zealots_seq_s = &assembled_agent_output.zealots.as_ref().unwrap();

    let frac_final_active = compute_fractions(final_active_seq_s, degree_seq_s);
    let frac_final_prevalence = compute_fractions(final_prevalence_seq_s, degree_seq_s);
    let frac_final_vaccinated = compute_fractions(final_vaccinated_seq_s, degree_seq_s);
    let frac_initial_active_susceptible =
        compute_fractions(initial_active_susceptible_seq_s, degree_seq_s);
    let frac_initial_vaccinated = compute_fractions(initial_vaccinated_seq_s, degree_seq_s);
    let frac_zealots = compute_fractions(zealots_seq_s, degree_seq_s);

    let convinced_when_stats = compute_stats(
        &convinced_when_seq_s
            .iter()
            .map(|vec| convert_to_f64(vec))
            .collect(),
    );
    let degree_stats = compute_stats(&degree_seq_s.iter().map(|vec| convert_to_f64(vec)).collect());
    let final_active_stats = compute_stats(&frac_final_active);
    let final_prevalence_stats = compute_stats(&frac_final_prevalence);
    let final_vaccinated_stats = compute_stats(&frac_final_vaccinated);
    let infected_when_stats = compute_stats(
        &infected_when_seq_s
            .iter()
            .map(|vec| convert_to_f64(vec))
            .collect(),
    );
    let initial_active_susceptible_stats = compute_stats(&frac_initial_active_susceptible);
    let initial_vaccinated_stats = compute_stats(&frac_initial_vaccinated);
    let removed_when_stats = compute_stats(
        &removed_when_seq_s
            .iter()
            .map(|vec| convert_to_f64(vec))
            .collect(),
    );
    let vaccinated_when_stats = compute_stats(
        &vaccinated_when_seq_s
            .iter()
            .map(|vec| convert_to_f64(vec))
            .collect(),
    );
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

fn compute_cluster_distribution(
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

fn compute_cluster_sim_distribution(simulations: &[Vec<usize>]) -> ClusterDistribution {
    let mut distribution = ClusterDistribution::new();

    for sim_vector in simulations {
        for &cluster_size in sim_vector {
            distribution.add_cluster(cluster_size);
        }
    }

    distribution
}

fn compute_cluster_stats(
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
    numerators: &[Vec<usize>],
    denominators: &[Vec<usize>],
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
    let variance = flat_values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (flat_values.len() - 1) as f64;
    let std = variance.sqrt();
    let z = 1.96;

    let l95 = mean - z * (std / (flat_values.len() as f64).sqrt());
    let u95 = mean + z * (std / (flat_values.len() as f64).sqrt());

    let min = *flat_values
        .iter()
        .min_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .unwrap();
    let max = *flat_values
        .iter()
        .max_by(|x, y| x.partial_cmp(y).unwrap_or(Ordering::Equal))
        .unwrap();

    StatPacker::new(mean, std, l95, u95, min, max)
}

pub fn construct_string_epidemic(model_pars: &InputMultilayer) -> String {
    format!("ua{0}_fa{1}_th{2}_rv{3}_fz{4}_mh{5}_mo{6}_ms{7}_nse{8}_nsi{9}_pv{10}_qv{11}_r0{12}_rr{13}_tm{14}_ta{15}",
        model_pars.flag_underage,
        model_pars.fraction_active,
        model_pars.threshold_opinion,
        model_pars.rate_vaccination,
        model_pars.fraction_zealot,
        convert_enum_hesitancy_to_string(model_pars.model_hesitancy, true),
        convert_enum_opinion_to_string(model_pars.model_opinion, true),
        convert_enum_seed_to_string(model_pars.model_seed, true),
        model_pars.nseeds,
        model_pars.nsims,
        convert_enum_vaccination_to_string(model_pars.policy_vaccination, true),
        model_pars.quota_vaccination,
        model_pars.r0,
        model_pars.rate_removal,
        model_pars.t_max,
        model_pars.threshold_age,
    )
}

pub fn construct_string_multilayer(model_region: Region, size: usize) -> String {
    format!("ml{0}_n{1}", model_region, size)
}

fn convert_to_f64(vec: &[usize]) -> Vec<f64> {
    vec.iter().map(|&value| value as f64).collect()
}

pub fn compute_interlayer_probability_matrix(contact_matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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

    for (alpha, row) in contact_matrix.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        intralayer_average_degree[alpha] = sum;
    }

    intralayer_average_degree
}

fn construct_file_path(
    base_path: &Path,
    header: &str,
    file_name_base: &str,
    extension: &str,
) -> Result<PathBuf, std::io::Error> {
    Ok(base_path.join(format!(
        "{}{}{}{}",
        HEADER_PROJECT, header, file_name_base, extension
    )))
}

fn convert_enum_hesitancy_to_string(
    model_hesitancy: HesitancyModel,
    flag_short: bool,
) -> String {
    match model_hesitancy {
        HesitancyModel::Adult => {
            if flag_short {
                "ADU".to_owned()
            } else {
                "Adult".to_owned()
            }
        }
        HesitancyModel::DataDriven => {
            if flag_short {
                "DD".to_owned()
            } else {
                "DataDriven".to_owned()
            }
        }
        HesitancyModel::Elder => {
            if flag_short {
                "ELD".to_owned()
            } else {
                "Elder".to_owned()
            }
        }
        HesitancyModel::ElderToYoung => {
            if flag_short {
                "EtY".to_owned()
            } else {
                "ElderToYoung".to_owned()
            }
        }
        HesitancyModel::Middleage => {
            if flag_short {
                "MID".to_owned()
            } else {
                "Middleage".to_owned()
            }
        }
        HesitancyModel::Random => {
            if flag_short {
                "RAN".to_owned()
            } else {
                "Random".to_owned()
            }
        }
        HesitancyModel::Underage => {
            if flag_short {
                "UND".to_owned()
            } else {
                "Underage".to_owned()
            }
        }
        HesitancyModel::Young => {
            if flag_short {
                "YOU".to_owned()
            } else {
                "Young".to_owned()
            }
        }
        HesitancyModel::YoungToElder => {
            if flag_short {
                "YtE".to_owned()
            } else {
                "YoungToElder".to_owned()
            }
        }
    }
}

fn convert_enum_opinion_to_string(
    model_opinion: OpinionModel, 
    flag_short: bool,
) -> String {
    match model_opinion {
        OpinionModel::DataDrivenThresholds => {
            if flag_short {
                "DD".to_owned()
            } else {
                "DataDriven".to_owned()
            }
        }
        OpinionModel::ElderCare => {
            if flag_short {
                "EC".to_owned()
            } else {
                "ElderCare".to_owned()
            }
        }
        OpinionModel::HomogeneousThresholds => {
            if flag_short {
                "HOT".to_owned()
            } else {
                "HomogeneousThresholds".to_owned()
            }
        }
        OpinionModel::HomogeneousWithZealots => {
            if flag_short {
                "HWZ".to_owned()
            } else {
                "HomogeneousWithZealots".to_owned()
            }
        }
        OpinionModel::Majority => {
            if flag_short {
                "MAJ".to_owned()
            } else {
                "Majority".to_owned()
            }
        }
    }
}

fn convert_enum_seed_to_string(
    model_seed: SeedModel, 
    flag_short: bool,
) -> String {
    match model_seed {
        SeedModel::BottomDegreeNeighborhood => {
            if flag_short {
                "BNE".to_owned()
            } else {
                "BottomDegreeNeighborhood".to_owned()
            }
        }
        SeedModel::BottomDegreeMultiLocus => {
            if flag_short {
                "BML".to_owned()
            } else {
                "BottomMultiLocus".to_owned()
            }
        }
        SeedModel::RandomMultiLocus => {
            if flag_short {
                "RML".to_owned()
            } else {
                "RandomMultiLocus".to_owned()
            }
        }
        SeedModel::RandomNeighborhood => {
            if flag_short {
                "RNE".to_owned()
            } else {
                "RandomNeighborhood".to_owned()
            }
        }
        SeedModel::TopDegreeMultiLocus => {
            if flag_short {
                "TML".to_owned()
            } else {
                "TopDegreMultiLocus".to_owned()
            }
        }
        SeedModel::TopDegreeNeighborhood => {
            if flag_short {
                "TNE".to_owned()
            } else {
                "TopDegreeNeighborhood".to_owned()
            }
        }
    }
}

fn convert_enum_vaccination_to_string(
    model_vaccination: VaccinationPolicy,
    flag_short: bool
) -> String {
    match model_vaccination {
        VaccinationPolicy::AgeAdult => {
            if flag_short {
                "AAD".to_owned()
            } else {
                "AgeAdult".to_owned()
            }
        }
        VaccinationPolicy::AgeElder => {
            if flag_short {
                "AEL".to_owned()
            } else {
                "AgeElder".to_owned()
            }
        }
        VaccinationPolicy::AgeMiddleage => {
            if flag_short {
                "AMI".to_owned()
            } else {
                "AgeMiddleage".to_owned()
            }
        }
        VaccinationPolicy::AgeTop => {
            if flag_short {
                "ATO".to_owned()
            } else {
                "AgeTop".to_owned()
            }
        }
        VaccinationPolicy::AgeUnderage => {
            if flag_short {
                "AUN".to_owned()
            } else {
                "AgeUnderage".to_owned()
            }
        }
        VaccinationPolicy::AgeYoung => {
            if flag_short {
                "AYO".to_owned()
            } else {
                "AgeYoung".to_owned()
            }
        }
        VaccinationPolicy::AgeYoungToElder => {
            if flag_short {
                "AYtE".to_owned()
            } else {
                "AgeYoungToElder".to_owned()
            }
        }
        VaccinationPolicy::Automatic => {
            if flag_short {
                "AUTO".to_owned()
            } else {
                "Automatic".to_owned()
            }
        }
        VaccinationPolicy::ComboElderTop => {
            if flag_short {
                "CET".to_owned()
            } else {
                "ComboElderTop".to_owned()
            }
        }
        VaccinationPolicy::ComboYoungTop => {
            if flag_short {
                "CYT".to_owned()
            } else {
                "ComboYoungTop".to_owned()
            }
        }
        VaccinationPolicy::DataDriven => {
            if flag_short {
                "DD".to_owned()
            } else {
                "DataDriven".to_owned()
            }
        }
        VaccinationPolicy::DegreeBottom => {
            if flag_short {
                "DBO".to_owned()
            } else {
                "DegreeBottom".to_owned()
            }
        }
        VaccinationPolicy::DegreeRandom => {
            if flag_short {
                "DRAN".to_owned()
            } else {
                "DegreeRandom".to_owned()
            }
        }
        VaccinationPolicy::DegreeTop => {
            if flag_short {
                "DTO".to_owned()
            } else {
                "DegreeTop".to_owned()
            }
        }
        VaccinationPolicy::Random=> {
            if flag_short {
                "RAN".to_owned()
            } else {
                "Random".to_owned()
            }
        }
    }
}

pub fn convert_hm_value_to_bool(hash_map: HashMap<String, Value>) -> HashMap<String, bool> {
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

pub fn convert_hm_value_to_f64(hash_map: HashMap<String, Value>) -> HashMap<String, f64> {
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
    population_vector.iter().take(18).sum()
}

pub fn create_output_files(
    file_name_base: String,
    output_pars: OutputFlags,
) -> Result<HashMap<String, File>, std::io::Error> {
    let mut output_file_map: HashMap<String, File> = HashMap::new();

    let base_path = env::current_dir()?.join(FOLDER_RESULTS);
    let extension = EXTENSION_RESULTS_PICKLE;

    let outputs = vec![
        (output_pars.age, HEADER_AGE),
        (output_pars.agent, HEADER_AGENT_STATS),
        (output_pars.attitude, HEADER_ATTITUDE),
        (output_pars.cluster, HEADER_CLUSTER_STATS),
        (output_pars.degree, HEADER_DEGREE),
        (output_pars.global, HEADER_GLOBAL),
        (output_pars.time, HEADER_TIME_STATS),
    ];

    for (enabled, header) in outputs {
        if enabled {
            let file_path = construct_file_path(&base_path, header, &file_name_base, extension)?;
            let file = File::create(&file_path)?;
            output_file_map.insert(header.to_owned(), file);
        }
    }

    Ok(output_file_map)
}

pub fn extract_region_and_nagents(input: &str) -> Result<(Region, usize), Box<dyn std::error::Error>> {
    let re = Regex::new(r"ml([A-Za-z\-]+)_n(\d+)_")?;
    let mut region_map: HashMap<&str, Region> = HashMap::new();

    region_map.insert("Alabama", Region::Alabama);
    region_map.insert("Alaska", Region::Alaska);
    region_map.insert("Arizona", Region::Arizona);
    region_map.insert("Arkansas", Region::Arkansas);
    region_map.insert("California", Region::California);
    region_map.insert("Colorado", Region::Colorado);
    region_map.insert("Connecticut", Region::Connecticut);
    region_map.insert("Delaware", Region::Delaware);
    region_map.insert("DistrictOfColumbia", Region::DistrictOfColumbia);
    region_map.insert("Florida", Region::Florida);
    region_map.insert("Georgia", Region::Georgia);
    region_map.insert("Hawaii", Region::Hawaii);
    region_map.insert("Idaho", Region::Idaho);
    region_map.insert("Illinois", Region::Illinois);
    region_map.insert("Indiana", Region::Indiana);
    region_map.insert("Iowa", Region::Iowa);
    region_map.insert("Kansas", Region::Kansas);
    region_map.insert("Kentucky", Region::Kentucky);
    region_map.insert("Louisiana", Region::Louisiana);
    region_map.insert("Maine", Region::Maine);
    region_map.insert("Maryland", Region::Maryland);
    region_map.insert("Massachusetts", Region::Massachusetts);
    region_map.insert("Michigan", Region::Michigan);
    region_map.insert("Minnesota", Region::Minnesota);
    region_map.insert("Mississippi", Region::Mississippi);
    region_map.insert("Missouri", Region::Missouri);
    region_map.insert("Montana", Region::Montana);
    region_map.insert("Nebraska", Region::Nebraska);
    region_map.insert("Nevada", Region::Nevada);
    region_map.insert("NewHampshire", Region::NewHampshire);
    region_map.insert("NewJersey", Region::NewJersey);
    region_map.insert("NewMexico", Region::NewMexico);
    region_map.insert("NewYork", Region::NewYork);
    region_map.insert("NorthCarolina", Region::NorthCarolina);
    region_map.insert("NorthDakota", Region::NorthDakota);
    region_map.insert("Ohio", Region::Ohio);
    region_map.insert("Oklahoma", Region::Oklahoma);
    region_map.insert("Oregon", Region::Oregon);
    region_map.insert("Pennsylvania", Region::Pennsylvania);
    region_map.insert("RhodeIsland", Region::RhodeIsland);
    region_map.insert("SouthCarolina", Region::SouthCarolina);
    region_map.insert("SouthDakota", Region::SouthDakota);
    region_map.insert("Tennessee", Region::Tennessee);
    region_map.insert("Texas", Region::Texas);
    region_map.insert("Utah", Region::Utah);
    region_map.insert("Vermont", Region::Vermont);
    region_map.insert("Virginia", Region::Virginia);
    region_map.insert("Washington", Region::Washington);
    region_map.insert("WestVirginia", Region::WestVirginia);
    region_map.insert("Wisconsin", Region::Wisconsin);
    region_map.insert("Wyoming", Region::Wyoming);
    region_map.insert("National", Region::National);
    region_map.insert("None", Region::None);

    if let Some(caps) = re.captures(input) {
        let region_str = &caps[1];
        let nagents_str = &caps[2];

        if let Some(&region) = region_map.get(region_str) {
            let nagents = nagents_str.parse::<usize>()?;
            Ok((region, nagents))
        } else {
            Err("Region not found.".into())
        }
    } else {
        Err("Input string does not match the expected pattern.".into())
    }
}

pub fn load_json_config_to_input_multilayer(
    filename: &str,
    subfolder: Option<&str>,
) -> Result<InputMultilayer, Box<dyn std::error::Error>> {
    let mut path = env::current_dir().expect("Failed to get current directory");

    let subfolder = subfolder.unwrap_or("config");
    path.push(subfolder);
    if !path.exists() {
        fs::create_dir(&path)?;
    }

    path.push(format!("{}.json", filename));

    if !path.exists() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("File not found: {:?}", path),
        )));
    }

    let mut file = File::open(&path)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;

    let config: InputMultilayer = serde_json::from_str(&contents)?;

    Ok(config)
}

pub fn load_multilayer_object(path_multilayer: &PathBuf) -> Multilayer {
    println!("{:?}", path_multilayer);
    let file = File::open(path_multilayer).expect("Unable to open file");
    let r = BufReader::new(file);
    let options = DeOptions::new();
    let result: Result<Multilayer, _> = serde_pickle::from_reader(r, options);

    result.expect("Failed to deserialize data")
}

pub fn measure_attitude_clusters(agent_ensemble: &AgentEnsemble) -> ClusterAttitudeOutput {
    let mut already_clusters = Vec::new();
    let mut soon_clusters = Vec::new();
    let mut someone_clusters = Vec::new();
    let mut majority_clusters = Vec::new();
    let mut never_clusters = Vec::new();

    // Initialize visited array and stack
    let nagents = agent_ensemble.number_of_agents();
    let mut clustered = vec![false; nagents];
    let mut stack = Vec::new();

    // DFS algorithm for building cluster distribution
    for a in 0..nagents {
        if !clustered[a] {
            // Initialize new cluster
            let mut cluster = Vec::new();
            stack.push(a);
            clustered[a] = true;

            // Get attitude & threshold for departing node
            let a_attitude = agent_ensemble.inner()[a].attitude;
            let a_threshold = agent_ensemble.inner()[a].threshold;

            // Check for zealots
            if a_threshold > 1.0 {
                // DFS
                while let Some(u) = stack.pop() {
                    // Add u to the current cluster
                    cluster.push(u);
                    clustered[u] = true;

                    // Add unvisited neighbors with matching status or threshold to stack
                    for v in agent_ensemble.inner()[u].neighbors.clone() {
                        if !clustered[v] {
                            let v_threshold = agent_ensemble.inner()[v].threshold;
                            if v_threshold > 1.0 && a_threshold > 1.0 {
                                stack.push(v);
                                clustered[v] = true;
                            }
                        }
                    }
                }

                // Add cluster to corresponding vector
                never_clusters.push(cluster.len());
            } else {
                // DFS
                while let Some(u) = stack.pop() {
                    // Add u to the current cluster
                    cluster.push(u);
                    clustered[u] = true;

                    // Add unvisited neighbors with matching status or threshold to stack
                    for v in agent_ensemble.inner()[u].neighbors.clone() {
                        if !clustered[v] {
                            let v_status = agent_ensemble.inner()[v].attitude;
                            let v_threshold = agent_ensemble.inner()[v].threshold;
                            if v_status == a_attitude && (v_threshold < 1.0) {
                                stack.push(v);
                                clustered[v] = true;
                            }
                        }
                    }
                }
                // Add cluster to corresponding vector
                match a_attitude.unwrap() {
                    Attitude::Vaccinated => already_clusters.push(cluster.len()),
                    Attitude::Soon => soon_clusters.push(cluster.len()),
                    Attitude::Someone => someone_clusters.push(cluster.len()),
                    Attitude::Most => majority_clusters.push(cluster.len()),
                    Attitude::Never => never_clusters.push(cluster.len()),
                }
            }
        }
    }

    ClusterAttitudeOutput::new(
        already_clusters,
        soon_clusters,
        someone_clusters,
        majority_clusters,
        never_clusters,
    )
}

pub fn measure_cascading_clusters(agent_ensemble: &mut AgentEnsemble) -> ClusterCascadingOutput {
    let mut cascading_clusters = Vec::new();
    let mut nonzealot_clusters = Vec::new();

    // Initialize visited array and stack
    let nagents = agent_ensemble.number_of_agents();
    let mut ca_clustered = vec![false; nagents];
    let mut ca_stack = Vec::new();
    let mut nz_clustered = vec![false; nagents];
    let mut nz_stack = Vec::new();

    for a in 0..nagents {
        if !ca_clustered[a] {
            // Initialize new cluster
            let mut cluster = Vec::new();
            ca_stack.push(a);
            ca_clustered[a] = true;

            let a_threshold = agent_ensemble.inner()[a].threshold;
            let a_cascading_threshold = agent_ensemble.inner()[a].cascading_threshold.unwrap();

            // Check for zealots
            if a_threshold < 1.0 {
                // DFS
                while let Some(u) = ca_stack.pop() {
                    // Add u to the current cluster
                    cluster.push(u);
                    ca_clustered[u] = true;

                    // Add unvisited neighbors with matching status or threshold to stack
                    for v in agent_ensemble.inner()[u].neighbors.clone() {
                        if !ca_clustered[v] {
                            let v_cascading_threshold =
                                agent_ensemble.inner()[v].cascading_threshold.unwrap();
                            if v_cascading_threshold == a_cascading_threshold
                                && a_cascading_threshold == 0
                            {
                                ca_stack.push(v);
                                ca_clustered[v] = true;
                            }
                        }
                    }
                }

                // Add cluster to corresponding vector
                cascading_clusters.push(cluster.len());
            }
        }

        if !nz_clustered[a] {
            // Initialize new cluster
            let mut cluster = Vec::new();
            nz_stack.push(a);
            nz_clustered[a] = true;

            let a_threshold = agent_ensemble.inner()[a].threshold;

            // Check for zealots
            if a_threshold < 1.0 {
                // DFS
                while let Some(u) = ca_stack.pop() {
                    // Add u to the current cluster
                    cluster.push(u);
                    nz_clustered[u] = true;

                    // Add unvisited neighbors with matching status or threshold to stack
                    for v in agent_ensemble.inner()[u].neighbors.clone() {
                        if !nz_clustered[v] {
                            let v_threshold = agent_ensemble.inner()[v].threshold;
                            if v_threshold < 1.0 && a_threshold < 1.0 {
                                nz_stack.push(v);
                                nz_clustered[v] = true;
                            }
                        }
                    }
                }

                // Add cluster to corresponding vector
                nonzealot_clusters.push(cluster.len());
            }
        }
    }

    ClusterCascadingOutput::new(cascading_clusters, nonzealot_clusters)
}

pub fn measure_opinion_health_clusters(
    agent_ensemble: &AgentEnsemble,
) -> ClusterOpinionHealthOutput {
    let mut as_clusters = Vec::new();
    let mut hs_clusters = Vec::new();
    let mut ai_clusters = Vec::new();
    let mut hi_clusters = Vec::new();
    let mut ar_clusters = Vec::new();
    let mut hr_clusters = Vec::new();
    let mut av_clusters = Vec::new();
    let mut hv_clusters = Vec::new();
    let mut ze_clusters = Vec::new();

    // Initialize visited array and stack
    let nagents = agent_ensemble.number_of_agents();
    let mut clustered = vec![false; nagents];
    let mut stack = Vec::new();

    // DFS algorithm for building cluster distribution
    for a in 0..nagents {
        if !clustered[a] {
            // Initialize new cluster
            let mut cluster = Vec::new();
            stack.push(a);
            clustered[a] = true;

            // Get status & threshold for departing node
            let a_status = agent_ensemble.inner()[a].status;
            let a_threshold = agent_ensemble.inner()[a].threshold;

            // Check for zealots
            if a_threshold > 1.0 {
                // DFS
                while let Some(u) = stack.pop() {
                    // Add u to the current cluster
                    cluster.push(u);
                    clustered[u] = true;

                    // Add unvisited neighbors with matching status or threshold to stack
                    for v in agent_ensemble.inner()[u].neighbors.clone() {
                        if !clustered[v] {
                            let v_threshold = agent_ensemble.inner()[v].threshold;
                            if v_threshold > 1.0 && a_threshold > 1.0 {
                                stack.push(v);
                                clustered[v] = true;
                            }
                        }
                    }
                }

                // Add cluster to corresponding vector
                ze_clusters.push(cluster.len());
            } else {
                // DFS
                while let Some(u) = stack.pop() {
                    // Add u to the current cluster
                    cluster.push(u);
                    clustered[u] = true;

                    // Add unvisited neighbors with matching status or threshold to stack
                    for v in agent_ensemble.inner()[u].neighbors.clone() {
                        if !clustered[v] {
                            let v_status = agent_ensemble.inner()[v].status;
                            let v_threshold = agent_ensemble.inner()[v].threshold;
                            if v_status == a_status && (v_threshold < 1.0) {
                                stack.push(v);
                                clustered[v] = true;
                            }
                        }
                    }
                }
                // Add cluster to corresponding vector
                match a_status {
                    Status::ActSus => as_clusters.push(cluster.len()),
                    Status::HesSus => hs_clusters.push(cluster.len()),
                    Status::ActInf => ai_clusters.push(cluster.len()),
                    Status::HesInf => hi_clusters.push(cluster.len()),
                    Status::ActRem => ar_clusters.push(cluster.len()),
                    Status::HesRem => hr_clusters.push(cluster.len()),
                    Status::ActVac => av_clusters.push(cluster.len()),
                    Status::HesVac => hv_clusters.push(cluster.len()),
                }
            }
        }
    }

    ClusterOpinionHealthOutput::new(
        ai_clusters,
        as_clusters,
        ar_clusters,
        av_clusters,
        hi_clusters,
        hr_clusters,
        hs_clusters,
        hv_clusters,
        ze_clusters,
    )
}

pub fn measure_neighborhood(agent_id: usize, agent_ensemble: &mut AgentEnsemble, t: usize) {
    let neighbors = agent_ensemble.inner_mut()[agent_id].neighbors.clone();
    let mut active_susceptible = 0;
    let mut vaccinated = 0;
    let mut zealots = 0;
    let mut prevalence = 0;

    for neigh in neighbors {
        let status = agent_ensemble.inner()[neigh].status;
        let threshold = agent_ensemble.inner()[neigh].threshold;
        if threshold > 1.0 {
            zealots += 1;
        } else {
            match status {
                Status::ActSus => active_susceptible += 1,
                Status::ActVac => vaccinated += 1,
                Status::ActRem | Status::HesRem => prevalence += 1,
                _ => (),
            }
        }
    }

    if t == 0 {
        agent_ensemble.inner_mut()[agent_id].initial_active_susceptible = Some(active_susceptible);
        agent_ensemble.inner_mut()[agent_id].initial_vaccinated = Some(vaccinated);
        agent_ensemble.inner_mut()[agent_id].zealots = Some(zealots);
    } else {
        agent_ensemble.inner_mut()[agent_id].final_active_susceptible = Some(active_susceptible);
        agent_ensemble.inner_mut()[agent_id].final_vaccinated = Some(vaccinated);
        agent_ensemble.inner_mut()[agent_id].zealots = Some(zealots);
        agent_ensemble.inner_mut()[agent_id].final_prevalence = Some(prevalence);
    }
}

pub fn read_key_and_f64_from_json(state: Region, filename: &str) -> f64 {
    let mut path = env::current_dir().expect("Failed to get current directory");
    path.push("data");
    path.push(format!("{}.json", filename));

    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let data: Value = serde_json::from_str(&contents).unwrap();

    let state_str = state.to_string();
    let state_data = &data[state_str];
    let data_f64: f64 =
        serde_json::from_value(state_data.clone()).expect("Failed to parse state data");

    data_f64
}

pub fn read_key_and_matrixf64_from_json(state: Region, filename: &str) -> Vec<Vec<f64>> {
    let mut path = env::current_dir().expect("Failed to get current directory");

    path.push(FOLDER_DATA_CURATED);
    path.push(format!("{}.json", filename));

    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let data: Value = serde_json::from_str(&contents).unwrap();

    let state_str = state.to_string();
    let state_data = &data[state_str];
    let data_matrixf64: Vec<Vec<f64>> =
        serde_json::from_value(state_data.clone()).expect("Failed to parse state data");

    data_matrixf64
}

pub fn read_key_and_vecf64_from_json(state: Region, filename: &str) -> Vec<f64> {
    let mut path = env::current_dir().expect("Failed to get current directory");
    path.push(FOLDER_DATA_CURATED);
    path.push(format!("{}.json", filename));

    let mut file = File::open(&path).unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents).unwrap();

    let data: Value = serde_json::from_str(&contents).unwrap();

    let state_str = state.to_string();
    let state_data = &data[state_str];
    let data_vecf64: Vec<f64> =
        serde_json::from_value(state_data.clone()).expect("Failed to parse state data");

    data_vecf64
}

pub fn rebuild_age_distribution(node_ensemble: &Multilayer) -> Vec<f64> {
    let nagents = node_ensemble.number_of_nodes();
    let mut age_distribution = vec![0.0; PAR_AGE_GROUPS];

    for (_, node) in node_ensemble.inner().iter().enumerate() {
        let age_i = node.layer;
        age_distribution[age_i] += 1.0 / nagents as f64;
    }

    age_distribution
}

pub fn rebuild_contact_matrix(node_ensemble: &Multilayer) -> Vec<Vec<f64>> {
    let mut age_distribution = vec![0.0; PAR_AGE_GROUPS];
    let mut contact_matrix = vec![vec![0.0; PAR_AGE_GROUPS]; PAR_AGE_GROUPS];

    for (_, node) in node_ensemble.inner().iter().enumerate() {
        let a_i = node.layer;
        let neighbors = node.neighbors.clone();
        for j in neighbors {
            let a_j = node_ensemble.inner()[j].layer;
            contact_matrix[a_i][a_j] += 1.0;
        }
        age_distribution[a_i] += 1.0;
    }

    for a_i in 0..PAR_AGE_GROUPS {
        for a_j in 0..PAR_AGE_GROUPS {
            contact_matrix[a_i][a_j] /= age_distribution[a_i];
        }
    }
    contact_matrix
}

pub fn rebuild_contact_data_from_multilayer(node_ensemble: &Multilayer) -> ContactOutput {
    let age_distribution = rebuild_age_distribution(node_ensemble);
    let contact_matrix = rebuild_contact_matrix(node_ensemble);
    let degree_distribution = rebuild_degree_distribution(node_ensemble);

    ContactOutput {
        age_distribution,
        contact_matrix,
        degree_distribution,
    }
}

pub fn rebuild_degree_distribution(node_ensemble: &Multilayer) -> Vec<f64> {
    let nagents = node_ensemble.number_of_nodes();
    let mut degree_distribution = vec![0.0; PAR_AGE_GROUPS];

    for (_, node) in node_ensemble.inner().iter().enumerate() {
        degree_distribution[node.degree] += 1.0 / nagents as f64;
    }

    degree_distribution
}

pub fn remove_duplicates(vec: Vec<usize>) -> Vec<usize> {
    let set: HashSet<_> = vec.into_iter().collect();
    set.into_iter().collect()
}

pub fn sample_from_cdf(cdf: &[f64], rng: &mut ThreadRng) -> usize {
    let u: f64 = rng.gen();
    cdf.iter().position(|&value| value >= u).unwrap_or(0)
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
