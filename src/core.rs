use netrust::network::Network;
use rand::prelude::*;

use crate::{
    agent::{AgentEnsemble, Status}, 
    utils::{Input, OutputResults, GlobalOutput, TimeOutput, TimeUnitPop, OutputEnsemble, remove_duplicates, AgentEnsembleOutput, ClusterOutput}, 
    analysis::{sir_prevalence, compute_beta_from_r0}};

pub fn measure_neighborhood(
    agent_id: usize, 
    agent_ensemble: &mut AgentEnsemble, 
    t: usize,
) {
    let neighbors = agent_ensemble.inner_mut()[agent_id].neighbors.as_ref().unwrap().clone();
    let mut active_susceptible = 0;
    let mut vaccinated = 0;
    let mut zealots = 0;
    let mut prevalence = 0;
    for neigh in neighbors {
        let status = agent_ensemble.inner()[neigh].status;
        let threshold = agent_ensemble.inner()[neigh].threshold;
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

pub fn measure_clusters(agent_ensemble: &AgentEnsemble) -> ClusterOutput {
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
                    for v in agent_ensemble.inner()[u].neighbors.as_ref().unwrap() {
                        if !clustered[*v] {
                            let v_threshold = agent_ensemble.inner()[*v].threshold;
                            if v_threshold > 1.0 && a_threshold > 1.0 {
                                stack.push(*v);
                                clustered[*v] = true;
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
                    for v in agent_ensemble.inner()[u].neighbors.as_ref().unwrap() {
                        if !clustered[*v] {
                            let v_status = agent_ensemble.inner()[*v].status;
                            let v_threshold = agent_ensemble.inner()[*v].threshold;
                            if v_status == a_status && (v_threshold < 1.0) {
                                stack.push(*v);
                                clustered[*v] = true;
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

    ClusterOutput::new(
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

pub fn watts_threshold_step_agent(
    agent_id: usize, 
    agent_ensemble: &AgentEnsemble
) -> bool {
    // Get focal agent status, neighbors and degree
    let status = agent_ensemble.inner()[agent_id].status;
    let neighbors = agent_ensemble.inner()[agent_id].neighbors.as_ref().unwrap();
    let k = neighbors.len();
    // Get focal agent vaccinated neighbor fraction
    let mut vaccinated_neighbors = 0.0;
    for neigh in neighbors {
        if agent_ensemble.inner()[*neigh].status == Status::ActVac {
            vaccinated_neighbors += 1.0;
        }
    }
    let vaccinated_fraction = vaccinated_neighbors / k as f64;
    // Check if there is an opinion update based on focal agent's threshold
    if vaccinated_fraction >= agent_ensemble.inner()[agent_id].threshold {
        match status {
            Status::HesSus => {
                true
            },
            Status::HesInf => {
                true
            },
            Status::HesRem => {
                true
            },
            _ => false,
        } 
    } else {
        false
    }
}

pub fn watts_threshold_step_subensemble(
    original_list: &[usize], 
    agent_ensemble: &mut AgentEnsemble,
    branching_list: &mut Vec<usize>,
    _status_to_update: Status,
    t: usize,
) -> Vec<usize> {
    let mut new_list = Vec::new();
    // Loop over hesitant susceptible agents
    for a in original_list.iter() {
        let agent_id = *a;
        let status = agent_ensemble.inner()[agent_id].status;
        let change = watts_threshold_step_agent(agent_id, agent_ensemble);
        if change {
            match status {
                Status::HesSus => {
                    agent_ensemble.inner_mut()[agent_id].status = Status::ActSus;
                    agent_ensemble.inner_mut()[agent_id].convinced_when = Some(t);
                    branching_list.push(agent_id);
                },
                Status::HesInf => {
                    agent_ensemble.inner_mut()[agent_id].status = Status::ActInf;
                    agent_ensemble.inner_mut()[agent_id].convinced_when = Some(t);
                    branching_list.push(agent_id);
                },
                Status::HesRem => {
                    agent_ensemble.inner_mut()[agent_id].status = Status::ActRem;
                    agent_ensemble.inner_mut()[agent_id].convinced_when = Some(t);
                    branching_list.push(agent_id);
                }
                _ => {},
            } 
        } else {
            match status {
                Status::HesSus => {
                    new_list.push(agent_id);
                },
                Status::HesInf => {
                    new_list.push(agent_id);
                },
                Status::HesRem => {
                    new_list.push(agent_id);
                }
                _ => {},
            } 
        }
    }
    new_list
}

pub fn vaccination_step_agent(vaccination_rate: f64) -> bool {
    let mut rng = rand::thread_rng();
    let trial: f64 = rng.gen();
    trial < vaccination_rate
}

pub fn vaccination_step_subensemble(
    as_list: &[usize], 
    agent_ensemble: &mut AgentEnsemble,
    av_list: &mut Vec<usize>,
    pars: &Input,
    t: usize,
) -> Vec<usize> {
    let vaccination_rate = pars.epidemic.vaccination_rate;
    let mut new_as_list = Vec::new();
    for a in as_list.iter() {
        let agent_id = *a;
        let change = vaccination_step_agent(vaccination_rate);
        if change {
            agent_ensemble.inner_mut()[agent_id].status = Status::ActVac;
            agent_ensemble.inner_mut()[agent_id].vaccinated_when = Some(t);
            av_list.push(agent_id);
        } else {
            new_as_list.push(agent_id);
        }
    }
    new_as_list
}

pub fn infection_step_agent(
    agent_id: usize, 
    agent_ensemble: &AgentEnsemble, 
    pars: &Input
) -> (Vec<usize>, Vec<usize>) {
    let mut hes_inf_a = Vec::new();
    let mut act_inf_a = Vec::new();
    let mut rng = rand::thread_rng();

    let neighbors = agent_ensemble.inner()[agent_id].neighbors.as_ref().unwrap();
    for neighbor in neighbors {
        let neighbor_id = *neighbor;
        let neighbor_status = agent_ensemble.inner()[neighbor_id].status;
        if neighbor_status == Status::HesSus {
            let trial: f64 = rng.gen();
            if trial < pars.epidemic.infection_rate {
                hes_inf_a.push(neighbor_id);
            }
        } else if neighbor_status == Status::ActSus {
            let trial: f64 = rng.gen();
            if trial < pars.epidemic.infection_rate {
                act_inf_a.push(neighbor_id);
            }
        }
    }
    (hes_inf_a, act_inf_a)
}

pub fn removal_step_agent(removal_rate: f64) -> bool {
    let mut rng = rand::thread_rng();
    let trial: f64 = rng.gen();
    trial < removal_rate
}

pub fn infection_and_removal_step_subensemble(
    focal_list: &[usize], 
    collateral_list: &[usize],
    agent_ensemble: &mut AgentEnsemble,
    branching_list: &mut Vec<usize>,
    focal_status: Status,
    pars: &Input,
    t: usize,
) -> (Vec<usize>, Vec<usize>) {
    // Infection & removal steps for all types of infected agents //hi_list: focal, ai_list: collateral, hr_list: branching
    let removal_rate = pars.epidemic.infection_decay;
    let mut new_focal_list = Vec::new(); //focal_list.clone();
    let mut new_collateral_list = collateral_list.to_owned();
    for a in focal_list.iter() {
        let agent_id = *a;
        // Perform infection step for focal agent
        let (new_hes_inf, new_act_inf) = infection_step_agent(agent_id, agent_ensemble, pars);
        // Update new infected statuses
        for hi in new_hes_inf.iter() {
            agent_ensemble.inner_mut()[*hi].status = Status::HesInf;
            agent_ensemble.inner_mut()[*hi].infected_when = Some(t);
            agent_ensemble.inner_mut()[*hi].infected_by = Some(*a);
        }
        for ai in new_act_inf.iter() {
            agent_ensemble.inner_mut()[*ai].status = Status::ActInf;
            agent_ensemble.inner_mut()[*ai].infected_when = Some(t);
            agent_ensemble.inner_mut()[*ai].infected_by = Some(*a);
        }
        // Update infected lists
        if focal_status == Status::HesInf {
            new_focal_list.extend(new_hes_inf);
            new_collateral_list.extend(new_act_inf);
        } else {
            new_focal_list.extend(new_act_inf);
            new_collateral_list.extend(new_hes_inf);
        }
        // Perform removal step for focal agent. Then update status & list
        let change = removal_step_agent(removal_rate);
        if change {
            if focal_status == Status::HesInf {
                agent_ensemble.inner_mut()[agent_id].status = Status::HesRem;
                agent_ensemble.inner_mut()[agent_id].removed_when = Some(t);
            }
            else if focal_status == Status::ActInf {
                agent_ensemble.inner_mut()[agent_id].status = Status::ActRem;
                agent_ensemble.inner_mut()[agent_id].removed_when = Some(t);
            }
            branching_list.push(agent_id);
        } else {
            new_focal_list.push(agent_id);
        }
    }
    let new_collateral_list = remove_duplicates(new_collateral_list);
    (new_focal_list, new_collateral_list)
}

pub fn opinion_activation_step(agent_id: usize, agent_ensemble: &AgentEnsemble) -> bool {
    // Get focal agent status, neighbors and degree
    let status = agent_ensemble.inner()[agent_id].status;
    let neighbors = agent_ensemble.inner()[agent_id].neighbors.as_ref().unwrap();
    let k = neighbors.len();
    // Get focal agent vaccinated neighbor fraction
    let mut vaccinated_neighbors = 0.0;
    for neigh in neighbors {
        if agent_ensemble.inner()[*neigh].status == Status::ActVac {
            vaccinated_neighbors += 1.0;
        }
    }
    let vaccinated_fraction = vaccinated_neighbors / k as f64;
    // Check if there is an opinion update based on focal agent's threshold
    if vaccinated_fraction >= agent_ensemble.inner()[agent_id].threshold {
        match status {
            Status::HesSus => {
                true
            },
            Status::HesInf => {
                true
            },
            Status::HesRem => {
                true
            },
            _ => false,
        } 
    } else {
        false
    }
}

pub fn opinion_deactivation_step(agent_id: usize, agent_ensemble: &AgentEnsemble) -> bool {
    // Get focal agent status, neighbors and degree
    let status = agent_ensemble.inner()[agent_id].status;
    let neighbors = agent_ensemble.inner()[agent_id].neighbors.as_ref().unwrap();
    let k = neighbors.len();
    // Get focal agent vaccinated neighbor fraction
    let mut vaccinated_neighbors = 0.0;
    for neigh in neighbors {
        if agent_ensemble.inner()[*neigh].status == Status::ActVac {
            vaccinated_neighbors += 1.0;
        }
    }
    let vaccinated_fraction = vaccinated_neighbors / k as f64;
    // Check if there is an opinion update based on focal agent's threshold
    if vaccinated_fraction < agent_ensemble.inner()[agent_id].threshold {
        match status {
            Status::ActSus => {
                true
            },
            Status::ActInf => {
                true
            },
            Status::ActRem => {
                true
            },
            _ => false,
        } 
    } else {
        false
    }
}

pub fn symmetrical_opinion_and_vaccination_step_subensemble(
    new_hs_list: &mut Vec<usize>,
    new_as_list: &mut Vec<usize>,
    new_hi_list: &mut Vec<usize>,
    new_ai_list: &mut Vec<usize>,
    new_hr_list: &mut Vec<usize>,
    new_ar_list: &mut Vec<usize>,
    new_av_list: &mut Vec<usize>,
    new_hv_list: &mut Vec<usize>,
    status_new: &mut Vec<Status>,
    agent_ensemble: &AgentEnsemble,
    pars: &Input,
) {
    let status_old = status_new.clone();
    let vaccination_rate = pars.epidemic.vaccination_rate;

    for (agent_id, &status) in status_old.iter().enumerate() {
        match status {
            Status::HesSus => {
                let change = opinion_activation_step(agent_id, agent_ensemble);
                if change {
                    let change = vaccination_step_agent(vaccination_rate);
                    if change {
                        status_new[agent_id] = Status::ActVac;
                        new_av_list.push(agent_id);
                    }
                    else {
                        status_new[agent_id] = Status::ActSus;
                        new_as_list.push(agent_id);
                    }
                    let index = new_hs_list.iter().position(|&x| x == agent_id).unwrap();
                    new_hs_list.remove(index);
                }
            },
            Status::HesInf => {
                let change = opinion_activation_step(agent_id, agent_ensemble);
                if change {
                    status_new[agent_id] = Status::ActInf;
                    new_ai_list.push(agent_id);
                    let index = new_hi_list.iter().position(|&x| x == agent_id).unwrap();
                    new_hi_list.remove(index);
                }
            },
            Status::HesRem => {
                let change = opinion_activation_step(agent_id, agent_ensemble);
                if change {
                    status_new[agent_id] = Status::ActRem;
                    new_ar_list.push(agent_id);
                    let index = new_hr_list.iter().position(|&x| x == agent_id).unwrap();
                    new_hr_list.remove(index);
                }
            },
            Status::HesVac => {
                let change = opinion_activation_step(agent_id, agent_ensemble);
                if change {
                    status_new[agent_id] = Status::ActVac;
                    new_av_list.push(agent_id);
                    let index = new_hv_list.iter().position(|&x| x == agent_id).unwrap();
                    new_hv_list.remove(index);
                }
            },
            Status::ActSus => {
                let change = opinion_deactivation_step(agent_id, agent_ensemble);
                if change {
                    status_new[agent_id] = Status::HesSus;
                    new_hs_list.push(agent_id);
                } else {
                    let change = vaccination_step_agent(vaccination_rate);
                    if change {
                        status_new[agent_id] = Status::ActVac;
                        new_av_list.push(agent_id);
                    } 
                }
                let index = new_as_list.iter().position(|&x| x == agent_id).unwrap();
                new_as_list.remove(index);
            },
            Status::ActInf => {
                let change = opinion_deactivation_step(agent_id, agent_ensemble);
                if change {
                    status_new[agent_id] = Status::HesInf;
                    new_hi_list.push(agent_id);
                    let index = new_ai_list.iter().position(|&x| x == agent_id).unwrap();
                    new_ai_list.remove(index);
                }
            },
            Status::ActRem => {
                let change = opinion_deactivation_step(agent_id, agent_ensemble);
                if change {
                    status_new[agent_id] = Status::HesRem;
                    new_hr_list.push(agent_id);
                    let index = new_ar_list.iter().position(|&x| x == agent_id).unwrap();
                    new_ar_list.remove(index);
                }
            },
            Status::ActVac => {
                let change = opinion_deactivation_step(agent_id, agent_ensemble);
                if change {
                    status_new[agent_id] = Status::HesVac;
                    new_hv_list.push(agent_id);
                    let index = new_av_list.iter().position(|&x| x == agent_id).unwrap();
                    new_av_list.remove(index);
                }
            },
        }
    }
}

pub fn symmetrical_infection_and_removal_step_subensemble(
    new_hs_list: &mut Vec<usize>,
    new_as_list: &mut Vec<usize>,
    new_hi_list: &mut Vec<usize>,
    new_ai_list: &mut Vec<usize>,
    new_hr_list: &mut Vec<usize>,
    new_ar_list: &mut Vec<usize>,
    status_new: &mut Vec<Status>,
    agent_ensemble: &AgentEnsemble,
    pars: &Input,
) {
    let hi_list = new_hi_list.clone();
    let ai_list = new_ai_list.clone();
    let removal_rate = pars.epidemic.infection_decay;

    for a in hi_list.iter() {
        let agent_id = *a;
        // Perform infection step for focal agent
        let (new_hes_inf, new_act_inf) = infection_step_agent(agent_id, agent_ensemble, pars);
        for hi in new_hes_inf.iter() {
            status_new[*hi] = Status::HesInf;
            let index = new_hs_list.iter().position(|&x| x == agent_id).unwrap();
            new_hs_list.remove(index);
        }
        for ai in new_act_inf.iter() {
            status_new[*ai] = Status::ActInf;
            let index = new_as_list.iter().position(|&x| x == agent_id).unwrap();
            new_as_list.remove(index);
        }
        
        let change = removal_step_agent(removal_rate);
        if change {
            status_new[agent_id] = Status::HesRem;
            new_hr_list.push(agent_id);
            let index = new_hi_list.iter().position(|&x| x == agent_id).unwrap();
            new_hi_list.remove(index);
        }
    }

    for a in ai_list.iter() {
        let agent_id = *a;
        // Perform infection step for focal agent
        let (new_hes_inf, new_act_inf) = infection_step_agent(agent_id, agent_ensemble, pars);
        for hi in new_hes_inf.iter() {
            status_new[*hi] = Status::HesInf;
            let index = new_hs_list.iter().position(|&x| x == agent_id).unwrap();
            new_hs_list.remove(index);
        }
        for ai in new_act_inf.iter() {
            status_new[*ai] = Status::ActInf;
            let index = new_as_list.iter().position(|&x| x == agent_id).unwrap();
            new_as_list.remove(index);
        }
        let change = removal_step_agent(removal_rate);
        if change {
            status_new[agent_id] = Status::ActRem;
            new_ar_list.push(agent_id);
            let index = new_ai_list.iter().position(|&x| x == agent_id).unwrap();
            new_ai_list.remove(index);
        }
    }
}

pub fn immunity_decay_step_agent(immunity_decay_rate: f64) -> bool {
    let mut rng = rand::thread_rng();
    let trial: f64 = rng.gen();
    trial < immunity_decay_rate
}

pub fn vaccine_decay_step_agent(vaccine_decay_rate: f64) -> bool {
    let mut rng = rand::thread_rng();
    let trial: f64 = rng.gen();
    trial < vaccine_decay_rate
}

pub fn symmetrical_immunity_and_vaccination_decay_step_subensemble(
    new_hs_list: &mut Vec<usize>,
    new_as_list: &mut Vec<usize>,
    new_hr_list: &mut Vec<usize>,
    new_ar_list: &mut Vec<usize>,
    new_av_list: &mut Vec<usize>,
    new_hv_list: &mut Vec<usize>,
    status_new: &mut Vec<Status>,
    pars: &Input,
) {
    let ar_list = new_ar_list.clone();
    let hr_list = new_hr_list.clone();
    let av_list = new_av_list.clone();
    let hv_list = new_hv_list.clone();
    let immunity_decay_rate = pars.epidemic.immunity_decay;
    let vaccine_decay_rate = pars.epidemic.vaccination_decay;

    for a in ar_list.iter() {
        let agent_id = *a;
        let change = immunity_decay_step_agent(immunity_decay_rate);
        if change {
            status_new[agent_id] = Status::ActSus;
            new_as_list.push(agent_id);
            let index = new_ar_list.iter().position(|&x| x == agent_id).unwrap();
            new_ar_list.remove(index);
        }
    }

    for a in hr_list.iter() {
        let agent_id = *a;
        let change = immunity_decay_step_agent(immunity_decay_rate);
        if change {
            status_new[agent_id] = Status::HesSus;
            new_hs_list.push(agent_id);
            let index = new_hr_list.iter().position(|&x| x == agent_id).unwrap();
            new_hr_list.remove(index);
        }
    }

    for a in av_list.iter() {
        let agent_id = *a;
        let change = vaccine_decay_step_agent(vaccine_decay_rate);
        if change {
            status_new[agent_id] = Status::ActSus;
            new_as_list.push(agent_id);
            let index = new_av_list.iter().position(|&x| x == agent_id).unwrap();
            new_av_list.remove(index);
        }
    }

    for a in hv_list.iter() {
        let agent_id = *a;
        let change = vaccine_decay_step_agent(vaccine_decay_rate);
        if change {
            status_new[agent_id] = Status::HesSus;
            new_hs_list.push(agent_id);
            let index = new_hv_list.iter().position(|&x| x == agent_id).unwrap();
            new_hv_list.remove(index);
        }
    }
}

pub fn dynamical_loop(agent_ensemble: &mut AgentEnsemble, pars: &Input) -> OutputResults {
    // Gather individuals by opinion-health status
    let mut hs_list = agent_ensemble.gather_hesitant_susceptible();
    let mut as_list = agent_ensemble.gather_active_susceptible();
    let mut hi_list = agent_ensemble.gather_hesitant_infected();
    let mut ai_list = agent_ensemble.gather_active_infected();
    let mut hr_list = agent_ensemble.gather_hesitant_removed();
    let mut ar_list = agent_ensemble.gather_active_removed();
    let mut av_list = agent_ensemble.gather_active_vaccinated();
    let hv_list = agent_ensemble.gather_hesitant_vaccinated();
    
    // Get initial susceptible density for the analytical comparison
    let sus_den_0 = (hs_list.len() + as_list.len()) as f64 / agent_ensemble.number_of_agents() as f64;
    
    // Initialize status-based population time series
    let mut pop_tseries = TimeOutput::new();

    // Initialize peak variables
    let mut time_to_peak = 0;
    let mut peak_incidence = 0;
    let mut vaccinated_at_peak = 0;
    let mut convinced_at_peak = 0;
    
    // Initialize loop conditions
    let t_max = pars.algorithm.t_max;
    let mut t = 0;
    let delta_t = 1;
    let mut total_infected = 1;
 
    // Run dynamical loop
    while (t < t_max) && total_infected > 0 {
        // Opinion step for all types of hesitant agents
        let mut new_hs_list = watts_threshold_step_subensemble(&hs_list, agent_ensemble, &mut as_list, Status::HesSus, t);
        let new_hi_list = watts_threshold_step_subensemble(&hi_list, agent_ensemble, &mut ai_list, Status::HesInf, t);
        let mut new_hr_list = watts_threshold_step_subensemble(&hr_list, agent_ensemble, &mut ar_list, Status::HesRem, t);
        
        // Vaccination step for all active susceptible agents
        let mut new_as_list = vaccination_step_subensemble(&as_list, agent_ensemble, &mut av_list, pars, t);
        
        // Infection & removal steps for all infected agents
        let (new_hi_list, ai_list_rep) = infection_and_removal_step_subensemble(
            &new_hi_list, 
            &ai_list, 
            agent_ensemble, 
            &mut new_hr_list, 
            Status::HesInf, 
            pars,
            t,
        );
        let (mut new_ai_list, new_hi_list) = infection_and_removal_step_subensemble(
            &ai_list, 
            &new_hi_list, 
            agent_ensemble, 
            &mut ar_list, 
            Status::ActInf, 
            pars,
            t,
        );
        
        // Update lists
        new_ai_list.extend(ai_list_rep);
        ai_list = remove_duplicates(new_ai_list);
        ai_list = agent_ensemble.update_list(&mut ai_list, Status::ActInf);
        hi_list = remove_duplicates(new_hi_list);
        hs_list = agent_ensemble.update_list(&mut new_hs_list, Status::HesSus);
        as_list = agent_ensemble.update_list(&mut new_as_list, Status::ActSus);
        hr_list = new_hr_list;
        // Update population time series
        let pop_t = TimeUnitPop::new(
            as_list.len(), 
            hs_list.len(), 
            ai_list.len(), 
            hi_list.len(), 
            ar_list.len(), 
            hr_list.len(), 
            av_list.len(),
            hv_list.len(),
        );
        pop_tseries.update_time_series(t, &pop_t);
        
        // Check exit condition
        let total_hi = hi_list.len();
        let total_ai = ai_list.len();
        if total_hi + total_ai >= total_infected {
            peak_incidence = total_hi + total_ai;
            time_to_peak = t;
            vaccinated_at_peak = av_list.len() + hv_list.len();
            convinced_at_peak = as_list.len() + ai_list.len() + ar_list.len() + av_list.len();
        }
        total_infected = total_hi + total_ai;

        // Update time step
        t += delta_t;  
        //println!("t={t}, inf={total_infected}");  
    }

    let end_time = t;

    // Complete time series
    let pop_t = TimeUnitPop::new(
        as_list.len(), 
        hs_list.len(), 
        ai_list.len(), 
        hi_list.len(), 
        ar_list.len(), 
        hr_list.len(), 
        av_list.len(),
        hv_list.len(),
    );
    //t += delta_t;
    while t < t_max {
        pop_tseries.update_time_series(t, &pop_t);
        t += delta_t;
    }

    // Measure global results
    let prevalence = ar_list.len() + hr_list.len();
    let vaccinated = av_list.len();
    let active = as_list.len() + ar_list.len() + av_list.len();

    println!("Time={end_time}, total infected={total_infected}");
    println!("Prevalence={prevalence}, vaccinated={vaccinated}, active={active}");
    let r_inf = sir_prevalence(pars.epidemic.r0, sus_den_0);
    let prev_ratio = prevalence as f64 / pars.network.n as f64;
    println!("Analytical homogeneous prevalence={r_inf} vs {prev_ratio}");

    // Collect global results
    let global_output = GlobalOutput { 
        convinced_at_peak,
        prevalence, 
        vaccinated, 
        vaccinated_at_peak,
        active, 
        peak_incidence, 
        time_to_peak, 
        time_to_end: end_time, 
    };
    // Prepare output struct
    let mut output = OutputResults {
        global: global_output,
        cluster: None,
        agent_ensemble: None,
        time: None,
    };
    // Collect cluster results
    if pars.output.cluster {
        let cluster_output = measure_clusters(&agent_ensemble);
        output.cluster = Some(cluster_output);
    }
    // Collect agent results
    if pars.output.agent {
        // Measure final agent/local properties if enabled
        if pars.output.agent {
            for agent_id in 0..agent_ensemble.number_of_agents() {
                measure_neighborhood(agent_id, agent_ensemble, t);
            }
        }
        let agent_ensemble_output = AgentEnsembleOutput::new(agent_ensemble);
        output.agent_ensemble = Some(agent_ensemble_output);
    }
    // Collect time series results
    if pars.output.time {
        output.time = Some(pop_tseries);
    }
    output
}

pub fn symmetric_dynamical_loop(agent_ensemble: &mut AgentEnsemble, pars: &Input) -> OutputResults {
    // Gather individuals by opinion-health status
    let mut hs_list = agent_ensemble.gather_hesitant_susceptible();
    let mut as_list = agent_ensemble.gather_active_susceptible();
    let mut hi_list = agent_ensemble.gather_hesitant_infected();
    let mut ai_list = agent_ensemble.gather_active_infected();
    let mut hr_list = agent_ensemble.gather_hesitant_removed();
    let mut ar_list = agent_ensemble.gather_active_removed();
    let mut av_list = agent_ensemble.gather_active_vaccinated();
    let mut hv_list = agent_ensemble.gather_hesitant_vaccinated();
    // Build status arrays old and new
    let mut status = agent_ensemble.build_status_array();
    // Get initial susceptible density for the analytical comparison with classical SIR
    let sus_den_0 = (hs_list.len() + as_list.len()) as f64 / agent_ensemble.number_of_agents() as f64;
    // Initialize status-based population time series
    let mut pop_tseries = TimeOutput::new();
    // Initialize loop conditions
    let t_max = pars.algorithm.t_max;
    let mut t = 0;
    let delta_t = 1;
    let mut total_infected = 1;
 
    // Run dynamical loop
    while (t < t_max) && total_infected > 0 {

        // Opinion & vaccination steps for all agents (if applies)
        symmetrical_opinion_and_vaccination_step_subensemble(
            &mut hs_list,
            &mut as_list,
            &mut hi_list,
            &mut ai_list,
            &mut hr_list,
            &mut ar_list,
            &mut av_list,
            &mut hv_list,
            &mut status,
            agent_ensemble,
            pars,
        );

        // Intermediate updating
        for (agent_id, &status) in status.iter().enumerate() {
            agent_ensemble.inner_mut()[agent_id].status = status;
        }

        // Infection & removal steps for all infected agents
        symmetrical_infection_and_removal_step_subensemble(
            &mut hs_list,
            &mut as_list,
            &mut hi_list,
            &mut ai_list,
            &mut hr_list,
            &mut ar_list,
            &mut status,
            agent_ensemble,
            pars,
        );

        // Natural immunity and vaccination decay for all involved agents
        symmetrical_immunity_and_vaccination_decay_step_subensemble( 
            &mut hs_list,
            &mut as_list,
            &mut hr_list,
            &mut ar_list,
            &mut av_list,
            &mut hv_list,
            &mut status,
            pars,
        );
        
        // End-loop list updating
        for (agent_id, &status) in status.iter().enumerate() {
            agent_ensemble.inner_mut()[agent_id].status = status;
        }

        // Update population time series
        let pop_t = TimeUnitPop::new(
            as_list.len(), 
            hs_list.len(), 
            ai_list.len(), 
            hi_list.len(), 
            ar_list.len(), 
            hr_list.len(), 
            av_list.len(),
            hv_list.len(),
        );
        pop_tseries.update_time_series(t, &pop_t);
        
        // Check exit condition
        let total_hi = hi_list.len();
        let total_ai = ai_list.len();
        total_infected = total_hi + total_ai;
        // Update time step
        t += delta_t;  
        //println!("t={t}, inf={total_infected}");  
    }

    let end_time = t;

    // Complete time series
    let pop_t = TimeUnitPop::new(
        as_list.len(), 
        hs_list.len(), 
        ai_list.len(), 
        hi_list.len(), 
        ar_list.len(), 
        hr_list.len(), 
        av_list.len(),
        hv_list.len(),
    );
    //t += delta_t;
    while t < t_max {
        pop_tseries.update_time_series(t, &pop_t);
        t += delta_t;
    }

    // Measure global results
    let prevalence = ar_list.len() + hr_list.len();
    let vaccinated = av_list.len();
    let active = as_list.len() + ar_list.len() + av_list.len();
    // TODO: PEAK INCIDENCE & TIMES

    println!("Time={end_time}, total infected={total_infected}");
    println!("Prevalence={prevalence}, vaccinated={vaccinated}, active={active}");
    let r_inf = sir_prevalence(pars.epidemic.r0, sus_den_0);
    let prev_ratio = prevalence as f64 / pars.network.n as f64;
    println!("Analytical homogeneous prevalence={r_inf} vs {prev_ratio}");

    // Collect global results
    let global_output = GlobalOutput {
        convinced_at_peak: 0, 
        prevalence, 
        vaccinated, 
        vaccinated_at_peak: 0,
        active, 
        peak_incidence: 0, 
        time_to_peak: 0, 
        time_to_end: end_time, 
    };
    // Prepare output struct
    let mut output = OutputResults {
        global: global_output,
        cluster: None,
        agent_ensemble: None,
        time: None,
    };
    // Collect agent results
    if pars.output.agent {
        // Measure final agent/local properties if enabled
        if pars.output.agent {
            for agent_id in 0..agent_ensemble.number_of_agents() {
                measure_neighborhood(agent_id, agent_ensemble, t);
            }
        }
        let agent_ensemble_output = AgentEnsembleOutput::new(agent_ensemble);
        output.agent_ensemble = Some(agent_ensemble_output);
    }
    // Collect time series results
    if pars.output.time {
        output.time = Some(pop_tseries);
    }
    output
}

pub fn watts_sir_coupled_model(pars: &Input, graph: &Network, output_ensemble: &mut OutputEnsemble) {
    // Loop over dynamical realizations
    for nsd in 0..pars.algorithm.nsims_dyn {
        println!("Dynamical realization={nsd}");

        // Instantiate the ensemble of agents
        let mut agent_ensemble = AgentEnsemble::new(pars.network.n);
        // Assign neighbors
        agent_ensemble.get_neighbors(graph);
        // Introduce infected individuals
        agent_ensemble.introduce_infections(pars.epidemic.seed_model, pars.epidemic.seeds);
        // Set thresholds to agents
        agent_ensemble.set_opinion_threshold(pars.opinion.threshold);
        // Introduce pro-active individuals
        agent_ensemble.introduce_opinions(
            pars.opinion.active_fraction, 
            pars.opinion.zealot_fraction,
        );

        // Measure agent's neighborhood at t=0 if enabled
        if pars.output.agent {
            for agent_id in 0..agent_ensemble.number_of_agents() {
                measure_neighborhood(agent_id, &mut agent_ensemble, 0);
            }
        }
        // Measure clusters if enabled
        //let mut cluster_output = ClusterOutput {
        //    ai_cluster: Vec::new(),
        //    ar_cluster: Vec::new(),
        //    as_cluster: Vec::new(),
        //    av_cluster: Vec::new(),
        //    hi_cluster: Vec::new(),
        //    hr_cluster: Vec::new(),
        //    hs_cluster: Vec::new(),
        //    hv_cluster: Vec::new(),
        //    ze_cluster: Vec::new(),
        //};
        //if pars.output.cluster {
        //    cluster_output = measure_clusters(&agent_ensemble);
        //}

        // State info before dynamical loop
        let info = true;
        if info {
            let active_fraction = agent_ensemble.total_active() as f64 / pars.network.n as f64;
            let hesitant_fraction = agent_ensemble.total_hesitant() as f64 / pars.network.n as f64;
            let zealot_fraction = agent_ensemble.total_zealot() as f64 / pars.network.n as f64;
            println!("Macrostate by opinion: a={active_fraction}, h={hesitant_fraction}, z={zealot_fraction}");
            let susceptible_fraction = agent_ensemble.total_susceptible() as f64 / pars.network.n as f64;
            let infected_fraction = agent_ensemble.total_infected() as f64 / pars.network.n as f64;
            let vaccinated_fraction = agent_ensemble.total_vaccinated() as f64 / pars.network.n as f64;
            println!("Macrostate by health: s={susceptible_fraction}, i={infected_fraction}, v={vaccinated_fraction}");
        }

        // Coupled dynamics
        let output = dynamical_loop(&mut agent_ensemble, pars);

        // Collect clusters into output if enabled
        //if pars.output.cluster {
        //    output.cluster = Some(cluster_output);
        //}

        // Add outbreak output to ensemble
        output_ensemble.add_outbreak(output, pars);
    }
}

pub fn datadriven_watts_sir_coupled_model(
    pars: &mut Input,
    graph: &Network, 
    output_ensemble1: &mut OutputEnsemble,
    output_ensemble2: &mut OutputEnsemble,
) {
    // Loop over dynamical realizations
    for nsd in 0..pars.algorithm.nsims_dyn {
        println!("Dynamical realization={nsd}");
        println!("Primary outbreak starts");

        // Instantiate the ensemble of agents
        let mut agent_ensemble = AgentEnsemble::new(pars.network.n);
        // Assign neighbors
        agent_ensemble.get_neighbors(graph);

        // Wave 1
        // Introduce infected & vaccinated individuals
        let vpars = pars.vaccination.unwrap();
        let nseeds = pars.epidemic.seeds;
        agent_ensemble.introduce_vaccination_thresholds(&vpars);
        agent_ensemble.introduce_infections_dd(nseeds);

        // Measure agent's neighborhood at t=0
        if pars.output.agent {
            for agent_id in 0..agent_ensemble.number_of_agents() {
                measure_neighborhood(agent_id, &mut agent_ensemble, 0);
            }
        }
        // Measure clusters if enabled
        let mut cluster_output = ClusterOutput {
            as_cluster: Vec::new(),
            hs_cluster: Vec::new(),
            ai_cluster: Vec::new(),
            hi_cluster: Vec::new(),
            ar_cluster: Vec::new(),
            hr_cluster: Vec::new(),
            av_cluster: Vec::new(),
            hv_cluster: Vec::new(),
            ze_cluster: Vec::new(),
        };
        if pars.output.cluster {
            cluster_output = measure_clusters(&agent_ensemble);
        }
        
        // Coupled dynamics
        let mut output1 = dynamical_loop(&mut agent_ensemble, pars);
        
        // Collect clusters into output if enabled
        if pars.output.cluster {
            output1.cluster = Some(cluster_output);
        }

        // Add outbreak output to ensemble
        output_ensemble1.add_outbreak(output1, pars);
    
        // Wave 2
        if pars.epidemic.secondary_outbreak {
            println!("Secondary outbreak starts");
            
            let r0 = pars.epidemic.r0_w2;
            let infection_decay = pars.epidemic.infection_decay;
            pars.epidemic.infection_rate = compute_beta_from_r0(r0, infection_decay, &pars.network, graph);
            
            // Reintroduce infections
            agent_ensemble.reintroduce_infections(nseeds);

            // Measure clusters if enabled
            let mut cluster_output = ClusterOutput {
                as_cluster: Vec::new(),
                hs_cluster: Vec::new(),
                ai_cluster: Vec::new(),
                hi_cluster: Vec::new(),
                ar_cluster: Vec::new(),
                hr_cluster: Vec::new(),
                av_cluster: Vec::new(),
                hv_cluster: Vec::new(),
                ze_cluster: Vec::new(),
            };
            if pars.output.cluster {
                cluster_output = measure_clusters(&agent_ensemble);
            }
            
            // Coupled dynamics
            let mut output2 = dynamical_loop(&mut agent_ensemble, pars);

            // Collect clusters into output if enabled
            if pars.output.cluster {
                output2.cluster = Some(cluster_output);
            }
            
            // Add outbreak output to ensemble
            output_ensemble2.add_outbreak(output2, pars);
        }
    }
}

/* 
pub fn symmetric_watts_sirs_coupled_model(pars: &Input, graph: &Network, output_ensemble: &mut OutputEnsemble) {
    // Loop over dynamical realizations
    for nsd in 0..pars.algorithm.nsims_dyn {
        println!("Dynamical realization={nsd}");
        
        // Instantiate the ensemble of agents
        let mut agent_ensemble = AgentEnsemble::new(pars.network.n);
        // Assign neighbors
        agent_ensemble.get_neighbors(graph);
        // Introduce infected individuals
        agent_ensemble.introduce_infections(pars.epidemic.seeds);
        // Set thresholds to agents
        agent_ensemble.set_opinion_threshold(pars.opinion.threshold);
        // Introduce pro-active individuals
        agent_ensemble.introduce_opinions(
            pars.opinion.active_fraction, 
            pars.opinion.zealot_fraction,
        );

        // Measure initial agent/local properties if enabled
        if pars.output.agent {
            for agent_id in 0..agent_ensemble.number_of_agents() {
                measure_neighborhood(agent_id, &mut agent_ensemble, 0);
            }
        }

        // Coupled dynamics
        let output = symmetric_dynamical_loop(&mut agent_ensemble, pars);

        // Add outbreak output to ensemble
        output_ensemble.add_outbreak(output, pars);
    }
}
*/