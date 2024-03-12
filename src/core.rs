use netrust::network::Network;
use rand::prelude::*;

use crate::{
    agent::{AgentEnsemble, Attitude, Status}, cons::{FLAG_VERBOSE, PAR_EPIDEMIC_DIEOUT, PAR_TIME_STEP}, utils::{
        compute_beta_from_r0, remove_duplicates, sir_prevalence, AgentEnsembleOutput, ClusterAttitudeOutput, ClusterCascadingOutput, ClusterOpinionHealthOutput, ClusterOutput, GlobalOutput, Input, OutputEnsemble, OutputResults, TimeOutput, TimeUnitPop
    } 
};

fn dynamical_loop(
    agent_ensemble: &mut AgentEnsemble, 
    pars: &Input,
) -> OutputResults {
        let mut hs_list = agent_ensemble.gather_hesitant_susceptible();
        let mut as_list = agent_ensemble.gather_active_susceptible();
        let mut hi_list = agent_ensemble.gather_hesitant_infected();
        let mut ai_list = agent_ensemble.gather_active_infected();
        let mut hr_list = agent_ensemble.gather_hesitant_removed();
        let mut ar_list = agent_ensemble.gather_active_removed();
        let mut av_list = agent_ensemble.gather_active_vaccinated();
        let hv_list = agent_ensemble.gather_hesitant_vaccinated();
        
        let sus_den_0 = (hs_list.len() + as_list.len()) as f64 / agent_ensemble.number_of_agents() as f64;

        let mut pop_tseries = TimeOutput::new();

        let mut time_to_peak = 0;
        let mut peak_incidence = 0;
        let mut vaccinated_at_peak = 0;
        let mut convinced_at_peak = 0;

        let t_max = pars.algorithm.unwrap().t_max;
        let mut t = 0;
        let delta_t = PAR_TIME_STEP;
        let mut total_infected = 1;

        while (t < t_max) && total_infected > PAR_EPIDEMIC_DIEOUT {
            let mut new_hs_list = watts_threshold_step_subensemble(&hs_list, agent_ensemble, &mut as_list, t);
            let new_hi_list = watts_threshold_step_subensemble(&hi_list, agent_ensemble, &mut ai_list, t);
            let mut new_hr_list = watts_threshold_step_subensemble(&hr_list, agent_ensemble, &mut ar_list, t);

            // Vaccination step for all active susceptible agents //TODO: INTRODUCE VACCINATION MODEL SELECTION
            let mut new_as_list = vaccination_step_subensemble(&as_list, agent_ensemble, &mut av_list, pars.epidemic.vaccination_rate, t);

            let (new_hi_list, ai_list_rep) = infection_and_removal_step_subensemble(
                &new_hi_list, 
                &ai_list, 
                agent_ensemble, 
                &mut new_hr_list, 
                Status::HesInf, 
                pars.epidemic.infection_rate,
                pars.epidemic.infection_decay,
                t,
            );
            let (mut new_ai_list, new_hi_list) = infection_and_removal_step_subensemble(
                &ai_list, 
                &new_hi_list, 
                agent_ensemble, 
                &mut ar_list, 
                Status::ActInf, 
                pars.epidemic.infection_rate,
                pars.epidemic.infection_decay,
                t,
            );

            new_ai_list.extend(ai_list_rep);
            ai_list = remove_duplicates(new_ai_list);
            ai_list = agent_ensemble.update_list(&mut ai_list, Status::ActInf);
            hi_list = remove_duplicates(new_hi_list);
            hs_list = agent_ensemble.update_list(&mut new_hs_list, Status::HesSus);
            as_list = agent_ensemble.update_list(&mut new_as_list, Status::ActSus);
            hr_list = new_hr_list;

            let pop_t = TimeUnitPop::new(
                ai_list.len(), 
                ar_list.len(), 
                as_list.len(),
                av_list.len(), 
                hi_list.len(), 
                hr_list.len(),
                hs_list.len(), 
                hv_list.len(),
            );
            pop_tseries.update_time_series(t, &pop_t);
    
            let total_hi = hi_list.len();
            let total_ai = ai_list.len();
            if total_hi + total_ai >= total_infected {
                peak_incidence = total_hi + total_ai;
                time_to_peak = t;
                vaccinated_at_peak = av_list.len() + hv_list.len();
                convinced_at_peak = as_list.len() + ai_list.len() + ar_list.len() + av_list.len();
            }
            total_infected = total_hi + total_ai;
    
            t += delta_t;  
            if FLAG_VERBOSE {
                println!("t={t}, inf={total_infected}");
            }
        }
    
        let end_time = t;
    
        let pop_t = TimeUnitPop::new(
            ai_list.len(), 
            ar_list.len(),
            as_list.len(),
            av_list.len(),
            hi_list.len(),
            hr_list.len(),
            hs_list.len(),
            hv_list.len(),
        );
        //t += delta_t;
        while t < t_max {
            pop_tseries.update_time_series(t, &pop_t);
            t += delta_t;
        }
    
        let prevalence = ar_list.len() + hr_list.len();
        let vaccinated = av_list.len();
        let active = as_list.len() + ar_list.len() + av_list.len();
    
        println!("Time={end_time}, total infected={total_infected}");
        println!("Prevalence={prevalence}, vaccinated={vaccinated}, active={active}");
        let r_inf = sir_prevalence(pars.epidemic.r0, sus_den_0);
        let prev_ratio = prevalence as f64 / pars.network.unwrap().size as f64;
        println!("Analytical homogeneous prevalence={r_inf} vs {prev_ratio}");
    
        let global_output = GlobalOutput {
            active,
            convinced_at_peak,
            peak_incidence, 
            prevalence,
            time_to_end: end_time,
            time_to_peak,
            vaccinated, 
            vaccinated_at_peak, 
        };

        let mut output = OutputResults {
            agent_ensemble: None,
            cluster: None,
            global: global_output,
            rebuild: None,
            time: None,
        };
        
        if pars.output.unwrap().cluster {
            let cluster_output = measure_opinion_health_clusters(&agent_ensemble);
            output.cluster.as_mut().unwrap().opinion_health = Some(cluster_output);
        }

        if pars.output.unwrap().agent {
            if pars.output.unwrap().agent {
                for agent_id in 0..agent_ensemble.number_of_agents() {
                    measure_neighborhood(agent_id, agent_ensemble, t);
                }
            }
            let agent_ensemble_output = AgentEnsembleOutput::new(agent_ensemble);
            output.agent_ensemble = Some(agent_ensemble_output);
        }

        if pars.output.unwrap().time {
            output.time = Some(pop_tseries);
        }
        output
}

fn infection_and_removal_step_subensemble(
        focal_list: &[usize], 
        collateral_list: &[usize],
        agent_ensemble: &mut AgentEnsemble,
        branching_list: &mut Vec<usize>,
        focal_status: Status,
        infection_rate: f64,
        infection_decay: f64,
        t: usize,
) -> (Vec<usize>, Vec<usize>) {
        // Infection & removal steps for all types of infected agents //hi_list: focal, ai_list: collateral, hr_list: branching
        let removal_rate = infection_decay;
        let mut new_focal_list = Vec::new(); //focal_list.clone();
        let mut new_collateral_list = collateral_list.to_owned();
        for a in focal_list.iter() {
            let agent_id = *a;
            // Perform infection step for focal agent
            let (new_hes_inf, new_act_inf) = infection_step_agent(agent_id, agent_ensemble, infection_rate);
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

fn infection_step_agent(
        agent_id: usize, 
        agent_ensemble: &AgentEnsemble, 
        infection_rate: f64,
) -> (Vec<usize>, Vec<usize>) {
        let mut hes_inf_a = Vec::new();
        let mut act_inf_a = Vec::new();
        let mut rng = rand::thread_rng();
    
        let neighbors = agent_ensemble.inner()[agent_id].neighbors.clone();
        for neighbor in neighbors {
            let neighbor_id = neighbor;
            let neighbor_status = agent_ensemble.inner()[neighbor_id].status;
            if neighbor_status == Status::HesSus {
                let trial: f64 = rng.gen();
                if trial < infection_rate {
                    hes_inf_a.push(neighbor_id);
                }
            } else if neighbor_status == Status::ActSus {
                let trial: f64 = rng.gen();
                if trial < infection_rate {
                    act_inf_a.push(neighbor_id);
                }
            }
        }
        (hes_inf_a, act_inf_a)
    }

fn measure_attitude_clusters(agent_ensemble: &AgentEnsemble) -> ClusterAttitudeOutput {
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

fn measure_cascading_clusters(agent_ensemble: &mut AgentEnsemble) -> ClusterCascadingOutput {
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
                            let v_cascading_threshold = agent_ensemble.inner()[v].cascading_threshold.unwrap();
                            if v_cascading_threshold == a_cascading_threshold && a_cascading_threshold == 0 {
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

    ClusterCascadingOutput::new(
        cascading_clusters, 
        nonzealot_clusters, 
    )
}

fn measure_opinion_health_clusters(agent_ensemble: &AgentEnsemble) -> ClusterOpinionHealthOutput {
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

fn measure_neighborhood(
    agent_id: usize, 
    agent_ensemble: &mut AgentEnsemble, 
    t: usize,
) {
    let neighbors = agent_ensemble.inner_mut()[agent_id].neighbors.clone();
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

fn removal_step_agent(removal_rate: f64) -> bool {
    let mut rng = rand::thread_rng();
    let trial: f64 = rng.gen();
    trial < removal_rate
}

fn vaccination_step_agent(vaccination_rate: f64) -> bool {
    let mut rng = rand::thread_rng();
    let trial: f64 = rng.gen();
    trial < vaccination_rate
}

fn vaccination_step_subensemble(
    as_list: &[usize], 
    agent_ensemble: &mut AgentEnsemble,
    av_list: &mut Vec<usize>,
    vaccination_rate: f64,
    t: usize,
) -> Vec<usize> {
    let mut new_as_list = Vec::new();
    for a in as_list.iter() {
        let agent_id = *a;
        let vaccination_target = agent_ensemble.inner()[agent_id].vaccination_target;
        match vaccination_target {
            true => {
                if vaccination_step_agent(vaccination_rate) {
                    agent_ensemble.inner_mut()[agent_id].status = Status::ActVac;
                    agent_ensemble.inner_mut()[agent_id].vaccinated_when = Some(t);
                    av_list.push(agent_id);
                } else {
                    new_as_list.push(agent_id);
                }
            },
            false => {
                new_as_list.push(agent_id);
            },
        };
    }
    new_as_list
}

pub fn watts_sir_coupled_model(
    pars: &Input, 
    graph: &Network, 
    output_ensemble: &mut OutputEnsemble,
) {
    for nsd in 0..pars.algorithm.unwrap().nsims_dyn {
        println!("Dynamical realization={nsd}");

        // Instantiate the ensemble of agents
        let mut agent_ensemble = AgentEnsemble::new(pars.network.unwrap().size);
        // Assign neighbors
        agent_ensemble.neighbors(graph);
        // Introduce infected individuals
        agent_ensemble.introduce_infections(pars.epidemic.seed_model, pars.epidemic.nseeds);
        // Set thresholds to agents
        agent_ensemble.set_opinion_threshold(pars.opinion.unwrap().threshold);
        // Introduce pro-active individuals
        agent_ensemble.introduce_opinions(
            pars.opinion.unwrap().active_fraction, 
            pars.opinion.unwrap().zealot_fraction,
        );

        // Measure agent's neighborhood at t=0 if enabled
        if pars.output.unwrap().agent {
            for agent_id in 0..agent_ensemble.number_of_agents() {
                measure_neighborhood(agent_id, &mut agent_ensemble, 0);
            }
        }

        // Coupled dynamics 
        let output = dynamical_loop(&mut agent_ensemble, pars);

        // Collect clusters into output if enabled
        //if pars.output.cluster {
        //    output.cluster = Some(cluster_output);
        //}

        // Add outbreak output to ensemble
        output_ensemble.add_outbreak(output, pars.network.unwrap().size, pars.epidemic.r0);
    }
}

pub fn watts_sir_coupled_model_datadriven_thresholds(
    pars: &mut Input,
    graph: &Network, 
    output_ensemble1: &mut OutputEnsemble,
    output_ensemble2: &mut OutputEnsemble,
) {
    // Loop over dynamical realizations
    for nsd in 0..pars.algorithm.unwrap().nsims_dyn {
        println!("Dynamical realization={nsd}");
        println!("Primary outbreak starts");

        // Instantiate the ensemble of agents
        let mut agent_ensemble = AgentEnsemble::new(pars.network.unwrap().size);
        // Assign neighbors
        agent_ensemble.neighbors(graph);

        // Wave 1
        // Introduce infected & vaccinated individuals
        let vpars = pars.vaccination.unwrap();
        agent_ensemble.introduce_vaccination_thresholds(&vpars);
        agent_ensemble.introduce_infections_dd(pars.epidemic.seed_model, pars.epidemic.nseeds);
    
        // Measure agent's neighborhood at t=0
        if pars.output.unwrap().agent {
            for agent_id in 0..agent_ensemble.number_of_agents() {
                measure_neighborhood(agent_id, &mut agent_ensemble, 0);
            }
        }
        // Measure clusters if enabled
        let mut cluster_output = ClusterOpinionHealthOutput {
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
        if pars.output.unwrap().cluster {
            cluster_output = measure_opinion_health_clusters(&agent_ensemble);
        }

        // Coupled dynamics
        let mut output1 = dynamical_loop(&mut agent_ensemble, pars);

        // Collect clusters into output if enabled
        if pars.output.unwrap().cluster {
            output1.cluster.as_mut().unwrap().opinion_health = Some(cluster_output);
        }

        // Add outbreak output to ensemble
        output_ensemble1.add_outbreak(output1, pars.network.unwrap().size, pars.epidemic.r0);

        // Wave 2
        if pars.epidemic.secondary_outbreak {
            println!("Secondary outbreak starts");
            
            let r0 = pars.epidemic.r0_w2;
            let infection_decay = pars.epidemic.infection_decay;
            pars.epidemic.infection_rate = compute_beta_from_r0(r0, infection_decay, &(pars.network.unwrap()), graph);

            // Reintroduce infections
            agent_ensemble.introduce_infections_dd(pars.epidemic.seed_model, pars.epidemic.nseeds);

            // Measure clusters if enabled
            let mut cluster_output = ClusterOpinionHealthOutput {
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
            if pars.output.unwrap().cluster {
                cluster_output = measure_opinion_health_clusters(&agent_ensemble);
            }

            // Coupled dynamics
            let mut output2 = dynamical_loop(&mut agent_ensemble, pars);

            // Collect clusters into output if enabled
            if pars.output.unwrap().cluster {
                output2.cluster.as_mut().unwrap().opinion_health = Some(cluster_output);
            }

            // Add outbreak output to ensemble
            output_ensemble2.add_outbreak(output2, pars.network.unwrap().size, pars.epidemic.r0);
        }
    }
}

pub fn watts_sir_coupled_model_multilayer(
    pars: &mut Input,
    agent_ensemble: &mut AgentEnsemble, 
    output_ensemble: &mut OutputEnsemble,
) {
    for nsd in 0..pars.algorithm.unwrap().nsims_dyn {
        println!("Dynamical realization={nsd}");

        agent_ensemble.introduce_vaccination_attitudes(&pars.vaccination.unwrap());

        agent_ensemble.introduce_infections_dd(pars.epidemic.seed_model, pars.epidemic.nseeds);

        if pars.output.unwrap().agent {
            for agent_id in 0..agent_ensemble.number_of_agents() {
                measure_neighborhood(agent_id, agent_ensemble, 0);
            }
        }

        let cluster_output = if pars.output.unwrap().cluster {
            let attitude_cluster = measure_attitude_clusters(agent_ensemble);
            agent_ensemble.cascading_threshold();
            let cascading_cluster = measure_cascading_clusters(agent_ensemble);
            Some(ClusterOutput::new(Some(attitude_cluster), Some(cascading_cluster), None))
        } else {
            None
        };
    
        let mut output = dynamical_loop(agent_ensemble, pars);

        if pars.output.unwrap().cluster {
            output.cluster = cluster_output;
        }

        output_ensemble.add_outbreak(output, pars.network.unwrap().size, pars.epidemic.r0);

        agent_ensemble.clear_epidemic_consequences();
    }
}

fn watts_threshold_step_agent(
    agent_id: usize, 
    agent_ensemble: &AgentEnsemble,
) -> bool {
    // Get focal agent status, neighbors and degree
    let status = agent_ensemble.inner()[agent_id].status;
    let neighbors = agent_ensemble.inner()[agent_id].neighbors.clone();
    let k = neighbors.len();
    // Get focal agent vaccinated neighbor fraction
    let mut vaccinated_neighbors = 0.0;
    for neigh in neighbors {
        if agent_ensemble.inner()[neigh].status == Status::ActVac {
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

fn watts_threshold_step_subensemble(
    original_list: &[usize], 
    agent_ensemble: &mut AgentEnsemble,
    branching_list: &mut Vec<usize>,
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