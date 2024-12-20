use crate::{
    agent::{AgentEnsemble, Status},
    cons::{FLAG_VERBOSE, PAR_EPIDEMIC_DIEOUT, PAR_TIME_STEP},
    utils::{
        measure_attitude_clusters, measure_cascading_clusters, measure_neighborhood,
        measure_opinion_health_clusters, remove_duplicates, sir_prevalence, AgentEnsembleOutput,
        GlobalOutput, InputMultilayer, OutputFlags, OutputResults, TimeOutput, TimeUnitPop,
    },
};
use rand::prelude::*;

pub fn dynamical_loop(
    agent_ensemble: &mut AgentEnsemble,
    pars: &InputMultilayer,
    flags_output: &OutputFlags,
) -> OutputResults {
    if flags_output.agent {
        for agent_id in 0..agent_ensemble.number_of_agents() {
            measure_neighborhood(agent_id, agent_ensemble, 0);
        }
    }

    let nagents = agent_ensemble.number_of_agents();

    let mut hs_list = agent_ensemble.gather_hesitant_susceptible();
    let mut as_list = agent_ensemble.gather_active_susceptible();
    let mut hi_list = agent_ensemble.gather_hesitant_infected();
    let mut ai_list = agent_ensemble.gather_active_infected();
    let mut hr_list = agent_ensemble.gather_hesitant_removed();
    let mut ar_list = agent_ensemble.gather_active_removed();
    let mut av_list = agent_ensemble.gather_active_vaccinated();
    let hv_list = agent_ensemble.gather_hesitant_vaccinated();

    let sus_den_0 = (hs_list.len() + as_list.len()) as f64 / nagents as f64;

    let mut pop_tseries = TimeOutput::new();

    let mut time_to_peak = 0;
    let mut peak_incidence = 0;
    let mut vaccinated_at_peak = 0;
    let mut convinced_at_peak = 0;

    let t_max = pars.t_max;
    let mut t = 0;
    let delta_t = PAR_TIME_STEP;
    let mut total_infected = 1;

    while (t < t_max) && total_infected > PAR_EPIDEMIC_DIEOUT {
        let mut new_hs_list =
            watts_threshold_step_subensemble(&hs_list, agent_ensemble, &mut as_list, t);
        let new_hi_list =
            watts_threshold_step_subensemble(&hi_list, agent_ensemble, &mut ai_list, t);
        let mut new_hr_list =
            watts_threshold_step_subensemble(&hr_list, agent_ensemble, &mut ar_list, t);

        let mut new_as_list = vaccination_step_subensemble(
            &as_list,
            agent_ensemble,
            &mut av_list,
            pars.rate_vaccination,
            t,
        );

        let (new_hi_list, ai_list_rep) = infection_and_removal_step_subensemble(
            &new_hi_list,
            &ai_list,
            agent_ensemble,
            &mut new_hr_list,
            Status::HesInf,
            pars.rate_infection,
            pars.rate_removal,
            t,
        );

        let (mut new_ai_list, new_hi_list) = infection_and_removal_step_subensemble(
            &ai_list,
            &new_hi_list,
            agent_ensemble,
            &mut ar_list,
            Status::ActInf,
            pars.rate_infection,
            pars.rate_removal,
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

    while t < t_max {
        pop_tseries.update_time_series(t, &pop_t);
        t += delta_t;
    }

    let prevalence = ar_list.len() + hr_list.len();
    let vaccinated = av_list.len();
    let active = as_list.len() + ar_list.len() + av_list.len();

    println!("Time={end_time}, total infected={total_infected}");
    println!("Prevalence={prevalence}, vaccinated={vaccinated}, active={active}");
    let r_inf = sir_prevalence(pars.r0, sus_den_0);
    let prev_ratio = prevalence as f64 / nagents as f64;
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
        time: None,
    };

    if flags_output.cluster {
        let attitude_cluster = measure_attitude_clusters(agent_ensemble);
        agent_ensemble.cascading_threshold();
        let cascading_cluster = measure_cascading_clusters(agent_ensemble);
        let opinion_health_cluster = measure_opinion_health_clusters(agent_ensemble);
        output.cluster.as_mut().unwrap().attitude = Some(attitude_cluster);
        output.cluster.as_mut().unwrap().cascading = Some(cascading_cluster);
        output.cluster.as_mut().unwrap().opinion_health = Some(opinion_health_cluster);
    }

    if flags_output.agent {
        for agent_id in 0..agent_ensemble.number_of_agents() {
            measure_neighborhood(agent_id, agent_ensemble, t);
        }
        let agent_ensemble_output = AgentEnsembleOutput::new(agent_ensemble);
        output.agent_ensemble = Some(agent_ensemble_output);
    }

    if flags_output.time {
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
    let removal_rate = infection_decay;
    let mut new_focal_list = Vec::new();
    let mut new_collateral_list = collateral_list.to_owned();
    for a in focal_list.iter() {
        let agent_id = *a;

        let (new_hes_inf, new_act_inf) =
            infection_step_agent(agent_id, agent_ensemble, infection_rate);

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

        if focal_status == Status::HesInf {
            new_focal_list.extend(new_hes_inf);
            new_collateral_list.extend(new_act_inf);
        } else {
            new_focal_list.extend(new_act_inf);
            new_collateral_list.extend(new_hes_inf);
        }

        let change = removal_step_agent(removal_rate);
        if change {
            if focal_status == Status::HesInf {
                agent_ensemble.inner_mut()[agent_id].status = Status::HesRem;
                agent_ensemble.inner_mut()[agent_id].removed_when = Some(t);
            } else if focal_status == Status::ActInf {
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
            }
            false => {
                new_as_list.push(agent_id);
            }
        };
    }
    new_as_list
}

fn watts_threshold_step_agent(agent_id: usize, agent_ensemble: &AgentEnsemble) -> bool {
    let status = agent_ensemble.inner()[agent_id].status;
    let neighbors = agent_ensemble.inner()[agent_id].neighbors.clone();
    let degree_opinion = agent_ensemble.inner()[agent_id].degree_opinion;

    let mut vaccinated_neighbors = 0.0;
    for neigh in neighbors {
        if agent_ensemble.inner()[neigh].status == Status::ActVac {
            vaccinated_neighbors += 1.0;
        }
    }

    let vaccinated_fraction = if degree_opinion == 0 {
        0.0
    } else {
        vaccinated_neighbors / degree_opinion as f64
    };

    if vaccinated_fraction >= agent_ensemble.inner()[agent_id].threshold {
        matches!(status, Status::HesSus | Status::HesInf | Status::HesRem)
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
                }
                Status::HesInf => {
                    agent_ensemble.inner_mut()[agent_id].status = Status::ActInf;
                    agent_ensemble.inner_mut()[agent_id].convinced_when = Some(t);
                    branching_list.push(agent_id);
                }
                Status::HesRem => {
                    agent_ensemble.inner_mut()[agent_id].status = Status::ActRem;
                    agent_ensemble.inner_mut()[agent_id].convinced_when = Some(t);
                    branching_list.push(agent_id);
                }
                _ => {}
            }
        } else {
            match status {
                Status::HesSus => {
                    new_list.push(agent_id);
                }
                Status::HesInf => {
                    new_list.push(agent_id);
                }
                Status::HesRem => {
                    new_list.push(agent_id);
                }
                _ => {}
            }
        }
    }
    new_list
}
