use netrust::{utils::{NetworkPars, NetworkModel}, network::Network};

use crate::utils::{ClusterAssembledVectors, calculate_cluster_sim_stats, ClusterStatPacker, ClusterDistribution, ClusterDistributionPacker, AssembledAgentOutput, AgentStatPacker, compute_stats, convert_to_f64, AgentDistribution, AgentDistributionPacker};

pub fn compute_beta_from_r0(r0: f64, removal_rate: f64, npars: &NetworkPars, graph: &Network) -> f64 {
    match npars.model {
        NetworkModel::Complete => {
            r0 * removal_rate / (npars.n - 1) as f64
        },
        NetworkModel::Regular => {
            let k_avg = graph.average_degree();
            r0 * removal_rate / k_avg
        },
        NetworkModel::ErdosRenyi => {
            let k_avg = graph.average_degree();
            r0 * removal_rate / k_avg
        },
        NetworkModel::ScaleFree => {
            let k_avg = graph.average_degree();
            //let k2_avg = graph.second_moment();
            r0 * removal_rate / k_avg // (k2_avg - k_avg))
        }
        NetworkModel::BarabasiAlbert => {
            let k_avg = graph.average_degree();
            //let k2_avg = graph.second_moment();
            r0 * removal_rate / k_avg // (k2_avg - k_avg))
        }
        NetworkModel::WattsStrogatz => {
            let k_avg = graph.average_degree();
            r0 * removal_rate / k_avg
        }
    }
}

pub fn sir_prevalence(r0: f64, sus0: f64) -> f64 {
    let mut r_inf = 0.0;
    let mut guess = 0.8;
    let mut escape = 0;
    let mut condition = true;
    while condition {
        r_inf = 1.0 - sus0 * (-r0 * guess).exp();
        if r_inf == guess {
            condition = false;
        }
        guess = r_inf;
        escape += 1;
        if escape > 10000 {
            r_inf = 0.0;
            condition = false;
        }
    }
    r_inf
}

pub fn compute_cluster_stats(assembled_cluster_output: &ClusterAssembledVectors) -> ClusterStatPacker {
    let ai_cluster_s = &assembled_cluster_output.ai_cluster;
    let ar_cluster_s = &assembled_cluster_output.ar_cluster;
    let as_cluster_s = &assembled_cluster_output.as_cluster;
    let av_cluster_s = &assembled_cluster_output.av_cluster;
    let hi_cluster_s = &assembled_cluster_output.hi_cluster;
    let hr_cluster_s = &assembled_cluster_output.hr_cluster;
    let hs_cluster_s = &assembled_cluster_output.hs_cluster;
    let hv_cluster_s = &assembled_cluster_output.hv_cluster;

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

pub fn compute_cluster_distribution(assembled_cluster_output: &ClusterAssembledVectors) -> ClusterDistributionPacker {
    let ai_cluster_s = &assembled_cluster_output.ai_cluster;
    let ar_cluster_s = &assembled_cluster_output.ar_cluster;
    let as_cluster_s = &assembled_cluster_output.as_cluster;
    let av_cluster_s = &assembled_cluster_output.av_cluster;
    let hi_cluster_s = &assembled_cluster_output.hi_cluster;
    let hr_cluster_s = &assembled_cluster_output.hr_cluster;
    let hs_cluster_s = &assembled_cluster_output.hs_cluster;
    let hv_cluster_s = &assembled_cluster_output.hv_cluster;

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

pub fn compute_cluster_sim_distribution(simulations: &[Vec<usize>]) -> ClusterDistribution {
    let mut distribution = ClusterDistribution::new();

    for sim_vector in simulations {
        for &cluster_size in sim_vector {
            distribution.add_cluster(cluster_size);
        }
    }

    distribution
}

pub fn compute_agent_stats(assembled_agent_output: &AssembledAgentOutput) -> AgentStatPacker {
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

pub fn compute_fractions(
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



pub fn compute_agent_sim_distribution(simulations: &Vec<Vec<f64>>) -> AgentDistribution {
    const NUM_BINS: usize = 30; // Adjust as needed

    let mut all_values = Vec::new();
    for sim_vector in simulations {
        all_values.extend_from_slice(sim_vector);
    }

    let min_value = *all_values.iter().min_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)).unwrap() as usize;
    let max_value = *all_values.iter().max_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal)).unwrap() as usize;

    let bin_size = (max_value - min_value) / NUM_BINS as usize;
    let bin_edges: Vec<usize> = (0..=NUM_BINS).map(|i| min_value + i * bin_size).collect();

    let mut distribution = AgentDistribution::new(bin_edges.clone());

    for sim_vector in simulations {
        for &value in sim_vector {
            distribution.add_value(value);
        }
    }

    distribution
}

pub fn compute_agent_distribution(assembled_agent_output: &AssembledAgentOutput) -> AgentDistributionPacker {
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
