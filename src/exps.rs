use clap::Parser;
use serde_pickle::ser::SerOptions;

use crate::agent::{AgentEnsemble, HesitancyAttributionModel, SeedModel, VaccinationPolicy};
use crate::cons::{FILENAME_CONFIG, FILENAME_DATA_AVERAGE_CONTACT, FILENAME_DATA_CONTACT_MATRIX, FILENAME_DATA_POPULATION_AGE, FILENAME_DATA_VACCINATION_ATTITUDE, FOLDER_CONFIG, FOLDER_RESULTS, HEADER_AGENT, HEADER_CLUSTER, HEADER_GLOBAL, HEADER_TIME, PAR_NETWORK_TRIALS};
use crate::utils::{
    build_normalized_cdf, compute_beta_from_r0, compute_cluster_distribution, compute_cluster_stats, compute_interlayer_probability_matrix, compute_intralayer_average_degree, count_underaged, load_json_config, read_key_and_f64_from_json, read_key_and_matrixf64_from_json, read_key_and_vecf64_from_json, write_file_name, AlgorithmPars, EpidemicPars, Input, OpinionPars, OutputEnsemble, OutputPars, SerializeClusterAssemblyTwoWaves, SerializeGlobalAssemblyTwoWaves, SerializedAgentAssemblyTwoWaves, SerializedTimeSeriesAssemblyTwoWaves, USState, VaccinationPars
};
use crate::core::{
    watts_sir_coupled_model, 
    watts_sir_coupled_model_datadriven_thresholds, 
    watts_sir_coupled_model_multilayer, 
};

use netrust::network::Network;
use netrust::utils::{
    NetworkModel, 
    NetworkPars,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub active_fraction: f64,
    #[clap(long, value_parser, default_value_t = true)]
    pub age_flag: bool,
    #[clap(long, value_parser, default_value_t = 18)]
    pub age_threshold: usize,
    #[clap(long, value_parser, default_value_t = true)]
    pub agent_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub agent_raw_flag: bool,
    #[clap(long, value_parser, default_value_t = 10)]
    pub average_degree: usize,
    #[clap(long, value_parser, default_value_t = false)]
    pub cluster_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub cluster_raw_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub config_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub degree_flag: bool,
    #[clap(long, value_parser, default_value_t = 3)]
    pub experiment_id: usize,
    #[clap(long, value_parser, default_value_t = true)]
    pub global_flag: bool,
    #[clap(long, value_parser, default_value = "random")]
    pub hesitancy_attribution_model: HesitancyAttributionModel,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub immunity_decay: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub infection_rate: f64,
    #[clap(long, value_parser, default_value_t = 0.2)]
    pub infection_decay: f64,
    #[clap(long, value_parser, default_value_t = 0)]
    pub maximum_degree: usize,
    #[clap(long, value_parser, default_value_t = 0)]
    pub minimum_degree: usize,
    #[clap(long, value_parser, default_value = "erdos-renyi")]
    pub network_model: NetworkModel,
    #[clap(long, value_parser, default_value_t = 100000)]
    pub network_size: usize,
    #[clap(long, value_parser, default_value_t = 10)]
    pub nsims_dyn: usize,
    #[clap(long, value_parser, default_value_t = 30)]
    pub nsims_net: usize,
    #[clap(long, value_parser, default_value_t = 20)]
    pub nseeds: usize,
    #[clap(long, value_parser, default_value_t = 1.0)]
    pub powerlaw_exponent: f64,
    #[clap(long, value_parser, default_value_t = true)]
    pub rebuild_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub rebuild_raw_flag: bool,
    #[clap(long, value_parser, default_value_t = 1.0)]
    pub rewiring_probability: f64,
    #[clap(long, value_parser, default_value_t = 1.5)]
    pub r0: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub r0_w2: f64,
    #[clap(long, value_parser, default_value_t = false)]
    pub secondary_outbreak: bool,
    #[clap(long, value_parser, default_value = "top-degree-neighborhood")]
    pub seed_model: SeedModel,
    #[clap(long, value_parser, default_value_t = 500)]
    pub t_max: usize,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub threshold: f64,
    #[clap(long, value_parser, default_value_t = false)]
    pub time_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub time_raw_flag: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub underage_correction_flag: bool,
    #[clap(long, value_parser, default_value = "massachusetts")]
    pub usa_id: USState,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub vaccination_decay: f64,
    #[clap(long, value_parser, default_value = "top-degree-neighborhood")]
    pub vaccination_policy: VaccinationPolicy,
    #[clap(long, value_parser, default_value_t = 1.0)]
    pub vaccination_quota: f64,
    #[clap(long, value_parser, default_value_t = 0.005)]
    pub vaccination_rate: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub zealot_fraction: f64,
}

// EXPERIMENT 1: WATTS-SIR DYNAMICS UNDER HOMOGENEOUS THRESHOLD
pub fn run_exp1_homogeneous(args: Args) {

    let mut pars = match args.config_flag {
        false => {
            // Set network parameters
            let npars = NetworkPars {
                attachment: None,
                average_degree: Some(args.average_degree),
                connection_probability: None,
                initial_cluster_size: None,
                maximum_degree: Some(args.maximum_degree),
                minimum_degree: Some(args.minimum_degree),
                model: args.network_model,
                powerlaw_exponent: Some(args.powerlaw_exponent),
                rewiring_probability: Some(args.rewiring_probability),
                size: args.network_size,
            };

            // Set opinion parameters
            let opars = OpinionPars::new(
                args.active_fraction, 
                Some(args.hesitancy_attribution_model),
                args.threshold, 
                args.zealot_fraction,
            );

            // Set epidemic parameters
            let epars = EpidemicPars::new(
                0.0, 
                args.infection_decay, 
                args.infection_rate, 
                args.nseeds,
                args.r0,
                0.0, 
                false, 
                args.seed_model,  
                0.0, 
                args.vaccination_rate,
            );

            // Set experiment/algorithm parameters
            let apars = AlgorithmPars::new(
                args.experiment_id,
                args.nsims_net, 
                args.nsims_dyn, 
                args.t_max
            );

            // Set output flags
            let oflags = OutputPars::new(
                args.age_flag,
                args.agent_flag,
                args.agent_raw_flag,
                args.cluster_flag,
                args.cluster_raw_flag,
                args.degree_flag,
                args.global_flag,
                args.rebuild_flag,
                args.rebuild_raw_flag,
                args.time_flag,
                args.time_raw_flag,
           );

           Input::new(Some(apars), epars, Some(npars), Some(opars), Some(oflags), None)
        },
        true => {
            load_json_config(FILENAME_CONFIG, Some(FOLDER_CONFIG)).unwrap()
        },
    };

    // Prepare output ensemble to store all realizations results
    let mut output_ensemble = OutputEnsemble::new();

    // Loop over network realizations
    for _ in 0..pars.algorithm.unwrap().nsims_net {
        // Generate network
        let mut graph = Network::new();
        let mut count = 0;
        let max_count = PAR_NETWORK_TRIALS;
        while count < max_count {
            graph = graph.set_model(&pars.network.unwrap());
            if graph.is_connected_dfs() {
                println!("Single component network at trial {count}");
                break;
            }
            count += 1;
        }
        // Save network
        //graph.to_pickle(&npars);

        // Compute beta from given R0 and empirical network topology
        let beta = compute_beta_from_r0(
            pars.epidemic.r0, 
            pars.epidemic.infection_decay, 
            &pars.network.unwrap(), 
            &graph,
        );
        pars.epidemic.infection_rate = beta;

        // Watts-SIR coupled dynamics
        watts_sir_coupled_model(&pars, &graph, &mut output_ensemble);
    }

    output_ensemble.save_to_pickle(&pars);
}

// EXPERIMENT 2: WATTS-SIR DYNAMICS WITH SURVEY-BASED VACCINATION THRESHOLDS
pub fn run_exp2_datadriven_thresholds(args: Args) {

    let mut pars = match args.config_flag {
        false => {
            // Set network parameters
            let npars = NetworkPars {
                attachment: None,
                average_degree: Some(args.average_degree),
                connection_probability: None,
                initial_cluster_size: None,
                maximum_degree: Some(args.maximum_degree),
                minimum_degree: Some(args.minimum_degree),
                model: args.network_model,
                powerlaw_exponent: Some(args.powerlaw_exponent),
                rewiring_probability: Some(args.rewiring_probability),
                size: args.network_size,
            };

            // Set opinion parameters
            let opars = OpinionPars::new(
                args.active_fraction, 
                Some(args.hesitancy_attribution_model),
                args.threshold, 
                args.zealot_fraction,
            );

            // Set epidemic parameters
            let epars = EpidemicPars::new(
                0.0, 
                args.infection_decay,
                args.infection_rate, 
                args.nseeds,
                args.r0,
                0.0, 
                false, 
                args.seed_model, 
                0.0, 
                args.vaccination_rate,
            );

            // Set experiment/algorithm parameters
            let apars = AlgorithmPars::new(
                args.experiment_id,
                args.nsims_net, 
                args.nsims_dyn, 
                args.t_max
            );

            // Set output flags
            let oflags = OutputPars::new(
                args.age_flag,
                args.agent_flag,
                args.agent_raw_flag,
                args.cluster_flag,
                args.cluster_raw_flag,
                args.degree_flag,
                args.global_flag,
                args.rebuild_flag,
                args.rebuild_raw_flag,
                args.time_flag,
                args.time_raw_flag,
           );

           Input::new(Some(apars), epars, Some(npars), Some(opars), Some(oflags), None)
        },
        true => {
            load_json_config(FILENAME_CONFIG, Some(FOLDER_CONFIG)).unwrap()
        },
    };

    // Set data-driven average degree
    let dd_degree_filename =  FILENAME_DATA_AVERAGE_CONTACT;
    let average_degree: f64 = read_key_and_f64_from_json(args.usa_id, dd_degree_filename);
    let average_degree: usize = average_degree.ceil() as usize;
    pars.network.unwrap().average_degree = Some(average_degree); 
    
    // Set data-driven vaccination parameters
    let dd_vac_filename = FILENAME_DATA_VACCINATION_ATTITUDE;
    let fractions: Vec<f64> = read_key_and_vecf64_from_json(args.usa_id, dd_vac_filename);
    pars.vaccination.unwrap().already = fractions[0];
    pars.vaccination.unwrap().soon = fractions[1];
    pars.vaccination.unwrap().someone = fractions[2];
    pars.vaccination.unwrap().majority = fractions[3];
    pars.vaccination.unwrap().never = fractions[4];

    // Set data-driven opinion parameters
    pars.opinion.unwrap().active_fraction = fractions[0] + fractions[1];
    pars.opinion.unwrap().threshold = 0.0;
    pars.opinion.unwrap().zealot_fraction = fractions[4];

    // Prepare output ensemble to store all realizations results during 2 waves
    let mut output_ensemble1 = OutputEnsemble::new();
    let mut output_ensemble2 = OutputEnsemble::new();

    // Loop over network realizations
    for nsn in 0..pars.algorithm.unwrap().nsims_net {
        println!("Network realization={nsn}");
        
        // Generate network
        let mut graph = Network::new();
        let mut count = 0;
        let max_count = PAR_NETWORK_TRIALS;
        while count < max_count {
            graph = graph.set_model(&pars.network.unwrap());
            if graph.is_connected_dfs() {
                println!("Single component network at trial {count}");
                break;
            }
            count += 1;
        }
        // Save network
        //graph.to_pickle(&npars);
        
        // Compute beta from given R0 and empirical network topology
        let beta = compute_beta_from_r0(
            pars.epidemic.r0, 
            pars.epidemic.infection_decay, 
            &pars.network.unwrap(), 
            &graph,
        );
        pars.epidemic.infection_rate = beta;

        // Watts-SIR coupled dynamics
        watts_sir_coupled_model_datadriven_thresholds(
            &mut pars, 
            &graph, 
            &mut output_ensemble1, 
            &mut output_ensemble2
        );
    }

    // Store results
    if pars.output.unwrap().global {

        let pars_replica = Input::new(pars.algorithm, pars.epidemic, pars.network, pars.opinion, pars.output, pars.vaccination);
    
        let assembled_global_output1 = 
        output_ensemble1.assemble_global_observables();
        let assembled_global_output2 = 
        output_ensemble2.assemble_global_observables();
    
        let output_to_serialize = SerializeGlobalAssemblyTwoWaves {
            global_w1: assembled_global_output1,
            global_w2: assembled_global_output2,
            pars: pars_replica,
        };
    
        let serialized = 
        serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();

        let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, args.usa_id);
        let global_string = HEADER_GLOBAL.to_owned() + &exp_string.to_string();
        let file_name = write_file_name(&pars, global_string, false);
        let path = FOLDER_RESULTS.to_owned() + &file_name + ".pickle";
        std::fs::write(path, serialized).unwrap();
    }

    if pars.output.unwrap().cluster {
        let pars_replica = Input::new(pars.algorithm, pars.epidemic, pars.network, pars.opinion, pars.output, pars.vaccination);
    
        let assembled_cluster_output1 = 
        output_ensemble1.assemble_cluster_observables();
        let assembled_cluster_output2 = 
        output_ensemble2.assemble_cluster_observables();

        if pars.output.unwrap().cluster_raw {
            let output_to_serialize = SerializeClusterAssemblyTwoWaves {
                attitude_w1: None,
                attitude_w2: None,
                cascading_w1: None,
                cascading_w2: None,
                opinion_health_w1: Some(assembled_cluster_output1),
                opinion_health_w2: Some(assembled_cluster_output2),
                pars: pars_replica,
            };

            let serialized = 
            serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();

            let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, args.usa_id);        
            let clusters_string = HEADER_CLUSTER.to_owned() + &exp_string;
            let file_name = write_file_name(&pars, clusters_string, false);
            let path = FOLDER_RESULTS.to_owned() + &file_name + ".pickle";
            std::fs::write(path, serialized).unwrap();
        } else {
            let cluster_stat_package1 = compute_cluster_stats(&assembled_cluster_output1);
            let csp1_serialized = serde_pickle::to_vec(&cluster_stat_package1, SerOptions::new(),).unwrap();
            let csp1_string = "cspw1_".to_owned() + &pars.algorithm.unwrap().experiment_id.to_string();
            let file_name1 = write_file_name(&pars, csp1_string, false);
            let path1 = FOLDER_RESULTS.to_owned() + &file_name1 + ".pickle";
            std::fs::write(path1, csp1_serialized).unwrap();

            let cluster_distribution1 = compute_cluster_distribution(&assembled_cluster_output1);
            let cd1_serialized = serde_pickle::to_vec(&cluster_distribution1, SerOptions::new(),).unwrap();
            let cd1_string = "cdw1_".to_owned() + &pars.algorithm.unwrap().experiment_id.to_string();
            let file_name1 = write_file_name(&pars, cd1_string, false);
            let path1 = FOLDER_RESULTS.to_owned() + &file_name1 + ".pickle";
            std::fs::write(path1, cd1_serialized).unwrap();

            let cluster_stat_package2 = compute_cluster_stats(&assembled_cluster_output2);
            let csp2_serialized = serde_pickle::to_vec(&cluster_stat_package2, SerOptions::new(),).unwrap();
            let csp2_string = "cspw2_".to_owned() + &pars.algorithm.unwrap().experiment_id.to_string();
            let file_name2 = write_file_name(&pars, csp2_string, false);
            let path2 = FOLDER_RESULTS.to_owned() + &file_name2 + ".pickle";
            std::fs::write(path2, csp2_serialized).unwrap();

            let cluster_distribution2 = compute_cluster_distribution(&assembled_cluster_output2);
            let cd2_serialized = serde_pickle::to_vec(&cluster_distribution2, SerOptions::new(),).unwrap();
            let cd2_string = "cdw2_".to_owned() + &pars.algorithm.unwrap().experiment_id.to_string();
            let file_name2 = write_file_name(&pars, cd2_string, false);
            let path2 = FOLDER_RESULTS.to_owned() + &file_name2 + ".pickle";
            std::fs::write(path2, cd2_serialized).unwrap();
        }
    }

    if pars.output.unwrap().agent {
        let pars_replica = 
        Input::new(pars.algorithm, pars.epidemic, pars.network, pars.opinion, pars.output, pars.vaccination);
    
        let assembled_agent_output1 = 
        output_ensemble1.assemble_agent_observables(&pars);
        let assembled_agent_output2 = 
        output_ensemble2.assemble_agent_observables(&pars);

        if pars.output.unwrap().agent_raw {
            let output_to_serialize = SerializedAgentAssemblyTwoWaves {
                agent_w1: assembled_agent_output1,
                agent_w2: assembled_agent_output2,
                pars: pars_replica,
            };

            let serialized = 
            serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();
            let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, args.usa_id);  
            let agents_string = HEADER_AGENT.to_owned() + &exp_string;
            let file_name = write_file_name(&pars, agents_string, false);
            let path = FOLDER_RESULTS.to_owned() + &file_name + ".pickle";
            std::fs::write(path, serialized).unwrap();
        } else {
            todo!()
        }
    }

    if pars.output.unwrap().time {
        let pars_replica = Input::new(pars.algorithm, pars.epidemic, pars.network, pars.opinion, pars.output, pars.vaccination);

        let assembled_time_series1 = output_ensemble1.assemble_time_series(pars.algorithm.unwrap().t_max);
        let assembled_time_series2 = output_ensemble2.assemble_time_series(pars.algorithm.unwrap().t_max);

        if pars.output.unwrap().time_raw {
            let output_to_serialize = SerializedTimeSeriesAssemblyTwoWaves {
                time_w1: assembled_time_series1,
                time_w2: assembled_time_series2,
                pars: pars_replica,
            };

            let serialized = 
            serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();
            let exp_string = format!("{}_{}", pars.algorithm.unwrap().experiment_id, args.usa_id);  
            let time_string = HEADER_TIME.to_owned() + &exp_string;
            let file_name = write_file_name(&pars, time_string, false);
            let path = FOLDER_RESULTS.to_owned() + &file_name + ".pickle";
            std::fs::write(path, serialized).unwrap();
        } else {
            todo!()
        }
    }
}

// EXPERIMENT 3: WATTS-SIR DYNAMICS WITH SURVEY-BASED VACCINATION THRESHOLDS IN AGE-STRUCTURED MULTILAYER NETWORK
pub fn run_exp3_multilayer_thresholds(args: Args) {
    let mut pars = match args.config_flag {
        false => {
            let opars = OpinionPars::new(
                args.active_fraction, 
                Some(args.hesitancy_attribution_model),
                args.threshold, 
                args.zealot_fraction,
            );

            let epars = EpidemicPars::new(
                0.0, 
                args.infection_decay,
                args.infection_rate, 
                args.nseeds,
                args.r0,
                0.0,
                false,
                args.seed_model, 
                0.0, 
                args.vaccination_rate,
            );

            let npars = NetworkPars {
                attachment: None,
                average_degree: Some(0),
                connection_probability: None,
                initial_cluster_size: None,
                maximum_degree: None,
                minimum_degree: None,
                model: args.network_model,
                powerlaw_exponent: None,
                rewiring_probability: None,
                size: args.network_size,
            };

            let apars = AlgorithmPars::new(
                args.experiment_id,
                args.nsims_dyn, 
                args.nsims_net, 
                args.t_max
            );

            let oflags = OutputPars::new(
                args.age_flag,
                args.agent_flag,
                args.agent_raw_flag,
                args.cluster_flag,
                args.cluster_raw_flag,
                args.degree_flag,
                args.global_flag,
                args.rebuild_flag,
                args.rebuild_raw_flag,
                args.time_flag,
                args.time_raw_flag,
            );

            let vpars = VaccinationPars::new(
                args.age_threshold,
                0.0, 
                args.hesitancy_attribution_model,
                0.0,
                0.0, 
                0.0, 
                0.0, 
                args.underage_correction_flag,
                Some(args.usa_id), 
                0.0, 
                args.vaccination_policy,
                args.vaccination_quota,
                args.vaccination_rate,
            );

           Input::new(Some(apars), epars, Some(npars), Some(opars), Some(oflags), Some(vpars))
        },
        true => {
            load_json_config(FILENAME_CONFIG, Some(FOLDER_CONFIG)).unwrap()
        },
    };

    let dd_vac_filename = FILENAME_DATA_VACCINATION_ATTITUDE;
    let fractions: Vec<f64> = read_key_and_vecf64_from_json(
        pars.vaccination.unwrap().us_state.unwrap(), 
        dd_vac_filename,
    );

    let popultion_filename = FILENAME_DATA_POPULATION_AGE;
    let mut population_vector: Vec<f64> = read_key_and_vecf64_from_json(pars.vaccination.unwrap().us_state.unwrap(), popultion_filename);
    let population_cdf = build_normalized_cdf(&mut population_vector);
    let ngroups =population_vector.len();

    let contact_filename = FILENAME_DATA_CONTACT_MATRIX;
    let contact_matrix: Vec<Vec<f64>> = read_key_and_matrixf64_from_json(pars.vaccination.unwrap().us_state.unwrap(), contact_filename); 
    let interlayer_probability_matrix = compute_interlayer_probability_matrix(&contact_matrix);
    let intralayer_average_degree = compute_intralayer_average_degree(&contact_matrix);

    let eligible_fraction = if args.underage_correction_flag {
        let underaged_fraction = count_underaged(&population_vector);
        1.0 - underaged_fraction 
    } else {
        pars.vaccination.as_mut().unwrap().age_threshold = 0;
        1.0
    };

    pars.vaccination.as_mut().unwrap().already = eligible_fraction * fractions[0];
    pars.vaccination.as_mut().unwrap().soon = eligible_fraction * fractions[1];
    pars.vaccination.as_mut().unwrap().someone = eligible_fraction * fractions[2];
    pars.vaccination.as_mut().unwrap().majority = eligible_fraction * fractions[3];
    pars.vaccination.as_mut().unwrap().never = eligible_fraction * fractions[4];

    pars.opinion.as_mut().unwrap().active_fraction = pars.vaccination.unwrap().already + pars.vaccination.unwrap().soon;
    pars.opinion.as_mut().unwrap().threshold = 0.0;
    pars.opinion.as_mut().unwrap().zealot_fraction = pars.vaccination.unwrap().never;

    let mut output_ensemble = OutputEnsemble::new();

    for nsn in 0..pars.algorithm.unwrap().nsims_net {
        println!("Network realization={nsn}");

        let mut agent_ensemble = AgentEnsemble::new(pars.network.unwrap().size);

        agent_ensemble.sample_age(&population_cdf);

        agent_ensemble.sample_degree_conditioned_to_age(&intralayer_average_degree);

        let mut intralayer_stubs = agent_ensemble.collect_intralayer_stubs(ngroups);

        agent_ensemble.generate_age_multilayer_network(&interlayer_probability_matrix, &mut intralayer_stubs);

        let r0 = pars.epidemic.r0;
        let infection_decay = pars.epidemic.infection_decay;
        let average_degree = agent_ensemble.average_degree();
        let beta = r0 * infection_decay / average_degree;
        pars.epidemic.infection_rate = beta;

        agent_ensemble.set_vaccination_policy_model(
            pars.vaccination.unwrap().vaccination_policy,
            pars.vaccination.unwrap().vaccination_quota,
        );

        watts_sir_coupled_model_multilayer(
            &mut pars,
            &mut agent_ensemble, 
            &mut output_ensemble,
        );
    }

    output_ensemble.save_to_pickle(&pars);
}