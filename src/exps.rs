//use std::path::PathBuf;
use clap::Parser;
use serde_pickle::ser::SerOptions;
//use config::{Config, File};
//use std::time::SystemTime;

use crate::agent::SeedModel;
use crate::analysis::{compute_beta_from_r0, compute_cluster_stats, compute_cluster_distribution, compute_agent_stats, compute_agent_distribution};
use crate::utils::{
    OpinionPars, EpidemicPars, AlgorithmPars, Input, 
    OutputPars, OutputEnsemble, SerializeGlobalAssembly, write_file_name, 
    USState, read_categories, VaccinationPars, 
    SerializeGlobalAssemblyTwoWaves, SerializedAgentAssemblyTwoWaves, 
    SerializedTimeSeriesAssemblyTwoWaves, SerializeClusterAssemblyTwoWaves, 
    select_network_model, convert_hm_value_to_f64, load_json_data, 
    convert_hm_value_to_bool, read_degree
};
use crate::core::{
    watts_sir_coupled_model, 
    datadriven_watts_sir_coupled_model, 
};

use netrust::network::Network;
use netrust::utils::{
    NetworkModel, 
    NetworkPars, 
    build_network_parameter_enum_from_map, 
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsLine {
    #[clap(long, value_parser, default_value_t = 10)]
    pub average_degree: usize,
    #[clap(long, value_parser, default_value_t = 20000)]
    pub n: usize,
    #[clap(long, value_parser, default_value = "erdos-renyi")]
    pub net_id: NetworkModel,
    #[clap(long, value_parser, default_value_t = 0.5)]
    pub active_fraction: f64,
    #[clap(long, value_parser, default_value_t = 0.1)]
    pub threshold: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub zealot_fraction: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub immunity_decay: f64,
    #[clap(long, value_parser, default_value_t = 0.46)]
    pub infection_rate: f64,
    #[clap(long, value_parser, default_value_t = 0.2)]
    pub infection_decay: f64,
    #[clap(long, value_parser, default_value_t = false)]
    pub secondary_outbreak: bool,
    #[clap(long, value_parser, default_value = "top-degree-neighborhood")]
    pub seed_model: SeedModel,
    #[clap(long, value_parser, default_value_t = 10)]
    pub seeds: usize,
    #[clap(long, value_parser, default_value_t = 1.5)]
    pub r0: f64,
    #[clap(long, value_parser, default_value_t = 1.5)]
    pub r0_w2: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub vaccination_decay: f64,
    #[clap(long, value_parser, default_value_t = 0.005)]
    pub vaccination_rate: f64,
    #[clap(long, value_parser, default_value = "massachusetts")]
    pub usa_id: USState,
    #[clap(long, value_parser, default_value_t = 2)]
    pub exp_flag: usize,
    #[clap(long, value_parser, default_value_t = 25)]
    pub nsims_dyn: usize,
    #[clap(long, value_parser, default_value_t = 25)]
    pub nsims_net: usize,
    #[clap(long, value_parser, default_value_t = 500)]
    pub t_max: usize,
    #[clap(long, value_parser, default_value_t = false)]
    pub agent_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub cluster_flag: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub global_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub time_flag: bool,
}

/// Input arguments
#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsNetwork {
    #[clap(long, value_parser, default_value_t = 10)]
    pub average_degree: usize,
    #[clap(long, value_parser, default_value_t = 20000)]
    pub n: usize,
    #[clap(long, value_parser, default_value = "erdos-renyi")]
    pub net_id: NetworkModel,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsOpinion {
    #[clap(long, value_parser, default_value_t = 0.1)]
    pub active_fraction: f64,
    #[clap(long, value_parser, default_value_t = 0.1)]
    pub threshold: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub zealot_fraction: f64,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsEpidemic {
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub immunity_decay: f64,
    #[clap(long, value_parser, default_value_t = 0.46)]
    pub infection_rate: f64,
    #[clap(long, value_parser, default_value_t = 0.2)]
    pub infection_decay: f64,
    #[clap(long, value_parser, default_value_t = false)]
    pub secondary_outbreak: bool,
    #[clap(long, value_parser, default_value = "top-degree-neighborhood")]
    pub seed_model: SeedModel,
    #[clap(long, value_parser, default_value_t = 10)]
    pub seeds: usize,
    #[clap(long, value_parser, default_value_t = 1.5)]
    pub r0: f64,
    #[clap(long, value_parser, default_value_t = 1.5)]
    pub r0_w2: f64,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsVaccination {
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub vaccination_decay: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub vaccination_rate: f64,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsDataDrivenUS {
    #[clap(long, value_parser, default_value = "massachusetts")]
    pub usa_id: USState,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsExperiment {
    #[clap(long, value_parser, default_value_t = 1)]
    pub exp_flag: usize,
    #[clap(long, value_parser, default_value_t = 35)]
    pub nsims_dyn: usize,
    #[clap(long, value_parser, default_value_t = 35)]
    pub nsims_net: usize,
    #[clap(long, value_parser, default_value_t = 500)]
    pub t_max: usize,
}

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct ArgsOutput {
    #[clap(long, value_parser, default_value_t = false)]
    pub agent_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub cluster_flag: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub global_flag: bool,
    #[clap(long, value_parser, default_value_t = false)]
    pub time_flag: bool,
}

// SETTING 1: WATTS-SIR DYNAMICS UNDER HOMOGENEOUS THRESHOLD
pub fn run_exp1_homogeneous(
    args_lin: ArgsLine,
    //args_net: ArgsNetwork, 
    //args_opi: ArgsOpinion,  
    //args_epi: ArgsEpidemic,
    //args_vac: ArgsVaccination, 
) {
    // Load network parameters into 
    let network_model = args_lin.net_id;
    let network_size = args_lin.n;
    let network_hm = select_network_model(args_lin.net_id);
    let mut network_hm = convert_hm_value_to_f64(network_hm);
    network_hm.insert("average_degree".to_string(), args_lin.average_degree as f64);
    let network_enum = build_network_parameter_enum_from_map(
        network_model, 
        network_size, 
        &network_hm
    );

    // Set network parameters
    let npars = NetworkPars {
         n: network_size,
         model: network_model,
         pars: network_enum,
    };

    // Set opinion parameters
    let opars = OpinionPars::new(
        args_lin.active_fraction, 
        args_lin.threshold, 
        args_lin.zealot_fraction,
    );

    // Load epidemic parameters
    let epi_filename = "config_epidemic";
    let epidemic_hm = load_json_data(epi_filename);
    let epidemic_hm = convert_hm_value_to_f64(epidemic_hm);

    let seeds = *epidemic_hm.get("seeds").unwrap();
    let r0 = *epidemic_hm.get("r0").unwrap();
    let infection_rate = *epidemic_hm.get("infection_rate").unwrap();
    let infection_decay = *epidemic_hm.get("infection_decay").unwrap();

    // Set epidemic parameters
    let epars = EpidemicPars::new(
        0.0, 
        infection_rate, 
        infection_decay, 
        r0,
        0.0, 
        false, 
        args_lin.seed_model, 
        seeds as usize, 
        0.0, 
        args_lin.vaccination_rate,
    );

    // Load experiment/algorithm parameters
    let exp_filename = "config_experiment";
    let experiment_hm = load_json_data(exp_filename);
    let experiment_hm = convert_hm_value_to_f64(experiment_hm);

    let exp_flag = *experiment_hm.get("exp_flag").unwrap() as usize;
    let nsims_net = *experiment_hm.get("nsims_net").unwrap() as usize;
    let nsims_dyn = *experiment_hm.get("nsims_dyn").unwrap() as usize;
    let t_max = *experiment_hm.get("t_max").unwrap() as usize;
    
    // Set experiment/algorithm parameters
    let apars = AlgorithmPars::new(
        nsims_net, 
        nsims_dyn, 
        t_max
    );

    // Load output flags
   let out_filename = "config_output";
   let output_hm = load_json_data(out_filename);
   let output_hm = convert_hm_value_to_bool(output_hm);

   let global_flag = *output_hm.get("global").unwrap();
   let agent_flag = *output_hm.get("agent").unwrap();
   let agent_raw = *output_hm.get("agent_raw").unwrap();
   let cluster_flag = *output_hm.get("cluster").unwrap();
   let cluster_raw = *output_hm.get("cluster_raw").unwrap();
   let time_flag = *output_hm.get("time").unwrap();
   let time_raw = *output_hm.get("time_raw").unwrap();

   // Set output flags
   let oflags = OutputPars::new(
        agent_flag,
        agent_raw,
        cluster_flag,
        cluster_raw,
        time_flag,
        time_raw,
    );

    // Pack all parameters
    let mut pars = Input::new(npars, opars, epars, None, apars, oflags);

    // Prepare output ensemble to store all realizations results
    let mut output_ensemble = OutputEnsemble::new();

    // Loop over network realizations
    for _ in 0..nsims_net {
        // Generate network
        let mut graph = Network::new();
        let mut count = 0;
        let max_count = 100;
        while count < max_count {
            graph = graph.set_model(&npars);
            if graph.is_connected_dfs() {
                println!("Single component network at trial {count}");
                break;
            }
            count += 1;
        }
        // Save network
        //graph.to_pickle(&npars);

        // Compute beta from given R0 and empirical network topology
        let beta = compute_beta_from_r0(r0, infection_decay, &npars, &graph);
        pars.epidemic.infection_rate = beta;

        // Watts-SIR coupled dynamics
        watts_sir_coupled_model(&pars, &graph, &mut output_ensemble);
    }

    // Store results
    if global_flag {
        // Create a replica for input parameters object
        let pars_replica = 
        Input::new(npars, opars, epars, None, apars, oflags);
        // Assemble global observables
        let assembled_global_output = 
        output_ensemble.assemble_global_observables();
        // Output data to serialize
        let output_to_serialize = SerializeGlobalAssembly {
            global: assembled_global_output,
            pars: pars_replica,
        };
        // Serialize output
        let serialized = serde_pickle::to_vec(
            &output_to_serialize, 
            SerOptions::new()
        ).unwrap();
      
        // Write the serialized byte vector to file
        let global_string = "global_".to_owned() + &exp_flag.to_string();
        let file_name = write_file_name(&pars, global_string);
        let path = "results/".to_owned() + &file_name + ".pickle";
        std::fs::write(path, serialized).unwrap();
    }
    
    if cluster_flag {
        let assembled_cluster_output = 
        output_ensemble.assemble_cluster_observables();
        
        if cluster_raw {
            // Load to pickle
            let serialized = serde_pickle::to_vec(
                &assembled_cluster_output, 
                SerOptions::new(),
            ).unwrap();
            let cluster_string = "clusters_".to_owned() + &exp_flag.to_string();
            let file_name = write_file_name(&pars, cluster_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, serialized).unwrap();
        } else {
            let cluster_stat_package = compute_cluster_stats(&assembled_cluster_output);
            let csp_serialized = serde_pickle::to_vec(&cluster_stat_package, SerOptions::new(),).unwrap();
            let csp_string = "csp_".to_owned() + &exp_flag.to_string();
            let file_name = write_file_name(&pars, csp_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, csp_serialized).unwrap();

            let cluster_distribution = compute_cluster_distribution(&assembled_cluster_output);
            let cd_serialized = serde_pickle::to_vec(&cluster_distribution, SerOptions::new(),).unwrap();
            let cd_string = "cd_".to_owned() + &exp_flag.to_string();
            let file_name = write_file_name(&pars, cd_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, cd_serialized).unwrap();
        }
    }
    
    if agent_flag {
        let assembled_agent_output = 
        output_ensemble.assemble_agent_observables(&pars);

        if agent_raw {
            // Load to pickle
            let serialized = serde_pickle::to_vec(
                &assembled_agent_output, 
                SerOptions::new(),
            ).unwrap();
            let agents_string = "agents_".to_owned() + &exp_flag.to_string();
            let file_name = write_file_name(&pars, agents_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, serialized).unwrap();
        } else {
            let agent_stat_package = compute_agent_stats(&assembled_agent_output);
            let asp_serialized = serde_pickle::to_vec(&agent_stat_package, SerOptions::new(),).unwrap();
            let asp_string = "asp_".to_owned() + &exp_flag.to_string();
            let file_name = write_file_name(&pars, asp_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, asp_serialized).unwrap();

            let agent_distribution = compute_agent_distribution(&assembled_agent_output);
            let ad_serialized = serde_pickle::to_vec(&agent_distribution, SerOptions::new(),).unwrap();
            let ad_string = "ad_".to_owned() + &exp_flag.to_string();
            let file_name = write_file_name(&pars, ad_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, ad_serialized).unwrap();
        }
    }
    
    if time_flag {
        // Assemble time observables
        let assembled_time_series = 
        output_ensemble.assemble_time_series(t_max);
        
        if time_raw {
            // Load to pickle
            let serialized = serde_pickle::to_vec(
                &assembled_time_series, 
                SerOptions::new()
            ).unwrap();
            let time_string = "time_".to_owned() + &exp_flag.to_string();
            let file_name = write_file_name(&pars, time_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, serialized).unwrap();
        } else {
            todo!()
        }
    }
}

// SETTING 2: WATTS-SIR DYNAMICS WITH DATA-DRIVEN VACCINATION THRESHOLDS
pub fn run_exp2_datadriven(
    args_lin: ArgsLine,
    //args_net: ArgsNetwork, 
    //args_epi: ArgsEpidemic,
    //args_dat: ArgsDataDrivenUS,
) {

    // Average degree data
    let dd_degree_filename = "average_contacts_data";
    let average_degree: f64 = read_degree(args_lin.usa_id, dd_degree_filename);
    let average_degree: usize = average_degree.ceil() as usize;

    // Load network parameters into 
    let network_model = args_lin.net_id;
    let network_size = args_lin.n;
    let network_hm = select_network_model(args_lin.net_id);
    let mut network_hm = convert_hm_value_to_f64(network_hm);
    network_hm.insert("average_degree".to_string(), average_degree as f64);
    let network_enum = build_network_parameter_enum_from_map(
        network_model, 
        network_size, 
        &network_hm
    );

    let npars = NetworkPars {
         n: network_size,
         model: network_model,
         pars: network_enum,
    };

    // Set opinion parameters
    let opars = OpinionPars::new(
        0.0, 
        0.0, 
        0.0,
    );

    // Load epidemic parameters into EpidemicPars struct
    let epi_filename = "config_epidemic";
    let epidemic_hm = load_json_data(epi_filename);
    let epidemic_hm = convert_hm_value_to_f64(epidemic_hm);
    
    let vac_filename = "config_vaccination";
    let vaccination_hm = load_json_data(vac_filename);
    let _vaccination_hm = convert_hm_value_to_f64(vaccination_hm);

    let seeds = *epidemic_hm.get("seeds").unwrap();
    let r0 = *epidemic_hm.get("r0").unwrap();
    let infection_rate = *epidemic_hm.get("infection_rate").unwrap();
    let infection_decay = *epidemic_hm.get("infection_decay").unwrap();
    
    let vaccination_rate = args_lin.vaccination_rate; //*vaccination_hm.get("vaccination_rate").unwrap();
    let secondary_outbreak = epidemic_hm.get("secondary_outbreak").map_or(false, |&value| value != 0.0);
    let r0_w2 = *epidemic_hm.get("r0_w2").unwrap();
    
    let epars = EpidemicPars::new(
        0.0, 
        infection_rate, 
        infection_decay, 
        r0, 
        r0_w2, 
        secondary_outbreak, 
        args_lin.seed_model, 
        seeds as usize, 
        0.0, 
        vaccination_rate
    );

    // Load configuration file
    //let mut config = Config::default();
    //config.merge(File::with_name("config/default")).unwrap();
    //let path: String = config.get("general.path").unwrap();

    // Set vaccination parameters
    //let file_path = PathBuf::from(path).join("data").join("vaccination_data.json");
    //let file_path = file_path.as_path().to_str().unwrap();
    let dd_vac_filename = "vaccination_data";
    let fractions: Vec<f64> = read_categories(args_lin.usa_id, dd_vac_filename);
    let vpars = VaccinationPars::new(
       vaccination_rate,
       0.0,
       args_lin.usa_id, 
       fractions[0], 
       fractions[1], 
       fractions[2], 
       fractions[3], 
       fractions[4],
    );

    // Load experiment/algorithm parameters
    let exp_filename = "config_experiment";
    let experiment_hm = load_json_data(exp_filename);
    let experiment_hm = convert_hm_value_to_f64(experiment_hm);

    let exp_flag = *experiment_hm.get("exp_flag").unwrap() as usize;
    let nsims_net = *experiment_hm.get("nsims_net").unwrap() as usize;
    let nsims_dyn = *experiment_hm.get("nsims_dyn").unwrap() as usize;
    let t_max = *experiment_hm.get("t_max").unwrap() as usize;
    
    let apars = AlgorithmPars::new(
        nsims_net, 
        nsims_dyn, 
        t_max
    );

    // Set output flags
   let out_filename = "config_output";
   let output_hm = load_json_data(out_filename);
   let output_hm = convert_hm_value_to_bool(output_hm);

   let global_flag = *output_hm.get("global").unwrap();
   let agent_flag = *output_hm.get("agent").unwrap();
   let agent_raw = *output_hm.get("agent_raw").unwrap();
   let cluster_flag = *output_hm.get("cluster").unwrap();
   let cluster_raw = *output_hm.get("cluster_raw").unwrap();
   let time_flag = *output_hm.get("time").unwrap();
   let time_raw = *output_hm.get("time_raw").unwrap();

   let oflags = OutputPars::new(
        agent_flag,
        agent_raw,
        cluster_flag,
        cluster_raw,
        time_flag,
        time_raw,
    );

    // Pack all parameters
    let mut pars = Input::new(npars, opars, epars, Some(vpars), apars, oflags);

    // Prepare output ensemble to store all realizations results during 2 waves
    let mut output_ensemble1 = OutputEnsemble::new();
    let mut output_ensemble2 = OutputEnsemble::new();

    // Loop over network realizations
    for nsn in 0..nsims_net {
        println!("Network realization={nsn}");
        
        // Generate network
        let mut graph = Network::new();
        let mut count = 0;
        let max_count = 100;
        while count < max_count {
            graph = graph.set_model(&npars);
            if graph.is_connected_dfs() {
                println!("Single component network at trial {count}");
                break;
            }
            count += 1;
        }
        // Save network
        //graph.to_pickle(&npars);
        
        // Compute beta from given R0 and empirical network topology
        let beta = compute_beta_from_r0(r0, infection_decay, &npars, &graph);
        pars.epidemic.infection_rate = beta;

        // Watts-SIR coupled dynamics
        datadriven_watts_sir_coupled_model(
            &mut pars, 
            &graph, 
            &mut output_ensemble1, 
            &mut output_ensemble2
        );
    }

    // Store results
    if global_flag {
        // Create a replica for input parameters object
        let vpars = vpars;
        let pars_replica = 
        Input::new(npars, opars, epars, Some(vpars), apars, oflags);
        // Assemble global observables
        let assembled_global_output1 = 
        output_ensemble1.assemble_global_observables();
        let assembled_global_output2 = 
        output_ensemble2.assemble_global_observables();
        // Output data to serialize
        let output_to_serialize = SerializeGlobalAssemblyTwoWaves {
            global_w1: assembled_global_output1,
            global_w2: assembled_global_output2,
            pars: pars_replica,
        };
        // Serialize output
        let serialized = 
        serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();

        // Write the serialized byte vector to file
        let exp_string = format!("{}_{}", exp_flag, args_lin.usa_id);
        let global_string = "global_".to_owned() + &exp_string.to_string();
        let file_name = write_file_name(&pars, global_string);
        let path = "results/".to_owned() + &file_name + ".pickle";
        std::fs::write(path, serialized).unwrap();
    }

    if cluster_flag {
        // Create a replica for input parameters object
        let vpars = vpars;
        let pars_replica = 
        Input::new(npars, opars, epars, Some(vpars), apars, oflags);
        // Assemble agent observables
        let assembled_cluster_output1 = 
        output_ensemble1.assemble_cluster_observables();
        let assembled_cluster_output2 = 
        output_ensemble2.assemble_cluster_observables();

        if cluster_raw {
            // Output data to serialize
            let output_to_serialize = SerializeClusterAssemblyTwoWaves {
                cluster_w1: assembled_cluster_output1,
                cluster_w2: assembled_cluster_output2,
                pars: pars_replica,
            };
            // Load to pickle
            let serialized = 
            serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();

            let exp_string = format!("{}_{}", exp_flag, args_lin.usa_id);        
            let clusters_string = "clusters_".to_owned() + &exp_string;
            let file_name = write_file_name(&pars, clusters_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, serialized).unwrap();
        } else {
            let cluster_stat_package1 = compute_cluster_stats(&assembled_cluster_output1);
            let csp1_serialized = serde_pickle::to_vec(&cluster_stat_package1, SerOptions::new(),).unwrap();
            let csp1_string = "cspw1_".to_owned() + &exp_flag.to_string();
            let file_name1 = write_file_name(&pars, csp1_string);
            let path1 = "results/".to_owned() + &file_name1 + ".pickle";
            std::fs::write(path1, csp1_serialized).unwrap();

            let cluster_distribution1 = compute_cluster_distribution(&assembled_cluster_output1);
            let cd1_serialized = serde_pickle::to_vec(&cluster_distribution1, SerOptions::new(),).unwrap();
            let cd1_string = "cdw1_".to_owned() + &exp_flag.to_string();
            let file_name1 = write_file_name(&pars, cd1_string);
            let path1 = "results/".to_owned() + &file_name1 + ".pickle";
            std::fs::write(path1, cd1_serialized).unwrap();

            let cluster_stat_package2 = compute_cluster_stats(&assembled_cluster_output2);
            let csp2_serialized = serde_pickle::to_vec(&cluster_stat_package2, SerOptions::new(),).unwrap();
            let csp2_string = "cspw2_".to_owned() + &exp_flag.to_string();
            let file_name2 = write_file_name(&pars, csp2_string);
            let path2 = "results/".to_owned() + &file_name2 + ".pickle";
            std::fs::write(path2, csp2_serialized).unwrap();

            let cluster_distribution2 = compute_cluster_distribution(&assembled_cluster_output2);
            let cd2_serialized = serde_pickle::to_vec(&cluster_distribution2, SerOptions::new(),).unwrap();
            let cd2_string = "cdw2_".to_owned() + &exp_flag.to_string();
            let file_name2 = write_file_name(&pars, cd2_string);
            let path2 = "results/".to_owned() + &file_name2 + ".pickle";
            std::fs::write(path2, cd2_serialized).unwrap();
        }
    }

    if agent_flag {
        // Create a replica for input parameters object
        let vpars = vpars;
        let pars_replica = 
        Input::new(npars, opars, epars, Some(vpars), apars, oflags);
        // Assemble agent observables
        let assembled_agent_output1 = 
        output_ensemble1.assemble_agent_observables(&pars);
        let assembled_agent_output2 = 
        output_ensemble2.assemble_agent_observables(&pars);
        
        if agent_raw {
            // Output data to serialize
            let output_to_serialize = SerializedAgentAssemblyTwoWaves {
                agent_w1: assembled_agent_output1,
                agent_w2: assembled_agent_output2,
                pars: pars_replica,
            };
            // Load to pickle
            let serialized = 
            serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();
            let exp_string = format!("{}_{}", exp_flag, args_lin.usa_id);  
            let agents_string = "agents_".to_owned() + &exp_string;
            let file_name = write_file_name(&pars, agents_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, serialized).unwrap();
        } else {
            todo!()
        }
    }

    if time_flag {
        // Create a replica for input parameters object
        let vpars = vpars;
        let pars_replica = 
        Input::new(npars, opars, epars, Some(vpars), apars, oflags);
        // Assemble time observables
        let assembled_time_series1 = 
        output_ensemble1.assemble_time_series(t_max);
        let assembled_time_series2 = 
        output_ensemble2.assemble_time_series(t_max);

        if time_raw {
            // Output data to serialize
            let output_to_serialize = SerializedTimeSeriesAssemblyTwoWaves {
                time_w1: assembled_time_series1,
                time_w2: assembled_time_series2,
                pars: pars_replica,
            };
            // Load to pickle
            let serialized = 
            serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();
            let exp_string = format!("{}_{}", exp_flag, args_lin.usa_id);  
            let time_string = "time_".to_owned() + &exp_string;
            let file_name = write_file_name(&pars, time_string);
            let path = "results/".to_owned() + &file_name + ".pickle";
            std::fs::write(path, serialized).unwrap();
        } else {
            todo!()
        }
    }
}

// SETTING 3: SYMMETRIC WATTS-SIRS DYNAMICS UNDER HOMOGENEOUS THRESHOLD
/* 
pub fn run_exp3_symmetric(args: Args) {
    // Set network parameters
    let model = args.net_id;
    let npars_vec = vec![args.n as f64, args.net_par1, args.net_par2, args.net_par3, args.net_par4];
    let npars_enum = build_network_parameter_enum(model, &npars_vec);
    let npars_hm = build_network_parameter_hashmap(args.net_id, &npars_vec);
    let npars = NetworkPars {
         n: args.n as usize,
         model,
         pars: npars_enum,
    };

    // Set opinion parameters
    let opars = OpinionPars::new(
        args.active_fraction, 
        args.theta, 
        0.0
    );

    // Set epidemic parameters
    let epars = EpidemicPars::new(
        args.seeds,
        args.r0,
        args.infection_rate, 
        args.infection_decay,
        args.immunity_decay,
        args.vaccination_rate,
        args.vaccination_decay, 
        args.secondary_outbreak,
        args.r0_w2,
    ); 

    // Set algorithm parameters
    let apars = AlgorithmPars::new(
        args.nsims_net, 
        args.nsims_dyn, 
        args.t_max
    );

    // Set output flags
    let oflags = OutputPars::new(
        args.agent_flag,
        args.cluster_flag,
        args.time_flag
    );

    // Pack all parameters
    let mut pars = Input::new(npars, opars, epars, None, apars, oflags);

    // Prepare output ensemble to store all realizations results
    let mut output_ensemble = OutputEnsemble::new();

    // Loop over network realizations
    for _ in 0..args.nsims_net {
        // Generate network
        let mut graph = Network::new();
        let mut count = 0;
        let max_count = 100;
        while count < max_count {
            graph = graph.set_model(args.net_id, &npars_hm);
            if graph.is_connected_dfs() {
                println!("Single component network at trial {count}");
                break;
            }
            count += 1;
        }
        // Save network
        let path = "results";
        graph.to_pickle(model, &npars_vec, path);
        // Compute beta from given R0 and empirical network topology
        let beta = compute_beta_from_r0(args.r0, args.infection_decay, &npars, &graph);
        pars.epidemic.infection_rate = beta;
        // Symmetric Watts-SIRS coupled dynamics
        symmetric_watts_sirs_coupled_model(&pars, &graph, &mut output_ensemble);
    }

    // Store results
    if args.global_flag {
        // Create a replica for input parameters object
        let pars_replica = Input::new(npars, opars, epars, None, apars, oflags);
        // Assemble global observables
        let assembled_global_output = output_ensemble.assemble_global_observables();
        // Output data to serialize
        let output_to_serialize = SerializeGlobalAssembly {
            global: assembled_global_output,
            pars: pars_replica,
        };
        // Serialize output
        let serialized = serde_pickle::to_vec(&output_to_serialize, SerOptions::new()).unwrap();
      
        // Write the serialized byte vector to file
        let global_string = "global_".to_owned() + &args.exp_flag.to_string();
        let file_name = write_file_name(&pars, &npars_vec, global_string);
        let path = "results/".to_owned() + &file_name + ".pickle";
        std::fs::write(path, serialized).unwrap();
    }
    if args.cluster_flag {
        // Assemble agent observables
        let assembled_cluster_output = output_ensemble.assemble_cluster_observables();
        // Load to pickle
        let serialized = serde_pickle::to_vec(&assembled_cluster_output, SerOptions::new()).unwrap();
        let agents_string = "clusters_".to_owned() + &args.exp_flag.to_string();
        let file_name = write_file_name(&pars, &npars_vec, agents_string);
        let path = "results/".to_owned() + &file_name + ".pickle";
        std::fs::write(path, serialized).unwrap();
    }
    if args.agent_flag {
        // Assemble agent observables
        let assembled_agent_output = output_ensemble.assemble_agent_observables(&pars);
        // Load to pickle
        let serialized = serde_pickle::to_vec(&assembled_agent_output, SerOptions::new()).unwrap();
        let agents_string = "agents_".to_owned() + &args.exp_flag.to_string();
        let file_name = write_file_name(&pars, &npars_vec, agents_string);
        let path = "results/".to_owned() + &file_name + ".pickle";
        std::fs::write(path, serialized).unwrap();
    }
    if args.time_flag {
        // Assemble time observables
        let assembled_time_series = output_ensemble.assemble_time_series(args.t_max);
        // Load to pickle
        let serialized = serde_pickle::to_vec(&assembled_time_series, SerOptions::new()).unwrap();
        let time_string = "time_".to_owned() + &args.exp_flag.to_string();
        let file_name = write_file_name(&pars, &npars_vec, time_string);
        let path = "results/".to_owned() + &file_name + ".pickle";
        std::fs::write(path, serialized).unwrap();
    }
}
*/
