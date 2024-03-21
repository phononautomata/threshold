use clap::Parser;

use crate::agent::{AgentEnsemble, HesitancyAttributionModel, SeedModel, VaccinationPolicy};
use crate::cons::{FILENAME_CONFIG, FILENAME_DATA_CONTACT_MATRIX, FILENAME_DATA_POPULATION_AGE, FILENAME_DATA_VACCINATION_ATTITUDE, FOLDER_CONFIG};
use crate::utils::{
    build_normalized_cdf, compute_interlayer_probability_matrix, compute_intralayer_average_degree, count_underaged, load_json_config, read_key_and_matrixf64_from_json, read_key_and_vecf64_from_json, AlgorithmPars, EpidemicPars, Input, OpinionPars, OutputEnsemble, OutputPars, USState, VaccinationPars
};
use crate::core::watts_sir_coupled_model_multilayer;

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
    #[clap(long, value_parser, default_value_t = true)]
    pub attitude_flag: bool,
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

pub fn build_multilayer(args: Args) {
    todo!()
}

pub fn run_epidemic(args: Args) {
    todo!()
}

pub fn run_full(args: Args) {
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
                args.attitude_flag,
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