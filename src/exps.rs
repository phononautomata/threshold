use std::env;
use std::path::PathBuf;

use clap::Parser;
use serde_pickle::SerOptions;
use uuid::Uuid;

use crate::agent::{
    AgentEnsemble, HesitancyModel, Multilayer, OpinionModel, SeedModel, VaccinationPolicy,
};
use crate::cons::{
    EXTENSION_RESULTS, FILENAME_CONFIG, FILENAME_DATA_CONTACT_MATRIX, FILENAME_DATA_POPULATION_AGE,
    FILENAME_DATA_VACCINATION_ATTITUDE, FOLDER_CONFIG, FOLDER_RESULTS, PATH_DATA_MULTILAYER,
};
use crate::core::dynamical_loop;
use crate::utils::{
    build_normalized_cdf, compute_interlayer_probability_matrix, compute_intralayer_average_degree,
    construct_string_epidemic, construct_string_multilayer, count_underaged,
    load_json_config_to_input_multilayer, load_multilayer_object, read_key_and_matrixf64_from_json,
    read_key_and_vecf64_from_json, rebuild_contact_data_from_multilayer, InputMultilayer, OutputEnsemble,
    OutputFlags, Region, VaccinationPars,
};

#[derive(Parser, Debug)]
#[clap(author, version, about, long_about = None)]
pub struct Args {
    #[clap(long, value_parser, default_value_t = false)]
    pub flag_config: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub flag_output_age: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub flag_output_agent: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub flag_output_attitude: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub flag_output_cluster: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub flag_output_contact: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub flag_output_degree: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub flag_output_global: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub flag_output_time: bool,
    #[clap(long, value_parser, default_value_t = true)]
    pub flag_underage: bool,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub fraction_active: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub fraction_majority: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub fraction_someone: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub fraction_soon: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub fraction_vaccinated: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub fraction_zealot: f64,
    #[clap(long, value_parser, default_value_t = 3)]
    pub id_experiment: usize,
    #[clap(long, value_parser, default_value = "random")]
    pub model_hesitancy: HesitancyModel,
    #[clap(long, value_parser, default_value = "homogeneous")]
    pub model_opinion: OpinionModel,
    #[clap(long, value_parser, default_value = "massachusetts")]
    pub model_region: Region,
    #[clap(long, value_parser, default_value = "top-degree-neighborhood")]
    pub model_seed: SeedModel,
    #[clap(long, value_parser, default_value_t = 100000)]
    pub nagents: usize,
    #[clap(long, value_parser, default_value_t = 10)]
    pub nsims: usize,
    #[clap(long, value_parser, default_value_t = 20)]
    pub nseeds: usize,
    #[clap(long, value_parser, default_value = "top-degree-neighborhood")]
    pub policy_vaccination: VaccinationPolicy,
    #[clap(long, value_parser, default_value_t = 1.0)]
    pub quota_vaccination: f64,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub rate_infection: f64,
    #[clap(long, value_parser, default_value_t = 0.2)]
    pub rate_removal: f64,
    #[clap(long, value_parser, default_value_t = 0.005)]
    pub rate_vaccination: f64,
    #[clap(long, value_parser, default_value_t = 1.5)]
    pub r0: f64,
    #[clap(long, value_parser, default_value_t = 500)]
    pub t_max: usize,
    #[clap(long, value_parser, default_value_t = 18)]
    pub threshold_age: usize,
    #[clap(long, value_parser, default_value_t = 0.0)]
    pub threshold_opinion: f64,
    #[clap(
        long,
        value_parser,
        default_value = "mlMassachusetts_n100000_47f7d63f-005a-4c49-97e0-776986135c84"
    )]
    pub uuid_multilayer: String,
}

pub fn generate_multilayer(args: Args) {
    let mut rng = rand::thread_rng();

    let population_filename = FILENAME_DATA_POPULATION_AGE;
    let mut population_vector: Vec<f64> =
        read_key_and_vecf64_from_json(args.model_region, population_filename);
    let population_cdf = build_normalized_cdf(&mut population_vector);
    let ngroups = population_vector.len();

    let contact_filename = FILENAME_DATA_CONTACT_MATRIX;
    let contact_matrix: Vec<Vec<f64>> =
        read_key_and_matrixf64_from_json(args.model_region, contact_filename);
    let interlayer_probability_matrix = compute_interlayer_probability_matrix(&contact_matrix);
    let intralayer_average_degree = compute_intralayer_average_degree(&contact_matrix);

    let mut node_ensemble = Multilayer::new(args.nagents);

    node_ensemble.sample_layer_from_cdf(&population_cdf, &mut rng);

    node_ensemble.sample_degree_conditioned_to_layer(&intralayer_average_degree, &mut rng);

    let mut intralayer_stubs = node_ensemble.collect_intralayer_stubs(ngroups);

    node_ensemble.generate_multilayer_network(
        &interlayer_probability_matrix,
        &mut intralayer_stubs,
        &mut rng,
    );

    let uuid_ml = Uuid::new_v4().to_string();
    let string_multilayer = construct_string_multilayer(args.model_region, args.nagents);
    let path_base = PATH_DATA_MULTILAYER;

    node_ensemble.to_pickle(&uuid_ml, &string_multilayer, &args.model_region.to_string(), &path_base);

    if args.flag_output_contact {
        let output_contact = rebuild_contact_data_from_multilayer(&node_ensemble);

        let serialized = serde_pickle::to_vec(&output_contact, SerOptions::new()).unwrap();

        let mut path = PathBuf::from(env::current_dir().expect("Failed to get current directory"));
        path.push(FOLDER_RESULTS);
        path.push(format!(
            "contact_{}_{}{}",
            string_multilayer, uuid_ml, EXTENSION_RESULTS
        ));
        std::fs::write(path, serialized).unwrap();
    }
}

pub fn run_epidemic(args: Args) {
    let mut pars_model = match args.flag_config {
        false => InputMultilayer::new(
            args.flag_underage,
            args.fraction_active,
            args.fraction_majority,
            args.fraction_someone,
            args.fraction_soon,
            args.fraction_vaccinated,
            args.fraction_zealot,
            args.id_experiment,
            args.model_hesitancy,
            args.model_opinion,
            args.model_region,
            args.model_seed,
            args.nseeds,
            args.nsims,
            args.policy_vaccination,
            args.quota_vaccination,
            args.r0,
            args.rate_infection,
            args.rate_removal,
            args.rate_vaccination,
            args.t_max,
            args.threshold_age,
            args.threshold_opinion,
        ),
        true => load_json_config_to_input_multilayer(FILENAME_CONFIG, Some(FOLDER_CONFIG)).unwrap(),
    };

    let string_multilayer = construct_string_multilayer(args.model_region, args.nagents);

    let path_base = PATH_DATA_MULTILAYER;
    let path_multilayer = PathBuf::from(path_base)
        .join(args.model_region.to_string())
        .join(format!("ml{}_n{}_{}.pickle", args.model_region, args.nagents, args.uuid_multilayer));

    let node_ensemble = load_multilayer_object(&path_multilayer);
    let nagents = node_ensemble.number_of_nodes();

    let mut agent_ensemble = AgentEnsemble::new_from_multilayer(&node_ensemble);

    let population_filename = FILENAME_DATA_POPULATION_AGE;
    let population_vector: Vec<f64> =
        read_key_and_vecf64_from_json(args.model_region, population_filename);

    let eligible_fraction = if args.flag_underage {
        let underaged_fraction = count_underaged(&population_vector);
        1.0 - underaged_fraction
    } else {
        pars_model.threshold_age = 0;
        1.0
    };

    if pars_model.model_opinion == OpinionModel::DataDrivenThresholds {
        let dd_vac_filename = FILENAME_DATA_VACCINATION_ATTITUDE;
        let fractions: Vec<f64> =
            read_key_and_vecf64_from_json(pars_model.model_region, dd_vac_filename);

        pars_model.fraction_vaccinated = eligible_fraction * fractions[0];
        pars_model.fraction_soon = eligible_fraction * fractions[1];
        pars_model.fraction_someone = eligible_fraction * fractions[2];
        pars_model.fraction_majority = eligible_fraction * fractions[3];
        pars_model.fraction_zealot = eligible_fraction * fractions[4];

        pars_model.fraction_active = pars_model.fraction_vaccinated + pars_model.fraction_soon;
        pars_model.threshold_opinion = 0.0;
        pars_model.fraction_zealot = pars_model.fraction_zealot;
    } else {
        pars_model.fraction_active = pars_model.fraction_active;
        pars_model.threshold_opinion = pars_model.threshold_opinion;
        pars_model.fraction_zealot = pars_model.fraction_zealot;
    }

    let r0 = pars_model.r0;
    let rate_removal = pars_model.rate_removal;
    let average_degree = agent_ensemble.average_degree();
    let beta = r0 * rate_removal / average_degree;
    pars_model.rate_infection = beta;

    let pars_vaccination = VaccinationPars::new(
        pars_model.flag_underage,
        pars_model.fraction_majority,
        pars_model.fraction_someone,
        pars_model.fraction_soon,
        pars_model.fraction_vaccinated,
        pars_model.fraction_zealot,
        pars_model.model_hesitancy,
        pars_model.model_region,
        pars_model.policy_vaccination,
        pars_model.quota_vaccination,
        pars_model.rate_vaccination,
        pars_model.threshold_age,
    );

    agent_ensemble.set_vaccination_policy_model(
        pars_vaccination.policy_vaccination,
        pars_vaccination.quota_vaccination,
    );

    let flags_output = OutputFlags::new(
        args.flag_output_age,
        args.flag_output_agent,
        args.flag_output_attitude,
        args.flag_output_cluster,
        args.flag_output_degree,
        args.flag_output_global,
        args.flag_output_time,
    );

    let mut output_ensemble = OutputEnsemble::new();

    println!("Outbreak in {}", args.model_region);

    for sim in 0..pars_model.nsims {
        println!("Dynamical realization={sim}");

        agent_ensemble.introduce_vaccination_attitudes(&pars_vaccination);

        agent_ensemble.introduce_infections_dd(pars_model.model_seed, pars_model.nseeds);

        let output = dynamical_loop(&mut agent_ensemble, &pars_model, &flags_output);

        output_ensemble.add_outbreak(output, nagents, pars_model.r0);

        agent_ensemble.clear_epidemic_consequences();
    }

    let string_epidemic = construct_string_epidemic(&pars_model);

    output_ensemble.to_pickle(
        &pars_model,
        &flags_output,
        &string_epidemic,
        &string_multilayer,
        &args.uuid_multilayer,
    )
}
