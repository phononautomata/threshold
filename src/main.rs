use clap::Parser;
use threshold::exps::{
    Args,
    run_exp1_homogeneous, 
    run_exp2_datadriven_thresholds,
    run_exp3_multilayer_thresholds,
};

fn main() {
    let input_args = Args::parse();

    match input_args.experiment_id {
        1 => run_exp1_homogeneous(input_args),
        2 => run_exp2_datadriven_thresholds(input_args),
        3 => run_exp3_multilayer_thresholds(input_args),
        _ => panic!(),
    };
}
