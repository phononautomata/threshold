use clap::Parser;
use threshold::exps::{
    Args,
    build_multilayer, 
    run_epidemic,
    run_full,
};

fn main() {
    let input_args = Args::parse();

    match input_args.experiment_id {
        1 => build_multilayer(input_args),
        2 => run_epidemic(input_args),
        3 => run_full(input_args),
        _ => panic!(),
    };
}
