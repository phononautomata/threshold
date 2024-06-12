use clap::Parser;
use threshold::exps::{generate_multilayer, run_epidemic, Args};

fn main() {
    let input_args = Args::parse();

    match input_args.id_experiment {
        1 => generate_multilayer(input_args),
        2 => run_epidemic(input_args),
        _ => panic!(),
    };
}
