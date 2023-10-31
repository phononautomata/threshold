use clap::Parser;
use threshold::exps::{
    ArgsLine,
    //ArgsNetwork, 
    //ArgsOpinion, 
    //ArgsEpidemic,
    //ArgsVaccination, 
    //ArgsDataDrivenUS, 
    //ArgsExperiment, 
    run_exp1_homogeneous, 
    run_exp2_datadriven
};

fn main() {
    let input_args = ArgsLine::parse();

    match input_args.exp_flag {
        1 => run_exp1_homogeneous(
            input_args,
            //ArgsNetwork::parse(), 
            //ArgsOpinion::parse(), 
            //ArgsEpidemic::parse(),
            //ArgsVaccination::parse(), 
        ),
        2 => run_exp2_datadriven(
            input_args,
            //ArgsNetwork::parse(), 
            //ArgsEpidemic::parse(),
            //ArgsDataDrivenUS::parse(),
        ),
        _ => panic!(),
    };
}
