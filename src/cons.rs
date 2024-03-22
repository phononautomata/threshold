use crate::agent::{Status, Attitude};

pub const CONST_ALREADY_THRESHOLD: f64 = 0.0;
pub const CONST_ELDER_THRESHOLD: usize = 65;
pub const CONST_EPIDEMIC_THRESHOLD: f64 = 1.0;
pub const CONST_SOON_THRESHOLD: f64 = 0.0;
pub const CONST_MAJORITY_THRESHOLD: f64 = 0.5;
pub const CONST_MIDDLEAGE_THRESHOLD: usize = 45;
pub const CONST_UNDERAGE_THRESHOLD: usize = 18;
pub const CONST_YOUNG_ADULT_UPPER_THRESHOLD: usize = 30;
pub const CONST_ZEALOT_THRESHOLD: f64 = 1.00000001;

pub const EXTENSION_CONFIG: &str = ".json";
pub const EXTENSION_DATA: &str = ".json";
pub const EXTENSION_RESULTS: &str = ".json";

pub const FLAG_VERBOSE: bool = false;

pub const FILENAME_CONFIG: &str = "config";
pub const FILENAME_DATA_AVERAGE_CONTACT: &str = "average_contacts_data";
pub const FILENAME_DATA_CONTACT_AGE: &str = "contacts_age_data";
pub const FILENAME_DATA_CONTACT_MATRIX: &str = "contact_matrix_data";
pub const FILENAME_DATA_DEGREE: &str = "norm_degree_data";
pub const FILENAME_DATA_POPULATION: &str = "population_data";
pub const FILENAME_DATA_POPULATION_AGE: &str = "norm_population_age_data";
pub const FILENAME_DATA_VACCINATION_ATTITUDE: &str = "vaccination_attitude_data";

pub const FOLDER_CONFIG: &str = "config";
pub const FOLDER_DATA: &str = "data";
pub const FOLDER_DATA_CUR: &str = "data/curated";
pub const FOLDER_DATA_RAW: &str = "data/raw";
pub const FOLDER_RESULTS: &str = "results";
pub const FOLDER_RESULTS_TEMP: &str = "results/temp";

pub const HEADER_AGE: &str = "age_";
pub const HEADER_AGENT: &str = "agent_";
pub const HEADER_AGENT_DISTRIBUTION: &str = "ad_";
pub const HEADER_AGENT_STATS: &str = "asp_";
pub const HEADER_ATTITUDE: &str = "att_";
pub const HEADER_CLUSTER: &str = "cluster_";
pub const HEADER_CLUSTER_DISTRIBUTION: &str = "cd_";
pub const HEADER_CLUSTER_STATS: &str = "csp_";
pub const HEADER_DEGREE: &str = "degree_";
pub const HEADER_GLOBAL: &str = "global_";
pub const HEADER_PROJECT: &str = "thr_";
pub const HEADER_REBUILD: &str = "rebuild_";
pub const HEADER_REBUILD_STATS: &str = "rebstat_";
pub const HEADER_TIME: &str = "time_";
pub const HEADER_TIME_STATS: &str = "ts_";

pub const INIT_ATTITUDE: Attitude = Attitude::Never;
pub const INIT_STATUS: Status = Status::HesSus;
pub const INIT_USIZE: usize = std::usize::MAX;

pub const PAR_AGE_GROUPS: usize = 85;
pub const PAR_ATTITUDE_GROUPS: usize = 5;
pub const PAR_EPIDEMIC_DIEOUT: usize = 0;
pub const PAR_NBINS: usize = 30;
pub const PAR_NETWORK_TRIALS: usize = 100;
pub const PAR_OUTBREAK_PREVALENCE_FRACTION_CUTOFF: f64 = 0.0;
pub const PAR_TIME_STEP: usize = 1;