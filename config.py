import torch

# model
NUM_INPUT_TIMESTEPS = 12
NUM_OUTPUT_TIMESTEPS = 6
CHECKPOINT_PATH = (
    "./models/checkpoints/downtown12in6out32168421_undirected_01234.pt"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEANS = [4.43166567, 2.55106077, 0.08281905, 4.0460931, 11.53766421]
STDS = [1.6744557, 1.55828848, 0.04885751, 1.98377927, 6.93358581]
OUTPUT_PATH = "./data/processed/output.json"

# data collection and processing
DATE_TIME_FORMAT = "%d_%m_%Y_%H_%M_%S"
MAX_TIME_DELTA = 9  # in minutes
TIMEZONE_STRING = "GMT+8"
SAVE_RAW = False
INPUT_SAMPLE_PATH = "./data/processed/input.npy"
LATEST_TIMESTEP_PATH = "./data/processed/latest.txt"
ADJACENCY_PATH = "./data/processed/adj.npy"
METADATA_PATH = "./data/processed/metadata.json"
CAT2INDEX_PATH = "./data/processed/cat2index.json"
RAW_DIR_PATH = "./data/raw"
TEMP_DIR_PATH = "./data/temp"
SAMPLE_DATA_PATH = "./data/sample"
AREA_ROADS_PATH = "./data/downtown.csv"
QUERY_INTERVAL = 5 # in minutes

# datamall
API_KEY = ""
SPEEDBANDS_URL = "http://datamall2.mytransport.sg/ltaodataservice/TrafficSpeedBandsv2"
