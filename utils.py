import numpy as np
import math
import json
from datetime import datetime, timedelta
from pathlib import Path
from dateutil import tz
import requests
import torch

import config
from models import stgcn


def get_symmetric_normalized_adj(A):
    """
    Calculates the symmetrically normalized adjacency matrix.
    Assumes that the adjacency matrix received is undirected,
    adds self loops, and returns D^-1/2 X A X D^-1/2
    -----------------------------
    :params:
        numpy.ndarray (2 dimensions of roads, roads) A: undirected adjacency matrix from road network
    -----------------------------
    :returns:
        numpy.ndarray (2 dimensions of roads, roads) : symmetrically normalised adjacency matrix
    """
    A = A + np.diag(np.ones(A.shape[0]))
    D_inv = np.diag(np.reciprocal(np.sum(A, axis=1) + 1))
    D_inv_sqrt = np.sqrt(D_inv)
    DAD = np.matmul(D_inv_sqrt, np.matmul(A, D_inv_sqrt))
    return DAD


def generate_adjacency_and_metadata(file_path):
    """
    Generates the Adjacency matrix of the road network, together with other metadata
    -----------------------------
    :params:
        str file_path: the file path of one of any raw data files
    -----------------------------
    :returns:
        numpy.ndarray (2 dimensions of roads, roads): Adjacency matrix (undirected)
        dict: Metadata (which index in the adjacency matrix corresponds to which road)
        dict: Road category to integer for use as feature
    """
    with open(file_path, "r") as traffic_data_file:
        traffic_records = json.load(traffic_data_file)

    # Get start, end, length, and find all road categories. Also remove all non-essential metadata features
    roadcategory_list = []
    nodes_params_dict = {}
    for (i, record) in enumerate(traffic_records):
        lat_long_positions = record["Location"].split()
        record["start_pos"] = " ".join(lat_long_positions[0:2])
        record["end_pos"] = " ".join(lat_long_positions[2:4])
        record["length"] = link_length(record["start_pos"], record["end_pos"])
        del record["Location"]
        del record["MaximumSpeed"]
        del record["MinimumSpeed"]
        del record["SpeedBand"]

        if record["RoadCategory"] not in roadcategory_list:
            roadcategory_list.append(record["RoadCategory"])

        nodes_params_dict[i] = record

    traffic_records.sort(key=lambda x: int(x.get("LinkID")))
    roadcategory_list.sort()
    RoadCat2Index = {}
    for i, cat in enumerate(roadcategory_list):
        RoadCat2Index[cat] = i

    # Generating adjacency matrix
    nodes_count = len(nodes_params_dict)
    A_u = np.zeros((nodes_count, nodes_count))
    # Finding the directed edges of the nodes
    for i, i_record in nodes_params_dict.items():
        for j, j_record in nodes_params_dict.items():
            if i_record["end_pos"] == j_record["start_pos"]:
                A_u[i, j] = 1
                A_u[j, i] = 1

    return A_u, nodes_params_dict, RoadCat2Index


def link_length(start_pos, end_pos):
    """
    Calculation of distance between two lat-long geo positions, using Haversine distance
    ------------------------------------
    :params:
        str start_pos: lat & long separated with a space
        str end_pos: lat & long separated with a space
    ------------------------------------
    :returns:
        float: total length of the link
    """
    lat1, lon1 = [float(pos) for pos in start_pos.split()]
    lat2, lon2 = [float(pos) for pos in end_pos.split()]
    radius = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(
        math.radians(lat1)
    ) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    d = radius * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))
    return d


def get_features(file_path, metadata, cat2index):
    """
    Generates a Feature matrix
    Note: Feature Matrix, X, would contain the output speedband as well.
    Positions of Features
        0. SpeedBand
        1. RoadCategory
        2. Length of Link
        3. Day
        4. Hour
    -----------------------------
    :params:
        str file_path: the file path of the dataset
        dict metadata: the road metadata from the previous processing step
        dict cat2index: the mapping from road category to int
    -----------------------------
    :returns:
        numpy.ndarray (2 dimensions of roads, features): Feature matrix
    """

    X = []
    parts = Path(file_path).parts
    timestamp = parts[-1]
    datetime_obj = timestamp_to_datetime(timestamp)
    day_int = datetime_obj.weekday() + 1
    hour = datetime_obj.hour

    with open(file_path, "r") as traffic_data_file:
        traffic_records = json.load(traffic_data_file)

    traffic_records.sort(key=lambda x: int(x.get("LinkID")))
    for i, record in enumerate(traffic_records):
        features = [
            record["SpeedBand"],
            cat2index[record["RoadCategory"]],
            metadata[str(i)]["length"],  # json converts int keys to strings
            day_int,
            hour,
        ]
        X.append(features)

    return np.array(X)


def timestamp_to_datetime(timestamp):
    """
    Converts a timestamp to a datetime object
    -----------------------------
    :params:
        str timestamp: timestamp string
    -----------------------------
    :returns:
        datetime: datetime object corresponding to the timestamp
    """
    return datetime.strptime(timestamp, config.DATE_TIME_FORMAT)


def datetime_to_timestamp(datetime_obj):
    """
    Converts a datetime object to a timestamp string
    -----------------------------
    :params:
        datetime: datetime object corresponding to the timestamp
    -----------------------------
    :returns:
        str timestamp: timestamp string
    """
    return datetime.strftime(datetime_obj, config.DATE_TIME_FORMAT)


def is_consecutive(t1, t2):
    """
    Checks whether two timestamps are within some delta minutes of each other
    -----------------------------
    :params:
        str t1: first timestamp
        str t2: second timestamp
    -----------------------------
    :returns:
        bool: whether the two timestamps are within delta of one another
    """
    t1 = timestamp_to_datetime(t1)
    t2 = timestamp_to_datetime(t2)
    delta = timedelta(minutes=config.MAX_TIME_DELTA)
    return t2 < t1 + delta


def write_json(file_path, json_dict):
    """
    Writes a python dictionary into a json file
    -----------------------------
    :params:
        str file_path: destination path of the json file
        dict json_dict: json data to be saved
    """
    with open(file_path, "w") as f:
        json.dump(json_dict, f, sort_keys=True, indent=4, ensure_ascii=False)


def update_input():
    """
    Updates the model input sample with the latest file in the temp data directory.
    Deletes files in temp directory.
    """
    temp_data_dir = Path(config.TEMP_DIR_PATH)
    files = [file for file in temp_data_dir.iterdir() if file.is_file()]
    files.sort()
    files.reverse()
    if len(files) == 0:
        return
    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)
    with open(config.CAT2INDEX_PATH) as f:
        cat2index = json.load(f)
    current_features = get_features(files[0], metadata, cat2index)
    current_features = np.expand_dims(current_features, 0)

    # starting from clean slate or not
    if Path(config.INPUT_SAMPLE_PATH).is_file():
        with open(config.LATEST_TIMESTEP_PATH) as f:
            prev_timestep = f.read()
        if is_consecutive(prev_timestep, files[0].parts[-1]):
            input_sample = np.load(config.INPUT_SAMPLE_PATH)
            input_sample = np.concatenate((input_sample, current_features))
            input_sample = input_sample[
                -config.NUM_INPUT_TIMESTEPS :
            ]  # remove old values if too long
            np.save(config.INPUT_SAMPLE_PATH, input_sample)
        else:
            np.save(config.INPUT_SAMPLE_PATH, current_features)
    else:
        np.save(config.INPUT_SAMPLE_PATH, current_features)

    with open(config.LATEST_TIMESTEP_PATH, "w") as f:
        f.write(files[0].parts[-1])

    for file in files:
        file.unlink()


def setup():
    """
    Generates adjacency matrix and metadata files from the sample data file
    """
    adj, metadata, cat2index = generate_adjacency_and_metadata(config.SAMPLE_DATA_PATH)
    np.save(config.ADJACENCY_PATH, adj)
    write_json(config.METADATA_PATH, metadata)
    write_json(config.CAT2INDEX_PATH, cat2index)


def cut_json(json_roads):
    """
    Cuts down the input list of roads to only those specified in the area file
    -----------------------------
    :params:
        list json_roads: list of road links from datamall api
    -----------------------------
    :returns:
        list: list of road links corresponding to road ids in area file
    """
    with open(config.AREA_ROADS_PATH) as f:
        selected_ids = [road_id.strip() for road_id in f.readlines()]
    output_json = []
    for road in json_roads:
        if road["LinkID"] in selected_ids:
            output_json.append(road)
    return output_json


def fetch_all():
    """
    Queries the datamall api for the current traffic speed bands,
    attaches the current timestamp to the file, and saves it into
    the raw data directory after truncation to the specified area.
    """
    headers = {"AccountKey": config.API_KEY, "accept": "application/json"}
    results = []

    while True:
        new_results = requests.get(
            config.SPEEDBANDS_URL,
            headers=headers,
            params={"$skip": len(results)},  # 500 records/call
        ).json()["value"]

        if len(new_results) == 0:
            break
        else:
            results += new_results

    current_time = datetime.now(tz=tz.gettz(config.TIMEZONE_STRING))
    timestamp = datetime_to_timestamp(current_time)
    if config.SAVE_RAW:
        write_json(Path(config.RAW_DIR_PATH) / timestamp, results)
    results = cut_json(results)
    write_json(Path(config.TEMP_DIR_PATH) / timestamp, results)


def predict():
    """
    Predicts future speed bands given the prevailing input file.
    First checks whether input file is of the correct length.
    """
    input_sample = np.load(config.INPUT_SAMPLE_PATH)
    if len(input_sample) != config.NUM_INPUT_TIMESTEPS:
        return
    input_sample = (input_sample - config.MEANS) / config.STDS
    model = stgcn.STGCN(
        input_sample.shape[1],
        input_sample.shape[2],
        config.NUM_INPUT_TIMESTEPS,
        config.NUM_OUTPUT_TIMESTEPS,
    )
    model.load_state_dict(torch.load(config.CHECKPOINT_PATH))
    adj = np.load(config.ADJACENCY_PATH)
    adj = get_symmetric_normalized_adj(adj)
    adj = torch.from_numpy(adj).float()
    input_sample = input_sample.reshape(
        (input_sample.shape[1], input_sample.shape[0], input_sample.shape[2])
    )
    input_sample = np.expand_dims(input_sample, 0)
    input_sample = torch.from_numpy(input_sample).float()

    adj = adj.to(config.DEVICE)
    input_sample = input_sample.to(config.DEVICE)
    model.to(config.DEVICE)

    output = {"roads": []}

    with open(config.LATEST_TIMESTEP_PATH) as f:
        output["current_timestamp"] = f.read()

    with torch.no_grad():
        model.eval()
        predicted = model(adj, input_sample)

    predicted = predicted.cpu()
    predicted = predicted * config.STDS[0] + config.MEANS[0]
    predicted = predicted[0]

    with open(config.METADATA_PATH) as f:
        metadata = json.load(f)

    for k, v in metadata.items():
        k = int(k)
        v["predicted_speedbands"] = [float(i) for i in np.array(predicted[k])]
        v["current_speedband"] = round(
            input_sample[0][k][-1][0].item() * config.STDS[0] + config.MEANS[0]
        )
        output["roads"].append(v)

    write_json(config.OUTPUT_PATH, output)
