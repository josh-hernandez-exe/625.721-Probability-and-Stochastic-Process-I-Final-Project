import collections
import functools
import multiprocessing

import numpy as np
import pandas as pd

def distance(lat1, lon1, lat2, lon2, R=6371e+3, deg_to_rag = np.pi/180):
    '''
    https://www.movable-type.co.uk/scripts/latlong.html

    https://numpy.org/doc/stable/reference/generated/numpy.arctan2.html
    https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Math/atan2

    const R = 6371e3; // metres
    const φ1 = lat1 * Math.PI/180; // φ, λ in radians
    const φ2 = lat2 * Math.PI/180;
    const Δφ = (lat2-lat1) * Math.PI/180;
    const Δλ = (lon2-lon1) * Math.PI/180;

    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ/2) * Math.sin(Δλ/2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));

    const d = R * c; // in metres
    '''

    phi1 = lat1 * deg_to_rag
    phi2 = lat2 * deg_to_rag
    delta_phi = (lat2-lat1) * deg_to_rag
    delta_lambda = (lon2-lon1) * deg_to_rag

    a_val = (
        +np.sin(delta_phi/2) **2
        +np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2) **2
    )
    c_val = 2 * np.arctan2(
        np.sqrt(a_val),
        np.sqrt(1-a_val),
    )
    d_val = R * c_val

    return d_val


def _get_station_data_helper(path, station_id_set=None):
    df = None
    with path.open() as flink:
        df = pd.read_csv(flink, encoding= 'unicode_escape')

    if df is None:
        return

    elif isinstance(station_id_set, set):
        station_mask = (
            df['Climate ID']
            .astype(str)
            .apply(lambda item: item in station_id_set)
        )

        df.drop(df[~station_mask].index, axis='index', inplace=True)

    elif station_id_set is not None:
        raise Exception('station_id_set is not given as a set')

    df['Date/Time (LST)'] = pd.to_datetime(df['Date/Time (LST)'])

    return df

def get_station_data(
    raw_data_dir,
    station_id_set = None,
    columns_interested = [
        'Longitude (x)', 'Latitude (y)', 'Station Name', 'Climate ID',
        'Date/Time (LST)', 'Year', 'Month', 'Day', 'Time (LST)',
        'Wind Dir (10s deg)', 'Wind Spd (km/h)',
        # 'Wind Dir Flag', 'Wind Spd Flag',
    ],
):
    if isinstance(station_id_set, (set, list, tuple)):
        station_id_set = set([
            str(item)
            for item in station_id_set
        ])

    dfs = []

    with multiprocessing.Pool(processes=None) as pool:
        for df in pool.imap(
            func=functools.partial(
                _get_station_data_helper,
                station_id_set=station_id_set,
            ),
            iterable=raw_data_dir.iterdir(),
            chunksize=64,
        ):
            if df is None:
                continue

            dfs.append(df)

    full_df = pd.concat(
        [
            df[columns_interested]
            for df in dfs
        ],
        axis='index',
        ignore_index=True,
    )

    if len(full_df) == 0:
        raise Exception('No data')

    full_df.sort_values(
        by=['Date/Time (LST)', 'Climate ID'],
        inplace=True,
        ignore_index=True,
    )

    return full_df

def calculate_n_ijh(state_ts, max_h=3):
    n_ijh = collections.defaultdict(collections.Counter)

    for hh in range(1,max_h+1):
        for ii,jj in zip(state_ts, state_ts[hh:]):
            if pd.isna(ii) or pd.isna(jj):
                continue

            n_ijh[hh][ii,jj] += 1

    return n_ijh

def calculate_p_ijh(state_ts, n_ijh, min_state=0, max_state=25):
    max_h = len(n_ijh)
    h_list = list(range(1,max_h+1))
    p_ijh = [
        np.matrix(
            [
                [
                    n_ijh[hh][ii,jj]
                    for jj in range(min_state, max_state+1)
                ]
                for ii in range(min_state, max_state+1)
            ],
            dtype=float,
        )
        for hh in h_list
    ]

    for hh, p_ij in zip(h_list, p_ijh):
        rows, cols = p_ij.shape
        # Note that pre normalization
        # p_ijh[hh][ii,jj] == n_ijh[hh][ii,jj]
        for ii in range(rows):
            total_count = np.sum(p_ij[ii,:])

            if total_count == 0:
                continue

            p_ij[ii,:] /= total_count

    return p_ijh



def round_away_from_zero(x):
    a = np.abs(x)
    r1 = np.floor(a) + np.floor(2 * (a % 1))
    r2 = np.copy(r1)
    # r if x >= 0 else -r
    r2[x < 0] = -r1[x < 0]
    return r2


def get_train_test_time_ranges():

    timestamp_checkpoints = [
        pd.Timestamp(year=2016,month=1, day=1),
        pd.Timestamp(year=2018,month=1, day=1),
        pd.Timestamp(year=2020,month=1, day=1),
        pd.Timestamp(year=2022,month=1, day=1),
    ]

    train_range_1 = (timestamp_checkpoints[0], timestamp_checkpoints[1])
    test_range_1 = (timestamp_checkpoints[1], timestamp_checkpoints[-1])

    train_range_2 = (timestamp_checkpoints[0], timestamp_checkpoints[2])
    test_range_2 = (timestamp_checkpoints[2], timestamp_checkpoints[-1])

    train_test_time_ranges = [
        (train_range_1, test_range_1),
        (train_range_2, test_range_2),
    ]

    return train_test_time_ranges


def evaluate_markov_chain(truth_ts, transition_matrix_vec, vec_size, max_iter=100):
    num_elements = transition_matrix_vec.size
    side = int(np.sqrt(num_elements))
    transition_matrix = transition_matrix_vec.reshape(side,side)
    initial_vec = np.zeros(vec_size)
    start_state = truth_ts.iloc[0]
    initial_vec[start_state] = 1
    initial_vec = np.matrix(initial_vec)

    vec = initial_vec
    test_states = np.array([
        np.argmax(vec := vec @ transition_matrix)
        for _ in range(max_iter)
    ])

    return test_states
