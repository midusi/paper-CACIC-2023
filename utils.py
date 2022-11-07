import logging
import os
from typing import Tuple, cast, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize

# Name of the class column in the DataFrame
NEW_CLASS_NAME = 'class'

# To prevent some errors with SVM
# EPSILON = 1.E-03
EPSILON = 1

# Structure of times dict for reporting execution and idle times. Worker name -> Partition -> Execution/Idle time
WorkerTimeDict = Dict[str, Dict[int, List[float]]]


def specificity(y_true, y_pred):
    """
    Gets the specificity metric
    Taken from https://stackoverflow.com/questions/33275461/specificity-in-scikit-learn
    :param y_true: Y true
    :param y_pred: Y pred
    :return: Specificity value
    """
    conf_res = confusion_matrix(y_true, y_pred).ravel()
    tn, fp = conf_res[0], conf_res[1]
    return tn / (tn + fp)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes NaN and Inf values
    :param df: DataFrame to clean
    :return: Cleaned DataFrame
    """
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True, axis='columns')
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any('columns')
    return df[indices_to_keep].astype(np.float64)


def read_survival_data(add_epsilon: bool) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Reads and preprocess survival dataset
    :param add_epsilon: If True it adds an epsilon to 0s in Y data to prevent errors in SVM training
    :return: Tuple with the filtered DataFrame, Y data
    """
    # Gets X
    x_file_path = os.path.join(os.path.dirname(__file__), 'Datasets/data_RNA_Seq_v2_mRNA_median_Zscores.txt')
    x = pd.read_csv(x_file_path, sep='\t', index_col=0)

    # Removes '-1' suffix to make the join
    x.columns = x.columns.str.replace("-01$", "", regex=True)
    patients_x = x.columns.values

    # Gets Y
    y_file_path = os.path.join(os.path.dirname(__file__), 'Datasets/data_clinical_patient.txt')
    y = pd.read_csv(y_file_path, sep='\t', skiprows=4, index_col=0)
    patients_y = y.index.values
    y = y.loc[:, ['OS_STATUS', 'OS_MONTHS']]  # Keeps only survival columns
    cond_living = y['OS_STATUS'] == '0:LIVING'
    y.loc[cond_living, 'OS_STATUS'] = False
    y.loc[~cond_living, 'OS_STATUS'] = True

    # Gets in common patients
    patients_intersect = np.intersect1d(patients_x, patients_y)
    y = y.loc[patients_intersect, :]

    # Removes zeros
    if add_epsilon:
        zeros_cond = y['OS_MONTHS'] == 0
        y.loc[zeros_cond, 'OS_MONTHS'] = y.loc[zeros_cond, 'OS_MONTHS'] + 1
        assert y[y['OS_MONTHS'] == 0].empty

    # Removes unneeded column and tranpose to keep samples as columns
    x.drop('Entrez_Gene_Id', axis=1, inplace=True)
    x = x.transpose()
    x = x.loc[patients_intersect, :]

    # Removes NaN and Inf values
    x = clean_dataset(x)

    # Formats Y to a structured array
    y = np.core.records.fromarrays(y.to_numpy().transpose(), names='event, time', formats='bool, float')

    return x, y


def rename_class_column_name(df: pd.DataFrame, class_name_old: str):
    """
    Renames the DataFrame class column to generalize the algorithms
    :param df: DataFrame
    :param class_name_old: Current class column name
    """
    df.rename(columns={class_name_old: NEW_CLASS_NAME}, inplace=True)


def binarize_y(y: pd.Series) -> Tuple[np.ndarray, int]:
    """
    Generates a binary array indicating the class
    :param y: Class array
    :return: Binary array
    """
    classes = y.unique()
    return label_binarize(y, classes=classes).ravel(), classes.shape[0]


def get_columns_by_categorical(columns_index: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the column from a categorical array
    :param columns_index: Numpy Array with a {0, 1} in the column index to indicate absence/presence of the column
    :param df: DataFrame to retrieve the columns data
    :return: DataFrame with only the specified columns
    """
    non_zero_idx = np.nonzero(columns_index)
    return df.iloc[:, non_zero_idx[0]]


def get_columns_from_df(columns_list: np.array, df: pd.DataFrame) -> pd.DataFrame:
    """Returns a set of columns of a DataFrame. The usefulness of this method is that it works
    for categorical indexes or strings"""
    if np.issubdtype(columns_list.dtype, np.number):
        # Gets by integer indexes
        return get_columns_by_categorical(columns_list, df)
    # Gets by column name
    return df[columns_list]


def report_exec_and_idle_times(workers_benchmarks: WorkerTimeDict, workers_idle: WorkerTimeDict):
    """
    Reports execution and idle times by worker
    :param workers_benchmarks: Execution times dict
    :param workers_idle: Idle times dict
    """
    for (worker_host_name, execution_time_partitions), (_, idle_time_partitions) \
            in zip(workers_benchmarks.items(), workers_idle.items()):
        worker_execution_means = []
        worker_idle_means = []
        for (partition_id, exec_times), (_, idle_times) \
                in zip(execution_time_partitions.items(), idle_time_partitions.items()):
            mean_exec_times = round(cast(float, np.mean(exec_times)), 3)
            worker_execution_means.append(mean_exec_times)

            mean_idle_times = round(cast(float, np.mean(idle_times)), 3)
            worker_idle_means.append(mean_idle_times)

        # Execution time
        worker_exec_mean = round(cast(float, np.mean(worker_execution_means)), 3)
        worker_exec_std = round(cast(float, np.std(worker_execution_means)), 3)
        logging.info(f'Worker "{worker_host_name}" execution time mean -> {worker_exec_mean} (+- {worker_exec_std})')

        # Idle time
        worker_idle_mean = round(cast(float, np.mean(worker_idle_means)), 3)
        worker_idle_std = round(cast(float, np.std(worker_idle_means)), 3)
        logging.info(f'Worker "{worker_host_name}" idle time mean -> {worker_idle_mean} (+- {worker_idle_std})')


def store_times(star_idx: int, current_data: Tuple[float, float, int, str, int, str, float, float, float, float],
                total_iteration_time: float, workers_benchmarks: WorkerTimeDict,
                workers_idle: WorkerTimeDict, debug: bool):
    """
    Stores execution and idle times in respective dicts
    :param star_idx: Star index to retrieve data
    :param current_data: Data tuples where the current stars' fitness results are
    :param total_iteration_time: Total iteration time to compute idle time
    :param workers_benchmarks: Execution times dict
    :param workers_idle: Idle times dict
    :param debug: If True, prints some debugging info
    """
    fitness_value = current_data[0]
    worker_time = current_data[1]
    partition_id = current_data[2]
    host_name = current_data[3]
    number_features_worker = current_data[4]
    time_lapse_description = current_data[5]
    if host_name not in workers_benchmarks:
        workers_benchmarks[host_name] = {}
        workers_idle[host_name] = {}
    if partition_id not in workers_benchmarks[host_name]:
        workers_benchmarks[host_name][partition_id] = []
        workers_idle[host_name][partition_id] = []
    workers_benchmarks[host_name][partition_id].append(worker_time)
    workers_idle[host_name][partition_id].append(total_iteration_time - worker_time)
    if debug:
        logging.info(f'{star_idx} star took {round(worker_time, 3)} seconds ({time_lapse_description}) '
                     f'for {number_features_worker} features. Partition: {partition_id} | '
                     f'Host name: {host_name}. Fitness: {fitness_value}')
