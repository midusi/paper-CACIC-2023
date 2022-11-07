from typing import Tuple, Union
from pyspark import TaskContext, Broadcast
from utils import get_columns_from_df
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.svm import FastKernelSurvivalSVM
from core import run_times_experiment, run_times_experiment_sequential
import logging
import time
from pyspark import SparkConf, SparkContext
from typing import Optional
import socket
from datetime import datetime
from multiprocessing import Process, Queue
from filelock import FileLock

# Enables logging
logging.getLogger().setLevel(logging.INFO)

# Number of cores used by the worker to compute the Cross Validation. -1 = use all
N_JOBS = -1

# To get the training score or not
RETURN_TRAIN_SCORE = True

# To replicate randomness
RANDOM_STATE: Optional[int] = None

# If True runs in Spark, otherwise in sequential
RUN_IN_SPARK: bool = True

# Number of iterations
N_ITERATIONS = 30

# If True Random Forest is used as classificator. SVM otherwise
USE_RF: bool = False

# To use a Broadcast value instead of a pd.DataFrame
USE_BROADCAST = True

# Only if RUN_IN_SPARK is set to True the following parameters are used
# Executors per instance of each worker
EXECUTORS: Optional[str] = "1"

# Cores on each executor
# CORES_PER_EXECUTOR: Optional[str] = "2"
CORES_PER_EXECUTOR: Optional[str] = None

# RAM to use per executor
MEMORY_PER_EXECUTOR: str = "6g"

# If True, the workers log Garbage Collector actions. Useful to debug performance problems during mapPartitions()
LOG_GC = False

# Classificator
if USE_RF:
    CLASSIFIER = RandomSurvivalForest(n_estimators=100,
                                      min_samples_split=10,
                                      min_samples_leaf=15,
                                      max_features="sqrt",
                                      n_jobs=-1,
                                      random_state=RANDOM_STATE)
else:
    CLASSIFIER = FastKernelSurvivalSVM(rank_ratio=0.0, max_iter=1000, tol=1e-5, random_state=RANDOM_STATE)


def compute_cross_validation_spark_f(subset: pd.DataFrame, y: np.ndarray, q: Queue):
    """
    Computes a cross validations to get the concordance index in a Spark environment
    :param subset: Subset of features to compute the cross validation
    :param y: Y data
    :param q: Queue to return Process result
    """
    # Locks to prevent multiple partitions in one worker getting all cores and degrading the performance
    with FileLock(f"/home/big_data/svm-surv.lock"):
        start = time.time()
        res = cross_validate(
            CLASSIFIER,
            subset,
            y,
            cv=10,
            n_jobs=N_JOBS,
            return_estimator=True,
            return_train_score=RETURN_TRAIN_SCORE
        )
        concordance_index_mean = res['test_score'].mean()
        end_time = time.time()

        worker_time = end_time - start
        logging.info(f'Cross validation with {subset.shape[1]} features -> {worker_time} seconds | '
                     f'Concordance Index -> {concordance_index_mean}')

        partition_id = TaskContext().partitionId()

        # Gets a time-lapse description to check if some worker is lazy
        start_desc = datetime.fromtimestamp(start).strftime("%H:%M:%S")
        end_desc = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
        time_description = f'{start_desc} - {end_desc}'

        # Gets number of iterations
        times_by_iteration = []
        total_number_of_iterations = []
        for estimator, fit_time in zip(res['estimator'], res['fit_time']):
            # Scikit-surv doesn't use BaseLibSVM. So it doesn't have 'n_iter_' attribute
            # number_of_iterations += np.sum(estimator.n_iter_)
            number_of_iterations = estimator.optimizer_result_.nit
            time_by_iterations = fit_time / number_of_iterations
            times_by_iteration.append(time_by_iterations)
            total_number_of_iterations.append(number_of_iterations)

        train_score = res['train_score'].mean() if RETURN_TRAIN_SCORE else 0.0

        q.put([
            concordance_index_mean,
            worker_time,
            partition_id,
            socket.gethostname(),
            subset.shape[1],
            time_description,
            np.mean(times_by_iteration),
            np.mean(res['score_time']),
            np.mean(total_number_of_iterations),
            train_score
        ])


def compute_cross_validation_spark(
        subset: Union[pd.DataFrame, Broadcast],
        index_array: np.ndarray,
        y: np.ndarray,
        is_broadcast: bool
) -> Tuple[float, float, int, str, int, str, float, float, float, float]:
    """
    Calls fitness inside a Process to prevent issues with memory leaks in Python.
    More info: https://stackoverflow.com/a/71700592/7058363
    :param is_broadcast:
    :param index_array:
    :param subset: Subset of features to compute the cross validation
    :param y: Y data
    :return: Result tuple with [0] -> fitness value, [1] -> execution time, [2] -> Partition ID, [3] -> Hostname,[4] ->\
    number of evaluated features, [5] -> time lapse description, [6] -> time by iteration and [7] -> avg test time
    """
    x_values = subset.value if is_broadcast else subset

    q = Queue()
    parsed_data = get_columns_from_df(index_array, x_values)
    p = Process(target=compute_cross_validation_spark_f, args=(parsed_data, y, q))
    p.start()
    process_result = q.get()
    p.join()
    return process_result


def compute_cross_validation_spark_original(subset: pd.DataFrame, y: np.ndarray):
    """
    Computes a cross validations to get the concordance index in a Spark environment
    :param subset: Subset of features to compute the cross validation
    :param y: Y data
    """
    start = time.time()
    res = cross_val_score(
        CLASSIFIER,
        subset,
        y,
        cv=10,
        n_jobs=N_JOBS
    )
    concordance_index_mean = res.mean()

    end_time = time.time()
    worker_time = end_time - start
    logging.info(f'Cross validation with {subset.shape[1]} features -> {worker_time} seconds | '
                 f'Concordance Index -> {concordance_index_mean}')

    partition_id = TaskContext().partitionId()

    # Gets a time-lapse description to check if some worker is lazy
    start_desc = datetime.fromtimestamp(start).strftime("%H:%M:%S")
    end_desc = datetime.fromtimestamp(end_time).strftime("%H:%M:%S")
    time_description = f'{start_desc} - {end_desc}'

    return concordance_index_mean, worker_time, partition_id, socket.gethostname(), subset.shape[1], time_description


def compute_cross_validation(subset: pd.DataFrame, y: np.ndarray) -> float:
    """
    Computes a cross validations to get the concordance index in a single node (sequentially)
    :param subset: Subset de features a utilizar en el RandomForest evaluado en el CrossValidation
    :param y: Clases
    :return: Promedio del accuracy obtenido en cada fold del CrossValidation
    """
    start = time.time()
    res = cross_val_score(
        CLASSIFIER,
        subset,
        y,
        cv=10,
        n_jobs=N_JOBS
    )
    end_time = time.time() - start
    concordance_index_mean = res.mean()
    logging.info(f'Cross validation with {subset.shape[1]} features -> {end_time} seconds | '
                 f'Concordance Index -> {concordance_index_mean}')

    return concordance_index_mean


def main():
    model_description = 'RF' if USE_RF else 'SVM'

    if RUN_IN_SPARK:
        # Spark settings
        conf = SparkConf().setMaster("spark://master-node:7077").setAppName(f"BBHA {time.time()}")

        if EXECUTORS is not None:
            conf = conf.set("spark.executor.instances", EXECUTORS)

        if CORES_PER_EXECUTOR is not None:
            conf = conf.set("spark.executor.cores", CORES_PER_EXECUTOR)

        if MEMORY_PER_EXECUTOR is not None:
            conf = conf.set("spark.executor.memory", MEMORY_PER_EXECUTOR)

        if LOG_GC:
            conf = conf.set("spark.executor.extraJavaOptions", '-verbose:gc -XX:+PrintGCDetails -XX:+PrintGCDateStamps -XX:+PrintGCTimeStamps')

        sc = SparkContext(conf=conf)
        sc.setLogLevel("ERROR")

        run_times_experiment(
            compute_cross_validation=compute_cross_validation_spark,
            n_iterations=N_ITERATIONS,
            sc=sc,
            metric_description='concordance index',
            add_epsilon=not USE_RF,  # Epsilon is needed by the SVM, not the RF
            model_description=f'{model_description} Survival',
            use_broadcasts_in_spark=USE_BROADCAST
        )
    else:
        # Runs sequentially
        run_times_experiment_sequential(
            compute_cross_validation=compute_cross_validation,
            n_iterations=N_ITERATIONS,
            metric_description='concordance index',
            add_epsilon=not USE_RF,  # Epsilon is needed by the SVM, not the RF
            model_description=f'{model_description} Survival',
        )


if __name__ == '__main__':
    main()
