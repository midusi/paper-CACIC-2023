import logging
from typing import Optional, Callable, Tuple, Union, List, Iterable
import random
import numpy as np
from math import tanh
from pyspark import SparkContext
import time
from utils import report_exec_and_idle_times, WorkerTimeDict, store_times

logging.getLogger().setLevel(logging.INFO)

NUMBER_OF_WORKERS = 3

# Result tuple with [0] -> fitness value, [1] -> execution time, [2] -> Partition ID, [3] -> Hostname,
# [4] -> number of evaluated features, [5] -> time lapse description, [6] -> time by iteration and [7] -> avg test time
# [8] -> mean of number of iterations of the model inside the CV, [9] -> train score
CrossValidationSparkResult = Tuple[float, float, int, str, int, str, float, float, float, float]


def get_best_spark(
        subsets: np.ndarray,
        workers_results: Union[np.ndarray, List[CrossValidationSparkResult]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtiene el mayor valor de un conjunto de fitness"""
    workers_results_np = np.array(workers_results)
    best_idx = np.argmax(workers_results_np[:, 0])  # Mantengo el idx para evitar comparaciones ambiguas
    return best_idx, subsets[best_idx][1], workers_results[best_idx]


def parallelize_fitness_execution(
        sc: SparkContext,
        stars_subsets: np.ndarray,
        fitness_function: Callable[[np.ndarray], CrossValidationSparkResult]
) -> List[CrossValidationSparkResult]:
    """
    Parallelize the fitness function computing on an Apache Spark cluster and return fitness metrics
    @param sc: Spark context
    @param stars_subsets: Stars subset to parallelize
    @param fitness_function: Fitness function
    @return: All the fitness metrics from all the star, and a execution time by worker
    """
    stars_parallelized = sc.parallelize(stars_subsets)

    return stars_parallelized \
        .map(lambda star_features: fitness_function(star_features[1])) \
        .collect()


def map_partition(
        fitness_function: Callable[[np.ndarray], CrossValidationSparkResult],
        records: Iterable[CrossValidationSparkResult]
) -> Iterable[CrossValidationSparkResult]:
    """Returns fitness result for all the elements in partition records"""
    for key, elem in records:
        yield fitness_function(elem)


def parallelize_fitness_execution_by_partitions(
        sc: SparkContext,
        stars_subsets: np.ndarray,
        fitness_function: Callable[[np.ndarray], CrossValidationSparkResult]
) -> Tuple[List[CrossValidationSparkResult], float]:
    """
    Parallelize the fitness function computing on an Apache Spark cluster and return fitness metrics.
    This function generates a partitioning that distributes the load equally to all nodes
    @param sc: Spark context
    @param stars_subsets: Stars subset to parallelize
    @param fitness_function: Fitness function
    @return: All the fitness metrics from all the star, and a execution time by worker
    """
    stars_parallelized = sc.parallelize(stars_subsets)

    start_parallelization = time.time()

    # NOTE: the mapPartitions() allows Scikit-surv models to use all the worker's cores during CrossValidation.
    # This avoids problems of Spark parallelization interfering with Scikit-surv parallelized algorithms
    # .partitionBy(NUMBER_OF_WORKERS, partitionFunc=lambda key: key * NUMBER_OF_WORKERS // len(stars_subsets))  # TODO: check if below solution fix problems
    result = stars_parallelized \
        .partitionBy(None, partitionFunc=lambda key: key) \
        .mapPartitions(lambda records: map_partition(fitness_function, records), preservesPartitioning=True) \
        .collect()
    total_iteration_time = time.time() - start_parallelization
    logging.info(f'partitionBy(), mapPartitions() and collect() time -> {total_iteration_time} seconds')

    return result, total_iteration_time


def get_random_subset_of_features(n_features: int) -> np.ndarray:
    """
    Generates a random subset of Features. Answer taken from https://stackoverflow.com/a/47942584/7058363
    :param n_features: Total number of features
    :return: Categorical array with {0, 1} values indicate the absence/presence of the feature in the index
    """
    res = np.zeros(n_features, dtype=int)  # Gets an array of all the features in zero

    # Generates a random number of features in 1. Then, shuffles and returns as boolean
    random_number_of_features = random.randint(1, n_features)
    res[:random_number_of_features] = 1
    np.random.shuffle(res)
    return res


def get_best(
        subsets: np.ndarray,
        fitness_values: Union[np.ndarray, List[float]]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Obtiene el mayor valor de un conjunto de fitness"""
    best_idx = np.argmax(fitness_values)  # Mantengo el idx para evitar comparaciones ambiguas
    return best_idx, subsets[best_idx], fitness_values[best_idx]


def binary_black_hole(
        n_stars: int,
        n_features: int,
        n_iterations: int,
        fitness_function: Callable[[np.ndarray], CrossValidationSparkResult],
        binary_threshold: Optional[float] = 0.6,
        debug: bool = False
):
    """
    Computa la metaheuristica Binary Black Hole sacada del paper
    "Binary black hole algorithm for feature selection and classification on biological data"
    Authors: Elnaz Pashaei, Nizamettin Aydin.
    NOTA: esta hecho de manera lenta para que se pueda entender el algoritmo, usar la version vectorizada para
    produccion
    :param n_stars: Number of stars
    :param n_features: Number of features
    :param n_iterations: Number of iterations
    :param fitness_function: Fitness function to compute on every star
    :param binary_threshold: Binary threshold to set 1 or 0 the feature. If None it'll be computed randomly
    :param debug: If True logs everything is happening inside BBHA
    :return:
    """
    # Preparo las estructuras de datos
    stars_subsets = np.empty((n_stars, n_features), dtype=int)
    stars_fitness_values = np.empty((n_stars,), dtype=float)

    # Inicializo las estrellas con sus subconjuntos y sus valores fitness
    if debug:
        logging.info('Initializing stars...')
    for i in range(n_stars):
        random_features_to_select = get_random_subset_of_features(n_features)
        stars_subsets[i] = random_features_to_select  # Initialize 'Population'
        stars_fitness_values[i] = fitness_function(random_features_to_select)

    # El que mejor fitness tiene es el Black Hole
    black_hole_idx, black_hole_subset, black_hole_fitness = get_best(stars_subsets, stars_fitness_values)
    if debug:
        logging.info(f'Black hole starting at star with index {black_hole_idx}')

    # Iteraciones
    for i in range(n_iterations):
        if debug:
            logging.info(f'Iteration -> {i + 1} of {n_iterations}')
        for a in range(n_stars):
            # Si es la misma estrella que se convirtio en agujero negro, no hago nada
            if a == black_hole_idx:
                continue

            # Compute the current star fitness
            current_star_subset = stars_subsets[a]
            current_fitness = fitness_function(current_star_subset)

            # Si la estrella tiene mejor fitness que el agujero negro, hacemos swap
            if current_fitness > black_hole_fitness:
                if debug:
                    logging.info(f'Changing Black hole for star {a},'
                                 f' BH fitness -> {black_hole_fitness} | Star {a} fitness -> {current_fitness}')
                black_hole_idx = a
                black_hole_subset, current_star_subset = current_star_subset, black_hole_subset
                black_hole_fitness, current_fitness = current_fitness, black_hole_fitness

            # Si la funcion de fitness es igual, pero tengo menos features en la estrella (mejor!), hacemos swap
            elif current_fitness == black_hole_fitness and np.count_nonzero(current_star_subset) < np.count_nonzero(
                    black_hole_subset):
                if debug:
                    logging.info(f'Changing Black hole for star {a},'
                                 f' BH fitness -> {black_hole_fitness} | Star {a} fitness -> {current_fitness}')
                black_hole_idx = a
                black_hole_subset, current_star_subset = current_star_subset, black_hole_subset
                black_hole_fitness, current_fitness = current_fitness, black_hole_fitness

            # Calculo el horizonte de eventos
            event_horizon = black_hole_fitness / np.sum(stars_fitness_values)

            # Me fijo si la estrella cae en el horizonte de eventos
            dist_to_black_hole = np.linalg.norm(black_hole_subset - current_star_subset)  # Dist. Euclidea
            if dist_to_black_hole < event_horizon:
                if debug:
                    logging.info(f'Star {a} has fallen inside event horizon. '
                                 f'Event horizon -> {event_horizon} | Star distance -> {dist_to_black_hole}')
                stars_subsets[a] = get_random_subset_of_features(n_features)

        # Actualizo de manera binaria los subsets de cada estrella
        for a in range(n_stars):
            # Salteo el agujero negro
            if black_hole_idx == a:
                continue
            for d in range(n_features):
                x_old = stars_subsets[a][d]
                threshold = binary_threshold if binary_threshold is not None else random.uniform(0, 1)
                x_new = x_old + random.uniform(0, 1) * (black_hole_subset[d] - x_old)  # Position
                stars_subsets[a][d] = 1 if abs(tanh(x_new)) > threshold else 0

    return black_hole_subset, black_hole_fitness


def improved_binary_black_hole(
        n_stars: int,
        n_features: int,
        n_iterations: int,
        fitness_function: Callable[[np.ndarray], CrossValidationSparkResult],
        coeff_1: float,
        coeff_2: float,
        binary_threshold: Optional[float] = 0.6,
        debug: bool = False
):
    """
    Computa la metaheuristica Binary Black Hole con algunas mejoras sacada del paper
    "Improved black hole and multiverse algorithms fordiscrete sizing optimization of planar structures"
    Authors: Saeed Gholizadeh, Navid Razavi & Emad Shojaei
    NOTA: esta hecho de manera lenta para que se pueda entender el algoritmo, usar la version vectorizada para
    produccion
    :param n_stars: Number of stars
    :param n_features: Number of features
    :param n_iterations: Number of iterations
    :param fitness_function: Fitness function to compute on every star
    :param coeff_1: Parametro especificado en el paper. Valores posibles = [2.2, 2.35]
    :param coeff_2: Parametro especificado en el paper. Valores posibles = [0.1, 0.2, 0.3]
    :param binary_threshold: Binary threshold to set 1 or 0 the feature. If None it'll be computed randomly
    :param debug: If True logs everything is happening inside BBHA
    :return:
    """
    coef_1_possible_values = [2.2, 2.35]
    coef_2_possible_values = [0.1, 0.2, 0.3]
    if coeff_1 not in coef_1_possible_values:
        print(f'El parámetro coef_1 debe ser alguno de los siguientes valores -> {coef_1_possible_values}')
        exit(1)

    if coeff_2 not in coef_2_possible_values:
        print(f'El parámetro coef_2 debe ser alguno de los siguientes valores -> {coef_2_possible_values}')
        exit(1)

    # Preparo las estructuras de datos
    stars_subsets = np.empty((n_stars, n_features), dtype=int)
    stars_best_subset = np.empty((n_stars, n_features), dtype=int)
    stars_fitness_values = np.empty((n_stars,), dtype=float)
    stars_best_fitness_values = np.empty((n_stars,), dtype=float)

    # Inicializo las estrellas con sus subconjuntos y sus valores fitness
    if debug:
        logging.info('Initializing stars...')
    for i in range(n_stars):
        random_features_to_select = get_random_subset_of_features(n_features)
        stars_subsets[i] = random_features_to_select  # Inicializa 'Population'
        stars_fitness_values[i] = fitness_function(random_features_to_select)
        # Best fitness and position
        stars_best_fitness_values[i] = stars_fitness_values[i]
        stars_best_subset[i] = stars_subsets[i]

    # El que menor fitness tiene es el Black Hole
    black_hole_idx, black_hole_subset, black_hole_fitness = get_best(stars_subsets, stars_fitness_values)
    if debug:
        logging.info(f'Black hole starting at star with index {black_hole_idx}')

    # Iteraciones
    for i in range(n_iterations):
        if debug:
            logging.info(f'Iteration -> {i + 1} of {n_iterations}')
        for a in range(n_stars):
            # Si es la misma estrella que se convirtio en agujero negro, no hago nada
            if a == black_hole_idx:
                continue

            # Compute the current star fitness
            current_star_subset = stars_subsets[a]
            current_fitness = fitness_function(current_star_subset)

            # Sets best fitness and position
            if current_fitness > stars_best_fitness_values[a]:
                stars_best_fitness_values[a] = current_fitness
                stars_best_subset[a] = current_star_subset

            # Si la estrella tiene mejor fitness que el agujero negro, hacemos swap
            if current_fitness > black_hole_fitness:
                if debug:
                    logging.info(f'Changing Black hole for star {a},'
                                 f' BH fitness -> {black_hole_fitness} | Star {a} fitness -> {current_fitness}')
                black_hole_idx = a
                black_hole_subset, current_star_subset = current_star_subset, black_hole_subset
                black_hole_fitness, current_fitness = current_fitness, black_hole_fitness

            # Si la funcion de fitness es igual, pero tengo menos features en la estrella (mejor!), hacemos swap
            elif current_fitness == black_hole_fitness and np.count_nonzero(current_star_subset) < np.count_nonzero(
                    black_hole_subset):
                if debug:
                    logging.info(f'Changing Black hole for star {a},'
                                 f' BH fitness -> {black_hole_fitness} | Star {a} fitness -> {current_fitness}')
                black_hole_idx = a
                black_hole_subset, current_star_subset = current_star_subset, black_hole_subset
                black_hole_fitness, current_fitness = current_fitness, black_hole_fitness

            # Calculo el horizonte de eventos
            # MEJORA 1: nueva funcion para definir el horizonte de eventos
            event_horizon = (1 / black_hole_fitness) / np.sum(1 / stars_fitness_values)

            # Me fijo si la estrella cae en el horizonte de eventos
            dist_to_black_hole = np.linalg.norm(black_hole_subset - current_star_subset)  # Dist. Euclidea
            if dist_to_black_hole < event_horizon:
                if debug:
                    logging.info(f'Star {a} has fallen inside event horizon. '
                                 f'Event horizon -> {event_horizon} | Star distance -> {dist_to_black_hole}')

                # MEJORA 2: se cambia solo UNA dimension del arreglo de features
                random_feature_idx = random.randint(0, n_features - 1)
                stars_subsets[a][random_feature_idx] ^= 1  # Invierte el 0 o 1

        # Actualizo de manera binaria los subsets de cada estrella
        # MEJORA 3: Nueva formula de corrimiento de una estrella
        w = 1 - (i / n_iterations)
        d1 = coeff_1 + w
        d2 = coeff_2 + w

        for a in range(n_stars):
            # Salteo el agujero negro
            if black_hole_idx == a:
                continue
            for d in range(n_features):
                x_old = stars_subsets[a][d]
                x_best = stars_best_subset[a][d]
                threshold = binary_threshold if binary_threshold is not None else random.uniform(0, 1)
                bh_star_diff = black_hole_subset[d] - x_old
                star_best_fit_diff = x_best - x_old
                x_new = x_old + (d1 * random.uniform(0, 1) * bh_star_diff) + (
                        d2 * random.uniform(0, 1) * star_best_fit_diff)
                stars_subsets[a][d] = 1 if abs(tanh(x_new)) > threshold else 0

    return black_hole_subset, black_hole_fitness


def binary_black_hole_spark(
        n_stars: int,
        n_features: int,
        n_iterations: int,
        fitness_function: Callable[[np.ndarray], CrossValidationSparkResult],
        sc: SparkContext,
        binary_threshold: Optional[float] = 0.6,
        debug: bool = False
):
    """
    Computa la metaheuristica Binary Black Hole sacada del paper
    "Binary black hole algorithm for feature selection and classification on biological data"
    Authors: Elnaz Pashaei, Nizamettin Aydin.
    NOTA: esta hecho de manera lenta para que se pueda entender el algoritmo, usar la version vectorizada para
    produccion
    :param n_stars: Number of stars
    :param n_features: Number of features
    :param n_iterations: Number of iterations
    :param fitness_function: Fitness function to compute on every star
    :param sc: Spark Context
    :param binary_threshold: Binary threshold to set 1 or 0 the feature. If None it'll be computed randomly
    :param debug: If True logs everything is happening inside BBHA
    :return:
    """
    # Preparo las estructuras de datos
    stars_subsets = np.empty((n_stars, 2), dtype=object)  # 2 = (1, features)

    # Inicializo las estrellas con sus subconjuntos y sus valores fitness
    if debug:
        logging.info('Initializing stars...')

    for i in range(n_stars):
        random_features_to_select = get_random_subset_of_features(n_features)
        stars_subsets[i] = (i + 1, random_features_to_select)  # Initializes 'Population' with a key for partitionBy()

    initial_benchs: WorkerTimeDict = {}
    initial_idle: WorkerTimeDict = {}
    initial_stars_results_values, initial_total = parallelize_fitness_execution_by_partitions(sc, stars_subsets,
                                                                                              fitness_function)

    # If requested, reports initial times
    if debug:
        for init_idx in range(n_stars):
            current_data = initial_stars_results_values[init_idx]
            store_times(init_idx, current_data, initial_total, initial_benchs, initial_idle, debug)

        report_exec_and_idle_times(initial_benchs, initial_idle)

    # Workers by partition benchmarks
    workers_benchmarks: WorkerTimeDict = {}
    workers_idle: WorkerTimeDict = {}

    # El que mejor fitness tiene es el Black Hole
    black_hole_idx, black_hole_subset, black_hole_data = get_best_spark(stars_subsets, initial_stars_results_values)
    black_hole_fitness = black_hole_data[0]

    if debug:
        logging.info(f'Black hole starting at star with index {black_hole_idx}')

    # Iteraciones
    for i in range(n_iterations):
        if debug:
            logging.info(f'Iteration -> {i + 1} of {n_iterations}')

        stars_results_values, total_iteration_time = parallelize_fitness_execution_by_partitions(sc, stars_subsets,
                                                                                                 fitness_function)

        for a in range(n_stars):
            # Si es la misma estrella que se convirtio en agujero negro, no hago nada
            if a == black_hole_idx:
                continue

            # Compute the current star fitness
            current_star_subset = stars_subsets[a][1]
            current_data = stars_results_values[a]
            current_fitness = current_data[0]

            store_times(a, current_data, total_iteration_time, workers_benchmarks, workers_idle, debug)

            # Si la estrella tiene mejor fitness que el agujero negro, hacemos swap
            if current_fitness > black_hole_fitness:
                if debug:
                    logging.info(f'Changing Black hole for star {a},'
                                 f' BH fitness -> {black_hole_fitness} | Star {a} fitness -> {current_fitness}')
                black_hole_idx = a
                black_hole_subset, current_star_subset = current_star_subset, black_hole_subset
                black_hole_data, current_data = current_data, black_hole_data
                black_hole_fitness, current_fitness = current_fitness, black_hole_fitness

            # Si la funcion de fitness es igual, pero tengo menos features en la estrella (mejor!), hacemos swap
            elif current_fitness == black_hole_fitness and np.count_nonzero(current_star_subset) < np.count_nonzero(
                    black_hole_subset):
                if debug:
                    logging.info(f'Changing Black hole for star {a},'
                                 f' BH fitness -> {black_hole_fitness} | Star {a} fitness -> {current_fitness}')
                black_hole_idx = a
                black_hole_subset, current_star_subset = current_star_subset, black_hole_subset
                black_hole_data, current_data = current_data, black_hole_data
                black_hole_fitness, current_fitness = current_fitness, black_hole_fitness

            # Calculo el horizonte de eventos
            event_horizon = black_hole_fitness / np.sum(current_fitness)

            # Me fijo si la estrella cae en el horizonte de eventos
            dist_to_black_hole = np.linalg.norm(black_hole_subset - current_star_subset)  # Dist. Euclidea
            if dist_to_black_hole < event_horizon:
                if debug:
                    logging.info(f'Star {a} has fallen inside event horizon. '
                                 f'Event horizon -> {event_horizon} | Star distance -> {dist_to_black_hole}')
                stars_subsets[a] = (a, get_random_subset_of_features(n_features))

        # Actualizo de manera binaria los subsets de cada estrella
        for a in range(n_stars):
            # Salteo el agujero negro
            if black_hole_idx == a:
                continue
            for d in range(n_features):
                x_old = stars_subsets[a][1][d]
                threshold = binary_threshold if binary_threshold is not None else random.uniform(0, 1)
                x_new = x_old + random.uniform(0, 1) * (black_hole_subset[d] - x_old)  # Position
                stars_subsets[a][1][d] = 1 if abs(tanh(x_new)) > threshold else 0

    # Logs all the workers data
    report_exec_and_idle_times(workers_benchmarks, workers_idle)

    return black_hole_subset, black_hole_fitness, black_hole_data
