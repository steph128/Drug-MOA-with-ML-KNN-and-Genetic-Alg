import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from cuml.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm.notebook import tqdm, trange

DRUGS = pd.read_csv('kaggle_data/train_drug.csv')
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_folds(y, n_folds: int, random_state: int):
    # y has 'sig_id' as the first column
    targets = y.columns[1:]
    y_drugs = y.merge(DRUGS, on='sig_id', how='left')

    vc = y_drugs['drug_id'].value_counts()
    vc1 = vc.loc[vc <= 18].index.sort_values()  # drugs with n <= 18
    vc2 = vc.loc[vc > 18].index.sort_values()  # drugs with n > 18

    drug_count_1, drug_count_2 = {}, {}

    # Stratify n <= 18
    ml_skf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    temp = y_drugs.groupby('drug_id')[targets].mean().loc[vc1]

    for fold, (idxT, idxV) in enumerate(ml_skf.split(temp, temp[targets])):
        dd = {k: fold for k in temp.index[idxV].values}
        drug_count_1.update(dd)

    # Stratify n > 18
    ml_skf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    temp = y_drugs.loc[y_drugs['drug_id'].isin(vc2)].reset_index(drop=True)

    for fold, (idxT, idxV) in enumerate(ml_skf.split(temp, temp[targets])):
        dd = {k: fold for k in temp['sig_id'][idxV].values}
        drug_count_2.update(dd)

    y_drugs['fold'] = np.nan
    y_drugs['fold'] = y_drugs['drug_id'].map(drug_count_1)
    y_drugs.loc[y_drugs['fold'].isna(), 'fold'] = y_drugs.loc[
        y_drugs['fold'].isna(), 'sig_id'].map(drug_count_2)

    return y_drugs[['sig_id', 'fold']]


def oof_probas(X, y, v, n_folds=5, random_state=42, n_neighbors=1000):
    # First scale inputs
    X_scaled = StandardScaler().fit_transform(X.iloc(axis=1)[3:].values)
    X_scaled = np.hstack([X.iloc(axis=1)[:3].values, X_scaled])

    # Get Multi-Label Stratified K-Folds
    df_fold = get_folds(y, n_folds=n_folds, random_state=random_state)

    # Initialize array to store out-of-fold probabilities
    oof = np.zeros((X.shape[0], 206))
    for fold in range(n_folds):
        fold_idx = df_fold[df_fold['fold'] != fold].index
        pp_fold = df_fold[df_fold['fold'] == fold].index

        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_scaled[fold_idx][:, v], y.iloc[fold_idx].values[:, 1:])

        pp = model.predict_proba(X_scaled[pp_fold][:, v])
        pp = np.stack([(1 - pp[x][:, 0]) for x in range(len(pp))]).T
        oof[pp_fold, ] = pp

    return oof


class GeneticAlgorithm:
    def __init__(self, population, n_folds=5, random_state=42, n_neighbors=1000):
        self.n_folds = n_folds
        self.random_state = random_state
        self.n_neighbors = n_neighbors
        self.population = population

    def fit(self, X, y, generations=20, cross_over=0.5, mutation_rate=0.05):
        print('Fitting {} generations...'.format(str(generations)), '\n')

        for generation in range(generations):
            print('=========== GENERATION', generation+1, '=========== \n', 'Computing log loss scores...')
            fitness_scores = []

            for sample in tqdm(self.population):
                prediction_probas = oof_probas(X, y, sample, n_folds=self.n_folds,
                                               random_state=self.random_state, n_neighbors=self.n_neighbors)
                fitness_scores.append(
                    log_loss(y.iloc[:, 1:].values.flatten(), prediction_probas.flatten()))

            # Smallest log loss values to select parents
            print('Selecting best parents...')
            n_parents = len(fitness_scores) // 2
            sorted_idx = np.argsort(fitness_scores)[:n_parents]
            parents = np.array([self.population[i] for i in sorted_idx])

            # Create offspring
            print('Generating offspring...\n')
            offspring_shape = (self.population.shape[0] - n_parents, self.population.shape[1])
            offspring = np.ndarray(offspring_shape)
            cross_over_point = int(offspring.shape[1] * cross_over)
            for i in range(offspring.shape[0]):
                parent1_idx = i % parents.shape[0]
                parent2_idx = (i + 1) % parents.shape[0]
                offspring[i, :cross_over_point] = parents[parent1_idx, :cross_over_point]
                offspring[i, cross_over_point:] = parents[parent2_idx, cross_over_point:]

                # Mutation involves randomly flipping booleans with a 5% chance
                for j in range(len(offspring[i])):
                    if np.random.rand(1) <= mutation_rate:
                        if offspring[i, j]:
                            offspring[i, j] = False
                        else:
                            offspring[i, j] = True

            self.population[:parents.shape[0], :] = parents
            self.population[parents.shape[0]:, :] = offspring
        print('Evolution complete.')


class EnsembleClassifier:
    def __init__(self, feature_set, n_ensemble=10):
        self.feature_set = feature_set
        self.prediction_probas = None
        self.weights = None
        self.n_ensemble = n_ensemble
        self.weighted_predictions = None
        self.ensemble_log_loss = None

    def fit(self, X, y, max_generations=1000, solution_per_population=100, mutation_rate=0.05, eps=1e-20, n_folds=5,
            random_state=42, n_neighbors=1000, verbose=True):

        log_losses = []
        prediction_probas = []

        if verbose:
            print('Fitting {} models...'.format(str(self.n_ensemble)))

        for feature_subset in tqdm(self.feature_set[:self.n_ensemble]):
            prediction_proba = oof_probas(
                X, y, feature_subset,  n_folds=n_folds, random_state=random_state, n_neighbors=n_neighbors)
            prediction_probas.append(prediction_proba)
            log_losses.append(log_loss(y.iloc[:, 1:].values.flatten(), prediction_proba.flatten()))

        self.prediction_probas = np.array(prediction_probas)
        log_losses = np.array(log_losses)

        if verbose:
            print('Running ensemble weight scoring optimization...')

        # Implement GA for ensemble weighting scores
        population_size = (solution_per_population, self.n_ensemble)
        weights = np.random.uniform(0.0, 1.0, size=population_size)  # randomly initialize weights
        weights = weights / weights.sum(axis=1)[:, None]  # normalize to enforce summing to one

        tmp_weights = weights.copy()
        n_parents = solution_per_population // 2
        gen_counter = 0

        for _ in range(max_generations):
            gen_counter += 1
            fitness = np.sum(weights * log_losses, axis=1)
            smallest_idx = np.argsort(fitness)[:n_parents]
            parents = np.array([weights[i] for i in smallest_idx])

            offspring_shape = (weights.shape[0] - n_parents, weights.shape[1])
            offspring = np.ndarray(offspring_shape)
            crossover_point = offspring.shape[1] // 2

            for k in range(offspring.shape[0]):
                parent1_idx = k % parents.shape[0]
                parent2_idx = (k + 1) % parents.shape[0]
                offspring[k, :crossover_point] = parents[parent1_idx, :crossover_point]
                offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]

                x = np.random.uniform(0, 1, offspring.shape[1])
                mutation_idx = np.where(x < mutation_rate)[0]
                offspring[k, mutation_idx] = np.random.uniform(0, 1, len(mutation_idx)) ** 2.0

            tmp_weights[:parents.shape[0], :] = parents
            tmp_weights[parents.shape[0]:, :] = offspring
            tmp_weights = tmp_weights / tmp_weights.sum(axis=1)[:, None]  # sums to one

            if (np.abs(tmp_weights - weights) < eps).all():
                self.weights = tmp_weights[0]
                print('Threshold achieved in {} generations.\n'.format(str(gen_counter)))
                break
            else:
                weights = tmp_weights

        if gen_counter == max_generations:
            self.weights = tmp_weights[0]
            if verbose:
                print("Max iterations reached without convergence.\n")

        self.weighted_predictions = self.prediction_probas.T.dot(self.weights).T
        self.ensemble_log_loss = log_loss(y.iloc[:, 1:].values.flatten(), self.weighted_predictions.flatten())

        if verbose:
            print('Final log loss:', str(self.ensemble_log_loss))


















































