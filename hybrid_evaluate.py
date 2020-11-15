import pandas as pd
import numpy as np
import random
from scipy import sparse
import time
import logging
logging.basicConfig(filename='logs/logs.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
pd.set_option('display.max_rows', 50)

from hybrid_recommend import Recommend

class Evaluate(Recommend):
    def __init__(self, k, cf_1, cf_2, cf_3, cb_1, test_matrix):
        super().__init__(k, cf_1, cf_2, cf_3, cb_1)
        if test_matrix == 'initial':
            self.using_initial_data = True
            self.path = 'db_initial/'
            self.get_initial_data()
            self.rating_matrix_08 = pd.read_csv(f'{self.path}rating_matrix_08.csv')
        elif test_matrix == 'create':
            self.using_initial_data = False
            self.rating_matrix_08 = self.create_matrix_08()
        else:
            self.using_initial_data = False
            self.rating_matrix_08 = pd.read_csv(test_matrix)

    def create_matrix_08(self):
        print('Tworze zbior testowy...')
        rating_matrix_08 = self.rating_matrix.copy()
        no_of_ratings = self.rating_matrix.count().sum()
        random.seed(12345)
        random_ratings_to_remove = random.sample(range(no_of_ratings), no_of_ratings // 5)
        counter = 0
        coordinates_of_removed_items = {}
        for i, row in self.rating_matrix.iterrows():
            cols = []
            for col, value in enumerate(row):
                if pd.notna(value):
                    if counter in random_ratings_to_remove:
                        rating_matrix_08.iloc[i, col] = np.nan
                        cols.append(col)
                    counter += 1
            coordinates_of_removed_items[i] = cols
        print('Utworzono zbior testowy')
        return rating_matrix_08

    def fill_matrix_08(self):
        rating_matrix_predicted = self.rating_matrix_08.copy()
        where_ratings_were_removed = self.rating_matrix.notnull() ^ self.rating_matrix_08.notnull()
        for user in self.rating_matrix.index:
            if user%50 == 0:
                print(f'Obliczono oceny dla {user} z {self.no_of_users} uzytkownikow')
            if self.alpha_cb_1 > 0:
                ratings_list, ids_list, movie_ids_list = self.get_rated_movies(userId=user)
                Dl = self.tf_idf[ids_list]
                y = sparse.csr_matrix(np.array(ratings_list)).T
                wt_1 = sparse.linalg.inv((np.dot(Dl.T, Dl) + self.lam_I))
                wt_2 = np.dot(Dl.T, y)
                Wt = np.dot(wt_1, wt_2)
                predicted_ratings_CB = np.dot(self.tf_idf, Wt).todense()
            if len(self.active_cf_modes) > 0:
                for movieId in where_ratings_were_removed.columns:
                    if where_ratings_were_removed.loc[user, movieId]:
                        cf_score = 0
                        for index, mode in enumerate(self.active_cf_modes):
                            cf_score += self.alphas[index] * np.nan_to_num(self.predict_collaborative(user, movieId, mode))
                        if self.alpha_cb_1 > 0:
                            cb_score = self.alphas[-1] * np.nan_to_num(predicted_ratings_CB[self.ids_for_movie_ids[movieId]])
                            weighted_rating = cf_score + cb_score
                            rating_matrix_predicted.at[user, movieId] = weighted_rating
                        else:
                            rating_matrix_predicted.at[user, movieId] = cf_score

            else:
                for movieId in where_ratings_were_removed.columns:
                    if where_ratings_were_removed.loc[user, movieId]:
                        cb_score = np.nan_to_num(predicted_ratings_CB[self.ids_for_movie_ids[movieId]])
                        rating_matrix_predicted.at[user, movieId] = cb_score

        return rating_matrix_predicted

    def rmse(self, predicted_matrix):
        matrix_difference = predicted_matrix.sub(self.rating_matrix)
        matrix_difference_squared = matrix_difference.pow(2)
        no_of_ratings = self.rating_matrix.count().sum()  # tyle ocen bylo w oryginalnej macierzy ocen
        no_of_ratings_to_calculate = no_of_ratings // 5  # tyle ocen nalezalo policzyc
        no_of_nan_ratings = no_of_ratings - predicted_matrix.count().sum()  # tyle z policzonych ocen wyszlo jako nan
        no_of_observed_ratings = no_of_ratings_to_calculate - no_of_nan_ratings  # tyle brakujacych ocen policzono
        mse = matrix_difference_squared.sum(axis=1).sum(axis=0) / (no_of_observed_ratings)
        rmse = np.sqrt(mse)
        mae = matrix_difference.abs().sum(axis=1).sum(axis=0) / (no_of_observed_ratings)
        return mse, rmse, mae

    def evaluate_filled_matrix(self, save):
        print('\nProsze czekac, trwa dokonywanie ewaluacji...')
        start_time = time.perf_counter()
        predicted_matrix = self.fill_matrix_08()
        end_time = time.perf_counter()
        run_time = end_time - start_time
        mse, rmse, mae = self.rmse(predicted_matrix)
        print(f'\nWyniki ewaluacji systemu:')
        print(f'Wagi poszczegolny algorytmow: [{self.alpha_cf_1}, {self.alpha_cf_2}, {self.alpha_cf_3}, {self.alpha_cb_1}]. MSE: {mse:.4f}. RMSE: {rmse:.4f}. MAE: {mae:.4f}. Czas wykonywania obliczen: {run_time:.4f} s.')
        logging.info(f'Hybrid. Alphas: [{self.alpha_cf_1}, {self.alpha_cf_2}, {self.alpha_cf_3}, {self.alpha_cb_1}]. MSE: {mse:.4f}. RMSE: {rmse:.4f}. MAE: {mae:.4f}. Time: {run_time:.4f}')

