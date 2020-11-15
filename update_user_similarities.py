import pandas as pd
import numpy as np
import math
import time
import warnings
warnings.filterwarnings("ignore")
from collaborative_initial_data import CollaborativeInitial
import logging
logging.basicConfig(filename='logs/logs.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

class Offline(CollaborativeInitial):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.matrix_name = 'user_similarity'
        self.get_initial_data()

    def get_common_items(self, user_a, user_b):
        movies_watched_by_user_a = set(self.rating_matrix.iloc[user_a].index.where(self.rating_matrix.iloc[user_a].notnull()).dropna())
        movies_watched_by_user_b = set(self.rating_matrix.iloc[user_b].index.where(self.rating_matrix.iloc[user_b].notnull()).dropna())
        movies_watched_by_both = list(movies_watched_by_user_a & movies_watched_by_user_b)
        return movies_watched_by_both

    def pearson_similarity(self, user_a, user_b, *args):
        common_items = self.get_common_items(user_a, user_b)
        if args:
            w = args[0]
            if common_items:
                licznik = sum([(w[i] * self.rating_matrix_mean_centered_column_wise.at[user_a, i]) *
                               (self.rating_matrix_mean_centered_column_wise.at[user_b, i]) for i in common_items])
                mianownik = np.sqrt(sum([(w[i] * self.rating_matrix_mean_centered_column_wise.at[user_a, i]) ** 2 for i in
                     common_items]) * sum([(w[i] * self.rating_matrix_mean_centered_column_wise.at[user_b, i]) ** 2 for i in common_items]))
                return licznik / mianownik
            else:
                return 0
        else:
            if common_items:
                licznik = sum([(self.rating_matrix_mean_centered_column_wise.at[user_a, i]) *
                               (self.rating_matrix_mean_centered_column_wise.at[user_b, i]) for i in common_items])
                mianownik = math.sqrt(sum([(self.rating_matrix_mean_centered_column_wise.at[user_a, i]) ** 2 for i in
                     common_items]) * sum([(self.rating_matrix_mean_centered_column_wise.at[user_b, i]) ** 2 for i in common_items]))
                return licznik / mianownik
            else:
                return 0

    def pearson_similarity_discounted(self, user_a, user_b, beta, *args):
        common_items = self.get_common_items(user_a, user_b)
        discount = min(len(common_items), beta) / beta
        if args:
            w = args[0]
            if common_items:
                licznik = sum([(w[i] * self.rating_matrix_mean_centered_column_wise.at[user_a, i]) *
                               (self.rating_matrix_mean_centered_column_wise.at[user_b, i]) for i in common_items])
                mianownik = math.sqrt(sum([(w[i] * self.rating_matrix_mean_centered_column_wise.at[user_a, i]) ** 2 for i in common_items])
                                      * sum([(w[i] * self.rating_matrix_mean_centered_column_wise.at[user_b, i]) ** 2 for i in common_items]))
                return (licznik / mianownik) * discount
            else:
                return 0
        else:
            if common_items:
                licznik = sum([(self.rating_matrix_mean_centered_column_wise.at[user_a, i]) *
                               (self.rating_matrix_mean_centered_column_wise.at[user_b, i]) for i in
                               common_items])
                mianownik = math.sqrt(sum([(self.rating_matrix_mean_centered_column_wise.at[user_a, i]) ** 2 for i in common_items])
                                      * sum([(self.rating_matrix_mean_centered_column_wise.at[user_b, i]) ** 2 for i in common_items]))
                return (licznik / mianownik) * discount
            else:
                return 0

    def pearson_similarity_alpha(self, user_a, user_b, alpha, *args):
        common_items = self.get_common_items(user_a, user_b)
        if args:
            w = args[0]
            if common_items:
                licznik = sum([(w[i] * self.rating_matrix_mean_centered_column_wise.at[user_a, i]) *
                               (self.rating_matrix_mean_centered_column_wise.at[user_b, i]) for i in
                               common_items])
                mianownik = math.sqrt(sum([(w[i] * self.rating_matrix_mean_centered_column_wise.at[user_a, i]) ** 2 for i in
                     common_items])* sum([(w[i] * self.rating_matrix_mean_centered_column_wise.at[user_b, i]) ** 2 for i in
                     common_items]))
                return (licznik / mianownik) ** alpha
            else:
                return 0
        else:
            if common_items:
                licznik = sum([(self.rating_matrix_mean_centered_column_wise.at[user_a, i]) *
                               (self.rating_matrix_mean_centered_column_wise.at[user_b, i]) for i in
                               common_items])
                mianownik = math.sqrt(sum([(self.rating_matrix_mean_centered_column_wise.at[user_a, i]) ** 2 for i in
                     common_items]) * sum([(self.rating_matrix_mean_centered_column_wise.at[user_b, i]) ** 2 for i in
                     common_items]))
                return (licznik / mianownik) ** alpha
            else:
                return 0

    def create_similarity_matrix(self):
        user_similarity_np = np.eye(self.no_of_users)
        weight = np.log10(self.no_of_users / self.rating_matrix.count())
        if self.mode == 'nw':
            func = self.pearson_similarity
        elif self.mode == 'aw3':
            func = self.pearson_similarity_alpha
            alfa = 3
        elif self.mode == 'dw8':
            func = self.pearson_similarity_discounted
            beta = 8
        for i in range(self.no_of_users):
            if i % 50 == 0:
                print(f'Obliczono podobienstwo dla {i} z {self.no_of_users} uzytkownikow')
            for j in range(self.no_of_users):
                if i < j:
                    if self.mode == 'nw':
                        user_similarity_np[i, j] = func(i, j, weight)
                    elif self.mode == 'aw3':
                        user_similarity_np[i, j] = func(i, j, alfa, weight)
                    elif self.mode == 'dw8':
                        user_similarity_np[i, j] = func(i, j, beta, weight)
        user_similarity_np = user_similarity_np + user_similarity_np.T
        np.fill_diagonal(user_similarity_np, 1)
        similarity_matrix_pd = pd.DataFrame(user_similarity_np)
        return similarity_matrix_pd

    def update_similarity_matrix(self):
        weight = np.log10(self.no_of_users / self.rating_matrix.count())
        if self.mode == 'nw':
            func = self.pearson_similarity
        elif self.mode == 'aw3':
            func = self.pearson_similarity_alpha
            alfa = 3
        elif self.mode == 'dw8':
            func = self.pearson_similarity_discounted
            beta = 8

        old_similarity_matrix = pd.read_csv(f'db_updated/user_similarities/user_similarity{self.similarities_paths[self.mode]}')
        previous_number_of_users = old_similarity_matrix.shape[0]
        number_of_added_users = self.no_of_users - previous_number_of_users
        new_users_ids = [previous_number_of_users + i for i in range(number_of_added_users)]

        for i in range(self.no_of_users):
            for new_user_id in new_users_ids:
                old_similarity_matrix.loc[new_user_id,new_user_id] = 1
                if self.mode == 'nw':
                    old_similarity_matrix.loc[new_user_id][i] = func(new_user_id, i, weight)
                    old_similarity_matrix.loc[i][new_user_id] = func(new_user_id, i, weight)
                elif self.mode == 'aw3':
                    old_similarity_matrix.loc[new_user_id][i] = func(new_user_id, i, alfa, weight)
                    old_similarity_matrix.loc[i][new_user_id] = func(new_user_id, i, alfa, weight)
                elif self.mode == 'dw8':
                    old_similarity_matrix.loc[new_user_id][i] = func(new_user_id, i, beta, weight)
                    old_similarity_matrix.loc[i][new_user_id] = func(new_user_id, i, beta, weight)
            old_similarity_matrix.loc[new_user_id][new_user_id] = 1

        updated_similarity_matrix = old_similarity_matrix
        return updated_similarity_matrix

    def update_similarity_matrix_single_user(self, user_id):
        weight = np.log10(self.no_of_users / self.rating_matrix.count())
        if self.mode == 'nw':
            func = self.pearson_similarity
        elif self.mode == 'aw3':
            func = self.pearson_similarity_alpha
            alfa = 3
        elif self.mode == 'dw8':
            func = self.pearson_similarity_discounted
            beta = 8

        old_similarity_matrix = pd.read_csv(f'db_updated/user_similarities/user_similarity{self.similarities_paths[self.mode]}')
        for i in range(self.no_of_users):
            if self.mode == 'nw':
                old_similarity_matrix.loc[user_id][i] = func(user_id, i, weight)
                old_similarity_matrix.loc[i][user_id] = func(user_id, i, weight)
            elif self.mode == 'aw3':
                old_similarity_matrix.loc[user_id][i] = func(user_id, i, alfa, weight)
                old_similarity_matrix.loc[i][user_id] = func(user_id, i, alfa, weight)
            elif self.mode == 'dw8':
                old_similarity_matrix.loc[user_id][i] = func(user_id, i, beta, weight)
                old_similarity_matrix.loc[i][user_id] = func(user_id, i, beta, weight)
        old_similarity_matrix.loc[user_id][user_id] = 1

        updated_similarity_matrix = old_similarity_matrix
        return updated_similarity_matrix

    def save_similarity_matrix(self, similarity_matrix_pd):
        similarity_matrix_pd.to_csv(f'{self.path}user_similarities/{self.matrix_name}{self.similarities_paths[self.mode]}', index=None)

def calculate_and_save_all_similarity_matrices(save, action, *args):
    modes = ['nw', 'aw3', 'dw8']
    if args:
        user = args[0]
    modes_info = {'nw': ['Normal Pearson similarity, weighted', 'Wazone podobiestwo pearsona'],
                  'aw3': ['Pearson similarity alpha, alpha=3, weighted',
                          'Wazone podobienstwo pearsona podniesione do potegi alfa, alfa=3'],
                  'dw8': ['Discounted Pearson similarity, beta=8, weighted',
                          'Wazone podobienstwo Pearsona ze znizka, beta=8']}
    for mode in modes:
        print('\nAktualizowanie macierzy podobienstw...')
        print(f'Sposob obliczania podobienstwa: {modes_info[mode][1]}')
        off = Offline(mode)
        start_time = time.perf_counter()
        txt1 = 'aktualizowania'
        txt2 = 'Updating'
        if action == 'create':
            txt1 = 'tworzenia'
            txt2 = 'Creating'
            sim_matrix = off.create_similarity_matrix()
        if action == 'new_user':
            sim_matrix = off.update_similarity_matrix()
        elif action == 'new_ratings':
            sim_matrix = off.update_similarity_matrix_single_user(user)

        end_time = time.perf_counter()
        run_time = end_time - start_time
        print(f'Czas {txt1} macierzy: {run_time:.4f} sekund')
        logging.info(f'{txt2} user similarity matrix. {modes_info[mode][0]}. Time: {run_time:.4f} s')
        if save:
            off.save_similarity_matrix(sim_matrix)

    print('\nMacierze podobienstw zostaly zaktualizowane. Mozesz przystapic do rekomendacji')
