import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import logging
logging.basicConfig(filename='logs/logs.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y/%m/%d %H:%M:%S')
import warnings
warnings.filterwarnings("ignore")
from collaborative_initial_data import CollaborativeInitial
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_colwidth', -1)

class Recommend(CollaborativeInitial):
    def __init__(self, k, cf_1, cf_2, cf_3, cb_1):
        super().__init__()
        self.k = k
        self.full_info_bag_of_words = pd.read_csv('db_initial/full_info_bag_of_words.csv')
        self.full_info = pd.read_csv('db_initial/full_info.csv')
        self.ratings = pd.read_csv('db_initial/ratings.csv')
        self.no_of_ratings = self.rating_matrix.count().sum()
        lam = 0.01
        no_of_features = 1000

        weights_sum = cf_1 + cf_2 + cf_3 + cb_1
        if weights_sum == 0:
            print('Podano nieprawidlowe wagi. Kazdemu algorytmowi zostanie przydzielona rowna waga (1/suma algorytmow))')
            self.alpha_cf_1, self.alpha_cf_2, self.alpha_cf_3, self.alpha_cb_1 = 0.25, 0.25, 0.25, 0.25
        else:
            self.alpha_cf_1 = round(cf_1/weights_sum, 2)
            self.alpha_cf_2 = round(cf_2/weights_sum, 2)
            self.alpha_cf_3 = round(cf_3/weights_sum, 2)
            self.alpha_cb_1 = round(cb_1/weights_sum, 2)

        self.alphas = []
        self.active_cf_modes = []

        if self.alpha_cf_1 > 0:
            self.active_cf_modes.append('nw')
            self.alphas.append(self.alpha_cf_1)
            self.user_sim_normal = pd.read_csv(f'{self.path}user_similarities/user_similarity{self.similarities_paths["nw"]}')
        if self.alpha_cf_2 > 0:
            self.active_cf_modes.append('aw3')
            self.alphas.append(self.alpha_cf_2)
            self.user_sim_alpha_3 = pd.read_csv(f'{self.path}user_similarities/user_similarity{self.similarities_paths["aw3"]}')
        if self.alpha_cf_3 > 0:
            self.active_cf_modes.append('dw8')
            self.alphas.append(self.alpha_cf_3)
            self.user_sim_beta_8 = pd.read_csv(f'{self.path}user_similarities/user_similarity{self.similarities_paths["dw8"]}')
        if self.alpha_cb_1 > 0:
            self.alphas.append(self.alpha_cb_1)
            vectorizer = TfidfVectorizer(max_features=no_of_features)
            self.tf_idf = vectorizer.fit_transform(self.full_info_bag_of_words['bag_of_words'])
            self.lam_I = sparse.csr_matrix(lam * np.eye(self.tf_idf.shape[1]))
            self.feature_names = vectorizer.get_feature_names()

        self.movie_ids_for_ids = {i: self.rating_matrix.columns[i] for i in range(self.rating_matrix.shape[1])}
        self.ids_for_movie_ids = {self.rating_matrix.columns[i]: i for i in range(self.rating_matrix.shape[1])}

        print('\nZebrano poczatkowe dane.')

    @staticmethod
    def sigmoid(x):
        if x < 10:
            return (math.e ** (0.6 * x + 0.3)) / (math.e ** (0.6 * x + 0.3) + 1)
        else:
            return 1

    def get_k_most_similar(self, user, movie, user_similarity_pd):
        users_which_watched_this_movie = self.rating_matrix[self.rating_matrix[str(movie)].notnull()].index.tolist()
        penalty = self.sigmoid(len(users_which_watched_this_movie) - 1)

        similarities = user_similarity_pd.iloc[user, users_which_watched_this_movie]
        similarities = similarities[similarities > 0.01]  # usuniecie peersow o bardzo niskim stopniu podobienstwa

        most_similar_which_watched_this_movie = similarities.sort_values(ascending=False).index.astype('int64').tolist()
        if user in most_similar_which_watched_this_movie:
            most_similar_which_watched_this_movie.remove(user)
        if len(most_similar_which_watched_this_movie) < self.k:
            return most_similar_which_watched_this_movie, penalty
        else:
            return most_similar_which_watched_this_movie[:self.k], penalty

    def get_top_5_peers(self, user):
        if len(self.active_cf_modes)>0:
            if self.alpha_cf_1 > 0:
                user_similarity_pd = pd.read_csv(f'{self.path}user_similarities/user_similarity{self.similarities_paths["nw"]}')
            if self.alpha_cf_2 > 0:
                user_similarity_pd = pd.read_csv(f'{self.path}user_similarities/user_similarity{self.similarities_paths["aw3"]}')
            if self.alpha_cf_3 > 0:
                user_similarity_pd = pd.read_csv(f'{self.path}user_similarities/user_similarity{self.similarities_paths["dw8"]}')
            similarities = user_similarity_pd.iloc[user, :]
            top_peers = similarities.sort_values(ascending=False).index.astype('int64').tolist()
            top_peers.remove(user)
            top_peers = top_peers[:5]
            return f'\nId 5 najbardziej podobnych uzytkownikow: {", ".join(str(v) for v in top_peers)}'
        else:
            return'\nNie mozna wskazac najbardziej podobnych uzytkownikow, nie korzystano z filtrowania kolaboratywnego'

    def predict_single_rating(self, user, movie, user_similarity_pd):
        k_peers, penalty = self.get_k_most_similar(user, movie, user_similarity_pd)
        if len(k_peers) == 0:
            return 0
        else:
            movie = str(movie)
            licznik = sum([user_similarity_pd.iat[user, i] * self.rating_matrix_mean_centered_column_wise.at[i, movie]for i in k_peers])
            mianownik = sum([abs(user_similarity_pd.iat[user, i]) for i in k_peers])
            if mianownik == 0:
                return 0
            else:
                predicted_rating = (self.mean_of_users[user] + self.users_std[user]*(licznik / mianownik)) * penalty
                return predicted_rating

    def predict_collaborative(self, user, movie, mode):
        if mode == 'nw':
            user_similarities = self.user_sim_normal
        elif mode == 'aw3':
            user_similarities = self.user_sim_alpha_3
        elif mode == 'dw8':
            user_similarities = self.user_sim_beta_8

        rating = self.predict_single_rating(user, movie, user_similarities)
        return rating

    def get_rated_movies(self, userId):
        rated_movies = self.rating_matrix.iloc[userId].dropna()
        ratings_list = rated_movies.tolist()
        movie_ids_list = rated_movies.index.tolist()
        ids_list = self.full_info_bag_of_words[self.full_info_bag_of_words['movieId'].isin(movie_ids_list)].index.tolist()
        return ratings_list, ids_list, movie_ids_list

    def predict_content_based(self, user):
        ratings_list, ids_list, movie_ids_list = self.get_rated_movies(userId=user)
        Dl = self.tf_idf[ids_list]
        y = sparse.csr_matrix(np.array(ratings_list)).T
        wt_1 = sparse.linalg.inv((np.dot(Dl.T, Dl) + self.lam_I))
        wt_2 = np.dot(Dl.T, y)
        Wt = np.dot(wt_1, wt_2)
        Wt_dense = Wt.todense()
        Wt_dense = [i.item() for i in Wt_dense]
        top_5_features = np.argsort(Wt_dense)[-5:][::-1].tolist()
        top_5_features_names = [self.feature_names[i] for i in top_5_features]

        predicted_ratings = np.dot(self.tf_idf, Wt).todense()
        return [predicted_ratings, top_5_features_names]

    def predict_hybrid_for_user(self, user):
        if len(self.active_cf_modes) > 0:
            ratings_cf = {mode: [] for mode in self.active_cf_modes}
            for i in self.rating_matrix.columns:
                for mode in self.active_cf_modes:
                    ratings_cf[mode].append(self.predict_collaborative(user, i, mode))
            if self.alpha_cb_1 > 0:
                ratings_cb = self.predict_content_based(user)
                return (ratings_cf, ratings_cb)
            else:
                return (ratings_cf)
        else:
            if self.alpha_cb_1 > 0:
                ratings_cb = self.predict_content_based(user)
                return (ratings_cb)

    def weight_ratings(self, user):
        ratings_from_each = self.predict_hybrid_for_user(user)
        final_ratings = []
        if isinstance(ratings_from_each, tuple):
            ratings_cf = ratings_from_each[0]
            ratings_cb = ratings_from_each[1][0]
            top_features_names = ', '.join(ratings_from_each[1][1])
            info_msg = f"\nNajwazniejsze cechy profilu uzytkownika: {top_features_names}"
            for i in range(self.no_of_movies):
                cf_score = 0
                for index, mode in enumerate(self.active_cf_modes):
                    cf_score += self.alphas[index] * np.nan_to_num(ratings_cf[mode][i])
                cb_score = self.alphas[-1] * np.nan_to_num(ratings_cb[i].item())
                final_ratings.append(cf_score + cb_score)

        elif self.alpha_cb_1 > 0:   #sa wagi tylko dla content-based
            ratings_cb = ratings_from_each[0]
            top_features_names = ', '.join(ratings_from_each[1])
            info_msg = f"\nNajwazniejsze cechy profilu uzytkownika: {top_features_names}"
            for i in range(self.no_of_movies):
                cb_score = self.alphas[-1] * np.nan_to_num(ratings_cb[i].item())
                final_ratings.append(cb_score)

        else:   # sa wagi tylko dla collaborative
            ratings_cf = ratings_from_each
            info_msg = '\nNie mozna okreslic najwazniejszych cech profilu uzytkownika. Nie wybrano metody opartej na zawartosci'
            for i in range(self.no_of_movies):
                cf_score = 1
                for index, mode in enumerate(self.active_cf_modes):
                    cf_score += self.alphas[index] * np.nan_to_num(ratings_cf[mode][i])
                final_ratings.append(cf_score)
        return final_ratings, info_msg

    def recommend_top_n(self, user, n):
        print('\nProsze czekac, trwa dokonywanie rekomendacji...')
        weighted_ratings, msg = self.weight_ratings(user)
        top_peers = self.get_top_5_peers(user)
        top_ids = np.argsort(weighted_ratings)[::-1]
        movies_watched_by_this_user = self.rating_matrix.iloc[user].dropna().index.tolist()
        top_n_movie_ids = []
        counter = 0
        for i in top_ids:
            movie_id = self.movie_ids_for_ids[i]
            if(movie_id) not in movies_watched_by_this_user:
                top_n_movie_ids.append(movie_id)
                counter += 1
            if counter >= n:
                break
        print('\nDOKONANO REKOMENDACJI')
        print(msg)
        print(top_peers)
        print(f'\nOto lista {n} filmow wybranych dla uzytkownika nr {user}:\n')
        top_movies = (self.full_info[self.full_info['movieId'].isin(top_n_movie_ids)])
        top_movies.loc[:,'order'] = np.argsort(list(map(int, top_n_movie_ids)))
        print(top_movies.sort_values(by='order').loc[:, ['movieId', 'title']])
        return top_movies['title'].tolist()
