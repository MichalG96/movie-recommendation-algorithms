import pandas as pd

class CollaborativeInitial:
    def __init__(self):
        self.path = 'db_updated/'
        self.get_initial_data()
        self.similarities_paths = {'nw': '_normal_weighted.csv',
                                  'aw3': '_alpha_3_weighted.csv',
                                  'dw8': '_discounted_beta_8_weighted.csv'}
    def get_initial_data(self):
        self.rating_matrix = pd.read_csv(f'{self.path}rating_matrix.csv')
        self.no_of_users = self.rating_matrix.shape[0]
        self.no_of_movies = self.rating_matrix.shape[1]
        self.mean_of_users = self.rating_matrix.mean(axis=1)
        self.rating_matrix_mean_centered_column_wise = self.rating_matrix.sub(self.mean_of_users, axis=0)
        self.users_std = self.rating_matrix.std(axis=1)
        self.rating_matrix_mean_centered_column_wise = self.rating_matrix_mean_centered_column_wise.divide(self.users_std,axis=0)

data = CollaborativeInitial()