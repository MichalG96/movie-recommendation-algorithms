import pandas as pd
pd.set_option('display.max_colwidth', None)
from update_user_similarities import calculate_and_save_all_similarity_matrices

movies = pd.read_csv('db_updated/movies.csv')
ratings = pd.read_csv('db_updated/ratings.csv')
if 'timestamp' in ratings.columns:
    ratings = ratings.drop(columns='timestamp')
movie_ratings_info = pd.merge(movies, ratings, how='left')
rating_matrix = pd.read_csv('db_updated/rating_matrix.csv')

ratings_count = movie_ratings_info.groupby(['title']).count().sort_values(by='movieId', ascending=False)['movieId']
movies_count = pd.merge(movies, ratings_count, on='title').rename(columns={"movieId_x": "movieId", "movieId_y": "ratings_count"})
movies_count['ratings_count_squared'] = movies_count['ratings_count'].pow(3)

variance = pd.DataFrame(rating_matrix.var())
variance['movieId'] = variance.index.astype('int64')
variance = variance.rename(columns={0: 'variance'})
movies_count_variance = pd.merge(movies_count, variance, how='left')

movies_sampled = movies_count_variance.sample(n=500, weights='ratings_count_squared')
movies_sampled = movies_sampled.sort_values(by='variance', ascending=False)[:80]

user_id = rating_matrix.shape[0]+1

print(f'Witaj. Twoj nr id to: {user_id-1}')
print(f'Ocen nastepujace filmy w skali od 1 do 10\n'
      f'Jesli nie widziales danego filmu, pozostaw puste pole, klikajac enter\n')
ratings_with_added = ratings.copy()
for i, row in movies_sampled.iterrows():
    correct_input = False
    while not correct_input:
        rating = (input(f"Ocen film: {row['title']}\n"))
        if rating.isdigit():
            rating = int(rating)
            if (rating >= 1) and (rating <= 10):
                correct_input = True
                rating = rating/2
            else:
                print('Bledna wartosc, ocena musi miescic sie w przedziale od 1 do 10')
        else:
            if rating == '':
                correct_input = True
            else:
                print('Bledna wartosc. Musisz podac liczbe naturalna')

    if rating != '':
        added_rating = pd.DataFrame({'userId': [user_id], 'movieId': [row['movieId']], 'rating':[rating]})
        ratings_with_added = ratings_with_added.append(added_rating, ignore_index=True)

print('Dziekuje za ocenienie filmow')
ratings_with_added.to_csv('db_updated/ratings.csv', index=None)

movie_ratings = pd.merge(movies, ratings_with_added, how='left').drop(['title', 'genres'], axis=1)
rating_matrix_updated = pd.pivot_table(movie_ratings, values='rating', index='userId', columns='movieId', dropna=False)
rating_matrix_updated.index = rating_matrix_updated.index.astype('int64')

rating_matrix_updated.to_csv('db_updated/rating_matrix.csv', index=None)
print('Uzytkownik zostal dodany do bazy danych')

calculate_and_save_all_similarity_matrices(True, 'new_user')

