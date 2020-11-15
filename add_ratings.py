import pandas as pd
from update_user_similarities import calculate_and_save_all_similarity_matrices
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_colwidth', -1)

rating_matrix = pd.read_csv('db_updated/rating_matrix.csv')
movies = pd.read_csv('db_updated/movies.csv')

no_of_users = rating_matrix.shape[0]
correct_user_id_input = False

while not correct_user_id_input:
    user_id_input = input(f'Podaj id uzytkownika, dla ktorego chcesz dodac ocene\n')
    if user_id_input.isdigit():
        user_id_int = int(user_id_input)
        if user_id_int >= 0 and user_id_int < no_of_users:
            user_id = user_id_int
            correct_user_id_input = True
        else:
            print('W bazie nie ma uzytkownika o takim id')
    else:
        print('Bledna wartosc, podaj liczbe oznaczajaca numer id uzytkownika')

ask_for_movies = True
correct_movie_id_input = False
correct_rating_input = False
wrong_movie = False
no_of_added_ratings = 0
while ask_for_movies:
    wrong_movie = False
    phrase = input(f'Podaj fragment tytulu filmu, ktory chcialbys ocenic\n')
    titles = movies[movies['title'].str.lower().str.contains(phrase.lower())].iloc[:,:-1]
    if titles.empty:
        print('Nie ma tytulow zawierajacych podana fraze')
    else:
        print('Oto filmy, ktore zawieraja wpisana fraze. Jesli chcialbys ocenic ktorys z nich:')
        print(titles)
        while not correct_movie_id_input:
            movie_id = input(f'Podaj movieId(jesli nie znalazles filmu, ktory chcesz ocenic, wpisz "0"):\n')
            if movie_id.isdigit():
                if int(movie_id) in titles['movieId'].values:
                    correct_movie_id_input = True
                else:
                    print('Film o takim ID nie znajduje sie wsrod listy wyszukanych tytulow')
                    wrong_movie = True
                    break
            else:
                print('Bledne dane. Wpisz wartosc liczbowa')
        while not correct_rating_input:
            if wrong_movie:
                break
            rating = (input(f"Ocen film: \n{movies[movies['movieId']==int(movie_id)]}\n"))
            if rating.isdigit():
                rating = int(rating)
                if (rating >= 1) and (rating <= 10):
                    correct_rating_input = True
                    print(f'Dodano ocene {rating} dla filmu {movies[movies["movieId"]==int(movie_id)]["title"].item()}')
                    rating = rating/2
                    rating_matrix.loc[user_id][movie_id] = rating
                    no_of_added_ratings += 1
                else:
                    print('Bledna wartosc, ocena musi miescic sie w przedziale od 1 do 10')
            else:
                print('Bledna wartosc. Musisz podac liczbe naturalna')
    correct_movie_id_input = False
    correct_rating_input = False
    decision = ''
    while not (decision == 'n' or decision == 'z'):
        decision = input('Jesli chcesz ocenic nastepny film, wpisz "n". Jesli chcesz zakonczyc dodawanie filmow, wpisz "z"\n')
    if decision == 'z':
        ask_for_movies = False

print('\nAktualizowanie macierzy ocen...')
rating_matrix.to_csv('db_updated/rating_matrix.csv', index=None)
print('\nMacierz ocen zaktualizowana')

if no_of_added_ratings >= 5:
    calculate_and_save_all_similarity_matrices(True, 'new_ratings', user_id)
