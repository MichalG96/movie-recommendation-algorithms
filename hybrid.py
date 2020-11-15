import argparse

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--e', help='Ocen zadana wersje algorytmu', action='store_true')
group.add_argument('--r', help='Dokonaj rekomendacji', action='store_true')
parser.add_argument('--user', type=int, help='Id uzytkownika, dla ktorego ma byc dokonana rekomendacja')
parser.add_argument('-a1', type=int, help='Liczba z przedzialu od 0 do 10, okreslajaca wage, z jaka bedzie wykorzystane'
                                          'przy rekomendacji filtrowanie kolaboratywne, wykorzystujace normalne, wazone'
                                          'podobienstwo Pearsona.', required=True)
parser.add_argument('-a2', type=int, help='Liczba z przedzialu od 0 do 10, okreslajaca wage, z jaka bedzie wykorzystane'
                                          'przy rekomendacji filtrowanie kolaboratywne, wykorzystujace wazone'
                                          'podobienstwo Pearsona podniesione do potegi alfa, alfa = 3.', required=True)
parser.add_argument('-a3', type=int, help='Liczba z przedzialu od 0 do 10, okreslajaca wage, z jaka bedzie wykorzystane'
                                          'przy rekomendacji filtrowanie kolaboratywne, wykorzystujace wazone'
                                          'podobienstwo Pearsona ze znizka, beta = 8.', required=True)
parser.add_argument('-a4', type=int, help='Liczba z przedzialu od 0 do 10, okreslajaca wage, z jaka bedzie wykorzystane'
                                          'przy rekomendacji predykcja oparta na zawartosci.', required=True)
parser.add_argument('-k', type=int, help='Liczba sasiadow, jaka ma byc brana pod uwage przy dokonywaniu predykcji '
                                          'z uzyciem filtrowania kolaboratywnego. Minimalna wartosc: 5, maksymalna wartosc: 40', required=True)
parser.add_argument('--n', type=int, help='Liczba filmow, jaka ma zostac zarekomendowana. Minimalna wartosc: 1, maksymalna wartosc: 30')
args = parser.parse_args()

eva = args.e
rec = args.r
user_id = args.user
a1 = args.a1
a2 = args.a2
a3 = args.a3
a4 = args.a4
k = args.k
n = args.n

if args.r and (args.n is None or args.user is None):
    parser.error("--r requires --user and --n.")

if user_id:
    if user_id < 0:
        parser.error("value of --user needs to be at least 0")
if n:
    if n < 1 or n > 30:
        parser.error("value of --n needs to be between 1 and 30")

if k < 5 or k > 40:
    parser.error("value of --k needs to be between 5 and 40")
if a1 < 0 or a1 > 10 or a2 < 0 or a2 > 10 or a3 < 0 or a3 > 10 or a4 < 0 or a4 > 10:
    parser.error("values of -a1, -a2, -a3, -a4 need to be between 0 and 10")

if eva:
    from hybrid_evaluate import Evaluate
    evaluation = Evaluate(k, a1, a2, a3, a4, test_matrix='create')
    print('\nROZPOCZETO PROCES EWALUACJI')
    evaluation.evaluate_filled_matrix(False)

elif rec:
    from hybrid_recommend import Recommend
    recommendation = Recommend(k, a1, a2, a3, a4)
    if user_id >= recommendation.no_of_users:
        print('Uzytkownik o podanym ID nie znajduje sie w bazie, rekomendacja niemozliwa')
        quit()
    print('\nROZPOCZETO PROCES REKOMENDACJI')
    recommendation.recommend_top_n(user_id, n)




