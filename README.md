## Introduction
The main goal of the project was to implement multiple variations of two of the most widely used algorithms for movie recommendation: **content - based recommendation** and **collaborative filltering**.
Each variation was tested separately, and the most eficcient versions were combined into hybrid system.
## Technologies
* Python 3.x
* Pandas
* NumPy
* scikit-learn
## Installation
In order to install all required dependencies, run:
```
pip install -r requirements.txt
```
## The database
All algorithms were implemented and tested, using [MovieLens 100k dataset](http://files.grouplens.org/datasets/movielens/ml-latest-small-README.html). The data were processed ang merged, in order to obtain the most suitable and convenient form to perform calculations.
## Usage
The scripts responsible for interaction with the user are :
* *add_user.py*
* *add_rating.py*
* *hybrid.py*

#### add_user.py

This script enables to add new user to the database. After running it with command:
```
python add_user.py
```
it asks the user to rate 60 movies, in order to establish his/her preferences. 
Presented films are chosen in a very specific way:
* first, all movies are assigned a weight, based on the number of ratings (the idea is to ask about rather popular movies, because there's a higher chance, that the user has seen them)
* then, 500 movies are sampled, with the more popular ones having bigger chance to be drawn
* out of the sampled movies, the script choses 60 ones with the highest variance of ratings (we want to ask about polarizing movies, because the are more desctiptive - a user giving high rating to a commonly loved movie, such as Forrest Gump, doesn't provide much useful information).

After the user is done rating movies, he/her is being added to the database (in form of a Pandas DataFrame), and similatities with all the other users are being calculated.
#### add_rating.py
This script is responsible for adding new ratings for existing users. It is run with a command:
```
python add_rating.py
```
After running the script, the user is asked to give his ID. After that he can searched the movie titles, and rate them.
#### hybrid.py
The main script, responsible for provinding recommendations, as well as evaluating different versions of the algorithm. It is run with a command:
```
python hybrid.py (--r| --e) -a1 -a2 -a3 -a4 –k [--user] [--n]     
```
##### Required arguments:
| Argument    | Description                                               |
|:-----------:|-----------------------------------------------------------| 
| --e         | evaluate algorithms by calculating MSE, RMSE and MAE      |
| --r         | recommend movies using specified version of the algorithm |
| -a1 `<int>` | the weight given to the first version of the algorithm    |
| -a2 `<int>` | the weight given to the second version of the algorithm   |
| -a3 `<int>` | the weight given to the third version of the algorithm    |
| -a4 `<int>` | the weight given to the fourth version of the algorithm   |
| -k `<int>`  | number of neighbors user in collaborative filltering      |

##### Optional arguments:
| Argument    | Description                                               |
|:-----------:|-----------------------------------------------------------| 
| --user      | ID of a user that the movies are being recommended to     |
| --n         | number of movies we want to have recommended              |
