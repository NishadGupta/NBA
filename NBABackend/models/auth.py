from flask import Blueprint, render_template, request, flash, redirect, url_for
from sklearn.model_selection import train_test_split
import pickle
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import csv
import NBABackend
import requests

auth = Blueprint("auth", __name__)

df2 = pd.read_csv("NBABackend/csv/Player Totals.csv")

@auth.route("/",  methods = ["GET", "POST"])
def login():
    if request.method == "POST":
        playerOne = request.form.get("playerOne")
        playerTwo = request.form.get("playerTwo")
        playerThree = request.form.get("playerThree")
        playerFour = request.form.get("playerFour")
        playerFive = request.form.get("playerFive")
        print(playerOne, playerTwo, playerThree, playerFour, playerFive)
    return render_template("/index.html", text = "Trial")


def calculate_win_percentage(wins_losses):
    num_wins = wins_losses.count('W')
    num_losses = wins_losses.count('L')
    win_percentage = num_wins / (num_wins + num_losses)
    return win_percentage


@auth.route("/loadGame", methods = ["GET", "POST"])
def loadGame():
    dataSetGame = pd.read_csv("NBABackend/csv/game.csv")
    data_shape = dataSetGame.shape
    teamNames = ['MIL', 'TOT', 'MIN', 'DAL', 'DEN', 'ATL', 'IND', 'OKC', 'ORL',
       'BOS', 'DET', 'CHI', 'SAC', 'SAS', 'BRK', 'HOU', 'LAC', 'GSW',
       'POR', 'LAL', 'WAS', 'MIA', 'PHO', 'MEM', 'NOP', 'CHO', 'NYK',
       'CLE', 'TOR', 'UTA', 'PHI']
    dataSetGame = dataSetGame[dataSetGame["team_abbreviation_home"].isin(teamNames)]
    dataSetGame = dataSetGame[dataSetGame["game_date"] >="2015-01-01 00:00:00"]
    dataSetGame = dataSetGame.drop(columns = ["season_id", "team_id_home", "game_id", "min", "plus_minus_home", "video_available_home",
                                         "plus_minus_away", "video_available_away"])
    columnsMissing = dataSetGame.columns[dataSetGame.isna().any()].tolist()
    dataSetGame = dataSetGame.drop(columns = ['wl_away'])
    dataSetGame = dataSetGame.dropna()
    matchDataSet = dataSetGame.groupby(['team_abbreviation_home',"team_abbreviation_away"])['wl_home'].sum()    
    newMatchDataSet = pd.DataFrame(matchDataSet) 
    newMatchDataSet['winPercentage'] = newMatchDataSet['wl_home'].apply(calculate_win_percentage)

    teamList = np.array(newMatchDataSet.index[:])
    homelist = [t[0] for t in teamList]
    awaylist = [t[1] for t in teamList]
    newerDataSet = pd.DataFrame({"home": homelist, "away": awaylist, "winPercentage": newMatchDataSet['winPercentage'] })
    # render the template with data
    print(data_shape)
    return render_template('csv.html', text =newerDataSet)
        
@auth.route("/loadplayerTotals", methods = ["GET", "POST"])
def loadplayerTotals():
    dataSet = pd.read_csv("NBABackend/csv/Player Totals.csv")
    dataSet = dataSet[dataSet["lg"] == "NBA"]
    dataSet = dataSet[dataSet["season"] >= 2015]
    dataSet = dataSet.drop(columns = "birth_year")
    teamNames = []
    teamNames = dataSet["tm"].unique()
    missingValues = dataSet.isna().sum().sum()
    columnsMissing = dataSet.columns[dataSet.isna().any()].tolist()
    dataSet = dataSet.drop(columns = "ft_percent")
    dataSet = dataSet.fillna(0)
    columnsMissing = dataSet.columns[dataSet.isna().any()].tolist()
    dataSet = dataSet.drop(columns = ["player_id", "seas_id"])
    playerStastics = dataSet.drop(columns =["g", "gs", 'season', 'player', 'age', 'experience', 'lg', 'tm', "mp"])
    playerStastics["pos"] = playerStastics["pos"].str.split('-').str.get(0)
    middlePlayerStatistics = playerStastics.drop(columns = "pos")
    plotStatics = pd.DataFrame(MinMaxScaler(feature_range=(0, 100)).fit_transform(middlePlayerStatistics), columns=middlePlayerStatistics.columns)
    plotStatics["pos"] = playerStastics["pos"]
    plotStatics = plotStatics.groupby("pos").mean()
    scaledPlotStatistics = plotStatics.drop(columns =["fg_percent", "x3p_percent", "x2p_percent", "e_fg_percent"])
    return render_template('csv.html', text = scaledPlotStatistics)

@auth.route("/loadTeamTotals", methods = ["GET", "POST"])
def loadTeamTotals():
    df = pd.read_csv('NBABackend/csv/Team Totals.csv')
    df = df[df.season >= 2015]
    df = df[df.team != 'League Average']
    playoffs_df = df.groupby('abbreviation')['playoffs'].sum()
    playoffs_df = pd.DataFrame({'Team': playoffs_df.index, 'Count': playoffs_df.values})
    stats_df = df[df.columns[4:]]
    stats_df = stats_df.drop(['g', 'mp'], axis=1)
    stats_df.playoffs = stats_df.playoffs.replace([True, False], [1, 0])
    global df2
    df2 = stats_df
    pl_stats_df = pd.DataFrame(MinMaxScaler(feature_range=(0, 100)).fit_transform(stats_df), columns=stats_df.columns)
    pl_stats_df = pl_stats_df.groupby('playoffs').mean()    
    pl_stats_nop = pl_stats_df.drop(['fg_percent', 'x3p_percent', 'x2p_percent', 'ft_percent'], axis=1)
    return render_template('csv.html', text = stats_df)

@auth.route("/trial", methods = ["GET", "POST"])
def trial():
    global df2
    return render_template('csv.html', text = df2)

@auth.route("/modelTrain", methods = ["GET", "POST"])
def modelTrain():
    global df2
    inputDataSet = df2.drop(columns = ["playoffs"])
    outDataSet = df2["playoffs"]
    X_train, X_test, Y_train, Y_test = train_test_split( inputDataSet, outDataSet, test_size=0.2)
    logit=linear_model.LogisticRegression(C=1e10,max_iter=1e5)
    logit.fit(X_train,Y_train)
    print('Logistic Regression Accuracy Score on Test data:', logit.score(X_test,Y_test))
    print('Logistic Regression Accuracy Score on train data:', logit.score(X_train,Y_train))
    
    predictions = logit.predict(X_test)
    print(confusion_matrix(Y_test,predictions))
    print(classification_report(Y_test,predictions))
    # logit.save("firstTry.h5")
    with open('my_model.pkl', 'wb') as f:
        pickle.dump(logit, f)
    return render_template('csv.html', text = predictions)


@auth.route("/modelPredict", methods = ["GET", "POST"])
def modelPredict():
    global df2
    df2 = df2.drop(columns = ["playoffs"])
    rowTry = df2.iloc[0]
    # rowTry["fg_percent"]  = 50
    # rowTry["fg"] = 2
    # rowTry["ast"] = 100
    rowTry = [rowTry]
    with open('my_model.pkl', 'rb') as file:
        model = pickle.load(file)
    prediction = model.predict(rowTry)
    print(prediction)
    return render_template('csv.html', text = prediction)