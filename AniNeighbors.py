# Get the hoh value for all of following
## OR get ALL anilist users (iterate through every user id), 
## with 90 requests a minute this is 130k people in 24 hours
## filter by some number (top 10000?)
## 
# To speed up the process, skip users who don't have a lot of ratings (don't get their scores)
# Get the top x% of users (make this adjustable)
# Use the number of shared anime to get a new ranking
# Get all the ratings for each user
# Get the # shared 10s for each user based on an input file of 10/10s to check for. Adjust how much this adds based on how many 10s are in the input file.
# Calculate the pearson coefficient for each user
# Get each users average score
# Get my average score with that user
# Get my average score on my profile

# USE ALL OF THESE TO COMPUTE FINAL RANKING. Maybe make the weights adjustable?? 
# be able to sort by different metrics? show a histogram of rating distribution


import csv
import json
import os
import webbrowser
from pip._vendor import requests
import sys
import math
import pandas as pd
import csv
import random as random

import configparser

config = configparser.RawConfigParser()
config.read('config.cfg')

weights = dict(config.items('WEIGHTS'))
print(weights)
max_score = (float(weights['pearson_weight']) + float(weights['shared_count_weight']) + float(weights['shared_10s_weight']) + float(weights['hoh_weight'])) * 10
print(max_score)
global_username = dict(config.items('USERNAME'))['username']
print(global_username)

options = dict(config.items('OPTIONS'))
calculate_all = options['calculate_all']

## TO DO ADD SHARED 10s to config

# Make query to get user Id
queryUserId = '''
query($name:String){User(name:$name){id}}
'''

# Define our query variables and values that will be used in the query request
variablesUserId = {
    'name': global_username
}

url = 'https://graphql.anilist.co'

# Make the HTTP Api request to get the user id of the username
response = requests.post(url, json={'query': queryUserId, 'variables': variablesUserId}).json()
global_user_id = response['data']['User']['id']

# Define our query variables and values that will be used in the query request
variables = {
    'userId': global_user_id
}

AnimeLists = {}
global_user_anime_count = -1

def makeUserDFFromResponse(response, userId):
    global global_user_anime_count
    user = response['data']['User']
    if(user == None):
        return -1
    userList = pd.DataFrame(columns='userid, title, media_id, score, status'.split(', '))
    for animelist in response['data']['MediaListCollection']['lists']:
        for anime in animelist['entries']:
            if(anime['status'] == "PLANNING"):
                continue
            if(anime['scoreRaw'] == 0):
                continue
            userList.loc[anime['mediaId']] = [userId, anime['media']['title']['romaji'], anime['mediaId'], anime['scoreRaw'], anime['status']]
    global_user_anime_count = len(userList)
    return userList

def getUserListFromAPI(userId):
    queryUserList = '''
    query ($userId: Int!, $listType: MediaType) {
        MediaListCollection(userId: $userId, type: $listType) {
            lists {
            entries {
                media {
                title {
                    romaji
                }
                }
                mediaId
                status
                scoreRaw: score(format: POINT_100)
            }
            }
        }
        User(id: $userId) {
            name
    }
    }
    '''
    variables = {
        'userId': userId,
        'listType': "ANIME"
    }
    response = requests.post(url, json={'query': queryUserList, 'variables': variables}).json()
    userList = makeUserDFFromResponse(response, userId)
    return userList        

userList = getUserListFromAPI(global_user_id)


def getShared10s(userList):
    with open("shared 10s.csv", newline="", encoding='latin1') as csvfile:
        rows = csv.reader(csvfile, delimiter=",", dialect="excel")
        shared_10s = 0
        total_10s = 0
        userList.sort_values(by='score_x')
        average = userList['score_x'].mean()
        stdev = userList['score_x'].std()
        userList['zscore_x'] = (userList['score_x'] - average) / stdev
        userList = userList.sort_values(by='zscore_x', ascending=False)
        normalizedUserList = userList[userList['zscore_x'] > 0.9]
        for row in rows :
            total_10s += 1
            anime = row[0]
            # Search for an entry of anime in any entry of title_x
            for index, row in normalizedUserList.iterrows():
                if(anime.lower() in row['title_x'].lower()):
                    shared_10s += 1
                    break
    
    return shared_10s, total_10s

def get_hoh_value(userList):
    y_average = userList['score_y'].mean()
    y_stdev = userList['score_y'].std()
    userList['zscore_y'] = (userList['score_y'] - y_average) / y_stdev
    # find the difference in z score for each row and average it
    userList['zscore_diff'] = abs(userList['zscore_y'] - userList['zscore_x'])
    hoh_value = userList['zscore_diff'].mean()
    userList.drop('zscore_diff', axis=1, inplace=True)
    return hoh_value

def get_anime_count_score(stats_df, scores_df):
    # With 800 anime
    # Should be aiming for 200 - 500 which is 0.25 - 0.625

    # With 200 anime
    # should be aiming for 60-120 which is 0.3 - 0.6

    # Calculate count_percentage based on the anime_count column
    stats_df['count_percentage'] = stats_df['anime_count'] / global_user_anime_count

    # # Map the input value to the desired range using a non-linear function based on
    # 0.15 - 1
    # 0.3 - 5
    # 0.6 - 10
    scores_df['anime_count_score'] = stats_df['count_percentage'].apply(lambda x: min(36.6666 * x +  -22.22222 * pow(x, 2) - 4, 10) * float(weights['shared_count_weight']))
    
    # # Drop the intermediate 'count_percentage' column
    scores_df = stats_df.drop('count_percentage', axis=1)

    return scores_df

def get_shared_10s_score(stats_df, scores_df):
        # TO DO : make max shared 10s the max a user gets, not the maximum put in
        scores_df['shared_10s_score'] = stats_df['shared10s'].apply(lambda x: min(10 * x, 10) * float(weights['shared_10s_weight']))

def get_mean_score_scores(stats_df, scores_df):
    # global user mean score diff
    # 0 diff = 0
    # 5 diff = -10
    scores_df['global_user_mean_diff_score'] = stats_df['reduces mean by'].apply(lambda x: max(2 *-abs(x), -10) * float(weights['global_user_avg_diff_weight']))

    # user mean score diff
    # 0 diff = 0
    # 10 diff = -10
    scores_df['user_mean_diff_score'] = stats_df['user mean diff'].apply(lambda x: max(-abs(x), -10) * float(weights['user_avg_diff_weight']))
    

def get_hoh_score(stats_df, scores_df):
    # 0.6 = 10
    # 1 = 0
    scores_df['hoh_score'] = stats_df['hoh'].apply(lambda x: min((-25*x+25),10) * float(weights['hoh_weight']))

def get_pearson_score(stats_df, scores_df):
    # 1 = 10
    # 0 = 0
    # -1 = -10
    scores_df['pearson_score'] = stats_df['pearson'].apply(lambda x: 10 * x * float(weights['pearson_weight']))


def calculate_scores(stats_df):
    scores_df = pd.DataFrame()
    scores_df['name'] = stats_df['name']
    get_anime_count_score(stats_df, scores_df)
    get_shared_10s_score(stats_df, scores_df)
    get_mean_score_scores(stats_df, scores_df)
    get_hoh_score(stats_df, scores_df)
    get_pearson_score(stats_df, scores_df)
    scores_df['final_score'] = scores_df['anime_count_score'] + scores_df['shared_10s_score'] + scores_df['global_user_mean_diff_score'] + scores_df['user_mean_diff_score'] + scores_df['hoh_score'] + scores_df['pearson_score']
    scores_df = scores_df.sort_values(by='final_score', ascending=False)
    return scores_df

user_data = pd.read_csv("DataSets/higui_following_edited.csv")
user_data.columns = ['userid', 'title', 'media_id', 'score', 'status']
print("user data", user_data)

unique_users = user_data['userid'].unique()
print(unique_users)

# skip users already calculating to not be high
low_users_path = f"Results/{global_username}_low_users.csv"
from os.path import exists
file_exists = exists(low_users_path)
low_users_list = []
if(file_exists and calculate_all == "False"):
    low_users = pd.read_csv(f"Results/{global_username}_low_users.csv")
    low_users_list = low_users['name'].tolist()

AnimeListsFromFile = {}

import time
start = time.time()
user_count = 0

for user in unique_users:
    user_count += 1
    if(user in low_users_list):
        continue
    # use this if you want to go through fast and skip users
    # if(user_count % 10 != 0):
    #     continue

    # Create a DataFrame for the anime only from the specific user
    filtered_df = user_data[user_data['userid'] == user]

    # Store the filtered DataFrame in the dictionary
    AnimeListsFromFile[user] = filtered_df
    
    if(user_count % 100 == 0):
        end = time.time()
        avg = (end - start) / user_count
        print("time left: ", avg * (len(unique_users) - user_count) / 60, " minutes")


stats = pd.DataFrame(columns='name, anime_count, merged mean, user mean diff, reduces mean by, hoh, pearson, shared10s'.split(', '))

print(f"Calculating the stats for {len(AnimeListsFromFile)} users")
for username in AnimeListsFromFile:
    user_list = AnimeListsFromFile[username]
    list_owner = username

    # Get the average of the user BEFORE merging
    user_list_old_average = userList['score'].mean()

    # Attach scores from userList that match the mediaId of the filtered_responses
    merged_data = pd.merge(user_list, userList, on='media_id', how='left')

    # Remove NaNs
    merged_data = merged_data.dropna()

    # Remove Duplicates
    merged_data = merged_data.drop_duplicates(subset='media_id', keep='first')

    # Sort by title_x
    merged_data = merged_data.sort_values(by='score_y')

    # Find the average of all scores that are above 0
    meanscore = merged_data['score_x'].mean()

    # Find the number of anime the user has watched
    anime_count = len(merged_data)

    # Find the amount of shared 10s
    shared_10s, total_10s = getShared10s(merged_data)

    # Find the hoh value
    hoh_value = get_hoh_value(merged_data)

    # Find the users new mean after merging
    user_list_new_average = merged_data['score_y'].mean()
    mean_diff = user_list_old_average - user_list_new_average
    user_mean_diff = user_list_new_average - meanscore

    # Find the pearson coefficient
    ratings_df = merged_data[['score_x', 'score_y']]
    pearson = ratings_df.corr(method='pearson')['score_x']['score_y']
    stats.loc[list_owner] = [list_owner, anime_count, meanscore, user_mean_diff, mean_diff, hoh_value, pearson, shared_10s / total_10s]
print(stats)
scores = calculate_scores(stats)
print(scores)

# normalize final score based on max score
scores['final_score'] = scores['final_score'] / max_score * 100
print(scores)

low_users = scores[scores['final_score'] < 20]
print(low_users)
low_users.to_csv(f'Results/{global_username}_low_users.csv', index=False)

import matplotlib.pyplot as plt

# make a histogram with the final scores
plt.hist(scores['final_score'], bins=50, range=[0,100])
plt.show()

## TO DO: other metrics?? favorites lists?? 

# write the scores to a csv
scores.to_csv(f'Results/{global_username}_scores.csv', index=False)



# get the score of the user in the second row
user_score = scores.iloc[1]['final_score']

high_users = scores[scores['final_score'] > user_score * 0.7]
top_users = high_users['name'].tolist()
anime_list = []
for username in top_users:
    user_list = AnimeListsFromFile[username]
    list_owner = username
    anime_list = anime_list + user_list['title'].tolist()

# drop duplicates in anime_list
anime_list = list(set(anime_list))
print(f"{len(anime_list)} unique anime")

# make a users x anime matrix using each username and each unique anime
user_anime_matrix = pd.DataFrame(columns=anime_list)
for username in top_users:
    list = AnimeListsFromFile[username]
    user_anime_matrix.loc[username] = [0] * len(anime_list)
    for index, row in list.iterrows():
        anime = row['title']
        user_anime_matrix.loc[username][anime] = row['score']

# transpose the user anime matrix
user_anime_matrix = user_anime_matrix.T

# turn all the 0s into NaN
user_anime_matrix = user_anime_matrix.replace(0, float('nan'))

# get the z scores for each users anime
user_anime_matrix = user_anime_matrix.apply(lambda x: (x - x.mean()) / x.std(), axis=0)
print(user_anime_matrix)

# calculate stats
user_anime_matrix['mean z-score'] = user_anime_matrix.mean(axis=1)
user_anime_matrix['number of ratings'] = user_anime_matrix.count(axis=1) - 1

# remove lesser seen anime
min_threshold = len(top_users) * 0.3
user_anime_matrix = user_anime_matrix[user_anime_matrix['number of ratings'] > min_threshold]

# remove anime on the users list
user_anime_matrix = user_anime_matrix[~user_anime_matrix.index.isin(userList['title'])]

# move the stats columns to the front
rating_num = user_anime_matrix.pop('number of ratings')
user_anime_matrix.insert(0, 'number of ratings', rating_num)
mean_zscore = user_anime_matrix.pop('mean z-score')
user_anime_matrix.insert(0, 'mean z-score', mean_zscore)

# sort by mean z-score
user_anime_matrix = user_anime_matrix.sort_values(by='mean z-score', ascending=False)

user_anime_matrix.to_csv(f'Results/{global_username}_recommendations.csv', index=True)