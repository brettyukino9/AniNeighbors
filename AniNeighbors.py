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

username = dict(config.items('USERNAME'))['username']
print(username)

# Make query to get user Id
queryUserId = '''
query($name:String){User(name:$name){id}}
'''

# Define our query variables and values that will be used in the query request
variablesUserId = {
    'name': username
}

url = 'https://graphql.anilist.co'

# Make the HTTP Api request to get the user id of the username
response = requests.post(url, json={'query': queryUserId, 'variables': variablesUserId}).json()
global_user_id = response['data']['User']['id']

# Define our query variables and values that will be used in the query request
variables = {
    'userId': global_user_id
}

queryUserList = '''
query {
  MediaListCollection(userName: "USERNAME") {
    lists {
      entries {
        media {
          title {
            romaji
          }
        }
      }
    }
  }
}

'''

AnimeLists = {}
global_user_anime_count = -1

def makeUserDFFromResponse(response, userId):
    global global_user_anime_count
    user = response['data']['User']
    if(user == None):
        return -1
    list_owner = user['name']
    print(list_owner, userId)
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
        userList['zscore'] = (userList['score_x'] - average) / stdev
        userList = userList.sort_values(by='zscore', ascending=False)
        normalizedUserList = userList[userList['zscore'] > 0.85]
        for row in rows :
            total_10s += 1
            anime = row[0]
            # Search for an entry of anime in any entry of title_x
            for index, row in normalizedUserList.iterrows():
                if(anime.lower() in row['title_x'].lower()):
                    shared_10s += 1
                    break
    
    return shared_10s, total_10s

def get_anime_count_score(stats_df, scores_df):
    # With 800 anime
    # Should be aiming for 250-500 which is 0.3125 - 0.625

    # With 200 anime
    # should be aiming for 60-120 which is 0.3 - 0.6

    # Calculate count_percentage based on the anime_count column
    stats_df['count_percentage'] = stats_df['anime_count'] / global_user_anime_count

    # # Map the input value to the desired range using a non-linear function based on 0.6 = 8 and 0.3 = 2
    scores_df['anime_count_score'] = stats_df['count_percentage'].apply(lambda x: min(-1.66666666 * x + 27.77777* pow(x, 2), 10) * float(weights['shared_count_weight']))
    
    # # Drop the intermediate 'count_percentage' column
    print(stats_df)
    scores_df = stats_df.drop('count_percentage', axis=1)

    return scores_df

def get_shared_10s_score(stats_df, scores_df):
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
    get_pearson_score(stats_df, scores_df)
    scores_df['final_score'] = scores_df['anime_count_score'] + scores_df['shared_10s_score'] + scores_df['global_user_mean_diff_score'] + scores_df['user_mean_diff_score'] + scores_df['pearson_score']
    scores_df = scores_df.sort_values(by='final_score', ascending=False)
    print(scores_df)
    return scores_df

user_data = pd.read_csv("UserData/data.csv")
user_data.columns = ['userid', 'title', 'media_id', 'score', 'status']
print("user data", user_data)
unique_users = user_data['userid'].unique()
print(unique_users)
AnimeListsFromFile = {}
for user in unique_users:
    # Create a DataFrame filtered by the unique value
    filtered_df = user_data[user_data['userid'] == user]
    
    # Store the filtered DataFrame in the dictionary
    AnimeListsFromFile[user] = filtered_df

stats = pd.DataFrame(columns='name, anime_count, merged mean, user mean diff, reduces mean by, hoh, pearson, shared10s'.split(', '))
for username in AnimeListsFromFile:
    
    list = AnimeListsFromFile[username]
    list_owner = username

    # Get the average of the user BEFORE merging
    user_list_old_average = userList['score'].mean()

    # Attach scores from userList that match the mediaId of the filtered_responses
    merged_data = pd.merge(list, userList, on='media_id', how='left')

    # Remove NaNs
    merged_data = merged_data.dropna()

    # Remove Duplicates
    merged_data = merged_data.drop_duplicates(subset='media_id', keep='first')

    # Sort by title_x
    merged_data = merged_data.sort_values(by='score_y')
    print(merged_data)

    # Find the average of all scores that are above 0
    meanscore = merged_data['score_x'].mean()

    # Find the number of anime the user has watched
    anime_count = len(merged_data)

    # Find the amount of shared 10s
    shared_10s, total_10s = getShared10s(merged_data)

    # Find the users new mean after merging
    user_list_new_average = merged_data['score_y'].mean()
    mean_diff = user_list_old_average - user_list_new_average
    user_mean_diff = user_list_new_average - meanscore

    # Find the pearson coefficient
    ratings_df = merged_data[['score_x', 'score_y']]
    pearson = ratings_df.corr(method='pearson')['score_x']['score_y']

    stats.loc[list_owner] = [list_owner, anime_count, meanscore, user_mean_diff, mean_diff, 0, pearson, shared_10s / total_10s]
print(stats)
scores = calculate_scores(stats)
print(scores)

# write the scores to a csv
scores.to_csv('Results/scores.csv', index=False)
