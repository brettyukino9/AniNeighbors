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


map = {}
username = "brettyoshi9"

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
# Query to get all following 
# {
#   Page(page: 1) {
#     pageInfo {
#       lastPage
#     }
#     following(userId: 242203) {
#       name
#     }
#   }
# }

AnimeLists = {}
global_user_anime_count = -1

def makeUserDFFromResponse(response, userId):
    global global_user_anime_count
    list_owner = response['data']['User']['name']
    anime_count = 0
    for animelist in response['data']['MediaListCollection']['lists']:
        anime_count += len(animelist['entries'])
    if(global_user_anime_count and anime_count < global_user_anime_count * 0.15):
        return -1
    userList = pd.DataFrame(columns='userid, title, media_id, score, status'.split(', '))
    for animelist in response['data']['MediaListCollection']['lists']:
        for anime in animelist['entries']:
            if(anime['status'] == "PLANNING"):
                continue
            if(anime['scoreRaw'] == 0):
                continue
            userList.loc[anime['mediaId']] = [userId, anime['media']['title']['romaji'], anime['mediaId'], anime['scoreRaw'], anime['status']]
    # if(userId != global_user_id):
    if(userId == global_user_id):
        global_user_anime_count = anime_count
    AnimeLists[list_owner] = userList
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


for i in range(1, 3) :
    getUserListFromAPI(i)
    

print(AnimeLists.keys())

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

stats = pd.DataFrame(columns='name, anime_count, merged mean, reduces mean by, hoh, pearson, shared10s'.split(', '))
for username in AnimeLists:
    list = AnimeLists[username]
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
    print(shared_10s, total_10s)

    # Find the users new mean after merging
    user_list_new_average = merged_data['score_y'].mean()
    mean_diff = user_list_old_average - user_list_new_average

    ratings_df = merged_data[['score_x', 'score_y']]
    pearson = ratings_df.corr(method='pearson')['score_x']['score_y']
    print(pearson)
    stats.loc[list_owner] = [list_owner, anime_count, meanscore, mean_diff, 0, pearson, shared_10s]
    print(stats)
