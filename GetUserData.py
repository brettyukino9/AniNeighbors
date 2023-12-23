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
import time

import configparser

with open('UserData/data.csv', 'w', newline='') as file:
    file.write('')

config = configparser.RawConfigParser()
config.read('config.cfg')

weights = dict(config.items('WEIGHTS'))
print(weights)

username = "higui"

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


## FILL ALL USER LIST DATA

global_user_anime_count = -1

def makeUserDFFromResponse(response, userId):
    global global_user_anime_count
    if(response['data'] == None):
        return -1
    user = response['data']['User']
    if(user == None):
        return -1
    list_owner = user['name']
    print(list_owner, userId)
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
            userList.loc[anime['mediaId']] = [list_owner, anime['media']['title']['romaji'], anime['mediaId'], anime['scoreRaw'], anime['status']]
    # if(userId != global_user_id):
    if(userId == global_user_id):
        global_user_anime_count = len(userList)
    if(len(userList) > 0):
        userList.to_csv('UserData/data.csv', mode='a', header=False, index=False)
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


queryAllFollowing = '''
query ($page: Int, $userId: Int!) {
    Page(page: $page) {
        pageInfo {
        lastPage
        }
        following(userId: $userId) {
        name
        id
        }
    }
    }
'''
# for i in range(1, 101):
#     variables = {
#         'userId': global_user_id,
#         'page': i
#     }
#     print("page ", i)
#     response = requests.post(url, json={'query': queryAllFollowing, 'variables': variables}).json()
#     for user in response['data']['Page']['following']:
#         getUserListFromAPI(user['id'])


# Scrape User data
users = [5996645, 300309, 452263]
for userid in users:
    getUserListFromAPI(userid)

# print(AnimeLists.keys())

## FILL ALL USER LIST DATA END
