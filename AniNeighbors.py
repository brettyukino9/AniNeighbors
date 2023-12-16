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


map = {}
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
print(response)
userId = response['data']['User']['id']

# Define our query variables and values that will be used in the query request
variables = {
    'userId': userId
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

print(userId)

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
def makeUserDFFromResponse(response):
    list_owner = response['data']['User']['name']
    anime_count = 0
    for animelist in response['data']['MediaListCollection']['lists']:
        anime_count += len(animelist['entries'])
    print(anime_count)
    if(anime_count < 10):
        return -1
    userList = pd.DataFrame(columns='title, media_id, score, status'.split(', '))
    for animelist in response['data']['MediaListCollection']['lists']:
        for anime in animelist['entries']:
            if(anime['status'] == "PLANNING"):
                continue
            userList.loc[anime['mediaId']] = [anime['media']['title']['romaji'], anime['mediaId'], anime['scoreRaw'], anime['status']]
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
    makeUserDFFromResponse(response)
    return response        

for i in range(1, 3) :
    getUserListFromAPI(i)
    

print(AnimeLists.keys())

stats = pd.DataFrame(columns='name, anime_count, meanscore, hoh, pearson, shared10s'.split(', '))
for username in AnimeLists:
    list = AnimeLists[username]
    print(f"{username} response{list}")
    # Get the amount of anime that don't have the status of PLANNING
    list_owner = username

    filtered_responses = list[list['score'] > 0]
    
    # Find the average of all scores that are above 0
    meanscore = filtered_responses['score'].mean()
    anime_count = len(list)
    print(meanscore)
    stats.loc[list_owner] = [list_owner, anime_count, meanscore, 0, 0, 0]
    print(stats)

                     