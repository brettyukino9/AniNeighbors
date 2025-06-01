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
import cProfile
import pstats
import io
import configparser
import MySQLdb
import time
import numpy as np

def init():
    global db_host
    db_host = "localhost"
    global db_user
    db_user = "root"
    global db_password
    db_password = ""
    global db_name
    db_name = "anineighbors"

    config = configparser.RawConfigParser()
    config.read('config.cfg')
    global weights
    weights = dict(config.items('WEIGHTS'))
    print(weights)
    global max_score
    max_score = (float(weights['pearson_weight']) + float(weights['shared_count_weight']) + float(weights['shared_10s_weight']) + float(weights['hoh_weight'])) * 10
    print(max_score)
    global global_username
    global_username = dict(config.items('USERNAME'))['username']
    print(global_username)
    global users_file
    users_file = dict(config.items('OPTIONS'))['users_file']
    print(users_file)
    global options
    options = dict(config.items('OPTIONS'))
    global calculate_all
    calculate_all = options['calculate_all']

    global url
    url = 'https://graphql.anilist.co'

    global global_user_anime_count
    global_user_anime_count = -1

def get_global_user_list():
    global_user_id = get_user_id_from_username(global_username)
    userList = getUserListFromAPI(global_user_id)
    return userList

def calculate_user_stats_from_file(userList):
 # Read in all the user data
    user_data = pd.read_csv("DataSets/{file}".format(file=users_file))
    user_data.columns = ['userid', 'title', 'media_id', 'score', 'status']
    print("user data", user_data)

    unique_users = user_data['userid'].unique()
    print(unique_users)

    # skip users already calculating to not be high; TAKE THIS OUT?
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
    every_n_users = int(options['every_n_users'])

    print("First turning all user CSV data into Anime Lists:")
    # Convert the CSV to a bunch of anime lists
    # Filter out users in low_users_list
    filtered_users = user_data[~user_data['userid'].isin(low_users_list)]

    # If every_n_users > 1, filter users to process every nth user
    if every_n_users > 1:
        filtered_users = filtered_users[filtered_users['userid'].apply(lambda x: unique_users.tolist().index(x) % every_n_users == 0)]

    # Group by user and create a dictionary of DataFrames
    AnimeListsFromFile = {user: df for user, df in filtered_users.groupby('userid')}

    stats = pd.DataFrame(columns='name, anime_count, merged mean, user mean diff, reduces mean by, hoh, pearson, shared10s, brett'.split(', '))
    print(f"Calculating the stats for {len(AnimeListsFromFile)} users")
    start = time.time()
    user_count = 0
    for username in AnimeListsFromFile: 
        user_count += 1
        get_neighbor_stats_from_list(stats, AnimeListsFromFile[username], userList, username)
        if(user_count % 10 == 0):
            end = time.time()
            avg = (end - start) / user_count
            print("time left: ", avg * (len(AnimeListsFromFile) - user_count) / 60, " minutes")  

    print(stats)
    scores = calculate_scores(stats)
    print(scores)

    # normalize final score based on max score
    scores['final_score'] = scores['final_score'] / max_score * 100
    print(scores)

    # save a list of low scoring users so you can save time ignoring them next time its run
    low_users = scores[scores['final_score'] < 20]
    print(low_users)

    if(file_exists) :
        low_users.to_csv(f'Results/{global_username}_low_users.csv', mode='a', index=False, header=False)
    else:
        low_users.to_csv(f'Results/{global_username}_low_users.csv', index=False)

    import matplotlib.pyplot as plt

    # make a histogram with the final scores
    # plt.hist(scores['final_score'], bins=50, range=[0,100])
    # plt.xlabel('Final Score')
    # plt.ylabel('Number of Users')
    # plt.title('Final Score Distribution')
    # plt.show()

    ## TO DO: other metrics?? favorites lists?? 

    # write the scores to a csv
    scores.to_csv(f'Results/{global_username}_scores.csv', index=False)

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
    if global_user_anime_count == -1: # GLOBAL USER MUST BE THE FIRST ONE TO ENTER THIS FUNCTION; cannot call insert on it's own without fixing this
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
    if response == None or response['data'] == None or response['data']['User'] == None:
        print("Error3: No response from API for userId", userId)
        return None
    userList = makeUserDFFromResponse(response, userId)
    return userList        

def getShared10s(userList):
    path = "Shared10s Lists/{username}.txt".format(username=global_username)
    if not os.path.exists(path): # if a shared 10s file does not exist, use the default
        path = "shared 10s.csv"
    with open(path, newline="", encoding='latin1') as csvfile:
        rows = csv.reader(csvfile, delimiter=",", dialect="excel")
        tens_list_lower = [row[0].lower() for row in rows]
        shared_10s = 0
        total_10s = len(tens_list_lower)
        userList.sort_values(by='score_x')
        average = userList['score_x'].mean()
        stdev = userList['score_x'].std()
        userList['zscore_x'] = (userList['score_x'] - average) / stdev
        userList = userList.sort_values(by='zscore_x', ascending=False)
        normalizedUserList = userList[userList['zscore_x'] > 0.9]
        
        unique_counts = {substr: (normalizedUserList["title_x"].str.lower().str.contains(substr)).any(axis=0).sum() for substr in tens_list_lower}
        # return how many counts are greater than 0
        for key in unique_counts:
            if unique_counts[key] > 0:
                shared_10s += 1
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

def get_brett_value(userList):
    # compare the two users by iterating through every anime. check if anime one is rated higher than anime two for a user. check if the other users agrees or disagrees. get the percentage of matches.
    agrees = 0
    disagrees = 0
    for index, row in userList.iterrows():
        for index2, row2 in userList.iterrows():
            if(index2  <= index):
                continue
            user1_score_diff = row['score_x'] - row2['score_x']
            user2_score_diff = row['score_y'] - row2['score_y']
            if(user1_score_diff > 0 and user2_score_diff > 0):
                agrees+=1
            elif(user1_score_diff < 0 and user2_score_diff < 0):
                agrees+=1
            elif(user1_score_diff == 0 and user2_score_diff == 0):
                agrees+=1
            else:
                disagrees+=1
    print("agrees", agrees)
    print("disagrees", disagrees)
    return agrees / (agrees + disagrees)

def get_brett_value_new(userList):
    user_diff_df = userList['score_x'].values[:, None] - userList['score_x'].values
    user_diff_df = pd.DataFrame(user_diff_df, index=userList.index, columns=userList.index)
    binary_user_df = (user_diff_df >= 0).astype(int)
    
    neighbor_diff_df = userList['score_y'].values[:, None] - userList['score_y'].values
    neighbor_diff_df = pd.DataFrame(neighbor_diff_df, index=userList.index, columns=userList.index)
    binary_neighbor_df = (neighbor_diff_df >= 0).astype(int)

    combined_xor_df = binary_user_df ^ binary_neighbor_df # if both positive 0 or if both negative 0 (agree), if one positive and one negative 1 (disagree)
    total_ones = np.sum(combined_xor_df.values == 1)
    total_zeros = np.sum(combined_xor_df.values == 0)
    
    # Because both 0 and positive are 1, it counts it as an agree when one is 0 and the other is positive. 
    # To fix this, find the amount of 0s and positive and subtract these from the totals at the end.
    binary_user_df_zero = (user_diff_df == 0).astype(int) # 1 if zero
    binary_user_df_positive = (user_diff_df > 0).astype(int) # 1 if positive
    binary_neighbor_df_zero = (neighbor_diff_df == 0).astype(int) # 1 if zero
    binary_neighbor_df_positive = (neighbor_diff_df > 0).astype(int) # 1 if positive
    zero_df_1 = binary_user_df_zero & binary_neighbor_df_positive # 1 is all the bugged ones
    zero_df_2 = binary_user_df_positive & binary_neighbor_df_zero # 1 is all the bugged ones
    bugs = (np.sum(zero_df_1.values == 1) + np.sum(zero_df_2.values == 1)) / 2

    agrees = (total_zeros - len(userList)) / 2 - bugs
    disagrees = (total_ones) / 2 + bugs 
    percentage = agrees / (agrees + disagrees)
    return percentage
    

def get_anime_count_score(stats_df, scores_df):
    # With 800 anime
    # Should be aiming for 200 - 500 which is 0.25 - 0.625

    # With 200 anime
    # should be aiming for 60-120 which is 0.3 - 0.6
    print("global user anime count", global_user_anime_count)
    print("stats df", stats_df['anime_count'])
    # Calculate count_percentage based on the anime_count column
    stats_df['count_percentage'] = stats_df['anime_count'] / global_user_anime_count
    print(stats_df['count_percentage'])
    # # Map the input value to the desired range using a non-linear function based on
    # 0.15 - 1
    # 0.3 - 5
    # 0.6 - 10
    scores_df['anime_count_score'] = stats_df['count_percentage'].apply(lambda x: min(36.6666 * x +  -22.22222 * pow(x, 2) - 4, 10) * float(weights['shared_count_weight']))
    print(scores_df['anime_count_score'])
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

def insert_scores_into_db(scores_df):
    scores_df['final_score'] = scores_df['anime_count_score'] + scores_df['shared_10s_score'] + scores_df['global_user_mean_diff_score'] + scores_df['user_mean_diff_score'] + scores_df['hoh_score'] + scores_df['pearson_score']
    scores_df = scores_df.sort_values(by='final_score', ascending=False)
    for index, row in scores_df.iterrows():
        # Insert the neighbor into the database
        insert_neighbor_into_db(row['name'], row['final_score'], row['anime_count_score'], row['shared_10s_score'], row['hoh_score'], row['pearson_score'], row['global_user_mean_diff_score'])

def calculate_scores(stats_df):
    scores_df = pd.DataFrame()
    scores_df['name'] = stats_df['name']
    get_anime_count_score(stats_df, scores_df)
    get_shared_10s_score(stats_df, scores_df)
    get_mean_score_scores(stats_df, scores_df)
    get_hoh_score(stats_df, scores_df)
    get_pearson_score(stats_df, scores_df)
    insert_scores_into_db(scores_df)
    return scores_df

def create_database_structure():
    connection = None
    try:
        connection = MySQLdb.connect(
            host=db_host,
            user=db_user,
            password=db_password
        )
        cursor = connection.cursor()
        cursor.execute("CREATE DATABASE IF NOT EXISTS anineighbors;")
        print("Database created successfully!")
    except MySQLdb.Error as e:
        print(f"Error: {e}")
    finally:
        if connection:
            connection.close()
    
    try:
        connection = MySQLdb.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = connection.cursor()
        
        # Create the user_neighbors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_neighbors (
                id INT AUTO_INCREMENT PRIMARY KEY,
                reference_username VARCHAR(50) NOT NULL,
                neighbor_username VARCHAR(50) NOT NULL,
                similarity_score FLOAT NOT NULL,
                anime_count INT,
                shared_10s_ratio FLOAT,
                hoh_value FLOAT,
                pearson_coefficient FLOAT,
                mean_score_diff FLOAT,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                
                # Create a unique constraint to prevent duplicate neighbor relationships
                UNIQUE KEY unique_neighbor (reference_username, neighbor_username)
            );
        """)
        
        connection.commit()
        print("User neighbors table created successfully!")
        
    except MySQLdb.Error as e:
        print(f"Error creating neighbors table: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

def insert_neighbor_into_db(neighbor_username, similarity_score, anime_count, shared_10s_ratio, hoh_value, pearson_coefficient, mean_score_diff):
    connection = None
    normalized_similarity_score = similarity_score / max_score * 100
    try:
        connection = MySQLdb.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = connection.cursor()
        print(f"inserting user {neighbor_username} with score {normalized_similarity_score}")
        # Insert the neighbor into the database
        cursor.execute("""
            INSERT INTO anineighbors.user_neighbors (reference_username, neighbor_username, similarity_score, anime_count, shared_10s_ratio, hoh_value, pearson_coefficient, mean_score_diff)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                similarity_score = VALUES(similarity_score),
                anime_count = VALUES(anime_count),
                shared_10s_ratio = VALUES(shared_10s_ratio),
                hoh_value = VALUES(hoh_value),
                pearson_coefficient = VALUES(pearson_coefficient),
                mean_score_diff = VALUES(mean_score_diff),
                last_updated = CURRENT_TIMESTAMP
        """, (global_username, neighbor_username, normalized_similarity_score, anime_count, shared_10s_ratio, hoh_value, pearson_coefficient, mean_score_diff))
        
        connection.commit()
    except MySQLdb.Error as e:
        print(f"Error inserting into database: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

def show_neighbors_in_db():
    connection = None
    try:
        connection = MySQLdb.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = connection.cursor()
        
        # Select all neighbors for the given user
        cursor.execute("""
            SELECT * FROM anineighbors.user_neighbors
            WHERE reference_username = %s
            ORDER BY similarity_score DESC
            LIMIT 100;
        """, (global_username,))
        
        results = cursor.fetchall()
        with open(f"Results/{global_username}_neighbors.csv", 'w', newline='') as csvfile:
            for row in results:
                print(row)
                csvfile.write(','.join(map(str, row)) + '\n')
    except MySQLdb.Error as e:
        print(f"Error fetching from database: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

def get_neighbor_stats_from_list(stats_df, neighbor_list, user_list, neighbor_name):
    # print("doing list", list_owner)
    # Get the average of the user BEFORE merging
    user_list_old_average = user_list['score'].mean()
    
    # Attach scores from userList that match the mediaId of the filtered_responses
    merged_data = pd.merge(neighbor_list, user_list, on='media_id', how='left')

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

    
    start = time.time()
    # brett_value = get_brett_value(merged_data)
    # end = time.time()
    # print("got brett value {} in {} seconds".format(brett_value, end - start))
    # start = time.time()
    # brett_value = get_brett_value_new(merged_data)
    # end = time.time()
    # print("got brett value2 {} in {} seconds".format(brett_value, end - start))
    brett_value = 0

    
    # Find the users new mean after merging
    user_list_new_average = merged_data['score_y'].mean()
    mean_diff = user_list_old_average - user_list_new_average
    user_mean_diff = user_list_new_average - meanscore

    # Find the pearson coefficient
    ratings_df = merged_data[['score_x', 'score_y']]
    pearson = ratings_df.corr(method='pearson')['score_x']['score_y']
    stats_df.loc[neighbor_name] = [neighbor_name, anime_count, meanscore, user_mean_diff, mean_diff, hoh_value, pearson, shared_10s / total_10s, brett_value]


def get_user_id_from_username(username):
    # Make query to get user Id
    url = 'https://graphql.anilist.co'
    queryUserId = '''
    query($name:String){User(name:$name){id}}
    '''
    variablesUserId = {
        'name': username.strip()
    }
    response_raw = requests.post(url, json={'query': queryUserId, 'variables': variablesUserId})
    response_json = response_raw.json()
    print("response", response_json)
    if response_json is None or response_json['data'] is None or response_json['data']['User'] is None:
        if response_json and response_json['errors'][0]['message'] == "Too Many Requests.":
            while(True):
                print("Too many requests, sleeping")
                time.sleep(30)
                response_raw = requests.post(url, json={'query': queryUserId, 'variables': variablesUserId})
                response_json = response_raw.json()
                if response_json is not None and response_json['data'] is not None and response_json['data']['User'] is not None:
                    break
                elif response_json and response_json['errors'][0]['message'] != "Too Many Requests.":
                    print("Error: No response from API for userId", username)
                    return None

        else:
            print("Error4: No response from API for userId", username)
            return None
    userid = response_json['data']['User']['id']
    return userid

def get_user_api_stats_and_insert(username, userList):
    userid = get_user_id_from_username(username)
    if userid is None:
        print("Error1: No response from API for user", username)
        return None
    print("user id for user ", username, "is", userid)
    time.sleep(2.5)

    # get list of user
    list = getUserListFromAPI(userid)
    if list is None:
        print("Error2: No response from API for user", username)
        return None
    print(list)

    # get user stats and add them
    stats_df = pd.DataFrame(columns='name, anime_count, merged mean, user mean diff, reduces mean by, hoh, pearson, shared10s, brett'.split(', '))
    get_neighbor_stats_from_list(stats_df, list, userList, username)
    calculate_scores(stats_df)

# add all of the users this user follows to the database
def expand_from_user(userList, username):
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
    user_id = get_user_id_from_username(username)
    if user_id is None:
        print("Error1: No response from API for user", username)
        return None
    for i in range(1, 20):
        variables = {
            'userId': user_id,
            'page': i
        }
        print("page ", i)
        time.sleep(2.5)
        response = requests.post(url, json={'query': queryAllFollowing, 'variables': variables}).json()
        last_page = response['data']['Page']['pageInfo']['lastPage']
        if i > last_page:
            break
        for user in response['data']['Page']['following']:
            time.sleep(2)
            following_user_id = user['id']
            following_username = user['name']
            list = getUserListFromAPI(following_user_id)
            if list is None:
                print("Error2: No response from API for user", following_username)
                return None
            print(list)

            # get user stats and add them
            stats_df = pd.DataFrame(columns='name, anime_count, merged mean, user mean diff, reduces mean by, hoh, pearson, shared10s, brett'.split(', '))
            get_neighbor_stats_from_list(stats_df, list, userList, following_username)
            calculate_scores(stats_df)

def expand_top_100(userList):
    connection = None
    try:
        connection = MySQLdb.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = connection.cursor()
        
        # Select all neighbors for the given user
        cursor.execute("""
            SELECT * FROM anineighbors.user_neighbors
            WHERE reference_username = %s
            ORDER BY similarity_score DESC
            LIMIT 200;
        """, (global_username,))
        
        results = cursor.fetchall()
        found = False
        for row in results:
            username = row[2]
            if username == "brettyoshi9" or username == "hyogotou" or username == "higui" or username == "Problem02" or username == "MrRaindropDa":
                continue
            if username == "RideIsAfraid":
                found = True
            if not found:
                continue
            
            print("expanding user", username)
            expand_from_user(userList, username)
    except MySQLdb.Error as e:
        print(f"Error fetching from database: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()    

def top_100_brett(userList):
    connection = None
    try:
        connection = MySQLdb.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = connection.cursor()
        
        # Select all neighbors for the given user
        cursor.execute("""
            SELECT * FROM anineighbors.user_neighbors
            WHERE reference_username = %s
            ORDER BY similarity_score DESC
            LIMIT 200;
        """, (global_username,))
        
        results = cursor.fetchall()
        with open(f"Results/{global_username}_top_100_brett_values.csv", 'w', newline='') as csvfile:
            for row in results:
                username = row[2]
                print("getting brett value of user", username)
                userid = get_user_id_from_username(username)
                if userid is None:
                    print("Error1: No response from API for user", username)
                    return None
                print("user id for user ", username, "is", userid)
                time.sleep(2.5)

                # get list of user
                neighbor_list = getUserListFromAPI(userid)
                if neighbor_list is None:
                    print("Error2: No response from API for user", username)
                    return None
                print(neighbor_list)
                merged_data = pd.merge(neighbor_list, userList, on='media_id', how='left')

                # Remove NaNs
                merged_data = merged_data.dropna()

                # Remove Duplicates
                merged_data = merged_data.drop_duplicates(subset='media_id', keep='first')

                # Sort by title_x
                merged_data = merged_data.sort_values(by='score_y')

                brett_value = get_brett_value_new(merged_data)
                anime_count = len(merged_data)
                print("brett value for user", username, "is", brett_value)
                csvfile.write(f"{username},{anime_count},{brett_value}\n")


    except MySQLdb.Error as e:
        print(f"Error fetching from database: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()    

def update_top_scores(userList):
    connection = None
    try:
        connection = MySQLdb.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name
        )
        cursor = connection.cursor()
        
        # Select all neighbors for the given user
        cursor.execute("""
            SELECT * FROM anineighbors.user_neighbors
            WHERE reference_username = %s
            ORDER BY similarity_score DESC
            LIMIT %s;
        """, (global_username, int(options['number_top_scores_to_update'])))
        
        results = cursor.fetchall()
        recommendations = pd.DataFrame(columns=['Anime Name'])
        user_count = 0
        user_titles = userList['title'].tolist()
        for row in results:
            username = row[2]
            if username == global_username or username == "hyoghotou" or username == "higui":
                continue
            print("updating user", username)
            userid = get_user_id_from_username(username)
            if userid is None:
                print("Error1: No response from API for user", username)
                continue
            print("user id for user ", username, "is", userid)
            time.sleep(2.5)

            # get list of user
            list = getUserListFromAPI(userid)
            if list is None:
                print("Error2: No response from API for user", username)
                continue
            user_count += 1

            # get user stats and add them
            stats_df = pd.DataFrame(columns='name, anime_count, merged mean, user mean diff, reduces mean by, hoh, pearson, shared10s, brett'.split(', '))
            get_neighbor_stats_from_list(stats_df, list, userList, username)
            calculate_scores(stats_df)

            titles = list['title'].tolist()
            current_titles = recommendations['Anime Name'].tolist()
            titles = [title for title in titles if title not in current_titles] # Exclude titles already in recommendations
            titles = [title for title in titles if title not in user_titles]  # Exclude titles already in user's list
            new_rows = []
            for title in titles:
                    new_rows.append({'Anime Name': title})            
            if new_rows:
                recommendations = pd.concat([recommendations, pd.DataFrame(new_rows)], ignore_index=True)
            score_map = dict(zip(list['title'], list['score']))
            recommendations[username] = recommendations['Anime Name'].map(score_map) # This will create a new column for each user with their scores for the anime in recommendations
        
        recommendations['Average Score'] = recommendations.iloc[:, 1:(user_count+1)].mean(axis=1)
        recommendations['Count'] = recommendations.iloc[:, 1:(user_count+1)].count(axis=1)
        recommendations['Median Score'] = recommendations.iloc[:, 1:(user_count+1)].median(axis=1)
        recommendations[' > 80 %'] = recommendations.iloc[:, 1:(user_count+1)].apply(lambda x: (x >= 80).sum(), axis=1) / recommendations['Count']
        recommendations[' > 90 %'] = recommendations.iloc[:, 1:(user_count+1)].apply(lambda x: (x >= 90).sum(), axis=1) / recommendations['Count']
        recommendations['100 %'] = recommendations.iloc[:, 1:(user_count+1)].apply(lambda x: (x >= 99).sum(), axis=1) / recommendations['Count']
        recommendations['Council Score'] = recommendations[' > 80 %'] * 6.4 + recommendations[' > 90 %'] * 7.8 + recommendations['100 %'] * 5.4 + recommendations['Average Score'] / 10 + recommendations['Median Score'] / 10
        recommendations = recommendations.sort_values(by='Council Score', ascending=False)
        recommendations = recommendations[recommendations['Count'] >= (user_count * 0.05)] # only keep recommendations with at least 5% of the users rating it
        recommendations = recommendations[recommendations['Count'] > 2]
        cols_to_front = [
            'Council Score',
            'Average Score',
            'Median Score',
            'Count',
            ' > 80 %',
            ' > 90 %',
            '100 %'
        ]
        for i, col in enumerate(cols_to_front):
            col_data = recommendations.pop(col)
            recommendations.insert(i + 1, col, col_data)  # +1 to keep 'Anime Name' as the first column

        recommendations.to_csv(f'Results/{global_username}_recommendations.csv', index=False)
    except MySQLdb.Error as e:
        print(f"Error fetching from database: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()

def main():
    profile = cProfile.Profile()
    profile.enable()
    ## TO DO ADD SHARED 10s to config

    init()
    userList = get_global_user_list()
    create_database_structure()

    if options['batch_file'] == "True":
        calculate_user_stats_from_file(userList)

    if options['api_single'] == "True":
        user_to_add = options['api_username_to_add']
        get_user_api_stats_and_insert(user_to_add, userList)
    
    if options['update_top_scores'] == "True":
        update_top_scores(userList)

    if options['expand_top_100'] == "True":
        expand_top_100(userList)
    
    if options['expand_from_user'] == "True":
        user_to_expand = options['expand_username']
        expand_from_user(userList, user_to_expand)
    
    if options['calc_brett_value'] == "True":
        top_100_brett(userList)

    
    show_neighbors_in_db()

   

    profile.disable()
    s = io.StringIO()
    ps = pstats.Stats(profile, stream=s).sort_stats('tottime')
    ps.print_stats()

    with open('profile.txt', 'w+') as f:
        f.write(s.getvalue())

    sys.exit()


if __name__ == "__main__":
    main()