# AniNeighbors

Find what similars on AniList have the most similar taste to you

# Running the script
Enter your username into the config. You may also change the weights for each metric. Then run the script.
Make sure to enter your favorites into the "shared_favorites.csv" file. Use romaji titles, one per line, and comma separated. 

The AniNeighbors script currently takes a minute to run with the default file. To run this, set 'batch_file' to True and set 'users_file' to higui_following_edited.csv. You can additionally add individual users by setting 'api_single' to True and giving a name. You can also add every user someone is following by setting 'expand_from_user' to True and giving a name. You can generate recommendations by setting 'update_top_scores' to True.

