from winsound import PlaySound
from flask import Flask, render_template, request,url_for,flash,redirect
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util

app = Flask(__name__,template_folder='templates')
scope = 'user-library-read'

# app.secret_key = SSK
API_BASE = 'https://accounts.spotify.com'


# Pull songs from a user's playlist.
def create_necessary_outputs(playlist_name, df,sp):
    
    #generate playlist dataframe
    playlist = pd.DataFrame()
    playlist_name = playlist_name.split('?')[0]

    playlist_name = playlist_name.split('/')[4]

    for ix, i in enumerate(sp.playlist(playlist_name)['tracks']['items']):

        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id'] # ['uri'].split(':')[2]
        playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[ix, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
    
    playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added',ascending = False)
    #all songs in the playlist
    return playlist


# Summarize a user's playlist into a single vector
def generate_playlist_feature(complete_feature_set, playlist_df, weight_factor):

       
    complete_feature_set_playlist = complete_feature_set[complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1).mean(axis =0)
    complete_feature_set_playlist = complete_feature_set_playlist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')
    complete_feature_set_nonplaylist = complete_feature_set[~complete_feature_set['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
    
    playlist_feature_set = complete_feature_set_playlist.sort_values('date_added',ascending=False)

    most_recent_date = playlist_feature_set.iloc[0,-1]
    
    for ix, row in playlist_feature_set.iterrows():
        playlist_feature_set.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
        
    playlist_feature_set['weight'] = playlist_feature_set['months_from_recent'].apply(lambda x: weight_factor ** (-x))
    
    playlist_feature_set_weighted = playlist_feature_set.copy()
    playlist_feature_set_weighted.update(playlist_feature_set_weighted.iloc[:,:-4].mul(playlist_feature_set_weighted.weight,0))
    playlist_feature_set_weighted_final = playlist_feature_set_weighted.iloc[:, :-4]
    # playlist_feature_set_weighted_final single feature that summarizes the playlist
    # complete_feature_set_nonplaylist
    return playlist_feature_set_weighted_final.sum(axis = 0), complete_feature_set_nonplaylist



#Pull songs from a specific playlist.
def generate_playlist_recos(df, features, nonplaylist_features, sp):
   
    non_playlist_df = df[df['id'].isin(nonplaylist_features['id'].values)]
    non_playlist_df['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    non_playlist_df_top_10 = non_playlist_df.sort_values('sim',ascending = False).head(10)
    non_playlist_df_top_10['url'] = non_playlist_df_top_10['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    

    response = non_playlist_df_top_10.to_dict('records')
    
    # Top 5 recommendations for that playlist as a dict.
    return response


def spotify(link,spotify_df,sp):
 
    playlist = create_necessary_outputs(link,spotify_df,sp)
    return playlist

@app.route("/",methods=['GET', 'POST'])
def render():
    if request.method == 'POST':
        return redirect(url_for("submit"))  

    return render_template("index.html")


@app.route("/home",methods=['GET', 'POST'])
def home():

    return render_template("home.html")
    
@app.route('/submit',methods=['GET', 'POST'])
def submit():
    #take playlist url
    link= request.form.get("inputName")
    spotify_df=pd.read_csv("spotify_last.csv")
    ####AUTHENTICATION####
    #client id and secret FROM SPOTIFY DEVELOPERS
    client_id = '941e4cf6c2294d018f5dde685dbc5e0c'
    client_secret= '6f2b6dbaffdb4ba8ae4ea06b8d40c45d'
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='"http://127.0.0.1:5000/submit')
    sp = spotipy.Spotify(auth=token)
    
    playlist=spotify(link,spotify_df,sp)
    complete_feature_set=pd.read_csv("final.csv")

    complete_feature_set_playlist_vector_EDM, complete_feature_set_nonplaylist_EDM = generate_playlist_feature(complete_feature_set, playlist, 1.09)
    # #complete_feature_set_playlist_vector_chill, complete_feature_set_nonplaylist_chill = generate_playlist_feature(complete_feature_set, playlist_chill, 1.09)
    top5 = generate_playlist_recos(spotify_df, complete_feature_set_playlist_vector_EDM, complete_feature_set_nonplaylist_EDM, sp)

    return render_template("songs.html", top5=top5) 


if __name__ == "__main__":
    app.run()


