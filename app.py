from flask import Flask, render_template, request,url_for,flash,redirect
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy.util as util

app = Flask(__name__,template_folder='templates')
scope = 'user-library-read'

# app.secret_key = SSK
API_BASE = 'https://accounts.spotify.com'


# Pull songs from a user's playlist.
def createPlaylistOutputs(playlistName, df,sp):
    
    #generate playlist dataframe
    playlist = pd.DataFrame()
    playlistName = playlistName.split('?')[0]

    playlistName = playlistName.split('/')[4]

    for ix, i in enumerate(sp.playlist(playlistName)['tracks']['items']):

        playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
        playlist.loc[ix, 'name'] = i['track']['name']
        playlist.loc[ix, 'id'] = i['track']['id'] # ['uri'].split(':')[2]
        playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
        playlist.loc[ix, 'date_added'] = i['added_at']

    playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
    
    playlist = playlist[playlist['id'].isin(df['id'].values)].sort_values('date_added',ascending = False)
    #all songs in the playlist
    return playlist


# Summarize the user's playlist into a single vector
# weightFactor: 1.09
def generatePlaylistFeature(completeFeatureSet, playlist_df, weightFactor):

    completeFeatureSetPlaylist = completeFeatureSet[completeFeatureSet['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1).mean(axis =0)
    completeFeatureSetPlaylist = completeFeatureSetPlaylist.merge(playlist_df[['id','date_added']], on = 'id', how = 'inner')
    completeFeatureSetNonplaylist = completeFeatureSet[~completeFeatureSet['id'].isin(playlist_df['id'].values)]#.drop('id', axis = 1)
    
    playlistFeatureSet = completeFeatureSetPlaylist.sort_values('date_added',ascending=False)
    most_recent_date = playlistFeatureSet.iloc[0,-1]
    
    for ix, row in playlistFeatureSet.iterrows():
        playlistFeatureSet.loc[ix,'months_from_recent'] = int((most_recent_date.to_pydatetime() - row.iloc[-1].to_pydatetime()).days / 30)
        
    playlistFeatureSet['weight'] = playlistFeatureSet['months_from_recent'].apply(lambda x: weightFactor ** (-x))
    ##weighted playlist
    playlistFeatureSet_weighted = playlistFeatureSet.copy()
    playlistFeatureSet_weighted.update(playlistFeatureSet_weighted.iloc[:,:-4].mul(playlistFeatureSet_weighted.weight,0))
    playlistFeatureSet_weighted_last = playlistFeatureSet_weighted.iloc[:, :-4]
    
    # playlistFeatureSet_weighted_last single feature that summarizes the playlist
    # completeFeatureSetNonplaylist
    return playlistFeatureSet_weighted_last.sum(axis = 0), completeFeatureSetNonplaylist



#Pull songs from a specific playlist.
def generatePlaylistCosineSim(df, features, nonplaylist_features, sp):
   
    nonplaylistDf = df[df['id'].isin(nonplaylist_features['id'].values)]
    ##cosine similarity is used
    nonplaylistDf['sim'] = cosine_similarity(nonplaylist_features.drop('id', axis = 1).values, features.values.reshape(1, -1))[:,0]
    # take the 10 nearest songs
    nonplaylistDf_top10 = nonplaylistDf.sort_values('sim',ascending = False).head(10)
    nonplaylistDf_top10['url'] = nonplaylistDf_top10['id'].apply(lambda x: sp.track(x)['album']['images'][1]['url'])
    # convert to an dictionary to help the passing the data.
    response = nonplaylistDf_top10.to_dict('records')

    # Top 5 recommendations for that playlist as a dict.
    return response


def spotify(link,spotifyDf,sp):
 
    playlist = createPlaylistOutputs(link,spotifyDf,sp)
    return playlist

#first 
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
    spotifyDf=pd.read_csv("spotify_last.csv")
    ####AUTHENTICATION####
    #client id and secret FROM SPOTIFY DEVELOPERS
    client_id = '941e4cf6c2294d018f5dde685dbc5e0c'
    client_secret= '6f2b6dbaffdb4ba8ae4ea06b8d40c45d'
    auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(auth_manager=auth_manager)

    token = util.prompt_for_user_token(scope, client_id= client_id, client_secret=client_secret, redirect_uri='"http://127.0.0.1:5000/submit')
    sp = spotipy.Spotify(auth=token)
    
    #take the user's playlist and make it meaningful
    playlist=spotify(link,spotifyDf,sp)
    # comes from spotify_csv.py
    completeFeatureSet=pd.read_csv("final.csv")

    #completeFeatureSetNonplaylist: not in the user's playlist
    completeFeatureSetPlaylistVector, completeFeatureSetNonplaylist = generatePlaylistFeature(completeFeatureSet, playlist, 1.09)
    top10 = generatePlaylistCosineSim(spotifyDf, completeFeatureSetPlaylistVector, completeFeatureSetNonplaylist, sp)

    return render_template("songs.html", top10=top10) 


if __name__ == "__main__":
    app.run()


