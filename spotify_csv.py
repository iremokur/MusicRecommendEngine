import pandas as pd
import numpy as np
import re 
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template
###  FOR CSV SETTINGS ###
### RUNS ONCE ###
app = Flask(__name__,template_folder='templates')

scope = 'user-library-read'

#simple function to create OHE features
def ohe_prep(df, column, new_name): 

    """ 
    Create One Hot Encoded features of a specific column

    Parameters: 
        df (pandas dataframe): Spotify Dataframe
        column (str): Column to be processed
        new_name (str): new column name to be used
        
    Returns: 
        tf_df: One hot encoded features 
    """
    
    tf_df = pd.get_dummies(df[column])
    feature_names = tf_df.columns
    tf_df.columns = [new_name + "|" + str(i) for i in feature_names]
    tf_df.reset_index(drop = True, inplace = True)    
    return tf_df

#function to build entire feature set
def create_feature_set(df, float_cols):
    #tfidf genre lists
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    genre_df.reset_index(drop = True, inplace=True)

    #explicity_ohe = ohe_prep(df, 'explicit','exp')    
    year_ohe = ohe_prep(df, 'year','year') * 0.5
    popularity_ohe = ohe_prep(df, 'popularity_red','pop') * 0.15

    #scale float columns
    floats = df[float_cols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    #concanenate all features
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
     
    #add song id
    final['id']=df['id'].values
    
    return final

def preprocess_dataset():
        spotify_df = pd.read_csv('data/data.csv')
        data_w_genre = pd.read_csv('data/data_w_genres.csv')
        data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
        spotify_df['artists_upd_v1'] = spotify_df['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
        spotify_df[spotify_df['artists_upd_v1'].apply(lambda x: not x)].head(5)

        spotify_df['artists_upd_v2'] = spotify_df['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
        spotify_df['artists_upd'] = np.where(spotify_df['artists_upd_v1'].apply(lambda x: not x), spotify_df['artists_upd_v2'], spotify_df['artists_upd_v1'] )

        #need to create my own song identifier because there are duplicates of the same song with different ids. I see different
        spotify_df['artists_song'] = spotify_df.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
        spotify_df.sort_values(['artists_song','release_date'], ascending = False, inplace = True)

        spotify_df.drop_duplicates('artists_song',inplace = True)
        artists_exploded = spotify_df[['artists_upd','id']].explode('artists_upd')

        artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'artists')
        artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]

        artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
        artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

        spotify_df = spotify_df.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')
    
        #Feature Engineering

        spotify_df['year'] = spotify_df['release_date'].apply(lambda x: x.split('-')[0])
        
        
        # create 5 point buckets for popularity 
        spotify_df['popularity_red'] = spotify_df['popularity'].apply(lambda x: int(x/5))
        # tfidf can't handle nulls so fill any null values with an empty list
        
        spotify_df['consolidates_genre_lists'] = spotify_df['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])
        spotify_df.to_csv("spotify_last.csv")
        return spotify_df

def main():

    spotify_df=preprocess_dataset()
   
    float_cols = spotify_df.dtypes[spotify_df.dtypes == 'float64'].index.values
    complete_feature_set = create_feature_set(spotify_df, float_cols=float_cols)#.mean(axis = 0)
   
    complete_feature_set.to_csv("final.csv")



if __name__ == "__main__":
    main()
