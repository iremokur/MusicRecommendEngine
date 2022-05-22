import pandas as pd
import numpy as np
import re 
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")
from flask import Flask
###  FOR CSV SETTINGS ###
### RUNS ONCE ###
app = Flask(__name__,template_folder='templates')

scope = 'user-library-read'

#simple function to create OHE features
def oneHotEncode(df, column, new_name): 
    #take the categorical variables and convert it to dummy variable!
    tfDF = pd.get_dummies(df[column])
    feature_names = tfDF.columns
    tfDF.columns = [new_name + "|" + str(i) for i in feature_names]
    tfDF.reset_index(drop = True, inplace = True)    
    return tfDF

#function to build entire feature set
def create_feature_set(df, floatCols):
    #tfdf genre lists
    tfidf = TfidfVectorizer()
    tfidf_matrix =  tfidf.fit_transform(df['consolidates_genre_lists'].apply(lambda x: " ".join(x)))
    #genre
    genre_df = pd.DataFrame(tfidf_matrix.toarray())
    genre_df.columns = ['genre' + "|" + i for i in tfidf.get_feature_names()]
    genre_df.reset_index(drop = True, inplace=True)
    #year
    year_ohe = oneHotEncode(df, 'year','year') * 0.5
    #popularity
    popularity_ohe = oneHotEncode(df, 'popularity_red','pop') * 0.15

    #scale float columns
    floats = df[floatCols].reset_index(drop = True)
    scaler = MinMaxScaler()
    floats_scaled = pd.DataFrame(scaler.fit_transform(floats), columns = floats.columns) * 0.2

    #concanenate all features
    final = pd.concat([genre_df, floats_scaled, popularity_ohe, year_ohe], axis = 1)
     
    #add song id
    final['id']=df['id'].values

    return final

#preprocessing the dataset
#concat all unecessary chars from the strings(genres, artists etc.)
def preprocess_dataset():
        #these csv's comes from KAGGLE DATASET
        #KAGGLE links are in the PROJECT REPORT!!!!
        spotifyDf = pd.read_csv('data/data.csv')
        data_w_genre = pd.read_csv('data/data_w_genres.csv')
        data_w_genre['genres_upd'] = data_w_genre['genres'].apply(lambda x: [re.sub(' ','_',i) for i in re.findall(r"'([^']*)'", x)])
        spotifyDf['artists_upd_v1'] = spotifyDf['artists'].apply(lambda x: re.findall(r"'([^']*)'", x))
        spotifyDf[spotifyDf['artists_upd_v1'].apply(lambda x: not x)].head(5)

        spotifyDf['artists_upd_v2'] = spotifyDf['artists'].apply(lambda x: re.findall('\"(.*?)\"',x))
        spotifyDf['artists_upd'] = np.where(spotifyDf['artists_upd_v1'].apply(lambda x: not x), spotifyDf['artists_upd_v2'], spotifyDf['artists_upd_v1'] )

        spotifyDf['artists_song'] = spotifyDf.apply(lambda row: row['artists_upd'][0]+row['name'],axis = 1)
        spotifyDf.sort_values(['artists_song','release_date'], ascending = False, inplace = True)

        spotifyDf.drop_duplicates('artists_song',inplace = True)
        artists_exploded = spotifyDf[['artists_upd','id']].explode('artists_upd')

        artists_exploded_enriched = artists_exploded.merge(data_w_genre, how = 'left', left_on = 'artists_upd',right_on = 'artists')
        artists_exploded_enriched_nonnull = artists_exploded_enriched[~artists_exploded_enriched.genres_upd.isnull()]

        artists_genres_consolidated = artists_exploded_enriched_nonnull.groupby('id')['genres_upd'].apply(list).reset_index()
        artists_genres_consolidated['consolidates_genre_lists'] = artists_genres_consolidated['genres_upd'].apply(lambda x: list(set(list(itertools.chain.from_iterable(x)))))

        spotifyDf = spotifyDf.merge(artists_genres_consolidated[['id','consolidates_genre_lists']], on = 'id',how = 'left')
    
        #Feature Engineering
        spotifyDf['year'] = spotifyDf['release_date'].apply(lambda x: x.split('-')[0])
        # create 5 point buckets for popularity 
        spotifyDf['popularity_red'] = spotifyDf['popularity'].apply(lambda x: int(x/5))
        # tfidf can't handle nulls so fill any null values with an empty list
        spotifyDf['consolidates_genre_lists'] = spotifyDf['consolidates_genre_lists'].apply(lambda d: d if isinstance(d, list) else [])
        # assign it to a csv to use it app.py!!
        spotifyDf.to_csv("spotify_last.csv")
        return spotifyDf

def main():

    spotifyDf=preprocess_dataset()
    floatCols = spotifyDf.dtypes[spotifyDf.dtypes == 'float64'].index.values
    complete_feature_set = create_feature_set(spotifyDf, floatCols=floatCols)#.mean(axis = 0)
    #######################FINAL CSV (used in app.py)
    complete_feature_set.to_csv("final.csv")



if __name__ == "__main__":
    main()
