def app():
    import pandas as pd 
    import numpy as np
    import random
    import streamlit as st
    import re 
    from surprise import Reader, Dataset, SVD
    
    from surprise import SVD
    from surprise import BaselineOnly
    from surprise.model_selection import cross_validate
    import matplotlib.pyplot as plt
    import seaborn as sns

    #importing the dataset
    movies=pd.read_csv('movies.csv')
    ratings=pd.read_csv('ratings.csv')

    


    #Converting the format of Genre column to a list and then appending to the new list
    Genre=[]
    Genres={}
    for num in range(0,len(movies)):
        key=movies.iloc[num]['title']
        value=movies.iloc[num]['genres'].split('|')
        Genres[key]=value
        Genre.append(value)

        
    #Making a new column in our original Dataset         
    movies['Gen'] =Genre

    #Getting the year from the movie column 
    p = re.compile(r"(?:\((\d{4})\))?\s*$")
    years=[]
    for movie in movies['title']:
        m = p.search(movie)
        year = m.group(1)
        years.append(year)  
    movies['year']=years

    #Deleting the year from the movies title column
    movies_name=[]
    raw=[]
    for movie in movies['title']:
        m = p.search(movie)
        year = m.group(0)
        new=re.split(year,movie)
        raw.append(new)  
    for i in range(len(raw)):
        movies_name.append(raw[i][0][:-2].title())
        

    movies['movie_name']=movies_name


    #Converting the datatype of new column from list to string as required by the function
    movies['Gen']=movies['Gen'].apply(' '.join)
    
    """Applying the Content Based Filtering"""
    # Feature extraction 
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfid=TfidfVectorizer(stop_words='english')
    matrix=tfid.fit_transform(movies['Gen'])

    #Compute the cosine similarity of every genre
    from sklearn.metrics.pairwise import cosine_similarity
    cosine_sim=cosine_similarity(matrix,matrix)

    """Applying the Collaborative Filtering"""
    #Intialising the Reader which is used to parse the file containing the ratings 
    reader=Reader()
    dataset=Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)

    #Benchmarking with various Algorithms
    """benchmark = []
    # Iterating over all algorithms
    for algorithm in [BaselineOnly()]:
        # Performing  cross validation
        results = cross_validate(algorithm, dataset, measures=['RMSE'], cv=2 ,verbose=True)
        
        # Get results & append algorithm name
        tmp = pd.DataFrame.from_dict(results).mean(axis=0)
        tmp = tmp.append(pd.Series([str(algorithm).split(' ')[0].split('.')[-1]], index=['Algorithm']))
        benchmark.append(tmp)
        
    bench = pd.DataFrame(benchmark).set_index('Algorithm').sort_values('test_rmse')"""
    def recommendation(movie):
        result=[]
        #Getting the id of the movie for which the user want recommendation
        ind=indices[movie]
        #Getting all the similar cosine score for that movie
        sim_scores=list(enumerate(cosine_sim[ind]))
        #Sorting the list obtained
        sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)    
        #Getting all the id of the movies that are related to the movie Entered by the user
        movie_id=[i[0] for i in sim_scores]    
        print('The Movie You Should Watched Next Are --')
        print('ID ,   Name ,  Average Ratings , Year ')
        #Varible to print only top 10 movies
        count=0
        for id in range(0,len(movie_id)):
        #to ensure that the movie entered by the user is doesnot come in his/her recommendation
            if(ind != movie_id[id]):
                rating=ratings[ratings['movieId']==movie_id[id]]['rating']
                avg_ratings=round(np.mean(rating),2)
                #To print only thoese movies which have an average ratings that is more than 3.5
                if(avg_ratings >3.5):
                    count+=1
                    print(f'{movie_id[id]} , {titles[movie_id[id]]} ,{avg_ratings}')
                    result.append([titles[movie_id[id]],str(avg_ratings)])
                if(count >=10):
                        break
        
        print('Accumulating your Recommendations')
        return result

    #Movie name and movie id 
    movies_dataset = movies.reset_index()
    titles = movies['movie_name']
    indices = pd.Series(movies.index, index=movies  ['movie_name'])
    #Function to make recommendation to the user

    movie = st.text_input("Input Your Recent Movie","Text")  
    if  st.button("Recommend"):
        try:        
            result=recommendation(movie)
            df = pd.DataFrame(result, columns = ['Movie', 'Rating']) 
            df.index = np.arange(1,len(df)+1)
            df = df.style.set_properties(**{'text-align': 'left'})
            print(df)
            
            st.dataframe(df)
        except:
            st.write("Movie not Found in Database")
