# Author: Mostafa ElAraby
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD,PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
import pymysql
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
import math
import config



plotData = False
def tfIDFeats(ids,data):


    # the infamous tfidf vectorizer (Do you remember this one?)
    tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 5), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
    # Fit TFIDF
    tfv.fit(data)
    X =  tfv.transform(data) 
        # Initialize SVD

    svd = TruncatedSVD(n_components=350)
    
    # Initialize the standard scaler 
    scl = StandardScaler( with_mean=False)
    
    
    
    if X.shape[1]>350:
        X = svd.fit_transform(X)
    X = scl.fit_transform(X,ids)
    if plotData:
        X = PCA(n_components=2).fit_transform(X)
    return (X,ids)

def initMysql():
	db =  pymysql.connect(host=config.MYSQL['MYSQLDB_SERVER'], # your host, usually localhost
                     user=config.MYSQL["MYSQLDB_USER"], # your username
                      passwd=config.MYSQL["MYSQLDB_PWD"], # your password
                      db=config.MYSQL["MYSQLDB_DB"])  # name of the data base
	db.set_charset('utf8')
	cur = db.cursor() 
	return (db,cur)
def getArticles(cursor):
	cursor.execute ("select articles.id, title , summary,content from articles LEFT outer join relatedarticles on articles.id=articleId where distance is NULL ;")
	# fetch all of the rows from the query
	data = cursor.fetchall ()
	ids = []
	resultArticles = []
	for row in data :
		ids.append(row[0])
		resultArticles.append(BeautifulSoup(row[1]+' '+row[2]).get_text())
	return (ids,resultArticles)

def clusterArticlesKMeans(n_clusters,reduced_data):
    
    kmeans = KMeans( n_clusters=n_clusters,n_jobs =5)
    results = kmeans.fit_predict(reduced_data)
    if plotData:
        plot_data(reduced_data,kmeans)
    return (results,kmeans.cluster_centers_ ) 
 
def plot_data(reduced_data,kmeans):
    h = .02  
    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
def insertDataDB(db,cursor,results,centers,ids,feats):
    curIndex = 0
    #cursor.execute('DELETE FROM relatedArticles')
    #database.commit()
    for result in results:
        distance = euclidean_distances(feats[curIndex],centers[result])
        cursor.execute('INSERT INTO relatedArticles(articleId,clusterId,distance) VALUES('+str(ids[curIndex])+','+str(result)+','+str(distance[0][0])+')')
        curIndex = curIndex+1
    db.commit()
    cursor.close()
    db.close()

def clusterArticles(feats,ids,cursor,database):
    articlesClustered = []
    distances = np.zeros((len(ids), len(ids)))
    for i,id in enumerate(ids):
        for j in range(0,len(ids)):
            distances[i][j] = euclidean_distances(feats[i],feats[j])
    clusterIndex = 0
    cursor.execute('DELETE FROM relatedArticles')
    database.commit()
    for i,id in enumerate(ids):
        if id in articlesClustered:
            continue
        outIds = getTopIndeces(distances[i][1:])
        
        articlesClustered.append(id)
        isClusteredBefore = False;
        for j,currentId in enumerate(outIds):
            if ids[currentId] in articlesClustered:
                isClusteredBefore = True
                break
        for j,currentId in enumerate(outIds):
            cursor.execute('INSERT INTO relatedArticles(articleId,clusterId,distance) VALUES('+str(ids[currentId])+','+str(clusterIndex)+','+str(distances[i][currentId])+')')
            articlesClustered.append(ids[currentId])
        cursor.execute('INSERT INTO relatedArticles(articleId,clusterId,distance) VALUES('+str(id)+','+str(clusterIndex)+','+str(distances[i][0])+')')
        
        clusterIndex = clusterIndex+1
    database.commit()
def getTopIndeces(distances):
    output = np.argpartition(distances, -1*config.learningParams['n_clusters'])[:config.learningParams['n_clusters']]
    return output
if __name__ == '__main__':
    (database,cursor) = initMysql()
    (ids,resultArticles) = getArticles(cursor)
    (feats,ids) = tfIDFeats(ids,resultArticles)
    if config.learningParams['isKmeans']:
        n_clusters = int(math.ceil(len(ids)/config.learningParams['n_clusters']))
        (results,centers) = clusterArticlesKMeans(n_clusters,feats)
        insertDataDB(database,cursor,results,centers,ids,feats)
    else:
        clusterArticles(feats,ids,cursor,database)
    print 'clustering done using TF-IDF'