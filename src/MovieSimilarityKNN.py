# -*- coding: utf-8 -*-

from surprise import PredictionImpossible
import math
import numpy as np

class MovieSimilarityKNN:

    def __init__(self, dl, k=40):
        self.k = k
        self.dl = dl

    def fit(self, trainset, train=True):

        # Compute movie similarity matrix based on generes, release year, average rating and popularity ranks
        
        self.trainset = trainset
        
        # Load up genre vectors, years and popularity for every movie
        print("\nLoading and Prcoessing movie feature vectors...")
        genres, genomes = self.preProcessMovieFeatureVectors()
        years = self.dl.getYears()
        popular = self.dl.getPopularityRanks()
        print("\nMovie feature vectors Ready")
        
        #Get average movie ratings
        avgMovieRating = self.dl.getAvgRating()      
        
        print("\nComputing Movie similarity matrix...")
            
        self.similarities = np.zeros((self.trainset.n_items, self.trainset.n_items), dtype=np.float32)
        
        if train:
            
            # Compute genre distance for every movie combination
            for thisRating in range(self.trainset.n_items):

                if ((thisRating+1) % 50 == 0):
                    print("\nComputed all movie similarities of Moive:  " + str(thisRating) + "  of  " + str(self.trainset.n_items))

                #Some optimazation to not re-read already set values
                for otherRating in range(thisRating+1, self.trainset.n_items):   
                
                    thisMovieID = int(self.trainset.to_raw_iid(thisRating))
                    otherMovieID = int(self.trainset.to_raw_iid(otherRating))
                
                    #Get similarities in genre
                    genreSimilarity = self.computeCosineSimilarity(thisMovieID, otherMovieID, genres)
                    genomeSimilarity = self.computeCosineSimilarity(thisMovieID, otherMovieID, genomes)
                    #To make sure movies with two many years in difference are penalized
                    yearSimilarity = self.computeYearSimilarity(thisMovieID, otherMovieID, years)
                
                    #To make similarity relations 95% genre and 5% year difference
                    thisSimilarity = (genreSimilarity + genomeSimilarity) *0.95 + yearSimilarity[0] *0.1
                    otherSimilarity = (genreSimilarity + genomeSimilarity) *0.95 + yearSimilarity[1]*0.1
                
                    #Compute rewards and penalties for popularity
                    thisPopularity = 1-1.0*popular[thisMovieID]/len(popular)
                    otherPopularity = 1-1.0*popular[otherMovieID]/len(popular)
                
                    #Compute rewards and penalties for popularity and average rating
                    thisAvgRating = avgMovieRating[thisMovieID]/5.0
                    otherAvgRating = avgMovieRating[otherMovieID]/5.0
                
                    #To make average rating and popularity relations 90% ratings 5% popularity
                    ratingPopularity = 1.0 - abs((0.9*thisAvgRating + 0.1*thisPopularity) - (0.9*otherAvgRating + 0.1*otherPopularity))
                
                    #Set desired relationships between movies 95% similarity 5% ratings and popularity
                    self.similarities[thisRating, otherRating] = 0.95*thisSimilarity + 0.05*ratingPopularity
                    self.similarities[otherRating, thisRating] = 0.95*otherSimilarity + 0.05*ratingPopularity
                
                print("*", end="")
                
            #save movie similarity matrix
            self.dl.writeLearnedWeightsToFile(knn=self.similarities)
                
        else:
            
            #Load pre-computed movie similarity matrix
            self.similarities = self.dl.readLearnedWeightsFromFile(knn=True)
            
          
        print("Movie Similarity Matrix Computed")
                
        return self
   

    def mergeMovieSimilaritiesUserRatings(self, trainingMatrix, userName=None):
        
        print("\nFilling User Ratings with Item Similarity Ratings...")
        
        start=0
        end=trainingMatrix.shape[0]    
        if userName is not None:
            userInt = int(userName)
            start=userInt
            end=start+1        
        
        for user in range(start, end, 1):           
            for movie in range(0, trainingMatrix.shape[1]):  
                
                #Fill Unknown Ratings with K-neighbour item similarity ratings
                if trainingMatrix[user, movie] <= 0.0: 

                     trainingMatrix[user, movie] = self.computeMovieNeighbourhoodRatings(user, movie)  
                    
            print("*", end="")
            if ((user+1) % 50 == 0):
                print("\n"+str(user) + " of " + str(trainingMatrix.shape[0]) + " User's predicted Movie Similarity Ratings Filled")
            
        print("\nEmpty User Ratings filled with Item Similarity Ratings")
        
        return trainingMatrix


    def computeMovieNeighbourhoodRatings(self, u, i):

        if not (self.trainset.knows_user(u)):
            raise PredictionImpossible('User is unkown.')
        elif not self.trainset.knows_item(i):
            raise PredictionImpossible('Item is unkown.')
        
        # make list with item and similarity to user rated movie and user rating
        neighbors = []

        for rating in self.trainset.ur[u]:
            neighbors.append( (self.similarities[i,rating[0]], rating[1]) )
        
        # Sort K most similar items
        neighbors.sort(key=lambda t: t[0])

        # Compute average similarity score of K closest neighbours rated by user and their user ratings
        simTotal = 0
        weightedSum = 0

        k = self.k
        if len(neighbors) < k:
            k = len(neighbors)
        for i in range(k):
            weightedSum += neighbors[i][0] * neighbors[i][1]           
            simTotal += 1
            
        if (simTotal == 0):
            raise PredictionImpossible('No neighbors')

        predictedRating = weightedSum / simTotal
        
        return predictedRating / 5.0


    #Combine genre vectors and genome data vectors for every movie
    #Tags were removed as genome scores convers that as well
    def preProcessMovieFeatureVectors(self):
        
        genres = self.dl.getGenres()
        genomes = self.dl.getGenomeData()
        #tags = self.dl.getTags()

        #All set to float since float calculations seemed to work faster than int calculation (most likely cause of BLAST)
        for movieID in genres:
            
            #Set movies without genome data have a zero array
            if movieID in genomes:
                genomes[movieID] = np.asarray(genomes[movieID], dtype=np.float32)
            else:
                genomes[movieID] = np.zeros(self.dl.numGenomes, dtype=np.float32)
            
        return genres, genomes
                    
    
    #Find cosine distance between movie genre or genome vectors - Dense rows as sparse calculation was slower
    def computeCosineSimilarity(self, movie1, movie2, features):
        
        movie1 = features[movie1]
        movie2 = features[movie2]
        
        if (not np.any(movie1)) or (not np.any(movie2)):
            return 0.0            
        elif (np.array_equal(movie1, movie2)):
            return 1.0
            
        #Been vectorized to improve efficiency upto 500 times, used loops earlier
        # The data seemed to be too sparse, to the point using scipy.sparse performed worse than numpy.dot
        sumxx = np.dot(movie1, movie1)
        sumyy = np.dot(movie2, movie2)
        sumxy = np.dot(movie1, movie2)
        
        if sumxx==0.0:
            sumxx=1.0
        if sumyy==0.0:
            sumyy=1.0
            
        return sumxy/math.sqrt(sumxx*sumyy)
    
    
    #Generate similarity between movie years
    def computeYearSimilarity(self, movie1, movie2, years):
        if years[movie1] == years[movie2]:
            return [1.0,1.0]
        
        diff = years[movie1] - years[movie2]
        sim = [None,None]
            
        if diff >= 0:
            sim[1] = math.exp(-diff / 90.0)
            sim[0] = math.exp(-diff / 120.0)
        else:
            sim[0] = math.exp(diff / 90.0)
            sim[1] = math.exp(diff / 120.0)
            
        return sim    
 
    
    def GetName(self):
        return "MovieSimilarityKNN"
    
