# -*- coding: utf-8 -*-

import numpy as np
import math

class UserSimilarityKNN:
    
    def __init__(self, dl, k=40):
        self.k = k
        self.dl = dl
        
    def computeUserKNNAlgorithm(self, sparseMatrix, denseMatrix, userName=None):
        
       print("\nComputing All Users Similarity Based Ratings...")
       
       trainingMatrix = np.copy(sparseMatrix)
       
       start=0
       end=trainingMatrix.shape[0]       
       if userName is not None:
           userInt = int(userName)
           start=userInt
           end=start+1        
       
       userSimilarities = np.zeros([trainingMatrix.shape[0], trainingMatrix.shape[0]]) 
       
       for user in range(start, end, 1):
           
           otherStart = user+1
           #To account for computing for only a single user
           if (end-start)==1:
               otherStart = 0
           
           for otherUser in range(otherStart, trainingMatrix.shape[0]):               
               userSimilarities[user, otherUser] = self.computeCosineSimilarity(trainingMatrix[user], trainingMatrix[otherUser])
               userSimilarities[otherUser, user] = userSimilarities[user, otherUser]
               
           trainingMatrix[user] = self.computeUserNeighbourhoodRatings(userSimilarities[user], denseMatrix)  
                
           print("*", end="")
           if ((user+1) % 50 == 0):
               print("\n"+str(user) + " of " + str(trainingMatrix.shape[0]) + " Users Similarity Based Ratings Computed")
           
       print("\nAll Users Similarity Based Ratings Computed")
       return trainingMatrix

    
    #Find top-k similar users in user's neighbourhood
    def computeUserNeighbourhoodRatings(self, similarities, denseMatrix):
        
        tmpMatrix = np.zeros([self.k, denseMatrix.shape[1]], dtype=np.float32)
        userSimTmp = np.copy(similarities)

        for i in range(self.k):
            index = userSimTmp.argmax()            
            tmpMatrix[i] = denseMatrix[index] * userSimTmp[index]
            userSimTmp[index] = -1.0
            
        tmpMatrix = tmpMatrix.T
        userSimilarityMovieScore = np.zeros([denseMatrix.shape[1]], dtype=np.float32)
        for i in range(denseMatrix.shape[1]):
            maxRatingIndex = tmpMatrix[i].argmax()
            itemSimSum = 0.0
            maxRatingSum = 0.0
            if np.amax(tmpMatrix[i]) <= 0.0:
                itemSimSum = 0.0
                maxRatingSum = 1.0
            else:
                for j in range(self.k):
                    itemSimSum += tmpMatrix[i][j]
                    maxRatingSum += tmpMatrix[i][maxRatingIndex]
            userSimilarityMovieScore[i] = (itemSimSum/maxRatingSum)

        return userSimilarityMovieScore
    
 
   #Find Cosine similarities between users
    def computeCosineSimilarity(self, userRatings, otherRatings): 
        if (np.array_equal(userRatings, otherRatings)):
            return 1.0

        #Been vectorized to improve efficiency upto 500 times, used loops earlier
        # The data seemed to be too sparse, to the point using scipy.sparse performed worse than numpy.dot
        sumxx = np.dot(userRatings, userRatings)
        sumyy = np.dot(otherRatings, otherRatings)
        sumxy = np.dot(userRatings, otherRatings)
        
        if sumxx==0.0:
            sumxx=1.0
        if sumyy==0.0:
            sumyy=1.0
            
        return sumxy/math.sqrt(sumxx*sumyy)
    
    
    def normalizeArray(self, row):
        diff = (row-row.min())
        maxDiff = row.max() - row.min()
        if maxDiff==0:
            maxDiff=1
        return diff/maxDiff 