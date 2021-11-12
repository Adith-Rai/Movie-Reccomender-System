import math

from DataProcessing import DataProcessing

class GenerateRecommendations:

    
    def __init__(self, dataset, train):
        
        self.processedData = DataProcessing(dataset)
        self.train = train
        
 
    
    def ComputeRecs(self, algo, dl, user, k=25, genreFilter=[]):
        
        
        print("\nUsing recommender ", algo.GetName())
            
        print("\nBuilding recommendation model...")
        trainSet = self.processedData.GetFullTrainSet()
            
        algo.fit(trainSet, self.train, user)

        
        print("Computing recommendations...")
        trainSet = self.processedData.GetUserAntiTestSet(user)
            

        predictions = algo.test(trainSet)
            
        self.buildStoplist(predictions, dl)
            
        recommendations = []
        
        genreFilterBits, myGenresMode, allGenres = self.filterGenresToBitwise(genreFilter, dl)
        avgUsersYear, allYears = dl.getAvgUserYear()
        popularityWeights = dl.computePopularityWeights()
        avgMovieRating = dl.getAvgRating()
        
        print ("\nWe recommend:")
        for userID, movieID, actualRating, estimatedRating, _ in predictions:
            intUserID = int(userID)
            intMovieID = int(movieID)
            if not self.stoplistLookup[intMovieID]:
                
                #calculate genre weights
                myGenresWeight = 0.0
                for i in range(len(myGenresMode)):
                    myGenresWeight += allGenres[intMovieID][i] * myGenresMode[i]

                #Calculate year weight
                yearWeight = self.computeYearSimilarity(avgUsersYear[intUserID][0], allYears[intMovieID], avgUsersYear[intUserID][1])
                    
                #add genre and year weights to predictions
                estimatedRating = estimatedRating*0.8 + 0.1*(myGenresWeight * yearWeight) + 0.1*(avgMovieRating[intMovieID]*0.2 * popularityWeights[intMovieID])

                if len(genreFilter) == 0:
                    recommendations.append((intMovieID, estimatedRating))
                else:
                    genersAND = []
                    for i in range(len(genreFilterBits)):
                        genersAND.append(genreFilterBits[i] * allGenres[intMovieID][i])
                    if genersAND == genreFilterBits:
                        recommendations.append((intMovieID, estimatedRating))

                    
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        if k > len(recommendations):
            k = len(recommendations)
            
        for ratings in recommendations[:k]:
            print(dl.getMovieName(ratings[0]), ratings[1])
        if len(recommendations)==0:
            print("Sorry, we could not find any movies that match your criteria.")
                
        dl.writeReccomendationsToFile(recommendations)
            
            

    def buildStoplist(self, trainset, dl):
        self.stoplistLookup = {}
        for userID, movieID, actualRating, estimatedRating, _ in trainset:
            self.stoplistLookup[int(movieID)] = False
            title = dl.getMovieName(int(movieID))
            if (title):
                title = title.lower()
                for term in dl.getStopList():
                    if term in title:
                        #print ("Blocked ", title)
                        self.stoplistLookup[int(movieID)] = True  
  
                        
    def filterGenresToBitwise(self, genreFilter, dl):
        
        allGenres = dl.getGenres()
        maxGenreID = len(allGenres[1])
        
        if len(genreFilter)==0:
            print("\nNo Genre Filter Applied")
            return [], [], allGenres
        
        genreMap = dl.genreIDs
        genreFilterBit = [0] * maxGenreID
        
        for genre in genreFilter:
            if (genre.strip()).title() not in genreMap:
                print("\n" + (genre.strip()).title() + " skipped, it is not a valid Genre in this data set.")
                continue
            genreFilterBit[genreMap[(genre.strip()).title()]] = 1
            
        myGenresMode = [0]*maxGenreID
        totalGenreCount = 0.0
        for i in range(len(dl.myMovieIDs)):
            for j in range(len(myGenresMode)):
                myGenresMode[j] += allGenres[dl.myMovieIDs[i][0]][j] * (dl.myMovieIDs[i][1]/5.0)
                totalGenreCount += allGenres[dl.myMovieIDs[i][0]][j]
        if totalGenreCount > 0.0:
            for i in range(len(myGenresMode)):
                myGenresMode[i] = (1.0*myGenresMode[i])/totalGenreCount
            
        
        return genreFilterBit, myGenresMode, allGenres
        
        
    def computeYearSimilarity(self, years1, years2, stDev):
        
        diff = years1 - years2
        if stDev == 0:
            stDev = 1
        
        if diff > 0:
            sim = math.exp(-diff / (stDev*18.0))
        else:
            sim = math.exp(diff / (stDev*51.0))
            
        return sim        
    

        


