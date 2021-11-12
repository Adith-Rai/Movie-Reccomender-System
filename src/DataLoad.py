import os
import csv
import sys
import re
import time
import shutil
import pickle as pk
import lzma
import statistics as stat
from collections import defaultdict

from surprise import Dataset
from surprise import Reader


class DataLoad:

    def __init__(self, maxRecords=sys.maxsize):
        paths={}
        with open('../config/config.config') as config:
            lines = config.readlines()
            for row in lines:
                row.replace(" ", "")
                paths[row.split("=")[0].strip()] = row.split("=")[1].strip()
            config.close()
        if not paths:
            print("\nCan NOT find file Paths in config file, please ensure config file is set up correctly.")
            
        self.ratingsPath = paths["ALL_RATINGS_PATH"]
        self.moviesPath = paths["MOVIES_PATH"]
        self.myRatingsPath = paths["MY_RATINGS_PATH"]
        self.cleanRatingsFile = paths["RATINGS_PATH_BKP"]
        self.rbmSavedData = paths["COMPUTED_RBM_WEIGHTS"]
        self.knnSavedData = paths["COMPUTED_SIMILARITY_MATRIX"]
        self.myReccomendationsPath = paths["MY_RECCOMENDATIONS_PATH"]
        self.genomeScorePath = paths["GENOME_SCORES_PATH"]
        self.tagsPath = paths["MOVIE_TAGS_PATH"]
        self.stoplistPath = paths["STOPLIST_PATH"]
        self.movieID_to_name = {}
        self.genreIDs = {}
        self.movieGenres = {}
        self.myMovieIDs = []
        self.maxRecords = maxRecords
        
    
    def pathsMissing(self):
        missing = 0
        if not os.path.exists(self.ratingsPath):
            print("Ratings Path specified in config file does NOT Exisit: " + self.ratingsPath)
            missing = 1
        if not os.path.exists(self.moviesPath):
            print("Movies Path specified in config file does NOT Exisit: " + self.moviesPath)
            missing = 1
        if not os.path.exists(self.myRatingsPath):
            print("Your Ratings Path specified in config file does NOT Exisit: " + self.myRatingsPath)
            missing = 1
        if not os.path.exists(self.rbmSavedData):
            print("Saved RBM Weights Path specified in config file does NOT Exisit: " + self.rbmSavedData)
            missing = 2
        if not os.path.exists(self.knnSavedData):
            print("Saved Content KNN Matrix Path specified in config file does NOT Exisit: " + self.knnSavedData)
            missing = 2
        if not os.path.exists(((self.myReccomendationsPath).rsplit("/",1))[0]):
            print("Path to Save Your Recommendations Not specified in config file: " + ((self.myReccomendationsPath).rsplit("/",1))[0])
            missing = 1
        if not os.path.exists(((self.genomeScorePath).rsplit("/",1))[0]):
            print("Path to Save Your Recommendations Not specified in config file: " + ((self.myReccomendationsPath).rsplit("/",1))[0])
            missing = 1
        if not os.path.exists(((self.stoplistPath).rsplit("/",1))[0]):
            print("Path to Save Your Recommendations Not specified in config file: " + ((self.myReccomendationsPath).rsplit("/",1))[0])
            missing = 1
        return missing
    
    def uploadUserData(self, userID):

        self.targetUser = int(userID)
        writeData = []
        
       
        #Create copy of ratings file
        if os.path.exists(self.cleanRatingsFile):
            self.cleanFiles()
        else:
            os.mkdir(((self.cleanRatingsFile).rsplit("/", 1))[0]) 
            shutil.copy2(self.ratingsPath, self.cleanRatingsFile)
        
        #Snip the ratings file to number of ratings specified in init
        self.snipDataFile()
        
        #Extract user ratings from my ratings file
        with open(self.myRatingsPath, newline='', encoding='ISO-8859-1') as csvfile:
            myRatingReader = csv.reader(csvfile)
            
            #Validate opened file fields and skip header line
            fields = next(myRatingReader)
            if len(fields) == 6 and fields == ['movie_id', 'imdb_id', 'tmdb_id', 'rating', 'average_rating', 'title']:
                print('File fields validation successful.')
            else:
                return 'File fields validation Failed. \nPlease enter a Legitimate MovieLens ratings csv file: movielens-ratings.csv - in the directory: ../MyRatings/'
            
            #Extract Data to be written to all users rating file, un-matched movie IDs will be dropped
            for row in myRatingReader:
                movieID = int(row[0])
                movieRating = float(row[3])
                if movieID in self.movieID_to_name:
                    writeData.append([userID, movieID, movieRating, str(int(time.time()))])
                    self.myMovieIDs.append([movieID, movieRating])
                    
            csvfile.close()
            
        #Write my ratings to all users ratings file
        with open(self.ratingsPath, 'a', newline='', encoding='ISO-8859-1') as csvwrite:
            writer = csv.writer(csvwrite)

            for row in writeData:
                writer.writerow(row)
            print("Ratings added to the collective ratings file for processing")
  
            csvwrite.close()
        
        #Load in Movie Lens data on init
        self.data = self.loadMovieLensRatingsData()
        
        return "Your ratings have Successfully been added to the user ratings file!"
    
    
    def snipDataFile(self):
        
        self.movieID_to_name = {}
        ratedMovies = {}
        
        #Reduce ratings file to only 3M records
        #copy ratings file into tmp file, and rename to raings file, delete temp file
        recordNo=0
        with open(self.ratingsPath, newline='', encoding='ISO-8859-1') as inp, open(self.ratingsPath+"-tmp", 'w+', newline='', encoding='ISO-8859-1') as out:
                movieReader = csv.reader(inp)
                movieWriter = csv.writer(out) 
                next(movieReader)  #Skip header line
                for row in movieReader:
                    if recordNo <= self.maxRecords:
                        movieID = int(row[1])                   
                        ratedMovies[movieID] = "rated"
                        movieWriter.writerow(row)
                        recordNo+=1
                    else:
                        break                    
                inp.close()
                out.close()
        os.remove(self.ratingsPath)
        shutil.copy2(self.ratingsPath+"-tmp", self.ratingsPath)
        os.remove(self.ratingsPath+"-tmp")
        
        #Get Movie ID to Movie Name Mapping
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  #Skip header line
            for row in movieReader:
                movieID = int(row[0])
                movieName = row[1]
                if (movieID in ratedMovies):
                    self.movieID_to_name[movieID] = movieName
                
            csvfile.close()
 
    
    def loadMovieLensRatingsData(self):

        # Look for files relative to the directory we are running from
        os.chdir(os.path.dirname(sys.argv[0]))

        reader = Reader(line_format='user item rating timestamp', sep=',', skip_lines=1)

        ratingsDataset = Dataset.load_from_file(self.ratingsPath, reader=reader)
        
        return ratingsDataset

    
    def getStopList(self):
        tmpstoplist=[]

        with open(self.stoplistPath) as stoplist:
            tmpstoplist = stoplist.readlines()
            for i in range(0, len(tmpstoplist)):
                tmpstoplist[i] = tmpstoplist[i].strip()
            stoplist.close()
        return tmpstoplist
    
    
    def writeReccomendationsToFile(self, reccomendations):
        
        genres = self.movieGenres
        popularity = self.getPopularityRanks()
        avgMovieRatings = self.getAvgRating()
            
        writeFileNameSplit = (self.myReccomendationsPath).split(".csv")
        writeFileName = writeFileNameSplit[0] + str(time.time()) + ".csv"
        with open(writeFileName, 'w+', newline='', encoding='ISO-8859-1') as csvwrite:
            writer = csv.writer(csvwrite)
            
            i = 1
            writer.writerow(['Rank','Movie Name', 'Your Projected Rating','Genres','Average Rating', 'Popular Rank'])
            for row in reccomendations:
                insert = [i, self.getMovieName(row[0]), row[1], genres[row[0]], avgMovieRatings[row[0]], popularity[row[0]]]
                writer.writerow(insert)
                i += 1
            print("\nYour reccomendations have been saved to the path: " + writeFileName )
  
            csvwrite.close()
            
            
    def writeLearnedWeightsToFile(self, knn=None, rbm=None):
        
        data = None
        Type = ""
            
        writeFileName = ""
        
        if (knn is not None):
            writeFileName = self.knnSavedData
            data = knn
            Type = "KNN"
            
        elif (rbm is not None):
            writeFileName = self.rbmSavedData
            data = rbm
            Type = "ExplainableRBM"
            
        else:
            print("\nNo Learned Data found" )
            return        

        if writeFileName == "":
            print("\nNo path to .dat file found")
            return 
        
        if data is None:
            print("\nNo Data to write found")
            return
            
        with lzma.open(writeFileName,'wb') as dataWriter:
            pk.dump(data, dataWriter)
            print("\n" + Type +" Learned Weights have been saved to the path: " + writeFileName + "\n")


    def readLearnedWeightsFromFile(self, knn=None, rbm=None):
        
        weights = None
        Type = ""
            
        readFileName = ""
        
        if (knn is not None):
            readFileName = self.knnSavedData
            Type = "KNN"
            
        elif (rbm is not None):
            readFileName = self.rbmSavedData
            Type = "RBM"
            
        else:
            print("\nNo Learned Data found" )
            return        

        if readFileName == "":
            print("\nNo path to .dat file found")
            return 
        try:
            with lzma.open(readFileName,'rb') as dataReader:
                weights = pk.load(dataReader)
                print("\n" + Type +" Learned Weights have been read from the path: " + readFileName)                     
        except (OSError, IOError) as e:
            print("No Saved Weights, Please Re-Train Model.")
 
            
        return weights
    
    
    
    def getPopularityRanks(self):
        ratings = defaultdict(int)
        rankings = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                ratings[movieID] += 1
                
            csvfile.close()
            
        rank = 1
        for movieID, ratingCount in sorted(ratings.items(), key=lambda x: x[1], reverse=True):
            rankings[movieID] = rank
            rank += 1
        return rankings

    
    def computePopularityWeights(self):
        popularity = self.getPopularityValues()
        maxPopularityValue = popularity[max(popularity, key = popularity.get)]
        
        for key in popularity:
            popularity[key] = 1.0*popularity[key]/maxPopularityValue  
           
        return popularity
    
    
    def getPopularityValues(self):
        ratings = defaultdict(int)
        ratingsTime = self.getRatingTime()
        
        minRateTime = ratingsTime[min(ratingsTime, key = ratingsTime.get)]-1
        maxRateTime = ratingsTime[max(ratingsTime, key = ratingsTime.get)]
        maxRateTimeDiff = maxRateTime - minRateTime
        
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)  
            
            for row in ratingReader:
                movieID = int(row[1])
                rateTime = int(row[3])
                ratings[movieID] += 1 * (1.0*(rateTime - minRateTime)/maxRateTimeDiff)
                
            csvfile.close()
            
        return ratings   
    
    
    def getRatingTime(self):

        ratingsTime = defaultdict(int)
        with open(self.ratingsPath, newline='') as csvfile:
            ratingReader = csv.reader(csvfile)
            next(ratingReader)
            for row in ratingReader:
                movieID = int(row[1])
                rateTime = int(row[3])
                ratingsTime[movieID] = rateTime    
                
            csvfile.close()
            
        return ratingsTime        
    
    
    def getGenres(self):
        genres = defaultdict(list)
        genreIDs = {}
        maxGenreID = 0
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  
            for row in movieReader:
                movieID = int(row[0])
                genreList = row[2].split('|')
                genreIDList = []
                for genre in genreList:
                    if genre in genreIDs:
                        genreID = genreIDs[genre]
                    else:
                        genreID = maxGenreID
                        genreIDs[genre] = genreID
                        maxGenreID += 1
                    genreIDList.append(genreID)
                genres[movieID] = genreIDList
                self.movieGenres[movieID] = row[2]
                
            csvfile.close()
            
        # Convert genre IDs to a bitfield vector
        for (movieID, genreIDList) in genres.items():
            bitfield = [0] * maxGenreID
            for genreID in genreIDList:
                bitfield[genreID] = 1
            genres[movieID] = bitfield            
        
        self.genreIDs = genreIDs
        return genres
    
    
    def getTags(self):
        tags = defaultdict(list)
        tagIDs = {}
        maxTagID = 0
        with open(self.tagsPath, newline='', encoding='ISO-8859-1') as csvfile:
            tagReader = csv.reader(csvfile)
            next(tagReader)  
            for row in tagReader:
                movieID = int(row[1])
                extractedTags = (row[2].strip()).upper()                
                if extractedTags in tagIDs:
                    tagID = tagIDs[extractedTags]
                else:
                    tagID = maxTagID
                    tagIDs[extractedTags] = tagID
                    maxTagID += 1
                tagIDList = tagID
                if movieID in tags:
                    tags[movieID].append(tagIDList)
                else:
                    tags[movieID] = [tagIDList]
                
            csvfile.close()
            
        # Convert tag IDs to a bitfield vector
        for (movieID, tagIDList) in tags.items():
            bitfield = [0] * maxTagID
            for tagID in tagIDList:
                bitfield[tagID] = 1
            tags[movieID] = bitfield   
            

        self.numTags = maxTagID
        
        return tags


    def getGenomeData(self):
        
        self.numGenomes = 0
        genomes = defaultdict(list)
        
        with open(self.genomeScorePath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)  
            movieID = 0
            for row in movieReader:
                movieID = int(row[0].strip())
                extractedGenome = float(row[2].strip())
                if movieID not in genomes:
                    genomes[movieID] = [extractedGenome]
                else:
                    genomes[movieID].append(extractedGenome)
            if movieID != 0:
                self.numGenomes = len(genomes[movieID])
            csvfile.close()
        
        return genomes    
 
    
    def getAvgRating(self):
        
        data = self.data
        data = data.build_full_trainset()
        
        avgMovieRatingTmp = {}
        
        for (uid, iid, rating) in data.all_ratings():
            
            thisMovieID = int(data.to_raw_iid(iid))
           
            #Generate a matrix for sum of ratings of each movie
            if thisMovieID in avgMovieRatingTmp:
                avgMovieRatingTmp[thisMovieID][0] += rating
                avgMovieRatingTmp[thisMovieID][1] += 1.0
            else:
                avgMovieRatingTmp[thisMovieID] = [rating,1.0]
           
        #Complete matrix of average rating of each movie, from sum of all ratings for each movie      
        avgMovieRating = {}
        for movieID in avgMovieRatingTmp:
            avgMovieRating[movieID] = avgMovieRatingTmp[movieID][0] / avgMovieRatingTmp[movieID][1]
            
        return avgMovieRating
    
    
    def getAvgUserYear(self):
        
        years = self.getYears()  
        data = self.data
        data = data.build_full_trainset()
        
        avgUserYearTmp = {}
        
        for (uid, iid, rating) in data.all_ratings():
            
            #Generate user all movie year dictionary
            thisUserID = int(data.to_raw_uid(uid))
            thisMovieID = int(data.to_raw_iid(iid))
            if thisUserID in avgUserYearTmp:
                avgUserYearTmp[thisUserID].append(years[thisMovieID])
            else:
                avgUserYearTmp[thisUserID] = [years[thisMovieID]]
        
        #Complete user average movie year and standard deviation dictionary
        avgUserYear = {}
        for user in avgUserYearTmp:
            if len(avgUserYearTmp[user])==1:
                avgUserYearTmp[user].append(avgUserYearTmp[user][0])  
                
            if user in avgUserYear:
                avgUserYear[user][0] = stat.median(avgUserYearTmp[user])
                avgUserYear[user][1] = stat.stdev(avgUserYearTmp[user])
            else:
                avgUserYear[user] = [stat.median(avgUserYearTmp[user]), stat.stdev(avgUserYearTmp[user])]
            
        return avgUserYear, years
  
    
    def getYears(self):
        extractYear = re.compile(r"(?:\((\d{4})\))?\s*$")
        years = defaultdict(int)
        with open(self.moviesPath, newline='', encoding='ISO-8859-1') as csvfile:
            movieReader = csv.reader(csvfile)
            next(movieReader)
            for row in movieReader:
                movieID = int(row[0])
                title = row[1]
                m = extractYear.search(title)
                year = m.group(1)
                if year:
                    years[movieID] = int(year)
                    
            csvfile.close()
                    
        return years


    def getMovieName(self, movieID):
        if movieID in self.movieID_to_name:
            return self.movieID_to_name[movieID]
        else:
            return ""

        
    #Restores all user ratings file to original
    def cleanFiles(self):
        copyTo = shutil.copy2(self.cleanRatingsFile, self.ratingsPath)
        
        print("\nMovie lens ratings csv file reset: " + copyTo)
        return "All user ratings csv file restored to path: " + copyTo
        
