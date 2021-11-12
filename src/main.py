import time

from UserCLI import UserCLI
from DataLoad import DataLoad
from DataProcessing import DataProcessing
from ExplainableRBM import ExplainableRBM
from GenerateRecommendations import GenerateRecommendations

import random
import numpy as np


#Generate unique user ID using date time
myNewUserId = int(time.time())
numRecs = 25

np.random.seed(0)
random.seed(0)

#Initialize File Paths
#dl = DataLoad(1000000) - max 1M ratings for 4GB RAM - will take many many hours
#dl = DataLoad(3000000) - max 3M ratings for 8GB RAM - will take many many hours, or even upto a day
#dl = DataLoad(4500000) - max 4.5M ratings for 12BG RAM - will take many many hours, or even upto a day or two
#dl = DataLoad(7000000) - max 7M ratings for 16GB RAM - will take many many hours, or even upto a day or two
dl = DataLoad(100000)



#Determine if Required files Exist
missing = dl.pathsMissing()

# User option selecter
userCLI = UserCLI(missing)
train = userCLI.optionSelecter()
        
#Insert my ratings data into all user ratings
dl.uploadUserData(myNewUserId)  

# Construct an DataProcessing to get train and test set
dataProcessor = DataProcessing(dl.data)
reccomender = GenerateRecommendations(dl.data, train)

#Enter Genre Filter
genreFilter = userCLI.enterGenreFilter(train)

#KNN Item Smilarities to make rating prediction for unrated movies and Densify Sparse Movie Ratings Matrix
#KNN User Similarities to generate movie ratiends for each user based on ratings of similar users -
# - with previously densified matrix as input
#Both Previously computed Matrices as two visible visible dimentions to RBM To Predict Movie Ratings - 
# - for user/or all users if re-training
explainableRBM = ExplainableRBM(dl)

#Reccomend the movies
reccomender.ComputeRecs(explainableRBM, dl, myNewUserId, numRecs, genreFilter)

    


    