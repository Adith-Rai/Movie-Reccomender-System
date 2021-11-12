# -*- coding: utf-8 -*-

class DataProcessing:
    
    def __init__(self, dataset):
        
        #Construct Train and Test data sets
        self.fullTrainSet = dataset.build_full_trainset()
        #self.fullAntiTestSet = self.fullTrainSet.build_anti_testset()
        
        #get train set and test set
        #self.trainSet, self.testSet = train_test_split(dataset, test_size=.1, random_state=1)
    
    
    def GetFullTrainSet(self):
        return self.fullTrainSet

    
    def GetUserAntiTestSet(self, user):
        trainset = self.fullTrainSet
        fill = trainset.global_mean
        anti_testset = []
        u = trainset.to_inner_uid(str(user))
        user_items = set([j for (j, _) in trainset.ur[u]])
        anti_testset += [(trainset.to_raw_uid(u), trainset.to_raw_iid(i), fill) for
                                 i in trainset.all_items() if
                                 i not in user_items]
        return anti_testset

    
    