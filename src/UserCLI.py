import sys

class UserCLI:
    
    def __init__(self, missing):
        if missing==1:
            print("\nPlease set up all Paths in config file")
            sys.exit()  
            
        self.missing = missing

    
    def optionSelecter(self):
        
        train = None
        
        # User option selecter
        while train is None:
            choice = input("What would you like to do?\n (1) Give my recommendations, computing results only for me\n (2) Re-train the model and Re-build all similarity matrices and give my recommendations, computing results for all users\n (3) Exit\n\nPlease enter 1, 2 or 3 for option (1), option (2) or option (3) respectively: ")
            if choice.strip() == '2':
                choice2 = input("\nRe-training may take several minutes to hours or longer based on system specifications.\nThere is a chance your system may not be able to perform or complete the training.\nAre you sure you would like to re-train? [y/n]")
                if choice2.strip() == 'y' or choice2 == 'Y':
                    train = True
                    print("\nModel will be re-trained\n")
                else:
                    print("\nPlease choose an option again")
            elif choice.strip()=='3':
                train = False
                sys.exit()  
            elif choice.strip()=='1':
                if (self.missing==2):
                    print("No Saved Data for Trained models, Please Re-Train")
                    train = True
                    sys.exit()  
                else:
                    train = False
            else:
                print("Your Choice was invalid, please enter your choice again.")
                
        return train
    
    
    def enterGenreFilter(self, train):
        genreFilter = []
        if not train:
            genreChoice = input("\nWould you like to get only specific combination of Genres? [y/n] ")
            if genreChoice.strip() == 'y' or genreChoice == 'Y':
                print("\nPlease Enter Genres: [type 'done' when finished]")
                inputGenre = "Empty"
                while (inputGenre.strip()).title()!="Done" and (inputGenre.strip()).title()!="" and (inputGenre.strip()).title()!="None":
                    inputGenre = input("Enter Genre: ")
                    if (inputGenre.strip()).title()!="Done" and (inputGenre.strip()).title()!="" and (inputGenre.strip()).title()!="None":
                        genreFilter.append(inputGenre.strip())
                        
        return genreFilter

