import json
import pandas as pd
import contractions
import string

pd.options.mode.chained_assignment = None  # default='warn'

def remove_punctuation(org_str):
  org_str = org_str.replace("-", " ")
  org_str = org_str.translate(str.maketrans('', '', string.punctuation))
  return org_str

def remove_contractions(text):
    return contractions.fix(text)

train_data = pd.read_csv("cocotrain.csv")
test_data = pd.read_csv("cocoval.csv")

if __name__ == "__main__":

    print("Train Captions")
    print()
    
    del_train_idx = []
    del_test_idx = []

    for idx in range(0, len(train_data)):
    
        caption = train_data["captions"][idx].lower()

        caption = remove_contractions(caption)
        caption = remove_punctuation(caption)
        
        train_data["captions"][idx] = caption

        if idx % 10000 == 0:
            print("Train image ", idx, " completed")
    
   
    train_data.drop(del_train_idx, axis = 0, inplace=True)
    train_data.reset_index(inplace = True)

    print()
    print("Test Captions")
    print()

    for idx in range(len(test_data)):
        caption = test_data["captions"][idx].lower()
        
        caption = remove_contractions(caption)
        caption = remove_punctuation(caption)

        test_data["captions"][idx] = caption
            

        if idx % 10000 == 0:
            print("Test image ", idx, " completed")
    
    test_data.drop(del_test_idx, axis = 0, inplace=True)
    test_data.reset_index(inplace = True)
         

    for idx in range(len(train_data.path)):
        train_data["path"][idx] = "../../train2014/" + train_data.image_file[idx]
        # assuming path is directly outside current directory; change if need 

    for idx in range(len(test_data.path)):
        test_data["path"][idx] = "../../val2014/" + test_data.image_file[idx]
        # assuming path is directly outside current directory; change if need

    train_data.to_csv('edited_cocotrain.csv', index = False)
    test_data.to_csv('edited_cocoval.csv', index = False)
