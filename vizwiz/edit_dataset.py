import json
from PIL import Image
import pandas as pd
import contractions
import string
import inflect

pd.options.mode.chained_assignment = None  # default='warn'

def remove_punctuation(org_str):
  org_str = org_str.replace("-", " ")
  org_str = org_str.translate(str.maketrans('', '', string.punctuation))
  return org_str

def remove_contractions(text):
    return contractions.fix(text)

with open('annotations/train.json') as f:
    train_annotations = json.load(f)

with open('annotations/val.json') as f:
    val_annotations = json.load(f)

train_captions = []
train_image_id = []
val_captions = []
val_image_id = []

for caption_data in train_annotations["annotations"]:
    train_captions.append(caption_data["caption"])
    train_image_id.append(caption_data["image_id"])

for caption_data in val_annotations["annotations"]:
    val_captions.append(caption_data["caption"])
    val_image_id.append(caption_data["image_id"])

train_image_file = [train_annotations['images'][img]['file_name'] for img in train_image_id]
val_image_file = [val_annotations['images'][img - 23431]['file_name'] for img in val_image_id]  # val image ids come after train ids

train_data = pd.DataFrame({'image_file': train_image_file, 'captions': train_captions})
test_data = pd.DataFrame({'image_file': val_image_file, 'captions': val_captions})

if __name__ == "__main__":

    print("Train Captions")
    print()
    
    del_train_idx = []
    del_test_idx = []

    for idx in range(0, len(train_data)):
    
        caption = train_data["captions"][idx].lower()
        
        if "quality issues" in caption:
            del_train_idx.append(idx)
            continue

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
        
        if "quality issues" in caption:
            del_test_idx.append(idx)
            continue
        
        caption = remove_contractions(caption)
        caption = remove_punctuation(caption)

        test_data["captions"][idx] = caption
            

        if idx % 10000 == 0:
            print("Test image ", idx, " completed")
    
    test_data.drop(del_test_idx, axis = 0, inplace=True)
    test_data.reset_index(inplace = True)
         

    for idx in range(len(train_data.image_file)):
        train_data["image_file"][idx] = "../train/" + train_data.image_file[idx]
        # assuming path is directly outside current directory; change if need 

    for idx in range(len(test_data.image_file)):
        test_data["image_file"][idx] = "../val/" + test_data.image_file[idx]
        # assuming path is directly outside current directory; change if need

    train_data.to_csv('train.csv', index = False)
    test_data.to_csv('test.csv', index = False)
