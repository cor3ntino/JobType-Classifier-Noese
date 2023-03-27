import random
import json

def split_train_val(dataset_dir, proportion_validation):

    with open(dataset_dir, "r") as f:
        data = json.load(f)

    train_data, val_data = {}, {}
    for k, v in data.items():
    
        if random.random() < proportion_validation:
            val_data[k] = v
        else:
            train_data[k] = v
        
    return train_data, val_data
