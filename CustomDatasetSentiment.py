import sys
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import json

class CustomDatasetSentiment(Dataset):

    def __init__(self, labels_path: str, data_path: str) -> None:
        self.labels_path = Path(labels_path)
        self.data_path = Path(data_path)

        if not self.labels_path.exists():
            print("File non esistente")
            sys.exit(-1)
        
        if not self.data_path.exists():
            print("File non esistente")
            sys.exit(-1)
        
        if not self.__is_txt_file(self.labels_path):
            sys.exit(-1)
        
        if not self.__is_txt_file(self.data_path):
            sys.exit(-1)

        print("--- Caricamento del Dataset ---")
        
        self.encoded_labels = self.__open_file(self.labels_path)
        self.encoded_data = self.__open_file(self.data_path)
                
        self.classes = []

        for x in self.encoded_labels:
            if x not in self.classes:
                self.classes.append(x)

    def __open_file(self, file_path) -> str:
        with open(file_path) as f:
            data = json.load(f)
        return data
            
    def __is_txt_file(self, file) -> bool:
        if file.suffix != ".json":
            print("Il file deve essere .json")
            return False
        return True
    
    def __getitem__(self, index):
        data = np.asarray(self.encoded_data[index])
        return data, self.encoded_labels[index]

    def __len__(self) -> int:
        return len(self.encoded_data)

if __name__ == "__main__":
    a = CustomDatasetSentiment('./dataset/training/tr_labels.json', './dataset/training/tr_data.json')