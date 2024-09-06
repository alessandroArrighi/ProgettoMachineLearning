import numpy as np

class Metrics():

    def __init__(self, classes: list[int], real_y: np.array, pred_y: np.array) -> None:
        self.classes = classes
        self.real_y = real_y
        self.pred_y = pred_y
        self.cofusion_matrix = None
    
    def calc_confusion_matrix(self) -> np.array:
        n = len(self.classes)
        self.cofusion_matrix = np.zeros((n, n), dtype = int)

        for pred, real in zip(self.pred_y, self.real_y):
            self.cofusion_matrix[real][int(pred)] += 1
        
        return self.cofusion_matrix
    
    def accuracy(self) -> float:
        if self.cofusion_matrix is None:
            self.calc_confusion_matrix()
        
        return np.sum(self.cofusion_matrix.diagonal()) / np.sum(self.cofusion_matrix)

if __name__ == "__main__":
    classes = [1, 0]
    
    real_y = np.array([1, 0, 1, 0], dtype = int)
    pred_y = np.array([1, 0, 1, 0], dtype = int)
    
    mt = Metrics(classes, real_y, pred_y)

    print("--- Confusion Matrix ---")
    print(mt.calc_confusion_matrix())
    print("Accuracy:", mt.accuracy())