from pathlib import Path
import sys
from string import punctuation
import json
import pandas as pd
import matplotlib.pyplot as plt

class DatasetHendler():

    def __init__(self, reviews_path: str, labels_path: str, dataset_output_path: str) -> None:
        data_path = Path(reviews_path)
        labels_path = Path(labels_path)
        self.output_path = Path(dataset_output_path)

        self.data = self.__open_file(data_path)
        self.labels = self.__open_file(labels_path)

        self.__encode_text()
        self.__split_dataset()
        self.__save_datasets()

    def __open_file(self, file: Path) -> list:
        if not file.is_file():
            print("File not exist")
            sys.exit(-1)

        if not file.suffix == '.txt':
            print("File is not .txt")
            sys.exit(-1)

        with open(file) as f:
            data = f.read()

        return data
    
    def __split_dataset(self) -> None:
        split = int(0.8 * len(self.encoded_data))

        # Separazione dal dataset la parte per il training (80%)
        self.train_x = self.encoded_data[:split]
        self.train_y = self.encoded_labels[:split]
        x_val_test = self.encoded_data[split:]
        y_val_test = self.encoded_labels[split:]

        split = int(0.5 * len(x_val_test))

        # Separazione del dataset rimanente per la validazione e il testing
        self.val_x = x_val_test[:split]
        self.val_y = y_val_test[:split]
        self.test_x = x_val_test[split:]
        self.test_y = y_val_test[split:]

        print("Numero di recensioni per il training: ", len(self.train_x))
        print("Numero di recensioni per la validazione: ", len(self.val_x))
        print("Numero di recensioni per il test: ", len(self.test_x))

    def __save_datasets(self) -> None:
        output_path = self.output_path

        if not output_path.is_dir():
            output_path.mkdir()

        tr = Path(output_path / 'training')
        va = Path(output_path / 'validation')
        te = Path(output_path / 'test')
        
        if not tr.is_dir():
            tr.mkdir()
        if not va.is_dir():
            va.mkdir()
        if not te.is_dir():
            te.mkdir()
        
        with open(tr / 'tr_data.json', 'w') as f:
            json.dump(self.train_x, f)

        with open(tr / 'tr_labels.json', 'w') as f:
            json.dump(self.train_y, f)

        with open(va / 'va_data.json', 'w') as f:
            json.dump(self.val_x, f)

        with open(va / 'va_labels.json', 'w') as f:
            json.dump(self.val_y, f)

        with open(te / 'te_data.json', 'w') as f:
            json.dump(self.test_x, f)

        with open(te / 'te_labels.json', 'w') as f:
            json.dump(self.test_y, f)

    def __encode_text(self) -> None:
        # Rimozione di tutti i caratteri speciali e conversione delle lettere maiuscole in lettere minuscole
        self.data = ''.join([c for c in self.data if c not in punctuation]).lower()

        # Divisione di tutte le parole all'interno di una lista
        words = self.data.split()

        # Creazione di un dizionario momentaneo per eliminare tutti i valori duplicati
        words = list(dict.fromkeys(words))

        # Creazione di un dizionario con tutte le parole del dataset e il ralativo valore intero
        self.vocab = { x : i for i, x in enumerate(words, 1)}

        # Separazione dei dati
        self.data = self.data.split('\n')
        self.labels = self.labels.split('\n')

        # Rimozione di dati duplicati
        a = {}
        i = 0

        for l, d in zip(self.labels, self.data):
            a[i] = (l, d)
            i += 1
            
        del i

        a = list(dict.fromkeys(a.values()))
        self.labels = [x[0] for x in a]
        self.data = [x[1] for x in a]

        # Eliminazione di tutte le stringhe con lunghezza == 0
        deleted_index = []
        for i, x in enumerate(self.data):
            if(len(x) == 0):
                deleted_index.append(i)
                self.data.pop(i)
        
        # Codifica delle parole nel corrispettivo valore intero
        encoded_data = []
        for sentence in self.data:
            encoded_data.append([self.vocab[word] for word in sentence.split()])
        
        # Modificati a 200 elementi tutti i dati con maggiori parole.
        # Quelle con lunghezza minore verranno aggiunti zeri all'inizio
        sentence_length = 200
        for i, sentence in enumerate(encoded_data):
            actual_length = len(sentence)
            if actual_length > sentence_length:
                encoded_data[i] = sentence[:sentence_length]
            else:
                if actual_length < sentence_length:
                    tmp = [0 for _ in range(sentence_length - actual_length)]
                    tmp.extend(sentence)
                    encoded_data[i] = tmp
        
        self.encoded_data = encoded_data
        print("Prime 3 recensioni codificate:\n", encoded_data[:3])

        # Cancellazione delle etichette delle recensioni eliminate
        for i in deleted_index:
            self.labels.pop(i)

        # Codifica delle etichette. 0 se negativa e 1 se positiva
        encoded_labels = [1 if label == 'positive' else 0 for label in self.labels]
        self.encoded_labels = encoded_labels
        print("Prime 3 etichette codificate:\n", encoded_labels[:3])

        print("Numero di rencsioni: ", len(encoded_data))
        print("Numero di etichette: ", len(encoded_labels))
    
    def analyze_dataset(self) -> None:
        # Creazione del dataframe pandads
        data = pd.DataFrame({'data': self.data})
        labels = self.labels
        data['labels'] = labels
        
        # Quantità di dati
        print("\n--- Quantità di dati ---")
        print(data.count())
        plt.bar(data.columns, data.count())
        plt.title("Quantità di dati")
        plt.show()

        # Calcolo delle etichette presenti nel dataset
        print("\n--- Etichette presenti nel dataset ---")
        print(data["labels"].unique())

        # Confronto numero di etichette
        print("\n--- Numero di etichette ---")
        tmp = data.groupby('labels').count()
        print(tmp)
        plt.bar(tmp.index, tmp['data'])
        plt.title("Numero di etichette")
        plt.show()

        # Numero di parole utilizzate
        print("\n--- Numero di parole ---")
        print(len(self.vocab))

        # Lunghezza media dei dati
        print("\n--- Media del numero di parole per frase ---")
        print(data['data'].apply(len).mean() / 4)

        # Numero di dati con meno di 200 parole
        d = data['data'].apply(len)
        print("\n--- Numero di dati aventi meno di 200 parole")
        print(d[d < 800].count())

        # Numero di utilizzo di una parola
        vocab = {word : 0 for word in self.vocab}
        for x in data['data']:
            split = x.split()
            for i in split:
                vocab[i] += 1
        d = pd.DataFrame(vocab.keys(), columns=['indice'])
        d['valore'] = vocab.values()
        print("\n--- Numero di utilizzo per parola ---")
        print(d.sort_values(by = 'valore', ascending = False))

if __name__ == "__main__":
    dataset = DatasetHendler('./data/reviews.txt', './data/labels.txt', './dataset')

    dataset.analyze_dataset()