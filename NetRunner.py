from ConfigChecker import ConfigChecker
from Net import Net
from CustomDatasetSentiment import CustomDatasetSentiment
from Metrics import Metrics
from DatasetHendler import DatasetHendler

from pathlib import Path
import sys
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class NetRunner():
    
    def __init__(self, conf: object, classes: list[int], vocab_size: int) -> None:
        self.conf = conf
        self.classes = classes

        self.__load_data()

        self.out_root = Path(conf.data.output_path)

        if not self.out_root.exists():
            self.out_root.mkdir()

        self.last_model_outpath_sd = self.out_root / 'last_model_sd.pth'
        self.last_model_outpath = self.out_root / 'last_model.pth'
        self.best_model_outpath_sd = self.out_root / 'best_model_sd.pth'
        self.best_model_outpath = self.out_root / 'best_model.pth'

        # Creazione della rete
        self.net = Net(classes, vocab_size, conf.network_params.embedding_dim, conf.network_params.hidden_size, conf.network_params.num_layers)

        if self.conf.train_params.use_last_model:
            self.__load_model(self.last_model_outpath_sd)
        
        if self.conf.train_params.use_best_model:
            self.__load_model(self.best_model_outpath_sd)

        # Verifica della GPU
        self.train_on_gpu = torch.cuda.is_available()

        if(self.train_on_gpu):
            print('Training on GPU.')
            self.net.cuda()
        else:
            print('No GPU available, training on CPU.')

        # Funzione di loss
        self.criterion = nn.BCELoss()
        # Ottimizzatore
        self.optimizer = optim.Adam(self.net.parameters(), lr = self.conf.hyper_params.learning_rate)

    def train(self) -> None:

        print("--- Training Mode ---")

        net = self.net

        # Numero di cicli effettuati
        global_step = 0
        
        epochs = self.conf.hyper_params.epochs
        
        # Hyper parametri
        validation_step = self.conf.train_params.validation_step
        train_step_monitor = self.conf.train_params.step_monitor
        accuracy_target = self.conf.train_params.accuracy_target

        # Valori delle loss
        tr_losses_x, tr_losses_y = [], []
        tr_run_losses_x, tr_run_losses_y = [], []
        va_losses_x, va_losses_y = [], []

        # Parametri dell'early stop
        epochs_check = self.conf.early_stop_params.epochs_check
        start_early_stop_check = self.conf.early_stop_params.start_check
        early_stop_improve_rate = self.conf.early_stop_params.improve_rate
        va_loss_not_improve_target = self.conf.early_stop_params.va_loss_target
        
        # Contatore in caso di non miglioramento
        va_not_improve_counter = 0

        # Flag per il funzionamento dell'early stop
        early_stop_check = False
        early_stop = False

        # Flag per il raggiungimento del target dell'accuracy
        on_target_acc = False

        # Valori migliori dell'addestramento
        best_tr_acc = None
        best_va_acc = None
        best_va_loss = None

        for epoch in range(epochs):
            
            # Controllo per l'early stop
            if (epoch + 1) == start_early_stop_check:
                early_stop_check = True
            
            if early_stop:
                print("Early stop triggered")
                break

            if on_target_acc:
                print("Target accuracy reached")
                break
            
            running_loss = 0.0
            for i, data in enumerate(self.tr_loader, 0):
                
                # Modalità training
                net.train()

                # Creazione di due variabili per salvare ad ogni ciclo il valore dell'etichetta e del dato in input
                inputs, labels = data

                if self.train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                net.zero_grad()

                # Chimata della forward
                outputs = net(inputs)

                # Funzione di costo in base a valori predetti e reali
                loss = self.criterion(outputs, labels.float())

                loss.backward()

                nn.utils.clip_grad_norm_(net.parameters(), 5)
                self.optimizer.step()

                running_loss += loss.item()

                # Passati ogni step_monitor cicli
                if (i + 1) % train_step_monitor == 0:
                    tr_run_losses_y.append(running_loss / train_step_monitor)
                    tr_run_losses_x.append(global_step)
                    print(f'global_step: {global_step:5d} - [ep: {epoch + 1:3d}, step: {i + 1:5d}] loss: {loss.item():.6f} - running_loss: {(running_loss / train_step_monitor):.6f}')
                    running_loss = 0.0
                
                tr_losses_y.append(loss.item())
                tr_losses_x.append(global_step)

                global_step += 1
            
            # Controllo della loss
            if early_stop_check and (epoch + 1) % epochs_check == 0:
                va_loss = 0
                va_loss_counter = 0

                for i, data in enumerate(self.va_loader, 0):
                    # Modalità di testing
                    net.eval()

                    inputs, labels = data

                    if self.train_on_gpu:
                        inputs, labels = inputs.cuda(), labels.cuda()
                    
                    with torch.no_grad():
                        outputs = net(inputs)

                        loss += self.criterion(outputs, labels.float())

                        va_loss += loss.item()
                        va_loss_counter += 1
                
                if best_va_loss is None:
                    best_va_loss = va_loss

                # Calcolo della loss
                total_va_loss = va_loss / va_loss_counter
                va_losses_x.append(global_step)
                va_losses_y.append(total_va_loss)

                # Controlli sull'andamento della loss per eseguire l'ealry stop se necessario
                if va_loss < best_va_loss:
                    improve_ratio = abs((va_loss / best_va_loss) - 1) * 100

                    if improve_ratio >= early_stop_improve_rate:
                        best_va_loss = va_loss
                        va_not_improve_counter = 0
                    else:
                        va_not_improve_counter += 1
                else:
                    va_not_improve_counter += 1

            # Validazione            
            if (epoch + 1) % validation_step == 0 or (epoch + 1) == epochs:
                
                # Esecuzione dei test con il set di training e validazione
                tr_acc = self.test(self.tr_loader)
                va_acc = self.test(self.va_loader)

                tr_improved = False
                va_improved = False

                if best_tr_acc is None:
                    best_tr_acc = tr_acc
                if best_va_acc is None:
                    best_va_acc = va_acc

                if tr_acc > best_tr_acc:
                    best_tr_acc = tr_acc
                    tr_improved = True

                if va_acc > best_va_acc:
                    best_va_acc = va_acc
                    va_improved = True

                if tr_improved and va_improved:
                    torch.save(net.state_dict(), self.best_model_outpath_sd)
                    torch.save(net, self.best_model_outpath)                

                print(f'Training accuracy: {(tr_acc):.2f}% - Validation accuracy: {(va_acc):.2f}%')
            
            if best_tr_acc is not None and best_va_acc is not None:
                if best_tr_acc >= accuracy_target and best_va_acc >= accuracy_target:
                    on_target_acc = True

            if va_not_improve_counter >= va_loss_not_improve_target:
                early_stop = True
        
        torch.save(net.state_dict(), self.last_model_outpath_sd)
        torch.save(net, self.last_model_outpath)

        _, (ax1, ax2) = plt.subplots(2, 1)
        
        ax1.plot(tr_losses_x, tr_losses_y, label='train_loss')
        ax1.plot(tr_run_losses_x, tr_run_losses_y, label='train_running_loss')
        ax1.set_title('Training loss')
        ax1.legend()
        
        ax2.plot(va_losses_x, va_losses_y, label='validation_loss')
        ax2.set_title('Validation loss')
        ax2.legend()

        plt.tight_layout()

        try:
            plt.savefig(self.conf.data.plots_path)
        except Exception:
            print("ERROR --- Cartella non esistente")
    
    def test(self, dataLoader: DataLoader = None):

        print("--- Testing Mode ---")

        print_acc = False
        
        # Se non specificato viene utilizzato il dataset di test
        if dataLoader is None:
            dataLoader = self.te_loader
            print_acc = True

        net = self.net

        net.eval()

        pred_y = []
        real_y = []

        with torch.no_grad():

            for i, data in enumerate(dataLoader):
                inputs, labels = data

                if self.train_on_gpu:
                    inputs, labels = inputs.cuda(), labels.cuda()

                outputs = net(inputs)

                # Arrotondati i valori a 1 - 0
                predicted = torch.round(outputs.squeeze())

                # Nel caso il tesore avesse un solo valore
                if len(outputs) == 1:
                    # Viene creato un tensore contenete un array di un elemento
                    predicted = torch.Tensor([predicted.item()])

                real_y += labels.tolist()
                pred_y += predicted.tolist()

        mt = Metrics(self.classes, real_y, pred_y)

        if print_acc:
            print(f'Test accuracy: {mt.accuracy():.2f}%')
            print(mt.cofusion_matrix)
        
        return mt.accuracy()

    
    def __load_data(self) -> None:
        tr_data = self.conf.data.tr_data_path
        tr_labels = self.conf.data.tr_labels_path
        va_data = self.conf.data.va_data_path
        va_labels = self.conf.data.va_labels_path
        te_data = self.conf.data.te_data_path
        te_labels = self.conf.data.te_labels_path

        tr_dataset = CustomDatasetSentiment(tr_labels, tr_data)
        va_dataset = CustomDatasetSentiment(va_labels, va_data)
        te_dataset = CustomDatasetSentiment(te_labels, te_data)

        self.tr_loader = DataLoader(tr_dataset, batch_size = self.conf.hyper_params.batch_size, shuffle = True)
        self.va_loader = DataLoader(va_dataset, batch_size = self.conf.hyper_params.batch_size, shuffle = False)
        self.te_loader = DataLoader(te_dataset, batch_size = self.conf.hyper_params.batch_size, shuffle = False)
    
    # Carica il modello dal path indicato
    def __load_model(self, model: Path) -> None:
        try:
            self.net.load_state_dict(torch.load(model))
        except:
            print('Cannot load model')
            sys.exit(-1)

if __name__ == "__main__":
    conf = ConfigChecker('./conf/conf.json', './conf/conf_schema.json')

    dataset = DatasetHendler('./data/reviews.txt', './data/labels.txt', './dataset')
    classes = CustomDatasetSentiment(conf.data.te_labels_path, conf.data.te_data_path)

    runner = NetRunner(conf, classes.classes, len(dataset.vocab) + 1)