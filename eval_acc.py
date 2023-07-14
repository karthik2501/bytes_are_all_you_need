import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from prettytable import PrettyTable

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(device,batch_size):
    path1 = '/home/notebook/data/group/cifar-10-batches-py/data_batch_1'
    path2 = '/home/notebook/data/group/cifar-10-batches-py/data_batch_2'
    path3 = '/home/notebook/data/group/cifar-10-batches-py/data_batch_3'
    path4 = '/home/notebook/data/group/cifar-10-batches-py/data_batch_4'
    path5 = '/home/notebook/data/group/cifar-10-batches-py/data_batch_5'
    path_test = '/home/notebook/data/group/cifar-10-batches-py/test_batch'
    path_d = '/home/notebook/data/group/cifar-10-batches-py/batches.meta'
    
    data_batch_1 = unpickle(path1)
    data_batch_2 = unpickle(path2)
    data_batch_3 = unpickle(path3)
    data_batch_4 = unpickle(path4)
    data_batch_5 = unpickle(path5)
    test_batch = unpickle(path_test)
    batches_meta = unpickle(path_d)
    
    training_data1,training_labels1 = torch.tensor(data_batch_1[b'data']).to(torch.int),data_batch_1[b'labels']
    training_data2,training_labels2 = torch.tensor(data_batch_2[b'data']).to(torch.int),data_batch_2[b'labels']
    training_data3,training_labels3 = torch.tensor(data_batch_3[b'data']).to(torch.int),data_batch_3[b'labels']
    training_data4,training_labels4 = torch.tensor(data_batch_4[b'data']).to(torch.int),data_batch_4[b'labels']
    training_data5,training_labels5 = torch.tensor(data_batch_5[b'data']).to(torch.int),data_batch_5[b'labels']
    test_data,test_labels = torch.tensor(test_batch[b'data']).to(torch.int),test_batch[b'labels']
    
    training_data = torch.cat((training_data1,training_data2,training_data3,training_data4,training_data5),dim=0)
    training_labels = []
    training_labels.extend(training_labels1)
    training_labels.extend(training_labels2)
    training_labels.extend(training_labels3)
    training_labels.extend(training_labels4)
    training_labels.extend(training_labels5)
    training_labels = torch.tensor(training_labels)

    training_labels = torch.LongTensor(training_labels)
    test_labels = torch.LongTensor(test_labels)
    
    train_dataset = TensorDataset(training_data.to(device), training_labels.to(device))
    test_dataset = TensorDataset(test_data.to(device), test_labels.to(device))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader


def main():
    path_d = '/home/notebook/data/group/cifar-10-batches-py/batches.meta'
    batches_meta = unpickle(path_d)
    class_names = batches_meta[b'label_names']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model = torch.jit.load('traced_byteformer_2.pt')
    model.eval()
    model.to(device)
    batch_size = 128
    train_loader,test_loader = load_data(device,batch_size)
    
    #### Measuring Train Accuracy
    correct = torch.zeros(10,device=device)
    total = torch.zeros(10,device=device)
    for input,label in train_loader:
        output = model(input)
        _, predicted = torch.max(output.data, 1)
        for i in range(10):
            class_mask = (label.flatten() == i)
            total[i] += torch.sum(class_mask)
            correct[i] += torch.sum(predicted[class_mask] == i)
    print("Training Accuracies : ")
    for i in range(10):
        print("Class {} : {:.6f}".format(class_names[i],(100*correct[i]/total[i])))
    print("Overall accuracy : {:.6f}".format((100*correct.sum()/total.sum())))
    
    #### Measuring Test Accuracy
    correct = torch.zeros(10,device=device)
    total = torch.zeros(10,device=device)
    for input,label in test_loader:
        output = model(input)
        _, predicted = torch.max(output.data, 1)
        for i in range(10):
            class_mask = (label.flatten() == i)
            total[i] += torch.sum(class_mask)
            correct[i] += torch.sum(predicted[class_mask] == i)
    print("Test Accuracies : ")
    for i in range(10):
        print("Class {} : {:.6f}".format(class_names[i],(100*correct[i]/total[i])))
    print("Overall accuracy : {:.6f} ".format((100*correct.sum()/total.sum())))
    
    
if __name__ == "__main__":
    main()