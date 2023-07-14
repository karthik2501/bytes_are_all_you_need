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

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(0, max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pos_encoding = torch.zeros(max_seq_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        pos_encoding = pos_encoding.unsqueeze(0)

        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:, :x.size(1)].clone().detach()
        return self.dropout(x)
    
class ByteFormer(nn.Module):
    def __init__(self,d_model,nhead,num_layers,embedding_dim,kernel_size,stride,num_classes,max_seq_length,dropout_rate):
        super(ByteFormer, self).__init__()
        self.embedding_layer = nn.Embedding(256,embedding_dim)
        self.conv = nn.Conv1d(embedding_dim,embedding_dim,kernel_size,stride)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length+1)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead,dropout=dropout_rate)
        # self.cls_token = nn.Parameter(torch.rand(1, 1, embedding_dim))
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,num_layers)
        self.fc = nn.Linear(embedding_dim,num_classes)
        self.output = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.embedding_layer(x)
        x = torch.transpose(x,1,2)
        x = torch.transpose(self.conv(x),2,1)
        # x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        # x = x[:,-1,:]
        n = x.shape[1]
        x = torch.sum(x,1)/n
        # x = F.relu(self.fc1(x))
        x = self.fc(x)
        # x = self.output(x)
        return x

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

def train_model(model,train_loader,test_loader,learning_rate,num_epochs,criterion,optimizer,lambda_reg):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # Mini-batch training loop
        for inputs,labels in train_loader:

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
             # Regularization term
            regularization_loss = 0.0
            for param in model.parameters():
                regularization_loss += torch.norm(param, p=2)**2  # L2 regularization
                
            num_parameters = sum(p.numel() for p in model.parameters())
            regularization_loss /= (2*num_parameters)
            # Add regularization term to the loss
            loss += lambda_reg * regularization_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Track training loss and accuracy
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate average training loss and accuracy
        train_loss /= (len(train_loader))
        train_accuracy = (100 * correct) / total

        # Evaluate the model on the test set
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            # Evaluation on the test set
            for inputs,labels in test_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                regularization_loss = 0.0
                for param in model.parameters():
                    regularization_loss += torch.norm(param, p=2)**2  # L2 regularization

                num_parameters = sum(p.numel() for p in model.parameters())
                regularization_loss /= (2*num_parameters)
                # Add regularization term to the loss
                loss += lambda_reg * regularization_loss

                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                # print(predicted[0],labels[0])
                correct += (predicted == labels).sum().item()

        # Calculate average test loss and accuracy
        test_loss /= (len(test_loader))
        test_accuracy = 100 * correct / total

        print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}% - "
              f"Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.2f}%")
        
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 128
    learning_rate = 0.0005
    num_epochs = 200
    
    train_loader,test_loader = load_data(device,batch_size)
    
    lambda_reg = 0
    d_model = 36
    nhead = 3
    num_layers = 3
    embedding_dim = 36
    kernel_size = 32
    stride = 8
    num_classes = 10
    max_seq_length = (3072-kernel_size)//stride +1
    dropout_rate = 0.2
    
    model = ByteFormer(d_model,nhead,num_layers,embedding_dim,kernel_size,stride,num_classes,max_seq_length,dropout_rate)
    # model = torch.jit.load('traced_byteformer_2.pt')
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),weight_decay=0, lr=learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum = 0.9)
    # count_parameters(model)
    
    train_model(model,train_loader,test_loader,learning_rate,num_epochs,criterion,optimizer,lambda_reg)
    
    example = torch.randint(0, 256, (1,3072))
    model.eval()
    model.to('cpu')
    # print(model(example))
    traced_script_module = torch.jit.trace(model, example)
    traced_script_module.save("traced_byteformer_2.pt")
    
if __name__ == "__main__":
    main()
    
    
