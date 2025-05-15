import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms



def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    if training:
        train_set = datasets.FashionMNIST('./data', train=True, download=True, transform= custom_transform)
        loader = torch.utils.data.DataLoader(train_set, batch_size = 64)
    else:
        test_set = datasets.FashionMNIST('./data', train=False, transform= custom_transform)
        loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size = 64)
    #print(loader)
    #print(train_set.shape)
    #print(test_set.shape)
    return loader



def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 128),  
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
        )
    
    return model



def build_deeper_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),  
        nn.ReLU(),
        nn.Linear(256, 128),  
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),  
        nn.ReLU(),
        nn.Linear(32, 10)
        )
    
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        curr_loss = 0.0
        corr = 0
        total = 0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #print(loss)
            #print("loss")
            #print(labels.shape)
            loss.backward()
            opt.step()

            curr_loss += loss.item()

            index, predicted = torch.max(outputs.data, 1)
            #print(predicted)
            #print(predicted.shape)
            total += labels.size(0)
            corr += ((predicted == labels).long()).sum().item()

        accuracy = 100.0 * corr / total
        avg_loss = curr_loss / len(train_loader)

        print(f"Train Epoch: {epoch +1} Accuracy: {corr}/{total} ({accuracy:.2f}%) Loss: {avg_loss:.3f}")
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    model.eval()
    curr_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            curr_loss += loss.item()
            #print("loss")
            #print(loss)
            index, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += ((predicted == labels).long()).sum().item()

    accuracy = 100.0 * correct / total
    avg_loss = curr_loss / len(test_loader)
            
    if show_loss:
        print(f"Avvg loss: {avg_loss:.4f}")
        print(f'Accuracy: {accuracy:.2f}%')
    else:
        print(f'Accuracy: {accuracy:.2f}%')
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    with torch.no_grad():
        logits = model(test_images[index]) 
    prob = F.softmax(logits, dim=1)
    top_prob, top_class = torch.topk(prob, 3, dim=1)
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    for i in range(3):
        index = top_class[0][i] 
        name = class_names[index]
        perc = top_prob[0][i] * 100
        print(f"{name}: {perc:.2f}%")



if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)

    model = build_model()
    print(model)

    train_model(model, train_loader, criterion, 5)
    evaluate_model(model, test_loader, criterion, show_loss = False)
    evaluate_model(model, test_loader, criterion, show_loss = True)
    predict_label(model, next(iter(test_loader))[0], 1)
