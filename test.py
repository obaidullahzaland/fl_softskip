import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)  # reduce spatial dims
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.fc    = nn.Linear(64 * 8 * 8, 128)
        self.fc_flag = nn.Linear(128, 1)

    def forward(self, x, flag_adjustment):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 16x16 feature map
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 8x8 feature map
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc(x))
        raw_flag = self.fc_flag(features)
        # Flag for decision 0 and 1
        flag = torch.sigmoid(raw_flag + flag_adjustment)
        return features, flag


class BodyLarge(nn.Module):
    def __init__(self):
        super(BodyLarge, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class BodySmall(nn.Module):
    def __init__(self):
        super(BodySmall, self).__init__()
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x


class Tail(nn.Module):
    def __init__(self, num_classes=10):
        super(Tail, self).__init__()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(x)


class CompleteModel(nn.Module):
    def __init__(self, gamma):

        super(CompleteModel, self).__init__()
        self.head = Head()
        self.body_large = BodyLarge()
        self.body_small = BodySmall()
        self.tail = Tail()
        self.gamma = gamma

    def forward(self, x, flag_adjustment):

        features, flag = self.head(x, flag_adjustment)

        flag_value = flag.mean()  
        if flag_value > 0.5:
            selected_body = "Large Body Network"
            body_out = self.body_large(features)
            penalty = self.gamma
        else:
            selected_body = "Small Body Network"
            body_out = self.body_small(features)
            penalty = 0.0

        logits = self.tail(body_out)
        return logits, flag, selected_body, penalty
    

class BiggerModel(nn.Module):
    def __init__(self, gamma):

        super(BiggerModel, self).__init__()
        self.head = Head()
        self.body_large = BodyLarge()
        self.tail = Tail()

    def forward(self, x, flag_adjustment):

        features, flag = self.head(x, flag_adjustment)
        penalty = 0
        flag_value = flag.mean()  
        selected_body = "L"
        body_out = self.body_large(features)
        logits = self.tail(body_out)
        return logits, flag, selected_body, penalty
    
class SmallerModel(nn.Module):
    def __init__(self, gamma):

        super(SmallerModel, self).__init__()
        self.head = Head()
        self.body_large = BodySmall()
        self.tail = Tail()

    def forward(self, x, flag_adjustment):

        features, flag = self.head(x, flag_adjustment)
        penalty = 0
        flag_value = flag.mean()  
        selected_body = "S"
        body_out = self.body_large(features)
        logits = self.tail(body_out)
        return logits, flag, selected_body, penalty


def train(model, dataloader, num_epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    flag_adjustment = torch.tensor(0.0)
    prev_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            logits, flag, selected_body, penalty = model(inputs, flag_adjustment)
            loss = criterion(logits, labels)
            loss = loss + penalty
            loss.backward()
            optimizer.step()

            # Print main and gradient flowing body 
            # print(f"Iteration {i}: Selected body: {selected_body}, Gradient passed through: {selected_body}")
            
            if loss.item() < prev_loss:
                flag_adjustment = flag_adjustment - 0.01
            else:
                flag_adjustment = flag_adjustment + 0.01
            # flag adjustment
            flag_adjustment = torch.clamp(flag_adjustment, -5.0, 5.0)
            prev_loss = loss.item()

    return model 
def test(model):
    # Define transform (same as for training)
    
    model.eval()  # set model to evaluation mode
    correct = 0
    total = 0
    # Use a constant flag_adjustment (e.g. 0.0) during testing.
    flag_adjustment = torch.tensor(0.0)
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            # Forward pass through the model
            logits, flag, selected_body, penalty = model(inputs, flag_adjustment)
            _, predicted = torch.max(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


model = CompleteModel(gamma=0.1)
bmodel = BiggerModel(gamma=0.1)
smodel = SmallerModel(gamma=0.1)
model = train(model, train_loader, num_epochs=25, lr=0.001)
print("Main Network accuracy")
acc = test(model)
bmodel = train(bmodel, train_loader, num_epochs=25, lr=0.001)

print("Main Network accuracy")
acc = test(bmodel)
smodel = train(bmodel, train_loader, num_epochs=25, lr=0.001)

print("Main Network accuracy")
acc = test(smodel)