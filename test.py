import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


from torchvision import datasets, transforms
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

# -------------------------------
# Define the sub-networks
# -------------------------------

class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        # A simple two‐layer convolutional network for CIFAR‑10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)  # reduce spatial dims
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        # After two poolings the 32x32 image becomes 8x8.
        self.fc    = nn.Linear(64 * 8 * 8, 128)
        # Produce a raw flag value; later we will add an external adjustment and use sigmoid.
        self.fc_flag = nn.Linear(128, 1)

    def forward(self, x, flag_adjustment):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 16x16 feature map
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 8x8 feature map
        x = x.view(x.size(0), -1)
        features = F.relu(self.fc(x))
        # Compute the flag value from features plus an external adjustment.
        raw_flag = self.fc_flag(features)
        # The flag is a number between 0 and 1.
        flag = torch.sigmoid(raw_flag + flag_adjustment)
        return features, flag


class BodyLarge(nn.Module):
    def __init__(self):
        super(BodyLarge, self).__init__()
        # A larger (deeper) fully connected transformation.
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class BodySmall(nn.Module):
    def __init__(self):
        super(BodySmall, self).__init__()
        # A smaller (simpler) transformation.
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x


class Tail(nn.Module):
    def __init__(self, num_classes=10):
        super(Tail, self).__init__()
        # A final classification layer mapping 128 features to 10 class scores.
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.fc(x)


# -------------------------------
# Combine into a complete model
# -------------------------------

class CompleteModel(nn.Module):
    def __init__(self, gamma):
        """
        gamma: hyperparameter used to penalize selecting the heavier body network.
        """
        super(CompleteModel, self).__init__()
        self.head = Head()
        self.body_large = BodyLarge()
        self.body_small = BodySmall()
        self.tail = Tail()
        self.gamma = gamma

    def forward(self, x, flag_adjustment):
        """
        x: input image batch.
        flag_adjustment: a scalar tensor that is added to the head's flag output.
        """
        features, flag = self.head(x, flag_adjustment)
        # Here we decide which body branch to use.
        # For simplicity, we take the mean flag over the mini-batch.
        flag_value = flag.mean()  
        if flag_value > 0.5:
            selected_body = "Large Body Network"
            body_out = self.body_large(features)
            # Apply a penalty gamma when using the large branch.
            penalty = self.gamma
        else:
            selected_body = "Small Body Network"
            body_out = self.body_small(features)
            penalty = 0.0

        logits = self.tail(body_out)
        return logits, flag, selected_body, penalty


# -------------------------------
# Example training loop
# -------------------------------

def train(model, dataloader, num_epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize flag_adjustment (this bias is updated each iteration)
    flag_adjustment = torch.tensor(0.0)
    prev_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            # Forward pass: compute logits and also get the flag and which body was chosen.
            logits, flag, selected_body, penalty = model(inputs, flag_adjustment)
            loss = criterion(logits, labels)
            # If the large body network was selected, add the gamma penalty.
            loss = loss + penalty
            loss.backward()
            optimizer.step()

            # Print which body was selected and where the gradients flowed.
            # (Only the branch that was used in the forward pass receives gradients.)
            print(f"Iteration {i}: Selected body: {selected_body}, Gradient passed through: {selected_body}")
            
            # Update the flag_adjustment:
            # If loss decreased compared to previous iteration, decrease flag_adjustment;
            # otherwise, increase it.
            if loss.item() < prev_loss:
                flag_adjustment = flag_adjustment - 0.01
            else:
                flag_adjustment = flag_adjustment + 0.01
            # Clamp the flag_adjustment to a reasonable range to keep flag (after sigmoid) in (0,1)
            flag_adjustment = torch.clamp(flag_adjustment, -5.0, 5.0)
            prev_loss = loss.item()

-------------------------------
-------------------------------

  

model = CompleteModel(gamma=2)
train(model, train_loader, num_epochs=10, lr=0.001)

