import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import random
import torch.optim as optim

class SiameseDataset(Dataset):
    def __init__(self, uniform_dir, casual_dir, transform=None):
        self.uniform_images = [os.path.join(uniform_dir, f) for f in os.listdir(uniform_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
        self.casual_images = [os.path.join(casual_dir, f) for f in os.listdir(casual_dir) if f.endswith(('jpg', 'png', 'jpeg'))]
        self.transform = transform

    def __len__(self):
        return max(len(self.uniform_images), len(self.casual_images))

    def __getitem__(self, idx):
        if random.random() > 0.5:
            img1_path = random.choice(self.uniform_images)
            img2_path = random.choice(self.uniform_images)
            label = 1
        else:
            img1_path = random.choice(self.uniform_images)
            img2_path = random.choice(self.casual_images)
            label = 0

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor(label, dtype=torch.float32)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_dataset = SiameseDataset('./data/positive', './data/negative', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='efficientnet-b0'):
        super(FeatureExtractor, self).__init__()
        
        # Load the EfficientNet model pre-trained on ImageNet
        self.base_model = EfficientNet.from_pretrained(model_name)
        
        # Freeze the weights of the base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Remove the classification head
        self.base_model._fc = nn.Identity()
        
    def forward(self, x):
        # Forward pass through the base model
        x = self.base_model(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, feature_extractor, use_l1_dist=False):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.use_l1_dist = use_l1_dist
        
    def forward(self, input1, input2):
        output1 = self.feature_extractor(input1)
        output2 = self.feature_extractor(input2)
        # print("output shape: ", output1.shape)
        if self.use_l1_dist:
            distance = torch.abs(output1 - output2)
        else:
            distance = torch.sqrt(torch.sum((output1 - output2) ** 2, dim=1, keepdim=True))
        # print("distance shape: ", distance)
        return distance

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(1280, 1)
    
    def forward(self, x):
        # x = x.view(x.size(0), -1)  # Flatten the tensor
        # print("shape: ", x.shape)
        x = torch.sigmoid(self.fc(x))
        return x

def contrastive_loss(y_true, y_pred, margin=1.0):
    y_true = y_true.float()
    y_pred = y_pred.squeeze()
    loss = y_true * torch.pow(y_pred, 2) + (1 - y_true) * torch.pow(torch.clamp(margin - y_pred, min=0.0), 2)
    return loss.mean()

# Define the input shape (image dimensions)
input_shape = (3, 224, 224)  # PyTorch uses (C, H, W) format

# Build the feature extractor
feature_extractor = FeatureExtractor().cuda()
siamese_model = SiameseNetwork(feature_extractor, use_l1_dist=False).cuda()
classifier = Classifier().cuda()

# Print the model summary
# print(siamese_model)
# print(classifier)

# To get a detailed summary, you can use torchsummary library
# from torchsummary import summary
# summary(siamese_model, [(3, 224, 224), (3, 224, 224)])
# summary(classifier, (1280,))

def train_siamese(model, classifier, train_loader, criterion, optimizer, num_epochs=25):
    model.train()
    classifier.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs1, inputs2, labels in train_loader:
            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

            optimizer.zero_grad()

            distance = model(inputs1, inputs2)
            outputs = classifier(distance)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs1.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

    print('Training complete')

# Initialize the model, loss function, and optimizer
feature_extractor = FeatureExtractor().cuda()
siamese_model = SiameseNetwork(feature_extractor, use_l1_dist=False).cuda()
classifier = Classifier().cuda()
criterion = contrastive_loss
optimizer = optim.Adam(list(siamese_model.parameters()) + list(classifier.parameters()), lr=0.001)

# Train the model
train_siamese(siamese_model, classifier, train_loader, criterion, optimizer, num_epochs=100)

def evaluate_siamese(model, classifier, data_loader):
    model.eval()
    classifier.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs1, inputs2, labels in data_loader:
            inputs1, inputs2, labels = inputs1.cuda(), inputs2.cuda(), labels.cuda()

            distance = model(inputs1, inputs2)
            outputs = classifier(distance)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f}')

# Evaluate the model
test_dataset = SiameseDataset('./data/positive', './data/negative', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
evaluate_siamese(siamese_model, classifier, test_loader)

