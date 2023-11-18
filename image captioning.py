import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO

# Load pre-trained ResNet model
resnet = models.resnet50(pretrained=True)
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
resnet.eval()

# Define a simple LSTM-based captioning model
class CaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(CaptioningModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        captions = captions[:, :-1]
        embeddings = self.embedding(captions)
        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.linear(lstm_out)
        return outputs

# Download a sample image
image_url = "https://example.com/sample.jpg"
response = requests.get(image_url)
img = Image.open(BytesIO(response.content))

# Preprocess the image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

# Extract features using the ResNet model
with torch.no_grad():
    features = resnet(img_tensor)
features = features.squeeze()

# Define vocabulary and model parameters
vocab_size = 1000  # Replace with the actual vocabulary size
embed_size = 256
hidden_size = 512
num_layers = 1

# Instantiate the captioning model
caption_model = CaptioningModel(embed_size, hidden_size, vocab_size, num_layers)

# Generate a sample caption
sample_caption = torch.randint(1, vocab_size, (1, 20))  # Replace 20 with the maximum caption length
output = caption_model(features, sample_caption)
print("Sample Caption Output Shape:", output.shape)
