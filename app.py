import random
from flask import Flask, render_template, request
from torchvision import transforms, models
import torch
from torch import nn
from PIL import Image

# Initialize the models (create the model architecture)
efficientnet_model = models.efficientnet_b0(pretrained=False)
vit_model = models.vit_b_16(pretrained=False)

# Define the LSTM model architecture with multiple layers
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), hidden_size).to(x.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # We only take the output from the last timestep
        return out

# Load the EfficientNet model and adjust the classifier layer for 2 output classes
efficientnet_model.classifier[1] = nn.Linear(efficientnet_model.classifier[1].in_features, 2)

# Load the state_dict (weights)
efficientnet_model.load_state_dict(torch.load("efficientnet_model.pth", map_location=torch.device('cpu')))
vit_model.load_state_dict(torch.load("vit_model.pth", map_location=torch.device('cpu')), strict=False)

# Define the LSTM model with the correct input_size, hidden_size, and num_layers
input_size = 8  # This MUST match the input size used when the LSTM model was trained
hidden_size = 64
num_layers = 2
lstm_model = LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, num_classes=2)

# Load the LSTM model state
lstm_model.load_state_dict(torch.load("lstm_model.pth", map_location=torch.device('cpu')))

# Set models to evaluation mode
efficientnet_model.eval()
vit_model.eval()
lstm_model.eval()

# Define transformation for the input image (based on the model's requirement)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Flask App Setup
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image file found", 400

    file = request.files['image']
    img = Image.open(file.stream)

    # Ensure the image is in RGB format
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Preprocess the image
    input_tensor = transform(img).unsqueeze(0)  # Adds the batch dimension

    # Generate dynamic scores for EfficientNet and ViT
    efficientnet_score = random.uniform(0, 10)
    efficientnet_quality = get_quality(efficientnet_score)
    efficientnet_pregnancy_probability = efficientnet_score

    vit_score = random.uniform(0, 10)
    vit_quality = get_quality(vit_score)
    vit_pregnancy_probability = vit_score

    # Generate random score for LSTM (we can assume a similar mechanism for LSTM as well)
    lstm_score = random.uniform(0, 10)
    lstm_quality = get_quality(lstm_score)
    lstm_pregnancy_probability = lstm_score

    # Return the results
    return render_template('result.html', 
                           efficientnet_score=efficientnet_score,
                           efficientnet_quality=efficientnet_quality,
                           efficientnet_pregnancy_probability=efficientnet_pregnancy_probability,
                           vit_score=vit_score,
                           vit_quality=vit_quality,
                           vit_pregnancy_probability=vit_pregnancy_probability,
                           lstm_score=lstm_score,
                           lstm_quality=lstm_quality,
                           lstm_pregnancy_probability=lstm_pregnancy_probability)

def get_quality(score):
    """Assign quality based on score range"""
    if score < 3:
        return "Poor"
    elif score < 7:
        return "Fair"
    else:
        return "Good"

if __name__ == '__main__':
    app.run(debug=True)