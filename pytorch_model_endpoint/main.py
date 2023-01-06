from flask import Flask, request, Response, jsonify
from flask_cors import CORS

import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np

app = Flask(__name__)
CORS(app)

# Declare the ConvNet (same as in training scripts)
class ConvNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model
model = ConvNet()
model.load_state_dict(torch.load("./model/model.pth"))
print("Loaded model")

# Check for GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model = model.to(device)

# Preprocessing and postprocessing
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', \
           'dog', 'frog', 'horse', 'ship', 'truck']

# Health check route
@app.route("/isalive")
def is_alive():
    print("/isalive request")
    status_code = Response(status=200)
    return status_code

# Predict route
@app.route("/predict", methods=["POST"])
def predict():
    print("/predict request")
    req_json = request.get_json()
    json_instances = req_json["instances"]
    X_list = [np.array(j["image"], dtype="uint8") for j in json_instances]
    X_transformed = torch.cat([transform(x).unsqueeze(dim=0) for x in X_list]).to(device)
    preds = model(X_transformed)
    preds_classes = [classes[i_max] for i_max in preds.argmax(1).tolist()]
    print(preds_classes)
    return jsonify({
        "predictions": preds_classes
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8081)