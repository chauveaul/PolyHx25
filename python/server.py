from flask import Flask
from flask import request

# from appwrite.client import Client
# from appwrite.services.users import Users
# from appwrite.exception import AppwriteException
# from appwrite.services.storage import Storage
import os
import torch
import torchvision.transforms as transforms
import io
from PIL import Image


app = Flask(__name__)

# client = (
#    Client()
#    .set_endpoint("https://cloud.appwrite.io/v1")
#    .set_project("67a849a20022902df5d1")
# )
#
# storage = Storage(client)


@app.route("/", methods=["POST"])
def hello_world():
    # fileId = request.json.body["fileId"]

    # result = storage.get_file_preview(
    #    bucket_id="67a84b5e002984581076",
    #    file_id=fileId,
    # )
    # print("result")

    file = request.files["image"]

    model = torch.load(
        "EmberAlert.pth", map_location=torch.device("cpu"), weights_only=False
    )

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Apply the transformations
    img_tensor = transform(file).float()

    # Add the batch dimension
    img_tensor.unsqueeze_(0)

    # Load the model
    # model = torch.load('model_path')
    model.eval()

    # Get the model output
    output = model(img_tensor)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    print(f"Predicted class: {predicted_class.item()}")
    if predicted_class.item() == 0:
        return "wildfire", 200
    elif predicted_class.item() == 1:
        return "nowildfire", 200
    else:
        return "error", 400
