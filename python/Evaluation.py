import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights

#model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model = torch.load("EmberAlert.pth", map_location=torch.device('cpu'), weights_only=False)
#model = model.load_state_dict(torch.load('EmberAlert.pth', map_location=torch.device('cpu')))

# Load the image
path = r"C:\Users\lufai\Downloads\bus.jpg"
model_path = ''
def evaluation(image_path):
    img = Image.open(image_path)

    # Define the transformations
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Apply the transformations
    img_tensor = transform(img).float()

    # Add the batch dimension
    img_tensor.unsqueeze_(0)

    # Load the model
    #model = torch.load('model_path')
    model.eval()

    # Get the model output
    output = model(img_tensor)

    # Get the predicted class
    _, predicted_class = torch.max(output, 1)

    print(f'Predicted class: {predicted_class.item()}')
    return predicted_class.item()

answer = evaluation(path)