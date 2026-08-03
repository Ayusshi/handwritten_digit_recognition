import torch
import torchvision.transforms as transforms
from PIL import Image
import config
from model import CNN
from preprocess import preprocess_image

# print(config.__file__)
# print(dir(config))

# exit()

# -----------------------------------
# Device
# -----------------------------------
device = torch.device(config.DEVICE)


# -----------------------------------
# Load model
# -----------------------------------
model = CNN().to(device)

model.load_state_dict(
    torch.load(
        config.MODEL_PATH,
        map_location=device
    )
)

model.eval()


# -----------------------------------
# Transform
# -----------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])


# -----------------------------------
# Prediction Function
# -----------------------------------
def predict(image_path):

    processed = preprocess_image(image_path)

    image = Image.fromarray(processed)

    image = transform(image)

    image = image.unsqueeze(0)

    image = image.to(device)

    with torch.no_grad():

        output = model(image)

        prediction = output.argmax(dim=1).item()

    return prediction


# -----------------------------------
# Run
# -----------------------------------
if __name__ == "__main__":

    image_path = "test_images/digit3.jpg"

    prediction = predict(image_path)

    print("=" * 40)
    print(f"Predicted Digit : {prediction}")
    print("=" * 40)