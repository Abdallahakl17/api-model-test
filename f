# from fastapi import FastAPI, UploadFile, File
# from torchvision import models, transforms
# from PIL import Image
# import io
# import torch
# app = FastAPI()
# # Load the model
# model = models.resnet50(pretrained=True)
# num_features = model.fc.in_features
# model.fc = torch.nn.Linear(num_features, 2) 
#  # Assuming your model has 2 output classes
# torch.save(model.state_dict(), "graduation_progect_cnn.py")

# model.load_state_dict(torch.load("graduation_progect_cnn.py", map_location=torch.device('cpu')))
# model.eval()

# # Define transformations
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Define function to process uploaded image
# def process_image(file):
#     # Read the image file
#     contents = file.file.read()
    
#     # Open the image using PIL
#     image = Image.open(io.BytesIO(contents))
    
#     # Apply transformations
#     image = transform(image).unsqueeze(0)  # Add a batch dimension
#     return image

# # Define prediction endpoint
# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     # Process the uploaded image 
#     image = process_image(file)
    
#     # Make predictions using the model
#     with torch.no_grad():
#         prediction = model(image)
    
#     return {"prediction": prediction.tolist()}