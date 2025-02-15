import torch
from ultralytics import YOLO

# Check for CUDA availability
if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU.")
else:
    print("CUDA is not available. PyTorch is using the CPU.")

# Load the model (starting with a pre-trained model)
model = YOLO('./best.pt')  # YOLOv8 nano version

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Train the model
model.train(
    data='C:/Users/Kenan/ML Projects/License Plate Recognition - LPR/datasets/tr/data.yaml',
    epochs=50,
    imgsz=640,
    device=device,
    workers=0,
    batch=16,
    lr0=0.01,
    augment=True,
    project='license_plate_recognition',  # Directory for saving results
    name='exp1_custom_name',  # Subdirectory and base name for the output files
    save_period=10  # Save model every 10 epochs
)