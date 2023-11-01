import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.models import resnet50
# Load the image using OpenCV
image_path = '/home/bishoymoussas/Pictures/figure_to_heat.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load pre-trained ResNet50 model
model = resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-2])  # Remove the last fully connected layer and avgpool
model.eval()

# Transform the image and get feature maps
img_tensor = F.to_tensor(image).unsqueeze(0)
if torch.cuda.is_available():
    img_tensor = img_tensor.cuda()
    model = model.cuda()

with torch.no_grad():
    features = model(img_tensor)

# Generate heatmap based on feature activations
heatmap = torch.mean(features, dim=1).squeeze().cpu().numpy()
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-10)  # Normalize between 0 and 1

# Resize heatmap to match original image size
heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

# Overlay the heatmap on the original image
overlayed_image = image.copy()
alpha = 0.5
cmap = plt.get_cmap('jet')
heatmap_colored = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
overlayed_image = cv2.addWeighted(overlayed_image, 1 - alpha, heatmap_colored, alpha, 0)

# Display the overlayed image
plt.imshow(overlayed_image)
plt.axis('off')
plt.show()
