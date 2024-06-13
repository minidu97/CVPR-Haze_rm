import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image

# Placeholder for the ResINet class definition
class ResINet(nn.Module):
    def __init__(self):
        super(ResINet, self).__init__()
        # Define the layers as per the ResINet architecture
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        x4 = F.relu(self.conv4(x3))
        y = F.relu(self.deconv1(x4))
        y = F.relu(self.deconv2(y))
        y = F.relu(self.deconv3(y))
        y = self.deconv4(y)
        return y

# Load the pre-trained model
model = ResINet()
model.load_state_dict(torch.load('path_to_pretrained_resinet_model.pth'))
model.eval()

# Function to remove haze from an image
def remove_haze(image):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        dehazed_tensor = model(image_tensor)
    dehazed_image = transforms.ToPILImage()(dehazed_tensor.squeeze(0))  # Remove batch dimension
    return dehazed_image

# Function to estimate motion using optical flow
def estimate_motion(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame1 = cap.read()
    if not ret:
        print("Failed to read video")
        return

    # Remove haze from the first frame
    frame1_pil = Image.fromarray(cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB))
    dehazed_frame1 = remove_haze(frame1_pil)
    dehazed_frame1 = cv2.cvtColor(np.array(dehazed_frame1), cv2.COLOR_RGB2BGR)

    prev_gray = cv2.cvtColor(dehazed_frame1, cv2.COLOR_BGR2GRAY)

    while(cap.isOpened()):
        ret, frame2 = cap.read()
        if not ret:
            break

        # Remove haze from the current frame
        frame2_pil = Image.fromarray(cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB))
        dehazed_frame2 = remove_haze(frame2_pil)
        dehazed_frame2 = cv2.cvtColor(np.array(dehazed_frame2), cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(dehazed_frame2, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        # Visualize the flow
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        hsv[..., 0] = angle * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        rgb_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('Optical Flow', rgb_flow)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        prev_gray = gray

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = 'path_to_hazy_video.mp4'
estimate_motion(video_path)
