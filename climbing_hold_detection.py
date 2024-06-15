import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

def detect_climbing_holds(image_path):
    # Load the image
    image = Image.open(image_path)
    
    # Convert the image to numpy array and then to grayscale
    image_np = np.array(image)
    gray_image = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    edged = cv2.Canny(gray_image, 30, 200) 
    
    # Apply a threshold to get the holds (they should be bright spots on the gray image)
    _, thresh_image = cv2.threshold(edged, 150, 255, cv2.THRESH_BINARY)
    
    # Find contours which represent the holds
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the coordinates, height, and width of the holds
    holds_data = []
    for contour in contours:
        if cv2.contourArea(contour) > 20:  # Filter out very small areas
            x, y, w, h = cv2.boundingRect(contour)
            holds_data.append((x, y, w, h))
    
    # Sort coordinates by Y (top to bottom), then by X (left to right)
    holds_data.sort(key=lambda hold: (hold[1], hold[0]))
    
    # Convert the list of hold data to a DataFrame for better visualization
    holds_df = pd.DataFrame(holds_data, columns=["X", "Y", "Width", "Height"])
    
    # cv2.drawContours(image_np, contours, -1, (0, 255, 0), 3) 
    # plt.imshow(image_np)
    # plt.axis('on')
    # plt.show()
    return holds_df

# Path to your image file
image_path = 'bpump_competition_wall.png'

# Detect climbing holds
holds_df = detect_climbing_holds(image_path)

# Display the coordinates along with width and height
print(holds_df)

# Optionally, display the image with detected holds
# image = Image.open(image_path)

