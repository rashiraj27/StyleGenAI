import cv2
import numpy as np

def analyze_skin_tone(image_path):
    print(f"📷 Loading image: {image_path}...")
    
    # 1. Read the image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Error: Could not find 'sample.jpg'. Check the file name and folder!")
        return
        
    # 2. Get image dimensions and crop the center 100x100 pixels
    # (In a full app, this is where you'd use a face detection bounding box)
    height, width, _ = image.shape
    center_y, center_x = height // 2, width // 2
    
    # Crop a square from the center of the image
    skin_patch = image[center_y-50 : center_y+50, center_x-50 : center_x+50]
    
    # 3. Convert from BGR (OpenCV default) to HSV color space
    hsv_patch = cv2.cvtColor(skin_patch, cv2.COLOR_BGR2HSV)
    
    # 4. Calculate the average Hue (Color), Saturation (Intensity), and Value (Lightness)
    average_hsv = np.mean(hsv_patch, axis=(0, 1))
    avg_hue = average_hsv[0]
    
    print(f"📊 Extracted Hue Value: {avg_hue:.2f}")
    
    # 5. Algorithmic Undertone Classification
    # Hue values in OpenCV go from 0-179.
    # Lower hues (reds/yellows) indicate warm undertones. 
    # Higher/shifted hues (pinks/blues) indicate cool undertones.
    if avg_hue < 15 or avg_hue > 165:
        undertone = "Cool"
    elif 15 <= avg_hue <= 35:
        undertone = "Warm"
    else:
        undertone = "Neutral"
        
    print(f"✨ Detected Skin Undertone: {undertone.upper()} ✨")
    return undertone

# Run the pipeline
final_tone = analyze_skin_tone("sample.jpeg")