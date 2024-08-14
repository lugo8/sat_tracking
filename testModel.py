from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt


# Define a function to plot the results
def plot_results(image, results):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    # Loop through the detections
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box)
            conf = scores[i]
            cls = classes[i]
            
            # Draw bounding boxes and labels
            plt.gca().add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2))
            plt.text(x1, y1, f'{model.names[int(cls)]} {conf:.2f}', bbox=dict(facecolor='yellow', alpha=0.5))
    
    plt.axis('off')
    plt.show()

model = YOLO('runs/detect/train24/weights/best.pt') #best or last

# Load an image
image_path = 'trainData/images/test/sat_02205.0000.png'
image = cv2.imread(image_path)

# Convert the image to RGB (YOLO models expect RGB images)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Perform inference
results = model(image_rgb)

# Plot the results
plot_results(image_rgb, results)
