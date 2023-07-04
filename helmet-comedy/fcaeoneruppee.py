import cv2
import numpy as np

# Load YOLOv3-tiny model
net = cv2.dnn.readNet('yolov3-tiny.weights', 'yolov3-tiny.cfg')

# Load class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Get output layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Load face mesh and PNG image
face_mesh = cv2.imread('face_mesh.png', -1)

# Define the coordinates to place the face mesh
# Adjust these coordinates according to your requirements
mesh_x = 100
mesh_y = 100

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, None, fx=0.4, fy=0.4)

    # Detect objects using YOLOv3-tiny
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize face detection variables
    class_ids = []
    confidences = []
    boxes = []

    # Process YOLOv3-tiny output
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter detections for faces
            if classes[class_id] == 'face':
                if confidence > 0.5:
                    # Calculate object coordinates on the original frame
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])

                    # Calculate bounding box coordinates
                    x = int(center_x - width / 2)
                    y = int(center_y - height / 2)

                    # Add face detection results to respective lists
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, width, height])

    # Apply face mesh and PNG image overlay
    for i in range(len(boxes)):
        # Extract face region
        x, y, width, height = boxes[i]
        face = frame[y:y + height, x:x + width]

        # Resize face mesh to match the face region
        mesh_width = int(width * 1.2)
        mesh_height = int(height * 1.2)
        face_mesh_resized = cv2.resize(face_mesh, (mesh_width, mesh_height))

        # Overlay face mesh on the face region
        x_offset = int(x + (width - mesh_width) / 2)
        y_offset = int(y + (height - mesh_height) / 2)
        alpha_s = face_mesh_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            frame[y_offset:y_offset + mesh_height, x_offset:x_offset + mesh_width, c] = (
                alpha_s * face_mesh_resized[:, :, c] + alpha_l * frame[y_offset:y_offset + mesh_height, x_offset:x_offset + mesh_width, c]
            )

        # Overlay PNG image on the forehead of the face
        forehead_height = int(height * 0.25)
        forehead_width = int(width * 0.6)
        forehead_x = int(x + (width - forehead_width) / 2)
        forehead_y = int(y + forehead_height * 0.2)
        forehead_region = frame[forehead_y:forehead_y + forehead_height, forehead_x:forehead_x + forehead_width]

        # Resize PNG image to match the forehead region
        png_resized = cv2.resize(png_image, (forehead_width, forehead_height))

        # Overlay PNG image on the forehead region
        alpha_s = png_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            forehead_region[:, :, c] = (
                alpha_s * png_resized[:, :, c] + alpha_l * forehead_region[:, :, c]
            )

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

