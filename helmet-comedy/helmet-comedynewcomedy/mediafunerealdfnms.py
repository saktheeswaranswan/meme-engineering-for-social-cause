import cv2
import mediapipe as mp
import numpy as np

# Load YOLOv3-tiny model
net = cv2.dnn.readNet('face-yolov3-tiny_41000.weights', 'face-yolov3-tiny.cfg')

# Load class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load face mesh and PNG image
face_mesh = cv2.imread('removebg.png', -1)

# Request user input for screen resolution
width = 1280
height = 720

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLOv3-tiny
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Initialize face detection variables
    class_ids = []
    confidences = []
    boxes = []

    # Process YOLOv3-tiny output
    for detection in outs[0]:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        # Filter detections for faces
        if classes[class_id] == 'face' and confidence > 0.5:
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

    # Apply non-maximum suppression to eliminate overlapping detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.3)
    for i in indices:
        i = i
        x, y, width, height = boxes[i]
        face = frame[y:y + height, x:x + width]

        # Convert frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process face mesh using MediaPipe
        results = mp_face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            # Get the first face mesh
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Extract forehead coordinates from face mesh
            forehead_x = int(face_landmarks[10].x * frame.shape[1])
            forehead_y = int(face_landmarks[10].y * frame.shape[0])

            # Overlay face mesh on the frame
            frame = cv2.circle(frame, (forehead_x, forehead_y), radius=5, color=(0, 0, 255), thickness=-1)

            # Overlay PNG image on the forehead of the face
            forehead_height = int(height * 0.25)
            forehead_width = int(width * 0.6)
            forehead_x = int(x + (width - forehead_width) / 2)
            forehead_y = int(y + forehead_height * 0.2)
            forehead_region = frame[forehead_y:forehead_y + forehead_height, forehead_x:forehead_x + forehead_width]

            # Resize PNG image to match the forehead region
            face_mesh_resized = cv2.resize(face_mesh, (forehead_width, forehead_height))

            # Overlay PNG image on the forehead region
            alpha_s = face_mesh_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                forehead_region[:, :, c] = (
                        alpha_s * face_mesh_resized[:, :, c] + alpha_l * forehead_region[:, :, c]
                )

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

