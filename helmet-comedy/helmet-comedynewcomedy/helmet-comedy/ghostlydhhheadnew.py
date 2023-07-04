import cv2
import mediapipe as mp
import numpy as np

# Load YOLOv3-tiny model
net = cv2.dnn.readNet('face-yolov3-tiny_41000.weights', 'face-yolov3-tiny.cfg')

# Load class labels
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Load face mesh and PNG images
face_mesh = cv2.imread('removebg.png', -1)
head_hat = cv2.imread('head_hat.png', -1)

# Request user input for screen resolution
width = 1280
height = 720

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# Initialize MediaPipe face mesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize head hat position
hat_x = int((width - head_hat.shape[1]) / 2)  # Centered horizontally
hat_y = int(height / 3)  # Positioned vertically at one-third height

# Initialize circles' positions
circle1_x = int(width / 4)  # Positioned at one-fourth width
circle1_y = int(height / 2)  # Positioned at half height

circle2_x = int(width / 2)  # Positioned at half width
circle2_y = int(height / 4)  # Positioned at one-fourth height

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects using YOLOv3-tiny
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(net.getUnconnectedOutLayersNames())

    # Print the dimensions of outs and out for debugging
    print("outs shape:", outs[0].shape)
    print("out shape:", outs[1].shape)

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

    # Apply face mesh overlay and head hat
    for i in range(len(boxes)):
        # Extract face region
        x, y, width, height = boxes[i]
        face = frame[y:y + height, x:x + width]

        # Detect face landmarks using MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark

            # Extract forehead coordinates
            forehead_x = int(face_landmarks[10].x * width) + x
            forehead_y = int(face_landmarks[10].y * height) + y

            # Overlay face mesh image on the frame
            face_mesh_resized = cv2.resize(face_mesh, (width, height))
            alpha_s = face_mesh_resized[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(3):
                try:
                    frame[y:y + height, x:x + width, c] = (
                            alpha_s * face_mesh_resized[:, :, c] + alpha_l * frame[y:y + height, x:x + width, c]
                    )
                except ValueError:
                    pass

            # Draw white circles at the forehead and nose nostrils
            cv2.circle(frame, (forehead_x, forehead_y), radius=5, color=(255, 255, 255), thickness=-1)
            nose_x = int(face_landmarks[5].x * width) + x
            nose_y = int(face_landmarks[5].y * height) + y
            cv2.circle(frame, (nose_x, nose_y), radius=5, color=(255, 255, 255), thickness=-1)

        # Overlay head hat PNG on the face
        hat_height = int(height * 0.8)
        hat_width = int(width * 1.2)
        hat_region = frame[hat_y:hat_y + hat_height, hat_x:hat_x + hat_width]

        # Resize head hat PNG to match the region
        head_hat_resized = cv2.resize(head_hat, (hat_width, hat_height))

        # Overlay head hat PNG on the region
        alpha_s = head_hat_resized[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(3):
            try:
                hat_region[:, :, c] = (
                        alpha_s * head_hat_resized[:, :, c] + alpha_l * hat_region[:, :, c]
                )
            except ValueError:
                pass

    # Update the positions of the circles
    circle1_x += 1  # Increment x-coordinate of circle 1
    circle1_y += 1  # Increment y-coordinate of circle 1

    circle2_x -= 1  # Decrement x-coordinate of circle 2
    circle2_y -= 1  # Decrement y-coordinate of circle 2

    # Draw circles on the frame
    cv2.circle(frame, (circle1_x, circle1_y), radius=5, color=(255, 255, 255), thickness=-1)
    cv2.circle(frame, (circle2_x, circle2_y), radius=5, color=(255, 255, 255), thickness=-1)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()

