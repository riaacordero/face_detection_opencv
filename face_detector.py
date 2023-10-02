import os
import cv2

def detect_and_crop_faces(input_dir, output_dir):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Loop through all subdirectories in the input directory
    for root, dirs, files in os.walk(input_dir):
        for filename in files:
            if filename.endswith('.png') or filename.endswith('.jpg'):
                # Read the image
                img_path = os.path.join(root, filename)
                img = cv2.imread(img_path)

                # Convert to grayscale
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)

                # Loop through all detected faces
                for i, (x, y, w, h) in enumerate(faces):
                    # Crop the face
                    face = img[y:y + h, x:x + w]

                    # Convert cropped face to grayscale
                    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                    # Resize the cropped face to 48x48 pixels
                    resized_face = cv2.resize(gray_face, (48, 48))

                    # Save the cropped face in grayscale
                    output_path = os.path.join(output_dir, root, f'{filename.split(".")[0]}_face{i}.jpg')
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    cv2.imwrite(output_path, resized_face)


input_dir = './train'
output_dir = './new_ds'

for root, dirs, files in os.walk(input_dir):
    for dir in dirs:
        input_subdir = os.path.join(input_dir, dir)
        output_subdir = os.path.join(output_dir, dir)
        detect_and_crop_faces(input_subdir, output_subdir)
