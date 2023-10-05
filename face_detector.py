import os
import cv2

def detect_and_crop_faces(input_dir, output_dir):
    # Load the cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            # Read the image
            img_path = os.path.join(input_dir, filename)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Error: Could not read image file {img_path}")
                    continue
            except Exception as e:
                print(f"Error: Could not read image file {img_path}: {e}")
                continue

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
                output_path = os.path.join(output_dir, f'{filename.split(".")[0]}_face{i}.jpg')
                cv2.imwrite(output_path, resized_face)
    
    print("Completed upload on "+output_dir+" directory. Check the output directory for the cropped faces.")


input_dir = './'+input('Enter the input directory: ')
output_dir = './'+input('Enter the output directory: ')

if os.path.isdir(input_dir):
    os.makedirs(output_dir, exist_ok=True)
    detect_and_crop_faces(input_dir, output_dir)
else:
    print("Error: Input directory does not exist or is not a directory.")
