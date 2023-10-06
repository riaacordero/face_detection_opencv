# Ahoy there! 🙌

This is an OpenCV and Haar Cascade model-based face detection algorithm that resizes facial images and converts them to grayscale, allows saving them into the same subfolders.

This repository is specifically created as part of our dataset creation for our Thesis 1 project, which involves the detection of panic attack patterns.

### How does this work?

1. Upload folder with image files to process.
2. Run the .py file
3. Enter the input/output filename.
4. Check the output folder for the processed images.

### Limits:
- The model detects multiple faces in one image... some aren't even faces. So, I recommend manually cleaning the data to achieve optimal results.
- If the grayscale conversion produces an error, try downgrading to OpenCV 4.6.0 (which I used for this project)
