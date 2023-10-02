# Ahoy there! 🙌

This is an OpenCV and Haar Cascade model-based face detection algorithm that resizes facial images and converts them to grayscale, allows saving them into the same subfolders.

This repository is specifically created as part of our dataset creation for our Thesis 1 project, which involves the detection of panic attack patterns.

### How does this work?

1. Upload your own files to the "train" folder.
2. Empty the "new_ds" folder. The uploaded images are for demonstration only.
3. Run the .py file
4. Check the "new_ds" folder for the output.

**Limits:**
- The model detects multiple faces in one image... some aren't even faces. So, I recommend manually cleaning the data to achieve optimal results.
- If the grayscale conversion produces an error, try downgrading to OpenCV 4.6.0 (which I used for this project)
