## Eyetracker README.md

### Overview
This Python program, Eyetracker, utilizes the OpenCV library and the cvzone module to track the movement of the left eye's iris in both images and live video. The program provides a visual indication of the direction of gaze.

### Files
1. **Eyetracker.py**: Processes a static image (`data/face.jpg`) and displays the direction of gaze based on the position of the left eye's iris center.

2. **Eyetracker_vid.py**: Captures live video from a webcam or a specified video file and dynamically displays the direction of gaze.

3. **data folder**: Contains sample images for testing.

### Dependencies
- Python 3.x
- OpenCV
- cvzone module (Ensure it's installed, can be installed via `pip install cvzone`)

### Instructions

#### Eyetracker.py
1. Run the script using:
   ```
   python Eyetracker.py
   ```
2. View the processed image with gaze direction displayed.

#### Eyetracker_vid.py
1. Run the script using:
   ```
   python Eyetracker_vid.py
   ```
2. A window will open displaying the live video feed with the gaze direction overlay.
3. Press 'q' to exit the program.

### Notes
- Adjust the eye tracking threshold (`40` in the code) based on environmental conditions and image quality.
- Ensure the correct indices for left eye landmarks (`LEFT_EYE`) are used for your specific face model.

### Sample Output
Check the `data` folder for sample input and output images.

Feel free to explore and modify the code to suit your needs. For any issues or suggestions, please open an issue on GitHub.

Enjoy eye-tracking with Python and OpenCV! 👀
