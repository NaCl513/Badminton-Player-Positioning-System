# Badminton Player On-Court Localization System Based on YOLOv8 and Homography
## Execution Method: Download the files, then directly run the "detection1.py".<br>Stopping Playback: Press the 'q' while the script is running to stop the system.
This is the result of a special research project, which includes the trained YOLO model, a sample video, and the source code.  
Please ensure that all required software packages and dependencies are downloaded and installed.  


- This research aims to develop an automated visual analysis system for precisely locating players on the court during badminton matches and statistically analyzing their activity frequency within a nine-square court grid.

We used two core methods: 

1. Homography Matrix Estimation: The system first performs court geometry rectification on the match video frames. This is achieved by implementing an algorithm that combines the Probabilistic Hough Transform with Least Squares fitting to compute the Homography Matrix. This matrix is essential for mapping the pixel coordinates of the match video to a standardized court model.

2. Player Detection: The YOLOv8 deep learning model is employed for real-time object detection of the players, providing their Bounding Box locations within the image.

The core innovation lies in combining these two techniques: the calculated Homography Matrix is used to successfully transform the pixel positions of the players detected by the YOLO model into their actual 2D court coordinates on the standardized court model.

Data Analysis and Conclusion
Based on these coordinates, the system divides the court into a nine-square region and instantly detects which area the player is in. It then compiles a real-time statistical record of the player's activity frequency within each region.
