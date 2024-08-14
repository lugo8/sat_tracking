# sat_tracking
Experimenting with different preprocessing techniques (dark object subtraction, contrast enhancement, and background correction) to determine the best permutation for satellite detection.

First run trainingData.py to generate satsim images.

Then run convertSatsim2Yolo.py to convert to a format yolov8 can understand.

Ensure lastPermut.txt has a 0 in it to start.
Copy all images from trainData/val/images into ogValImgs and from trainData/train/images into ogTrainImgs

Then run batchModel in preprocess.py in "train" mode to create a model for every permutation and evaluate its performance.

"Train" refers to training a model for every permutation while "val" refers to only validating the permutations based on a single model trained on unprocessed data.
