from satsim import load_json
import matplotlib.pyplot as plt
from astropy.io import fits
import numpy as np
import os
from PIL import Image
import glob
import shutil
import cv2
import matplotlib.patches as patches


#TODO: Make it so that the names of the produced files will not overlap

def convert(path, numImg, numTrain, numVal, numTest):
    # Converts the .json file describing the image and bounding box output of satsim
    # into a .txt file that YOLO can work with. Also converts the .fits file output of
    # satsim into a .jpg for YOLO to work with
    #
    # Parameters:
    # - path: path to satsim's output directory (relative to current directory)
    # - numImg: number of images produced per directory
    # - numTrain: number of images in training 
    # - numVal: number of images in validation
    # - numTest: number of images in testing

    #List of all subdirectories in order they were made
    subDirs = [entry for entry in os.listdir(path) if os.path.isdir(os.path.join(path, entry))]
    subDirs = [os.path.join(path, tempDir) for tempDir in subDirs]
    subDirs= sorted(subDirs, key=lambda x: os.path.getmtime(x))
    #print(subDirs)

    #Running count of the directories
    dirCount = 0

    delete = True #delete old data?

    mode = "train"

    #Remove previous data
    if delete:
        delMode(mode)

    #Go through each directory in path
    for dir in subDirs:

        # Change so that each train, val, test gets correct number of images/labels
        if dirCount == numTrain:
            mode = "val"
            if delete:
                delMode(mode)

        elif dirCount == (numTrain + numVal):
            mode = "test"
            if delete:
                delMode(mode)

        #iterate through each image and make bounding box and image files for YOLO to understand
        for j in range(numImg):
            
            boxList = boxData(dir, dirCount, j, mode)

            img = imgData(dir, dirCount, j, mode)

            #visualBb(img, boxList)

            

        dirCount += 1

def boxData(path, dirCount, i, mode):
    # Converts bounding box data from satsim's JSON format to 
    # YOLO's .txt file
    # 
    # Parameters:
    # - path: name of subdirectory with the generated data
    # - dirCount: directory number
    # - i: image number
    # - mode: whether this image belongs in train, val, or test

    satName = f"sat_{str(dirCount).zfill(5)}.{str(i).zfill(4)}"
    jsonData = load_json(path + f"/Annotations/{satName}.json") 

    #print(path + f"/Annotations/sat_{str(dirCount).zfill(5)}.{str(i).zfill(4)}.json")

    # the BB data
    objects = jsonData['data']['objects']
    sensor = jsonData['data']['sensor']
    height = sensor['height']
    width = sensor['width']

    hlfSideLen = 10 #pixels len of half the bounding box

    # prepare YOLO format data
    yolo_data = []
    boxLables = []
    for obj in objects:
        class_id = 0

        center = obj['pixels'][0]
        cenX = center[1] # x coord of center
        cenY = center[0] # y coord of center

        lx = (cenX - hlfSideLen) #left x coord of bb
        rx = (cenX + hlfSideLen) # right x coord of bb
        uy = (cenY + hlfSideLen) # upper y coord of bb
        dy = (cenY - hlfSideLen) # lower y coord of bb

        lxNorm = lx / width #left x coord of bb
        rxNorm = rx / width # right x coord of bb
        uyNorm = uy / height # upper y coord of bb
        dyNorm = dy / height # lower y coord of bb

        yolo_data.append(f"{class_id} {lxNorm} {uyNorm} {rxNorm} {dyNorm} {lxNorm} {dyNorm} {rxNorm} {uyNorm}") # four corners of bb
        boxLables.append([[lx, uy], [rx, dy], [lx, dy], [rx, uy]])

    # Save YOLO data to a file
    output_filename = f"{satName}.txt"
    with open(os.path.abspath('.') + f"/trainData/{mode}/labels/" + output_filename, 'w') as f:
        for line in yolo_data:
            f.write(line + '\n')

    return boxLables

def imgData(path, dirCount, i, mode):
    # Converts image data from satsim's .fits format to 
    # YOLO's .jpg file
    # 
    # Parameters:
    # - path: name of subdirectory with the generated data
    # - dirCount: directory number
    # - i: image number
    # - mode: whether this image belongs in train, val, or test


    name = f"sat_{str(dirCount).zfill(5)}.{str(i).zfill(4)}"

    #Get the .fits file and convert to numpy array
    fits_hdulist = fits.open(path + f"/ImageFiles/{name}.fits") 
    hdu = fits_hdulist[0]
    b4data = hdu.data
   
    #b4data = np.flip(b4data, axis=1)

    #img = z_scale_normalization(b4data) #TODO: here is where preprocessing would be done

    # bkg = find_background(b4data)

    # img = apply_correction(b4data, bkg)
    img = b4data #because no preprocessing

    #Convert to images
    img = img.astype(np.uint8)
    
    # Save the image as png
    path2images = os.path.abspath('.') + f"/trainData/{mode}/images"
    # Save the image as PNG
    cv2.imwrite(path2images + f"/{name}.png", img)
    #image.save(path2images + f"/{name}.png") 

    # # Display the image

    # plt.imshow(img, cmap='gray')
    # plt.title('FITS Image ' + str(i))
    # plt.show()  

    return img

def delFilesandDirs(directory):
    # Deletes all files and directories in a directory
    # 
    # Parameters:
    # - directory: specified directory to delete all files
    
    for root, dirs, files in os.walk(directory, topdown=False):

        #delete files
        for name in files:
            file_path = os.path.join(root, name)
            os.remove(file_path)

        #delete directories
        for name in dirs:
            dir_path = os.path.join(root, name)
            shutil.rmtree(dir_path)

def delMode(mode):
    # Deletes all images in specified folder in trainData
    # 
    # Parameters:
    # - mode: train, val, or test

    modePath = os.path.abspath('.') + f"/trainData"
    delFilesandDirs(modePath + f"/{mode}/images")
    delFilesandDirs(modePath + f"/{mode}/labels")

def visualBb(image_array, bbox_corners):
    # Visualize bounding boxes on an image.
    
    # Parameters:
    # - image_array: numpy array representing the image
    # - bbox_corners: list of list of tuples containing the coordinates of the corners of the bounding box [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
    #

    fig, ax = plt.subplots(1)
    ax.imshow(image_array, cmap='gray')
    
    for box in bbox_corners:

        # Extract the top-left and bottom-right corners of the bounding box
        x_coords = [corner[0] for corner in box]
        y_coords = [corner[1] for corner in box]
        
        min_x = min(x_coords)
        min_y = min(y_coords)
        max_x = max(x_coords)
        max_y = max(y_coords)
        
        # Create a Rectangle patch
        width = max_x - min_x
        height = max_y - min_y
        rect = patches.Rectangle((min_x, min_y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        
        # Add the patch to the Axes
        ax.add_patch(rect)
    
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    convert("satsimOutput", 1, 2000, 200, 200)