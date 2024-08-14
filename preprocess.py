import cv2
import matplotlib.pyplot as plt
import numpy as np
from background import find_background, apply_correction
from convertSatsim2Yolo import delFilesandDirs
import os
import glob
from yoloTraining import val, train
import itertools
import csv
import shutil
from astropy.visualization import ZScaleInterval
import pandas as pd


def zScale(img, contrast=0.2):
    # Taken from https://github.com/kevinphaneos/Synthetic_Tracking/blob/main/utils.py
    # Z scale normalization on input numpy image
    #
    # Parameters:
    # - img: Image to normalize
    #
    # Returns:
    # - Normalized image

    norm = ZScaleInterval(contrast=contrast)
    return norm(img)

def bkgCorr(img):
    # Using functions from https://github.com/kevinphaneos/Synthetic_Tracking/tree/main 
    # Apply background correction to an image
    #
    # Parameters:
    # - img: Image to apply correction on
    #
    # Returns:
    # - Processed image

    bkg = find_background(img)

    processed = apply_correction(img, bkg)

    return processed

def contrast(img, contrast=1.2):
    # Apply contrast enhancement
    #
    # Parameters:
    # - img: Image to apply enhancement on
    #
    # Returns:
    # - Processed image

    #return cv2.convertScaleAbs(img, contrast, 1)
    return cv2.convertScaleAbs(img, alpha=contrast, beta=1)

def DOS(img, percent=.005):
    # Apply dark object subtraction
    #
    # Parameters:
    # - img: Image to apply subtraction on
    # - percent: Bottom percent of pixels to calculate the value to subtract from the entire image
    #
    # Returns:
    # - Processed image

    #TODO: I dont know if this is acceptable for DOS

    #Find lower percent of pixels
    pixels = img.flatten()
    sorted_pixels = np.sort(pixels)
    num_pixels = len(sorted_pixels)
    threshold = int(num_pixels * percent)
    lower = sorted_pixels[0:threshold]
    #print(lower[0:200])

    min = np.mean(lower) #Average of the lowest pixels

    # min = np.min(img)
    #print(min)
    processed = img - min
    return processed

def newImg(functions, img):
    # Apply every function to image in order they were passed in
    #
    # Parameters:
    # - functions: A list of functions to apply to image
    # - img: Image to apply functions on
    #
    # Returns:
    # - Processed image

    processed = img

    for func in functions:
        if func != None:
            processed = func(processed)
    
    return processed

def allImgs(mode):
    # Make all edited images
    #
    # Parameters:
    # - mode: validation or train ("Val" or "Train")


    # Get a list of all .png of the original images
    png_files = glob.glob(f'og{mode}Imgs/*.png')


    if mode == "val":
        delPrefix("runs/detect", "val") #Remove any file with the "val" prefix to start with a clean slate


    delFilesandDirs(os.path.join(os.path.abspath('.'), f"permutImgs{mode}")) #Clear validation permutations

    perms = allPerms()
    first = True
    iter = 1

    for perm in perms:

        #First permutation?
        if perm != perms[0]:
            first = False

        funcNames = [func.__name__ if func is not None else 'None' for func in perm]

        permName = '_'.join(funcNames)
        
        # Apply permutation of processes to each file 
        # and store in validation file
        for file in png_files:

            #Load image as numpy array
            b4img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            #b4img = cv2.cvtColor(b4img, cv2.COLOR_BGR2GRAY)

            new = newImg(perm, b4img) #Apply processing in desired order

            # Save the image as png
            saveImg = new.astype(np.uint8)

            #Make directory
            dirPath = os.path.join(f"permutImgs{mode}", permName)
            os.makedirs(dirPath, exist_ok=True)
            
            #Save file
            path2images = os.path.join(os.path.abspath('.'), dirPath)
            satName = file[len(f"og{mode}Imgs/"):]
            filePth = os.path.join(path2images, satName)

            cv2.imwrite(filePth, saveImg)
        
        print(f"Permutation {iter} finished")
        iter += 1

def allPerms():
    # Make all permutations of all the processing techniques
    #
    # Parameters:
    #

    funcList = [bkgCorr, contrast, DOS]

    # Generate all permutations of the functions without on or off
    b4permutations = list(itertools.permutations(funcList))
    b4permList = [list(p) for p in b4permutations] #make it a list

    binaryNums = [format(i, '03b') for i in range(8)] #all four digit binary numbers

    names = [func.__name__ if func is not None else 'None' for perm in b4permList for func in perm]

    # i = 0
    # for j in range(len(names)):
    #     print(names[j], end='')
    #     print(" | ", end='')
    #     if i == 3:
    #         print("\n")
    #         i = -1
        
    #     i += 1

    #print(len(permList[0]))
    
    permList = []
    #For each original permutation, add on or off
    for permutation in b4permList:

        for binary in binaryNums:

            switchedPerm = [None, None, None] #Permutation with on or off

            for i in range(3):
                if binary[i] == '1':
                    switchedPerm[i] = permutation[i]   
                    #print("HERE")         
            permList.append(switchedPerm)
            #print(switchedPerm)
        
    return remvDuplicates2d(permList)

def remvDuplicates2d(lst):
    # Removes all duplicates from a 2d list
    #
    # Parameters:
    # - lst: The list to remove duplicates from
    #
    # Return:
    # - The list without duplicates

    #return [list(x) for x in dict.fromkeys(tuple(sublist) for sublist in lst)]

    # Define a function to remove None entries from an inner list
    def remove_none(inner_list):
        return [item for item in inner_list if item is not None]

    # Use a set to store unique lists after removing None entries
    seen = set()
    unique_list = []

    for inner_list in lst:
        cleaned_list = tuple(remove_none(inner_list))  # Convert to tuple to make it hashable
        if cleaned_list not in seen:
            seen.add(cleaned_list)
            unique_list.append(inner_list)  # Append the original list

    return unique_list

def addValInfo(fileName, permName, metrics, first):
    # Adds the metrics from validation into a csv file
    #
    # Parameters:
    # - fileName: File name to save to
    # - permName: permutation name
    # - metrics: the results of the validation
    # - first: if this is the first time the file needs to be opened
    #
    
    #Get specific metrics
    precision = metrics.class_result(0)[0]
    recall = metrics.class_result(0)[1]
    map50 = metrics.class_result(0)[2]
    map50_95 = metrics.class_result(0)[3]

    #csv data
    headers = ['Permutation', 'mAP50', 'mAP50-95', 'Precision', 'Recall']
    row = [permName, map50, map50_95, precision, recall,]

    #add header if new
    if first:
        with open(fileName, mode='w', newline='') as file:
            writer = csv.writer(file)

            writer.writerow(headers)

            # Write the new row
            writer.writerow(row)  
    else:
        with open(fileName, mode='a', newline='') as file:
            writer = csv.writer(file)

            # Write the new row
            writer.writerow(row)   
    

def delPrefix(dir, prefix):
    # Deletes all files of a certain prefix
    #
    # Parameters:
    # - dir: Directory with files to delete
    # - prefix: Prefix of files to delete

    # List all entries in the given directory
    entries = os.listdir(dir)
    
    # Iterate over all entries
    for entry in entries:
        # Construct full path of the entry
        full_path = os.path.join(dir, entry)
        
        # Check if the entry is a directory and starts with the prefix
        if os.path.isdir(full_path) and entry.startswith(prefix):
            # Remove the directory and all its contents
            shutil.rmtree(full_path)
                

def testProcess():
    # Test preprocessing
    #
    # Parameters:


    #Load img
    path = "trainData/train/images/sat_00000.0000.png"
    b4img = cv2.imread(path)
    b4img = cv2.cvtColor(b4img, cv2.COLOR_BGR2GRAY)

    #Make a figure to display before and aftr
    plt.figure(figsize=(10, 5))

    numImg = 2

    #Show unprocessed image
    plt.subplot(1, numImg, 1)  
    plt.imshow(b4img, cmap='gray')
    print(b4img)
    print("---------------------------\n\n\n\n")

    #Show processed image
    plt.subplot(1, numImg, 2) 
    #aftrImg = zScale(b4img)
    #aftrImg = bkgCorr(b4img)
    aftrImg = contrast(b4img, 1.2)
    #aftrImg = DOS(b4img, .005)
    plt.imshow(aftrImg, cmap='gray')
    print(aftrImg)

    # plt.subplot(1, numImg, 3) 
    # aftrImg = contrast(aftrImg, 1.5)
    # plt.imshow(aftrImg, cmap='gray')

    plt.show()


def batchModel(model, file, mode, epochs=15, batch=3):
    # Preforms validation on a batch of the permutations
    #
    # Parameters: 
    # - model: path to the model
    # - file: file to store last permut num
    # - mode: "Val" or "Train"
    # - batch: batch size (number of permutations to edit)

    lowerMode = mode.lower()

    #Get list of subDirs in order they were created (4 consistency)
    source = f"permutImgs{mode}"
    sourceVal = "permutImgsVal"
    destination = f"trainData/{lowerMode}/images"
    destinationVal = "trainData/val/images"

    subDirs = [entry for entry in os.listdir(source) if os.path.isdir(os.path.join(source, entry))]
    subDirs = [os.path.join(source, tempDir) for tempDir in subDirs]
    subDirs= sorted(subDirs, key=lambda x: os.path.getmtime(x))

    subDirsVal = [entry for entry in os.listdir(sourceVal) if os.path.isdir(os.path.join(sourceVal, entry))]
    subDirsVal = [os.path.join(sourceVal, tempDir) for tempDir in subDirsVal]
    subDirsVal = sorted(subDirsVal, key=lambda x: os.path.getmtime(x))

    lastPermut = getPermutNum(file) #Permutation we are on currently

    #Repeat as many times as the batch size
    for i in range(batch):

        if(i + lastPermut > 15):
            return

        permutDir = subDirs[i + lastPermut] #directory of next permutation images
        permutDirVal = subDirsVal[i + lastPermut] #directory of next validation permutation img

        imgFiles = [f for f in os.listdir(permutDir)] #list of all the files in the permutation directory
        imgFilesVal = [f for f in os.listdir(permutDirVal)] #list of all the files in the permutation directory validation

        #copy all images from permutation directory to validation/train directory
        for filename in imgFiles:
            sourceFile = os.path.join(permutDir, filename)
            destinationFile = os.path.join(destination, filename)
            
            if os.path.isfile(sourceFile):
                shutil.copy(sourceFile, destinationFile)

        for filename in imgFilesVal:
            sourceFileVal = os.path.join(permutDirVal, filename)
            destinationFileVal = os.path.join(destinationVal, filename)
            
            if os.path.isfile(sourceFileVal):
                shutil.copy(sourceFileVal, destinationFileVal) 

        
        
        permName = permutDir.replace(f"permutImgs{mode}/", '', 1)

        if mode == "Val":
            #Run validation on each permutation
            metrics = val(model)
        
        elif mode == "Train":
            #Run train on each permutation
            metrics = train(epochs)

        first = False

        if (lastPermut == 0) and i == 0:
            first = True

        print(f"{permName} finished")
        addValInfo(f"permutResults{mode}.csv", permName, metrics, first)
        
    
    storePermutNum(lastPermut + batch, file)


def getPermutNum(file):
    # Gets the number of the last permutation
    #
    # Parameters:
    # - file: name of file last permut is stored in  

    # Open the file and read the first line
    with open(file, 'r') as f:
        # Read the first line of the file
        first_line = f.readline().strip()
        
        # Convert the read line to an integer
        readNum = int(first_line)  # Use float(first_line) if the number could be a decimal

        if readNum > 208: #Catch if we already ran this code
            return 0
        else:
            return readNum
    
def storePermutNum(num, file):
    # Stores the number of the last permutation
    #
    # Parameters:
    # - num: number to store
    # - file: name of file last permut is stored in

    with open(file, 'w') as f:
        f.write(str(num))

def plotResults(file):
    # Plots the results
    #
    # Parameters:
    # - file: plots the results 

    data = pd.read_csv(file)

    # Set the index to 'Permutation' for better readability
    data.set_index('Permutation', inplace=True)

    # Find the top 3 for each metric
    top3_map50 = data['mAP50'].nlargest(3)
    top3_map50_95 = data['mAP50-95'].nlargest(3)
    top3_precision = data['Precision'].nlargest(3)
    top3_recall = data['Recall'].nlargest(3)

    # Print the results for each metric
    print("Top 3 mAP50:")
    print(top3_map50)

    print("\nTop 3 mAP50-95:")
    print(top3_map50_95)

    print("\nTop 3 Precision:")
    print(top3_precision)

    print("\nTop 3 Recall:")
    print(top3_recall)

    # Find the top 3 permutations with the highest sum of all metrics
    data['Total'] = data['mAP50'] + data['mAP50-95'] + data['Precision'] + data['Recall']
    top3_total = data['Total'].nlargest(3)

    # Print the results
    print("\nTop 3 Total Sum of Metrics:")
    print(top3_total)

    # Create a figure with multiple subplots
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    # Plot mAP50
    axs[0, 0].bar(data.index, data['mAP50'], color='skyblue')
    axs[0, 0].set_title('mAP50 for Each Permutation')
    axs[0, 0].set_ylabel('mAP50')
    axs[0, 0].tick_params(axis='x', rotation=90)

    # Plot mAP50-95
    axs[0, 1].bar(data.index, data['mAP50-95'], color='lightgreen')
    axs[0, 1].set_title('mAP50-95 for Each Permutation')
    axs[0, 1].set_ylabel('mAP50-95')
    axs[0, 1].tick_params(axis='x', rotation=90)

    # Plot Precision
    axs[1, 0].bar(data.index, data['Precision'], color='salmon')
    axs[1, 0].set_title('Precision for Each Permutation')
    axs[1, 0].set_ylabel('Precision')
    axs[1, 0].tick_params(axis='x', rotation=90)

    # Plot Recall
    axs[1, 1].bar(data.index, data['Recall'], color='lightcoral')
    axs[1, 1].set_title('Recall for Each Permutation')
    axs[1, 1].set_ylabel('Recall')
    axs[1, 1].tick_params(axis='x', rotation=90)

    # Adjust layout
    if file == "permutResultsVal.csv":
        fig.suptitle("1 Model Performance", fontsize=16)
    elif file == "permutResultsTrain.csv":
        fig.suptitle("Permutation Model Performance", fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    #allImgs("Train")
    #batchModel("runs/detect/train24/weights/best.pt", "lastPermut.txt", "Train", batch=1)
    #allImgs("Val")
    #batchModel("runs/detect/train24/weights/best.pt", "lastPermut.txt", "Val", batch=2)
    #testProcess()
    plotResults("permutResultsTrain.csv")


 



    


