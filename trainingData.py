from satsim import gen_multi, load_json, config
import copy
from convertSatsim2Yolo import delFilesandDirs
import os

def genData(filename, numSamples):
    # Generates image and bounding box data on satellites using satsim
    #
    # Parameters:
    # - filename: name of the file with seed to generate the images

    # load a template json file
    ssp = load_json(filename)

    ssp['sim']['samples'] = numSamples

    dir = "satsimOutput/"
    delFilesandDirs(os.path.join(os.path.abspath('.'), dir))

    # generate SatNet files to the output directory
    gen_multi(ssp, eager=True, output_dir=dir)

if __name__ == "__main__":
    genData("satsim_dynamic_for_austin_v2.json", 2400)
