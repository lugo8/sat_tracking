from trainingData import genData
from convertSatsim2Yolo import convert
from yoloTraining import train

def makeModel():
    # print("1")
    genData("satsim_dynamic_for_austin_v2.json", 2400)
    # print("2")
    convert("satsimOutput", 1, 2000, 200, 200)
    # print("3")
    # train(5)
    # print("4")

if __name__ == "__main__":
    makeModel()
