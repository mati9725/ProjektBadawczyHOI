import pickle

path_detection = "/home/zgorek/Desktop/Projekt Badawczy/DRG/Data/Test_Faster_RCNN_R-50-PFN_2x_VCOCO.pkl"
test_detection_temp = pickle.load(open(path_detection, "rb"), encoding='latin1')
# test_detection = {}
# test_detection[image_id] = test_detection_temp[image_id]
for detection in test_detection_temp[294]:
    print(detection)