import json
import cv2 as cv
def motion_state(res):
    json_data_str = json.loads(res)
    if(json_data_str["data"]["cloudPos"]) ==None:
        x = -1
        y = -1
    else:
        y = int(json_data_str["data"]["cloudPos"]['nPTZTilt'])
        x = int(json_data_str["data"]["cloudPos"]['nPTZPan'])
    return x,y
def point_mate(json_data,x,y):
    point = json_data["point"]
    for i in range(int(len(point)/2)):
        xx,yy = point[i*2],point[i*2+1]
        if xx==x and yy == y:
            return True
        else:
            pass
def video_to_pic(json_data):
    cap = cv.VideoCapture(json_data["RTSP"])
    state,pic = cap.read()
    cap.release()
    return state,pic
    