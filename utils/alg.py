from utils.plots import  plot_one_box
from utils.general import  check_img_size,non_max_suppression,scale_coords
import cv2
import torch
from utils.torch_utils import select_device
import numpy as np
device = "cuda:1"
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import cv2 
import requests
from utils.http_seve import requests_load
from utils.json_data import motion_state,point_mate,video_to_pic
#时间管理区域
SHA_TZ = timezone(
    timedelta(hours=8),
    name='Asia/Shanghai',
)
headers = {"Content-Type": "application/json;charset=utf8"}
url = "http://10.11.96.41:8088/prod-api/miniapi/event/onaievent"

half = True  # half precision only supported on CUDA
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)
def YOLO_alg(model,img,idx,ii,STATE_CAIGANGWA,STATE_SULIAO,STATE_YANWU):
    
    utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
    beijing_now = utc_now.astimezone(SHA_TZ)
    result_id =[]
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(800, s=32)  # check img_size
    if half:
        model.half()  # to FP16
    img0 = img #yuan
    img = letterbox(img, imgsz, stride=stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half()   # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    pred = model(img, augment=None)[0]
    pred = non_max_suppression(pred, 0.3, 0.3, classes=None, agnostic=False)
    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

            # Print results
            for c in det[:, -1].unique():
                n = (det[:, -1] == c).sum()  # detections per class

            # Write results
            for *xyxy, conf, cls in reversed(det):
                result_id.append(int(cls))
                plot_one_box(xyxy, img0, label=None, color=(0,0,0), line_thickness=3)
    if(len(list(set(result_id)))==0):
            pass
    for i in range(len(list(set(result_id)))):
        if(list(set(result_id))[i]==2):
            if(int(beijing_now.strftime('%M'))!=STATE_YANWU):
                cv2.imwrite("/home/alg/pic/2{}{}.jpg".format( idx,beijing_now.strftime('%Y.%m.%d.%H.%M')), img0)
                _data_smoke = {
                            "cameraid": idx,
                            "eventid": "103",
                            "time": str(beijing_now.strftime('%Y-%m-%d %H:%M:%S')),
                            "img": "/pic/2{}{}.jpg".format(idx,beijing_now.strftime('%Y.%m.%d.%H.%M')),
                            "video": "/hls/{}/aaa.m3u8".format(ii),
                            "longitude":"118.214005",
                            "latitude":"33.875902"
                        }
                print(str(beijing_now.strftime('%Y-%m-%d %H:%M:%S')))
                res = requests.post(url=url, headers=headers, json=_data_smoke).text
                print(res)
                STATE_YANWU = int(beijing_now.strftime('%M'))
        if(list(set(result_id))[i]==4):
            if(int(beijing_now.strftime('%M'))!=STATE_SULIAO):
                cv2.imwrite("/home/alg/pic/4{}{}.jpg".format(idx, beijing_now.strftime('%Y.%m.%d.%H.%M')), img0)
                _data_smoke = {
                            "cameraid": idx,
                            "eventid": "101",
                            "time": str(beijing_now.strftime('%Y-%m-%d %H:%M:%S')),
                            "img": "/pic/4{}{}.jpg".format( idx,beijing_now.strftime('%Y.%m.%d.%H.%M')),
                            "video": "/hls/{}/aaa.m3u8".format(ii),
                            "longitude":"118.214005",
                            "latitude":"33.875902"
                        }
                print(str(beijing_now.strftime('%Y-%m-%d %H:%M:%S')))
                res = requests.post(url=url, headers=headers, json=_data_smoke).text
                STATE_SULIAO = int(beijing_now.strftime('%M'))
        if(list(set(result_id))[i]==5):
            if(int(beijing_now.strftime('%M'))!=STATE_CAIGANGWA):
                cv2.imwrite("/home/alg/pic/5{}{}.jpg".format(idx, beijing_now.strftime('%Y.%m.%d.%H.%M')), img0)
                _data_smoke = {
                            "cameraid": idx,
                            "eventid": "100",
                            "time": str(beijing_now.strftime('%Y-%m-%d %H:%M:%S')),
                            "img": "/pic/5{}{}.jpg".format(idx, beijing_now.strftime('%Y.%m.%d.%H.%M')),
                            "video": "/hls/{}/aaa.m3u8".format(ii),
                            "longitude":"118.214005",
                            "latitude":"33.875902"
                        }
                print(str(beijing_now.strftime('%Y-%m-%d %H:%M:%S')))
                res = requests.post(url=url, headers=headers, json=_data_smoke).text
                STATE_CAIGANGWA = int(beijing_now.strftime('%M'))
    return STATE_CAIGANGWA,STATE_SULIAO,STATE_YANWU
