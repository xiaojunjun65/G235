import requests
import json
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import cv2 
from utils.alg import YOLO_alg
from utils.http_seve import requests_load
from utils.json_data import motion_state,point_mate,video_to_pic
#时间管理区域
SHA_TZ = timezone(
    timedelta(hours=8),
    name='Asia/Shanghai',
)
utc_now = datetime.utcnow().replace(tzinfo=timezone.utc)
beijing_now = utc_now.astimezone(SHA_TZ)

def main_program(json_data,idx,ii,model_devic):
    STATE_CAIGANGWA=0
    STATE_SULIAO =0
    STATE_YANWU=0
    print(ii,idx)
    while True:
        beijing_now = utc_now.astimezone(SHA_TZ)
        if int(beijing_now.strftime('%H')) >=8 and int(beijing_now.strftime('%H')) <=17:
            res = requests_load(idx)
            x,y = motion_state(res)
            if x==-1 and y == -1:
                pass
            else:
                Point_result = point_mate(json_data,x,y)
                if Point_result:
                    PIC_STATE,pic =video_to_pic(json_data)
                    if PIC_STATE:
                        print("this pic is in the infer",idx)
                        STATE_CAIGANGWA,STATE_SULIAO,STATE_YANWU=YOLO_alg(model_devic,pic,idx,ii,STATE_CAIGANGWA,STATE_SULIAO,STATE_YANWU)
                