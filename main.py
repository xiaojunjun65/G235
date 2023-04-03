import threading
import json
from models.experimental import  attempt_load
from utils.master_control import main_program
MODEL_PATH = "utils/fire.pt"

if __name__ == '__main__':
    model1 = attempt_load(MODEL_PATH,map_location="cuda:0")
    model2 = attempt_load(MODEL_PATH,map_location="cuda:1")
    #config
    THREAD_IDS = []
    #####
    with open('data/config.json') as f:
        # 加载JSON数据
        data = json.load(f)
    for i in range(len(data)):
        if i<9:
            str_ids = "1000"+str(i+1)
        else:
            str_ids = "100"+str(i+1)
        if i>14:
            model_devic = model1
        else:
            model_devic = model2
        if data[str_ids]['state']:
            t = threading.Thread(target=main_program,args=(data[str_ids],str_ids,i+1,model_devic))
            THREAD_IDS.append(t)
    for t in THREAD_IDS:
        t.start()
        
    
