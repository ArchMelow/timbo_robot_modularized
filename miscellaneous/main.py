from serial_gui import SerialComm
from models import *
import time

def runner():
    max_episode_num = 500
    
    agent = DDPGagent()
    
    while True:
        ret = agent.train(max_episode_num)
        if ret != -1: # training success
            agent.plot_result()
        else:
            print(ret)
            print('training terminated.')
            agent.comm_obj.serialObj.close() # close the port
            time.sleep(1) # wait 1s to close the port
            agent.comm_obj.buttons[agent.comm_obj.selectedPortIndex]['state'] = 'normal'
            
    
    
if __name__ == '__main__':
    runner()
    
    
    