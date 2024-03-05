import sys

import UdpComms as U
import socket

import random
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from pickle import load
from scipy.spatial.transform import Rotation as R

from models import CNN2DGRU

import time

import concurrent.futures
from multiprocessing import Pool
import multiprocessing

import warnings
warnings.simplefilter(action='ignore')
     
class UdpSocket():
    def __init__(self, udpIP="192.168.1.2", portTX=8000, portRX=8001, 
                 enableRX=True, suppressWarnings=True):
        self.sock = U.UdpComms(udpIP=udpIP, portTX=portTX, portRX=portRX, 
                               enableRX=enableRX, suppressWarnings=suppressWarnings)
        print("Initialized UDP socket.")
        

class DataPreprocessor():
    def __init__(self, scaler=None):
        self.scaler = scaler
        self.valid_flag = False
        
    def _str_process(self, string):
        number_list = string.replace("(", "").replace(")", "") \
                            .replace(" ", "").split("/")
        number_list = [float(x) for x in number_list]
        return np.array(number_list)
    
    def preprocess(self, raw_features):
        if type(raw_features) == str:
            raw_features = raw_features.split(",")
        assert type(raw_features) == list

        # Discard the first 3 column: label, timestamp, counter
        rightRawX = raw_features[3:29]
        leftRawX = raw_features[29:55]

        # Check if features are valid
        leftValidFlag = True
        rightValidFlag = True
        if(rightRawX[0] == ' 0' or rightRawX[0] == " 0" or rightRawX[0] == 0):
            rightValidFlag = False
        if(leftRawX[0] == ' 0' or leftRawX[0] == " 0" or leftRawX[0] == 0):
            leftValidFlag = False

        rhand_p = []
        lhand_p = []
        for idx in range(0,26):
            if rightValidFlag:
                rhand_p.append(self._str_process(rightRawX[idx]))
            if leftValidFlag:
                lhand_p.append(self._str_process(leftRawX[idx]))

        # each hand has 26 joints --> 78 variables 
        leftX = np.zeros(78)
        rightX = np.zeros(78)
        # Center at the wrist
        for idx in range(0,25):
            if rightValidFlag:
                rightX[3*idx:3*(idx+1)] = rhand_p[idx] - rhand_p[-1]
            if leftValidFlag:
                leftX[3*idx:3*(idx+1)] = lhand_p[idx] - lhand_p[-1]

        if leftValidFlag:
            leftX = np.expand_dims(leftX, axis=0)
        if rightValidFlag:
            rightX = np.expand_dims(rightX, axis=0)

        return rightX, rightValidFlag, leftX, leftValidFlag
    

class HandFeatureBuffer:
    def __init__(self, window_length=30):
        self.buffer = np.zeros([window_length, 78])
        self.valid_length = 0
        self.ready_flag = False
        self.window_length = window_length
    
    def push(self, data, valid_flag):
        #data, valid_flag = self.data_preprocess(raw_data)
        if valid_flag == True:
            #print(data.shape)
            assert data.shape==(1,78)
            
            ## !!!!!!!!!!!!!!!!!!!!
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = data
            
            if self.valid_length < self.window_length:
                self.valid_length += 1
        else:
            self.valid_length = 0
    
    def sample(self):
        if self.valid_length < self.window_length:
            self.ready_flag = False
            return np.expand_dims(self.buffer, axis=0), self.ready_flag
        else:
            self.ready_flag = True
            return np.expand_dims(self.buffer, axis=0), self.ready_flag
        
    def reset(self):
        self.__init__()


def Udp2Buffer_Process(receive_lock, receive_share_data):
    ## Settings
    # Declare a local udp socket
    udpIP = "127.0.0.1" # local ip address
    udp_socket = UdpSocket(udpIP=udpIP, portTX=8000, portRX=8001, 
                           enableRX=True, suppressWarnings=True)
    
    leftHandFeatureBuffer = HandFeatureBuffer()
    rightHandFeatureBuffer = HandFeatureBuffer()
    dataPreprocessor = DataPreprocessor()

    print("udp2buffer process started")
    while True:
        # Keep running forever 
        try:
            # Receive from UDP
            data = udp_socket.sock.ReadReceivedData() # read data
        except:
            pass
            #with receive_lock:
            #    receive_share_data['hasNewFeature'] = False

        else:
            if data != None and len(data) > 10:
                #print("Receive Data: {}".format(data))
                # Preprocess and buffer
                rightX, rightValidFlag, leftX, leftValidFlag = dataPreprocessor.preprocess(data)
                with receive_lock:
                    rightHandFeatureBuffer.push(rightX, rightValidFlag)
                    leftHandFeatureBuffer.push(leftX, leftValidFlag)
                    receive_share_data['hasNewData'] = True
                    receive_share_data['rightHandBuffer'] = rightHandFeatureBuffer
                    receive_share_data['leftHandBuffer'] = leftHandFeatureBuffer
    
        time.sleep(0.005)
    print("udp2buffer process stopped")


class LabelBuffer:
    def __init__(self, length):
        self.length = length
        self.buffer = np.ones(length, dtype=int)*4 # 4 is the others label
    
    def push(self, label):
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = label
        
    def sample(self):
        return np.bincount(self.buffer).argmax()
    
    def reset(self):
        self.__init__(self.length)
    
        
def load_model(model_path, device):
    state_dict = torch.load(model_path, 
                            map_location=device)['state_dict']
    model = CNN2DGRU.CNN2DGRUModel(
                    input_dim=78, 
                    hidden_dim=128, 
                    layer_dim=3, 
                    output_dim=5, 
                    dropout_prob=0.0)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
        

    
def Buffer2ML_Process(receive_lock, receive_share_data,
                      label_lock, label_share_data):
    # check GPU availability               
    print("Check CUDA availability")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Run on {}...".format(device))
    
    print("Load CNN-GRU Model")
    model_path = './saved_model' +'/cnn2dgru.pth'
    model = load_model(model_path, device)
    model.eval()
    print(model)
    # Optional (for acceleration)
    #model = torch.compile(model, mode="reduce-overhead")

    X = torch.zeros(1, 30, 78).to(device, dtype=torch.float)
    y = model.forward(X)

    while True:
        with receive_lock:
            hasNewData = receive_share_data['hasNewData']
            rightHandFeatureBuffer = receive_share_data['rightHandBuffer']
            leftHandFeatureBuffer = receive_share_data['leftHandBuffer']

        if hasNewData:
            right_X, right_ready_flag = rightHandFeatureBuffer.sample()
            left_X, left_ready_flag =  leftHandFeatureBuffer.sample()

            if right_ready_flag:
                right_X = torch.tensor(right_X, dtype=torch.float).to(device)
                right_y = model.forward(right_X)
                right_y = int(right_y.argmax(dim=-1).detach().cpu())
                print("Right Hand Label: {}".format(right_y))
                with label_lock:
                    label_share_data['rightLabelBuffer'].push(right_y)
            else:
                with label_lock:
                    label_share_data['rightLabelBuffer'].reset()

            if left_ready_flag:
                left_X = torch.tensor(left_X, dtype=torch.float).to(device)
                left_y = model.forward(left_X)
                left_y = int(left_y.argmax(dim=-1).detach().cpu())
                print("Left Hand Label: {}".format(left_y))
                with label_lock:
                    label_share_data['leftLabelBuffer'].push(left_y)
            else:
                with label_lock:
                    label_share_data['leftLabelBuffer'].reset()

            with receive_lock:
                receive_share_data['hasNewData'] = False

        time.sleep(0.005)
        #     if dataPreprocessor.valid_flag:
        #         X, is_ready = feature_buffer.sample()
        #         if is_ready:
        #             try:
        #                 # As a batch but with batchsize = 1
        #                 X = np.expand_dims(X, axis=0)
        #                 X = torch.tensor(X).to(device, dtype=torch.float)
        #                 y_hat = model(X)
        #                 y = int(y_hat.argmax(dim=-1).detach().cpu())
        #                 label_buffer.push(y)
        #                 y = label_buffer.sample()
        #                 y = str(y)
        #             except:
        #                 print("Second Try Error")
        #                 pass
        #         else:
        #             print("Don't have enough data points")
        #             y = "Initializing"
        #     else:
        #         print("data is invalid")
        #         y = "Invalid Inputs"
        #     udp_socket.sock.SendDataToTarget(y, "192.168.1.38", 8000)
        #     print('Result from Jetson: ' + str(y))
        # except:
        #     print("First Try Error")
        #     pass


def LabelBuffer2UDP_Process(label_lock, label_share_data):
    # Initialize label buffer
    rightLabelBuffer = LabelBuffer(length=10)
    leftLabelBuffer = LabelBuffer(length=10)
    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Target device (Hololens 2) ip and port
    target_ip = "192.168.1.38"
    taret_port = 8000

    with label_lock:
        label_share_data['hasNewRightLabel'] = False
        label_share_data['hasNewLeftLabel'] = False
        label_share_data['rightLabelBuffer'] = rightLabelBuffer
        label_share_data['leftLabelBuffer'] = leftLabelBuffer

    while(True):
        with label_lock:
            hasNewRightLabel = label_share_data['hasNewRightLabel']
            hasNewLeftLabel = label_share_data['hasNewLeftLabel']
            if hasNewRightLabel:
                rightLabel = label_share_data['rightLabelBuffer'].sample()
                label_share_data['hasNewRightLabel'] = False
            if hasNewLeftLabel:
                leftLabel = label_share_data['leftLabelBuffer'].sample()
                label_share_data['hasNewLeftLabel'] = False
                
        if hasNewRightLabel:        
            udp_socket.sock.SendDataToTarget(rightLabel, target_ip, taret_port)
        if hasNewLeftLabel:    
            udp_socket.sock.SendDataToTarget(leftLabel, target_ip, taret_port)

        time.sleep(0.005)

def Timer_Process():
    while True:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        time.sleep(1)
      
        

if __name__ == "__main__":
    
    manager = multiprocessing.Manager()
    print("Initialize manager")

    receive_lock = manager.Lock()
    receive_share_data = manager.dict()
    receive_share_data['hasNewData'] = False
    receive_share_data['rightHandBuffer'] = None
    receive_share_data['leftHandBuffer'] = None


    label_lock = manager.Lock()
    label_share_data = manager.dict()
    label_share_data['hasNewRightLabel'] = False
    label_share_data['hasNewLeftLabel'] = False
    label_share_data['rightLabelBuffer'] = None
    label_share_data['leftLabelBuffer'] = None


    udp2buffer_process = multiprocessing.Process(target=Udp2Buffer_Process, 
                                              args=(receive_lock, receive_share_data))
    

    buffer2ml_process = multiprocessing.Process(target=Buffer2ML_Process, 
                                              args=(receive_lock, receive_share_data,
                                                    label_lock, label_share_data))

    labelbuffer2udp_process = multiprocessing.Process(target=LabelBuffer2UDP_Process, 
                                              args=(label_lock, label_share_data))
    
    timer_process = multiprocessing.Process(target=Timer_Process)
    
    # Activate all the processes
    udp2buffer_process.start()
    buffer2ml_process.start()
    labelbuffer2udp_process.start()
    timer_process.start()


    # Wait for all the processes to end
    udp2buffer_process.join()
    buffer2ml_process.join()
    labelbuffer2udp_process.join()
    timer_process.join()

