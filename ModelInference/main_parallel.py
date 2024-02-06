import sys
#from http.server import HTTPServer, BaseHTTPRequestHandler
import UdpComms as U

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

import warnings
warnings.simplefilter(action='ignore')
     
class UdpSocket():
    def __init__(self, udpIP="192.168.1.2", portTX=8000, portRX=8001, 
                 enableRX=True, suppressWarnings=True):
        self.sock = U.UdpComms(udpIP=udpIP, portTX=portTX, portRX=portRX, 
                               enableRX=enableRX, suppressWarnings=suppressWarnings)
        print("Initialized UDP socket.")
        

class DataPreprocessor():
    def __init__(self, scaler):
        self.scaler = scaler
        self.valid_flag = False
        
    def _str_process(self, string):
        number_list = string.replace("(", "").replace(")", "") \
                            .replace(" ", "").split("/")
        number_list = [float(x) for x in number_list]
        return np.array(number_list)
   
    def _quaternion_2_euler(self, quaternion):
        rotations = R.from_quat(quaternion)
        # Convert quaternion to euler in radians
        euler = rotations.as_euler(seq='xyz', degrees=False)
        return euler
    
    def preprocess(self, raw_features):
        error_code = []
        X = np.zeros(171)
        
        if type(raw_features) == str:
            raw_features = raw_features.split(",")
        assert type(raw_features) == list
        
        # the first 14 columns are irrelvant
        valid_flag = True
        for idx, feature in enumerate(raw_features[14:]):
            if feature == ' 0' or feature == " 0" or feature == 0:
                valid_flag = False
                self.valid_flag = valid_flag
                return X, self.valid_flag
        self.valid_flag = valid_flag
        if self.valid_flag == True:
            
            tool_p = self._str_process(raw_features[14])
            tool_e = self._str_process(raw_features[15])
            tool_e = self._quaternion_2_euler(tool_e)
            head_p = self._str_process(raw_features[16])
            head_e = self._str_process(raw_features[17])
            head_e = self._quaternion_2_euler(head_e)
            '''
            with Pool(4) as p:
                tool_p, tool_e, head_p, head_e = p.map(self._str_process,
                                                      [raw_features[14],
                                                       raw_features[15],
                                                       raw_features[16],
                                                       raw_features[17]])
            with Pool(2) as p:
                tool_e, head_e = p.map(self._quaternion_2_euler,
                                       [tool_e,
                                        head_e])
            '''
            rhand_p = []
            lhand_p = []
            
            for idx in range(0,26):
                rhand_p.append(self._str_process(raw_features[18+idx]))
                lhand_p.append(self._str_process(raw_features[44+idx]))
            '''
            with Pool(4) as p:
                rhand_p = p.map(self._str_process, [raw_features[18+idx] for idx in range(0,26)])
            with Pool(4) as p:
                lhand_p = p.map(self._str_process, [raw_features[44+idx] for idx in range(0,26)])
            '''
            X[0:3] = tool_p - head_p
            X[3:6] = tool_e
            X[6:9] = head_e
            X[9:12] = rhand_p[-1] - head_p
            X[12:15] = lhand_p[-1] - head_p
            X[15:18] = tool_p - lhand_p[-1]
            X[18:21] = tool_p - rhand_p[-1]
            for idx in range(0,25):
                X[21+3*idx:21+3*(idx+1)] = rhand_p[idx] - rhand_p[-1]
                X[96+3*idx:96+3*(idx+1)] = lhand_p[idx] - lhand_p[-1]
            X = np.expand_dims(X, axis=0)
            X = self.scaler.transform(X)
        return X, self.valid_flag


class FeatureBuffer:
    def __init__(self):
        self.buffer = np.zeros([20,171])
        self.valid_length = 0
        self.ready_flag = False
    
    def push(self, data, valid_flag):
        #data, valid_flag = self.data_preprocess(raw_data)
        if valid_flag == True:
            #print(data.shape)
            assert data.shape==(1,171)
            self.buffer = np.roll(self.buffer, 1, axis=0)
            self.buffer[0] = data
            if self.valid_length < 20:
                self.valid_length += 1
        else:
            self.valid_length = 0
    
    def sample(self):
        if self.valid_length < 20:
            self.ready_flag = False
            return self.buffer, self.ready_flag
        else:
            self.ready_flag = True
            return self.buffer, self.ready_flag
        
    def reset(self):
        self.__init__()

class LabelBuffer:
    def __init__(self, length):
        self.length = length
        self.buffer = np.zeros(length, dtype=int)
    
    def push(self, label):
        self.buffer = np.roll(self.buffer, 1, axis=0)
        self.buffer[0] = label
        
    def sample(self):
        return np.bincount(self.buffer).argmax()
        
def load_model(model_path, device):
    state_dict = torch.load(model_path, 
                            map_location=device)['state_dict']
    model = CNN2DGRU.CNN2DGRUModel(input_dim=171, 
                    hidden_dim=32, 
                    layer_dim=1, 
                    output_dim=7, 
                    dropout_prob=0.5)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
        

def udp2buffer():
    while True:
        try:
            data = udp_socket.sock.ReadReceivedData() # read data
            if data != None and len(data) > 10:
                print("Receive Data")
                x,is_valid = dataPreprocessor.preprocess(data)
                feature_buffer.push(x,is_valid)
        except:
            pass

    print("UDP Stop")

    
def buffer2ML2udp():
    while True:
        try:
            if dataPreprocessor.valid_flag:
                X, is_ready = feature_buffer.sample()
                if is_ready:
                    try:
                        # As a batch but with batchsize = 1
                        X = np.expand_dims(X, axis=0)
                        X = torch.tensor(X).to(device, dtype=torch.float)
                        y_hat = model(X)
                        y = int(y_hat.argmax(dim=-1).detach().cpu())
                        label_buffer.push(y)
                        y = label_buffer.sample()
                        y = str(y)
                    except:
                        print("Second Try Error")
                        pass
                else:
                    print("Don't have enough data points")
                    y = "Initializing"
            else:
                print("data is invalid")
                y = "Invalid Inputs"
            udp_socket.sock.SendDataToTarget(y, "192.168.1.38", 8000)
            print('Result from Jetson: ' + str(y))
        except:
            print("First Try Error")
            pass


def timer():
    while True:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print(current_time)
        time.sleep(1)
      
        
if __name__ == '__main__':      
    # check GPU availability               
    print("1. Check CUDA availability")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("Run on {}...".format(device))
    
    print("2. Load CNN-GRU Model")
    model_path = './saved_model' +'/cnn2dgru.pth'
    model = load_model(model_path, device)
    model.eval()
    print(model)
    
    # Initialize the model with a dummy input
    X = np.zeros([1,20,171])
    X = torch.tensor(X).to(device, dtype=torch.float)
    y = model(X)
    
    print("3. Load DataPreprocessor")
    scaler = load(open('scaler.pkl', 'rb'))
    dataPreprocessor = DataPreprocessor(scaler)
    feature_buffer = FeatureBuffer()
    feature_buffer.reset()
    label_buffer = LabelBuffer(10)
    
    print("4. Initialize UDP socket")
    udp_socket = UdpSocket(udpIP="192.168.1.40", portTX=8000, portRX=8001, 
                           enableRX=True, suppressWarnings=True)
              
    print("5. Start Send/Receive Data")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        executor.submit(udp2buffer)
        executor.submit(udp2buffer)
        executor.submit(udp2buffer)
        executor.submit(buffer2ML2udp)
        executor.submit(buffer2ML2udp)
        executor.submit(timer)