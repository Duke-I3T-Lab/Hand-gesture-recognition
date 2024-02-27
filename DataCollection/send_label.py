import os

import logging
import time
import json

import numpy as np
import msvcrt

# pip install keyboard
import keyboard

#import http.server as server
from http.server import HTTPServer, BaseHTTPRequestHandler

#class HTTPRequestHandler(server.SimpleHTTPRequestHandler):
class HTTPRequestHandler(BaseHTTPRequestHandler):
    """Extend SimpleHTTPRequestHandler to handle PUT requests"""
    def do_PUT(self):
        self.send_response(201, 'Created')
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        file_length = int(self.headers['Content-Length'])
        body = self.rfile.read(file_length)
        self.wfile.write(body)
        print("PUT Received: ", body) 
        
        #body = body.decode()
        #six_DOF = body.split(',')
        
        #print("Cube DOF: ", six_DOF)
        #self.wfile.write()
        
    def do_GET(self):        
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        #c = msvcrt.getch()
        c = keyboard.read_event().name
        print('You entered: ' + str(c))
        self.wfile.write(c.encode())
       
if __name__ == '__main__':
    # replace the following ip and port with
    # the labelling PC's ip address
    # the labelling PC will act as a http server
    # the hololens will send requests to the server to query the ground truth label
    httpd = HTTPServer(("192.168.1.31", 8000), HTTPRequestHandler)
    print("HTTP Running")
    print("="*20)
    print("Please press keyboard to label data")
    httpd.serve_forever()      
    #server.test(HandlerClass=HTTPRequestHandler)