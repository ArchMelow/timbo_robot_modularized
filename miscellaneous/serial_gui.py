from tkinter import *
import serial.tools.list_ports
import functools
import json
import numpy as np
import random


class SerialComm(object):
    
    def __init__(self):
        self.ports = serial.tools.list_ports.comports()
        self.flags = [False for _ in range(len(self.ports))]
        self.end_flags = {k:v for k,v in zip(self.ports, self.flags)}
        print(self.end_flags)
        self.selectedPort = None 
        self.serialObj = serial.Serial()
        self.root = Tk()
        self.root.config(bg='grey')
        self.dataFrame = None
        self.buttons = []
        self.exit_buttons = []
        self.selectedPortIndex = None
        
    def initComPort(self, index):
        # disable the button while its operation
        self.buttons[index]['state'] = 'disabled'
        
        currentPort = str(self.ports[index])
        comPortVar = str(currentPort.split(' ')[0])
        #print(comPortVar)
        self.serialObj.port = comPortVar
        self.serialObj.baudrate = 115200
        
        try:
            self.serialObj.open()
        except Exception as e:
            print(e)
            self.buttons[index]['state'] = 'normal'
        else:
            self.selectedPort = self.ports[index]
            self.selectedPortIndex = index
            print('opened the port.')
            
    def endComm(self, port):
        self.end_flags[port] = True
        

    def initGui(self):
    
        for i, onePort in enumerate(self.ports):
            self.buttons.append(Button(self.root, text=onePort, font=('Calibri', '13'), height=1, width=45, command = functools.partial(self.initComPort, index = self.ports.index(onePort))))
            self.buttons[i].grid(row=i, column=0)
            self.exit_buttons.append(Button(self.root, text=f'exit {i}', font=('Calibri', '13'), height=1, width=15, command = functools.partial(self.endComm, port = onePort)))
            self.exit_buttons[i].grid(row=len(self.ports) + self.ports.index(onePort), column = 0)
        self.label_training = Label(self.root, text="training log: ", bg="white", relief="sunken")
        self.label_training.grid(row=0, column=1)
        self.dataCanvas = Canvas(self.root, width=400, height=400, bg='white')
        self.dataCanvas.grid(row=1, column=1, rowspan=100)
        self.dataFrame = Frame(self.dataCanvas, bg="white")
        self.dataCanvas.create_window((10,5),window=self.dataFrame,anchor='nw')
        self.label_summary = Label(self.root, text="message received: ", bg="white", relief="sunken")
        self.label_summary.grid(row = 0, column = 2, columnspan = 3, padx = 10, pady = 10)
        self.summary_text = Text(self.root, height=10, width=60, wrap='word', state = 'disabled')
        self.summary_text.grid(row=1, column=2, columnspan=3, padx=10, pady=10)
        #self.summaryFrame = Frame(self.summary_text, bg="white")
    '''
    def send(self, msg = None):
        if self.serialObj.isOpen():
            # write some message (must be bytestring)
            if isinstance(msg, bytes):
                return self.serialObj.write(msg)
        return -1    
    '''

    def checkSerialPort(self):
        if self.serialObj.isOpen() and self.serialObj.in_waiting:
            recentPacket = b''
            tmp_len = self.serialObj.in_waiting
            recentPacket = self.serialObj.read(tmp_len)
                
            recentPacketString =recentPacket.decode('utf-8').rstrip('\n')
            #print(recentPacketString)
            
            
        
            # Check if a label already exists, if not, create one
            if not hasattr(self, 'label'):
                self.label = Label(self.dataFrame, text="", bg="white")
                self.label.pack()
            # Update the text of the existing label
            self.label.config(text=recentPacketString)
            
            try:
                data = json.loads(recentPacketString)
            except:
                data = recentPacketString
                #raise ValueError('send the packet in JSON format.')
            return data    

    def update_gui(self):
        self.root.update()  
        #   self.dataCanvas.config(scrollregion=self.dataCanvas.bbox("all"))