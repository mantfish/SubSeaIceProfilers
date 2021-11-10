import network
import socket
from machine import Pin
from time import sleep_ms as delay
import webrepl

p13 = Pin(13, Pin.OUT)
p14 = Pin(14, Pin.OUT)
p12 = Pin(12, Pin.OUT)
p27 = Pin(27, Pin.OUT)

delay_time = 5

def write(a,b,c,d):
    p13.value(a)
    p14.value(c)
    p12.value(b)
    p27.value(d)
    return None

def one_step():
    write(1, 1, 0, 0)
    delay(delay_time)
    write(0, 1, 1, 0)
    delay(delay_time)
    write(0, 0, 1, 1)
    delay(delay_time)
    write(1, 0, 0, 1)
    delay(delay_time)
    return None

def move_stepper(data):
    for i in range(int(data)):
        one_step()
        print(i)
    print("Done.")
    write(0,0,0,0)
    return None

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind(('',80))

s.listen(5)
print ('server started and listening')

(clientsocket, address) = s.accept()
print ("connection found!")

while True:
    data = clientsocket.recv(1024).decode()
    print(data)
    if data:
        data = str(data)
        print(data)
        move_stepper(data)
    else:
        pass



