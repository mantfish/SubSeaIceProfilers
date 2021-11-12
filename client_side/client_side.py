import socket

#CONVERSION FACTOR: 500 STEPS = 31mm

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(('192.168.4.1',80))

while True:
    data = input("Enter steps to write: ")
    s.send(data.encode())