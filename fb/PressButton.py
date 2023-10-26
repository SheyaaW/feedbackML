import keyboard
import time
import socket
import subprocess

# Raspberry Pi's IP address and port (make sure they match the server)
host = '192.168.119.78'  # Replace with Raspberry Pi's actual IP address
port = 4444

# Create a socket object
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the server
client_socket.connect((host, port))



while True:
    try:
        
        if keyboard.is_pressed('w'):
            action = "w\n"
            client_socket.send(action.encode('utf-8'))
        elif keyboard.is_pressed('a'):
            action = "a\n"
            client_socket.send(action.encode('utf-8'))
        elif keyboard.is_pressed('s'):
            action = "s\n"
            client_socket.send(action.encode('utf-8'))
        elif keyboard.is_pressed('d'):
            action = "d\n"
            client_socket.send(action.encode('utf-8'))
        elif keyboard.is_pressed('p'):
            action = "p\n"
            client_socket.send(action.encode('utf-8'))
        elif keyboard.is_pressed('i'):
            action = "i\n"
            client_socket.send(action.encode('utf-8'))
        else:
            # No key is pressed, stop the robot or do nothing.
            action = "p\n"
            client_socket.send(action.encode('utf-8'))
            pass
    except KeyboardInterrupt:
        break

# Release the keyboard hook when the program exits.
keyboard.unhook_all()
client_socket.close()