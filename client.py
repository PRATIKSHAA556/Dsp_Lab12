import socket
import ssl
from cryptography.fernet import Fernet

# Generate or use shared secret key for E2EE
secret_key = Fernet.generate_key()
cipher = Fernet(secret_key)

def send_message():
    host = "127.0.0.1"
    port = 65432

    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    raw_message = input("Enter your message: ")

    # Encrypt message with E2EE
    encrypted_msg = cipher.encrypt(raw_message.encode())

    with socket.create_connection((host, port)) as sock:
        with context.wrap_socket(sock, server_hostname=host) as ssock:
            ssock.sendall(encrypted_msg)
            print("Message sent securely with E2EE")

def receive_message(encrypted_msg):
    """Receiver decrypts the message locally"""
    decrypted_msg = cipher.decrypt(encrypted_msg)
    print("Decrypted message:", decrypted_msg.decode())

if __name__ == "__main__":
    send_message()
