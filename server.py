import socket
import ssl

def start_server():
    host = "127.0.0.1"
    port = 65432

    # TLS context
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile="server.crt", keyfile="server.key")

    # Create socket
    bindsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    bindsocket.bind((host, port))
    bindsocket.listen(5)
    print("Secure Relay Server started...")

    while True:
        newsocket, fromaddr = bindsocket.accept()
        conn = context.wrap_socket(newsocket, server_side=True)
        print(f"Connection from {fromaddr}")

        data = conn.recv(4096)
        if data:
            print("Relaying message...")
            conn.sendall(data)  # Relay back to client
        conn.close()

if __name__ == "__main__":
    start_server()
