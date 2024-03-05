import csv
import socket
import time

def send_csv_over_udp(file_path, udp_host, udp_port):
    # Create a UDP socket
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Open the CSV file
    with open(file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            # Convert the row to a string
            row_str = ','.join(row)
            # Send the row over UDP
            udp_socket.sendto(row_str.encode(), (udp_host, udp_port))
            time.sleep(0.03)


    # Close the socket after sending all rows
    udp_socket.close()

# Example usage:
file_path = 'test_left.csv'
udp_host = '127.0.0.1'  # UDP server host
udp_port = 8001        # UDP server port
send_csv_over_udp(file_path, udp_host, udp_port)
