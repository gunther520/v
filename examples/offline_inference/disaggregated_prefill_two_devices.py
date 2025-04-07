# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
We can run this script on two separate nodes:
- Prefill node: python disaggregated_prefill_two_devices.py --mode prefill --ip <prefill_node_ip>
- Decode node: python disaggregated_prefill_two_devices.py --mode decode --ip <prefill_node_ip>
"""
import os
import time
import argparse
import socket
import threading
from multiprocessing import Event
import sys

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Port for synchronization signal (different from KV transfer port)
SYNC_PORT = 12347

def start_signal_server(ip_address="localhost"):
    """Starts a simple socket server to signal when prefill is done."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((ip_address, SYNC_PORT))
    server.listen(1)
    
    print(f"Signal server listening on {ip_address}:{SYNC_PORT}")
    
    # Create a container for the connection and address
    connection_info = {"conn": None, "connected": False}
    
    # Function to accept connection in background
    def accept_connection():
        try:
            conn, addr = server.accept()
            print(f"Decode node connected from {addr}")
            connection_info["conn"] = conn
            connection_info["connected"] = True
        except Exception as e:
            print(f"Error accepting connection: {e}")
    
    # Start the server in a background thread
    thread = threading.Thread(target=accept_connection, daemon=True)
    thread.start()
    
    # Function to send done signal when called
    def send_done_signal():
        # Wait for connection to be established, with timeout
        max_wait = 30  # seconds
        wait_time = 0
        while not connection_info["connected"] and wait_time < max_wait:
            time.sleep(1)
            wait_time += 1
            print("Waiting for decode node to connect...")
        
        if not connection_info["connected"]:
            print("Timeout waiting for decode node to connect")
            server.close()
            return
            
        try:
            connection_info["conn"].sendall(b"DONE")
            connection_info["conn"].close()
        except Exception as e:
            print(f"Error sending done signal: {e}")
        finally:
            server.close()
    
    return send_done_signal

def wait_for_prefill_signal(ip_address):
    """Waits for the signal that prefill is complete."""
    print(f"Waiting for prefill node to finish at {ip_address}:{SYNC_PORT}...")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    # Keep trying to connect until successful
    while True:
        try:
            client.connect((ip_address, SYNC_PORT))
            break
        except (ConnectionRefusedError, socket.error):
            print("Waiting for prefill node to become available...")
            time.sleep(2)
    
    # Wait for the "DONE" signal
    data = client.recv(1024)
    client.close()
    if data == b"DONE":
        print("Received signal that prefill is complete")
        return True
    return False

# Add this function to listen for decode completion
def start_completion_listener(ip_address="localhost"):
    """Starts a socket server to listen for completion signal from decode node."""
    COMPLETION_PORT = 12348  # Using a different port for completion signal
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((ip_address, COMPLETION_PORT))
    server.listen(1)
    
    print(f"Completion listener started on {ip_address}:{COMPLETION_PORT}")
    
    # Create a flag to signal when decode is complete
    completion_flag = {"completed": False}
    
    def listen_for_completion():
        try:
            conn, addr = server.accept()
            print(f"Received completion connection from {addr}")
            data = conn.recv(1024)
            conn.close()
            server.close()
            
            if data == b"COMPLETED":
                print("Decode node reports successful completion")
                completion_flag["completed"] = True
        except Exception as e:
            print(f"Error in completion listener: {e}")
    
    thread = threading.Thread(target=listen_for_completion, daemon=True)
    thread.start()
    
    return completion_flag

# Add this function to send completion signal
def send_completion_signal(ip_address):
    """Sends a signal to the prefill node that decode is complete."""
    COMPLETION_PORT = 12348  # Same port as in the listener
    
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((ip_address, COMPLETION_PORT))
        client.sendall(b"COMPLETED")
        client.close()
        print("Sent completion signal to prefill node")
    except Exception as e:
        print(f"Error sending completion signal: {e}")

def run_prefill(ip_address="localhost"):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Start signal server in a separate thread
    send_done_signal = start_signal_server(ip_address)
    
    # Start listener for completion signal
    completion_flag = start_completion_listener(ip_address)

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        # The decode node will actually "prefill" this request.
        "Tell me a very long story"*100,
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    ktc = KVTransferConfig.from_cli(
        f'{{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_ip":"{ip_address}","kv_port":12345}}'
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(#model="meta-llama/Llama-3.2-3B-Instruct",
              model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
              kv_transfer_config=ktc,
              max_model_len=20000,
              gpu_memory_utilization=0.8)

    llm.generate(prompts, sampling_params)
    print("Prefill node is finished.")
    
    # Signal that prefill is complete
    send_done_signal()

    # Keep running until completion signal is received or user interrupts
    print("Waiting for decode node to complete processing...")
    try:
        while not completion_flag["completed"]:
            time.sleep(1)
        print("Received completion signal from decode node. Shutting down.")
        os._exit(0)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(ip_address="localhost"):
    # We use GPU 0 on the decode node (since it's a different machine)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Using GPU 0 since this is on a different machine

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story"*100,
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    ktc = KVTransferConfig.from_cli(
        f'{{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_ip":"{ip_address}","kv_port":12345}}'
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(#model="meta-llama/Llama-3.2-3B-Instruct",
              model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
              kv_transfer_config=ktc,
              max_model_len=20000,
              gpu_memory_utilization=0.8)

    print("Initialized LLM on decode node, now waiting for prefill to complete...")
    
    # Wait for the prefill node to finish via network signal
    if not wait_for_prefill_signal(ip_address):
        print("Failed to receive proper signal from prefill node")
        return

    # At this point the kv-cache should have been transferred to this decode node
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    
    # Send completion signal back to prefill node
    send_completion_signal(ip_address)
    print("Decode node completed successfully")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM in disaggregated prefill mode")
    parser.add_argument("--mode", type=str, choices=["prefill", "decode"], required=True,
                        help="Run mode: 'prefill' for prefill node, 'decode' for decode node")
    parser.add_argument("--ip", type=str, default="localhost", 
                        help="IP address of the prefill node")
    args = parser.parse_args()
    
    if args.mode == "prefill":
        print(f"Starting prefill node on {args.ip}")
        run_prefill(args.ip)
    else:
        print(f"Starting decode node, connecting to prefill node at {args.ip}")
        run_decode(args.ip)