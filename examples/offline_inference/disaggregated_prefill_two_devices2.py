# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
using AsyncLLMEngine for streaming and metrics measurement.

Run modes:
- Prefill node: python your_script_name.py --mode prefill --ip <prefill_node_ip>
- Decode node: python your_script_name.py --mode decode --ip <prefill_node_ip>
"""
import os
import time
import argparse
import socket
import threading
import asyncio # Added asyncio
import uuid # Added for unique request IDs
import sys
import numpy as np

# VLLM imports
from vllm.engine.arg_utils import AsyncEngineArgs # Changed
from vllm.engine.async_llm_engine import AsyncLLMEngine # Changed
from vllm import SamplingParams
from vllm.config import KVTransferConfig

# --- Signaling Code (largely unchanged, but ensure it doesn't block asyncio loop) ---

SYNC_PORT = 12347
COMPLETION_PORT = 12348

# start_signal_server, wait_for_prefill_signal, start_completion_listener, send_completion_signal
# remain the same functions as before, as they use standard sockets/threading.
# We will call the blocking parts using asyncio.to_thread later.

def start_signal_server(ip_address="localhost"):
    """Starts a simple socket server to signal when prefill is done."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((ip_address, SYNC_PORT))
    except OSError as e:
        print(f"Error binding to signal port {SYNC_PORT}: {e}. "
              f"Another process might be using it.")
        # Return a dummy function if binding fails
        return lambda: print("Error: Signal server could not start.")
    server.listen(1)

    print(f"Signal server listening on {ip_address}:{SYNC_PORT}")

    connection_info = {"conn": None, "connected": False, "server_socket": server} # Store server socket

    def accept_connection():
        try:
            conn, addr = server.accept()
            print(f"Decode node connected from {addr}")
            connection_info["conn"] = conn
            connection_info["connected"] = True
        except Exception as e:
            print(f"Error accepting connection: {e}")
            # Ensure server socket is closed if accept fails
            if connection_info["server_socket"]:
                try:
                    connection_info["server_socket"].close()
                    connection_info["server_socket"] = None
                except Exception: pass
        finally:
             # Make sure the server socket is closed after accepting one connection or error
             if connection_info["server_socket"]:
                try:
                    connection_info["server_socket"].close()
                    connection_info["server_socket"] = None
                except Exception: pass


    thread = threading.Thread(target=accept_connection, daemon=True)
    thread.start()

    def send_done_signal():
        max_wait = 60
        wait_time = 0
        while not connection_info["connected"] and wait_time < max_wait:
            time.sleep(1) # Using time.sleep is okay here as it's for the lambda
            wait_time += 1
            if wait_time % 5 == 0:
                 print("Waiting for decode node connection to send signal...")

        if not connection_info["connected"]:
            print("Timeout waiting for decode node connection for signal.")
            # Ensure server socket is closed on timeout too
            if connection_info["server_socket"]:
                try:
                    connection_info["server_socket"].close()
                    connection_info["server_socket"] = None
                except Exception: pass
            return

        try:
            if connection_info["conn"]:
                connection_info["conn"].sendall(b"DONE")
                connection_info["conn"].close()
                print("Sent DONE signal.")
            else:
                 print("Error: No connection available to send DONE signal.")
        except Exception as e:
            print(f"Error sending done signal: {e}")
        finally:
            # Clean up connection and server socket if still open
            if connection_info["conn"]:
                try: connection_info["conn"].close()
                except Exception: pass
            if connection_info["server_socket"]:
                try:
                    connection_info["server_socket"].close()
                    connection_info["server_socket"] = None
                except Exception: pass

    return send_done_signal


# This is a blocking function, call with asyncio.to_thread
def wait_for_prefill_signal(ip_address):
    """Waits for the signal that prefill is complete."""
    print(f"Waiting for prefill node signal at {ip_address}:{SYNC_PORT}...")
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    connected = False
    while not connected:
        try:
            client.connect((ip_address, SYNC_PORT))
            connected = True
            print("Connected to signal server.")
        except (ConnectionRefusedError, socket.error, OSError) as e:
            print(f"Waiting for prefill node signal server... ({e})")
            time.sleep(2) # Blocking sleep okay here, as it runs in a separate thread

    try:
        client.settimeout(600.0) # Add a timeout for receiving data
        data = client.recv(1024)
        client.close()
        if data == b"DONE":
            print("Received signal: Prefill is complete")
            return True
        else:
            print(f"Received unexpected signal data: {data}")
            return False
    except socket.timeout:
        print("Timeout waiting for DONE signal from prefill node.")
        client.close()
        return False
    except Exception as e:
        print(f"Error receiving signal: {e}")
        client.close()
        return False


def start_completion_listener(ip_address="localhost"):
    """Starts a socket server to listen for completion signal from decode node."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((ip_address, COMPLETION_PORT))
    except OSError as e:
        print(f"Error binding to completion port {COMPLETION_PORT}: {e}. ")
        return {"completed": False, "error": True}

    server.listen(1)
    print(f"Completion listener started on {ip_address}:{COMPLETION_PORT}")

    completion_flag = {"completed": False, "error": False, "server_socket": server} # Store socket

    def listen_for_completion():
        try:
            # Set a timeout for accept, e.g., 5 minutes (300 seconds)
            # This prevents the thread from hanging indefinitely if decode node never connects
            server.settimeout(300.0)
            conn, addr = server.accept()
            print(f"Received completion connection from {addr}")
            conn.settimeout(10.0) # Timeout for receiving data
            data = conn.recv(1024)
            conn.close()


            if data == b"COMPLETED":
                print("Decode node reports successful completion via signal.")
                completion_flag["completed"] = True
            else:
                 print(f"Completion listener received unexpected data: {data}")
        except socket.timeout:
             print("Timeout waiting for completion connection or data.")
             completion_flag["error"] = True # Indicate timeout error
        except Exception as e:
            print(f"Error in completion listener: {e}")
            completion_flag["error"] = True # Indicate general error
        finally:
            # Always close the server socket in the thread
            if completion_flag["server_socket"]:
                try:
                    completion_flag["server_socket"].close()
                    completion_flag["server_socket"] = None
                except Exception: pass

    thread = threading.Thread(target=listen_for_completion, daemon=True)
    thread.start()

    return completion_flag # Return the dictionary


# This is a blocking function, call with asyncio.to_thread
def send_completion_signal(ip_address):
    """Sends a signal to the prefill node that decode is complete."""
    retries = 3
    while retries > 0:
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.settimeout(10.0) # Connection timeout
            client.connect((ip_address, COMPLETION_PORT))
            client.sendall(b"COMPLETED")
            client.close()
            print("Sent completion signal to prefill node")
            return True # Success
        except (ConnectionRefusedError, socket.error, socket.timeout, OSError) as e:
            print(f"Error sending completion signal (retrying... {retries-1} left): {e}")
            retries -= 1
            time.sleep(2) # Blocking sleep okay here (runs in thread)
        except Exception as e:
            print(f"Unexpected error sending completion signal: {e}")
            break # Don't retry on unexpected errors
    print("Failed to send completion signal after multiple retries.")
    return False # Failure


# --- Async Prefill Function ---
async def run_prefill(ip_address="localhost"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Start signal server (non-blocking setup, returns lambda)
    send_done_signal_func = start_signal_server(ip_address)

    # Start listener for completion signal (non-blocking setup)
    completion_flag = start_completion_listener(ip_address)
    if completion_flag.get("error"):
        print("Could not start completion listener. Exiting prefill node.")
        return # Exit if listener failed

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story about " * 50 + "a brave knight.",
    ]
    # max_tokens=1 signals prefill-only behavior to the engine implicitly
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # --- Setup AsyncEngineArgs ---
    ktc_dict = KVTransferConfig.from_cli(
        f'{{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":0,"kv_parallel_size":2,"kv_ip":"{ip_address}","kv_port":12345}}'
    )

    engine_args = AsyncEngineArgs(
        # model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
        model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        max_model_len=700,
        gpu_memory_utilization=0.95,
        max_num_seqs=80,
        # tensor_parallel_size=1, # Adjust as needed
        kv_transfer_config=ktc_dict # Pass the dict here
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Prefill Engine Initialized.")


    # Perform prefill - generate triggers KV cache computation and transfer setup
    print("Starting prefill generation (computes and prepares KV cache)...")
    t_prefill_start = time.perf_counter()

    # Generate unique request IDs
    request_ids = [f"prefill-{uuid.uuid4()}" for _ in prompts]
    results_generators = [] # Although we don't process results, we need to await
    for i, prompt in enumerate(prompts):
        # We need to await generate, even if we discard the result for prefill
        results_generators.append(
             engine.generate(prompt, sampling_params, request_ids[i])
        )

    # Await all generation tasks concurrently to finish prefill calculation
    # We iterate through the generators to ensure they complete
    for gen in results_generators:
        try:
             async for _ in gen: # Consume the generator (yields one result due to max_tokens=1)
                  pass
        except Exception as e:
            print(f"Warning: Exception during prefill generation consumption: {e}")


    t_prefill_end = time.perf_counter()
    print(f"Prefill computation finished in {t_prefill_end - t_prefill_start:.4f} seconds.")

    # Signal the decode node (run blocking function in thread)
    print("Signaling decode node...")
    # Ensure the function is callable before calling it
    if callable(send_done_signal_func):
         # Running the signal sending in a thread isn't strictly necessary
         # as it happens after the main async work, but good practice if it could block.
         # await asyncio.to_thread(send_done_signal_func)
         send_done_signal_func() # Direct call might be fine here
    else:
         print("Error: send_done_signal function not available.")


    # Keep prefill node alive until decode node confirms completion
    print("Prefill node waiting for decode node completion signal...")
    wait_start_time = time.time()
    try:
        while not completion_flag.get("completed", False):
            if completion_flag.get("error", False):
                print("Error detected in completion listener. Stopping wait.")
                break
            if time.time() - wait_start_time > 300: # 5 min timeout
                print("Timeout waiting for completion signal from decode node.")
                break
            await asyncio.sleep(1) # Use asyncio.sleep

        if completion_flag.get("completed", False):
            print("Received completion signal. Prefill node shutting down.")
        else:
            print("Prefill node shutting down without confirmed completion signal.")

    except asyncio.CancelledError:
         print("Prefill task cancelled.")
    except KeyboardInterrupt:
        print("Prefill script stopped by user.")
    finally:
        print("Prefill process exiting.")
        # Force exit if tasks/threads are hanging
        os._exit(0)


# --- Async Decode Function ---
async def run_decode(ip_address="localhost"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "2" # Ensure correct GPU

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story about " * 50 + "a brave knight.",
    ]
    # Enable streaming via SamplingParams
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=128)

    # --- Setup AsyncEngineArgs ---
    ktc_dict = KVTransferConfig.from_cli(
        f'{{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_ip":"{ip_address}","kv_port":12345}}'
    )
    engine_args = AsyncEngineArgs(
        # model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95",
        model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
        max_model_len=2000,
        gpu_memory_utilization=0.8,
        # tensor_parallel_size=1, # Adjust as needed
        kv_transfer_config=ktc_dict
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Decode Engine Initialized.")


    # Wait for the prefill node signal (run blocking call in thread)
    print("Waiting for prefill node signal...")
    t_wait_start = time.perf_counter()
    try:
        prefill_ready = await asyncio.to_thread(wait_for_prefill_signal, ip_address)
        if not prefill_ready:
            print("Failed to receive signal or prefill node indicated failure. Exiting.")
            return
    except Exception as e:
        print(f"Error waiting for prefill signal: {e}")
        return

    t_wait_end = time.perf_counter()
    print(f"Received prefill signal after {t_wait_end - t_wait_start:.4f} seconds.")
    print("KV cache transfer should be complete or happening now.")

    # --- Metrics Initialization ---
    request_metrics = {}
    all_itls = []
    start_times = {} # Store per-request start time

    # --- Start Generation and Measurement ---
    print("Starting decode generation...")
    t_generate_call_time = time.perf_counter() # Time generation call is made

    # Generate unique request IDs
    request_ids = f"decode-{uuid.uuid4()}"

    # Submit all requests to the engine. engine.generate returns an async generator.
    results_generator = engine.generate(prompts, sampling_params, request_id=request_ids)

    # Process the stream of results
    async for request_output in results_generator:
        t_token_received = time.perf_counter() # Time the update arrived
        request_id = request_output.request_id
        num_output_tokens = len(request_output.outputs[0].token_ids)

        # Initialize metrics for this request if first time seeing it
        if request_id not in request_metrics:
            start_times[request_id] = t_generate_call_time # Record start time for this specific request
            request_metrics[request_id] = {
                'prompt': request_output.prompt,
                'prompt_len': len(request_output.prompt_token_ids),
                'start_time': start_times[request_id],
                'first_token_time': None,
                'token_timestamps': [],
                'finished': False,
                'final_output': None,
                'num_output_tokens': 0 # Track previous count
            }
            print(f"  Request {request_id[:8]}...: Started processing.")

        metrics = request_metrics[request_id]
        prev_num_tokens = metrics['num_output_tokens']

        # If new tokens have arrived in this update
        if num_output_tokens > prev_num_tokens:
            new_tokens_count = num_output_tokens - prev_num_tokens

            # Record timestamp for each *new* token
            for _ in range(new_tokens_count):
                 metrics['token_timestamps'].append(t_token_received)

            # Check if this is the *very first* token for *this* request
            if metrics['first_token_time'] is None and num_output_tokens > 0:
                metrics['first_token_time'] = t_token_received # Use the arrival time
                ttft = metrics['first_token_time'] - metrics['start_time']
                print(f"  Request {request_id[:8]}...: TTFT = {ttft:.4f}s")

            # Update the count
            metrics['num_output_tokens'] = num_output_tokens

        # Store final state when finished
        if request_output.finished:
            metrics['finished'] = True
            metrics['final_output'] = request_output # Store the last output
            print(f"  Request {request_id[:8]}...: Finished.")


    t_generate_end = time.perf_counter() # Time when the last result was processed
    print(f"\nTotal processing time on decode node (from generate call to last result): {t_generate_end - t_generate_call_time:.4f} seconds")
    print("-" * 30)
    print("Metrics Calculation:")
    print("-" * 30)

    # --- Calculate and Print Metrics ---
    total_output_tokens = 0
    total_prompt_tokens_decode = 0 # Check prompt tokens processed by decode

    for req_id in request_ids: # Iterate in original order
        if req_id not in request_metrics:
            print(f"Request ID: {req_id} - No metrics recorded (maybe failed?)")
            continue

        metrics = request_metrics[req_id]
        print(f"Request ID: {req_id}")
        print(f"Prompt: {metrics['prompt'][:50]!r}...")
        prompt_len = metrics['prompt_len']
        total_prompt_tokens_decode += prompt_len

        if not metrics['finished'] or metrics['num_output_tokens'] == 0:
            status = "did not complete" if not metrics['finished'] else "produced 0 tokens"
            print(f"  Generation {status}.")
            continue

        num_tokens = metrics['num_output_tokens']
        total_output_tokens += num_tokens
        print(f"  Generated Tokens: {num_tokens}")

        # TTFT
        if metrics['first_token_time']:
            ttft = metrics['first_token_time'] - metrics['start_time']
            print(f"  TTFT (Time To First Token): {ttft:.6f} s")
        else:
            print("  TTFT: Not measured (no tokens generated or first token time missing)")

        # ITL and TPOT
        if num_tokens > 1 and len(metrics['token_timestamps']) > 1:
            token_times = metrics['token_timestamps']
            # Ensure timestamps are sorted (should be, but safety check)
            token_times.sort()
            itls = np.diff(token_times)

            # Filter out zero or negative ITLs if they somehow occur
            itls = itls[itls > 0]

            if len(itls) > 0:
                avg_itl = np.mean(itls)
                p90_itl = np.percentile(itls, 90)
                p99_itl = np.percentile(itls, 99)

                # TPOT (Method 1: Avg ITL)
                tpot_itl = avg_itl

                # TPOT (Method 2: Based on total decode phase)
                # Ensure first_token_time is valid
                if metrics['first_token_time'] and metrics['first_token_time'] <= token_times[-1]:
                    total_decode_phase_time = token_times[-1] - metrics['first_token_time']
                    # Avoid division by zero if only one ITL interval exists
                    num_intervals = len(itls) # Number of valid inter-token intervals
                    tpot_total = total_decode_phase_time / num_intervals if num_intervals > 0 else 0.0
                else:
                    tpot_total = float('nan') # Not calculable


                all_itls.extend(itls) # Collect for global stats

                print(f"  ITL (Inter-Token Latency, based on {len(itls)} intervals):")
                print(f"      Average: {avg_itl:.6f} s")
                print(f"      P90:     {p90_itl:.6f} s")
                print(f"      P99:     {p99_itl:.6f} s")
                print(f"  TPOT (Time Per Output Token, based on Avg ITL): {tpot_itl:.6f} s")
                print(f"  TPOT (Time Per Output Token, based on total decode phase): {tpot_total:.6f} s")
            else:
                print("  ITL/TPOT: Not calculated (no valid positive intervals between tokens)")

        elif num_tokens == 1:
             print("  ITL/TPOT: Not applicable (only 1 token generated)")
        else:
             print("  ITL/TPOT: Not measured (less than 2 tokens or timestamps missing)")
        print("-" * 10)


    # --- Overall Metrics ---
    print("-" * 30)
    print("Overall Performance:")
    print("-" * 30)
    if all_itls:
        overall_avg_itl = np.mean(all_itls)
        overall_p90_itl = np.percentile(all_itls, 90)
        overall_p99_itl = np.percentile(all_itls, 99)
        print(f"Overall Average ITL: {overall_avg_itl:.6f} s")
        print(f"Overall P90 ITL:     {overall_p90_itl:.6f} s")
        print(f"Overall P99 ITL:     {overall_p99_itl:.6f} s")
        print(f"Overall TPOT (Avg ITL): {overall_avg_itl:.6f} s")

        # Calculate overall throughput (tokens per second)
        # Use the time from the first generate call to the last token received across all requests
        first_request_start_time = min(m['start_time'] for m in request_metrics.values() if 'start_time' in m) if request_metrics else t_generate_call_time
        last_token_time = max(m['token_timestamps'][-1] for m in request_metrics.values() if m.get('token_timestamps')) if any(m.get('token_timestamps') for m in request_metrics.values()) else t_generate_end
        total_duration = last_token_time - first_request_start_time

        if total_duration > 0:
             overall_throughput = total_output_tokens / total_duration
             print(f"Overall Throughput (total output tokens / effective duration): {overall_throughput:.2f} tokens/s")
        else:
             print("Overall Throughput: Not calculated (zero duration)")

    else:
        print("Overall ITL/TPOT/Throughput: Not calculated (no valid ITLs recorded).")

    print(f"Total Prompt Tokens Processed (decode node): {total_prompt_tokens_decode}") # Should be low if prefill worked
    print(f"Total Output Tokens Generated: {total_output_tokens}")

    # Send completion signal back to prefill node (run blocking call in thread)
    print("\nSending completion signal to prefill node...")
    try:
        success = await asyncio.to_thread(send_completion_signal, ip_address)
        if not success:
             print("Failed to send completion signal.")
    except Exception as e:
        print(f"Error sending completion signal via thread: {e}")

    print("Decode node script finished.")


# --- Main Execution Block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run vLLM in disaggregated prefill mode using AsyncLLMEngine")
    parser.add_argument("--mode", type=str, choices=["prefill", "decode"], required=True,
                        help="Run mode: 'prefill' or 'decode'")
    parser.add_argument("--ip", type=str, default="localhost",
                        help="IP address of the prefill node (used for KV transfer and signaling)")
    args = parser.parse_args()

    # Basic IP validation
    try:
        socket.gethostbyname(args.ip)
    except socket.gaierror:
         print(f"Error: Could not resolve hostname/IP: {args.ip}")
         sys.exit(1)

    if args.mode == "prefill":
        print(f"Starting async prefill node. Signal server IP: {args.ip}")
        try:
            asyncio.run(run_prefill(args.ip))
        except KeyboardInterrupt:
            print("Prefill node interrupted by user.")
        except Exception as e:
             print(f"Error running prefill node: {e}")
             os._exit(1) # Force exit on unhandled exception during run
    else:
        print(f"Starting async decode node. Connecting to prefill node at: {args.ip}")
        try:
            asyncio.run(run_decode(args.ip))
        except KeyboardInterrupt:
            print("Decode node interrupted by user.")
        except Exception as e:
            print(f"Error running decode node: {e}")
            os._exit(1) # Force exit