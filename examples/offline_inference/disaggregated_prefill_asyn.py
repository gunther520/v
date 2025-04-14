# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
using AsyncLLMEngine and asyncio.

It launches two async tasks simulating vLLM instances on different GPUs
(GPU 0 for prefill and GPU 1 for decode) within the same process,
and then transfers the KV cache between them using PyNcclConnector.
"""

#Asyn disaggregated prefill is not supported in vLLM yet.
import os
import time
import asyncio
import uuid  # For unique request IDs
from vllm.config import KVTransferConfig

# VLLM imports
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams
# KVTransferConfig is implicitly handled by passing the dict to AsyncEngineArgs

# Define Model and Base KVT Config
#MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct" # Or your local path
MODEL_NAME = "/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"


async def run_prefill(prefill_done_event: asyncio.Event):
    """Runs the prefill part asynchronously."""
    print("Starting Prefill Task...")
    # We use GPU 0 for prefill node.
    # Set this before initializing the engine.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story",
    ]
    # max_tokens=1 triggers prefill-only behavior
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Configure KVT for the producer role
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_ip":"localhost","kv_port":12345}'
    )

    # Setup AsyncEngineArgs
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.9, # Adjust per GPU
        # tensor_parallel_size=1, # Usually 1 for single-GPU prefill/decode nodes
    )

    # Initialize engine
    print("Prefill Task: Initializing Engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Prefill Task: Engine Initialized.")

    # Generate unique request IDs
    request_ids = [f"prefill-{uuid.uuid4()}" for _ in prompts]

    # Perform prefill generation (computes KV cache and sets up transfer)
    print("Prefill Task: Starting generation (KV Cache computation)...")
    results_generator = engine.generate(prompts, sampling_params, request_id="0")

    # We must iterate through the generator to ensure the requests are processed
    # by the engine, even though we only care about the side effect (KV transfer).
    try:
        async for result in results_generator:
             # Minimal logging to show progress
             if result.finished:
                  print(f"Prefill Task: Request {result.request_id[:8]}... processed.")
    except Exception as e:
         print(f"Prefill Task: Error during generation processing: {e}")
         # Decide how to handle errors - potentially signal failure
         return # Exit the task on error

    print("Prefill Task: Prefill generation processing complete.")

    # Signal that prefill computation and transfer setup is done
    prefill_done_event.set()
    print("Prefill Task: Signaled completion.")

    # The task will naturally exit after this. No infinite loop needed.
    # We keep it alive via the main task management if necessary.

async def run_decode(prefill_done_event: asyncio.Event):
    """Runs the decode part asynchronously."""
    print("Starting Decode Task...")
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    prompts = [
        "Hello, my name is",
        "Hi, your name is",
        "Tell me a very long story",
    ]
    # Set desired output length and enable streaming for async processing
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=64)

    # Configure KVT for the consumer role
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_ip":"localhost","kv_port":12345}'
    )

    # Setup AsyncEngineArgs
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.9, # Adjust per GPU
        # tensor_parallel_size=1,
    )

    # Initialize engine
    print("Decode Task: Initializing Engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    print("Decode Task: Engine Initialized.")

    # Wait for the prefill node to signal it's ready (KV cache transfer initiated)
    print("Decode Task: Waiting for prefill completion signal...")
    await prefill_done_event.wait()
    print("Decode Task: Received prefill completion signal.")
    print("Decode Task: KV cache transfer should be complete or in progress.")

    # Generate unique request IDs
    request_ids = [f"decode-{uuid.uuid4()}" for _ in prompts]

    # Start decoding generation
    print("Decode Task: Starting generation...")
    results_generator = engine.generate(prompts, sampling_params, request_id="0")

    # Process the streaming results
    final_outputs = {} # Store final outputs when finished
    try:
        async for request_output in results_generator:
            # Optional: Print token by token or intermediate results
            # current_text = request_output.outputs[0].text
            # print(f"Decode Task: Request {request_output.request_id[:8]}... partial output: {current_text!r}")

            if request_output.finished:
                print(f"Decode Task: Request {request_output.request_id[:8]}... finished.")
                final_outputs[request_output.request_id] = request_output
    except Exception as e:
         print(f"Decode Task: Error during generation processing: {e}")
         # Handle error appropriately
         return # Exit task

    # Print the final generated text after the stream is fully processed
    print("\n--- Final Decode Outputs ---")
    for req_id in request_ids: # Iterate in original order
         if req_id in final_outputs:
              output = final_outputs[req_id]
              prompt = output.prompt
              generated_text = output.outputs[0].text
              print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}\n---")
         else:
              print(f"Request ID {req_id}: No final output recorded (check for errors).")
    print("Decode Task: Finished.")


async def main():
    """Main function to create and manage async tasks."""
    prefill_done_event = asyncio.Event()

    # Create tasks for prefill and decode
    prefill_task = asyncio.create_task(run_prefill(prefill_done_event))
    # Add a small delay to ensure prefill starts listening before decode tries to connect?
    # Usually NCCL handles this, but sometimes helpful in scripts.
    # await asyncio.sleep(5) # Optional delay
    decode_task = asyncio.create_task(run_decode(prefill_done_event))


    try:
        await decode_task
        print("Main: Decode task completed.")
    except Exception as e:
        print(f"Main: Decode task failed with exception: {e}")

    # Once decode is done (successfully or not), the prefill task might still be running
    # if it encountered an issue after setting the event, or just waiting.
    # We can cancel it.
    if not prefill_task.done():
        print("Main: Cancelling prefill task...")
        prefill_task.cancel()
        try:
            await prefill_task # Wait for cancellation to be processed
        except asyncio.CancelledError:
            print("Main: Prefill task successfully cancelled.")
        except Exception as e:
            # Log if cancellation itself leads to an unexpected error in the task
            print(f"Main: Exception during prefill task cancellation processing: {e}")
    else:
         print("Main: Prefill task was already done.")

    print("Main: Script finished.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Main: Script interrupted by user.")
    except Exception as e:
         # Catch potential errors during asyncio.run() setup/teardown
         print(f"Main: An unexpected error occurred: {e}")