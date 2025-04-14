import asyncio
import time
import uuid # For generating unique request IDs
import os
# VLLM imports
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm import SamplingParams

# --- Configuration ---
# Option 1: Use a readily available small model (if installed/cached)
# MODEL_NAME = "facebook/opt-125m"
# Option 2: Use the model from previous examples (ensure it's downloaded)
# MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = "/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659" # Or your specific local path

# Ensure this points to the correct location of your downloaded model
if not os.path.exists(MODEL_NAME):
     # Fallback if local path doesn't exist - try HF hub name
     print(f"Warning: Local path {MODEL_NAME} not found. Trying Hugging Face Hub name.")
     MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"


async def run_inference():
    """Initializes engine and runs a single async inference request."""
    print("--- Starting Minimal Async Inference Example ---")

    # 1. Define AsyncEngineArgs
    # Most basic args: model path and disabling Ray (for single process)
    # Add other args as needed (GPU util, max len, dtype etc.)
    engine_args = AsyncEngineArgs(
        model=MODEL_NAME,
        gpu_memory_utilization=0.90, # Adjust as needed
        max_model_len=2048, # Adjust based on model/needs
        # tensor_parallel_size=1, # Default is 1
    )
    print(f"Using model: {engine_args.model}")
    print("Initializing AsyncLLMEngine...")
    t_init_start = time.perf_counter()

    # 2. Initialize AsyncLLMEngine
    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
    except Exception as e:
        print(f"\n!!! Error initializing engine: {e}")
        print("Please ensure the model path is correct and dependencies are installed.")
        return

    t_init_end = time.perf_counter()
    print(f"Engine initialized in {t_init_end - t_init_start:.2f} seconds.")

    # 3. Define Prompt and SamplingParams
    prompt = "What is the capital of France?"
    # Must use_streaming=True for async iteration
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=100,
    )
    request_id = f"simple-req-{uuid.uuid4()}" # Unique ID for the request

    print(f"\nGenerating response for request ID: {request_id}")
    print(f"Prompt: {prompt!r}")

    # 4. Start Generation (get async generator)
    # NOTE: We use engine.generate for text generation, not engine.encode
    results_generator = engine.generate(prompt, sampling_params, request_id)

    # 5. Process Streamed Results
    t_gen_start = time.perf_counter()
    full_text = ""
    last_output_len = 0

    async for request_output in results_generator:
        # The request_output object contains the full state up to this point
        current_text = request_output.outputs[0].text
        new_text = current_text[last_output_len:] # Get only the newly generated part

        print(f"{new_text}", end="", flush=True) # Print the new part incrementally

        last_output_len = len(current_text) # Update the length tracker
        full_text = current_text # Store the full text

        # Check if generation is finished for this request
        if request_output.finished:
            t_gen_end = time.perf_counter()
            print("\n--- Generation Finished ---")
            break # Exit the loop once finished

    # 6. Final Output and Timing
    print(f"\n\nFinal Output for request {request_id}:")
    print(f"Prompt: {prompt!r}")
    print(f"Generated Text: {full_text!r}")
    print(f"Total generation time: {t_gen_end - t_gen_start:.2f} seconds")
    print("--------------------------------------------------")

    # Optional: Cleanly shutdown engine background loop if needed,
    # although asyncio.run() usually handles this for simple scripts.
    # Consider engine.shutdown() or similar if managing engine lifetime explicitly.


# --- Main Execution ---
if __name__ == "__main__":
    try:
        # Check CUDA availability early
        import torch
        if not torch.cuda.is_available():
             print("!!! Error: CUDA is not available. vLLM requires a GPU.")
        else:
             print(f"Found {torch.cuda.device_count()} CUDA devices.")
             # Explicitly set device if needed, although AsyncEngine usually handles it
             # torch.cuda.set_device(0)
             print(f"Running on device: {torch.cuda.get_device_name(torch.cuda.current_device())}")

             # Run the async function
             asyncio.run(run_inference())

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()