# SPDX-License-Identifier: Apache-2.0

import os
import time

from vllm import LLM, SamplingParams

# enable torch profiler, can also be set on cmd line
os.environ["VLLM_TORCH_PROFILER_DIR"] = "./vllm_profile"

# Sample prompts.
prompts = [
    "yo, my name is",
    "The flag of France is",
    "The food of China is",
    "The meaning of education is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, )

if __name__ == "__main__":
    os.environ["VLLM_USE_V1"] = "0"
    # Create an LLM.
    llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", tensor_parallel_size=1,max_model_len=2000,disable_log_stats=False,kv_transfer_config=None)


    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)


    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)
