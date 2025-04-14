# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of disaggregated prefilling
We will launch 2 vllm instances (GPU 0 for prefill and GPU 1 for decode),
and then transfer the KV cache between them.
"""
import os
import time
from multiprocessing import Event, Process

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from datasets import load_dataset
import nvtx

# Load the dataset with streaming enabled
dataset = load_dataset("json",data_files="/home/hkngae/test/temp_dataset/output.json", split="train",)
dataset = list(dataset)
extracted_values = [record["conversations"][0]["value"] for record in dataset]
prompt_len=20
prompts = extracted_values[:prompt_len]



def run_prefill(prefill_done):
    # We use GPU 0 for prefill node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # The prefill node receives two requests, while the decode node receives
    # three requests. So the decode node will only receive the KV Cache for
    # requests 1 and 3. The decode node will use the KV Cache of requests 1
    # and 3 and do prefilling on request 2.
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the prefill node (kv_producer, rank 0).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2,"kv_ip":"localhost","kv_port":12345}'
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(#model="meta-llama/Llama-3.2-3B-Instruct",
              model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
              kv_transfer_config=ktc,
              max_model_len=700,
              gpu_memory_utilization=0.95,
              max_num_seqs=40,
              )
    nvtx.push_range("prefill", color="green")
    llm.generate(prompts, sampling_params)
    nvtx.pop_range()
    print("Prefill node is finished.")
    prefill_done.set()

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(prefill_done):
    # We use GPU 1 for decode node.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95,max_tokens=1000)

    # Using PyNcclConnector to transmit KV caches between vLLM instances.
    # This instance is the decode node (kv_consumer, rank 1).
    # The number of parallel instances for KV cache transfer is set to 2,
    # as required for PyNcclConnector.
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"PyNcclConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2,"kv_ip":"localhost","kv_port":12345}'
    )

    # Set GPU memory utilization to 0.8 for an A6000 GPU with 40GB
    # memory. You may need to adjust the value to fit your GPU.
    llm = LLM(#model="meta-llama/Llama-3.2-3B-Instruct",
              model="/home/hkngae/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659",
              kv_transfer_config=ktc,
              max_model_len=700,
              gpu_memory_utilization=0.95,
              max_num_seqs=40,
              disable_log_stats=False,)

    # Wait for the producer to start the pipe
    print("Waiting for prefill node to finish...")
    prefill_done.wait()

    # At this point when the prefill_done is set, the kv-cache should have been
    # transferred to this decode node, so we can start decoding.
    nvtx.push_range("decode", color="red")
    outputs = llm.generate(prompts, sampling_params)
    nvtx.pop_range()
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    prefill_done = Event()
    prefill_process = Process(target=run_prefill, args=(prefill_done, ))
    decode_process = Process(target=run_decode, args=(prefill_done, ))

    # Start prefill node
    prefill_process.start()

    # Start decode node
    decode_process.start()

    # Terminate the prefill node when decode is finished
    decode_process.join()
    prefill_process.terminate()
