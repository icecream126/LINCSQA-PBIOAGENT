CUDA_VISIBLE_DEVICES=0 vllm serve deepseek-ai/DeepSeek-R1-Distill-Llama-8B --port 8000 --max-model-len 32768 --reasoning-parser deepseek_r1 --enable-chunked-prefill --gpu-memory-utilization 0.9 --max-num-seqs 128 

CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-30B-A3B-Instruct-2507 --port 8001 --enable_expert_parallel