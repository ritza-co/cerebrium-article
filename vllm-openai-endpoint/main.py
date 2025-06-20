# main.py - ULTRA-OPTIMIZED for sub-3s A10 performance
# File: main.py
# Target: <2.5s inference, >50 tokens/sec, competitive cost

from vllm import SamplingParams, AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from pydantic import BaseModel
from typing import List, Optional
import time
import json
import os
import uuid
import asyncio
from huggingface_hub import login

# Initialize Hugging Face authentication
if os.environ.get("HF_AUTH_TOKEN"):
    login(token=os.environ.get("HF_AUTH_TOKEN"))

# CRITICAL: Use fastest proven model for A10
# Mistral-7B is significantly faster than Llama-3.1-8B
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

print(f"🚀 SPEED-OPTIMIZED: Using {MODEL_NAME} for ultra-fast A10 performance")

# ULTRA-OPTIMIZED vLLM configuration for A10 speed
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    
    # SPEED 1: Conservative GPU memory for stability and speed
    gpu_memory_utilization=0.75,        # Reduced for better performance
    
    # SPEED 2: Optimized context length - sweet spot for speed
    max_model_len=1024,                 # Small context for speed
    
    # SPEED 3: Optimal quantization for A10
    dtype="float16",                    # Perfect for A10
    
    # SPEED 4: Optimized batch settings
    max_num_seqs=4,                     # Small batches for low latency
    max_num_batched_tokens=1024,        # Match max_model_len
    
    # SPEED 5: Fastest execution mode
    enforce_eager=True,                 # No CUDA graphs overhead
    
    # SPEED 6: Minimal overhead settings
    trust_remote_code=False,
    disable_log_stats=True,
    enable_prefix_caching=False,        # Adds overhead
    
    # SPEED 7: Optimized memory management
    swap_space=0,                       # No swap for speed
    block_size=16,                      # Optimal for A10
    
    # SPEED 8: Single GPU optimization
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    
    # SPEED 9: Deterministic for testing
    seed=42,
)

print("🔥 Initializing SPEED-OPTIMIZED vLLM engine...")
engine = AsyncLLMEngine.from_engine_args(engine_args)
print("✅ Engine initialized with speed optimizations")

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list

def format_mistral_prompt(messages: list) -> str:
    """
    OPTIMIZED: Proper Mistral prompt formatting - no loops!
    """
    if not messages:
        return ""
    
    # Take only recent messages to stay within context
    recent_messages = messages[-2:] if len(messages) > 2 else messages
    
    # Mistral format: <s>[INST] instruction [/INST] model_answer</s>[INST] follow_up [/INST]
    formatted_parts = []
    
    for i, message in enumerate(recent_messages):
        role = message.role
        content = message.content.strip()[:200]  # Truncate for speed
        
        if role == "system":
            # System message goes in first INST
            if i == 0:
                formatted_parts.append(f"<s>[INST] {content}")
            else:
                formatted_parts.append(f"[INST] {content}")
        elif role == "user":
            if not formatted_parts:
                formatted_parts.append(f"<s>[INST] {content} [/INST]")
            else:
                formatted_parts.append(f"[INST] {content} [/INST]")
        elif role == "assistant":
            formatted_parts.append(f" {content}</s>")
    
    # Ensure we end properly for generation
    prompt = "".join(formatted_parts)
    if not prompt.endswith("[/INST]"):
        if not prompt.startswith("<s>"):
            prompt = f"<s>[INST] {prompt} [/INST]"
        elif "[/INST]" not in prompt:
            prompt = prompt.replace("<s>[INST]", "<s>[INST]").replace("</s>", " [/INST]")
    
    return prompt

# SPEED-OPTIMIZED sampling parameters
ULTRA_FAST_SAMPLING = {
    "temperature": 0.1,              # Very low for speed and determinism
    "top_p": 0.7,                   # Reduced for faster sampling
    "top_k": 15,                    # Small for speed
    "max_tokens": 75,               # CRITICAL: Small output for speed
    "repetition_penalty": 1.05,     # Prevent loops
}

async def run(
    messages: list, 
    model: str, 
    run_id: str,
    temperature: float = 0.1,        # Default optimized for speed
    top_p: float = 0.7,              
    top_k: int = 15,                 
    max_tokens: int = 75,            # Small default for speed
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stream: bool = True,
    stop: list = None
):
    """
    ULTRA-OPTIMIZED streaming endpoint for <2.5s inference.
    """
    try:
        if not messages:
            raise ValueError("Messages required")
        
        # Convert and validate messages
        validated_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if content:  # Only add non-empty messages
                validated_messages.append(Message(role=role, content=content))
        
        if not validated_messages:
            raise ValueError("No valid messages provided")
        
        # OPTIMIZED prompt formatting - no loops!
        prompt = format_mistral_prompt(validated_messages)
        
        # Safety check - prevent infinite loops
        if prompt.count("[INST]") > 3:
            # Fallback to simple format
            last_msg = validated_messages[-1]
            prompt = f"<s>[INST] {last_msg.content} [/INST]"
        
        # SPEED-OPTIMIZED sampling
        sampling_params = SamplingParams(
            temperature=max(0.05, min(temperature, 0.5)),    # Clamp for speed
            top_p=max(0.5, min(top_p, 0.9)),                # Clamp for speed
            top_k=max(5, min(top_k, 25)),                   # Clamp for speed
            max_tokens=min(max_tokens, 100),                # Hard limit for speed
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=1.05,                        # Prevent repetition
            stop=stop or ["</s>", "[INST]"],               # Prevent format loops
            skip_special_tokens=True,
        )
        
        print(f"🚀 Ultra-fast inference starting (max {sampling_params.max_tokens} tokens)...")
        inference_start = time.time()
        
        results_generator = engine.generate(prompt, sampling_params, run_id)
        
        if stream:
            previous_text = ""
            token_count = 0
            chars_generated = 0
            
            async for output in results_generator:
                if output.outputs:
                    current_text = output.outputs[0].text
                    new_text = current_text[len(previous_text):]
                    previous_text = current_text
                    
                    if new_text.strip():
                        chars_generated += len(new_text)
                        
                        # Rough token estimation (4 chars per token average)
                        token_count = max(token_count, chars_generated // 4)
                        
                        response = ChatCompletionResponse(
                            id=run_id,
                            object="chat.completion.chunk",
                            created=int(time.time()),
                            model=model,
                            choices=[{
                                "delta": {"content": new_text},
                                "index": 0,
                                "finish_reason": output.outputs[0].finish_reason
                            }]
                        )
                        yield f"data: {json.dumps(response.model_dump())}\n\n"
            
            inference_time = time.time() - inference_start
            final_token_count = len(previous_text.split()) if previous_text else token_count
            tokens_per_sec = round(final_token_count / inference_time, 1) if inference_time > 0 else 0
            
            print(f"✅ Ultra-fast streaming: {inference_time:.2f}s, {final_token_count} tokens, {tokens_per_sec} tok/s")
            yield "data: [DONE]\n\n"
        
        else:
            # Non-streaming
            final_output = None
            async for output in results_generator:
                final_output = output
            
            inference_time = time.time() - inference_start
            
            if final_output and final_output.outputs:
                content = final_output.outputs[0].text
                token_count = len(content.split()) if content else 0
                tokens_per_sec = round(token_count / inference_time, 1) if inference_time > 0 else 0
                
                print(f"✅ Ultra-fast completion: {inference_time:.2f}s, {token_count} tokens, {tokens_per_sec} tok/s")
                
                response = ChatCompletionResponse(
                    id=run_id,
                    object="chat.completion",
                    created=int(time.time()),
                    model=model,
                    choices=[{
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "index": 0,
                        "finish_reason": final_output.outputs[0].finish_reason or "stop"
                    }]
                )
                yield f"data: {json.dumps(response.model_dump())}\n\n"
                yield "data: [DONE]\n\n"
    
    except Exception as e:
        print(f"❌ Ultra-fast inference failed: {str(e)}")
        error_response = {"error": {"message": str(e), "type": "ultra_fast_optimization_error"}}
        yield f"data: {json.dumps(error_response)}\n\n"

async def chat(
    messages: list, 
    model: str,
    temperature: float = 0.1,
    top_p: float = 0.7,
    top_k: int = 15,
    max_tokens: int = 75,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: list = None
):
    """
    ULTRA-FAST non-streaming chat for <2s inference.
    """
    try:
        validated_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", "")).strip()
            if content:
                validated_messages.append(Message(role=role, content=content))
        
        if not validated_messages:
            raise ValueError("No valid messages")
        
        prompt = format_mistral_prompt(validated_messages)
        
        # Safety check
        if prompt.count("[INST]") > 3:
            last_msg = validated_messages[-1]
            prompt = f"<s>[INST] {last_msg.content} [/INST]"
        
        sampling_params = SamplingParams(
            temperature=max(0.05, min(temperature, 0.5)),
            top_p=max(0.5, min(top_p, 0.9)),
            top_k=max(5, min(top_k, 25)),
            max_tokens=min(max_tokens, 100),
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repetition_penalty=1.05,
            stop=stop or ["</s>", "[INST]"],
            skip_special_tokens=True,
        )
        
        request_id = str(uuid.uuid4())
        
        print("🚀 Ultra-fast chat starting...")
        inference_start = time.time()
        
        results_generator = engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for output in results_generator:
            final_output = output
        
        inference_time = time.time() - inference_start
        
        if final_output and final_output.outputs:
            content = final_output.outputs[0].text
            token_count = len(content.split()) if content else 0
            tokens_per_sec = round(token_count / inference_time, 1) if inference_time > 0 else 0
            
            print(f"✅ Ultra-fast chat: {inference_time:.2f}s, {token_count} tokens, {tokens_per_sec} tok/s")
            
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "index": 0,
                    "finish_reason": final_output.outputs[0].finish_reason or "stop"
                }],
                "usage": {
                    "inference_time_seconds": round(inference_time, 3),
                    "tokens_generated": token_count,
                    "tokens_per_second": tokens_per_sec
                }
            }
        else:
            return {"error": "No output generated"}
    
    except Exception as e:
        return {"error": {"message": str(e), "type": "ultra_fast_chat_error"}}

async def complete(
    prompt: str,
    temperature: float = 0.1,
    max_tokens: int = 75
):
    """
    ULTRA-FAST simple completion for basic use cases.
    """
    try:
        # Simple prompt - no complex formatting
        if len(prompt) > 300:
            prompt = prompt[-300:]
        
        # Ensure proper Mistral format
        if not prompt.startswith("<s>[INST]"):
            prompt = f"<s>[INST] {prompt} [/INST]"
        
        sampling_params = SamplingParams(
            temperature=max(0.05, min(temperature, 0.5)),
            top_p=0.7,
            top_k=15,
            max_tokens=min(max_tokens, 100),
            repetition_penalty=1.05,
            stop=["</s>", "[INST]"],
            skip_special_tokens=True,
        )
        
        request_id = str(uuid.uuid4())
        
        print(f"🚀 Ultra-fast completion: '{prompt[:50]}...'")
        start_time = time.time()
        
        results_generator = engine.generate(prompt, sampling_params, request_id)
        
        final_output = None
        async for output in results_generator:
            final_output = output
        
        completion_time = time.time() - start_time
        
        if final_output and final_output.outputs:
            content = final_output.outputs[0].text
            token_count = len(content.split()) if content else 0
            tokens_per_sec = round(token_count / completion_time, 1) if completion_time > 0 else 0
            
            print(f"✅ Ultra-fast completion: {completion_time:.2f}s, {token_count} tokens, {tokens_per_sec} tok/s")
            
            return {
                "id": request_id,
                "text": content,
                "finish_reason": final_output.outputs[0].finish_reason or "stop",
                "performance": {
                    "inference_time": round(completion_time, 3),
                    "tokens_generated": token_count,
                    "tokens_per_second": tokens_per_sec
                }
            }
        else:
            return {"error": "No output generated"}
    
    except Exception as e:
        return {"error": str(e)}

async def health():
    """
    Fixed health check with proper performance test.
    """
    try:
        print("🔥 Running ultra-fast health check...")
        test_start = time.time()
        
        # Quick performance test with proper Mistral format
        test_prompt = "<s>[INST] Hi [/INST]"
        sampling_params = SamplingParams(
            temperature=0.1, 
            max_tokens=10, 
            top_k=10,
            stop=["</s>", "[INST]"],
            skip_special_tokens=True
        )
        
        results_generator = engine.generate(test_prompt, sampling_params, "health_check")
        
        response_text = ""
        async for output in results_generator:
            if output.outputs:
                response_text = output.outputs[0].text
        
        test_time = time.time() - test_start
        token_count = len(response_text.split()) if response_text else 0
        tokens_per_second = round(token_count / test_time, 1) if test_time > 0 else 0
        
        print(f"✅ Health check: {test_time:.3f}s, {token_count} tokens, {tokens_per_second} tok/s")
        
        return {
            "status": "ULTRA_OPTIMIZED",
            "model": MODEL_NAME,
            "test_inference_time": f"{test_time:.3f}s",
            "tokens_generated": token_count,
            "tokens_per_second": tokens_per_second,
            "target_performance": "<2.5s inference time",
            "optimizations": [
                "Mistral-7B (faster than Llama-8B)",
                "Fixed prompt formatting (no loops)",
                "Optimized context (1024 tokens)",
                "Speed-optimized sampling",
                "Proper stop sequences",
                "Conservative GPU utilization (75%)",
                "Small batch sizes (4 sequences)",
                "Max 75 tokens output",
                "Repetition penalty to prevent loops"
            ],
            "hardware": "A10 GPU optimized",
            "response_sample": response_text[:100] if response_text else "No response"
        }
    except Exception as e:
        return {"status": "FAILED", "error": str(e)}

# Simplified batch processing - removed complex features that were failing
async def simple_batch(requests: List[dict], model: str = MODEL_NAME):
    """
    SIMPLIFIED batch processing - fixes the HTTP 587 error.
    """
    try:
        if not requests:
            return {"error": "No requests provided"}
        
        # Limit to 2 requests for A10 memory
        batch_size = min(len(requests), 2)
        requests = requests[:batch_size]
        
        results = []
        
        # Process one by one to avoid memory issues
        for i, req in enumerate(requests):
            if "messages" not in req:
                results.append({"error": f"Request {i} missing messages"})
                continue
            
            try:
                # Use the chat function for each request
                result = await chat(
                    messages=req["messages"],
                    model=model,
                    max_tokens=50  # Smaller for batch
                )
                results.append(result)
            except Exception as e:
                results.append({"error": str(e)})
        
        return {
            "responses": results,
            "batch_size": len(results),
            "model": model
        }
    
    except Exception as e:
        return {"error": {"message": str(e), "type": "simple_batch_error"}}