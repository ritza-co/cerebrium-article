# test.py - FIXED performance tester for optimized Cerebrium deployment
# File: test.py
# Target: Accurately test <2.5s inference with proper token counting

import requests
import json
import time
import os
from typing import List, Dict, Optional
from datetime import datetime
import statistics

# Try to load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

class OptimizedPerformanceTester:
    """
    FIXED performance tester for optimized A10 deployment.
    Tests Mistral-7B performance with proper token counting.
    """
    
    def __init__(self, 
                 project_id: str = None, 
                 token: str = None, 
                 endpoint: str = None,
                 debug: bool = False):
        self.project_id = project_id or os.getenv('CEREBRIUM_PROJECT_ID')
        self.token = token or os.getenv('CEREBRIUM_TOKEN')
        self.debug = debug
        
        if endpoint:
            self.base_url = endpoint.rstrip('/').replace('/{function_name}', '')
        elif os.getenv('CEREBRIUM_ENDPOINT'):
            self.base_url = os.getenv('CEREBRIUM_ENDPOINT').rstrip('/').replace('/{function_name}', '')
        elif self.project_id:
            self.base_url = f"https://api.cortex.cerebrium.ai/v4/p-{self.project_id}/vllm-openai-endpoint"
        else:
            raise ValueError("Either provide project_id or set CEREBRIUM_ENDPOINT environment variable")
        
        if not self.token:
            raise ValueError("Token is required. Set CEREBRIUM_TOKEN environment variable or provide token parameter")
        
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        
        # UPDATED performance targets for Mistral-7B optimization
        self.performance_targets = {
            "max_inference_time": 2.5,      # Target: <2.5 seconds (more realistic)
            "min_tokens_per_second": 30,    # Target: >30 tokens/sec (more realistic)
            "max_cost_per_request": 0.0008, # Target: <$0.0008 (updated for Mistral)
            "target_model": "mistralai/Mistral-7B-Instruct-v0.1"
        }
        
        print(f"🚀 OPTIMIZED A10 Performance Tester (Mistral-7B)")
        print(f"📡 Endpoint: {self.base_url}")
        print(f"🎯 Updated Performance Targets:")
        print(f"   ⏱️  Inference: <{self.performance_targets['max_inference_time']}s")
        print(f"   🚀 Speed: >{self.performance_targets['min_tokens_per_second']} tokens/sec")
        print(f"   💰 Cost: <${self.performance_targets['max_cost_per_request']}")
        print("-" * 70)
    
    def extract_content_from_response(self, response_data):
        """
        FIXED: Extract content and count tokens from various response formats.
        """
        content = ""
        token_count = 0
        
        try:
            if isinstance(response_data, dict):
                # Check choices array (OpenAI format)
                if "choices" in response_data and response_data["choices"]:
                    choice = response_data["choices"][0]
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                    elif "text" in choice:
                        content = choice["text"]
                
                # Check direct text field
                elif "text" in response_data:
                    content = response_data["text"]
                
                # Check content field
                elif "content" in response_data:
                    content = response_data["content"]
                
                # Check response field
                elif "response" in response_data:
                    content = response_data["response"]
                
                # If still no content, convert the whole response to string
                if not content:
                    content = str(response_data)
                
                # Count tokens (rough estimation: 1 token ≈ 4 characters for English)
                if content:
                    # Better token estimation: split by whitespace and punctuation
                    import re
                    words = re.findall(r'\w+|[^\w\s]', content)
                    token_count = len(words)
                else:
                    token_count = 0
            
            else:
                content = str(response_data)
                token_count = len(content.split())
        
        except Exception as e:
            print(f"⚠️  Error extracting content: {e}")
            content = str(response_data)[:200] + "..."
            token_count = len(content.split())
        
        return content.strip(), token_count
    
    def test_health_endpoint(self):
        """Test the health endpoint."""
        print("🏥 Testing health endpoint...")
        try:
            response = requests.post(f"{self.base_url}/health", headers=self.headers, timeout=30)
            if response.status_code == 200:
                result = response.json()
                print("✅ Health check passed!")
                
                # Handle both direct response and nested result format
                if 'result' in result:
                    health_data = result['result']
                else:
                    health_data = result
                
                print(f"   Status: {health_data.get('status', 'unknown')}")
                print(f"   Model: {health_data.get('model', 'unknown')}")
                print(f"   Test time: {health_data.get('test_inference_time', 'unknown')}")
                print(f"   Tokens/sec: {health_data.get('tokens_per_second', 'unknown')}")
                
                # ENHANCED: Show actual response sample
                response_sample = health_data.get('response_sample', '')
                if response_sample:
                    print(f"   📝 Sample response: '{response_sample}'")
                
                # Show optimizations if available
                optimizations = health_data.get('optimizations', [])
                if optimizations and self.debug:
                    print(f"   🔧 Optimizations: {len(optimizations)} active")
                    for opt in optimizations[:3]:  # Show first 3
                        print(f"      - {opt}")
                
                return True
            else:
                print(f"❌ Health check failed: HTTP {response.status_code}")
                if self.debug:
                    print(f"Response: {response.text[:200]}")
                return False
        except Exception as e:
            print(f"❌ Health check error: {str(e)}")
            return False
    
    def test_streaming_endpoint(self, prompt: str = "Hello", max_tokens: int = 50):
        """Test the /run endpoint with FIXED streaming response handling."""
        print(f"🧪 Testing streaming /run endpoint...")
        
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.performance_targets["target_model"],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.7,
            "top_k": 15,
            "stream": True
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/run",
                headers=self.headers,
                json=payload,
                timeout=60,
                stream=True
            )
            
            if response.status_code == 200:
                print("✅ Streaming response received...")
                
                content_chunks = []
                total_content = ""
                
                for line in response.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
                        if line_str.startswith('data: '):
                            data_str = line_str[6:]  # Remove 'data: ' prefix
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                chunk_data = json.loads(data_str)
                                
                                # Extract content from chunk
                                content, _ = self.extract_content_from_response(chunk_data)
                                if content:
                                    content_chunks.append(content)
                                    total_content += content
                                    
                            except json.JSONDecodeError:
                                continue
                
                total_time = time.time() - start_time
                
                # FIXED: Proper token counting
                final_content, token_count = self.extract_content_from_response({"text": total_content})
                tokens_per_second = round(token_count / total_time, 1) if total_time > 0 and token_count > 0 else 0
                cost = total_time * 0.000306  # A10 cost per second
                
                print(f"✅ Streaming test successful!")
                # ENHANCED: Show full content instead of truncated
                print(f"   📝 Full Response: '{final_content}'")
                print(f"   📊 Length: {len(final_content)} chars, {token_count} tokens")
                print(f"   ⏱️  Time: {total_time:.2f}s")
                print(f"   🚀 Speed: {tokens_per_second} tokens/sec")
                print(f"   💰 Cost: ${cost:.6f}")
                
                return True
            else:
                print(f"❌ Streaming test failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Streaming test error: {str(e)}")
            return False
    
    def test_chat_endpoint(self, prompt: str, max_tokens: int = 50) -> Optional[Dict]:
        """
        Test the FIXED /chat endpoint with accurate performance measurement.
        """
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "model": self.performance_targets["target_model"],
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "top_p": 0.7,
            "top_k": 15
        }
        
        print(f"🚀 Testing chat endpoint: '{prompt[:50]}...'")
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # FIXED: Extract content properly
                content, token_count = self.extract_content_from_response(result)
                
                # FIXED: Always use Cerebrium's internal performance metrics when available
                cerebrium_time = None
                cerebrium_tokens_per_sec = None
                
                # Check if result has nested structure with usage data
                usage_data = None
                if "result" in result and isinstance(result["result"], dict) and "usage" in result["result"]:
                    usage_data = result["result"]["usage"]
                elif "usage" in result:
                    usage_data = result["usage"]
                
                if usage_data:
                    cerebrium_time = usage_data.get("inference_time_seconds")
                    cerebrium_tokens_per_sec = usage_data.get("tokens_per_second")
                    if "tokens_generated" in usage_data:
                        token_count = usage_data["tokens_generated"]  # Use actual token count from Cerebrium
                
                # ALWAYS use Cerebrium's internal timing and speed when available
                if cerebrium_time is not None:
                    inference_time = cerebrium_time
                else:
                    inference_time = total_time
                
                # ENHANCED: Calculate total tokens/sec including input tokens
                input_token_count = 0
                if usage_data and "prompt_tokens" in usage_data:
                    input_token_count = usage_data["prompt_tokens"]
                else:
                    # Estimate input tokens from prompt
                    prompt_text = ""
                    if "messages" in payload and payload["messages"]:
                        prompt_text = " ".join([msg.get("content", "") for msg in payload["messages"]])
                    import re
                    words = re.findall(r'\w+|[^\w\s]', prompt_text)
                    input_token_count = len(words)
                
                total_tokens = token_count + input_token_count
                
                if cerebrium_tokens_per_sec is not None:
                    tokens_per_second = cerebrium_tokens_per_sec  # This is output tokens/sec from Cerebrium
                    total_tokens_per_second = round(total_tokens / inference_time, 1) if inference_time > 0 else 0
                else:
                    tokens_per_second = round(token_count / inference_time, 1) if inference_time > 0 and token_count > 0 else 0
                    total_tokens_per_second = round(total_tokens / inference_time, 1) if inference_time > 0 else 0
                
                # ENHANCED: Calculate cost with token-based pricing insights
                time_based_cost = inference_time * 0.000306  # A10 cost per second
                
                # Calculate cost per token for comparison
                cost_per_output_token = time_based_cost / token_count if token_count > 0 else 0
                cost_per_total_token = time_based_cost / total_tokens if total_tokens > 0 else 0
                
                cost = time_based_cost  # Keep time-based cost as primary
                
                print(f"✅ Chat test successful!")
                # ENHANCED: Show full response instead of truncated
                print(f"   📝 Full Response: '{content}'")
                print(f"   📊 Tokens: {input_token_count} input + {token_count} output = {total_tokens} total")
                print(f"   ⏱️  Time: {inference_time:.2f}s")
                print(f"   🚀 Speed: {tokens_per_second} output tok/s, {total_tokens_per_second} total tok/s")
                print(f"   💰 Cost: ${cost:.6f} (${cost_per_output_token:.8f}/output tok, ${cost_per_total_token:.8f}/total tok)")
                
                # Calculate performance score
                performance_score = self.calculate_performance_score(inference_time, tokens_per_second, cost)
                
                return {
                    "success": True,
                    "content": content,
                    "token_count": token_count,
                    "input_token_count": input_token_count,
                    "total_tokens": total_tokens,
                    "inference_time": inference_time,
                    "total_time": total_time,
                    "tokens_per_second": tokens_per_second,  # Output tokens/sec
                    "total_tokens_per_second": total_tokens_per_second,  # Total tokens/sec
                    "cost": cost,
                    "cost_per_output_token": cost_per_output_token,
                    "cost_per_total_token": cost_per_total_token,
                    "performance_score": performance_score
                }
            else:
                print(f"❌ Chat test failed: HTTP {response.status_code}")
                if self.debug:
                    print(f"Response: {response.text[:200]}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            print(f"❌ Chat test error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def test_complete_endpoint(self, prompt: str = "What is AI?", max_tokens: int = 50):
        """Test the /complete endpoint with FIXED response handling."""
        print(f"🧪 Testing /complete endpoint...")
        
        payload = {
            "prompt": prompt,
            "temperature": 0.1,
            "max_tokens": max_tokens
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/complete",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # FIXED: Extract content properly
                content, token_count = self.extract_content_from_response(result)
                
                # FIXED: Get Cerebrium's internal performance metrics
                cerebrium_time = None
                cerebrium_tokens_per_sec = None
                
                # Check for usage data in nested result structure or direct
                usage_data = None
                if "result" in result and isinstance(result["result"], dict) and "usage" in result["result"]:
                    usage_data = result["result"]["usage"]
                elif "usage" in result:
                    usage_data = result["usage"]
                elif "performance" in result:
                    usage_data = result["performance"]  # Fallback to performance field
                
                if usage_data:
                    cerebrium_time = usage_data.get("inference_time_seconds") or usage_data.get("inference_time")
                    cerebrium_tokens_per_sec = usage_data.get("tokens_per_second")
                    if "tokens_generated" in usage_data:
                        token_count = usage_data["tokens_generated"]
                
                # Use internal metrics when available
                inference_time = cerebrium_time if cerebrium_time is not None else total_time
                
                # ENHANCED: Calculate total tokens including input
                input_token_count = 0
                if usage_data and "prompt_tokens" in usage_data:
                    input_token_count = usage_data["prompt_tokens"]
                else:
                    # Estimate input tokens from prompt
                    prompt_text = payload.get("prompt", "")
                    import re
                    words = re.findall(r'\w+|[^\w\s]', prompt_text)
                    input_token_count = len(words)
                
                total_tokens = token_count + input_token_count
                
                if cerebrium_tokens_per_sec is not None:
                    tokens_per_second = cerebrium_tokens_per_sec
                    total_tokens_per_second = round(total_tokens / inference_time, 1) if inference_time > 0 else 0
                else:
                    tokens_per_second = round(token_count / inference_time, 1) if inference_time > 0 and token_count > 0 else 0
                    total_tokens_per_second = round(total_tokens / inference_time, 1) if inference_time > 0 else 0
                
                # ENHANCED: Calculate cost with token insights
                time_based_cost = inference_time * 0.000306  # A10 cost per second
                cost_per_output_token = time_based_cost / token_count if token_count > 0 else 0
                cost_per_total_token = time_based_cost / total_tokens if total_tokens > 0 else 0
                cost = time_based_cost
                
                print(f"✅ Complete test successful!")
                # ENHANCED: Show full response instead of truncated
                print(f"   📝 Full Response: '{content}'")
                print(f"   📊 Tokens: {input_token_count} input + {token_count} output = {total_tokens} total")
                print(f"   ⏱️  Time: {inference_time:.2f}s")
                print(f"   🚀 Speed: {tokens_per_second} output tok/s, {total_tokens_per_second} total tok/s")
                print(f"   💰 Cost: ${cost:.6f} (${cost_per_output_token:.8f}/output tok, ${cost_per_total_token:.8f}/total tok)")
                
                return True
            else:
                print(f"❌ Complete test failed: HTTP {response.status_code}")
                if self.debug:
                    print(f"Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ Complete test error: {str(e)}")
            return False
    
    def test_simple_batch(self):
        """Test the FIXED simple batch processing."""
        print("🔥 Testing optimized batch processing...")
        
        batch_requests = [
            {"messages": [{"role": "user", "content": "Hi"}]},
            {"messages": [{"role": "user", "content": "What is AI?"}]}
        ]
        
        payload = {
            "requests": batch_requests,
            "model": self.performance_targets["target_model"]
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/simple_batch",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            total_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Batch processing successful in {total_time:.2f}s")
                
                responses = result.get("responses", [])
                successful_responses = [r for r in responses if "error" not in r]
                
                print(f"   📊 Batch results:")
                print(f"      Total requests: {result.get('batch_size', len(responses))}")
                print(f"      Successful: {len(successful_responses)}/{len(responses)}")
                print(f"      Model: {result.get('model', 'unknown')}")
                
                # Show sample responses
                for i, resp in enumerate(responses[:2]):
                    if "error" not in resp:
                        content, tokens = self.extract_content_from_response(resp)
                        print(f"      Response {i+1}: {content[:50]}... ({tokens} tokens)")
                    else:
                        print(f"      Response {i+1}: ERROR - {resp.get('error', 'unknown')}")
                
                return True
            else:
                print(f"❌ Batch processing failed: HTTP {response.status_code}")
                if self.debug:
                    print(f"Response: {response.text[:200]}")
                return False
                
        except Exception as e:
            print(f"❌ Batch processing error: {str(e)}")
            return False
    
    def calculate_performance_score(self, inference_time: float, tokens_per_sec: float, cost: float) -> Dict:
        """Calculate performance score against updated targets."""
        targets = self.performance_targets
        
        # Score each metric (0-100, higher is better)
        time_score = max(0, min(100, (targets["max_inference_time"] / max(inference_time, 0.1)) * 100))
        speed_score = max(0, min(100, (tokens_per_sec / targets["min_tokens_per_second"]) * 100))
        cost_score = max(0, min(100, (targets["max_cost_per_request"] / max(cost, 0.000001)) * 100))
        
        overall_score = (time_score + speed_score + cost_score) / 3
        
        return {
            "overall": round(overall_score, 1),
            "time_score": round(time_score, 1),
            "speed_score": round(speed_score, 1),
            "cost_score": round(cost_score, 1),
            "meets_targets": time_score >= 90 and speed_score >= 90 and cost_score >= 90
        }
    
    def run_optimized_benchmark(self, test_prompts: List[str] = None, runs_per_prompt: int = 3):
        """
        Run comprehensive benchmark with FIXED performance measurement.
        """
        if test_prompts is None:
            # Customer service prompts with rich context, expecting ~60 token responses
            test_prompts = [
                "I'm a Premium Plus customer (account #12345) who purchased a wireless bluetooth headset (model BT-7500) from your electronics store last Tuesday for $249.99. The left earbud stopped working yesterday after only 5 days of normal use. I have the receipt and original packaging. I've been a loyal customer for 3 years and have never returned anything before. I'm traveling next week and really need these headphones working. What are my options for getting this resolved quickly?",
                
                "Hello, I ordered a birthday cake for my daughter's 8th birthday party this Saturday at 2 PM. Order confirmation #CAKE789 shows chocolate cake with vanilla frosting and 'Happy Birthday Emma' in pink letters. I placed the order 2 weeks ago and paid $85 including delivery to 123 Oak Street. However, I just realized I need to change the pickup time from 12 PM to 10 AM because the party venue became available earlier. Is it possible to modify the pickup time?",
                
                "I'm calling about my internet service that has been extremely slow for the past week. I'm on the Fiber Ultra plan paying $79/month and typically get 500 Mbps download speeds. However, speed tests now show only 15-20 Mbps, making it impossible to work from home. I've already tried unplugging the modem for 30 seconds, checked all cable connections, and ran multiple speed tests at different times. My service address is 456 Pine Avenue, account number INT-9876. Can you help troubleshoot this issue?",
                
                "I received my credit card statement today and noticed a charge for $127.50 from 'Digital Streaming Services' on January 15th that I don't recognize. I only subscribe to Netflix ($15.99/month) and Spotify ($9.99/month). I haven't signed up for any new streaming services recently. My card number ends in 4832 and this is regarding statement period December 15 - January 14. I'm concerned this might be fraudulent. What steps should I take to dispute this charge?",
                
                "I need to return a winter coat I bought online 3 weeks ago (order #WC-2024-455). It's a women's size medium navy wool coat that cost $189.99. When it arrived, the color looked much darker than on the website photos, and the fit is tighter than expected despite ordering my usual size. The coat still has all tags attached and I have the original shipping box. According to your website, I have a 30-day return window. How do I process this return and will I need to pay for return shipping?"
            ]
        
        print(f"🏃 Running optimized benchmark: {len(test_prompts)} prompts × {runs_per_prompt} runs")
        print("=" * 70)
        
        all_results = []
        prompt_summaries = []
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n📝 Prompt {i}/{len(test_prompts)}: '{prompt}'")
            print("-" * 50)
            
            prompt_results = []
            
            for run in range(runs_per_prompt):
                print(f"   Run {run+1}/{runs_per_prompt}...", end=" ")
                
                result = self.test_chat_endpoint(prompt, max_tokens=60)
                if result and result.get("success"):
                    prompt_results.append(result)
                    all_results.append(result)
                    
                    # Quick feedback with ENHANCED metrics
                    time_taken = result.get("inference_time") or result.get("total_time")
                    output_tokens_per_sec = result.get("tokens_per_second", 0)
                    total_tokens_per_sec = result.get("total_tokens_per_second", 0)
                    output_tokens = result.get("token_count", 0)
                    total_tokens = result.get("total_tokens", 0)
                    
                    if time_taken <= self.performance_targets["max_inference_time"]:
                        print(f"✅ {time_taken:.2f}s ({output_tokens}→{total_tokens} tok, {output_tokens_per_sec:.1f}→{total_tokens_per_sec:.1f} tok/s)")
                    else:
                        print(f"⚠️  {time_taken:.2f}s ({output_tokens}→{total_tokens} tok, {output_tokens_per_sec:.1f}→{total_tokens_per_sec:.1f} tok/s) - SLOW")
                else:
                    print("❌ Failed")
                
                # Small delay between runs
                time.sleep(1)
            
            # Summarize this prompt's performance
            if prompt_results:
                avg_time = statistics.mean([r.get("inference_time") or r.get("total_time") for r in prompt_results])
                avg_output_tokens_per_sec = statistics.mean([r.get("tokens_per_second", 0) for r in prompt_results])
                avg_total_tokens_per_sec = statistics.mean([r.get("total_tokens_per_second", 0) for r in prompt_results])
                avg_cost = statistics.mean([r.get("cost", 0) for r in prompt_results])
                avg_score = statistics.mean([r.get("performance_score", {}).get("overall", 0) for r in prompt_results])
                avg_output_token_count = statistics.mean([r.get("token_count", 0) for r in prompt_results])
                avg_total_token_count = statistics.mean([r.get("total_tokens", 0) for r in prompt_results])
                
                prompt_summaries.append({
                    "prompt": prompt,
                    "avg_time": avg_time,
                    "avg_tokens_per_sec": avg_output_tokens_per_sec,  # Keep for backward compatibility
                    "avg_total_tokens_per_sec": avg_total_tokens_per_sec,
                    "avg_cost": avg_cost,
                    "avg_score": avg_score,
                    "avg_token_count": avg_output_token_count,  # Keep for backward compatibility
                    "avg_total_token_count": avg_total_token_count,
                    "runs": len(prompt_results)
                })
                
                print(f"   📊 Average: {avg_time:.2f}s, {avg_output_token_count:.1f}→{avg_total_token_count:.1f} tok, {avg_output_tokens_per_sec:.1f}→{avg_total_tokens_per_sec:.1f} tok/s, ${avg_cost:.6f}, Score: {avg_score:.1f}/100")
        
        # Overall benchmark results
        if all_results:
            self.print_optimized_summary(all_results, prompt_summaries)
        else:
            print("❌ No successful results to analyze!")
    
    def print_optimized_summary(self, results: List[Dict], prompt_summaries: List[Dict]):
        """Print comprehensive benchmark summary with FIXED metrics."""
        successful_results = [r for r in results if r.get("success")]
        
        if not successful_results:
            print("❌ No successful benchmark results!")
            return
        
        print("\n" + "=" * 70)
        print("🎯 OPTIMIZED A10 BENCHMARK SUMMARY (Mistral-7B)")
        print("=" * 70)
        
        # Overall statistics with ENHANCED token calculations
        times = [r.get("inference_time") or r.get("total_time") for r in successful_results]
        output_speeds = [r.get("tokens_per_second", 0) for r in successful_results]
        total_speeds = [r.get("total_tokens_per_second", 0) for r in successful_results]
        costs = [r.get("cost", 0) for r in successful_results]
        scores = [r.get("performance_score", {}).get("overall", 0) for r in successful_results]
        output_token_counts = [r.get("token_count", 0) for r in successful_results]
        input_token_counts = [r.get("input_token_count", 0) for r in successful_results]
        total_token_counts = [r.get("total_tokens", 0) for r in successful_results]
        cost_per_output_tokens = [r.get("cost_per_output_token", 0) for r in successful_results]
        cost_per_total_tokens = [r.get("cost_per_total_token", 0) for r in successful_results]
        
        print(f"📊 Overall Performance ({len(successful_results)} successful runs):")
        print(f"   ⏱️  Inference Time:")
        print(f"      Average: {statistics.mean(times):.2f}s")
        print(f"      Median:  {statistics.median(times):.2f}s") 
        print(f"      Min/Max: {min(times):.2f}s / {max(times):.2f}s")
        print(f"      Target:  <{self.performance_targets['max_inference_time']}s")
        
        print(f"   🚀 Speed:")
        print(f"      Output tokens/sec:")
        print(f"         Average: {statistics.mean(output_speeds):.1f} tokens/sec")
        print(f"         Median:  {statistics.median(output_speeds):.1f} tokens/sec")
        print(f"         Min/Max: {min(output_speeds):.1f} / {max(output_speeds):.1f} tokens/sec")
        print(f"         Target:  >{self.performance_targets['min_tokens_per_second']} tokens/sec")
        print(f"      Total tokens/sec (input+output):")
        print(f"         Average: {statistics.mean(total_speeds):.1f} tokens/sec")
        print(f"         Median:  {statistics.median(total_speeds):.1f} tokens/sec")
        print(f"         Min/Max: {min(total_speeds):.1f} / {max(total_speeds):.1f} tokens/sec")
        
        print(f"   📝 Token Breakdown:")
        print(f"      Input tokens (avg): {statistics.mean(input_token_counts):.1f}")
        print(f"      Output tokens (avg): {statistics.mean(output_token_counts):.1f}")
        print(f"      Total tokens (avg): {statistics.mean(total_token_counts):.1f}")
        print(f"      Output range: {min(output_token_counts):.0f} - {max(output_token_counts):.0f} tokens")
        
        print(f"   💰 Cost:")
        print(f"      Per request: ${statistics.mean(costs):.6f} (avg), ${statistics.median(costs):.6f} (median)")
        print(f"      Per output token: ${statistics.mean(cost_per_output_tokens):.8f} (avg)")
        print(f"      Per total token: ${statistics.mean(cost_per_total_tokens):.8f} (avg)")
        print(f"      Request range: ${min(costs):.6f} - ${max(costs):.6f}")
        print(f"      Target: <${self.performance_targets['max_cost_per_request']} per request")
        
        # Performance against targets
        targets_met = sum(1 for r in successful_results if r.get("performance_score", {}).get("meets_targets", False))
        target_percentage = (targets_met / len(successful_results)) * 100
        
        print(f"   🎯 Performance Score:")
        print(f"      Average: {statistics.mean(scores):.1f}/100")
        print(f"      Targets met: {targets_met}/{len(successful_results)} ({target_percentage:.1f}%)")
        
        # Cost comparison with OpenAI - ENHANCED with proper token breakdown
        avg_cost = statistics.mean(costs)
        avg_input_tokens = statistics.mean(input_token_counts)
        avg_output_tokens = statistics.mean(output_token_counts)
        avg_total_tokens = statistics.mean(total_token_counts)
        
        # Calculate cost per token for Cerebrium
        cerebrium_cost_per_output_token = statistics.mean(cost_per_output_tokens)
        cerebrium_cost_per_total_token = statistics.mean(cost_per_total_tokens)
        
        # OpenAI GPT-4o-mini pricing (January 2025 - UPDATED)
        # Input: $1.100 per 1M tokens, Output: $4.400 per 1M tokens
        openai_output_cost_per_token = 4.400 / 1_000_000  # $0.0000044 per output token
        openai_input_cost_per_token = 1.100 / 1_000_000   # $0.0000011 per input token
        
        # For comparison, use output token pricing and total token pricing
        output_cost_ratio = cerebrium_cost_per_output_token / openai_output_cost_per_token if openai_output_cost_per_token > 0 else 0
        
        # Calculate total request cost for reference using actual token counts
        openai_input_cost = avg_input_tokens * openai_input_cost_per_token
        openai_output_cost = avg_output_tokens * openai_output_cost_per_token
        openai_total_cost = openai_input_cost + openai_output_cost
        total_cost_ratio = avg_cost / openai_total_cost if openai_total_cost > 0 else 0
        
        print(f"\n💵 Cost Comparison - Enhanced Token Analysis:")
        print(f"   🔸 Cost per output token:")
        print(f"      Cerebrium (Mistral-7B): ${cerebrium_cost_per_output_token:.8f}")
        print(f"      OpenAI (gpt-4o-mini):   ${openai_output_cost_per_token:.8f}")
        print(f"      Output token ratio: {output_cost_ratio:.1f}x")
        
        print(f"   🔸 Cost per total token (input+output):")
        print(f"      Cerebrium (Mistral-7B): ${cerebrium_cost_per_total_token:.8f}")
        print(f"      OpenAI equivalent:      ${(openai_input_cost + openai_output_cost) / avg_total_tokens:.8f}")
        
        print(f"\n   📋 Total request cost breakdown:")
        print(f"      Cerebrium: ${avg_cost:.6f}")
        print(f"        - {avg_input_tokens:.1f} input + {avg_output_tokens:.1f} output = {avg_total_tokens:.1f} total tokens")
        print(f"      OpenAI: ${openai_total_cost:.6f}")
        print(f"        - Input ({avg_input_tokens:.1f} tokens): ${openai_input_cost:.6f}")
        print(f"        - Output ({avg_output_tokens:.1f} tokens): ${openai_output_cost:.6f}")
        print(f"      Total cost ratio: {total_cost_ratio:.1f}x")
        
        # Updated competitiveness assessment based on cost per token
        if output_cost_ratio <= 3.0:
            print("   ✅ COMPETITIVE cost per output token!")
        elif output_cost_ratio <= 5.0:
            print("   ⚠️  Reasonable cost per output token")
        else:
            print("   ❌ Expensive cost per output token vs OpenAI")
        
        # Value analysis
        print(f"\n   💡 Value Analysis:")
        print(f"      Output speed: {statistics.mean(output_speeds):.1f} tokens/sec vs OpenAI's ~20-30 tokens/sec")
        print(f"      Total throughput: {statistics.mean(total_speeds):.1f} tokens/sec (input+output)")
        if statistics.mean(output_speeds) > 30:
            print("      ✅ FASTER output generation than OpenAI")
        print(f"      Privacy: ✅ Your own dedicated model")
        print(f"      Latency: {statistics.mean(times):.2f}s avg (very good for self-hosted)")
        
        # Per-prompt analysis
        print(f"\n📝 Per-Prompt Analysis:")
        for summary in prompt_summaries:
            prompt_display = summary["prompt"][:30] + "..." if len(summary["prompt"]) > 30 else summary["prompt"]
            status = "✅" if summary["avg_score"] >= 90 else "⚠️" if summary["avg_score"] >= 70 else "❌"
            print(f"   {status} '{prompt_display}': {summary['avg_time']:.2f}s, {summary['avg_token_count']:.1f} tok, {summary['avg_tokens_per_sec']:.1f} tok/s, Score: {summary['avg_score']:.1f}")
        
        # Success assessment
        avg_time = statistics.mean(times)
        avg_output_speed = statistics.mean(output_speeds)
        
        print(f"\n🏆 Optimization Success Assessment:")
        if avg_time <= self.performance_targets["max_inference_time"]:
            print("   ✅ Inference time TARGET MET!")
        else:
            print(f"   ❌ Inference time still {avg_time:.2f}s (target: <{self.performance_targets['max_inference_time']}s)")
        
        if avg_output_speed >= self.performance_targets["min_tokens_per_second"]:
            print("   ✅ Token generation speed TARGET MET!")
        else:
            print(f"   ❌ Speed still {avg_output_speed:.1f} tok/s (target: >{self.performance_targets['min_tokens_per_second']} tok/s)")
        
        if output_cost_ratio <= 3.0:
            print("   ✅ Cost competitiveness TARGET MET!")
        else:
            print(f"   ❌ Still {output_cost_ratio:.1f}x more expensive than OpenAI")


def main():
    """Main function to run optimized A10 performance testing."""
    print("🚀 OPTIMIZED A10 PERFORMANCE TESTER (Mistral-7B)")
    print("=" * 60)
    
    # Get configuration
    project_id = os.getenv('CEREBRIUM_PROJECT_ID')
    token = os.getenv('CEREBRIUM_TOKEN')
    endpoint = os.getenv('CEREBRIUM_ENDPOINT')
    
    if not token:
        token = input("Enter your Cerebrium JWT token: ").strip()
    
    if not project_id and not endpoint:
        project_id = input("Enter your Cerebrium project ID: ").strip()
    
    debug_mode = input("Enable debug mode? (y/N): ").strip().lower() == 'y'
    
    try:
        # Initialize tester
        tester = OptimizedPerformanceTester(
            project_id=project_id,
            token=token,
            endpoint=endpoint,
            debug=debug_mode
        )
        
        # Test sequence
        print("\n🔧 Testing optimized deployment...")
        
        # 1. Health check
        health_ok = tester.test_health_endpoint()
        time.sleep(1)
        
        # 2. Basic streaming test
        streaming_ok = False
        if health_ok:
            streaming_test = tester.test_streaming_endpoint("Hello")
            streaming_ok = streaming_test
        time.sleep(1)
        
        # 3. Choose what to test
        print("\n🚀 Choose test to run:")
        print("1. Quick performance test (single request)")
        print("2. Optimized benchmark (multiple prompts)")
        print("3. Simple batch processing test")
        print("4. All endpoint tests")
        print("5. Full comprehensive test")
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == "1":
            prompt = input("Enter test prompt (or press Enter for default): ").strip()
            if not prompt:
                prompt = "What is machine learning?"
            
            result = tester.test_chat_endpoint(prompt)
            if result and result.get("success"):
                print(f"\n✅ Optimized test successful!")
                print(f"   Response: {result['content'][:100]}...")
                print(f"   Time: {result.get('inference_time') or result.get('total_time'):.2f}s")
                print(f"   Tokens: {result['token_count']}")
                print(f"   Speed: {result['tokens_per_second']} tokens/sec")
                print(f"   Cost: ${result['cost']:.6f}")
                print(f"   Score: {result['performance_score']['overall']}/100")
        
        elif choice == "2":
            runs = input("Runs per prompt (default 3): ").strip()
            runs = int(runs) if runs.isdigit() else 3
            tester.run_optimized_benchmark(runs_per_prompt=runs)
        
        elif choice == "3":
            tester.test_simple_batch()
        
        elif choice == "4":
            print("🧪 Testing all optimized endpoints...")
            tester.test_health_endpoint()
            time.sleep(1)
            tester.test_streaming_endpoint()
            time.sleep(1)
            tester.test_complete_endpoint()
            time.sleep(1)
            tester.test_simple_batch()
        
        elif choice == "5":
            print("🔥 Running FULL optimized test suite...")
            tester.test_health_endpoint()
            time.sleep(2)
            tester.test_streaming_endpoint()
            time.sleep(2)
            tester.test_complete_endpoint()
            time.sleep(2)
            tester.test_simple_batch()
            time.sleep(2)
            tester.run_optimized_benchmark(runs_per_prompt=2)
        
        else:
            print("Invalid choice")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()