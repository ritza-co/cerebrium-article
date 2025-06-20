# README.md

# Optimized vLLM OpenAI-Compatible Endpoint

This project deploys a **high-performance, cost-effective** OpenAI-compatible API endpoint using vLLM on Cerebrium's serverless infrastructure. Optimized for **sub-2.5s inference times** and **50+ tokens/sec** throughput on A10 GPUs.

## 🎯 Performance Targets

- **Inference Time**: < 2.5 seconds
- **Throughput**: > 50 tokens/second  
- **Cost**: < $0.0008 per request

## 📋 Prerequisites

- Python 3.8+
- Cerebrium account ([sign up here](https://www.cerebrium.ai))
- Hugging Face account ([sign up here](https://huggingface.co))

## 🔧 Installation & Setup

### 1. Install Cerebrium CLI

```bash
pip install cerebrium
```

### 2. Login to Cerebrium

```bash
cerebrium login
```

Follow the prompts to authenticate with your Cerebrium account.

### 3. Clone/Download Project Files

Ensure you have these files in your project directory:
- `main.py` - Main application code
- `cerebrium.toml` - Deployment configuration  
- `test.py` - Performance testing script
- `.env` - Environment variables (create this)

### 4. Configure Environment Variables

Create a `.env` file in your project root:

```bash
# .env
# Required: Hugging Face token for model access
HF_AUTH_TOKEN=hf_your_token_here

# Optional: For testing
CEREBRIUM_PROJECT_ID=your_project_id
CEREBRIUM_TOKEN=your_jwt_token
CEREBRIUM_ENDPOINT=https://api.cortex.cerebrium.ai/v4/p-PROJECT_ID/vllm-openai-endpoint
```

**Where to find these values:**

- **HF_AUTH_TOKEN**: 
  1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
  2. Create a new token with "Read" permissions
  3. Copy the token (starts with `hf_`)

- **CEREBRIUM_PROJECT_ID**: Found in your Cerebrium dashboard URL after deployment

- **CEREBRIUM_TOKEN**: 
  1. Go to your Cerebrium dashboard
  2. Navigate to "API Keys" section
  3. Copy your JWT token

- **CEREBRIUM_ENDPOINT**: Generated after deployment (optional, for testing)

### 5. Add Secrets to Cerebrium Dashboard

1. Go to your [Cerebrium Dashboard](https://dashboard.cerebrium.ai)
2. Navigate to your project → "Secrets"
3. Add: `HF_AUTH_TOKEN` with your Hugging Face token value

## 🚀 Deployment

Deploy the optimized endpoint:

```bash
cerebrium deploy
```

This will:
- Build the Docker container with all dependencies
- Deploy to A10 GPU instances
- Return your API endpoint URL
- Run initial health checks

**Expected deployment time**: 3-5 minutes

## 📁 Project Structure

### `main.py` - Ultra-Optimized vLLM Implementation

The main application file containing the optimized vLLM server with multiple endpoints.

**Key Optimizations:**

- **Model Selection**: Uses Mistral-7B (faster) or Llama-3.1-8B for optimal A10 performance
- **Memory Management**: Conservative 75% GPU utilization for stability
- **Context Optimization**: 1024 token context length for speed
- **Batch Processing**: Small batches (4 sequences) for low latency  
- **Prompt Formatting**: Fixed Mistral prompt formatting to prevent loops
- **Sampling**: Speed-optimized parameters (low temperature, reduced top_k/top_p)
- **Stop Sequences**: Proper stop tokens to prevent infinite generation

**Available Endpoints:**

- `/run` - Streaming chat completions (OpenAI-compatible)
- `/chat` - Non-streaming chat completions  
- `/complete` - Simple text completion
- `/health` - Health check with performance metrics
- `/simple_batch` - Batch processing (up to 2 requests)

**Performance Features:**

- Async streaming with proper token counting
- Sub-2.5s inference targeting
- Comprehensive error handling
- Real-time performance metrics

### `cerebrium.toml` - Deployment Configuration

Contains all deployment settings and hardware specifications.

**Key Optimizations:**

**Hardware Configuration:**
```toml
[cerebrium.hardware]
compute = "AMPERE_A10"        # Optimal price/performance GPU
cpu = 4                       # Balanced CPU allocation
memory = 20.0                 # Sufficient for Mistral-7B + overhead  
gpu_count = 1                # Single GPU optimization
```

**Scaling Settings:**
```toml
[cerebrium.scaling]
min_replicas = 1              # CRITICAL: Eliminates cold starts
max_replicas = 3              # Conservative scaling
cooldown = 30                 # Balanced scaling response
```

**Dependencies:**
- **Stable versions**: Pinned ranges for reliability
- **Essential only**: Minimal dependencies for faster builds
- **Optimized stack**: vLLM 0.4.0-0.6.0, PyTorch 2.1.0-2.4.0

**Build Optimizations:**
- CUDA 12.1.1 base image for A10 compatibility
- Disabled animations and confirmations for speed
- Optimized include/exclude patterns

### `test.py` - Performance Testing & Validation

Comprehensive testing script to validate deployment performance and measure against targets.

**Features:**

- **Health Check**: Validates deployment status and basic functionality
- **Endpoint Testing**: Tests all available endpoints (`/run`, `/chat`, `/complete`, `/batch`)  
- **Performance Benchmarking**: Multi-prompt performance analysis
- **Token Counting**: Accurate token measurement and speed calculation
- **Cost Analysis**: Real-time cost calculation and OpenAI comparison
- **Target Validation**: Scores performance against optimization targets

**Usage:**

```bash
# Basic performance test
python test.py

# With environment variables set
export CEREBRIUM_PROJECT_ID=your_project_id
export CEREBRIUM_TOKEN=your_token
python test.py
```

**Test Options:**
1. Quick performance test (single request)
2. Optimized benchmark (multiple prompts × multiple runs)  
3. Simple batch processing test
4. All endpoint tests
5. Full comprehensive test suite

**Performance Metrics:**
- Inference time (target: <2.5s)
- Tokens per second (target: >50)
- Cost per request (target: <$0.0008)  
- Success rate and reliability
- Comparison with OpenAI pricing

## 🔍 Usage Examples

### Using with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.cortex.cerebrium.ai/v4/p-YOUR_PROJECT_ID/vllm-openai-endpoint/run",
    api_key="YOUR_CEREBRIUM_JWT_TOKEN",
)

# Streaming chat completion
response = client.chat.completions.create(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
    max_tokens=100,
    temperature=0.1,
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Direct API Usage

```bash
curl -X POST "https://api.cortex.cerebrium.ai/v4/p-YOUR_PROJECT_ID/vllm-openai-endpoint/chat" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain quantum computing briefly"}
    ],
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "max_tokens": 100,
    "temperature": 0.1
  }'
```

## 🎛️ Configuration Options

### Model Selection

The deployment supports Llama, Mistral has vllm compatibility issues:

- **Llama-3.1-8B-Instruct** (default) - Higher quality, slightly slower
- **Mistral-7B-Instruct-v0.1** - Fastest, most cost-effective

Change in `main.py`:
```python
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Quality
# MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.1"  # Fast
```

### Performance Tuning

Adjust in `main.py` for different speed/quality tradeoffs:

```python
# Ultra-fast (current settings)
ULTRA_FAST_SAMPLING = {
    "temperature": 0.1,
    "top_p": 0.7, 
    "top_k": 15,
    "max_tokens": 75,
}

# Balanced quality/speed
BALANCED_SAMPLING = {
    "temperature": 0.3,
    "top_p": 0.9,
    "top_k": 40, 
    "max_tokens": 150,
}
```

## 💰 Cost Optimization

Current optimized costs:

- **A10 GPU**: $0.000306/second
- **Typical request**: 1.5-2.5s = $0.0005-0.0008
- **vs OpenAI gpt-4o-mini**: ~1.5-2x cost

**Cost reduction strategies:**
1. Use shorter `max_tokens` (75 default)
2. Enable `min_replicas=0` if cold starts acceptable  
3. Use Mistral-7B vs Llama-3.1-8B
4. Batch multiple requests when possible
