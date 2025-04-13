import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
import deepspeed
from accelerate import init_empty_weights
from transformers import BitsAndBytesConfig

MODEL_PATH = "/models/deepseek-r1-chat"  # 改成你的實際路徑

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_skip_modules=None,
    llm_int8_enable_fp32_cpu_offload=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",  # DeepSpeed 在啟動時會覆蓋這個
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    trust_remote_code=True
)

streamer = TextStreamer(tokenizer)

print("Generating...")
inputs = tokenizer("Hello, what is DeepSeek?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50, streamer=streamer)
