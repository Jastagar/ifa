from transformers import AutoTokenizer, AutoModelForCausalLM
from snac import SNAC
import torch
import soundfile as sf


MODEL_NAME = "Indus-Labs/hinglish_tts_v1"

# Load models
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)

snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
snac_model = snac_model.to("cuda")

# Text
prompt = "Hello doston, main aapka dost hun"

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt")

# IMPORTANT: move inputs to model device
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# Generate
with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1200,
    )

print("Generated tokens:", outputs.shape)

# For now, just inspect the generated output
print(outputs)