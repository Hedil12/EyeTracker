import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Qwen model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map={"":"gpu"})
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Function to create the diagnostic prompt based on logs
def create_prompt(logs):
    log_text = '\n'.join([f"{row['timestamp']}: {row['eye movement']}: {row['eye dilation status']}" for _, row in logs.iterrows()])
    prompt = (
        "Analyze the following logs and provide possible diagnoses based on these rules:\n\n"
        "1. If the logs did not update for more than 5 seconds and the pupil/iris dilates or constricts, note it.\n"
        "2. If the logs show continuous eye movement (e.g., left, right, left) for 5 seconds or more, label it as rapid movement.\n"
        "3. If the logs show infrequent blinking or if the duration of each blink gets longer, note it as infrequent blinking.\n"
        "4. Possible diagnoses based on the above:\n"
        "   - Dilation indicates Heat Exhaustion\n"
        "   - Constriction indicates Heat Stroke\n"
        "   - Rapid eye movement indicates Heat Syncope\n"
        "   - Infrequent blinking can indicate any of the above\n\n"
        f"{log_text}"
    )
    return [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

# Function to analyze logs
def analyze_logs(logs_df):
    # Prepare input for the model
    log_entries = logs_df.apply(lambda row: f"Timestamp: {row['timestamp']}, Movement: {row['eye movement']}, Dilation: {row['eye dilation status']}.", axis=1)
    input_text = " ".join(log_entries)
    
    # Tokenize input
    inputs = tokenizer.encode(input_text, return_tensors="pt").to("gpu")  # Move to CPU

    # Generate response
    with torch.no_grad():
        output = model.generate(inputs, max_length=150)
    
    # Decode the generated response
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response
