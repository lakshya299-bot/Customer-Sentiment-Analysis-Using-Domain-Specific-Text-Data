import torch
import os
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType


INPUT_PATH = "/Users/lakshyasmac/Desktop/IMDB/models/roberta_imdb_cls" 
OUTPUT_DIR = "/Users/lakshyasmac/Desktop/IMDB/model_quantized_onnx"
ONNX_FILENAME = "roberta.onnx"
QUANT_FILENAME = "roberta.quant.onnx"

os.makedirs(OUTPUT_DIR, exist_ok=True)


print("⏳ Loading PyTorch model...")
tokenizer = RobertaTokenizer.from_pretrained(INPUT_PATH)
model = RobertaForSequenceClassification.from_pretrained(INPUT_PATH)
model.eval()


print("🔄 Exporting to ONNX...")
onnx_path = os.path.join(OUTPUT_DIR, ONNX_FILENAME)
dummy_input = tokenizer("This is a sample sentence", return_tensors="pt")

torch.onnx.export(
    model, 
    (dummy_input["input_ids"], dummy_input["attention_mask"]),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "logits": {0: "batch_size"}
    },
    opset_version=14
)


print("📉 Quantizing model to Int8...")
quantized_model_path = os.path.join(OUTPUT_DIR, QUANT_FILENAME)

quantize_dynamic(
    model_input=onnx_path,
    model_output=quantized_model_path,
    weight_type=QuantType.QUInt8 
)


tokenizer.save_pretrained(OUTPUT_DIR)


os.remove(onnx_path) 

print(f"✅ Success! Optimized model saved to: {OUTPUT_DIR}/{QUANT_FILENAME}")
print(f"👉 You can now deploy the '{OUTPUT_DIR}' folder.")