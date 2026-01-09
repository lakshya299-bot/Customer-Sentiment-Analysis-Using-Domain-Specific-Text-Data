import streamlit as st
import onnxruntime as ort
import numpy as np
import os
import re
from transformers import RobertaTokenizer
from scipy.special import softmax 


MODEL_DIR = "model_quantized_onnx"

ONNX_FILENAME = "roberta.quant.onnx"

st.set_page_config(
    page_title="IMDB Sentiment (ONNX)", 
    page_icon="⚡",
    layout="centered"
)




def preprocess_text(text):
    text = text.strip()
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


@st.cache_resource
def load_pipeline():
 
    if not os.path.exists(MODEL_DIR):
        st.error(f"❌ Directory not found: {MODEL_DIR}")
        return None, None
        
    try:
  
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_DIR)
        
        
        model_path = os.path.join(MODEL_DIR, ONNX_FILENAME)
        
        
        if not os.path.exists(model_path):
            st.error(f"❌ ONNX file not found at: {model_path}")
            return None, None

        
        session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        
        return tokenizer, session
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

tokenizer, session = load_pipeline()




def predict_onnx(text):
    
    inputs = tokenizer(
        text,
        return_tensors="np", 
        truncation=True,
        padding="max_length",
        max_length=256
    )

    
    
    ort_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }

    
    
    logits = session.run(None, ort_inputs)[0]

    
    
    probs = softmax(logits[0]) 
    return probs




st.title("⚡ IMDB Classifier (ONNX Optimized)")
st.markdown("Running via **Quantized ONNX**. This is optimized for speed and low memory usage.")


user_input = st.text_area("Enter your review:", height=150, placeholder="Example: The cinematography was breathtaking, but the script felt weak.")

if st.button("Analyze Sentiment", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    elif tokenizer is None or session is None:
        st.error("Model is not loaded.")
    else:
        with st.spinner("Processing..."):
            
            clean_text = preprocess_text(user_input)

            
            probs = predict_onnx(clean_text)
            
            
            
            pred_idx = np.argmax(probs)
            confidence = probs[pred_idx]
            
            
            st.divider()
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if pred_idx == 1:
                    st.success("## Positive 😃")
                else:
                    st.error("## Negative 😡")
            
            with col2:
                st.write(f"**Confidence Score:** {confidence:.2%}")
                if pred_idx == 1:
                    st.progress(float(confidence), text="Positive Confidence")
                else:
                    st.progress(float(confidence), text="Negative Confidence")

            with st.expander("See raw probabilities"):
                st.write(f"Negative: {probs[0]:.4f}")
                st.write(f"Positive: {probs[1]:.4f}")