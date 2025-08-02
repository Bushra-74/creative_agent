import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# تحميل النموذج
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("akhooli/gpt2-small-arabic")
    model = AutoModelForCausalLM.from_pretrained("akhooli/gpt2-small-arabic")
    return tokenizer, model

tokenizer, model = load_model()

# واجهة المستخدم
st.set_page_config(page_title="🎯 Creative Content Agent", layout="centered")
st.title("🎯 Creative Content Agent")

prompt = st.text_area("أدخل بداية النص بالعربية", placeholder="مثال: في عالم التكنولوجيا الحديثة،")

if st.button("🔮 توليد"):
    if prompt.strip():
        st.info("⏳ جاري التوليد...")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
        output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        st.success("🎉 النص الناتج:")
        st.write(output)
    else:
        st.warning("⚠️ يرجى إدخال نص للبدء.")
