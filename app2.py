import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# تحميل النموذج المحلي
@st.cache_resource
def load_model():
    model_name = "akhooli/gpt2-small-arabic"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# إعداد الواجهة
st.set_page_config(page_title="🎯 Creative Content Agent", layout="centered")
st.title("🎯 Creative Content Agent")

product_name = st.text_input("اسم المنتج", placeholder="مثال: ليمون ونعناع")
product_desc = st.text_area("وصف المنتج", placeholder="مثال: مشروب غازي جديد ومنعش مصنوع من مكونات طبيعية.")
target_audience = st.text_input("الجمهور المستهدف", placeholder="مثال: الشباب والطلاب في المدن السعودية")
tone_options = ["مرحة", "رسمية", "تحفيزية", "غامضة", "عاطفية"]
tone = st.selectbox("نبرة الكتابة", tone_options)

# توليد المحتوى
if st.button("✨ توليد المحتوى"):
    if product_name and product_desc and target_audience and tone:
        prompt = f"""اكتب وصفًا تسويقيًا إبداعيًا لمنتج اسمه '{product_name}'. 
الوصف: {product_desc} 
الجمهور المستهدف: {target_audience}. 
نبرة الكتابة: {tone}. 
استخدم أسلوبًا جذابًا يُشبه الإعلانات على السوشيال ميديا."""

        with st.spinner("جاري التوليد..."):
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(**inputs, max_new_tokens=200, pad_token_id=tokenizer.eos_token_id)
            result = tokenizer.decode(output[0], skip_special_tokens=True)

        st.success("تم توليد النص:")
        st.write(result[len(prompt):].strip())
    else:
        st.warning("يرجى تعبئة جميع الحقول.")
