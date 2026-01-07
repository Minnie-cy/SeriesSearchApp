import streamlit as st
import os

st.title("🔧 诊断测试")

# 测试1: 检查目录
st.header("1. 检查目录结构")
for dir_path in ["data", "data/database", "data/qdrant_data", "data/models"]:
    if os.path.exists(dir_path):
        st.success(f"✅ {dir_path} 存在")
        # 列出内容
        items = os.listdir(dir_path)[:5]
        st.write(f"   内容: {items}")
    else:
        st.error(f"❌ {dir_path} 不存在")

# 测试2: 检查关键文件
st.header("2. 检查关键文件")
files_to_check = [
    "data/database/final.db",
    "data/llm_summaries.json",
    "data/qdrant_data/meta.json",
    "data/models/bge-model/config.json"
]

for file_path in files_to_check:
    if os.path.exists(file_path):
        size = os.path.getsize(file_path) / 1024 / 1024
        st.success(f"✅ {file_path} ({size:.2f} MB)")
    else:
        st.error(f"❌ {file_path} 不存在")

# 测试3: 尝试加载模型
st.header("3. 尝试加载模型")
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    model_path = "data/models/bge-model"
    
    st.write(f"模型路径: {model_path}")
    st.write(f"路径存在: {os.path.exists(model_path)}")
    
    if os.path.exists(model_path):
        files = os.listdir(model_path)
        st.write(f"模型文件数量: {len(files)}")
        st.write(f"前10个文件: {files[:10]}")
    
    with st.spinner("加载模型..."):
        embed_model = HuggingFaceEmbedding(
            model_name=model_path,
            trust_remote_code=True
        )
        st.success("✅ 模型加载成功!")
        
except Exception as e:
    st.error(f"❌ 模型加载失败: {str(e)}")
    import traceback
    st.code(traceback.format_exc())

st.write("---")
st.write("✅ 诊断完成")
