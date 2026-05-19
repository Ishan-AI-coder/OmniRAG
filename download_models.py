import os
os.environ["FASTEMBED_CACHE_PATH"] = "./model_cache"

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import FastEmbedSparse

print("🚀 Starting download for Dense Model (BGE-Base)...")
dense = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
print("✅ Dense Model ready!")

print("🚀 Starting download for Sparse Model (SPLADE)...")
sparse = FastEmbedSparse(model_name="prithivida/Splade_PP_en_v1")
print("✅ Sparse Model ready!")

print("🎉 All models successfully downloaded and cached!")