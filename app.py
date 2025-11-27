# app.py
import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path
import pickle

# PM4Py
import pm4py
from pm4py.statistics.traces.generic.pandas import get_variants
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.log.util import dataframe_utils

# Embeddings & FAISS
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Document parsing
from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Generation (local LLM fallback)
from transformers import pipeline

st.set_page_config(page_title="PiRAG Open-Source Demo", layout="wide")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = "google/flan-t5-small"  # small generation model

@st.experimental_singleton
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.experimental_singleton
def load_gen_model():
    return pipeline("text2text-generation", model=GEN_MODEL, device=-1)

# ---------- Helpers ----------
def read_event_log(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    if name.lower().endswith('.xes'):
        # save temp
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xes')
        tmp.write(uploaded_file.getbuffer())
        tmp.close()
        log = pm4py.read_xes(tmp.name)
        df = pm4py.convert_to_dataframe(log)
        return df
    else:
        return pd.read_csv(uploaded_file)

def extract_variants(df: pd.DataFrame, case_id_col='case_id', activity_col='activity'):
    try:
        df = df.rename(columns={case_id_col:'case_id', activity_col:'activity'})
    except Exception:
        pass
    variants = get_variants.get_variants_dataframe(df, case_id='case_id', activity_key='activity')
    docs = []
    for idx, row in variants.iterrows():
        seq = row['variant']
        cnt = int(row['count'])
        text = f"Variant: {' -> '.join(seq)}. Occurrences: {cnt}."
        docs.append({"variant": seq, "count": cnt, "text": text})
    return docs

def build_faiss_index(texts: List[str], embed_model, index_path: Path):
    emb = embed_model.encode(texts, convert_to_numpy=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(emb, dtype='float32'))
    faiss.write_index(index, str(index_path))
    with open(index_path.with_suffix('.pkl'), 'wb') as f:
        pickle.dump(texts, f)
    return index

def load_faiss_index(index_path: Path, embed_model):
    index = faiss.read_index(str(index_path))
    with open(index_path.with_suffix('.pkl'), 'rb') as f:
        texts = pickle.load(f)
    return index, texts

def search_faiss(index, texts, query, embed_model, k=4):
    qv = embed_model.encode([query], convert_to_numpy=True)[0].astype('float32')
    D, I = index.search(np.array([qv]), k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        results.append({"text": texts[idx], "score": float(dist)})
    return results

# ---------- Streamlit UI ----------
st.title("PiRAG — Open-Source Cloud Demo (no-cost defaults)")

col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("Upload event log (CSV or XES)", type=['csv','xes'])
with col2:
    docs_zip = st.file_uploader("(Optional) Upload pirag_docs ZIP (or place pirag_docs/ in repo)", type=['zip'])

if uploaded:
    df = read_event_log(uploaded)
    st.write("Event log preview:")
    st.dataframe(df.head())
    if st.button("Run Pi: extract variants"):
        with st.spinner("Extracting variants..."):
            variants = extract_variants(df)
            st.write(f"Found {len(variants)} variants")
            for v in variants[:20]:
                st.markdown(f"- **{v['count']}x** — {' → '.join(v['variant'])}")
            st.session_state['variants'] = variants

st.markdown("---")

# Indexing documents (local FAISS)
st.header("Document ingestion & FAISS indexing (local)")
if st.button("Build local FAISS index from pirag_docs folder"):
    docs_folder = Path('pirag_docs')
    if not docs_folder.exists():
        st.error('pirag_docs folder not found in the working directory. Upload or create it.')
    else:
        texts = []
        for f in docs_folder.rglob('*'):
            if f.suffix.lower() == '.txt':
                loader = TextLoader(str(f), encoding='utf8')
                for d in loader.load():
                    texts.append(d.page_content)
            elif f.suffix.lower() == '.pdf':
                try:
                    from unstructured.partition.pdf import partition_pdf
                    txt = '\n'.join([p.get('text','') for p in partition_pdf(filename=str(f))])
                    texts.append(txt)
                except Exception:
                    texts.append(f"[PDF: {f.name}] Please extract text (unstructured not installed)")
        if not texts:
            st.warning('No text extracted from pirag_docs. Place .txt files or enable unstructured PDF parsing.')
        else:
            embed_model = load_embed_model()
            idx_path = Path('pirag_faiss.index')
            build_faiss_index(texts, embed_model, idx_path)
            st.success('Built faiss index at pirag_faiss.index')

st.markdown('---')

# Querying
st.header('RAG Q&A (using local FAISS + local generator)')
query = st.text_input('Ask a question about the process or docs')
if st.button('Get answer') and query.strip():
    idx_path = Path('pirag_faiss.index')
    if not idx_path.exists():
        st.error('FAISS index not found. Build it first.')
    else:
        embed_model = load_embed_model()
        index, texts = load_faiss_index(idx_path, embed_model)
        results = search_faiss(index, texts, query, embed_model, k=4)
        st.subheader('Retrieved snippets')
        context = ''
        for r in results:
            st.markdown(f"- (score {r['score']:.3f}) {r['text'][:400]}...")
            context += '\n' + r['text']
        st.subheader('Generated answer')
        gen = load_gen_model()
        prompt = f"Use the context below to answer the question.\nContext:\n{context}\nQuestion: {query}\nAnswer:"
        out = gen(prompt, max_length=256, do_sample=False)
        st.write(out[0]['generated_text'])

st.markdown('\n---\n')
st.caption('PiRAG Open-Source demo — defaults to local models and FAISS to avoid paid services. For production, swap to Pinecone/OpenAI as needed.')
