# ingest_docs.py
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle
from langchain.document_loaders import TextLoader

EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

def load_texts_from_folder(folder):
    p = Path(folder)
    texts = []
    for f in p.rglob('*'):
        if f.suffix.lower() == '.txt':
            with open(f, 'r', encoding='utf8') as fh:
                texts.append(fh.read())
        elif f.suffix.lower() == '.pdf':
            texts.append(f"[PDF: {f.name}] Please extract text or install unstructured library for PDF parsing.")
    return texts

def build_index(texts, model_name=EMBED_MODEL_NAME, out_index='pirag_faiss.index'):
    model = SentenceTransformer(model_name)
    emb = model.encode(texts, convert_to_numpy=True)
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(emb, dtype='float32'))
    faiss.write_index(index, out_index)
    with open(out_index + '.pkl', 'wb') as f:
        pickle.dump(texts, f)
    print('Index written:', out_index)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs-folder', type=str, default='pirag_docs')
    parser.add_argument('--out-index', type=str, default='pirag_faiss.index')
    args = parser.parse_args()

    texts = load_texts_from_folder(args.docs_folder)
    build_index(texts, out_index=args.out_index)
