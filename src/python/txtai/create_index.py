from txtai.embeddings import Embeddings
import json

emb = Embeddings({"path": "sentence-transformers/all-MiniLM-L6-v2"})
docs = []
with open(r"D:\backend\txtai\src\python\txtai\customagents\phiaagent\fewshot_examples.jsonl","r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        docs.append(item["content"])

emb.index(docs)
emb.save("fewshots_index")