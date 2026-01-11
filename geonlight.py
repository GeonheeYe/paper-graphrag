#%%
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from prompt import geonlight_prompt

PERSIST_DIRECTORY = "./chroma_db"
top_k = 15

embeddings = HuggingFaceEmbeddings(model_name="upskyy/bge-m3-korean")
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})


def rag_answer(question: str, retriever, max_context_docs=10):
    docs = retriever.invoke(question)
    docs = docs[:max_context_docs]
    context = "\n".join([f"{doc.metadata['source']}: {doc.page_content}" for doc in docs])

    prompt = geonlight_prompt.format(context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512)    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer


model_path = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
question = "GRPO는 어떤 기존 방법(RLHF, PPO 등)과 비교하여 제안되었나?"
answer = rag_answer(question, retriever, top_k)
print(answer)

# %%
