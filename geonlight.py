"""
RAG 기반 질의응답 시스템
HuggingFace 모델 또는 vLLM 서버를 사용하여 문서 기반 질문에 답변합니다.
"""

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from prompt import geonlight_prompt
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

# ============================================================================
# 설정
# ============================================================================
PERSIST_DIRECTORY = "./chroma_db"
MODEL_TYPE = "vllm"  # "vllm" or "HF"
MODEL_PATH = "Qwen/Qwen2.5-7B-instruct"
EMBEDDING_MODEL = "upskyy/bge-m3-korean"
TOP_K = 8  # 검색할 문서 개수
MAX_NEW_TOKENS = 512  # HF 모델용
VLLM_MAX_TOKENS = 1024  # vLLM 모델용
VLLM_TEMPERATURE = 0.1
VLLM_TOP_P = 0.9
VLLM_SEED = 42
VLLM_API_BASE = "http://localhost:8000/v1"

# vLLM 서버 실행 명령어 (참고용)
"""
vllm serve Qwen/Qwen2.5-7B-instruct \
    --max-model-len 10000 \
    --port 8000 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --dtype auto \
    --gpu_memory_utilization 0.25 \
    --seed 42
"""

# ============================================================================
# 초기화 함수
# ============================================================================

def initialize_vector_store(persist_directory: str, embedding_model: str, top_k: int):
    """벡터 스토어와 retriever 초기화"""
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )
    return retriever


def load_hf_model(model_path: str, device: str = "cuda:0"):
    """HuggingFace 모델 로드"""
    print(f"Loading HuggingFace model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    ).to("cuda")
    model.eval()
    return model, tokenizer


# ============================================================================
# RAG 답변 생성 함수
# ============================================================================

def hf_answer(
    question: str,
    retriever,
    model,
    tokenizer,
    top_k: int,
    max_new_tokens: int = MAX_NEW_TOKENS
):
    """HuggingFace 모델을 사용한 RAG 답변 생성"""
    docs = retriever.invoke(question)
    docs = docs[:top_k]
    context = "\n".join([f"{doc.metadata['source']}: {doc.page_content}" for doc in docs])
    
    prompt = geonlight_prompt.format(context=context, question=question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer


def vllm_answer(
    question: str,
    retriever,
    model_path: str,
    top_k: int,
    max_tokens: int = VLLM_MAX_TOKENS,
    temperature: float = VLLM_TEMPERATURE,
    top_p: float = VLLM_TOP_P,
    seed: int = VLLM_SEED,
    api_base: str = VLLM_API_BASE,
    debug: bool = False
):
    """vLLM 서버를 사용한 RAG 답변 생성"""
    docs = retriever.invoke(question)
    docs = docs[:top_k]
    
    context = "\n".join([f"[{doc.metadata['source']}]:\n {doc.page_content}\n\n" for doc in docs])
    prompt = geonlight_prompt.format(context=context, question=question)
    
    # 수식 등에서 쓰인 {}를 템플릿 변수로 인식하지 않게 이스케이프
    escape_prompt = prompt.replace("{", "{{").replace("}", "}}")
    
    if debug:
        print("=" * 50)
        print("프롬프트 (이스케이프 후):")
        print("=" * 50)
        print(escape_prompt)
        print("=" * 50)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            'You are a helpful assistant that can answer questions about the provided documents.'
        ),
        HumanMessagePromptTemplate.from_template(escape_prompt)
    ])
    
    llm = ChatOpenAI(
        model=model_path,
        openai_api_base=api_base,
        openai_api_key="EMPTY",
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
    )
    
    chain = chat_prompt | llm | StrOutputParser()
    answer = chain.invoke({})  # 템플릿 변수가 없으므로 빈 dict
    return answer


# ============================================================================
# 메인 실행
# ============================================================================

def main():
    """메인 실행 함수"""
    # 벡터 스토어 초기화
    print("Initializing vector store...")
    retriever = initialize_vector_store(PERSIST_DIRECTORY, EMBEDDING_MODEL, TOP_K)
    
    # 모델 로드 (HF 모드인 경우만)
    model = None
    tokenizer = None
    if MODEL_TYPE == "HF":
        model, tokenizer = load_hf_model(MODEL_PATH)
    
    # 질문 리스트
    questions = [
        "GRPO가 특히 유용한 상황(모델/데이터/학습환경)은 어떤 경우인가?",
        "figure 1번에서 설명하는 바가 뭐야?",
        "GRPO는 어떤 기존 방법(RLHF, PPO 등)과 비교하여 제안되었나?",
        "GRPO가 무엇의 약자이며, 한 줄로 정의하면?",
        "GRPO가 PPO와 비교해 '핵심적으로 바뀐 점' 2가지는?",
        "GRPO에서 'group'은 무엇을 의미하고, 왜 필요한가?",
        "GRPO가 해결하려는 문제(목표)는 무엇인가?",
    ]
    
    print(f"\nUsing {MODEL_TYPE} model")
    print("=" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] Question: {question}")
        print("-" * 50)
        
        if MODEL_TYPE == "vllm":
            answer = vllm_answer(
                question,
                retriever,
                MODEL_PATH,
                TOP_K,
                debug=False
            )
        else:
            answer = hf_answer(
                question,
                retriever,
                model,
                tokenizer,
                TOP_K,
                MAX_NEW_TOKENS
            )
        
        print(f"Answer: {answer}")
        print("=" * 50)


if __name__ == "__main__":
    main()
