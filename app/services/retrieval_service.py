import torch
from ..repositories.vector_repo import VectorRepo
from ..utils.prompt import geonlight_prompt
from ..config import Config

class RetrievalService:
    def __init__(self, vector_repo: VectorRepo, config: Config): 
        self.vector_repo = vector_repo
        self.model = config['HF_MODEL']
        self.tokenizer = config['HF_TOKENIZER']

    
    @classmethod
    def from_app_config(cls, config) -> 'RetrievalService':
        vector_repo = VectorRepo(
            chroma_dir=config['CHROMA_DIR'],
            embed_model=config['EMBED_MODEL']
        )
        return cls(vector_repo=vector_repo, config=config)

    def answer(self, doc_id: str, question: str, top_k: int) -> dict:
        hits = self.vector_repo.similarity_search(doc_id=doc_id, query=question, top_k=top_k)
        hits = hits[:top_k]
        # for text, meta, score in hits[:top_k]:
        #     page = meta.get('page_idx')
        #     block = meta.get('block_idx')
        #     table = meta.get('table_idx')
        #     figure = meta.get('figure_idx')
        #     section = meta.get('section')
        #     type = meta.get('type')
        #     text = text.replace('\n', ' ')
        contexts = "\n".join([f"{hit.metadata['source']}: {hit.page_content}" for hit in hits])

        prompt = geonlight_prompt.format(context=contexts, question=question)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=1024, temperature=0.7, top_p=0.9)

        with torch.no_grad():
            answer = self.tokenizer.decode(outputs[0].messages.content.strip())
        return answer