# 파일 시스템을 다루는 책임을 전담하는 저장소
import os 

class FileRepo:
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    def save_pdf(self, doc_id: str, file) -> str:
        doc_dir = os.path.join(self.upload_dir, doc_id) #./data/uploads/doc_id
        os.makedirs(doc_dir, exist_ok=True)
        pdf_path = os.path.join(doc_dir, "source.pdf")
        file.save(pdf_path)
        return pdf_path