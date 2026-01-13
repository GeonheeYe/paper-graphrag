#!/usr/bin/env python3
"""
HuggingFace 모델 다운로드 스크립트
프로젝트에서 사용하는 모든 모델을 미리 다운로드합니다.
"""

from huggingface_hub import snapshot_download
import os

def download_model(model_name: str, description: str = ""):
    """모델을 다운로드하는 헬퍼 함수"""
    print(f"\n{'='*50}")
    if description:
        print(f"[{description}] {model_name} 다운로드 중...")
    else:
        print(f"{model_name} 다운로드 중...")
    print('='*50)
    
    try:
        # local_dir_use_symlinks=False: 심볼릭 링크 대신 실제 파일 복사
        snapshot_download(
            repo_id=model_name,
            local_dir_use_symlinks=False,
            resume_download=True,  # 중단된 다운로드 재개
        )
        print(f"✓ {model_name} 다운로드 완료!")
        return True
    except Exception as e:
        print(f"✗ {model_name} 다운로드 실패: {e}")
        return False

def main():
    """메인 함수: 모든 모델 다운로드"""
    print("\n" + "="*50)
    print("모델 다운로드 시작")
    print("="*50)
    
    # 다운로드할 모델 리스트
    models = [
        ("Qwen/Qwen2.5-7B-instruct", "1/3 - LLM 모델 (geonlight.py)"),
        ("Qwen/Qwen3-VL-4B-Instruct", "2/3 - Vision-Language 모델 (PDF_to_RAG.py)"),
        ("upskyy/bge-m3-korean", "3/3 - Embedding 모델 (embeddings)"),
        ("Qwen/Qwen3-4B-Instruct-2507", "4/4 - LLM 모델 (geonlight.py)"),
    ]
    
    results = []
    for model_name, description in models:
        success = download_model(model_name, description)
        results.append((model_name, success))
    
    # 결과 요약
    print("\n" + "="*50)
    print("다운로드 결과 요약")
    print("="*50)
    for model_name, success in results:
        status = "✓ 성공" if success else "✗ 실패"
        print(f"{status}: {model_name}")
    
    print("\n" + "="*50)
    print("모델 다운로드 완료!")
    print("="*50)
    print(f"\n다운로드된 모델 위치: {os.path.expanduser('~/.cache/huggingface/hub/')}")
    print("\n개별 모델만 다운로드하려면:")
    print("  python download_models.py")
    print("  또는 코드에서 download_model('모델명') 호출")
    print()

if __name__ == "__main__":
    main()

