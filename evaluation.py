import torch
import os
from pytorch_fid import fid_score

# 실제 이미지 폴더와 생성된 이미지 폴더 지정
real_images_path = "data/evaluation/gt_image/common_words_13"  # 실제 손글씨 이미지가 있는 폴더
generated_images_path = "data/evaluation/generated/common_words_13"  # 생성된 손글씨 이미지가 있는 폴더

# FID Score 계산
fid_value = fid_score.calculate_fid_given_paths(
    [real_images_path, generated_images_path],  # 비교할 두 개의 이미지 폴더 경로
    batch_size=32,  # 배치 크기 (GPU 메모리 상황에 맞게 조정)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),  # GPU 사용 가능 여부 확인
    dims=2048  # Inception v3의 feature vector 차원
)

print(f"FID Score: {fid_value:.2f}")
