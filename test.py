from fastapi import FastAPI, File, UploadFile, Request, Form
import tensorflow as tf
import sys
import uvicorn
from multiprocessing import Process
from typing import List
import random
import torch
import numpy as np
from fastapi import File, UploadFile
from PIL import Image
import numpy as np
import io
from face_alignment import align
from inference import load_pretrained_model, to_input

PYTORCH_NO_CUDA_MEMORY_CACHING=1
seed = 50
tf.random.set_seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from pathlib import Path

"""
이 파일은 코드를 이해하는 목적 이외에 하등 쓸모 없습니다

이 파일 돌려보고 대강 이해하면서 app.py를 수정하시고 아예 삭제해도 댐다
"""

model = load_pretrained_model("ir_50")  # 모델을 로드합니다.
device = torch.device("cpu") 
#"cuda:3"을 써야할 수도 있습니다. 할당된 디바이스를 잘 봐야합니다

# 이미지를 전처리하고 얼굴을 정렬하는 함수입니다.
def preprocess_and_align(image_bytes):
    image_path = image_bytes
    #image_path = io.BytesIO(image_bytes)
    aligned_rgb_img = align.get_aligned_face(image_path,device)
    if aligned_rgb_img is not None:
        bgr_tensor_input = to_input(aligned_rgb_img)
        return bgr_tensor_input
    else:
        return None
    
    
# 사용자 ID에 따른 프로필 이미지 벡터를 가져오는 함수 (데이터베이스 구현 필요)
def get_profile_image_embeddings(user_ids):
    """ #TODO
    아래에 있는 코드는 로컬에서 테스트해보려고 작성한거고, 실제로는 DB의 User 정보에서 embedding을 잘 가져와야합니다.
    """
    user_profile_images = {
        "user_id_1": "path/to/image1.pt",
        "user_id_2": "path/to/image2.pt",
        # ...
    }
    return [user_profile_images[user_id] for user_id in user_ids if user_id in user_profile_images]


def profile_embedding(image):
    tensor_input = preprocess_and_align(image)
    if tensor_input is None:
        return {"error": "No face detected or alignment failed."}
    with torch.no_grad():
        features, _ = model(tensor_input.unsqueeze(0))  # 이미지가 하나만 있으므로 배치 차원 추가

    # 업로드된 파일의 원래 이름을 사용하여 임베딩 파일 이름을 설정합니다.
    # 파일 확장자를 변경하기 위해 원래 파일 이름에서 확장자를 제거하고 '.pt'를 추가합니다.
    # 당연하지만 실제 DB에 넣을때는 filename고려 안해도 됩니다. test용도로 넣었어요
    image_path = Path(image)
    embedding_filename = image_path.stem+".pt"
    embedding_path = Path("face_alignment/profile_images") / embedding_filename

    # 임베딩 벡터를 파일로 저장합니다.
    torch.save(features.squeeze(0), embedding_path)
    return {"embedding": features.squeeze(0).tolist()}  # 임베딩 벡터 반환

def get_users_with_image(images, user_ids):
    # 업로드된 이미지 처리
    uploaded_tensor_inputs = []
    for image in images:
        tensor_input = preprocess_and_align(image)
        if tensor_input is not None:
            uploaded_tensor_inputs.append(tensor_input)

    # 프로필 이미지 처리
    #profile_image_paths = get_profile_image_embeddings(user_ids) # image embedding 처리. DB에서 profile embedding들고오는거
    profile_image_paths = user_ids # test에서는 그냥 임베딩 pt를 리스트로 주니까 이거로 함. 
    profile_embeddings = []
    for path in profile_image_paths:
        embedding = torch.load(path)  # 임베딩 파일 로드
        profile_embeddings.append(embedding)
    profile_embeddings = torch.stack(profile_embeddings)

    # 모델을 통해 임베딩 생성
    with torch.no_grad():
        uploaded_features, _ = model(torch.stack(uploaded_tensor_inputs)) if uploaded_tensor_inputs else ([], [])
    
    # 유사도 계산
    similarity_scores = torch.mm(uploaded_features, profile_embeddings.T)  # 업로드된 이미지 임베딩과 프로필 이미지 임베딩 간의 유사도
    most_similar_indices = similarity_scores.argmax(dim=1)  # 각 업로드된 이미지에 대해 가장 유사한 프로필 이미지의 인덱스
    
    # 결과 매핑
    results = [{"uploaded_image_name":images[i],"uploaded_image_index": i, "matched_user_id": user_ids[idx], "score":similarity_scores[i][idx]} for i, idx in enumerate(most_similar_indices.tolist())]
    for result in results:
        print(result,"\n") # 점수가 높을수록 높은겁니다
    return {"results": results}



if __name__ == "__main__":
    if sys.argv[1]=='profile_embedding':
        image = "face_alignment/test_images/profile/q3.jpg"
        profile_embedding(image) # 이거 하면 face_alighment/profile_images에 embedding pt 생깁니다.

    if sys.argv[1]=='get_users_with_image':
        images = ["face_alignment/test_images/a2.jpg","face_alignment/test_images/a3.jpg","face_alignment/test_images/a4.jpg","face_alignment/test_images/c1.jpg","face_alignment/test_images/d2.jpg"]
        user_ids=["face_alignment/profile_images/a1.pt","face_alignment/profile_images/c2.pt","face_alignment/profile_images/d1.pt"]
        get_users_with_image(images, user_ids)
    """ 출력 예
    {'uploaded_image_name': 'face_alignment/test_images/a2.jpg', 'uploaded_image_index': 0, 'matched_user_id': 'face_alignment/profile_images/a1.pt', 'score': tensor(0.4841)} 

    {'uploaded_image_name': 'face_alignment/test_images/a3.jpg', 'uploaded_image_index': 1, 'matched_user_id': 'face_alignment/profile_images/a1.pt', 'score': tensor(0.2231)} 

    {'uploaded_image_name': 'face_alignment/test_images/a4.jpg', 'uploaded_image_index': 2, 'matched_user_id': 'face_alignment/profile_images/a1.pt', 'score': tensor(0.4291)} 

    {'uploaded_image_name': 'face_alignment/test_images/c1.jpg', 'uploaded_image_index': 3, 'matched_user_id': 'face_alignment/profile_images/c2.pt', 'score': tensor(0.7241)} 

    {'uploaded_image_name': 'face_alignment/test_images/d2.jpg', 'uploaded_image_index': 4, 'matched_user_id': 'face_alignment/profile_images/d1.pt', 'score': tensor(0.5014)} 
    """