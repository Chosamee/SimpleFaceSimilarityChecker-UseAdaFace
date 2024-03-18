import base64
import json
import pickle
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, Request, Form
import sys
from requests import Session
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

# from database import get_db
from face_alignment import align
from inference import load_pretrained_model, to_input
from user import *

PYTORCH_NO_CUDA_MEMORY_CACHING = 1
seed = 50
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from pathlib import Path

app = FastAPI()

model = load_pretrained_model("ir_50")  # 모델을 로드합니다.
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# nvidia-smi로 할당된 디바이스를 보고 cuda:{번호} 변경


# 이미지를 전처리하고 얼굴을 정렬하는 함수
def preprocess_and_align(image_bytes):
    image_path = io.BytesIO(image_bytes)
    aligned_rgb_img = align.get_aligned_face(image_path, device)
    if aligned_rgb_img is not None:
        bgr_tensor_input = to_input(aligned_rgb_img)
        return bgr_tensor_input
    else:
        return None


# 사용자 ID에 따른 프로필 이미지 벡터를 가져오는 함수 (데이터베이스 구현 필요)
def get_profile_image_embeddings(user_ids):
    """
    input : user_ids
    output : 해당 user의 image의 Embedding Tensor
    """
    user_profile_images = {
        "user_id_1": "path/to/image1.pt",
        "user_id_2": "path/to/image2.pt",
        # ...
    }
    return [user_profile_images[user_id] for user_id in user_ids if user_id in user_profile_images]


@app.post("/profile_embedding")  # 프로필 임베딩을 가져오는 함수. 처음 프로필을 생성할때 한번씩 돌리는거
async def profile_embedding(image: UploadFile = File(...), internal_id: str = Form(None)):
    image_bytes = await image.read()
    tensor_input = preprocess_and_align(image_bytes)
    if tensor_input is None:
        return {"error": "No face detected or alignment failed."}
    with torch.no_grad():
        features, _ = model(tensor_input.unsqueeze(0))  # 이미지가 하나만 있으므로 배치 차원 추가
    print(features)

    numpy_array = features.numpy()
    binary_data = numpy_array.tobytes()
    encoded_data = base64.b64encode(binary_data).decode("utf-8")

    # Base64 인코딩 문자열을 JSON 응답으로 반환
    return {"encoded_data": encoded_data}


"""
    # DB를 AI서버에서 direct로 이용할 때. "/profile_embedding"에 추가
    user = db.query(User).filter(User.internal_id == internal_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Not found User")
    user.embedded_profile = binary_data
    db.commit()
    # 업로드된 파일의 원래 이름을 사용하여 임베딩 파일 이름을 설정합니다.
    # 파일 확장자를 변경하기 위해 원래 파일 이름에서 확장자를 제거하고 '.pt'를 추가합니다.
    embedding_filename = f"{image.filename.rsplit('.', 1)[0]}.pt"
    embedding_path = Path("face_alignment/profile_images") / embedding_filename

    # 임베딩 벡터를 파일로 저장합니다.
    torch.save(features.squeeze(0), embedding_path)
    return {"embedding": features.squeeze(0).tolist()}  # 임베딩 벡터 반환
"""


@app.post("/get_users_with_image")
async def get_users_with_image(
    files: List[UploadFile] = File(...),
    embeddeds: str = Form(...),
):
    dtype = np.float32  # 원본 배열의 데이터 타입
    shape = (1, 512)  # 원본 배열의 모양
    # 업로드된 이미지 처리
    uploaded_tensor_inputs = []
    profile_embeddings = []
    ids = []
    not_found = []
    decoded_embeddeds = base64.b64decode(embeddeds)
    embs = pickle.loads(decoded_embeddeds)
    for idx, image in enumerate(files):
        image_bytes = await image.read()
        tensor_input = preprocess_and_align(image_bytes)
        internal_id = image.filename.split(".")[0]
        if tensor_input is not None:
            uploaded_tensor_inputs.append(tensor_input)
        else:
            not_found.append(internal_id)
        # 프로필 이미지 처리
        ids.append(internal_id)
        # embedded_profile = db.query(User).fil ter(User.internal_id == internal_id).first().embedded_profile
        print(idx)
        if embs[idx] is not None:
            numpy_array = np.frombuffer(embs[idx], dtype=dtype).reshape(shape)
            # NumPy 배열을 PyTorch 텐서로 변환
            tensor = torch.from_numpy(numpy_array)
            profile_embeddings.append(tensor)
    # profile_image_paths = get_profile_image_embeddings(user_ids)
    # for path in profile_image_paths:
    #     embedding = torch.load(path)  # 임베딩 파일 로드
    #     profile_embeddings.append(embedding)

    # 모델을 통해 임베딩 생성
    with torch.no_grad():
        uploaded_features, _ = model(torch.stack(uploaded_tensor_inputs)) if uploaded_tensor_inputs else ([], [])
    # 유사도 계산
    similarity_scores = torch.mm(
        uploaded_features, torch.cat(profile_embeddings).T
    )  # 업로드된 이미지 임베딩과 프로필 이미지 임베딩 간의 유사도
    most_similar_indices = similarity_scores.argmax(
        dim=1
    )  # 각 업로드된 이미지에 대해 가장 유사한 프로필 이미지의 인덱스

    # 결과 매핑
    results = [
        {
            "uploaded_user_id": files[i].filename.split(".")[0],
            "uploaded_image_index": i,
            "matched_user_id": ids[idx],
            "score": similarity_scores[i][idx].item(),
        }
        for i, idx in enumerate(most_similar_indices.tolist())
    ]
    print(results)
    print(not_found)
    return {"results": results, "errors": not_found}


def run_api():
    uvicorn.run("app:app", host="0.0.0.0", port=7000, log_level="debug", reload=True)


if __name__ == "__main__":
    if sys.argv[1] == "run_app":
        api_process = Process(target=run_api)
        api_process.start()
        api_process.join()
