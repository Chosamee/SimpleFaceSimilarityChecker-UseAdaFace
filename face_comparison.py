from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import torch
from face_alignment import align
from inference import load_pretrained_model, to_input

from PIL import Image
import os
import cv2


def normalize_images_in_directory():
    # 입력 디렉토리의 모든 파일 목록 가져오기\
    input_directory = "input_images"
    output_directory = "test_images"
    file_list = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f))]

    # 출력 디렉토리 생성
    os.makedirs(output_directory, exist_ok=True)

    for file_name in file_list:
        # 이미지 파일인지 확인
        if file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            input_path = os.path.join(input_directory, file_name)
            output_path = os.path.join(output_directory, file_name)

            # 이미지 로드
            img = cv2.imread(input_path)

            if img is not None:
                # 이미지 정규화
                normalized_img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX)

                # 정규화된 이미지 저장
                cv2.imwrite(output_path, normalized_img)
                print(f"{file_name} 이미지가 정규화되어 {output_path}에 저장되었습니다.")
            else:
                print(f"{file_name} 이미지를 불러올 수 없습니다.")


def find_second_largest_in_row(row):
    # 행에서 가장 큰 값을 찾고, 그 값을 제외한 후 다시 가장 큰 값을 찾음
    top_values, top_indices = torch.topk(row, 2)
    # 두 번째로 큰 값의 인덱스 반환
    return top_indices[1].item()


def face_comparison():
    """
    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    )
    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    """
    normalize_images_in_directory()
    model = load_pretrained_model("ir_50")

    test_image_path = "test_images"
    features = []
    fnames = []
    bgr_tensor_inputs = []
    for fname in sorted(os.listdir(test_image_path)):
        if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            fnames.append(fname)
            path = os.path.join(test_image_path, fname)
            aligned_rgb_img = align.get_aligned_face(path)
            if aligned_rgb_img is None:
                fnames.pop()
                continue
            bgr_tensor_input = to_input(aligned_rgb_img)
            bgr_tensor_inputs.append(bgr_tensor_input)

    bgr_tensor_inputs = torch.stack(bgr_tensor_inputs)  # list -> tensor
    features, _ = model(bgr_tensor_inputs)
    similarity_scores = features @ features.T
    second_largest_indices = [find_second_largest_in_row(row) for row in similarity_scores]

    for i, indices in enumerate(second_largest_indices):
        print(fnames[i], fnames[indices], similarity_scores[i][indices].item())
    for i, indices in enumerate(second_largest_indices):
        print(fnames[i], similarity_scores[i])

    """
    img = Image.open("1.jpg")
    img2 = Image.open("11.jpg")
    img3 = Image.open("22.jpg")
    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img)
    img_cropped2 = mtcnn(img2)
    img_cropped3 = mtcnn(img3)
    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))
    img_embedding2 = resnet(img_cropped2.unsqueeze(0))
    img_embedding3 = resnet(img_cropped3.unsqueeze(0))
    print(img_embedding, img_embedding2)
    print((img_embedding-img_embedding2).norm().item())
    print((img_embedding-img_embedding3).norm().item())
    """
