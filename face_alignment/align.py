import sys
import os

from face_alignment import mtcnn
import argparse
from PIL import Image
from tqdm import tqdm
import random
from datetime import datetime




def add_padding(pil_img, top, right, bottom, left, color=(0, 0, 0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result


def get_aligned_face(image_path, device, rgb_pil_image=None):
    mtcnn_model = mtcnn.MTCNN(device, crop_size=(112, 112)) # Device 설정때문에 모델은 보통 함수 내에 쓰는게 조음
    if rgb_pil_image is None:
        img = Image.open(image_path).convert("RGB")
    else:
        assert isinstance(rgb_pil_image, Image.Image), "Face alignment module requires PIL image or path to the image"
        img = rgb_pil_image
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=None)
        face = faces[0]
        # 사진에 얼굴이 두 개 이상 검출될 시 에러 발생 -> 추후에 cam에서 다른 사람이 같은 방에 있는지 확인할때 사용
        if len(faces) >= 2:
            print()
            print("###############################")
            print("Detected more than one face.")
            print("file name : " + image_path)
            print("Number of detected faces : " + str(len(faces)))
            print("###############################")
            print()
            face = None
    except Exception as e:
        print("Face detection Failed due to error.")
        print(e)
        face = None

    return face
