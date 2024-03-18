import net
import torch
import os
from face_alignment import align
import numpy as np


adaface_models = {
    "ir_50": "pretrained/adaface_ir50_ms1mv2.ckpt",
}


def load_pretrained_model(architecture="ir_50"):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    # 여기에 파일이 없으면 다운을 하는 코드 넣을 수 있음. google drive나 dropbox zenodo 등 링크는 "pretrained/adaface_ir50_ms1mv2.ckpt"
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture], map_location="cpu")["state_dict"]
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith("model.")}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.0) - 0.5) / 0.5
    tensor = torch.from_numpy(brg_img.transpose(2, 0, 1)).float()
    return tensor


if __name__ == "__main__":
    model = load_pretrained_model("ir_50")
    device = torch.device("cpu")
    feature, norm = model(torch.randn(2, 3, 112, 112))

    test_image_path = "face_alignment/test_images"
    features = []
    bgr_tensor_inputs = []
    for fname in sorted(os.listdir(test_image_path)):
        path = os.path.join(test_image_path, fname)
        aligned_rgb_img = align.get_aligned_face(path, device)
        bgr_tensor_input = to_input(aligned_rgb_img)
        bgr_tensor_inputs.append(bgr_tensor_input)

    bgr_tensor_inputs = torch.cat(bgr_tensor_inputs)  # list -> tensor
    print(bgr_tensor_inputs.shape)
    features = model(bgr_tensor_inputs)

    similarity_scores = features @ features.T
    print(similarity_scores)
