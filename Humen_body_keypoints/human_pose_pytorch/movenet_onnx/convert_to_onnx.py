import torch
import torch.onnx
import onnx
from Config import ConfigSet
from classification import Classification


if __name__ == "__main__":
    model_parameter = torch.load(r"G:\Module_Parameter\humen_pose\after_movenet\0\model_param.pt")
    config = ConfigSet(train_root=r"M:\data package\human_pose_detection\train_keypoint_dict.pt",
                       test_root=r"M:\data package\human_pose_detection\test_keypoint_dict.pt",
                       cnn_input=[2, 4, 8, 16, 8, 4, 2],
                       cnn_output=[4, 8, 16, 8, 4, 2, 1],
                       layer_input=[34, 64, 128, 256, 256, 128, 64, 32, 16],
                       layer_output=[64, 128, 256, 256, 128, 64, 32, 16, 5],
                       epoch_num=93,
                       lr=0.00002,
                       device="cuda" if torch.cuda.is_available() else "cpu",
                       )
    model = Classification(config["cnn_input"], config["cnn_output"],
                           config["layer_input"], config["layer_output"])
    model.load_state_dict(model_parameter)
    model.eval()
    model_input = (1, 17, 3)
    input_batch = 1
    x = torch.randn(input_batch, *model_input)
    export_onnx_file = r"G:\python_program\Humen_body_keypoints\human_pose_pytorch\classification.onnx"

    torch.onnx.export(model,
                      x,
                      export_onnx_file,
                      opset_version=14,
                      do_constant_folding=True,
                      input_names=["input"],
                      output_names=["output"],
                      dynamic_axes={"input": {0: "batch_size"},
                                    "output": {0: "batch_size"}},
                      )
