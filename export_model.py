import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights

model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
model.fc = torch.nn.Linear(2048, 4)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224, requires_grad=False)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy_input,
        "model_repository/resnet_medical/1/model.onnx",
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
    )

print("Model exported successfully")
