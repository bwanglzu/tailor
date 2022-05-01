import torch
import torchvision.models as models

from torchinspect.interpreter import Interpreter

# Model under test
rn18 = models.resnet18()

print(rn18.__class__.__name__)

input_ = torch.randn(1, 3, 224, 224)

int = Interpreter()
int.interpret(rn18, input_)
