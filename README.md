# Tailor

```python
from tailor import Interpreter
from torchvision.models import resnet18

model = resnet18()

interpreter = Interpreter()
interpreter.plot(module=model, input_shape=(1, 3, 224, 224))
```