# Tailor


### Features

1. model visualization
2. layer freezing
3. sub-layer re-writing (insert/delete/replace)
4. quantization (in progress)
5. fully support torchvision/timm (transformers in progress)

Build on top of [torch.fx](https://pytorch.org/docs/stable/fx.html)

### Plot the model structure.


```python
from tailor import Tailor
from torchvision.models import alexnet

tailor = Tailor(model=alexnet(), input_shape=(1, 3, 224, 224))
tailor.plot()

                          Model Structure: AlexNet                         
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃         name ┃         dtype ┃ num_params ┃            shape ┃ trainable ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│   features_0 │ torch.float32 │      23296 │  [1, 64, 55, 55] │      True │
│   features_1 │ torch.float32 │          0 │  [1, 64, 55, 55] │     False │
│   features_2 │ torch.float32 │          0 │  [1, 64, 27, 27] │     False │
│   features_3 │ torch.float32 │     307392 │ [1, 192, 27, 27] │      True │
│   features_4 │ torch.float32 │          0 │ [1, 192, 27, 27] │     False │
│   features_5 │ torch.float32 │          0 │ [1, 192, 13, 13] │     False │
│   features_6 │ torch.float32 │     663936 │ [1, 384, 13, 13] │      True │
│   features_7 │ torch.float32 │          0 │ [1, 384, 13, 13] │     False │
│   features_8 │ torch.float32 │     884992 │ [1, 256, 13, 13] │      True │
│   features_9 │ torch.float32 │          0 │ [1, 256, 13, 13] │     False │
│  features_10 │ torch.float32 │     590080 │ [1, 256, 13, 13] │      True │
│  features_11 │ torch.float32 │          0 │ [1, 256, 13, 13] │     False │
│  features_12 │ torch.float32 │          0 │   [1, 256, 6, 6] │     False │
│      avgpool │ torch.float32 │          0 │   [1, 256, 6, 6] │     False │
│ classifier_0 │ torch.float32 │          0 │        [1, 9216] │     False │
│ classifier_1 │ torch.float32 │   37752832 │        [1, 4096] │      True │
│ classifier_2 │ torch.float32 │          0 │        [1, 4096] │     False │
│ classifier_3 │ torch.float32 │          0 │        [1, 4096] │     False │
│ classifier_4 │ torch.float32 │   16781312 │        [1, 4096] │      True │
│ classifier_5 │ torch.float32 │          0 │        [1, 4096] │     False │
│ classifier_6 │ torch.float32 │    4097000 │        [1, 1000] │      True │
└──────────────┴───────────────┴────────────┴──────────────────┴───────────┘
```

### Layer Freezing

```python
from tailor import Tailor
from torchvision.models import alexnet

tailor = Tailor(model=alexnet(), input_shape=(1, 3, 224, 224))
tailor.freeze(from_='classifier.4', to='classifier.6')
tailor.plot()

                          Model Structure: AlexNet                          
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┓
┃         name ┃         dtype ┃ num_params ┃            shape ┃ trainable ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━┩
│   features.0 │ torch.float32 │      23296 │  [1, 64, 55, 55] │      True │
│   features.1 │ torch.float32 │          0 │  [1, 64, 55, 55] │     False │
│   features.2 │ torch.float32 │          0 │  [1, 64, 27, 27] │     False │
│   features.3 │ torch.float32 │     307392 │ [1, 192, 27, 27] │      True │
│   features.4 │ torch.float32 │          0 │ [1, 192, 27, 27] │     False │
│   features.5 │ torch.float32 │          0 │ [1, 192, 13, 13] │     False │
│   features.6 │ torch.float32 │     663936 │ [1, 384, 13, 13] │      True │
│   features.7 │ torch.float32 │          0 │ [1, 384, 13, 13] │     False │
│   features.8 │ torch.float32 │     884992 │ [1, 256, 13, 13] │      True │
│   features.9 │ torch.float32 │          0 │ [1, 256, 13, 13] │     False │
│  features.10 │ torch.float32 │     590080 │ [1, 256, 13, 13] │      True │
│  features.11 │ torch.float32 │          0 │ [1, 256, 13, 13] │     False │
│  features.12 │ torch.float32 │          0 │   [1, 256, 6, 6] │     False │
│      avgpool │ torch.float32 │          0 │   [1, 256, 6, 6] │     False │
│ classifier.0 │ torch.float32 │          0 │        [1, 9216] │     False │
│ classifier.1 │ torch.float32 │   37752832 │        [1, 4096] │      True │
│ classifier.2 │ torch.float32 │          0 │        [1, 4096] │     False │
│ classifier.3 │ torch.float32 │          0 │        [1, 4096] │     False │
│ classifier.4 │ torch.float32 │          0 │        [1, 4096] │     False │
│ classifier.5 │ torch.float32 │          0 │        [1, 4096] │     False │
│ classifier.6 │ torch.float32 │    4097000 │        [1, 1000] │      True │
└──────────────┴───────────────┴────────────┴──────────────────┴───────────┘
```

### Sub-Layer Rewriting

```python
import torch
from torch.fx import GraphModule
from torchvision.models import alexnet

from tailor import Tailor

tailor = Tailor(model=alexnet(), input_shape=(1, 3, 224, 224))
# Remove FC and turn model into feature extractor.
model_without_fc: GraphModule = tailor.delete(layer='classifier.6')
# After re-writing, please recompile.
model_without_fc.recompile()
# make sure fc removed and model produce 4096d vector.
rv = model_without_fc(torch.rand(1, 3, 224, 224))
assert rv.size() == (1, 4096)


# attach a Linear layer at the end of the model.
# for dimensionality reduction.
model_128d = tailor.insert(
    module=torch.nn.Linear(4096, 128),
    name='classifier.7'
)
model_128d.recompile()

rv = model_128d(torch.rand(1, 3, 224, 224))
assert rv.size() == (1, 128)
```

### Quantization

in progress

