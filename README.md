# Tailor

#### 1. Plot the model structure.
```python
from tailor import Tailor
from torchvision.models import alexnet

tailor = Tailor(model=alexnet())
tailor.plot(input_shape=(1, 3, 224, 224))

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

#### Freeze Layers

```python
from tailor import Tailor
from torchvision.models import alexnet

tailor = Tailor(model=alexnet())
tailor.freeze(from_='classifier.4', to='classifier.6')
tailor.plot(input_shape=(1, 3, 224, 224))

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