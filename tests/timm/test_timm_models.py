import pytest
import timm

from tailor import Tailor

models = timm.list_models(exclude_filters=['bat_resnext26ts'])[:30]


@pytest.fixture(
    params=models,
)
def timm_model_name(request):
    return request.param


def test_visualize_not_fail(timm_model_name):
    model_shape = timm_model_name.rsplit('_')[-1]
    if model_shape.isdigit():
        input_shape = (1, 3, int(model_shape), int(model_shape))
    else:
        input_shape = (1, 3, 224, 224)
    timm_model = timm.create_model(timm_model_name)
    tailor = Tailor(model=timm_model)
    tailor.plot(input_shape=input_shape)
