import pytest
import torchvision.models as models

from tailor import Tailor


@pytest.fixture(
    params=[
        models.resnet18(),
        models.vgg16(),
        models.efficientnet_b0(),
        models.vit_b_16(),
    ]
)
def representative_model(request):
    return request.param


def test_assert_shape_is_expected(representative_model):
    tailor = Tailor()
    summaries = tailor._interpret(representative_model, input_shape=(1, 3, 224, 224))
    assert summaries[-1]['trainable'] == True
    assert summaries[-1]['shape'] == [1, 1000]


def test_assert_layer_trainable_given_num_params_greater_than_zero(
    representative_model,
):
    tailor = Tailor()
    summaries = tailor._interpret(representative_model, input_shape=(1, 3, 224, 224))
    for summary in summaries:
        if summary['trainable'] == True:
            assert summary['num_params'] > 0


def test_visualize_not_fail(representative_model):
    tailor = Tailor()
    tailor.plot(representative_model, input_shape=(1, 3, 224, 224))
