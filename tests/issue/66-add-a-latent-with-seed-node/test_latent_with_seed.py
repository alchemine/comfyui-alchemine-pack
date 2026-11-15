import sys
import types

import torch
import pytest


@pytest.fixture
def run(pack):
    model_management = types.ModuleType("comfy.model_management")
    model_management.intermediate_device = lambda: torch.device("cpu")
    model_management.intermediate_dtype = lambda: torch.float32
    sys.modules["comfy.model_management"] = model_management
    sys.modules["comfy"].model_management = model_management
    node = pack("latent").LatentWithSeed

    def run(**kw):
        args = dict(
            seed=0,
            dimensions="1024 x 1024 (1:1)",
            invert=False,
            batch_size=1,
            scale_by=1.0,
        )
        args.update(kw)
        return node.execute(**args)

    return run


@pytest.mark.parametrize(
    "dimensions, invert, scale_by, shape",
    [
        ("512 x 512 (1:1)", False, 1.0, (64, 64)),
        ("1152 x 896 (1.286:1)", False, 1.0, (112, 144)),
        ("1152 x 896 (1.286:1)", True, 1.0, (144, 112)),
        ("1152 x 896 (1.286:1)", True, 1.5, (216, 168)),
        ("1152 x 896 (1.286:1)", False, 1.1, (123, 158)),
        ("1216 x 832 (1.46:1)", False, 1.3, (135, 198)),
        ("1536 x 640 (2.4:1)", True, 0.5, (96, 40)),
        ("1024 x 576 (1.778:1)", False, 1.0, (72, 128)),
    ],
)
def test_latent_size_follows_dimensions_invert_and_scale(
    run, dimensions, invert, scale_by, shape
):
    latents, _ = run(dimensions=dimensions, invert=invert, scale_by=scale_by)
    assert latents[0]["samples"].shape == (1, 4, *shape)


def test_each_item_is_one_image_with_its_own_seed_and_batch_index(run):
    latents, seeds = run(seed=885647646956553, batch_size=3)
    assert seeds == [885647646956553, 885647646956554, 885647646956555]
    assert [x["batch_index"] for x in latents] == [[0], [1], [2]]
    assert all(x["samples"].shape[0] == 1 and not x["samples"].any() for x in latents)


def test_batch_size_one_gives_one_item(run):
    latents, seeds = run(seed=7, batch_size=1)
    assert len(latents) == 1 and seeds == [7]
