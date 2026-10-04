import pytest

from infer import create_parser, expand_prompts


def test_single_prompt_expands_without_mutating_original():
    prompts = ["Hello"]
    assert expand_prompts(prompts, 128) == ["Hello"] * 128
    assert prompts == ["Hello"]
    assert expand_prompts(prompts, None) == prompts


def test_explicit_prompt_batch_preserves_order():
    prompts = ["Hello", "World"]
    assert expand_prompts(prompts, None) == prompts
    assert expand_prompts(prompts, 2) == prompts
    with pytest.raises(ValueError, match="match the number of prompts"):
        expand_prompts(prompts, 3)


@pytest.mark.parametrize("batch", [0, -1])
def test_nonpositive_batch_is_rejected(batch):
    with pytest.raises(ValueError, match="must be positive"):
        expand_prompts(["Hello"], batch)


def test_inference_cli_accepts_batch_size():
    args = create_parser().parse_args(["--model", "model", "--prompts", "Hello", "--batch-size", "32"])
    assert len(expand_prompts(args.prompts, args.batch_size)) == 32
