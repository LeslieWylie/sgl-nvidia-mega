# SPDX-License-Identifier: Apache-2.0
"""CPU launch-policy checks; actual host bodies run with finite CUDA stubs.

The compilation cache must separate cooperative policies, and non-cooperative
launches must still use the real compiled-kernel residency guard and PDL.
"""

import ast
import functools
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

PAYLOAD = Path(__file__).resolve().parents[2] / "sgl_nvidia_mega/kimi_k3/cutedsl"


def load_body(relative, names, scope, *, keep_cache=False):
    body = ast.parse((PAYLOAD / relative).read_text()).body
    for name in names:
        node = next(item for item in body if getattr(item, "name", None) == name)
        body = node.body
    if not keep_cache:
        node.decorator_list = []
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]), str(PAYLOAD / relative), "exec"
        ),
        scope,
    )
    return scope[node.name]


def test_compilation_cache_keeps_cooperative_launch_policies_separate():
    policies = []

    def kernel(*args, cooperative):
        policies.append(cooperative)
        return SimpleNamespace(
            grouped=False,
            paired=False,
            w13_scale_bytes=1,
            w2_scale_bytes=1,
            workspace_bytes=1,
        )

    scope = dict(
        functools=functools,
        streamed=lambda batch: False,
        ExpertsKernel=kernel,
        c=SimpleNamespace(Uint8=int, Int32=int, Int64=int, Float32=float, BFloat16=int),
        fake=lambda *args: object(),
        cute=SimpleNamespace(compile=lambda *args: object()),
        torch=SimpleNamespace(
            accelerator=SimpleNamespace(device_index=lambda index: nullcontext())
        ),
        make_fake_stream=lambda: object(),
    )
    compile_experts = load_body(
        "moe/experts_runtime.py", ("compile_experts",), scope, keep_cache=True
    )
    common = (8, 896, 4, 148, 0)
    first = compile_experts(*common, cooperative=False)
    assert compile_experts(*common, cooperative=False) is first
    legacy = compile_experts(*common, cooperative=True)
    assert compile_experts(*common, cooperative=True) is legacy
    assert first is not legacy and policies == [False, True]


@pytest.mark.parametrize("cooperative", [False, True])
def test_compiled_residency_guard_cannot_be_skipped_by_launch_policy(cooperative):
    compiled = object()
    compile_experts = Mock(return_value=compiled)
    residency = Mock(side_effect=RuntimeError("not co-resident"))
    prepare = load_body(
        "moe/experts_runtime.py",
        ("PersistentExperts", "prepare"),
        dict(
            pregrouped=lambda tp, batch: False,
            streamed=lambda batch: False,
            stealing=lambda batch: False,
            compile_experts=compile_experts,
            check_residency=residency,
        ),
    )
    owner = SimpleNamespace(
        tp=8,
        experts=896,
        cooperative=cooperative,
        device=SimpleNamespace(index=0),
        grid_for=lambda batch: 148,
    )
    with pytest.raises(RuntimeError, match="not co-resident"):
        prepare(owner, 37, {}, object())
    assert compile_experts.call_args.kwargs == {"cooperative": cooperative}
    residency.assert_called_once_with(compiled, 0, 148, 384, "Kimi K3 MoE experts")


def test_noncooperative_expert_entry_retains_programmatic_launch_dependency():
    # Exercise both actual entry bodies, not a string check on the launch flags.
    launch = load_body("common/primitives.py", ("launch",), {})
    run = load_body("moe/experts.py", ("ExpertsKernel", "__call__"), {"launch": launch})
    device_launch = Mock()
    owner = SimpleNamespace(
        cooperative=False,
        grid=148,
        kernel=Mock(return_value=SimpleNamespace(launch=device_launch)),
    )
    stream = object()
    run(owner, "input", stream)
    owner.kernel.assert_called_once_with(("input",))
    assert device_launch.call_args.kwargs == dict(
        grid=(148, 1, 1),
        block=(384, 1, 1),
        cluster=(1, 1, 1),
        stream=stream,
        use_pdl=True,
        cooperative=False,
    )
