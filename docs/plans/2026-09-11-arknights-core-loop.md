# 明日方舟代肝脚本 · 核心闭环实施计划 (M0–M2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付一个能无人值守自动刷体力到底的自研脚本内核：可替换的截图/点击后端、基于模板匹配的页面识别、语义化动作层，以及可在无模拟器环境下离线回放的测试体系。

**Architecture:** 严格单向数据流。任务层只通过 `Context` 原语与游戏交互，`Context` 之下是 `Vision`（模板匹配 + OCR）与 IO 后端。所有图像在捕获时归一化到 1280×720 逻辑坐标系，因此模板与 ROI 无需缩放。IO 后端可替换为 `FakeDevice`，使全部任务逻辑能离线测试。

**Tech Stack:** Python 3.11+、OpenCV、NumPy、PyYAML、Typer、pytest。OCR 用 PaddleOCR（可选依赖）。ADB 通过 subprocess 直调。

**Spec:** `docs/specs/2026-09-10-arknights-auto-design.md`

## Global Constraints

以下约束适用于**每一个**任务，不再逐条重复。

- Python 3.11+，源码放在 `src/` 布局下，包名 `arknights_auto`。
- 逻辑坐标系固定 **1280×720**；设备分辨率非 16:9 时直接抛错拒绝运行，绝不"尽力适配"。
- 运行时代码不得直接调用 `subprocess` 访问 ADB，唯一入口是 `arknights_auto.io`。
- 任务层与动作层不得自行处理图像，全部通过 `Vision`。
- 页面判定为"不确定"时禁止发出任何点击。
- OCR 只在需要读取数值的页面调用，不得每帧执行。
- 不做 GUI，除标注工具外只提供命令行。
- 游戏美术素材不入库；`assets/templates/` 与 `assets/captures/` 已在 `.gitignore` 中排除。
- 提交信息用中文，格式 `type: 描述`（type 取 feat / fix / test / docs / chore）。

### 对设计文档的两处有意细化

1. **依赖分层**：`paddleocr` 与 `onnxruntime` 体积大（数百 MB），且只有"读理智值"一处需要，因此放进 `[ocr]` 可选依赖组。未安装时，只有涉及 OCR 的功能会给出明确提示，其余功能不受影响。
2. **`Context.tap` 的职责**：设计文档写的是"找到元素 → 点击 → 确认状态已改变"。计划中拆成 `tap()`（找到并点击）与调用方显式 `wait_page()` / `expect_page()`。拆分是为了让"点完之后要等哪个界面"由任务显式表达，而不是藏在动作层里猜。

---

### Task 1: 项目骨架与 CLI 基线

**Files:**
- Create: `pyproject.toml`
- Create: `src/arknights_auto/__init__.py`
- Create: `src/arknights_auto/cli.py`
- Create: `tests/test_cli.py`
- Create: `assets/templates/.gitkeep`
- Create: `assets/captures/.gitkeep`

**Interfaces:**
- Consumes: 无（首个任务）
- Produces: `arknights_auto.__version__: str`；`arknights_auto.cli.app: typer.Typer`；测试用 `CliRunner` 调用 `app`

- [ ] **Step 1: 建目录与占位文件**

```powershell
New-Item -ItemType Directory -Force src/arknights_auto, tests, assets/templates, assets/captures, config, tools | Out-Null
New-Item -ItemType File -Force assets/templates/.gitkeep, assets/captures/.gitkeep | Out-Null
```

- [ ] **Step 2: 写 `pyproject.toml`**

```toml
[project]
name = "arknights-auto"
version = "0.1.0"
description = "明日方舟自动化脚本（自用）"
requires-python = ">=3.11"
dependencies = [
    "opencv-python>=4.9",
    "numpy>=1.26",
    "pyyaml>=6.0",
    "typer>=0.12",
]

[project.optional-dependencies]
ocr = ["onnxruntime>=1.17", "paddleocr>=2.7"]
dev = ["pytest>=8.0"]

[project.scripts]
arknights-auto = "arknights_auto.cli:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/arknights_auto"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
```

- [ ] **Step 3: 写失败测试 `tests/test_cli.py`**

```python
from typer.testing import CliRunner

from arknights_auto import __version__
from arknights_auto.cli import app


def test_version_option_prints_version():
    result = CliRunner().invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_help_shows_chinese_description():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "明日方舟" in result.stdout
```

- [ ] **Step 4: 实现 `__init__.py` 与 `cli.py`**

`src/arknights_auto/__init__.py`：

```python
__version__ = "0.1.0"
```

`src/arknights_auto/cli.py`：

```python
"""命令行入口。"""

import typer

from . import __version__

app = typer.Typer(help="明日方舟代肝脚本", no_args_is_help=True)


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="打印版本号后退出"),
) -> None:
    if version:
        typer.echo(f"arknights-auto {__version__}")
        raise typer.Exit()
```

- [ ] **Step 5: 安装并验证**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -v
```

Expected: `2 passed`

- [ ] **Step 6: 提交**

```bash
git add pyproject.toml src tests assets
git commit -m "feat: 项目骨架与 CLI 基线"
```

---

### Task 2: 帧与坐标系

**Files:**
- Create: `src/arknights_auto/io/__init__.py`
- Create: `src/arknights_auto/io/frame.py`
- Test: `tests/test_frame.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `LOGICAL_WIDTH = 1280`、`LOGICAL_HEIGHT = 720`、`LOGICAL_SIZE = (1280, 720)`
  - `Frame(image: np.ndarray, raw_size: tuple[int,int], ts: float)`，`image` 恒为 1280×720 BGR
  - `make_frame(image, ts) -> Frame`（不支持的宽高比抛 `UnsupportedResolution`）
  - `is_supported_aspect(width, height) -> bool`
  - `logical_to_device(x, y, raw_size) -> tuple[int,int]`
  - `UnsupportedResolution(ValueError)`

- [ ] **Step 1: 写失败测试 `tests/test_frame.py`**

```python
import numpy as np
import pytest

from arknights_auto.io.frame import (
    LOGICAL_SIZE,
    UnsupportedResolution,
    is_supported_aspect,
    logical_to_device,
    make_frame,
)


def test_accepts_16_9_resolutions():
    assert is_supported_aspect(1280, 720) is True
    assert is_supported_aspect(1920, 1080) is True
    assert is_supported_aspect(2560, 1440) is True


def test_rejects_non_16_9_resolutions():
    assert is_supported_aspect(1080, 2340) is False
    assert is_supported_aspect(1280, 800) is False
    assert is_supported_aspect(0, 0) is False


def test_make_frame_passes_through_720p_without_copy():
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame = make_frame(image, ts=1.0)
    assert frame.image.shape[:2] == (720, 1280)
    assert frame.image is image
    assert frame.raw_size == (1280, 720)


def test_make_frame_downscales_1080p_to_logical_size():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    frame = make_frame(image, ts=1.0)
    assert frame.image.shape[:2] == LOGICAL_SIZE[::-1]
    assert frame.raw_size == (1920, 1080)


def test_make_frame_rejects_unsupported_resolution():
    image = np.zeros((800, 1280, 3), dtype=np.uint8)
    with pytest.raises(UnsupportedResolution):
        make_frame(image, ts=1.0)


def test_logical_to_device_scales_coordinates():
    assert logical_to_device(0, 0, (1280, 720)) == (0, 0)
    assert logical_to_device(640, 360, (1280, 720)) == (640, 360)
    assert logical_to_device(640, 360, (1920, 1080)) == (960, 540)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_frame.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.io.frame'`

- [ ] **Step 3: 实现 `src/arknights_auto/io/frame.py`**

```python
"""逻辑坐标系与帧。所有下游模块只认 1280x720 的 Frame。"""

from dataclasses import dataclass

import cv2
import numpy as np

LOGICAL_WIDTH = 1280
LOGICAL_HEIGHT = 720
LOGICAL_SIZE = (LOGICAL_WIDTH, LOGICAL_HEIGHT)

# 允许 1% 的比例误差，吸收模拟器报告的轻微尺寸偏差
ASPECT_TOLERANCE = 0.01


class UnsupportedResolution(ValueError):
    """设备分辨率不是 16:9，拒绝运行。"""


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    raw_size: tuple[int, int]
    ts: float


def is_supported_aspect(width: int, height: int) -> bool:
    if width <= 0 or height <= 0:
        return False
    target = LOGICAL_WIDTH / LOGICAL_HEIGHT
    return abs(width / height - target) <= target * ASPECT_TOLERANCE


def normalize(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    if (width, height) == LOGICAL_SIZE:
        return image
    # 缩小用 INTER_AREA 抗锯齿，放大用 INTER_LINEAR
    interpolation = cv2.INTER_AREA if width > LOGICAL_WIDTH else cv2.INTER_LINEAR
    return cv2.resize(image, LOGICAL_SIZE, interpolation=interpolation)


def logical_to_device(x: int, y: int, raw_size: tuple[int, int]) -> tuple[int, int]:
    scale_x = raw_size[0] / LOGICAL_WIDTH
    scale_y = raw_size[1] / LOGICAL_HEIGHT
    return int(round(x * scale_x)), int(round(y * scale_y))


def make_frame(image: np.ndarray, ts: float) -> Frame:
    height, width = image.shape[:2]
    if not is_supported_aspect(width, height):
        raise UnsupportedResolution(
            f"设备分辨率 {width}x{height} 不是 16:9，拒绝运行。"
            f"请把模拟器分辨率设为 {LOGICAL_WIDTH}x{LOGICAL_HEIGHT}。"
        )
    return Frame(image=normalize(image), raw_size=(width, height), ts=ts)
```

`src/arknights_auto/io/__init__.py` 暂时留空。

- [ ] **Step 4: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_frame.py -v`
Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
git add src/arknights_auto/io tests/test_frame.py
git commit -m "feat: 逻辑坐标系与帧归一化"
```

---

### Task 3: 元素与页面配置加载

**Files:**
- Create: `src/arknights_auto/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `LOGICAL_WIDTH`、`LOGICAL_HEIGHT`（Task 2）
- Produces:
  - `ElementSpec(name, roi, kind, threshold, template, scale)`，ROI 顺序为 `(x, y, width, height)`
  - `PageSpec(name, anchors, min_score, margin, min_hits)`
  - `load_elements(path: Path) -> dict[str, ElementSpec]`
  - `load_pages(path: Path, elements: Mapping[str, ElementSpec]) -> dict[str, PageSpec]`
  - `ConfigError(ValueError)`
  - `validate_roi(name: str, roi: object) -> tuple[int,int,int,int]`（公开函数，Task 13 复用）

- [ ] **Step 1: 写失败测试 `tests/test_config.py`**

```python
from pathlib import Path

import pytest
import yaml

from arknights_auto.config import ConfigError, load_elements, load_pages


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_load_elements_applies_defaults(tmp_path):
    path = _write(
        tmp_path,
        "elements.yaml",
        {
            "elements": {
                "main.terminal": {"template": "t.png", "roi": [100, 200, 40, 30]},
                "sanity.value": {"kind": "ocr_int", "roi": [10, 10, 80, 20]},
            }
        },
    )
    elements = load_elements(path)
    assert elements["main.terminal"].kind == "template"
    assert elements["main.terminal"].threshold == 0.85
    assert elements["main.terminal"].roi == (100, 200, 40, 30)
    assert elements["sanity.value"].kind == "ocr_int"


def test_load_elements_rejects_roi_out_of_bounds(tmp_path):
    path = _write(
        tmp_path,
        "elements.yaml",
        {"elements": {"bad": {"template": "t.png", "roi": [1200, 700, 200, 200]}}},
    )
    with pytest.raises(ConfigError, match="超出逻辑坐标系"):
        load_elements(path)


def test_load_elements_requires_template_for_template_kind(tmp_path):
    path = _write(tmp_path, "elements.yaml", {"elements": {"bad": {"roi": [0, 0, 10, 10]}}})
    with pytest.raises(ConfigError, match="缺少 template"):
        load_elements(path)


def test_load_pages_validates_anchor_exists(tmp_path):
    elements_path = _write(
        tmp_path,
        "elements.yaml",
        {"elements": {"main.terminal": {"template": "t.png", "roi": [0, 0, 10, 10]}}},
    )
    elements = load_elements(elements_path)
    good = _write(tmp_path, "pages.yaml", {"pages": {"main": {"anchors": ["main.terminal"]}}})
    pages = load_pages(good, elements)
    assert pages["main"].anchors == ("main.terminal",)
    assert pages["main"].min_hits is None

    bad = _write(tmp_path, "pages.yaml", {"pages": {"main": {"anchors": ["nope"]}}})
    with pytest.raises(ConfigError, match="未定义的元素"):
        load_pages(bad, elements)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.config'`

- [ ] **Step 3: 实现 `src/arknights_auto/config.py`**

```python
"""元素与页面配置的加载与校验。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .io.frame import LOGICAL_HEIGHT, LOGICAL_WIDTH

VALID_KINDS = {"template", "ocr_int"}


class ConfigError(ValueError):
    """配置文件不合法。"""


@dataclass(frozen=True)
class ElementSpec:
    name: str
    roi: tuple[int, int, int, int]  # x, y, width, height
    kind: str = "template"
    threshold: float = 0.85
    template: str | None = None
    scale: int = 1


@dataclass(frozen=True)
class PageSpec:
    name: str
    anchors: tuple[str, ...]
    min_score: float = 0.75
    margin: float = 0.05
    min_hits: int | None = None


def validate_roi(name: str, roi: object) -> tuple[int, int, int, int]:
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ConfigError(f"元素 {name} 的 roi 必须是 4 个整数")
    x, y, width, height = (int(v) for v in roi)  # type: ignore[arg-type]
    if width <= 0 or height <= 0:
        raise ConfigError(f"元素 {name} 的 roi 宽高必须为正")
    if x < 0 or y < 0 or x + width > LOGICAL_WIDTH or y + height > LOGICAL_HEIGHT:
        raise ConfigError(
            f"元素 {name} 的 roi {roi} 超出逻辑坐标系 {LOGICAL_WIDTH}x{LOGICAL_HEIGHT}"
        )
    return (x, y, width, height)


def load_elements(path: Path) -> dict[str, ElementSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = raw.get("elements") or {}
    if not isinstance(entries, Mapping):
        raise ConfigError("elements.yaml 的 elements 字段必须是映射")

    result: dict[str, ElementSpec] = {}
    for name, body in entries.items():
        if not isinstance(body, Mapping):
            raise ConfigError(f"元素 {name} 的定义必须是映射")
        kind = str(body.get("kind", "template"))
        if kind not in VALID_KINDS:
            raise ConfigError(f"元素 {name} 的 kind {kind!r} 不支持，可选 {sorted(VALID_KINDS)}")
        template = body.get("template")
        if kind == "template" and not template:
            raise ConfigError(f"元素 {name} 缺少 template")
        result[name] = ElementSpec(
            name=name,
            roi=validate_roi(name, body.get("roi")),
            kind=kind,
            threshold=float(body.get("threshold", 0.85)),
            template=None if template is None else str(template),
            scale=int(body.get("scale", 1)),
        )
    return result


def load_pages(path: Path, elements: Mapping[str, ElementSpec]) -> dict[str, PageSpec]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    entries = raw.get("pages") or {}
    if not isinstance(entries, Mapping):
        raise ConfigError("pages.yaml 的 pages 字段必须是映射")

    result: dict[str, PageSpec] = {}
    for name, body in entries.items():
        if not isinstance(body, Mapping):
            raise ConfigError(f"页面 {name} 的定义必须是映射")
        anchors = tuple(str(anchor) for anchor in (body.get("anchors") or ()))
        if not anchors:
            raise ConfigError(f"页面 {name} 至少要有一个锚点")
        for anchor in anchors:
            if anchor not in elements:
                raise ConfigError(f"页面 {name} 引用了未定义的元素 {anchor}")
        min_hits = body.get("min_hits")
        result[name] = PageSpec(
            name=name,
            anchors=anchors,
            min_score=float(body.get("min_score", 0.75)),
            margin=float(body.get("margin", 0.05)),
            min_hits=None if min_hits is None else int(min_hits),
        )
    return result
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_config.py -v`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add src/arknights_auto/config.py tests/test_config.py
git commit -m "feat: 元素与页面配置加载"
```

---

### Task 4: 模板匹配

**Files:**
- Create: `src/arknights_auto/vision/__init__.py`（本任务先建成空文件）
- Create: `src/arknights_auto/vision/match.py`
- Create: `tests/helpers/__init__.py`（空文件）
- Create: `tests/helpers/synth.py`
- Test: `tests/test_match.py`

**Interfaces:**
- Consumes: `Frame`（Task 2）、`ElementSpec`（Task 3）
- Produces:
  - `Match(element: str, bbox: tuple[int,int,int,int], score: float)`，属性 `.center -> tuple[int,int]`
  - `TemplateMatcher(elements, templates_dir)`，方法 `template(name)`、`find(name, frame)`、`find_all(name, frame, max_results=10)`
  - 测试辅助：`synth.make_patch(seed, size=24)`、`synth.make_frame(width=1280, height=720)`、`synth.stamp(frame, patch, x, y)`

> **重要：测试素材必须是有纹理的图案，不能用纯色块。** `cv2.matchTemplate` 用 `TM_CCOEFF_NORMED`，纯色模板方差为 0，得分无定义。`synth.make_patch` 用固定种子的随机噪声保证方差非零且结果可复现。

- [ ] **Step 1: 写测试辅助 `tests/helpers/synth.py`**

```python
"""用合成图像构造可复现的测试帧，避免依赖真实游戏截图。"""

import numpy as np


def make_patch(seed: int, size: int = 24) -> np.ndarray:
    """生成一块有纹理的方形图案。有纹理是必须的：纯色块在 TM_CCOEFF_NORMED 下无意义。"""
    rng = np.random.default_rng(seed)
    return rng.integers(40, 216, size=(size, size, 3), dtype=np.uint8)


def make_frame(width: int = 1280, height: int = 720) -> np.ndarray:
    """生成一张带轻微底噪的空白帧，底噪让不同模板的相对得分有区分度。"""
    rng = np.random.default_rng(20260911)
    return rng.integers(0, 24, size=(height, width, 3), dtype=np.uint8)


def stamp(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> np.ndarray:
    """把图案贴到帧上，返回新帧（不修改原帧）。"""
    result = frame.copy()
    height, width = patch.shape[:2]
    result[y : y + height, x : x + width] = patch
    return result
```

`tests/helpers/__init__.py` 建成空文件。

- [ ] **Step 2: 写失败测试 `tests/test_match.py`**

```python
from pathlib import Path

import cv2
import numpy as np
import pytest

from arknights_auto.config import ElementSpec
from arknights_auto.io.frame import Frame
from arknights_auto.vision.match import TemplateMatcher
from helpers import synth


def _frame(image: np.ndarray) -> Frame:
    return Frame(image=image, raw_size=(1280, 720), ts=0.0)


@pytest.fixture
def setup(tmp_path: Path):
    patch = synth.make_patch(seed=1)
    assert cv2.imwrite(str(tmp_path / "main.terminal.png"), patch)
    elements = {
        "main.terminal": ElementSpec(
            name="main.terminal",
            roi=(900, 500, 300, 200),
            template="main.terminal.png",
            threshold=0.9,
        )
    }
    matcher = TemplateMatcher(elements, tmp_path)
    frame = _frame(synth.stamp(synth.make_frame(), patch, 1000, 560))
    return matcher, frame, patch


def test_find_locates_element_inside_roi(setup):
    matcher, frame, _ = setup
    match = matcher.find("main.terminal", frame)
    assert match is not None
    assert match.bbox[:2] == (1000, 560)
    assert match.center == (1012, 572)
    assert match.score > 0.99


def test_find_returns_none_when_patch_absent(setup):
    matcher, _, _ = setup
    assert matcher.find("main.terminal", _frame(synth.make_frame())) is None


def test_find_returns_none_when_patch_lies_outside_roi(setup):
    matcher, _, patch = setup
    frame = _frame(synth.stamp(synth.make_frame(), patch, 100, 100))
    assert matcher.find("main.terminal", frame) is None


def test_find_all_finds_every_occurrence(setup):
    matcher, _, patch = setup
    image = synth.make_frame()
    for x in (910, 1000, 1100):
        image = synth.stamp(image, patch, x, 560)
    matches = matcher.find_all("main.terminal", _frame(image))
    assert [m.bbox[0] for m in matches] == [910, 1000, 1100]
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_match.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.vision.match'`

- [ ] **Step 4: 实现 `src/arknights_auto/vision/match.py`**

```python
"""基于固定 ROI 的模板匹配。全屏搜索既慢又容易误匹配，因此 ROI 是强制项。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..config import ElementSpec
from ..io.frame import Frame


@dataclass(frozen=True)
class Match:
    element: str
    bbox: tuple[int, int, int, int]  # x, y, width, height
    score: float

    @property
    def center(self) -> tuple[int, int]:
        x, y, width, height = self.bbox
        return x + width // 2, y + height // 2


class TemplateMatcher:
    def __init__(self, elements: Mapping[str, ElementSpec], templates_dir: Path) -> None:
        self._elements = dict(elements)
        self._templates_dir = Path(templates_dir)
        self._cache: dict[str, np.ndarray] = {}

    def template(self, name: str) -> np.ndarray:
        if name not in self._cache:
            spec = self._elements[name]
            if not spec.template:
                raise KeyError(f"元素 {name} 没有模板文件")
            path = self._templates_dir / spec.template
            template = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if template is None:
                raise FileNotFoundError(f"模板文件不存在或无法读取: {path}")
            self._cache[name] = template
        return self._cache[name]

    def find(self, name: str, frame: Frame) -> Match | None:
        matches = self._search(name, frame, limit=1)
        return matches[0] if matches else None

    def find_all(self, name: str, frame: Frame, max_results: int = 10) -> list[Match]:
        return self._search(name, frame, limit=max_results)

    def _search(self, name: str, frame: Frame, limit: int) -> list[Match]:
        spec = self._elements[name]
        template = self.template(name)
        x, y, width, height = spec.roi
        region = frame.image[y : y + height, x : x + width]
        template_height, template_width = template.shape[:2]
        if region.shape[0] < template_height or region.shape[1] < template_width:
            return []

        result = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
        found: list[Match] = []
        working = result.copy()
        for _ in range(limit):
            _, max_value, _, max_location = cv2.minMaxLoc(working)
            if max_value < spec.threshold:
                break
            found.append(
                Match(
                    element=name,
                    bbox=(
                        x + max_location[0],
                        y + max_location[1],
                        template_width,
                        template_height,
                    ),
                    score=float(max_value),
                )
            )
            # 抑制邻域，避免同一个目标被反复检出
            x0 = max(0, max_location[0] - template_width // 2)
            y0 = max(0, max_location[1] - template_height // 2)
            x1 = min(working.shape[1], max_location[0] + template_width // 2)
            y1 = min(working.shape[0], max_location[1] + template_height // 2)
            working[y0:y1, x0:x1] = -1.0
        return found
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_match.py -v`
Expected: `4 passed`

若 `test_find_all_finds_every_occurrence` 失败，检查抑制范围：三个图案中心相距 90px，抑制区域必须比模板本身更宽。

- [ ] **Step 6: 提交**

```bash
git add src/arknights_auto/vision tests/helpers tests/test_match.py
git commit -m "feat: 固定 ROI 模板匹配"
```

---

### Task 5: 页面分类器

**Files:**
- Create: `src/arknights_auto/vision/page.py`
- Test: `tests/test_page.py`

**Interfaces:**
- Consumes: `PageSpec`、`ElementSpec`（Task 3），`TemplateMatcher`（Task 4）
- Produces:
  - `PageResult(page: str | None, score: float, runner_up: str | None, certain: bool)`
  - `PageDetector(elements, pages, matcher)`，方法 `detect(frame) -> PageResult`、`required_hits(spec) -> int`
  - `certain` 为真当且仅当：存在达标页面，其平均分 ≥ `min_score`，且与次高分差距 ≥ `margin`

- [ ] **Step 1: 写失败测试 `tests/test_page.py`**

```python
from pathlib import Path

import cv2
import pytest

from arknights_auto.config import ElementSpec, PageSpec
from arknights_auto.io.frame import Frame
from arknights_auto.vision.match import TemplateMatcher
from arknights_auto.vision.page import PageDetector
from helpers import synth


def _frame(image) -> Frame:
    return Frame(image=image, raw_size=(1280, 720), ts=0.0)


@pytest.fixture
def detector(tmp_path: Path):
    patches = {"a": synth.make_patch(1), "b": synth.make_patch(2), "c": synth.make_patch(3)}
    for name, patch in patches.items():
        assert cv2.imwrite(str(tmp_path / f"{name}.png"), patch)
    elements = {
        "main.a": ElementSpec("main.a", (900, 100, 200, 100), template="a.png", threshold=0.9),
        "main.b": ElementSpec("main.b", (900, 300, 200, 100), template="b.png", threshold=0.9),
        "level.c": ElementSpec("level.c", (900, 500, 200, 100), template="c.png", threshold=0.9),
    }
    pages = {
        "main": PageSpec("main", ("main.a", "main.b"), min_score=0.75, margin=0.05),
        "level": PageSpec("level", ("level.c",), min_score=0.75, margin=0.05),
    }
    return PageDetector(elements, pages, TemplateMatcher(elements, tmp_path)), patches


def test_detects_page_when_all_anchors_present(detector):
    page_detector, patches = detector
    image = synth.make_frame()
    image = synth.stamp(image, patches["a"], 950, 130)
    image = synth.stamp(image, patches["b"], 950, 330)
    result = page_detector.detect(_frame(image))
    assert result.page == "main"
    assert result.certain is True
    assert result.score > 0.9


def test_uncertain_when_nothing_matches(detector):
    page_detector, _ = detector
    result = page_detector.detect(_frame(synth.make_frame()))
    assert result.page is None
    assert result.certain is False


def test_uncertain_when_two_pages_score_close(detector):
    page_detector, patches = detector
    image = synth.make_frame()
    image = synth.stamp(image, patches["a"], 950, 130)
    image = synth.stamp(image, patches["b"], 950, 330)
    image = synth.stamp(image, patches["c"], 950, 530)
    result = page_detector.detect(_frame(image))
    assert result.certain is False
    assert result.page is None
    assert result.runner_up in {"main", "level"}


def test_page_with_too_few_anchor_hits_is_not_eligible(detector):
    page_detector, patches = detector
    frame = _frame(synth.stamp(synth.make_frame(), patches["a"], 950, 130))
    result = page_detector.detect(frame)
    assert result.certain is False


def test_required_hits_defaults_to_sixty_percent(detector):
    page_detector, _ = detector
    assert page_detector.required_hits(PageSpec("x", ("a", "b", "c", "d", "e"))) == 3
    assert page_detector.required_hits(PageSpec("x", ("a",), min_hits=1)) == 1
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_page.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.vision.page'`

- [ ] **Step 3: 实现 `src/arknights_auto/vision/page.py`**

```python
"""页面分类器：靠固定 ROI 内的锚点模板判定当前处于哪个界面。

核心约定：certain 为 False 时，调用方不得发出任何点击。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from ..config import ElementSpec, PageSpec
from ..io.frame import Frame
from .match import TemplateMatcher


@dataclass(frozen=True)
class PageResult:
    page: str | None
    score: float
    runner_up: str | None
    certain: bool


class PageDetector:
    def __init__(
        self,
        elements: Mapping[str, ElementSpec],
        pages: Mapping[str, PageSpec],
        matcher: TemplateMatcher,
    ) -> None:
        self._elements = dict(elements)
        self._pages = dict(pages)
        self._matcher = matcher

    def required_hits(self, spec: PageSpec) -> int:
        if spec.min_hits is not None:
            return spec.min_hits
        return max(1, math.ceil(len(spec.anchors) * 0.6))

    def detect(self, frame: Frame) -> PageResult:
        scored: list[tuple[str, float, int, int]] = []
        for spec in self._pages.values():
            total = 0.0
            hits = 0
            for anchor in spec.anchors:
                match = self._matcher.find(anchor, frame)
                if match is None:
                    continue
                total += match.score
                hits += 1
            scored.append((spec.name, total / len(spec.anchors), hits, self.required_hits(spec)))

        if not scored:
            return PageResult(page=None, score=0.0, runner_up=None, certain=False)

        ranked = sorted(scored, key=lambda item: item[1], reverse=True)
        eligible = [
            item
            for item in ranked
            if item[2] >= item[3] and item[1] >= self._pages[item[0]].min_score
        ]
        if not eligible:
            return PageResult(page=None, score=ranked[0][1], runner_up=None, certain=False)

        best_name, best_score = eligible[0][0], eligible[0][1]
        others = [item for item in ranked if item[0] != best_name]
        runner_up_name = others[0][0] if others else None
        runner_up_score = others[0][1] if others else 0.0
        certain = best_score - runner_up_score >= self._pages[best_name].margin

        return PageResult(
            page=best_name if certain else None,
            score=best_score,
            runner_up=runner_up_name,
            certain=certain,
        )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_page.py -v`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
git add src/arknights_auto/vision/page.py tests/test_page.py
git commit -m "feat: 页面分类器"
```

---

### Task 6: OCR 接口与理智值读取

**Files:**
- Create: `src/arknights_auto/vision/ocr.py`
- Test: `tests/test_ocr.py`

**Interfaces:**
- Consumes: 无
- Produces:
  - `Ocr` 协议：`read_int(image: np.ndarray) -> int | None`
  - `StubOcr(values: list[int | None])`：按调用顺序返回预设值，记录 `.calls`
  - `PaddleOcr(lang="ch", use_gpu=False)`：懒加载；引擎不可用时抛 `OcrUnavailable`
  - `parse_int(text: str) -> int | None`
  - `preprocess_int_region(image, scale=1) -> np.ndarray`（返回单通道二值图）
  - `OcrUnavailable(RuntimeError)`

- [ ] **Step 1: 写失败测试 `tests/test_ocr.py`**

```python
import numpy as np
import pytest

from arknights_auto.vision.ocr import (
    OcrUnavailable,
    PaddleOcr,
    StubOcr,
    parse_int,
    preprocess_int_region,
)


def test_parse_int_extracts_digits_from_noisy_text():
    assert parse_int(" 135 / 135 ") == 135
    assert parse_int("理智:42") == 42
    assert parse_int("") is None
    assert parse_int("无") is None


def test_preprocess_upscales_and_binarizes_to_single_channel():
    region = np.full((18, 60, 3), 200, dtype=np.uint8)
    result = preprocess_int_region(region, scale=3)
    assert result.shape == (54, 180)


def test_stub_ocr_returns_preset_values_in_order():
    ocr = StubOcr([135, None, 6])
    image = np.zeros((18, 60, 3), dtype=np.uint8)
    assert ocr.read_int(image) == 135
    assert ocr.read_int(image) is None
    assert ocr.read_int(image) == 6
    assert ocr.read_int(image) is None
    assert ocr.calls == 4


def test_paddle_ocr_reports_missing_dependency_clearly():
    ocr = PaddleOcr()
    ocr._import_failed = OcrUnavailable("未安装 paddleocr")
    with pytest.raises(OcrUnavailable, match="paddleocr"):
        ocr.read_int(np.zeros((18, 60, 3), dtype=np.uint8))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ocr.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.vision.ocr'`

- [ ] **Step 3: 实现 `src/arknights_auto/vision/ocr.py`**

```python
"""OCR：只用于读取数字与短文本，绝不每帧调用。"""

from __future__ import annotations

import re
from typing import Protocol

import cv2
import numpy as np


class OcrUnavailable(RuntimeError):
    """OCR 引擎不可用。"""


class Ocr(Protocol):
    def read_int(self, image: np.ndarray) -> int | None: ...


def preprocess_int_region(image: np.ndarray, scale: int = 1) -> np.ndarray:
    """放大、转灰度、二值化。放大是必要的：游戏里的理智数字只有十几像素高。"""
    if scale > 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def parse_int(text: str) -> int | None:
    digits = re.sub(r"\D", "", text or "")
    return int(digits) if digits else None


class StubOcr:
    """测试替身：按调用顺序返回预设数值，用尽后返回 None。"""

    def __init__(self, values: list[int | None]) -> None:
        self._values = list(values)
        self.calls = 0

    def read_int(self, image: np.ndarray) -> int | None:
        self.calls += 1
        if not self._values:
            return None
        return self._values.pop(0)


class PaddleOcr:
    def __init__(self, lang: str = "ch", use_gpu: bool = False) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._engine = None
        self._import_failed: Exception | None = None

    def _ensure_engine(self) -> None:
        if self._engine is not None:
            return
        if self._import_failed is not None:
            raise self._import_failed
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - 取决于可选依赖
            self._import_failed = OcrUnavailable(
                '未安装 paddleocr，请执行: pip install -e ".[ocr]"'
            )
            raise self._import_failed from exc
        self._engine = PaddleOCR(use_angle_cls=False, lang=self._lang, show_log=False)

    def read_int(self, image: np.ndarray) -> int | None:
        self._ensure_engine()
        binary = preprocess_int_region(image, scale=1)
        result = self._engine.ocr(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), cls=False)
        if not result or not result[0]:
            return None
        text = "".join(line[1][0] for line in result[0])
        return parse_int(text)
```

`_ensure_engine` 必须先检查 `_import_failed`，测试正是通过注入该字段验证报错路径的。

- [ ] **Step 4: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_ocr.py -v`
Expected: `4 passed`

- [ ] **Step 5: 提交**

```bash
git add src/arknights_auto/vision/ocr.py tests/test_ocr.py
git commit -m "feat: OCR 读取与降级提示"
```

---

### Task 7: Vision 门面

**Files:**
- Modify: `src/arknights_auto/vision/__init__.py`
- Test: `tests/test_vision.py`

**Interfaces:**
- Consumes: `TemplateMatcher`、`PageDetector`、`Ocr`、`ElementSpec`、`PageSpec`
- Produces: `Vision`
  - `Vision.build(elements, pages, templates_dir, ocr) -> Vision`
  - `find(element, frame) -> Match | None`
  - `find_all(element, frame, max_results=10) -> list[Match]`
  - `page(frame) -> PageResult`
  - `read_int(element, frame) -> int | None`（按 `ElementSpec.scale` 放大后交给 OCR；非 `ocr_int` 元素直接返回 `None` 且不调用 OCR）

- [ ] **Step 1: 写失败测试 `tests/test_vision.py`**

```python
from pathlib import Path

import cv2
import pytest

from arknights_auto.config import ElementSpec, PageSpec
from arknights_auto.io.frame import Frame
from arknights_auto.vision import Vision
from arknights_auto.vision.ocr import StubOcr
from helpers import synth


def _frame(image) -> Frame:
    return Frame(image=image, raw_size=(1280, 720), ts=0.0)


def test_read_int_crops_roi_and_delegates_to_ocr(tmp_path: Path):
    patch = synth.make_patch(7)
    elements = {
        "sanity.value": ElementSpec(
            name="sanity.value", roi=(100, 200, 24, 24), kind="ocr_int", scale=3
        )
    }
    ocr = StubOcr([135])
    vision = Vision.build(elements, {}, tmp_path, ocr)
    frame = _frame(synth.stamp(synth.make_frame(), patch, 100, 200))

    assert vision.read_int("sanity.value", frame) == 135
    assert ocr.calls == 1


def test_read_int_ignores_template_elements(tmp_path: Path):
    elements = {"main.terminal": ElementSpec("main.terminal", (0, 0, 10, 10), template="t.png")}
    ocr = StubOcr([5])
    vision = Vision.build(elements, {}, tmp_path, ocr)
    assert vision.read_int("main.terminal", _frame(synth.make_frame())) is None
    assert ocr.calls == 0


def test_page_delegates_to_detector(tmp_path: Path):
    patch = synth.make_patch(11)
    assert cv2.imwrite(str(tmp_path / "p.png"), patch)
    elements = {
        "main.a": ElementSpec("main.a", (900, 100, 200, 100), template="p.png", threshold=0.9)
    }
    pages = {"main": PageSpec("main", ("main.a",), min_score=0.75, margin=0.05)}
    vision = Vision.build(elements, pages, tmp_path, StubOcr([]))
    frame = _frame(synth.stamp(synth.make_frame(), patch, 950, 130))
    assert vision.page(frame).page == "main"


def test_find_returns_none_for_unknown_element(tmp_path: Path):
    vision = Vision.build({}, {}, tmp_path, StubOcr([]))
    assert vision.find("nope", _frame(synth.make_frame())) is None
    assert vision.find_all("nope", _frame(synth.make_frame())) == []


def test_find_propagates_missing_template_file(tmp_path: Path):
    elements = {"a": ElementSpec("a", (0, 0, 10, 10), template="missing.png")}
    vision = Vision.build(elements, {}, tmp_path, StubOcr([]))
    with pytest.raises(FileNotFoundError, match="missing.png"):
        vision.find("a", _frame(synth.make_frame()))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vision.py -v`
Expected: FAIL，`ImportError: cannot import name 'Vision' from 'arknights_auto.vision'`

- [ ] **Step 3: 实现 `src/arknights_auto/vision/__init__.py`**

```python
"""感知层门面。上层只能通过 Vision 访问图像识别能力。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import cv2
import numpy as np

from ..config import ElementSpec, PageSpec
from ..io.frame import Frame
from .match import Match, TemplateMatcher
from .ocr import Ocr
from .page import PageDetector, PageResult


class Vision:
    def __init__(
        self,
        elements: Mapping[str, ElementSpec],
        matcher: TemplateMatcher,
        detector: PageDetector,
        ocr: Ocr,
    ) -> None:
        self._elements = dict(elements)
        self._matcher = matcher
        self._detector = detector
        self._ocr = ocr

    @classmethod
    def build(
        cls,
        elements: Mapping[str, ElementSpec],
        pages: Mapping[str, PageSpec],
        templates_dir: Path,
        ocr: Ocr,
    ) -> "Vision":
        matcher = TemplateMatcher(elements, templates_dir)
        detector = PageDetector(elements, pages, matcher)
        return cls(elements, matcher, detector, ocr)

    def find(self, element: str, frame: Frame) -> Match | None:
        if element not in self._elements:
            return None
        return self._matcher.find(element, frame)

    def find_all(self, element: str, frame: Frame, max_results: int = 10) -> list[Match]:
        if element not in self._elements:
            return []
        return self._matcher.find_all(element, frame, max_results=max_results)

    def page(self, frame: Frame) -> PageResult:
        return self._detector.detect(frame)

    def read_int(self, element: str, frame: Frame) -> int | None:
        spec = self._elements.get(element)
        if spec is None or spec.kind != "ocr_int":
            return None
        x, y, width, height = spec.roi
        region = frame.image[y : y + height, x : x + width]
        return self._ocr.read_int(self._scale(region, spec.scale))

    @staticmethod
    def _scale(region: np.ndarray, scale: int) -> np.ndarray:
        if scale <= 1:
            return region
        return cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
```

- [ ] **Step 4: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_vision.py -v`
Expected: `5 passed`

- [ ] **Step 5: 提交**

```bash
git add src/arknights_auto/vision/__init__.py tests/test_vision.py
git commit -m "feat: Vision 门面"
```

---

### Task 8: IO 后端（ADB 与 Fake）

**Files:**
- Create: `src/arknights_auto/io/base.py`
- Create: `src/arknights_auto/io/adb.py`
- Create: `src/arknights_auto/io/fake.py`
- Modify: `src/arknights_auto/io/__init__.py`
- Test: `tests/test_adb.py`
- Test: `tests/test_fake_device.py`

**Interfaces:**
- Consumes: `Frame`、`make_frame`、`logical_to_device`（Task 2）
- Produces:
  - `CaptureBackend` 协议：`capture() -> Frame`、`close() -> None`
  - `InputBackend` 协议：`tap(x, y)`、`swipe(x1, y1, x2, y2, duration_ms)`、`back()`
  - `Device` 协议：同时具备以上两者
  - `AdbDevice(serial=None, adb_path="adb", timeout=20.0, raw_size=None)`，方法 `raw_size()`、`capture()`、`tap()`、`swipe()`、`back()`、`close()`
  - `AdbError(RuntimeError)`
  - `FakeDevice(frames)`，属性 `.taps`、`.swipes`、`.backs`、`.captures`
  - 对外接口一律使用**逻辑坐标**，设备坐标换算在后端内部完成

- [ ] **Step 1: 写失败测试 `tests/test_adb.py`**

```python
import cv2
import numpy as np

from arknights_auto.io.adb import AdbDevice


class _Recorder:
    """替身 subprocess.run：记录命令，返回预设字节。"""

    def __init__(self, payload: bytes = b"") -> None:
        self.payload = payload
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))

        class _Result:
            returncode = 0
            stdout = self.payload
            stderr = b""

        return _Result()


def test_tap_converts_logical_to_device_coordinates(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr("arknights_auto.io.adb.subprocess.run", recorder)
    device = AdbDevice(serial="emulator-5554", raw_size=(1920, 1080))
    device.tap(640, 360)
    assert recorder.calls[-1][-3:] == ["input", "tap", "960 540"]


def test_tap_uses_serial_flag(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr("arknights_auto.io.adb.subprocess.run", recorder)
    AdbDevice(serial="emulator-5554", raw_size=(1280, 720)).tap(1, 2)
    assert recorder.calls[-1][:3] == ["adb", "-s", "emulator-5554"]


def test_back_sends_keyevent(monkeypatch):
    recorder = _Recorder()
    monkeypatch.setattr("arknights_auto.io.adb.subprocess.run", recorder)
    AdbDevice(raw_size=(1280, 720)).back()
    assert "KEYCODE_BACK" in " ".join(recorder.calls[-1])


def test_raw_size_parses_wm_size_output(monkeypatch):
    recorder = _Recorder(payload=b"Physical size: 1920x1080\n")
    monkeypatch.setattr("arknights_auto.io.adb.subprocess.run", recorder)
    assert AdbDevice().raw_size() == (1920, 1080)


def test_capture_decodes_png_into_logical_frame(monkeypatch):
    raw = np.zeros((1080, 1920, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", raw)
    assert ok
    recorder = _Recorder(payload=encoded.tobytes())
    monkeypatch.setattr("arknights_auto.io.adb.subprocess.run", recorder)
    frame = AdbDevice(raw_size=(1920, 1080)).capture()
    assert frame.image.shape[:2] == (720, 1280)
    assert frame.raw_size == (1920, 1080)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_adb.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.io.adb'`

- [ ] **Step 3: 实现 `src/arknights_auto/io/base.py`**

```python
"""截图与输入后端协议。所有坐标均为 1280x720 逻辑坐标。"""

from typing import Protocol

from .frame import Frame


class CaptureBackend(Protocol):
    def capture(self) -> Frame: ...
    def close(self) -> None: ...


class InputBackend(Protocol):
    def tap(self, x: int, y: int) -> None: ...
    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None: ...
    def back(self) -> None: ...


class Device(CaptureBackend, InputBackend, Protocol):
    """同时具备截图与输入能力的设备。"""
```

- [ ] **Step 4: 实现 `src/arknights_auto/io/adb.py`**

```python
"""ADB 后端。直调 adb 可执行文件，不引入额外抽象层。"""

from __future__ import annotations

import shutil
import subprocess
import time

import cv2
import numpy as np

from .frame import Frame, logical_to_device, make_frame


class AdbError(RuntimeError):
    """ADB 调用失败。"""


class AdbDevice:
    def __init__(
        self,
        serial: str | None = None,
        adb_path: str = "adb",
        timeout: float = 20.0,
        raw_size: tuple[int, int] | None = None,
    ) -> None:
        self._adb = shutil.which(adb_path) or adb_path
        self._serial = serial
        self._timeout = timeout
        self._raw_size = raw_size

    def _args(self, *args: str) -> list[str]:
        command = [self._adb]
        if self._serial:
            command += ["-s", self._serial]
        return command + list(args)

    def _run(self, *args: str) -> bytes:
        result = subprocess.run(self._args(*args), capture_output=True, timeout=self._timeout)
        if result.returncode != 0:
            message = result.stderr.decode("utf-8", "replace").strip()
            raise AdbError(f"adb {' '.join(args)} 失败: {message}")
        return result.stdout

    def raw_size(self) -> tuple[int, int]:
        """查询设备分辨率（带缓存）。"""
        if self._raw_size is None:
            output = self._run("shell", "wm", "size").decode("utf-8", "replace")
            for line in output.splitlines():
                if "size:" not in line:
                    continue
                token = line.split("size:")[1].strip()
                width, _, height = token.partition("x")
                if width.isdigit() and height.isdigit():
                    self._raw_size = (int(width), int(height))
            if self._raw_size is None:
                raise AdbError(f"无法从 wm size 输出解析分辨率: {output!r}")
        return self._raw_size

    def capture(self) -> Frame:
        payload = self._run("exec-out", "screencap", "-p")
        buffer = np.frombuffer(payload, dtype=np.uint8)
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise AdbError("截图解码失败。可改用 adb shell screencap -p /sdcard/s.png 后 adb pull")
        return make_frame(image, ts=time.monotonic())

    def tap(self, x: int, y: int) -> None:
        device_x, device_y = logical_to_device(x, y, self.raw_size())
        self._run("shell", "input", "tap", f"{device_x} {device_y}")

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
        size = self.raw_size()
        ax, ay = logical_to_device(x1, y1, size)
        bx, by = logical_to_device(x2, y2, size)
        self._run("shell", "input", "swipe", f"{ax} {ay} {bx} {by}", str(duration_ms))

    def back(self) -> None:
        self._run("shell", "input", "keyevent", "KEYCODE_BACK")

    def close(self) -> None:
        return None
```

- [ ] **Step 5: 实现 `src/arknights_auto/io/fake.py`**

```python
"""离线回放用的假设备：喂给脚本预先准备好的帧，记录所有输入。"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .frame import Frame


class FakeDevice:
    def __init__(self, frames: Sequence[Frame] | Callable[[], Frame]) -> None:
        self._frames = frames
        self._index = 0
        self.taps: list[tuple[int, int]] = []
        self.swipes: list[tuple[int, int, int, int, int]] = []
        self.backs = 0
        self.captures = 0

    def capture(self) -> Frame:
        self.captures += 1
        if callable(self._frames):
            return self._frames()
        if not self._frames:
            raise AssertionError("FakeDevice 没有可用的帧")
        index = min(self._index, len(self._frames) - 1)
        frame = self._frames[index]
        self._index += 1
        return frame

    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int) -> None:
        self.swipes.append((x1, y1, x2, y2, duration_ms))

    def back(self) -> None:
        self.backs += 1

    def close(self) -> None:
        return None
```

`src/arknights_auto/io/__init__.py` 只导出不含 ADB 的部分，避免导入期就要求系统存在 adb：

```python
from .base import CaptureBackend, Device, InputBackend
from .fake import FakeDevice
from .frame import LOGICAL_HEIGHT, LOGICAL_SIZE, LOGICAL_WIDTH, Frame, make_frame

__all__ = [
    "CaptureBackend",
    "Device",
    "FakeDevice",
    "Frame",
    "InputBackend",
    "LOGICAL_HEIGHT",
    "LOGICAL_SIZE",
    "LOGICAL_WIDTH",
    "make_frame",
]
```

- [ ] **Step 6: 写 `tests/test_fake_device.py`**

```python
from arknights_auto.io.fake import FakeDevice
from arknights_auto.io.frame import Frame
from helpers import synth


def _frame(ts: float) -> Frame:
    return Frame(image=synth.make_frame(), raw_size=(1280, 720), ts=ts)


def test_replays_frames_then_holds_last():
    device = FakeDevice([_frame(0.0), _frame(1.0)])
    assert device.capture().ts == 0.0
    assert device.capture().ts == 1.0
    assert device.capture().ts == 1.0
    assert device.captures == 3


def test_records_input():
    device = FakeDevice([_frame(0.0)])
    device.tap(10, 20)
    device.swipe(0, 0, 100, 100, 300)
    device.back()
    assert device.taps == [(10, 20)]
    assert device.swipes == [(0, 0, 100, 100, 300)]
    assert device.backs == 1
```

- [ ] **Step 7: 运行 IO 层测试**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_adb.py tests/test_fake_device.py -v`
Expected: `7 passed`

- [ ] **Step 8: 提交**

```bash
git add src/arknights_auto/io tests/test_adb.py tests/test_fake_device.py
git commit -m "feat: ADB 与离线回放后端"
```

---

### Task 9: Context 动作层

**Files:**
- Create: `src/arknights_auto/context.py`
- Test: `tests/test_context.py`

**Interfaces:**
- Consumes: `Device`、`Vision`、`Frame`
- Produces:
  - `ContextError(Exception)`、`ActionError(ContextError)`、`StopRequested(ContextError)`
  - `Context(device, vision, *, clock=time.monotonic, sleeper=time.sleep, logger=print, stop_flag=None, poll_interval=0.4)`
  - 方法：`check_stop()`、`log(message)`、`sleep(seconds)`、`refresh() -> Frame`、`frame() -> Frame`、`page() -> str | None`、`wait_page(page, timeout=10.0) -> Frame`、`expect_page(page)`、`read_int(element) -> int | None`、`tap_xy(x, y)`、`tap(element, timeout=3.0, settle=0.3)`、`swipe(...)`、`back()`
  - `Context.stop_flag: threading.Event`，置位后任何原语都会抛 `StopRequested`

- [ ] **Step 1: 写失败测试 `tests/test_context.py`**

```python
import cv2
import pytest

from arknights_auto.config import ElementSpec, PageSpec
from arknights_auto.context import ActionError, Context, StopRequested
from arknights_auto.io.fake import FakeDevice
from arknights_auto.io.frame import Frame
from arknights_auto.vision import Vision
from arknights_auto.vision.ocr import StubOcr
from helpers import synth


class _Clock:
    """假时钟：sleep 会推进时间，让超时逻辑立即可测。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept += seconds
        self.now += seconds


def _frame(image) -> Frame:
    return Frame(image=image, raw_size=(1280, 720), ts=0.0)


def _build(tmp_path, frames, *, with_template: bool, ocr=None):
    if with_template:
        assert cv2.imwrite(str(tmp_path / "t.png"), synth.make_patch(1))
    elements = {
        "main.terminal": ElementSpec(
            "main.terminal", (900, 100, 200, 100), template="t.png", threshold=0.9
        )
    }
    pages = {"main": PageSpec("main", ("main.terminal",), min_score=0.75, margin=0.05)}
    vision = Vision.build(elements, pages, tmp_path, ocr or StubOcr([]))
    device = FakeDevice(frames)
    clock = _Clock()
    ctx = Context(
        device, vision, clock=clock.monotonic, sleeper=clock.sleep, logger=lambda _m: None
    )
    return ctx, device, clock


def test_page_detects_main(tmp_path):
    image = synth.stamp(synth.make_frame(), synth.make_patch(1), 950, 130)
    ctx, _, _ = _build(tmp_path, [_frame(image)], with_template=True)
    assert ctx.page() == "main"


def test_tap_clicks_element_center(tmp_path):
    image = synth.stamp(synth.make_frame(), synth.make_patch(1), 950, 130)
    ctx, device, _ = _build(tmp_path, [_frame(image)], with_template=True)
    ctx.tap("main.terminal")
    assert device.taps == [(962, 142)]


def test_tap_raises_action_error_when_element_missing(tmp_path):
    ctx, _, clock = _build(tmp_path, [_frame(synth.make_frame())], with_template=True)
    with pytest.raises(ActionError, match="找不到元素 main.terminal"):
        ctx.tap("main.terminal", timeout=1.0)
    assert clock.slept >= 1.0


def test_expect_page_raises_when_page_differs(tmp_path):
    ctx, _, _ = _build(tmp_path, [_frame(synth.make_frame())], with_template=True)
    with pytest.raises(ActionError, match="期望页面 main"):
        ctx.expect_page("main")


def test_wait_page_times_out(tmp_path):
    ctx, _, clock = _build(tmp_path, [_frame(synth.make_frame())], with_template=True)
    with pytest.raises(ActionError, match="等待页面 main 超时"):
        ctx.wait_page("main", timeout=2.0)
    assert clock.slept >= 2.0


def test_stop_flag_interrupts_primitives(tmp_path):
    ctx, _, _ = _build(tmp_path, [_frame(synth.make_frame())], with_template=True)
    ctx.stop_flag.set()
    with pytest.raises(StopRequested):
        ctx.check_stop()
    with pytest.raises(StopRequested):
        ctx.sleep(0.1)
    with pytest.raises(StopRequested):
        ctx.refresh()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_context.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.context'`

- [ ] **Step 3: 实现 `src/arknights_auto/context.py`**

```python
"""动作层：任务层唯一可用的接口。禁止绕过它直接碰设备或图像。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from .io.base import Device
from .io.frame import Frame
from .vision import Vision


class ContextError(Exception):
    """动作层错误基类。"""


class ActionError(ContextError):
    """一次动作无法完成。"""


class StopRequested(ContextError):
    """收到全局停机信号。"""


class Context:
    def __init__(
        self,
        device: Device,
        vision: Vision,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
        logger: Callable[[str], None] = print,
        stop_flag: threading.Event | None = None,
        poll_interval: float = 0.4,
    ) -> None:
        self.device = device
        self.vision = vision
        self.stop_flag = stop_flag if stop_flag is not None else threading.Event()
        self._clock = clock
        self._sleep_fn = sleeper
        self._log = logger
        self._poll_interval = poll_interval
        self._frame: Frame | None = None

    # ---- 基础 ----

    def check_stop(self) -> None:
        if self.stop_flag.is_set():
            raise StopRequested("收到停机信号")

    def log(self, message: str) -> None:
        self._log(message)

    def sleep(self, seconds: float) -> None:
        self.check_stop()
        self._sleep_fn(seconds)

    def refresh(self) -> Frame:
        self.check_stop()
        self._frame = self.device.capture()
        return self._frame

    def frame(self) -> Frame:
        self.check_stop()
        if self._frame is None:
            return self.refresh()
        return self._frame

    # ---- 感知 ----

    def page(self) -> str | None:
        return self.vision.page(self.refresh()).page

    def wait_page(self, page: str, timeout: float = 10.0) -> Frame:
        deadline = self._clock() + timeout
        while True:
            self.check_stop()
            current = self.refresh()
            if self.vision.page(current).page == page:
                return current
            if self._clock() >= deadline:
                raise ActionError(f"等待页面 {page} 超时（{timeout}s）")
            self._sleep_fn(self._poll_interval)

    def expect_page(self, page: str) -> None:
        current = self.page()
        if current != page:
            raise ActionError(f"期望页面 {page}，实际为 {current}")

    def read_int(self, element: str) -> int | None:
        return self.vision.read_int(element, self.refresh())

    # ---- 输入 ----

    def tap_xy(self, x: int, y: int) -> None:
        self.check_stop()
        self.device.tap(x, y)

    def tap(self, element: str, timeout: float = 3.0, settle: float = 0.3) -> None:
        deadline = self._clock() + timeout
        while True:
            self.check_stop()
            match = self.vision.find(element, self.refresh())
            if match is not None:
                x, y = match.center
                self.device.tap(x, y)
                self._log(f"点击 {element} @ ({x}, {y}) 得分 {match.score:.3f}")
                if settle > 0:
                    self._sleep_fn(settle)
                return
            if self._clock() >= deadline:
                raise ActionError(f"找不到元素 {element}（{timeout}s 内）")
            self._sleep_fn(self._poll_interval)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 300) -> None:
        self.check_stop()
        self.device.swipe(x1, y1, x2, y2, duration_ms)

    def back(self) -> None:
        self.check_stop()
        self.device.back()
        self._sleep_fn(0.5)
```

注意：`sleep` 内部必须调用 `self._sleep_fn`，不能调用自身（否则递归）。参数字段命名为 `_sleep_fn` 就是为了避免这个陷阱。

- [ ] **Step 4: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_context.py -v`
Expected: `6 passed`

- [ ] **Step 5: 提交**

```bash
git add src/arknights_auto/context.py tests/test_context.py
git commit -m "feat: Context 动作层"
```

---

### Task 10: 卡死恢复

**Files:**
- Create: `src/arknights_auto/recovery.py`
- Test: `tests/test_recovery.py`

**Interfaces:**
- Consumes: `Context`、`ContextError`（Task 9）
- Produces: `recover_to_home(ctx, *, home_page="main", max_attempts=8) -> bool`
- 恢复顺序：已在主界面 → 返回 `True`；识别到关闭类元素 → 点击；否则按返回键。每轮之间 `sleep(1.0)`。

- [ ] **Step 1: 写失败测试 `tests/test_recovery.py`**

```python
import cv2

from arknights_auto.config import ElementSpec, PageSpec
from arknights_auto.context import Context
from arknights_auto.io.fake import FakeDevice
from arknights_auto.io.frame import Frame
from arknights_auto.recovery import recover_to_home
from arknights_auto.vision import Vision
from arknights_auto.vision.ocr import StubOcr
from helpers import synth


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _elements():
    return {
        "main.a": ElementSpec("main.a", (100, 100, 200, 100), template="a.png", threshold=0.9),
        "common.close": ElementSpec(
            "common.close", (100, 300, 200, 100), template="c.png", threshold=0.9
        ),
    }


def _pages():
    return {"main": PageSpec("main", ("main.a",), min_score=0.75, margin=0.05)}


def _frame(image) -> Frame:
    return Frame(image=image, raw_size=(1280, 720), ts=0.0)


def _context(tmp_path, image):
    clock = _Clock()
    ctx = Context(
        FakeDevice([_frame(image)]),
        Vision.build(_elements(), _pages(), tmp_path, StubOcr([])),
        clock=clock.monotonic,
        sleeper=clock.sleep,
        logger=lambda _m: None,
    )
    return ctx


def test_returns_true_immediately_when_already_home(tmp_path):
    assert cv2.imwrite(str(tmp_path / "a.png"), synth.make_patch(1))
    ctx = _context(tmp_path, synth.stamp(synth.make_frame(), synth.make_patch(1), 150, 130))
    assert recover_to_home(ctx) is True
    assert ctx.device.backs == 0
    assert ctx.device.taps == []


def test_presses_back_when_nothing_recognized(tmp_path):
    assert cv2.imwrite(str(tmp_path / "a.png"), synth.make_patch(1))
    ctx = _context(tmp_path, synth.make_frame())
    assert recover_to_home(ctx, max_attempts=3) is False
    assert ctx.device.backs == 3


def test_clicks_close_button_when_visible(tmp_path):
    assert cv2.imwrite(str(tmp_path / "a.png"), synth.make_patch(1))
    assert cv2.imwrite(str(tmp_path / "c.png"), synth.make_patch(3))
    ctx = _context(tmp_path, synth.stamp(synth.make_frame(), synth.make_patch(3), 150, 330))
    recover_to_home(ctx, max_attempts=2)
    assert set(ctx.device.taps) == {(162, 342)}
    assert ctx.device.backs == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_recovery.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.recovery'`

- [ ] **Step 3: 实现 `src/arknights_auto/recovery.py`**

```python
"""从任意未知界面恢复到主界面。

这是无人值守运行的关键路径：半夜卡在某个弹窗上，等于整晚什么都没干。
"""

from __future__ import annotations

from .context import Context, ContextError

CLOSE_ELEMENTS = ("common.close", "common.confirm")
BACK_ELEMENTS = ("common.back_button",)


def recover_to_home(
    ctx: Context, *, home_page: str = "main", max_attempts: int = 8
) -> bool:
    for _ in range(max_attempts):
        ctx.check_stop()
        if ctx.vision.page(ctx.refresh()).page == home_page:
            return True

        acted = False
        for element in CLOSE_ELEMENTS + BACK_ELEMENTS:
            if ctx.vision.find(element, ctx.frame()) is None:
                continue
            try:
                ctx.tap(element, timeout=0.5)
                acted = True
            except ContextError:
                acted = False
            break
        if not acted:
            try:
                ctx.back()
            except ContextError:
                pass
        ctx.sleep(1.0)

    return ctx.vision.page(ctx.refresh()).page == home_page
```

`ctx.vision.find` 对未定义元素会返回 `None`（Task 7 的行为），因此未配置 `common.close` 时不会抛异常，而是直接走返回键分支。

- [ ] **Step 4: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_recovery.py -v`
Expected: `3 passed`

- [ ] **Step 5: 提交**

```bash
git add src/arknights_auto/recovery.py tests/test_recovery.py
git commit -m "feat: 卡死恢复流程"
```

---

### Task 11: 任务基类与调度器

**Files:**
- Create: `src/arknights_auto/tasks/__init__.py`
- Create: `src/arknights_auto/tasks/base.py`
- Create: `src/arknights_auto/scheduler.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `Context`、`ContextError`、`StopRequested`、`recover_to_home`
- Produces:
  - `TaskResult(success: bool, reason: str, rounds: int = 0)`
  - `Task(Protocol)`：属性 `name: str`，方法 `run(ctx) -> TaskResult`
  - `TaskError(Exception)`
  - `TaskOutcome(name, success, reason, rounds, attempts)`
  - `RunSummary(outcomes)`，属性 `.ok`，方法 `.render() -> str`
  - `Scheduler(ctx, tasks, *, retries=2, recover=None)`，方法 `run(names) -> RunSummary`

- [ ] **Step 1: 写失败测试 `tests/test_scheduler.py`**

```python
from arknights_auto.scheduler import Scheduler
from arknights_auto.tasks.base import TaskResult


class _Task:
    def __init__(self, name, results):
        self.name = name
        self._results = list(results)
        self.calls = 0

    def run(self, ctx):
        self.calls += 1
        return self._results.pop(0) if self._results else TaskResult(True, "完成")


class _Ctx:
    def __init__(self):
        self.logs = []

    def log(self, message):
        self.logs.append(message)


def test_runs_tasks_in_order_and_summarizes():
    first = _Task("combat", [TaskResult(True, "理智耗尽", rounds=3)])
    second = _Task("daily", [TaskResult(True, "完成")])
    scheduler = Scheduler(_Ctx(), {"combat": first, "daily": second}, recover=lambda _c: True)
    summary = scheduler.run(["combat", "daily"])
    assert summary.ok is True
    assert [o.name for o in summary.outcomes] == ["combat", "daily"]
    assert summary.outcomes[0].rounds == 3
    assert "理智耗尽" in summary.render()


def test_retries_failed_task_before_giving_up():
    flaky = _Task("combat", [TaskResult(False, "卡住"), TaskResult(True, "理智耗尽", rounds=1)])
    scheduler = Scheduler(_Ctx(), {"combat": flaky}, retries=1, recover=lambda _c: True)
    summary = scheduler.run(["combat"])
    assert flaky.calls == 2
    assert summary.outcomes[0].success is True
    assert summary.outcomes[0].attempts == 2


def test_gives_up_after_retries_and_still_runs_next_task():
    broken = _Task("combat", [TaskResult(False, "卡住")] * 3)
    other = _Task("daily", [TaskResult(True, "完成")])
    scheduler = Scheduler(
        _Ctx(), {"combat": broken, "daily": other}, retries=2, recover=lambda _c: False
    )
    summary = scheduler.run(["combat", "daily"])
    assert summary.ok is False
    assert broken.calls == 3
    assert other.calls == 1


def test_unknown_task_name_raises_key_error():
    scheduler = Scheduler(_Ctx(), {}, recover=lambda _c: True)
    try:
        scheduler.run(["nope"])
    except KeyError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("应当抛出 KeyError")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_scheduler.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.scheduler'`

- [ ] **Step 3: 实现 `src/arknights_auto/tasks/base.py`**

```python
"""任务层契约。每个任务是阻塞式函数，内部自行推进状态。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ..context import Context


@dataclass(frozen=True)
class TaskResult:
    success: bool
    reason: str
    rounds: int = 0


class TaskError(Exception):
    """任务无法继续执行。本计划暂未使用，留给日常杂项任务（第二份计划）。"""


class Task(Protocol):
    name: str

    def run(self, ctx: Context) -> TaskResult: ...
```

`src/arknights_auto/tasks/__init__.py`：

```python
from .base import Task, TaskError, TaskResult

__all__ = ["Task", "TaskError", "TaskResult"]
```

- [ ] **Step 4: 实现 `src/arknights_auto/scheduler.py`**

```python
"""任务调度：串行执行、失败重试、汇总报告。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from .context import Context, ContextError, StopRequested
from .recovery import recover_to_home
from .tasks.base import Task, TaskResult


@dataclass(frozen=True)
class TaskOutcome:
    name: str
    success: bool
    reason: str
    rounds: int
    attempts: int


@dataclass(frozen=True)
class RunSummary:
    outcomes: list[TaskOutcome] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(outcome.success for outcome in self.outcomes)

    def render(self) -> str:
        lines = ["运行摘要："]
        for outcome in self.outcomes:
            status = "成功" if outcome.success else "失败"
            lines.append(
                f"  [{status}] {outcome.name} 轮次={outcome.rounds} "
                f"尝试={outcome.attempts} 原因={outcome.reason}"
            )
        return "\n".join(lines)


class Scheduler:
    def __init__(
        self,
        ctx: Context,
        tasks: Mapping[str, Task],
        *,
        retries: int = 2,
        recover: Callable[[Context], bool] | None = None,
    ) -> None:
        self._ctx = ctx
        self._tasks = dict(tasks)
        self._retries = retries
        self._recover = recover if recover is not None else recover_to_home

    def run(self, names: Sequence[str]) -> RunSummary:
        outcomes: list[TaskOutcome] = []
        for name in names:
            task = self._tasks[name]
            outcomes.append(self._run_one(task))
        return RunSummary(outcomes=outcomes)

    def _run_one(self, task: Task) -> TaskOutcome:
        attempts = 0
        reason = "未执行"
        rounds = 0
        while attempts <= self._retries:
            attempts += 1
            self._ctx.log(f"开始任务 {task.name}（第 {attempts} 次）")
            try:
                result: TaskResult = task.run(self._ctx)
            except StopRequested:
                return TaskOutcome(task.name, False, "收到停机信号", rounds, attempts)
            except ContextError as exc:
                reason = f"动作层错误: {exc}"
                self._ctx.log(f"任务 {task.name} {reason}")
                if not self._recover(self._ctx):
                    self._ctx.log(f"任务 {task.name} 恢复失败")
            else:
                rounds = result.rounds
                if result.success:
                    return TaskOutcome(task.name, True, result.reason, rounds, attempts)
                reason = result.reason
                self._ctx.log(f"任务 {task.name} 失败：{reason}")
                if not self._recover(self._ctx):
                    self._ctx.log(f"任务 {task.name} 恢复失败")
        return TaskOutcome(task.name, False, reason, rounds, attempts)
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_scheduler.py -v`
Expected: `4 passed`

- [ ] **Step 6: 提交**

```bash
git add src/arknights_auto/tasks src/arknights_auto/scheduler.py tests/test_scheduler.py
git commit -m "feat: 任务调度与运行摘要"
```

---

### Task 12: 刷体力任务

**Files:**
- Create: `src/arknights_auto/tasks/combat.py`
- Create: `tests/helpers/scripted.py`
- Test: `tests/test_combat.py`

**Interfaces:**
- Consumes: `Context`、`TaskResult`、`recover_to_home`
- Produces:
  - `CombatTask(stage: str, sanity_cost: int, max_rounds: int = 100)`，`name = "combat"`
  - 页面常量：`HOME = "main"`、`LEVEL_SELECT = "level_select"`、`COMBAT_CONFIRM = "combat_confirm"`、`COMBAT_RUNNING = "combat_running"`、`COMBAT_RESULT = "combat_result"`
  - 测试辅助 `ScriptedDevice`

> **注意**：`ocr_int` 类元素不能当页面锚点——它们没有模板，`TemplateMatcher.template()` 会抛 `KeyError`。页面锚点必须是 `template` 类元素。

- [ ] **Step 1: 写测试辅助 `tests/helpers/scripted.py`**

```python
"""按状态机回放界面的假设备：点击某个元素会触发页面跳转。

这让刷体力任务能在完全没有模拟器的情况下端到端测试。
"""

from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np

from arknights_auto.io.frame import Frame


class ScriptedDevice:
    def __init__(
        self,
        pages: Mapping[str, np.ndarray],
        start: str,
        transitions: Mapping[tuple[str, str], str],
        element_boxes: Mapping[str, tuple[int, int, int, int]],
        auto_advance: Mapping[str, tuple[str, int]] | None = None,
    ) -> None:
        self._pages = dict(pages)
        self._current = start
        self._transitions = dict(transitions)
        self._boxes = dict(element_boxes)
        # 形如 {"combat_running": ("combat_result", 2)}：在该页面被抓取 N 次后自动跳转
        self._auto = dict(auto_advance or {})
        self._dwell = 0
        self.taps: list[tuple[int, int]] = []
        self.backs = 0
        self.visits: list[str] = [start]

    @property
    def current(self) -> str:
        return self._current

    def capture(self) -> Frame:
        self._dwell += 1
        rule = self._auto.get(self._current)
        if rule is not None:
            target, after = rule
            if self._dwell >= after:
                self._go(target)
        return Frame(
            image=self._pages[self._current], raw_size=(1280, 720), ts=time.monotonic()
        )

    def tap(self, x: int, y: int) -> None:
        self.taps.append((x, y))
        for element, (bx, by, width, height) in self._boxes.items():
            if bx <= x < bx + width and by <= y < by + height:
                target = self._transitions.get((self._current, element))
                if target is not None:
                    self._go(target)
                return

    def swipe(self, x1, y1, x2, y2, duration_ms) -> None:
        return None

    def back(self) -> None:
        self.backs += 1

    def close(self) -> None:
        return None

    def force(self, page: str, image: np.ndarray | None = None) -> None:
        """测试用：强行跳到某页，可同时替换该页画面并取消其自动跳转。"""
        if image is not None:
            self._pages[page] = image
        self._auto.pop(page, None)
        self._go(page)

    def _go(self, target: str) -> None:
        self._current = target
        self._dwell = 0
        self.visits.append(target)
```

- [ ] **Step 2: 写失败测试 `tests/test_combat.py`**

```python
from pathlib import Path

import cv2
import pytest

from arknights_auto.config import ElementSpec, PageSpec
from arknights_auto.context import Context, StopRequested
from arknights_auto.tasks.combat import CombatTask
from arknights_auto.vision import Vision
from arknights_auto.vision.ocr import StubOcr
from helpers import synth
from helpers.scripted import ScriptedDevice

# 元素名 -> (左上角坐标, 图案种子)
ELEMENTS = {
    "main.terminal_entry": ((1000, 560), 1),
    "stage.stage_1_7": ((200, 300), 2),
    "combat.start_button": ((900, 600), 4),
    "combat.in_progress": ((600, 600), 5),
    "result.confirm": ((800, 600), 6),
}
SANITY_ROI = (40, 20, 24, 24)

PAGE_SPECS = {
    "main": PageSpec("main", ("main.terminal_entry",), min_score=0.75, margin=0.05),
    "level_select": PageSpec("level_select", ("stage.stage_1_7",), min_score=0.75, margin=0.05),
    "combat_confirm": PageSpec("combat_confirm", ("combat.start_button",), min_score=0.75, margin=0.05),
    "combat_running": PageSpec("combat_running", ("combat.in_progress",), min_score=0.75, margin=0.05),
    "combat_result": PageSpec("combat_result", ("result.confirm",), min_score=0.75, margin=0.05),
}


def _build(tmp_path: Path, sanity_values):
    elements = {}
    boxes = {}
    for name, ((x, y), seed) in ELEMENTS.items():
        assert cv2.imwrite(str(tmp_path / f"{name}.png"), synth.make_patch(seed))
        elements[name] = ElementSpec(name, (x, y, 24, 24), template=f"{name}.png", threshold=0.9)
        boxes[name] = (x, y, 24, 24)
    elements["sanity.value"] = ElementSpec(
        "sanity.value", SANITY_ROI, kind="ocr_int", scale=1
    )

    def render(names):
        image = synth.make_frame()
        for name in names:
            (x, y), seed = ELEMENTS[name]
            image = synth.stamp(image, synth.make_patch(seed), x, y)
        return image

    pages = {
        "main": render(["main.terminal_entry"]),
        "level_select": render(["stage.stage_1_7"]),
        "combat_confirm": render(["combat.start_button"]),
        "combat_running": render(["combat.in_progress"]),
        "combat_result": render(["result.confirm"]),
    }
    transitions = {
        ("main", "main.terminal_entry"): "level_select",
        ("level_select", "stage.stage_1_7"): "combat_confirm",
        ("combat_confirm", "combat.start_button"): "combat_running",
        ("combat_result", "result.confirm"): "level_select",
    }
    device = ScriptedDevice(
        pages,
        start="main",
        transitions=transitions,
        element_boxes=boxes,
        auto_advance={"combat_running": ("combat_result", 2)},
    )
    vision = Vision.build(elements, PAGE_SPECS, tmp_path, StubOcr(sanity_values))
    ctx = Context(device, vision, logger=lambda _m: None, poll_interval=0.0)
    return ctx, device


def test_runs_one_round_then_stops_when_sanity_low(tmp_path):
    ctx, device = _build(tmp_path, [135, 5])
    result = CombatTask(stage="stage_1_7", sanity_cost=6, max_rounds=5).run(ctx)
    assert result.success is True
    assert result.reason == "理智耗尽"
    assert result.rounds == 1
    assert device.visits.count("combat_result") == 1


def test_stops_at_max_rounds(tmp_path):
    ctx, _ = _build(tmp_path, [135] * 20)
    result = CombatTask(stage="stage_1_7", sanity_cost=6, max_rounds=2).run(ctx)
    assert result.reason == "达到局数上限"
    assert result.rounds == 2


def test_aborts_on_stop_signal(tmp_path):
    ctx, _ = _build(tmp_path, [135] * 20)
    ctx.stop_flag.set()
    with pytest.raises(StopRequested):
        CombatTask(stage="stage_1_7", sanity_cost=6, max_rounds=2).run(ctx)


def test_unknown_page_triggers_recovery(tmp_path):
    ctx, device = _build(tmp_path, [135] * 20)
    device.force("combat_running", image=synth.make_frame())
    result = CombatTask(stage="stage_1_7", sanity_cost=6, max_rounds=1).run(ctx)
    assert result.success is False
    assert "无法回到主界面" in result.reason or "卡在未知页面" in result.reason
```

- [ ] **Step 3: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_combat.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.tasks.combat'`

- [ ] **Step 4: 实现 `src/arknights_auto/tasks/combat.py`**

```python
"""刷体力：选关 → 作战 → 结算 → 循环，直到理智不足。"""

from __future__ import annotations

from ..context import Context
from ..recovery import recover_to_home
from .base import TaskResult

HOME = "main"
LEVEL_SELECT = "level_select"
COMBAT_CONFIRM = "combat_confirm"
COMBAT_RUNNING = "combat_running"
COMBAT_RESULT = "combat_result"

# 单局战斗最长等待时间。剿灭这类长关卡需要调大。
BATTLE_TIMEOUT = 600.0


class CombatTask:
    name = "combat"

    def __init__(self, stage: str, sanity_cost: int, max_rounds: int = 100) -> None:
        self.stage = stage
        self.sanity_cost = sanity_cost
        self.max_rounds = max_rounds

    def run(self, ctx: Context) -> TaskResult:
        rounds = 0
        ctx.log(f"开始刷体力：{self.stage}，单局消耗 {self.sanity_cost}")
        if not recover_to_home(ctx):
            return TaskResult(False, "无法回到主界面")

        while rounds < self.max_rounds:
            ctx.check_stop()
            page = ctx.page()
            ctx.log(f"当前页面：{page}")

            if page is None:
                ctx.sleep(0.5)
                continue

            if page == HOME:
                ctx.tap("main.terminal_entry")
                ctx.wait_page(LEVEL_SELECT)
                continue

            if page == LEVEL_SELECT:
                sanity = ctx.read_int("sanity.value")
                if sanity is not None and sanity < self.sanity_cost:
                    ctx.log(f"理智不足（{sanity} < {self.sanity_cost}），结束刷体力")
                    return TaskResult(True, "理智耗尽", rounds)
                ctx.tap(f"stage.{self.stage}")
                ctx.wait_page(COMBAT_CONFIRM)
                continue

            if page == COMBAT_CONFIRM:
                ctx.tap("combat.start_button")
                ctx.wait_page(COMBAT_RUNNING, timeout=30.0)
                continue

            if page == COMBAT_RUNNING:
                ctx.wait_page(COMBAT_RESULT, timeout=BATTLE_TIMEOUT)
                continue

            if page == COMBAT_RESULT:
                ctx.tap("result.confirm")
                rounds += 1
                ctx.log(f"已完成第 {rounds} 局")
                ctx.wait_page(LEVEL_SELECT, timeout=30.0)
                continue

            ctx.log(f"未知页面 {page}，尝试恢复")
            if not recover_to_home(ctx):
                return TaskResult(False, f"卡在未知页面 {page}", rounds)

        return TaskResult(True, "达到局数上限", rounds)
```

注意 `stage` 参数是元素名后缀：`stage="stage_1_7"` 对应元素 `stage.stage_1_7`。

- [ ] **Step 5: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_combat.py -v`
Expected: `4 passed`

若 `test_runs_one_round_then_stops_when_sanity_low` 失败，检查 `StubOcr` 的预设值顺序：任务每轮到关卡选择页读一次理智，第一轮读到 `135`，第二轮读到 `5`，因此 `rounds == 1`。

- [ ] **Step 6: 提交**

```bash
git add src/arknights_auto/tasks/combat.py tests/helpers/scripted.py tests/test_combat.py
git commit -m "feat: 刷体力任务与脚本化回放测试"
```

---

### Task 13: 模板标注工具

**Files:**
- Create: `src/arknights_auto/annotate.py`
- Create: `tools/annotate.py`
- Test: `tests/test_annotate.py`

**Interfaces:**
- Consumes: `validate_roi`、`ConfigError`（Task 3），`normalize`（Task 2）
- Produces:
  - `load_frame_for_annotation(path: Path) -> np.ndarray`（归一化到 1280×720）
  - `crop_and_save(frame_path, rect, out_path) -> Path`
  - `upsert_element(elements_path, name, rect, template, threshold=0.85) -> None`

- [ ] **Step 1: 写失败测试 `tests/test_annotate.py`**

```python
from pathlib import Path

import cv2
import numpy as np
import yaml

from arknights_auto.annotate import crop_and_save, load_frame_for_annotation, upsert_element
from helpers import synth


def test_crop_and_save_writes_only_the_rect(tmp_path: Path):
    source = tmp_path / "frame.png"
    assert cv2.imwrite(str(source), synth.make_frame())
    out = crop_and_save(source, (100, 200, 40, 30), tmp_path / "templates" / "x.png")
    assert out.exists()
    assert cv2.imread(str(out)).shape[:2] == (30, 40)


def test_upsert_element_creates_file_then_appends(tmp_path: Path):
    path = tmp_path / "elements.yaml"
    upsert_element(path, "main.terminal", (1, 2, 3, 4), "main.terminal.png")
    upsert_element(path, "sanity.value", (5, 6, 7, 8), "sanity.value.png", threshold=0.8)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["elements"]["main.terminal"]["roi"] == [1, 2, 3, 4]
    assert data["elements"]["sanity.value"]["threshold"] == 0.8


def test_upsert_element_overwrites_existing_entry(tmp_path: Path):
    path = tmp_path / "elements.yaml"
    upsert_element(path, "main.terminal", (1, 2, 3, 4), "a.png")
    upsert_element(path, "main.terminal", (9, 9, 9, 9), "b.png")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data["elements"]["main.terminal"]["roi"] == [9, 9, 9, 9]
    assert data["elements"]["main.terminal"]["template"] == "b.png"


def test_load_frame_for_annotation_normalizes_1080p(tmp_path: Path):
    path = tmp_path / "big.png"
    assert cv2.imwrite(str(path), np.zeros((1080, 1920, 3), dtype=np.uint8))
    assert load_frame_for_annotation(path).shape[:2] == (720, 1280)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_annotate.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'arknights_auto.annotate'`

- [ ] **Step 3: 实现 `src/arknights_auto/annotate.py`**

```python
"""标注工具的纯逻辑部分。GUI 外壳在 tools/annotate.py。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

from .config import ConfigError, validate_roi
from .io.frame import normalize


def load_frame_for_annotation(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ConfigError(f"无法读取截图: {path}")
    return normalize(image)


def crop_and_save(
    frame_path: Path, rect: tuple[int, int, int, int], out_path: Path
) -> Path:
    x, y, width, height = validate_roi(Path(out_path).stem, rect)
    image = load_frame_for_annotation(frame_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), image[y : y + height, x : x + width]):
        raise ConfigError(f"模板写入失败: {out_path}")
    return out_path


def upsert_element(
    elements_path: Path,
    name: str,
    rect: tuple[int, int, int, int],
    template: str,
    threshold: float = 0.85,
) -> None:
    x, y, width, height = validate_roi(name, rect)
    elements_path = Path(elements_path)
    if elements_path.exists():
        data = yaml.safe_load(elements_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}
    data.setdefault("elements", {})
    data["elements"][name] = {
        "template": template,
        "roi": [x, y, width, height],
        "threshold": threshold,
    }
    elements_path.parent.mkdir(parents=True, exist_ok=True)
    elements_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=True), encoding="utf-8"
    )
```

- [ ] **Step 4: 实现 `tools/annotate.py`（tkinter 外壳）**

```python
"""交互式标注：打开截图，鼠标框选，生成模板并写回 elements.yaml。

用法: python tools/annotate.py --frame assets/captures/main.png --name main.terminal_entry
"""

from __future__ import annotations

import argparse
import sys
import tkinter as tk
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from arknights_auto.annotate import (  # noqa: E402
    crop_and_save,
    load_frame_for_annotation,
    upsert_element,
)

TEMPLATES_DIR = Path("assets/templates")
ELEMENTS_FILE = Path("assets/elements.yaml")


def main() -> int:
    parser = argparse.ArgumentParser(description="模板标注工具")
    parser.add_argument("--frame", required=True, type=Path)
    parser.add_argument("--name", required=True, help="元素名，如 main.terminal_entry")
    parser.add_argument("--threshold", type=float, default=0.85)
    args = parser.parse_args()

    image = load_frame_for_annotation(args.frame)
    root = tk.Tk()
    root.title(f"框选 {args.name}")
    canvas = tk.Canvas(root, width=image.shape[1], height=image.shape[0])
    canvas.pack()
    # PhotoImage 直接吃 PNG 字节，省掉 PIL 依赖
    photo = tk.PhotoImage(data=cv2.imencode(".png", image)[1].tobytes(), format="png")
    canvas.create_image(0, 0, anchor="nw", image=photo)

    selection: list[tuple[int, int, int, int]] = []
    start: list[int] = [0, 0]
    rectangle: list[int] = []

    def on_press(event) -> None:
        start[0], start[1] = event.x, event.y
        rectangle.clear()
        rectangle.append(canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="red"))

    def on_drag(event) -> None:
        if rectangle:
            canvas.coords(rectangle[0], start[0], start[1], event.x, event.y)

    def on_release(event) -> None:
        x0, x1 = sorted((start[0], event.x))
        y0, y1 = sorted((start[1], event.y))
        if x1 - x0 > 3 and y1 - y0 > 3:
            selection.append((x0, y0, x1 - x0, y1 - y0))
        root.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.mainloop()

    if not selection:
        print("未选择有效区域，已取消")
        return 1

    rect = selection[0]
    template_name = f"{args.name}.png"
    crop_and_save(args.frame, rect, TEMPLATES_DIR / template_name)
    upsert_element(ELEMENTS_FILE, args.name, rect, template_name, args.threshold)
    print(f"已写入 {args.name} roi={rect} template={template_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: 运行测试确认通过**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_annotate.py -v`
Expected: `4 passed`

tkinter 部分不做自动化测试——它需要图形界面。它的逻辑全部委托给已测试的 `annotate.py`，外壳只负责取一个矩形。

- [ ] **Step 6: 提交**

```bash
git add src/arknights_auto/annotate.py tools/annotate.py tests/test_annotate.py
git commit -m "feat: 模板标注工具"
```

---

### Task 14: 运行时组装、配置与 CLI 命令

**Files:**
- Create: `src/arknights_auto/runtime.py`
- Create: `config/config.yaml`
- Create: `assets/elements.yaml`
- Create: `assets/pages.yaml`
- Modify: `src/arknights_auto/cli.py`
- Test: `tests/test_cli_commands.py`

**Interfaces:**
- Consumes: 前序全部模块
- Produces:
  - `Settings`：`serial`、`adb_path`、`assets_dir`、`templates_dir`、`elements_file`、`pages_file`、`retries`、`logger`
  - `build_vision(settings) -> Vision`
  - `build_context(settings) -> Context`
  - CLI 命令：`doctor`、`capture`、`crop`、`run`

- [ ] **Step 1: 写默认配置文件**

`assets/elements.yaml`：

```yaml
# 坐标基于 1280x720 逻辑坐标系，roi 顺序为 x, y, width, height
# 用 tools/annotate.py 框选生成，或手工填写
elements: {}
```

`assets/pages.yaml`：

```yaml
# 每个页面 2-5 个锚点，且锚点必须是 template 类元素
pages: {}
```

`config/config.yaml`：

```yaml
device:
  serial: null        # null 表示使用 adb 默认设备；多设备时必须指定
  adb_path: adb

paths:
  assets_dir: assets
  templates_dir: assets/templates
  elements_file: assets/elements.yaml
  pages_file: assets/pages.yaml

combat:
  stage: "stage_1_7"  # 对应 elements 里的 stage.stage_1_7
  sanity_cost: 6
  max_rounds: 100

scheduler:
  retries: 2
```

- [ ] **Step 2: 实现 `src/arknights_auto/runtime.py`**

```python
"""把配置、后端、感知层组装成一个可用的 Context。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import load_elements, load_pages
from .context import Context
from .io.adb import AdbDevice
from .vision import Vision
from .vision.ocr import PaddleOcr


class NullOcr:
    """未启用 OCR 时的占位实现，永远返回 None。"""

    def read_int(self, image: np.ndarray) -> int | None:
        return None


@dataclass(frozen=True)
class Settings:
    serial: str | None = None
    adb_path: str = "adb"
    assets_dir: Path = Path("assets")
    templates_dir: Path = Path("assets/templates")
    elements_file: Path = Path("assets/elements.yaml")
    pages_file: Path = Path("assets/pages.yaml")
    retries: int = 2
    use_ocr: bool = True
    logger: Callable[[str], None] = print


def build_vision(settings: Settings) -> Vision:
    elements = load_elements(settings.elements_file)
    pages = load_pages(settings.pages_file, elements)
    ocr = PaddleOcr() if settings.use_ocr else NullOcr()
    return Vision.build(elements, pages, settings.templates_dir, ocr)


def build_context(settings: Settings) -> Context:
    device = AdbDevice(serial=settings.serial, adb_path=settings.adb_path)
    return Context(device, build_vision(settings), logger=settings.logger)
```

- [ ] **Step 3: 写失败测试 `tests/test_cli_commands.py`**

```python
from pathlib import Path

import cv2
from typer.testing import CliRunner

from arknights_auto.cli import app
from helpers import synth


def test_crop_command_writes_template_and_element(tmp_path: Path):
    frame = tmp_path / "frame.png"
    assert cv2.imwrite(str(frame), synth.make_frame())
    elements = tmp_path / "elements.yaml"
    templates = tmp_path / "templates"

    result = CliRunner().invoke(
        app,
        [
            "crop",
            "--frame", str(frame),
            "--rect", "100,200,40,30",
            "--name", "main.terminal",
            "--elements", str(elements),
            "--templates-dir", str(templates),
        ],
    )
    assert result.exit_code == 0, result.stdout
    assert (templates / "main.terminal.png").exists()
    assert "main.terminal" in elements.read_text(encoding="utf-8")


def test_crop_command_rejects_malformed_rect(tmp_path: Path):
    frame = tmp_path / "frame.png"
    assert cv2.imwrite(str(frame), synth.make_frame())
    result = CliRunner().invoke(
        app,
        [
            "crop",
            "--frame", str(frame),
            "--rect", "abc",
            "--name", "x",
            "--elements", str(tmp_path / "e.yaml"),
            "--templates-dir", str(tmp_path / "t"),
        ],
    )
    assert result.exit_code == 1
    assert "--rect 需要形如" in result.stdout


def test_run_command_rejects_missing_config(tmp_path: Path):
    result = CliRunner().invoke(app, ["run", "--config", str(tmp_path / "nope.yaml")])
    assert result.exit_code == 1
    assert "配置文件不存在" in result.stdout
```

- [ ] **Step 4: 运行测试确认失败**

Run: `.\.venv\Scripts\python.exe -m pytest tests/test_cli_commands.py -v`
Expected: FAIL，`crop` 命令不存在

- [ ] **Step 5: 扩展 `src/arknights_auto/cli.py`**

完整文件如下（替换 Task 1 的版本）：

```python
"""命令行入口。"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import typer
import yaml

from . import __version__
from .annotate import crop_and_save, upsert_element
from .io.adb import AdbDevice, AdbError
from .io.frame import LOGICAL_HEIGHT, LOGICAL_WIDTH

app = typer.Typer(help="明日方舟代肝脚本", no_args_is_help=True)

DEFAULT_ELEMENTS = Path("assets/elements.yaml")
DEFAULT_TEMPLATES = Path("assets/templates")


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(False, "--version", help="打印版本号后退出"),
) -> None:
    if version:
        typer.echo(f"arknights-auto {__version__}")
        raise typer.Exit()


@app.command()
def doctor(
    serial: str | None = typer.Option(None, "--serial", help="ADB 设备序列号"),
) -> None:
    """检查 ADB 连接、分辨率与截图耗时。"""
    device = AdbDevice(serial=serial)
    try:
        width, height = device.raw_size()
    except (AdbError, FileNotFoundError) as exc:
        typer.echo(f"[失败] 无法连接设备: {exc}")
        raise typer.Exit(code=1) from exc

    typer.echo(f"设备分辨率: {width}x{height}")
    if abs(width / height - LOGICAL_WIDTH / LOGICAL_HEIGHT) > 0.01:
        typer.echo(f"[失败] 不是 16:9，请把模拟器设为 {LOGICAL_WIDTH}x{LOGICAL_HEIGHT}")
        raise typer.Exit(code=1)
    if (width, height) != (LOGICAL_WIDTH, LOGICAL_HEIGHT):
        typer.echo(f"[提示] 建议直接把模拟器设为 {LOGICAL_WIDTH}x{LOGICAL_HEIGHT}，避免缩放损失")

    started = time.monotonic()
    frame = device.capture()
    elapsed = (time.monotonic() - started) * 1000
    typer.echo(f"截图耗时: {elapsed:.0f} ms，帧尺寸: {frame.image.shape[1]}x{frame.image.shape[0]}")
    if elapsed > 500:
        typer.echo("[提示] 截图偏慢，脚本可用但节奏会拖；可考虑模拟器自带的高速截图 API")
    typer.echo("[通过] 环境检查完成")


@app.command()
def capture(
    out: Path = typer.Option(Path("assets/captures"), "--out", help="输出目录"),
    count: int = typer.Option(1, "--count", help="抓取帧数"),
    interval: float = typer.Option(0.5, "--interval", help="抓帧间隔（秒）"),
    serial: str | None = typer.Option(None, "--serial"),
) -> None:
    """抓取当前画面，供标注使用。"""
    device = AdbDevice(serial=serial)
    out.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        frame = device.capture()
        path = out / f"{time.strftime('%Y%m%d-%H%M%S')}-{index:02d}.png"
        cv2.imwrite(str(path), frame.image)
        typer.echo(f"已保存 {path}")
        if index < count - 1:
            time.sleep(interval)


@app.command()
def crop(
    frame: Path = typer.Option(..., "--frame", help="源截图"),
    rect: str = typer.Option(..., "--rect", help="x,y,width,height"),
    name: str = typer.Option(..., "--name", help="元素名"),
    elements: Path = typer.Option(DEFAULT_ELEMENTS, "--elements"),
    templates_dir: Path = typer.Option(DEFAULT_TEMPLATES, "--templates-dir"),
    threshold: float = typer.Option(0.85, "--threshold"),
) -> None:
    """从截图裁剪模板并写入 elements.yaml。"""
    parts = [part.strip() for part in rect.split(",")]
    if len(parts) != 4 or not all(part.lstrip("-").isdigit() for part in parts):
        typer.echo("[失败] --rect 需要形如 100,200,40,30")
        raise typer.Exit(code=1)
    box = (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]))
    template_name = f"{name}.png"
    crop_and_save(frame, box, templates_dir / template_name)
    upsert_element(elements, name, box, template_name, threshold)
    typer.echo(f"已写入元素 {name}，roi={box}")


@app.command()
def run(
    config: Path = typer.Option(Path("config/config.yaml"), "--config"),
    task: str = typer.Option("combat", "--task", help="要执行的任务名"),
    dry_run: bool = typer.Option(False, "--dry-run", help="只识别，不点击"),
) -> None:
    """执行任务。"""
    if not config.exists():
        typer.echo(f"[失败] 配置文件不存在: {config}")
        raise typer.Exit(code=1)

    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    device_cfg = raw.get("device") or {}
    paths = raw.get("paths") or {}

    from .runtime import Settings, build_context

    settings = Settings(
        serial=device_cfg.get("serial"),
        adb_path=device_cfg.get("adb_path", "adb"),
        assets_dir=Path(paths.get("assets_dir", "assets")),
        templates_dir=Path(paths.get("templates_dir", "assets/templates")),
        elements_file=Path(paths.get("elements_file", "assets/elements.yaml")),
        pages_file=Path(paths.get("pages_file", "assets/pages.yaml")),
        retries=int((raw.get("scheduler") or {}).get("retries", 2)),
        logger=typer.echo,
    )
    ctx = build_context(settings)

    if dry_run:
        typer.echo("dry-run：只识别，不点击")
        result = ctx.vision.page(ctx.refresh())
        typer.echo(
            f"当前页面: {result.page} certain={result.certain} score={result.score:.3f}"
        )
        return

    from .recovery import recover_to_home
    from .scheduler import Scheduler
    from .tasks.combat import CombatTask

    combat_cfg = raw.get("combat") or {}
    tasks = {
        "combat": CombatTask(
            stage=str(combat_cfg.get("stage", "stage_1_7")),
            sanity_cost=int(combat_cfg.get("sanity_cost", 6)),
            max_rounds=int(combat_cfg.get("max_rounds", 100)),
        )
    }
    scheduler = Scheduler(ctx, tasks, retries=settings.retries, recover=recover_to_home)
    summary = scheduler.run([task])
    typer.echo(summary.render())
    if not summary.ok:
        raise typer.Exit(code=1)
```

注意 `Settings.logger` 传的是 `typer.echo`，它接受任意对象并转成字符串，满足 `Callable[[str], None]` 的用法。

- [ ] **Step 6: 运行全部测试**

Run: `.\.venv\Scripts\python.exe -m pytest -v`
Expected: 全部通过

- [ ] **Step 7: 提交**

```bash
git add config assets src/arknights_auto/cli.py src/arknights_auto/runtime.py tests/test_cli_commands.py
git commit -m "feat: 运行时组装与 CLI 命令"
```

---

### Task 15: 文档与端到端人工验收

**Files:**
- Create: `README.md`
- Create: `docs/元素标注指南.md`

**Interfaces:**
- Consumes: 全部前序模块
- Produces: 一份可在真实模拟器上逐条执行的人工验收清单

- [ ] **Step 1: 写 `README.md`**

必须覆盖以下小节，缺一不可：

1. **这是什么** + 免责声明：本工具违反游戏用户协议，使用风险自负，仅限自用。
2. **环境要求**：Python 3.11+、Android 模拟器设为 1280×720 横屏、`adb` 在 PATH 中。
3. **安装**：`python -m venv .venv` → `pip install -e ".[dev]"` → 需要 OCR 时再 `pip install -e ".[ocr]"`。
4. **上手五步**：`doctor` → `capture` → 标注（`tools/annotate.py` 或 `crop`）→ `run --dry-run` → `run`。
5. **配置文件说明**：`config/config.yaml` 每个字段的含义与默认值。
6. **常见问题**：截图慢、分辨率不对、元素找不到、`paddleocr` 未安装、模板失效了怎么办。

- [ ] **Step 2: 写 `docs/元素标注指南.md`**

必须覆盖：

- 抓哪一帧：选画面稳定、没有转场动画的一帧。
- 怎么选 ROI：尽量小、只包含该元素本身、避开会变化的内容。
- 阈值怎么调：先用 0.85；误命中就调高，漏检就调低；调到 0.95 还命中就说明模板选错了。
- 每页锚点数量：2–5 个；锚点必须分布在不同的 ROI，避免同时失效。
- **锚点禁忌**：不要选会随游戏状态变化的元素。理智数字本身绝不能当锚点——数字一变模板就失效，整个页面判定会跟着崩。
- 模板失效的征兆与重做流程。

- [ ] **Step 3: 真实模拟器验收清单**

逐条执行，把结果记在下面（实施者填写）：

| # | 命令 / 操作 | 期望 |
|---|---|---|
| 1 | `arknights-auto --version` | 打印版本号 |
| 2 | `arknights-auto doctor` | 输出 `[通过]`，并记录截图耗时 |
| 3 | `arknights-auto capture --count 5` | `assets/captures/` 下出现 5 张 1280×720 截图 |
| 4 | 标注主界面、关卡选择、作战确认、战斗中、结算页的元素 | `assets/elements.yaml` 与 `assets/templates/` 均被填充 |
| 5 | `arknights-auto run --dry-run` | 正确打印当前页面且 `certain=True` |
| 6 | `arknights-auto run --task combat` | 自动完成 ≥2 局，最终输出"理智耗尽"并停机 |
| 7 | 运行中手动弹出一个游戏内弹窗 | 脚本能恢复或安全报错，不出现乱点 |
| 8 | 断开模拟器连接后重新 `run` | 报出明确的 ADB 错误并退出，不静默卡死 |

- [ ] **Step 4: 提交**

```bash
git add README.md docs
git commit -m "docs: 上手流程与验收清单"
```

---

## 完成标准

计划全部完成后应满足：

1. `.\.venv\Scripts\python.exe -m pytest -v` 全绿，且不依赖模拟器、不依赖网络。
2. `arknights-auto doctor` 在真实模拟器上输出 `[通过]`。
3. `arknights-auto run --dry-run` 能正确报出当前页面。
4. `arknights-auto run --task combat` 能在无人干预下连续完成多局，并因理智耗尽安全停机。
5. 所有游戏素材均来自本机截图，未进入版本库。

## 已知的取舍与后续动作

- `Context.page()` 每次调用都会重新截图，比理论最优多花若干次 `screencap`。v1 接受这个开销换取实现简单；如果 `doctor` 实测截图超过 300ms，再考虑加帧缓存。
- `AdbDevice` 用 `adb shell input tap` 点击，单次延迟可能上百毫秒。若实测节奏不可接受，后续可接入 minitouch 或模拟器原生输入 API——只需要新增一个 `InputBackend` 实现，任务层不受影响。
- 假设备 `ScriptedDevice` 依赖元素位置与模板位置一致。真实素材的截图会有轻微差异，因此离线测试只能验证逻辑，不能替代第 15 步的真机验收。
- 设计文档 §8 要求把真实截图存进 `tests/fixtures/` 做回归测试。本计划只建立了合成帧测试体系；真实 fixture 库需要先有可用素材，因此推迟到第三份计划（M4）。

## 后续计划（不在本计划范围内）

- 第二份计划：日常杂项（剿灭、签到、领邮件、公招），对应设计文档 M3。
- 第三份计划：稳定性收尾，对应 M4，包括 fixture 库扩充、截图后端加速、多开支持评估。
