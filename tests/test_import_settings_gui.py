"""GUI「导入设置」回归：从 CSV 头必须回填所有数值字段（含 force_aspect）。

export_controller._import_settings 用轻量假控件直接驱动（不依赖 QApplication），
锁定"force_aspect_edit 从 CSV 导入时会被写入"这个历史 bug。
"""
from __future__ import annotations

import export_controller
from export_controller import ExportControllerMixin


class FakeLineEdit:
    def __init__(self, text: str = "") -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, t) -> None:
        self._text = str(t)


class FakeSpin:
    def __init__(self, value: int = 0) -> None:
        self._value = value

    def value(self) -> int:
        return self._value

    def setValue(self, v) -> None:
        self._value = int(v)

    def blockSignals(self, b) -> None:
        pass


class FakeCombo:
    def __init__(self, index: int = 0) -> None:
        self._index = index

    def currentIndex(self) -> int:
        return self._index

    def setCurrentIndex(self, i) -> None:
        self._index = int(i)


class FakeRadio:
    def __init__(self) -> None:
        self.checked = False

    def setChecked(self, b) -> None:
        self.checked = bool(b)


class FakePreview:
    def __init__(self) -> None:
        self.roi = None

    def set_roi(self, x1, y1, x2, y2) -> None:
        self.roi = (x1, y1, x2, y2)


class FakeStatus:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, t) -> None:
        self.text = str(t)


class FakeApp:
    def __init__(self, settings: dict) -> None:
        self._settings = settings
        self.roi_x1 = FakeSpin(0)
        self.roi_y1 = FakeSpin(0)
        self.roi_x2 = FakeSpin(100)
        self.roi_y2 = FakeSpin(40)
        self._preview_widget = FakePreview()
        self._status_label = FakeStatus()


def _settings_panel() -> dict:
    return {
        "max_speed_edit": FakeLineEdit("400"), "max_accel_edit": FakeLineEdit("50"),
        "force_aspect_edit": FakeLineEdit("0.0"),
        "frame_start_edit": FakeLineEdit("0"), "frame_end_edit": FakeLineEdit("1000"),
        "buffer_spin": FakeSpin(128), "fill_width_spin": FakeSpin(224),
        "backend_combo": FakeCombo("auto"), "ocr_backend_combo": FakeCombo("auto"),
        "format_ms": FakeRadio(), "format_kmh": FakeRadio(), "format_mph": FakeRadio(),
    }


def test_import_settings_reads_force_aspect(monkeypatch, tmp_path):
    csv = tmp_path / "in.csv"
    csv.write_text("# force_aspect=1.5, max_speed=320.0, fill_width=160\n"
                   "0,0.0,0,0\n", encoding="utf-8")

    class _Dialog:
        @staticmethod
        def getOpenFileName(*args, **kwargs):
            return (str(csv), "")

    monkeypatch.setattr(export_controller, "QFileDialog", _Dialog)

    s = _settings_panel()
    app = FakeApp(s)
    ExportControllerMixin._import_settings(app)

    assert s["force_aspect_edit"].text() == "1.5"
    assert s["max_speed_edit"].text() == "320.0"
    assert s["fill_width_spin"].value() == 160