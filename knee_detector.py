"""
knee_detector.py  -  실시간 Knee 감지기

SOH 측정값을 순서대로 넣으면:
  - 열화 속도(rate) 계산
  - Knee 통과 여부 감지 (causal — 현재까지의 데이터만 사용)

사용 예시:
    kd = KneeDetector()
    for soh in [92.1, 91.5, 90.8, ...]:
        kd.update(soh)
    print(kd.summary())
"""

import numpy as np


SMOOTH_WIN   = 7      # 스무딩 윈도우
KNEE_THRESH  = -0.4   # 2차 미분 임계값 (더 음수 = 더 엄격)
MIN_KNEE_CYC = 15     # 최소 Knee 감지 사이클
RATE_WIN     = 5      # 열화 속도 윈도우


class KneeDetector:
    """
    실시간 Knee 감지 + 열화 속도 계산.

    매 측정마다 update(soh) 호출 -> 자동으로 knee 감지 시도.
    """

    def __init__(
        self,
        smooth_window: int   = SMOOTH_WIN,
        threshold: float     = KNEE_THRESH,
        rate_window: int     = RATE_WIN,
        min_knee_cycles: int = MIN_KNEE_CYC,
    ):
        self.smooth_window   = smooth_window
        self.threshold       = threshold
        self.rate_window     = rate_window
        self.min_knee_cycles = min_knee_cycles

        self._history: list  = []
        self.knee_cycle_: int | None = None   # knee 감지된 사이클 번호

    # ── Public API ───────────────────────────────────────────────────────────

    def update(self, soh_pct: float) -> None:
        """새 SOH 측정값 추가 후 knee 감지 시도."""
        self._history.append(float(soh_pct))
        if self.knee_cycle_ is None:
            self._try_detect()

    def reset(self) -> None:
        """히스토리 및 감지 결과 초기화."""
        self._history.clear()
        self.knee_cycle_ = None

    @property
    def n_cycles(self) -> int:
        return len(self._history)

    @property
    def current_soh(self) -> float | None:
        return self._history[-1] if self._history else None

    def is_knee_detected(self) -> bool:
        return self.knee_cycle_ is not None

    def post_knee(self) -> float:
        """현재가 knee 이후면 1.0, 아니면 0.0 (RUL 모델 feature용)."""
        return 1.0 if self.is_knee_detected() else 0.0

    def current_rate(self) -> float:
        """
        최근 rate_window 사이클의 선형 열화 속도 (%/cycle).
        (음수 = 열화 중)
        """
        s = np.array(self._history)
        if len(s) < 2:
            return 0.0
        seg = s[max(0, len(s) - self.rate_window):]
        x   = np.arange(len(seg), dtype=float)
        return float(np.polyfit(x, seg, 1)[0])

    def cycles_since_knee(self) -> int | None:
        """Knee 감지 이후 경과 사이클 수. 감지 안됐으면 None."""
        if self.knee_cycle_ is None:
            return None
        return self.n_cycles - 1 - self.knee_cycle_

    def summary(self) -> dict:
        return {
            "n_cycles":       self.n_cycles,
            "current_soh":    self.current_soh,
            "rate_%/cyc":     round(self.current_rate(), 4),
            "knee_detected":  self.is_knee_detected(),
            "knee_cycle":     self.knee_cycle_,
            "cycles_since_knee": self.cycles_since_knee(),
            "post_knee_flag": self.post_knee(),
        }

    # ── Serialization ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "history":     list(self._history),
            "knee_cycle":  self.knee_cycle_,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KneeDetector":
        obj = cls()
        for soh in d.get("history", []):
            obj._history.append(float(soh))
        obj.knee_cycle_ = d.get("knee_cycle")
        return obj

    @classmethod
    def from_list(cls, soh_list: list) -> "KneeDetector":
        """SOH 리스트로부터 KneeDetector 생성 (일괄 초기화)."""
        obj = cls()
        for soh in soh_list:
            obj.update(float(soh))
        return obj

    # ── Internal ─────────────────────────────────────────────────────────────

    def _try_detect(self) -> None:
        s = np.array(self._history)
        if len(s) < self.min_knee_cycles + 2:
            return

        # 이동 평균 스무딩
        if len(s) >= self.smooth_window:
            kernel = np.ones(self.smooth_window) / self.smooth_window
            s_sm   = np.convolve(s, kernel, mode="valid")
        else:
            s_sm = s

        if len(s_sm) < 3:
            return

        d2 = np.diff(np.diff(s_sm))
        if d2.min() >= self.threshold:
            return

        # 스무딩 offset 보정
        idx = int(np.argmin(d2)) + self.smooth_window
        self.knee_cycle_ = min(max(idx, self.min_knee_cycles), len(s) - 1)
