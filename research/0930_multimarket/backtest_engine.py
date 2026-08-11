from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from data_pipeline import LOCAL_TZ, MARKET_SPECS


@dataclass(frozen=True)
class StrategyConfig:
    config_id: str
    family: str = "exact"
    confirmation: str = "above_high"
    midpoint: str = "range"
    retracement: str = "weekly_touch"
    ma_type: str = "ema"
    cross_condition: str = "fresh"
    setup_expiry_weeks: int = 26
    cross_wait_days: int = 60
    reference_mode: str = "next_day"
    high_source: str = "reference"
    entry_mode: str = "exact_dual_next_open"
    entry_valid_days: int = 1
    same_bar_policy: str = "adverse_first"
    stop_type: str = "structural"
    scale_out: str = "half_below20"
    remainder_exit: str = "ma80_or_target"
    target_r: float = 3.0
    max_hold_days: int = 60
    cost_multiplier: float = 1.0
    risk_model: str = "risk_only"


@dataclass(frozen=True)
class ReferenceEvent:
    anchor_week: str
    anchor_idx: int
    confirmation_idx: int
    retrace_idx: int
    cross_idx: int
    ref_idx: int
    midpoint_value: float


@dataclass(frozen=True)
class EntrySignal:
    route: str
    signal_idx: int
    entry_idx: int
    raw_entry: float
    structural_stop: float
    selected_high: float
    level: float | None = None


class PreparedMarket:
    def __init__(self, name: str, frame: pd.DataFrame):
        self.name = name
        self.spec = MARKET_SPECS[name]
        self.df = frame.copy().sort_index()
        self.index = self.df.index
        self.n = len(self.df)
        self.open = self.df["open"].to_numpy(dtype=float)
        self.high = self.df["high"].to_numpy(dtype=float)
        self.low = self.df["low"].to_numpy(dtype=float)
        self.close = self.df["close"].to_numpy(dtype=float)
        self.volume = self.df["volume"].to_numpy(dtype=float)
        close_s = self.df["close"]
        self.ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().to_numpy(dtype=float)
        self.ema80 = close_s.ewm(span=80, adjust=False, min_periods=80).mean().to_numpy(dtype=float)
        self.sma20 = close_s.rolling(20, min_periods=20).mean().to_numpy(dtype=float)
        self.sma80 = close_s.rolling(80, min_periods=80).mean().to_numpy(dtype=float)
        prev_close = close_s.shift(1)
        tr = pd.concat(
            [
                self.df["high"] - self.df["low"],
                (self.df["high"] - prev_close).abs(),
                (self.df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        self.atr14 = tr.rolling(14, min_periods=14).mean().to_numpy(dtype=float)
        self.prior20_low = self.df["low"].rolling(20, min_periods=1).min().shift(1).to_numpy(dtype=float)
        self.prior_ath = self.df["high"].cummax().shift(1).to_numpy(dtype=float)

        self.date_arr = np.array([d.toordinal() for d in self.index.date], dtype=np.int32)
        self.minute_arr = (self.index.hour * 60 + self.index.minute).to_numpy(dtype=np.int16)
        self.year_arr = self.index.year.to_numpy(dtype=np.int16)
        self.ref_indices = np.flatnonzero(self.minute_arr == 570)
        self.ref_dates = self.date_arr[self.ref_indices]
        self.unique_dates, first_pos = np.unique(self.date_arr, return_index=True)
        _, last_rev = np.unique(self.date_arr[::-1], return_index=True)
        last_pos = self.n - 1 - last_rev
        order = np.argsort(self.unique_dates)
        self.unique_dates = self.unique_dates[order]
        self.date_first = first_pos[order]
        self.date_last = last_pos[order]
        self.date_to_pos = {int(d): i for i, d in enumerate(self.unique_dates)}

        self.day_highs = np.array(
            [self.high[a : b + 1].max() for a, b in zip(self.date_first, self.date_last)],
            dtype=float,
        )
        rth_highs = []
        for a, b in zip(self.date_first, self.date_last):
            sl = np.arange(a, b + 1)
            mask = (self.minute_arr[sl] >= 570) & (self.minute_arr[sl] <= 945)
            rth_highs.append(float(self.high[sl[mask]].max()) if mask.any() else np.nan)
        self.rth_highs = np.asarray(rth_highs, dtype=float)

        naive = self.index.tz_localize(None)
        periods = naive.to_period(self.spec.week_ending)
        tmp = self.df.copy()
        tmp["_pos"] = np.arange(self.n)
        tmp["_period"] = periods.astype(str)
        grouped = tmp.groupby("_period", sort=True)
        self.weekly = grouped.agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            start_idx=("_pos", "min"),
            end_idx=("_pos", "max"),
        ).reset_index(names="period")
        self.week_end_indices = self.weekly["end_idx"].to_numpy(dtype=int)
        self.week_high = self.weekly["high"].to_numpy(dtype=float)
        self.week_close = self.weekly["close"].to_numpy(dtype=float)
        self.week_low = self.weekly["low"].to_numpy(dtype=float)
        self.week_open = self.weekly["open"].to_numpy(dtype=float)
        self._event_cache: Dict[Tuple, List[ReferenceEvent]] = {}

    def moving_averages(self, ma_type: str) -> Tuple[np.ndarray, np.ndarray]:
        if ma_type == "ema":
            return self.ema20, self.ema80
        if ma_type == "sma":
            return self.sma20, self.sma80
        raise ValueError(ma_type)

    def period_bounds(self, split: Dict[str, int], period: str) -> Tuple[int, int]:
        if period == "development":
            mask = (self.year_arr >= split["development_start"]) & (
                self.year_arr <= split["development_end"]
            )
        elif period == "validation":
            mask = self.year_arr == split["validation"]
        elif period == "holdout":
            mask = self.year_arr == split["holdout"]
        else:
            raise ValueError(period)
        positions = np.flatnonzero(mask)
        if len(positions) == 0:
            return -1, -1
        return int(positions[0]), int(positions[-1])

    def _event_key(self, cfg: StrategyConfig) -> Tuple:
        return (
            cfg.confirmation,
            cfg.midpoint,
            cfg.retracement,
            cfg.ma_type,
            cfg.cross_condition,
            cfg.setup_expiry_weeks,
            cfg.cross_wait_days,
            cfg.reference_mode,
        )

    def generate_events(self, cfg: StrategyConfig) -> List[ReferenceEvent]:
        key = self._event_key(cfg)
        cached = self._event_cache.get(key)
        if cached is not None:
            return cached

        fast, slow = self.moving_averages(cfg.ma_type)
        events: List[ReferenceEvent] = []
        w = self.weekly
        for i in range(len(w) - 2):
            a_high = self.week_high[i]
            a_low = self.week_low[i]
            a_open = self.week_open[i]
            a_close = self.week_close[i]
            b_close = self.week_close[i + 1]
            if cfg.confirmation == "above_high":
                confirmed = b_close > a_high
            elif cfg.confirmation == "above_close":
                confirmed = b_close > a_close
            elif cfg.confirmation == "above_body_top":
                confirmed = b_close > max(a_open, a_close)
            else:
                raise ValueError(cfg.confirmation)
            if not confirmed:
                continue

            midpoint = (
                (a_high + a_low) / 2
                if cfg.midpoint == "range"
                else (a_open + a_close) / 2
            )
            b_end = int(self.week_end_indices[i + 1])
            expiry_ts = self.index[b_end] + pd.Timedelta(weeks=cfg.setup_expiry_weeks)
            expiry_idx = int(self.index.searchsorted(expiry_ts, side="right") - 1)
            expiry_idx = min(expiry_idx, self.n - 1)
            if expiry_idx <= b_end:
                continue

            retrace_idx: Optional[int] = None
            if cfg.retracement == "intraday_touch":
                segment = self.low[b_end + 1 : expiry_idx + 1]
                hits = np.flatnonzero(segment <= midpoint)
                if len(hits):
                    retrace_idx = b_end + 1 + int(hits[0])
            elif cfg.retracement in {"weekly_touch", "weekly_close"}:
                for j in range(i + 2, len(w)):
                    j_end = int(self.week_end_indices[j])
                    if j_end > expiry_idx:
                        break
                    value = (
                        self.week_low[j]
                        if cfg.retracement == "weekly_touch"
                        else self.week_close[j]
                    )
                    if value <= midpoint:
                        retrace_idx = j_end
                        break
            else:
                raise ValueError(cfg.retracement)
            if retrace_idx is None or retrace_idx + 1 >= self.n:
                continue

            cross_deadline_ts = self.index[retrace_idx] + pd.Timedelta(
                days=cfg.cross_wait_days
            )
            cross_deadline_idx = min(
                int(self.index.searchsorted(cross_deadline_ts, side="right") - 1),
                self.n - 1,
            )
            if cross_deadline_idx <= retrace_idx:
                continue
            cross_idx: Optional[int] = None
            start = max(retrace_idx, 1)
            if (
                cfg.cross_condition == "already_or_fresh"
                and np.isfinite(fast[start])
                and np.isfinite(slow[start])
                and fast[start] > slow[start]
            ):
                cross_idx = start
            else:
                for k in range(max(start + 1, 1), cross_deadline_idx + 1):
                    if not (
                        np.isfinite(fast[k - 1])
                        and np.isfinite(slow[k - 1])
                        and np.isfinite(fast[k])
                        and np.isfinite(slow[k])
                    ):
                        continue
                    if fast[k - 1] <= slow[k - 1] and fast[k] > slow[k]:
                        cross_idx = k
                        break
            if cross_idx is None:
                continue

            pos = int(np.searchsorted(self.ref_indices, cross_idx + 1, side="left"))
            if cfg.reference_mode in {"next_day", "next_weekday"}:
                cross_date = self.date_arr[cross_idx]
                while (
                    pos < len(self.ref_indices)
                    and self.date_arr[self.ref_indices[pos]] <= cross_date
                ):
                    pos += 1
                if cfg.reference_mode == "next_weekday":
                    while (
                        pos < len(self.ref_indices)
                        and self.index[self.ref_indices[pos]].weekday() >= 5
                    ):
                        pos += 1
            elif cfg.reference_mode != "same_or_next":
                raise ValueError(cfg.reference_mode)
            if pos >= len(self.ref_indices):
                continue
            ref_idx = int(self.ref_indices[pos])
            if ref_idx <= cross_idx or ref_idx + 1 >= self.n:
                continue
            events.append(
                ReferenceEvent(
                    anchor_week=str(w.iloc[i]["period"]),
                    anchor_idx=int(self.week_end_indices[i]),
                    confirmation_idx=b_end,
                    retrace_idx=int(retrace_idx),
                    cross_idx=int(cross_idx),
                    ref_idx=ref_idx,
                    midpoint_value=float(midpoint),
                )
            )

        events.sort(key=lambda e: (e.ref_idx, e.anchor_idx))
        dedup: List[ReferenceEvent] = []
        seen = set()
        for event in events:
            signature = (
                event.anchor_idx,
                event.confirmation_idx,
                event.retrace_idx,
                event.cross_idx,
                event.ref_idx,
            )
            if signature not in seen:
                seen.add(signature)
                dedup.append(event)
        self._event_cache[key] = dedup
        return dedup

    def event_expiry_idx(self, ref_idx: int, valid_days: int) -> int:
        ref_date = int(self.date_arr[ref_idx])
        p = self.date_to_pos.get(ref_date)
        if p is None:
            return ref_idx
        target_p = min(p + max(valid_days, 1) - 1, len(self.unique_dates) - 1)
        return int(self.date_last[target_p])

    def selected_high(self, event: ReferenceEvent, source: str) -> float:
        r = event.ref_idx
        if source == "reference":
            return float(self.high[r])
        date_pos = self.date_to_pos.get(int(self.date_arr[r]), -1)
        if source == "previous_day":
            return float(self.day_highs[date_pos - 1]) if date_pos > 0 else np.nan
        if source == "previous_rth":
            if date_pos <= 0:
                return np.nan
            p = date_pos - 1
            while p >= 0 and not np.isfinite(self.rth_highs[p]):
                p -= 1
            return float(self.rth_highs[p]) if p >= 0 else np.nan
        if source == "previous_week":
            wi = int(np.searchsorted(self.week_end_indices, r, side="left") - 1)
            return float(self.week_high[wi]) if wi >= 0 else np.nan
        if source == "prior_ath":
            return float(self.prior_ath[r])
        if source == "since_cross":
            return float(np.nanmax(self.high[event.cross_idx : r + 1]))
        raise ValueError(source)


def _fill_breakout(mkt: PreparedMarket, i: int, level: float) -> float:
    return float(mkt.open[i]) if mkt.open[i] > level else float(level)


def _fill_buy_limit(mkt: PreparedMarket, i: int, level: float) -> float:
    return float(mkt.open[i]) if mkt.open[i] < level else float(level)


def find_entry(
    mkt: PreparedMarket,
    event: ReferenceEvent,
    cfg: StrategyConfig,
    period_end: int,
) -> Optional[EntrySignal]:
    r = event.ref_idx
    high_level = mkt.selected_high(event, cfg.high_source)
    ref_low = float(mkt.low[r])
    if not (
        np.isfinite(high_level)
        and high_level > 0
        and np.isfinite(ref_low)
        and ref_low > 0
    ):
        return None
    expiry = min(mkt.event_expiry_idx(r, cfg.entry_valid_days), period_end)
    start = r + 1
    if expiry < start:
        return None

    mode = cfg.entry_mode
    low_breached = False
    fallback_armed = False
    upper80 = ref_low + 0.80 * (high_level - ref_low)
    deep80 = high_level - 0.80 * (high_level - ref_low)
    literal80 = 0.80 * high_level

    for i in range(start, expiry + 1):
        o, h, l, c = mkt.open[i], mkt.high[i], mkt.low[i], mkt.close[i]
        breakout = h > high_level or o > high_level
        breach = l < ref_low or o < ref_low
        strict_reclaim = o < ref_low and c > ref_low
        touch_reclaim = l < ref_low and c > ref_low

        if mode in {
            "exact_dual_next_open",
            "exact_dual_reclaim_close",
            "dual_keep_both",
        }:
            if not low_breached:
                if breakout and breach:
                    if cfg.same_bar_policy == "breakout_first":
                        return EntrySignal(
                            "breakout",
                            i,
                            i,
                            _fill_breakout(mkt, i, high_level),
                            ref_low,
                            high_level,
                        )
                    low_breached = True
                elif breach:
                    low_breached = True
                elif breakout:
                    return EntrySignal(
                        "breakout",
                        i,
                        i,
                        _fill_breakout(mkt, i, high_level),
                        ref_low,
                        high_level,
                    )
            if low_breached:
                if mode == "dual_keep_both" and breakout:
                    return EntrySignal(
                        "breakout_after_breach",
                        i,
                        i,
                        _fill_breakout(mkt, i, high_level),
                        ref_low,
                        high_level,
                    )
                if strict_reclaim:
                    if mode == "exact_dual_reclaim_close":
                        return EntrySignal(
                            "strict_reclaim_close",
                            i,
                            i,
                            float(c),
                            float(l),
                            high_level,
                        )
                    if i + 1 <= expiry and i + 1 <= period_end:
                        return EntrySignal(
                            "strict_reclaim_next_open",
                            i,
                            i + 1,
                            float(mkt.open[i + 1]),
                            float(l),
                            high_level,
                        )
            continue

        if mode == "breakout_only":
            if breakout:
                return EntrySignal(
                    "breakout",
                    i,
                    i,
                    _fill_breakout(mkt, i, high_level),
                    ref_low,
                    high_level,
                )
            continue

        if mode in {"reclaim_only", "reclaim_touch_close"}:
            low_breached = low_breached or breach
            qualifies = strict_reclaim if mode == "reclaim_only" else touch_reclaim
            if low_breached and qualifies and i + 1 <= expiry and i + 1 <= period_end:
                return EntrySignal(
                    mode + "_next_open",
                    i,
                    i + 1,
                    float(mkt.open[i + 1]),
                    float(l),
                    high_level,
                )
            continue

        if mode in {
            "upper80_recovery",
            "upper80_limit",
            "deep80_limit",
            "literal80_limit",
            "breakout_then_upper80_recovery",
            "breakout_then_upper80_limit",
            "breakout_then_deep80_limit",
            "breakout_then_literal80_limit",
        }:
            after_fallback = mode.startswith("breakout_then_")
            if after_fallback and not fallback_armed:
                if breakout and breach:
                    if cfg.same_bar_policy == "breakout_first":
                        return EntrySignal(
                            "breakout",
                            i,
                            i,
                            _fill_breakout(mkt, i, high_level),
                            ref_low,
                            high_level,
                        )
                    fallback_armed = True
                elif breach:
                    fallback_armed = True
                elif breakout:
                    return EntrySignal(
                        "breakout",
                        i,
                        i,
                        _fill_breakout(mkt, i, high_level),
                        ref_low,
                        high_level,
                    )
                if not fallback_armed:
                    continue
            core = mode.replace("breakout_then_", "")
            if core == "upper80_recovery":
                level = upper80
                if l <= level:
                    fallback_armed = True
                prev_close = mkt.close[i - 1] if i > 0 else np.nan
                if (
                    fallback_armed
                    and c > level
                    and prev_close <= level
                    and i + 1 <= expiry
                    and i + 1 <= period_end
                ):
                    return EntrySignal(
                        "upper80_recovery",
                        i,
                        i + 1,
                        float(mkt.open[i + 1]),
                        ref_low,
                        high_level,
                        level,
                    )
            else:
                level = {
                    "upper80_limit": upper80,
                    "deep80_limit": deep80,
                    "literal80_limit": literal80,
                }[core]
                if level > 0 and l <= level:
                    return EntrySignal(
                        core,
                        i,
                        i,
                        _fill_buy_limit(mkt, i, level),
                        ref_low,
                        high_level,
                        level,
                    )
            continue

        raise ValueError(mode)
    return None


def compute_stop(
    mkt: PreparedMarket, signal: EntrySignal, cfg: StrategyConfig
) -> float:
    i = signal.entry_idx
    atr = mkt.atr14[min(max(signal.signal_idx, 0), mkt.n - 1)]
    structural = signal.structural_stop
    if cfg.stop_type == "structural":
        stop = structural
    elif cfg.stop_type == "structural_atr":
        stop = structural - 0.10 * atr
    elif cfg.stop_type == "prior20_atr":
        stop = mkt.prior20_low[i] - 0.10 * atr
    elif cfg.stop_type == "atr1_5":
        stop = signal.raw_entry - 1.5 * atr
    else:
        raise ValueError(cfg.stop_type)
    return float(stop)


def _net_exit_price(raw: float, cost: float) -> float:
    return float(raw) * (1.0 - cost)


def _entry_exec(raw: float, cost: float) -> float:
    return float(raw) * (1.0 + cost)


def simulate_trade(
    mkt: PreparedMarket,
    event: ReferenceEvent,
    signal: EntrySignal,
    cfg: StrategyConfig,
    equity: float,
    period_end: int,
) -> Optional[dict]:
    entry_i = signal.entry_idx
    if entry_i > period_end:
        return None
    stop_raw = compute_stop(mkt, signal, cfg)
    if not (
        np.isfinite(stop_raw)
        and stop_raw > 0
        and stop_raw < signal.raw_entry
    ):
        return None
    cost = mkt.spec.base_cost_rate * cfg.cost_multiplier
    entry_exec = _entry_exec(signal.raw_entry, cost)
    stop_exec_est = _net_exit_price(stop_raw, cost)
    risk_per_unit = entry_exec - stop_exec_est
    if not (risk_per_unit > 0 and np.isfinite(risk_per_unit)):
        return None
    risk_budget = equity * 0.01
    qty = risk_budget / risk_per_unit
    if cfg.risk_model == "notional_cap":
        qty = min(qty, equity / entry_exec)
    planned_risk = qty * risk_per_unit
    if qty <= 0 or planned_risk <= 0:
        return None

    remaining = qty
    realized = 0.0
    fills: List[Tuple[int, float, float, str]] = []
    scaled = False
    schedule_scale = False
    highest = signal.raw_entry
    target_raw = signal.raw_entry + cfg.target_r * (
        signal.raw_entry - stop_raw
    )
    entry_ts = mkt.index[entry_i]
    last_i = entry_i
    exit_reason = "period_end"

    start_i = entry_i + 1 if signal.route.endswith("_close") else entry_i
    for i in range(start_i, period_end + 1):
        last_i = i
        o, h, l, c = mkt.open[i], mkt.high[i], mkt.low[i], mkt.close[i]
        highest = max(highest, h)

        if o <= stop_raw:
            px = _net_exit_price(o, cost)
            realized += remaining * (px - entry_exec)
            fills.append((i, remaining, px, "gap_stop"))
            remaining = 0.0
            exit_reason = "gap_stop"
            break
        if schedule_scale and remaining > 0:
            amount = remaining if cfg.scale_out == "full_below20" else remaining * 0.5
            px = _net_exit_price(o, cost)
            realized += amount * (px - entry_exec)
            fills.append((i, amount, px, cfg.scale_out))
            remaining -= amount
            scaled = True
            schedule_scale = False
            if remaining <= 1e-12:
                remaining = 0.0
                exit_reason = cfg.scale_out
                break

        if l <= stop_raw:
            px = _net_exit_price(stop_raw, cost)
            realized += remaining * (px - entry_exec)
            fills.append((i, remaining, px, "stop"))
            remaining = 0.0
            exit_reason = "stop"
            break

        target_hit = h >= target_raw
        slow = mkt.ema80[i] if cfg.ma_type == "ema" else mkt.sma80[i]
        fast = mkt.ema20[i] if cfg.ma_type == "ema" else mkt.sma20[i]
        if i > 0:
            prev_fast = mkt.ema20[i - 1] if cfg.ma_type == "ema" else mkt.sma20[i - 1]
            prev_slow = mkt.ema80[i - 1] if cfg.ma_type == "ema" else mkt.sma80[i - 1]
        else:
            prev_fast = prev_slow = np.nan
        ma80_hit = np.isfinite(slow) and l <= slow
        bearish_cross = (
            np.isfinite(prev_fast)
            and np.isfinite(prev_slow)
            and np.isfinite(fast)
            and np.isfinite(slow)
            and prev_fast >= prev_slow
            and fast < slow
        )
        trail_raw = (
            highest - 3.0 * mkt.atr14[i]
            if np.isfinite(mkt.atr14[i])
            else -np.inf
        )
        trail_hit = l <= trail_raw and trail_raw > stop_raw

        remainder_reason = None
        remainder_raw = None
        if cfg.remainder_exit == "ma80_or_target":
            if target_hit:
                remainder_reason, remainder_raw = "target", target_raw
            elif scaled and ma80_hit:
                remainder_reason, remainder_raw = "ma80", slow
        elif cfg.remainder_exit == "target_only":
            if target_hit:
                remainder_reason, remainder_raw = "target", target_raw
        elif cfg.remainder_exit == "bearish_cross_or_target":
            if target_hit:
                remainder_reason, remainder_raw = "target", target_raw
            elif bearish_cross:
                remainder_reason, remainder_raw = "bearish_cross", c
        elif cfg.remainder_exit == "atr_trail_or_target":
            if target_hit:
                remainder_reason, remainder_raw = "target", target_raw
            elif trail_hit:
                remainder_reason, remainder_raw = "atr_trail", trail_raw
        elif cfg.remainder_exit == "ma80_only":
            if scaled and ma80_hit:
                remainder_reason, remainder_raw = "ma80", slow
        else:
            raise ValueError(cfg.remainder_exit)

        if remainder_reason is not None and remaining > 0:
            px = _net_exit_price(float(remainder_raw), cost)
            realized += remaining * (px - entry_exec)
            fills.append((i, remaining, px, remainder_reason))
            remaining = 0.0
            exit_reason = remainder_reason
            break

        if (
            not scaled
            and cfg.scale_out != "none"
            and np.isfinite(fast)
            and c < fast
            and i < period_end
        ):
            schedule_scale = True

        if mkt.index[i] >= entry_ts + pd.Timedelta(days=cfg.max_hold_days):
            px = _net_exit_price(c, cost)
            realized += remaining * (px - entry_exec)
            fills.append((i, remaining, px, "max_hold"))
            remaining = 0.0
            exit_reason = "max_hold"
            break

    if remaining > 0:
        i = min(last_i, period_end)
        px = _net_exit_price(mkt.close[i], cost)
        realized += remaining * (px - entry_exec)
        fills.append((i, remaining, px, "period_end"))
        remaining = 0.0
        exit_reason = "period_end"
        last_i = i

    r_mult = realized / planned_risk
    weighted_exit = sum(amount * px for _, amount, px, _ in fills) / qty
    return {
        "market": mkt.name,
        "config_id": cfg.config_id,
        "family": cfg.family,
        "anchor_week": event.anchor_week,
        "anchor_time": mkt.index[event.anchor_idx].isoformat(),
        "confirmation_time": mkt.index[event.confirmation_idx].isoformat(),
        "retrace_time": mkt.index[event.retrace_idx].isoformat(),
        "cross_time": mkt.index[event.cross_idx].isoformat(),
        "reference_time": mkt.index[event.ref_idx].isoformat(),
        "entry_route": signal.route,
        "signal_time": mkt.index[signal.signal_idx].isoformat(),
        "entry_time": mkt.index[entry_i].isoformat(),
        "exit_time": mkt.index[last_i].isoformat(),
        "raw_entry": signal.raw_entry,
        "entry_exec": entry_exec,
        "stop_raw": stop_raw,
        "selected_high": signal.selected_high,
        "entry_level": signal.level,
        "quantity": qty,
        "planned_risk_pct": planned_risk / equity,
        "net_pnl": realized,
        "r_multiple": r_mult,
        "weighted_exit_exec": weighted_exit,
        "exit_reason": exit_reason,
        "holding_hours": (
            mkt.index[last_i] - mkt.index[entry_i]
        ).total_seconds()
        / 3600,
        "equity_before": equity,
        "equity_after": equity + realized,
        "cost_rate_per_fill": cost,
        "fill_count": len(fills),
    }


def simulate_period(
    mkt: PreparedMarket,
    cfg: StrategyConfig,
    split: Dict[str, int],
    period: str,
) -> Tuple[dict, List[dict]]:
    start, end = mkt.period_bounds(split, period)
    if start < 0:
        return empty_metrics(mkt.name, cfg.config_id, period), []
    events = mkt.generate_events(cfg)
    equity = 10000.0
    peak = equity
    max_dd = 0.0
    trades: List[dict] = []
    last_exit = start - 1
    for event in events:
        if (
            event.ref_idx < start
            or event.ref_idx > end
            or event.ref_idx <= last_exit
        ):
            continue
        signal = find_entry(mkt, event, cfg, end)
        if (
            signal is None
            or signal.entry_idx < start
            or signal.entry_idx <= last_exit
        ):
            continue
        trade = simulate_trade(mkt, event, signal, cfg, equity, end)
        if trade is None:
            continue
        trade["period"] = period
        trades.append(trade)
        equity = float(trade["equity_after"])
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
        last_exit = int(
            mkt.index.searchsorted(pd.Timestamp(trade["exit_time"]), side="left")
        )

    metrics = metrics_from_trades(
        mkt.name, cfg.config_id, period, trades, equity, max_dd
    )
    return metrics, trades


def empty_metrics(market: str, config_id: str, period: str) -> dict:
    return {
        "market": market,
        "config_id": config_id,
        "period": period,
        "trades": 0,
        "total_return": 0.0,
        "win_rate": np.nan,
        "profit_factor": np.nan,
        "mean_r": np.nan,
        "median_r": np.nan,
        "max_drawdown": 0.0,
        "avg_holding_hours": np.nan,
        "breakout_trades": 0,
        "reclaim_trades": 0,
        "ending_equity": 10000.0,
    }


def metrics_from_trades(
    market: str,
    config_id: str,
    period: str,
    trades: List[dict],
    equity: float,
    max_dd: float,
) -> dict:
    if not trades:
        return empty_metrics(market, config_id, period)
    pnl = np.array([trade["net_pnl"] for trade in trades], dtype=float)
    r = np.array([trade["r_multiple"] for trade in trades], dtype=float)
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    routes = [str(trade["entry_route"]) for trade in trades]
    return {
        "market": market,
        "config_id": config_id,
        "period": period,
        "trades": int(len(trades)),
        "total_return": float(equity / 10000.0 - 1.0),
        "win_rate": float((pnl > 0).mean()),
        "profit_factor": (
            float(gains / losses)
            if losses > 0
            else (float("inf") if gains > 0 else np.nan)
        ),
        "mean_r": float(r.mean()),
        "median_r": float(np.median(r)),
        "max_drawdown": float(max_dd),
        "avg_holding_hours": float(
            np.mean([trade["holding_hours"] for trade in trades])
        ),
        "breakout_trades": int(sum("breakout" in route for route in routes)),
        "reclaim_trades": int(sum("reclaim" in route for route in routes)),
        "ending_equity": float(equity),
    }


def aggregate_metrics(market_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (config_id, period), group in market_metrics.groupby(
        ["config_id", "period"], sort=False
    ):
        trades = int(group["trades"].sum())
        weights = group["trades"].to_numpy(dtype=float)
        mean_r_values = group["mean_r"].fillna(0).to_numpy(dtype=float)
        weighted_r = (
            float(np.average(mean_r_values, weights=weights))
            if weights.sum() > 0
            else np.nan
        )
        rows.append(
            {
                "config_id": config_id,
                "period": period,
                "markets": int(len(group)),
                "trades": trades,
                "equal_weight_return": float(group["total_return"].mean()),
                "median_market_return": float(group["total_return"].median()),
                "positive_markets": int((group["total_return"] > 0).sum()),
                "negative_markets": int((group["total_return"] < 0).sum()),
                "trade_weighted_mean_r": weighted_r,
                "median_market_mean_r": (
                    float(group["mean_r"].median(skipna=True))
                    if group["mean_r"].notna().any()
                    else np.nan
                ),
                "mean_profit_factor": float(
                    group["profit_factor"]
                    .replace([np.inf, -np.inf], np.nan)
                    .mean(skipna=True)
                ),
                "worst_market_return": float(group["total_return"].min()),
                "best_market_return": float(group["total_return"].max()),
                "max_market_drawdown": float(group["max_drawdown"].max()),
            }
        )
    return pd.DataFrame(rows)


def catalog() -> List[StrategyConfig]:
    base = StrategyConfig(config_id="EXACT_BASELINE", family="exact_baseline")
    configs: List[StrategyConfig] = [base]

    high_sources = [
        "reference",
        "previous_day",
        "previous_rth",
        "previous_week",
        "prior_ath",
        "since_cross",
    ]
    entry_modes = [
        "exact_dual_next_open",
        "exact_dual_reclaim_close",
        "dual_keep_both",
        "breakout_only",
        "reclaim_only",
        "reclaim_touch_close",
        "upper80_recovery",
        "upper80_limit",
        "deep80_limit",
        "literal80_limit",
        "breakout_then_upper80_recovery",
        "breakout_then_upper80_limit",
        "breakout_then_deep80_limit",
        "breakout_then_literal80_limit",
    ]
    for high_source in high_sources:
        for entry_mode in entry_modes:
            for valid in [1, 3, 7]:
                configs.append(
                    replace(
                        base,
                        config_id=f"ENTRY__{high_source}__{entry_mode}__{valid}d",
                        family="entry_high_grid",
                        high_source=high_source,
                        entry_mode=entry_mode,
                        entry_valid_days=valid,
                    )
                )

    for confirmation in ["above_high", "above_close", "above_body_top"]:
        for midpoint in ["range", "body"]:
            for retracement in [
                "weekly_touch",
                "weekly_close",
                "intraday_touch",
            ]:
                for ma_type in ["ema", "sma"]:
                    for cross in ["fresh", "already_or_fresh"]:
                        configs.append(
                            replace(
                                base,
                                config_id=(
                                    f"WEEKLY__{confirmation}__{midpoint}__"
                                    f"{retracement}__{ma_type}__{cross}"
                                ),
                                family="weekly_indicator_grid",
                                confirmation=confirmation,
                                midpoint=midpoint,
                                retracement=retracement,
                                ma_type=ma_type,
                                cross_condition=cross,
                            )
                        )

    for stop in ["structural", "structural_atr", "prior20_atr", "atr1_5"]:
        configs.append(
            replace(
                base,
                config_id=f"STOP__{stop}",
                family="risk_exit_sweep",
                stop_type=stop,
            )
        )
    for scale in ["half_below20", "full_below20", "none"]:
        configs.append(
            replace(
                base,
                config_id=f"SCALE__{scale}",
                family="risk_exit_sweep",
                scale_out=scale,
            )
        )
    exit_variants = [
        ("ma80_or_target", 2.0),
        ("ma80_or_target", 3.0),
        ("ma80_or_target", 5.0),
        ("target_only", 2.0),
        ("target_only", 3.0),
        ("target_only", 5.0),
        ("bearish_cross_or_target", 3.0),
        ("atr_trail_or_target", 3.0),
        ("ma80_only", 3.0),
    ]
    for exit_mode, target in exit_variants:
        configs.append(
            replace(
                base,
                config_id=f"EXIT__{exit_mode}__{target:g}R",
                family="risk_exit_sweep",
                remainder_exit=exit_mode,
                target_r=target,
            )
        )
    for max_hold in [10, 30, 60]:
        configs.append(
            replace(
                base,
                config_id=f"HOLD__{max_hold}d",
                family="timing_sweep",
                max_hold_days=max_hold,
            )
        )
    for expiry in [12, 26, 52]:
        configs.append(
            replace(
                base,
                config_id=f"SETUP_EXPIRY__{expiry}w",
                family="timing_sweep",
                setup_expiry_weeks=expiry,
            )
        )
    for wait in [30, 60, 120]:
        configs.append(
            replace(
                base,
                config_id=f"CROSS_WAIT__{wait}d",
                family="timing_sweep",
                cross_wait_days=wait,
            )
        )
    for ref_mode in ["next_day", "same_or_next", "next_weekday"]:
        configs.append(
            replace(
                base,
                config_id=f"REFERENCE__{ref_mode}",
                family="timing_sweep",
                reference_mode=ref_mode,
            )
        )
    for policy in ["adverse_first", "breakout_first"]:
        configs.append(
            replace(
                base,
                config_id=f"SAME_BAR__{policy}",
                family="execution_sweep",
                same_bar_policy=policy,
            )
        )
    for multiplier in [0.5, 1.0, 2.0]:
        configs.append(
            replace(
                base,
                config_id=f"COST__{multiplier:g}x",
                family="execution_sweep",
                cost_multiplier=multiplier,
            )
        )
    for risk_model in ["risk_only", "notional_cap"]:
        configs.append(
            replace(
                base,
                config_id=f"RISK_MODEL__{risk_model}",
                family="execution_sweep",
                risk_model=risk_model,
            )
        )

    result: List[StrategyConfig] = []
    seen = set()
    for cfg in configs:
        params = tuple(
            value
            for key, value in asdict(cfg).items()
            if key not in {"config_id", "family"}
        )
        if params in seen:
            continue
        seen.add(params)
        result.append(cfg)
    return result


def rank_and_select(
    agg: pd.DataFrame, configs: Sequence[StrategyConfig]
) -> Tuple[pd.DataFrame, str]:
    rows = []
    config_map = {config.config_id: config for config in configs}
    for config_id in config_map:
        row = {
            "config_id": config_id,
            "family": config_map[config_id].family,
        }
        for period in ["development", "validation", "holdout"]:
            sub = agg[(agg.config_id == config_id) & (agg.period == period)]
            if sub.empty:
                continue
            series = sub.iloc[0]
            for column in [
                "trades",
                "equal_weight_return",
                "positive_markets",
                "trade_weighted_mean_r",
                "mean_profit_factor",
                "worst_market_return",
                "best_market_return",
                "max_market_drawdown",
            ]:
                row[f"{period}_{column}"] = series[column]
        dev_trades = float(row.get("development_trades", 0) or 0)
        dev_r = float(row.get("development_trade_weighted_mean_r", 0) or 0)
        dev_ret = float(row.get("development_equal_weight_return", 0) or 0)
        dev_pos = float(row.get("development_positive_markets", 0) or 0)
        row["development_score"] = (
            dev_r * math.sqrt(max(dev_trades, 1))
            + 4.0 * dev_ret
            + 0.10 * dev_pos
        )
        rows.append(row)
    ranking = pd.DataFrame(rows).sort_values(
        ["development_score", "development_trades"],
        ascending=[False, False],
    )
    ranking = ranking.reset_index(drop=True)
    ranking["development_rank"] = np.arange(1, len(ranking) + 1)

    top = ranking.head(min(20, len(ranking))).copy()
    top["validation_score"] = (
        top["validation_trade_weighted_mean_r"].fillna(-99)
        * np.sqrt(top["validation_trades"].fillna(0).clip(lower=1))
        + 4.0 * top["validation_equal_weight_return"].fillna(-99)
        + 0.10 * top["validation_positive_markets"].fillna(0)
    )
    selected = str(
        top.sort_values(
            ["validation_score", "validation_trades"],
            ascending=[False, False],
        ).iloc[0]["config_id"]
    )
    ranking["selected_before_holdout"] = ranking["config_id"].eq(selected)
    ranking["robustness_gate_pass"] = (
        (ranking["validation_trades"].fillna(0) >= 20)
        & (ranking["holdout_trades"].fillna(0) >= 20)
        & (ranking["validation_equal_weight_return"].fillna(-1) > 0)
        & (ranking["holdout_equal_weight_return"].fillna(-1) > 0)
        & (ranking["validation_trade_weighted_mean_r"].fillna(-1) > 0)
        & (ranking["holdout_trade_weighted_mean_r"].fillna(-1) > 0)
        & (ranking["validation_positive_markets"].fillna(0) >= 3)
        & (ranking["holdout_positive_markets"].fillna(0) >= 3)
    )
    return ranking, selected


def run_synthetic_tests() -> pd.DataFrame:
    rows = []
    idx = pd.date_range(
        "2024-01-02 09:30", periods=10, freq="15min", tz=LOCAL_TZ
    )
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0],
            "high": [102.0, 101.0, 103.0, 104.0, 104.0, 104.0, 104.0, 104.0, 104.0, 104.0],
            "low": [99.0, 99.5, 100.0, 101.0, 102.0, 102.0, 102.0, 102.0, 102.0, 102.0],
            "close": [101.0, 100.5, 102.5, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0, 103.0],
            "volume": 1.0,
        },
        index=idx,
    )
    market = PreparedMarket("AAPL", df)
    event = ReferenceEvent("x", 0, 0, 0, 0, 0, 100.0)
    config = StrategyConfig(config_id="test")
    signal = find_entry(market, event, config, 9)
    rows.append(
        {
            "test": "breakout_before_breach",
            "pass": bool(
                signal and signal.route == "breakout" and signal.entry_idx == 2
            ),
            "detail": repr(signal),
        }
    )

    df2 = df.copy()
    df2.loc[idx[1], ["open", "high", "low", "close"]] = [
        100.0,
        101.0,
        98.0,
        98.5,
    ]
    df2.loc[idx[2], ["open", "high", "low", "close"]] = [
        98.5,
        100.5,
        98.0,
        100.0,
    ]
    market2 = PreparedMarket("AAPL", df2)
    signal2 = find_entry(market2, event, config, 9)
    rows.append(
        {
            "test": "strict_reclaim_after_breach",
            "pass": bool(
                signal2
                and "reclaim" in signal2.route
                and signal2.entry_idx == 3
                and signal2.structural_stop == 98.0
            ),
            "detail": repr(signal2),
        }
    )

    df3 = df.copy()
    df3.loc[idx[1], ["open", "high", "low", "close"]] = [
        100.0,
        103.0,
        98.0,
        99.0,
    ]
    market3 = PreparedMarket("AAPL", df3)
    signal3 = find_entry(market3, event, config, 9)
    rows.append(
        {
            "test": "same_bar_adverse_first",
            "pass": bool(signal3 is None or signal3.entry_idx > 1),
            "detail": repr(signal3),
        }
    )

    if signal:
        market.atr14[:] = 1.0
        trade = simulate_trade(
            market,
            event,
            signal,
            replace(config, max_hold_days=1),
            10000.0,
            9,
        )
        passed = bool(
            trade and trade["planned_risk_pct"] <= 0.0100000001
        )
    else:
        trade, passed = None, False
    rows.append(
        {
            "test": "one_percent_risk",
            "pass": passed,
            "detail": repr(trade)[:500],
        }
    )
    return pd.DataFrame(rows)
