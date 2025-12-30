from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field


class SessionConfig(BaseModel):
    include_extended_hours: bool = False
    include_premarket_in_vwap: bool = False
    exchange_timezone: str = "America/New_York"
    rth_open_time: str = "09:30:00"  # HH:MM:SS in exchange timezone


class SnapshotConfig(BaseModel):
    interval_seconds: int = 10


class ZoneConfig(BaseModel):
    level_width_atr_default: float = 0.25


class ATRConfig(BaseModel):
    period_bars: int = 14
    atr_blend_alpha: float = 0.7
    fixed_default_atr: float = 0.50


class ResolutionConfig(BaseModel):
    success_threshold_atr: float = 0.50
    failure_threshold_atr: float = 0.35
    timeout_seconds: int = 300


class EMAConfig(BaseModel):
    confluence_ref: float = 0.25


class StorageConfig(BaseModel):
    root_dir: str = "data/output"
    compression: Literal["zstd", "snappy"] = "zstd"


class SmartMoneyConfig(BaseModel):
    return_floor_atr: float = 0.02
    epsilon: float = 1e-9


class AppConfig(BaseModel):
    session: SessionConfig = Field(default_factory=SessionConfig)
    snapshot: SnapshotConfig = Field(default_factory=SnapshotConfig)
    zone: ZoneConfig = Field(default_factory=ZoneConfig)
    atr: ATRConfig = Field(default_factory=ATRConfig)
    ema: EMAConfig = Field(default_factory=EMAConfig)
    smart_money: SmartMoneyConfig = Field(default_factory=SmartMoneyConfig)
    resolution: ResolutionConfig = Field(default_factory=ResolutionConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    provider: Optional[dict] = None  # provider-specific settings live here

    @staticmethod
    def load(path: str | Path) -> "AppConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return AppConfig.model_validate(data)



