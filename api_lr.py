from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from random import randint
from typing import Any, Dict, List, Optional, Tuple

from zoneinfo import ZoneInfo

import httpx
import dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field
from redis.asyncio import Redis

try:  # Optional dependencies for model inference
    import numpy as np
    import joblib
except ImportError:  # pragma: no cover - allow API to run without ML stack
    np = None  # type: ignore
    joblib = None  # type: ignore

dotenv.load_dotenv(dotenv.find_dotenv())

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("bicing_api")

RedisLike = Any


class FakeRedis:
    """Minimal async Redis-like store for local testing."""

    def __init__(self) -> None:
        self._strings: Dict[str, str] = {}
        self._zsets: Dict[str, List[Tuple[float, str]]] = {}
        self._lock = asyncio.Lock()

    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            self._strings[key] = value

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            return self._strings.get(key)

    async def zadd(self, key: str, mapping: Dict[str, float]) -> None:
        async with self._lock:
            entries = self._zsets.setdefault(key, [])
            members = set(mapping.keys())
            entries[:] = [item for item in entries if item[1] not in members]
            for member, score in mapping.items():
                entries.append((score, member))
            entries.sort(key=lambda item: item[0])

    async def zrange(
        self, key: str, start: int, end: int, withscores: bool = False
    ) -> List[Any]:
        async with self._lock:
            entries = self._zsets.get(key, [])
            length = len(entries)
            if length == 0:
                return []
            if start < 0:
                start = max(0, length + start)
            if end < 0:
                end = length + end
            start = max(0, start)
            end = min(length - 1, end)
            if start > end:
                return []
            slice_ = entries[start : end + 1]
            if withscores:
                return [(member, score) for score, member in slice_]
            return [member for _, member in slice_]

    async def zrevrange(
        self, key: str, start: int, end: int, withscores: bool = False
    ) -> List[Any]:
        async with self._lock:
            entries = list(reversed(self._zsets.get(key, [])))
            length = len(entries)
            if length == 0:
                return []
            if start < 0:
                start = max(0, length + start)
            if end < 0:
                end = length + end
            start = max(0, start)
            end = min(length - 1, end)
            if start > end:
                return []
            slice_ = entries[start : end + 1]
            if withscores:
                return [(member, score) for score, member in slice_]
            return [member for _, member in slice_]

    async def aclose(self) -> None:
        return None

    async def close(self) -> None:
        return None

LATEST_SNAPSHOT_KEY = "bicing:stations:latest"
SNAPSHOT_HISTORY_KEY = "bicing:stations:history"
POLL_INTERVAL_SECONDS = 60 * 5  # 5 minutes
BICING_STATUS_URL = (
    "https://opendata-ajuntament.barcelona.cat/data/dataset/6aa3416d-ce1a-494d-861b-7bd07f069600/"
    "resource/1b215493-9e63-4a12-8980-2d7e0fa19f85/download"
)
BICING_AUTH_TOKEN = os.getenv("BICING_API_TOKEN")
META_DIR = os.getenv("BICING_META_DIR", "bicing_lstm_artifacts_stateless")
LINEAR_MODEL_PATH = os.getenv(
    "BICING_LINEAR_MODEL_PATH", "bicing_linear_artifacts/best_linear_model.joblib"
)
ZONE_EUROPE_MADRID = ZoneInfo("Europe/Madrid")

@dataclass
class PreparedObservation:
    station_idx: int
    cont_std: "np.ndarray"
    status_id: int
    ts_utc: int
    delta_steps: int


class LinearOnlineBuffer:
    def __init__(self, seq_len: int) -> None:
        self.seq_len = seq_len
        self.cont: deque["np.ndarray"] = deque(maxlen=seq_len)
        self.status: deque[int] = deque(maxlen=seq_len)
        self.last_ts: Optional[int] = None

    def clear(self) -> None:
        self.cont.clear()
        self.status.clear()
        self.last_ts = None

    def append(self, ts: int, cont_vec_std: "np.ndarray", status_id: int) -> None:
        if np is None:
            raise RuntimeError("NumPy is required for inference.")
        cont = np.asarray(cont_vec_std, dtype=np.float32)
        self.last_ts = ts
        self.cont.append(cont)
        self.status.append(int(status_id))

    def ready(self) -> bool:
        return len(self.cont) >= self.seq_len

    def feature_vector(self) -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy is required for inference.")
        if not self.ready():
            raise RuntimeError("Insufficient history buffered for inference.")
        cont_seq = np.stack(self.cont, axis=0)
        return cont_seq.reshape(-1).astype(np.float32)

    def last_status(self) -> int:
        if not self.status:
            raise RuntimeError("Buffer is empty; no status available.")
        return self.status[-1]


class BicingLinearForecastService:
    """Handles loading and inference for the linear regression forecaster."""

    def __init__(
        self,
        meta_dir: str = META_DIR,
        model_path: str = LINEAR_MODEL_PATH,
    ) -> None:
        self.meta_dir = meta_dir
        self.model_path = model_path
        self.meta: Optional[Dict[str, Any]] = None
        self.station2id: Dict[int, int] = {}
        self.status2id: Dict[str, int] = {}
        self.scaler_mean: Dict[str, float] = {}
        self.scaler_std: Dict[str, float] = {}
        self.feature_order: List[str] = []
        self.feature_index: Dict[str, int] = {}
        self.horizons_steps: List[int] = []
        self.seq_len: int = 1
        self.base_step_sec: int = 4 * 60
        self.max_delta: int = 60
        self.short_gap_steps: int = 3
        self.long_gap_steps: int = 15
        self.backfill_short_gaps: bool = True
        self.model: Optional[Any] = None
        self.station_encoder: Any = None
        self.status_encoder: Any = None
        self.target_col: str = "num_bikes_available"
        self.delta_feature_names = ["delta_steps", "log1p_delta_steps", "is_gap"]
        self.buffers: Dict[int, LinearOnlineBuffer] = {}
        self.lock = asyncio.Lock()

    def dependencies_ready(self) -> bool:
        return all(dep is not None for dep in (np, joblib))

    def _meta_path(self) -> str:
        return os.path.join(self.meta_dir, "meta.json")

    def _load_sync(self, force: bool = False) -> None:
        if self.model is not None and not force:
            return
        if not self.dependencies_ready():
            raise RuntimeError("NumPy and joblib must be installed for inference.")

        meta_path = self._meta_path()
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Meta file not found: {meta_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Linear model artifact not found: {self.model_path}")

        with open(meta_path, "r") as handle:
            meta = json.load(handle)

        artifact = joblib.load(self.model_path)

        self.meta = meta
        self.station2id = {int(sid): idx for idx, sid in enumerate(meta["station_ids"])}
        self.status2id = {status: idx for idx, status in enumerate(meta["status_values"])}
        self.scaler_mean = {
            k: float(v) for k, v in artifact.get("scaler_mean", meta["scaler_mean"]).items()
        }
        self.scaler_std = {
            k: float(v) for k, v in artifact.get("scaler_std", meta["scaler_std"]).items()
        }
        self.feature_order = list(artifact["feat_cont"])
        self.feature_index = {name: idx for idx, name in enumerate(self.feature_order)}
        self.horizons_steps = [int(step) for step in artifact["horizons"]]
        self.seq_len = int(artifact["seq_len"])
        self.base_step_sec = int(meta.get("base_step_sec", 4 * 60))
        self.max_delta = int(meta.get("max_delta", 60))
        self.short_gap_steps = int(meta.get("short_gap_steps", 3))
        self.long_gap_steps = int(meta.get("long_gap_steps", 15))
        self.backfill_short_gaps = bool(meta.get("backfill_short_gaps", True))
        self.target_col = artifact.get("target_col", self.target_col)

        self.model = artifact["model"]
        self.station_encoder = artifact["station_encoder"]
        self.status_encoder = artifact["status_encoder"]

    async def ensure_loaded(self, force: bool = False) -> None:
        if self.model is not None and not force:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._load_sync, force)

    def _get_buffer(self, station_idx: int) -> LinearOnlineBuffer:
        buffer = self.buffers.get(station_idx)
        if buffer is None:
            buffer = LinearOnlineBuffer(self.seq_len)
            self.buffers[station_idx] = buffer
        return buffer

    def _standardize(self, values: Dict[str, float]) -> "np.ndarray":
        if np is None:
            raise RuntimeError("NumPy is required for inference.")
        cont = np.array([values.get(name, 0.0) for name in self.feature_order], dtype=np.float32)
        for idx, name in enumerate(self.feature_order):
            mean = self.scaler_mean[name]
            std = self.scaler_std[name]
            cont[idx] = (cont[idx] - mean) / (std + 1e-6)
        return cont.astype(np.float32)

    def _prepare(self, snapshot: "AvailabilitySnapshot") -> PreparedObservation:
        if self.model is None or not self.dependencies_ready():
            raise RuntimeError("Model not loaded or dependencies missing.")

        station_idx = self.station2id.get(snapshot.station_id)
        if station_idx is None:
            raise KeyError(f"Station {snapshot.station_id} not in training metadata.")

        dt = snapshot.collected_at
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ts_utc = int(dt.timestamp())
        dt_local = dt.astimezone(ZONE_EUROPE_MADRID)
        hour = dt_local.hour
        dow = dt_local.weekday()

        sin_hour = math.sin(2 * math.pi * hour / 24.0)
        cos_hour = math.cos(2 * math.pi * hour / 24.0)
        sin_dow = math.sin(2 * math.pi * dow / 7.0)
        cos_dow = math.cos(2 * math.pi * dow / 7.0)

        mechanical = snapshot.mechanical_bikes_available
        ebikes = snapshot.ebikes_available
        if mechanical is None and ebikes is None:
            mechanical = int(round(snapshot.bikes_available * 0.7))
            mechanical = max(0, min(snapshot.bikes_available, mechanical))
            ebikes = snapshot.bikes_available - mechanical
        else:
            mechanical = mechanical if mechanical is not None else max(
                snapshot.bikes_available - (ebikes or 0), 0
            )
            ebikes = ebikes if ebikes is not None else max(
                snapshot.bikes_available - mechanical, 0
            )

        feature_map: Dict[str, float] = {
            "num_bikes_available": float(snapshot.bikes_available),
            "num_bikes_available_types.mechanical": float(mechanical),
            "num_bikes_available_types.ebike": float(ebikes),
            "num_docks_available": float(snapshot.docks_available),
            "is_installed": float(1 if snapshot.is_installed else 0),
            "is_renting": float(1 if snapshot.is_renting else 0),
            "is_returning": float(1 if snapshot.is_returning else 0),
            "ttl": float(snapshot.ttl or 0),
            "sin_hour": float(sin_hour),
            "cos_hour": float(cos_hour),
            "sin_dow": float(sin_dow),
            "cos_dow": float(cos_dow),
        }

        buffer = self._get_buffer(station_idx)
        prev_ts = buffer.last_ts
        delta_steps = (
            0
            if prev_ts is None
            else max(0, int(round((ts_utc - prev_ts) / max(1, self.base_step_sec))))
        )
        delta_steps = min(delta_steps, self.max_delta)
        feature_map["delta_steps"] = float(delta_steps)
        feature_map["log1p_delta_steps"] = float(math.log1p(delta_steps))
        feature_map["is_gap"] = float(1.0 if delta_steps > 1 else 0.0)

        cont_std = self._standardize(feature_map)
        status = snapshot.status or ""
        status_id = self.status2id.get(status, len(self.status2id))

        return PreparedObservation(
            station_idx=station_idx,
            cont_std=cont_std,
            status_id=status_id,
            ts_utc=ts_utc,
            delta_steps=delta_steps,
        )

    def _append_short_gap_filler(self, buffer: LinearOnlineBuffer, steps: int) -> None:
        if np is None:
            raise RuntimeError("NumPy is required for inference.")
        if not buffer.cont:
            return
        last_cont = buffer.cont[-1]
        last_status = buffer.last_status()
        last_ts = buffer.last_ts or 0
        for offset in range(1, steps):
            filler = last_cont.copy()
            for name in self.delta_feature_names:
                if name not in self.feature_index:
                    continue
                idx = self.feature_index[name]
                raw_value = 1.0 if name != "log1p_delta_steps" else math.log1p(1.0)
                mean = self.scaler_mean[name]
                std = self.scaler_std[name]
                filler[idx] = (raw_value - mean) / (std + 1e-6)
            filler_ts = last_ts + offset * self.base_step_sec
            buffer.append(filler_ts, filler, last_status)

    def _ingest_snapshot_unlocked(self, snapshot: "AvailabilitySnapshot") -> None:
        try:
            prepared = self._prepare(snapshot)
        except KeyError:
            return
        except RuntimeError as exc:
            LOGGER.debug("Skipping snapshot ingestion: %s", exc)
            return

        buffer = self._get_buffer(prepared.station_idx)
        if buffer.last_ts is not None and prepared.ts_utc <= buffer.last_ts:
            return

        if prepared.delta_steps > self.long_gap_steps:
            buffer.clear()
        elif (
            self.backfill_short_gaps
            and 1 < prepared.delta_steps <= self.short_gap_steps
            and buffer.cont
        ):
            self._append_short_gap_filler(buffer, prepared.delta_steps)

        buffer.append(prepared.ts_utc, prepared.cont_std, prepared.status_id)

    def _predict_from_buffer_unlocked(self, station_idx: int) -> Optional[np.ndarray]:
        if self.model is None or np is None:
            return None
        buffer = self.buffers.get(station_idx)
        if buffer is None or not buffer.ready():
            return None

        x_cont = buffer.feature_vector().reshape(1, -1)
        station_arr = np.array([[station_idx]], dtype=np.int32)
        status_arr = np.array([[buffer.last_status()]], dtype=np.int32)

        station_ohe = self.station_encoder.transform(station_arr).astype(np.float32)
        status_ohe = self.status_encoder.transform(status_arr).astype(np.float32)

        X = np.hstack([x_cont.astype(np.float32), station_ohe, status_ohe])
        preds = self.model.predict(X)
        if preds.ndim == 1:
            preds = preds.reshape(1, -1)
        return preds[0].astype(float)

    async def ingest_batch(self, snapshots: List["AvailabilitySnapshot"]) -> None:
        if not snapshots or not self.dependencies_ready():
            return
        try:
            await self.ensure_loaded()
        except FileNotFoundError:
            return
        async with self.lock:
            for snapshot in sorted(
                snapshots, key=lambda item: item.collected_at or datetime.now(timezone.utc)
            ):
                self._ingest_snapshot_unlocked(snapshot)

    async def ensure_station_history(
        self, redis: RedisLike, station_id: int, max_batches: int = 200
    ) -> None:
        if not self.dependencies_ready():
            return
        try:
            await self.ensure_loaded()
        except FileNotFoundError:
            return

        station_idx = self.station2id.get(station_id)
        if station_idx is None:
            return

        async with self.lock:
            buffer = self.buffers.get(station_idx)
            last_ts = buffer.last_ts if buffer else None
        if last_ts is not None:
            return

        history = await fetch_snapshot_history(redis, max_batches)
        if not history:
            return

        snapshots: List[AvailabilitySnapshot] = []
        for batch in history:
            for snapshot in batch.stations:
                if snapshot.station_id == station_id:
                    snapshots.append(snapshot)
        await self.ingest_batch(snapshots)

    async def forecast(
        self,
        station_id: int,
        snapshot: "AvailabilitySnapshot",
        horizon_minutes: int,
    ) -> Tuple[List[Tuple[int, float]], "AvailabilitySnapshot"]:
        if not self.dependencies_ready():
            raise HTTPException(
                status_code=503,
                detail="ML dependencies (numpy/joblib) are not installed on the server.",
            )
        await self.ensure_loaded()
        async with self.lock:
            self._ingest_snapshot_unlocked(snapshot)
            station_idx = self.station2id.get(station_id)
            if station_idx is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Station {station_id} not present in trained model metadata.",
                )
            preds = self._predict_from_buffer_unlocked(station_idx)
        if preds is None:
            raise HTTPException(
                status_code=503,
                detail="Insufficient history for the requested station to run inference.",
            )

        step_minutes = [int(step * self.base_step_sec / 60) for step in self.horizons_steps]
        paired = list(zip(step_minutes, preds.tolist()))
        filtered = [item for item in paired if item[0] <= horizon_minutes]
        if not filtered:
            filtered = [paired[0]]
        return filtered, snapshot


class AvailabilitySnapshot(BaseModel):
    station_id: int
    bikes_available: int
    docks_available: int
    collected_at: datetime
    mechanical_bikes_available: Optional[int] = None
    ebikes_available: Optional[int] = None
    status: Optional[str] = None
    is_charging_station: Optional[bool] = None
    is_installed: Optional[bool] = None
    is_renting: Optional[bool] = None
    is_returning: Optional[bool] = None
    ttl: Optional[int] = None


class ForecastPoint(BaseModel):
    eta: datetime = Field(..., description="Estimated timestamp of the forecast slot.")
    bikes_available: int
    docks_available: int


class ForecastResponse(BaseModel):
    station_id: int
    horizon_minutes: int
    forecast: List[ForecastPoint]
    issued_at: datetime


class StationSnapshotBatch(BaseModel):
    collected_at: datetime
    stations: List[AvailabilitySnapshot]


def model_to_dict(model: BaseModel) -> Dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")  # Pydantic v2
    return json.loads(model.json())  # type: ignore[return-value]  # Pydantic v1 fallback


def parse_datetime(value: Any) -> datetime:
    if isinstance(value, str):
        normalised = value.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(normalised)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


async def create_redis_client() -> RedisLike:
    """Return a Redis client, falling back to FakeRedis when connection fails."""
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis = Redis.from_url(redis_url, encoding="utf-8", decode_responses=True)
    try:
        await redis.ping()
        LOGGER.info("Connected to Redis at %s", redis_url)
        return redis
    except Exception as exc:  # pragma: no cover - depends on runtime environment
        LOGGER.warning(
            "Failed to connect to Redis (%s). Using in-memory FakeRedis instead.", exc
        )
        try:
            await redis.aclose()
        except Exception:
            pass
        return FakeRedis()
        

async def store_station_snapshot(redis: RedisLike, snapshot: List[AvailabilitySnapshot]) -> None:
    if not snapshot:
        return

    collected_at = max(item.collected_at for item in snapshot)
    batch = StationSnapshotBatch(collected_at=collected_at, stations=snapshot)
    batch_serialised = json.dumps(model_to_dict(batch))
    await redis.zadd(SNAPSHOT_HISTORY_KEY, {batch_serialised: collected_at.timestamp()})

    latest_serialised = json.dumps([model_to_dict(item) for item in snapshot])
    await redis.set(LATEST_SNAPSHOT_KEY, latest_serialised)


async def fetch_latest_snapshot(redis: RedisLike) -> Optional[List[AvailabilitySnapshot]]:
    raw = await redis.get(LATEST_SNAPSHOT_KEY)
    if raw:
        payload: List[Dict[str, Any]] = json.loads(raw)
        return [AvailabilitySnapshot(**item) for item in payload]

    history = await redis.zrevrange(SNAPSHOT_HISTORY_KEY, 0, 0)
    if not history:
        return None

    latest_entry = json.loads(history[0])
    stations_payload = latest_entry.get("stations") or []
    return [AvailabilitySnapshot(**item) for item in stations_payload]


async def fetch_snapshot_history(redis: RedisLike, limit: int) -> List[StationSnapshotBatch]:
    if limit <= 0:
        return []

    raw_entries = await redis.zrevrange(SNAPSHOT_HISTORY_KEY, 0, limit - 1)
    if not raw_entries:
        return []

    # reverse so oldest snapshot is first
    batches: List[StationSnapshotBatch] = []
    for entry in reversed(raw_entries):
        payload = json.loads(entry)
        collected_value = payload.get("collected_at")
        stations_payload = payload.get("stations") or []
        collected_at = parse_datetime(collected_value)
        stations = [AvailabilitySnapshot(**item) for item in stations_payload]
        batches.append(StationSnapshotBatch(collected_at=collected_at, stations=stations))
    return batches


async def fetch_bicing_status() -> List[AvailabilitySnapshot]:
    """Fetch live Bicing station availability."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(BICING_STATUS_URL, 
                                        headers={"Authorization": BICING_AUTH_TOKEN})
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        LOGGER.warning("Failed to fetch Bicing data (%s).", exc)
        return []

    stations_data = payload.get("data", {}).get("stations")
    if not stations_data:
        LOGGER.warning("Bicing data payload missing stations.")
        return []

    last_updated_epoch = payload.get("last_updated")
    fallback_collected = (
        datetime.fromtimestamp(last_updated_epoch, tz=timezone.utc)
        if isinstance(last_updated_epoch, (int, float))
        else datetime.now(timezone.utc)
    )
    snapshots: List[AvailabilitySnapshot] = []

    ttl_value = payload.get("ttl")
    for station in stations_data:
        station_id_raw = station.get("station_id")
        if station_id_raw is None:
            continue
        try:
            station_id = int(station_id_raw)
        except (TypeError, ValueError):
            continue

        bikes = station.get("num_bikes_available", 0) or 0
        docks = station.get("num_docks_available", 0) or 0
        types = station.get("num_bikes_available_types") or {}
        mechanical = types.get("mechanical")
        ebikes = types.get("ebike")
        status = station.get("status")
        is_charging = station.get("is_charging_station")
        is_installed = station.get("is_installed")
        is_renting = station.get("is_renting")
        is_returning = station.get("is_returning")

        last_reported_epoch = station.get("last_reported")
        if isinstance(last_reported_epoch, (int, float)) and last_reported_epoch > 0:
            collected_at = datetime.fromtimestamp(last_reported_epoch, tz=timezone.utc)
        else:
            collected_at = fallback_collected

        snapshots.append(
            AvailabilitySnapshot(
                station_id=station_id,
                bikes_available=int(bikes),
                docks_available=int(docks),
                collected_at=collected_at,
                mechanical_bikes_available=(
                    int(mechanical) if mechanical is not None else None
                ),
                ebikes_available=int(ebikes) if ebikes is not None else None,
                status=status,
                is_charging_station=bool(is_charging) if is_charging is not None else None,
                is_installed=bool(is_installed) if is_installed is not None else None,
                is_renting=bool(is_renting) if is_renting is not None else None,
                is_returning=bool(is_returning) if is_returning is not None else None,
                ttl=int(ttl_value) if ttl_value is not None else None,
            )
        )

    return snapshots


def build_mock_forecast(snapshot: AvailabilitySnapshot, horizon_minutes: int) -> List[ForecastPoint]:
    steps = max(1, horizon_minutes // 15)
    base_time = datetime.now(timezone.utc)
    forecast: List[ForecastPoint] = []
    bikes = snapshot.bikes_available
    docks = snapshot.docks_available

    for step in range(1, steps + 1):
        drift = randint(-2, 2)
        bikes = max(0, bikes + drift)
        docks = max(0, docks - drift)
        eta = base_time + timedelta(minutes=step * (horizon_minutes // steps))
        forecast.append(
            ForecastPoint(
                eta=eta,
                bikes_available=bikes,
                docks_available=docks,
            )
        )
    return forecast


async def poll_bicing_status(
    redis: RedisLike,
    stop_event: asyncio.Event,
    forecast_service: "BicingLinearForecastService",
    interval_seconds: int,
) -> None:
    """Background job that keeps the latest Bicing snapshot cached in Redis."""
    LOGGER.info("Starting Bicing status poller (interval=%ss)", interval_seconds)
    while not stop_event.is_set():
        snapshot = await fetch_bicing_status()
        if snapshot:
            await store_station_snapshot(redis, snapshot)
            await forecast_service.ingest_batch(snapshot)
            LOGGER.debug("Stored snapshot for %d stations", len(snapshot))
        else:
            LOGGER.warning("Fetched snapshot is empty; skipping persistence.")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue
    LOGGER.info("Bicing status poller stopped")


@asynccontextmanager
async def lifespan(app: FastAPI):
    redis = await create_redis_client()
    app.state.redis = redis
    forecast_service = BicingLinearForecastService()
    app.state.forecast_service = forecast_service
    try:
        await forecast_service.ensure_loaded()
        LOGGER.info("Forecast model loaded successfully.")
    except FileNotFoundError:
        LOGGER.warning(
            "Forecast artifacts not found. Train the model before forecasting is available."
        )

    stop_event = asyncio.Event()
    poller_task = asyncio.create_task(
        poll_bicing_status(redis, stop_event, forecast_service, POLL_INTERVAL_SECONDS)
    )

    try:
        yield
    finally:
        stop_event.set()
        await poller_task
        close = getattr(redis, "aclose", None)
        if callable(close):
            await close()
        else:
            close = getattr(redis, "close", None)
            if callable(close):
                await close()


app = FastAPI(title="Barcelona Bicing Forecast API", lifespan=lifespan)


async def get_redis(request: Request) -> RedisLike:
    return request.app.state.redis


async def get_forecast_service(request: Request) -> BicingLinearForecastService:
    return request.app.state.forecast_service


@app.get("/stations/history", response_model=List[StationSnapshotBatch])
async def stations_history(
    limit: int = Query(
        10,
        ge=1,
        le=500,
        description="Number of cached snapshots to return, ordered from oldest to newest.",
    ),
    redis: RedisLike = Depends(get_redis),
) -> List[StationSnapshotBatch]:
    history = await fetch_snapshot_history(redis, limit)
    if not history:
        raise HTTPException(status_code=503, detail="Station snapshots not yet available.")
    return history


@app.get("/forecast", response_model=ForecastResponse)
async def forecast_availability(
    station_id: int = Query(..., ge=0, description="Target station identifier"),
    horizon_minutes: int = Query(
        60,
        ge=5,
        le=240,
        description="Forecast horizon in minutes (default 60, max 240).",
    ),
    redis: RedisLike = Depends(get_redis),
    forecast_service: BicingLinearForecastService = Depends(get_forecast_service),
) -> ForecastResponse:
    snapshot = await fetch_latest_snapshot(redis)
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Station data not yet available.")

    station = next((item for item in snapshot if item.station_id == station_id), None)
    if station is None:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found.")

    try:
        await forecast_service.ensure_station_history(redis, station_id)
        predictions, latest_snapshot = await forecast_service.forecast(
            station_id, station, horizon_minutes
        )
        base_time = latest_snapshot.collected_at
        if base_time.tzinfo is None:
            base_time = base_time.replace(tzinfo=timezone.utc)
        total_capacity = station.bikes_available + station.docks_available
        forecast_points: List[ForecastPoint] = []
        for minutes_ahead, bikes_pred in predictions:
            eta = base_time + timedelta(minutes=minutes_ahead)
            bikes_count = max(0, int(round(bikes_pred)))
            docks_count = max(0, total_capacity - bikes_count)
            forecast_points.append(
                ForecastPoint(
                    eta=eta,
                    bikes_available=bikes_count,
                    docks_available=docks_count,
                )
            )
    except HTTPException as exc:
        if exc.status_code != 503:
            raise
        LOGGER.warning("Forecast model unavailable (%s); using heuristic fallback.", exc.detail)
        forecast_points = build_mock_forecast(station, horizon_minutes)
    except Exception as exc:  # pragma: no cover - fallback safety
        LOGGER.warning("Forecast model error (%s); using heuristic fallback.", exc)
        forecast_points = build_mock_forecast(station, horizon_minutes)

    return ForecastResponse(
        station_id=station_id,
        horizon_minutes=horizon_minutes,
        forecast=forecast_points,
        issued_at=datetime.now(timezone.utc),
    )


@app.get("/stations/latest", response_model=List[AvailabilitySnapshot])
async def latest_station_snapshot(redis: RedisLike = Depends(get_redis)) -> List[AvailabilitySnapshot]:
    snapshot = await fetch_latest_snapshot(redis)
    if snapshot is None:
        raise HTTPException(status_code=503, detail="Station data not yet available.")
    return snapshot


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_lr:app", host="0.0.0.0", port=8000, reload=True)
