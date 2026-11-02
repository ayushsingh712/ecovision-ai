"""Dependency accessors backed by the application state."""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import Depends, Request

from backend.core.config import Settings
from backend.database.database import Database
from backend.services.advice_service import AdviceService
from backend.services.model_service import ModelService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def get_advice_service(request: Request) -> AdviceService:
    return request.app.state.advice_service


def get_inference_semaphore(request: Request) -> asyncio.Semaphore:
    return request.app.state.inference_semaphore


SettingsDep = Annotated[Settings, Depends(get_settings)]
DatabaseDep = Annotated[Database, Depends(get_database)]
ModelServiceDep = Annotated[ModelService, Depends(get_model_service)]
AdviceServiceDep = Annotated[AdviceService, Depends(get_advice_service)]
SemaphoreDep = Annotated[asyncio.Semaphore, Depends(get_inference_semaphore)]
