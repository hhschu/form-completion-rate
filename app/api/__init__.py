"""Initialise model container."""

from pydantic import BaseSettings

from app.api.deps import LRUModelContainer


class Settings(BaseSettings):
    max_num_models: int = 1
    model_dir: str = "model"


settings = Settings()
Models = LRUModelContainer(
    maxsize=settings.max_num_models, model_dir=settings.model_dir
)
Models.load()
