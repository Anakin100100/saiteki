from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class OptimizationTask(BaseModel):
    id: str
    running: bool
    userId: str
    optimizedFunc: str
    validateResultFunc: str
    generateMetricsFunc: str
    createdAt: datetime
    updatedAt: datetime
    logs: Optional[str] = None
