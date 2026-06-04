"""Dramatiq broker setup using Redis."""

import os
import dramatiq
from dramatiq.brokers.redis import RedisBroker
from dramatiq.middleware import TimeLimit

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

broker = RedisBroker(url=REDIS_URL)
dramatiq.set_broker(broker)
