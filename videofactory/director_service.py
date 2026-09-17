"""Create a validated, timestamp-aligned DirectorPlan from one provider call."""

from __future__ import annotations

from typing import Any

from .director_context import DirectorContextBuilder
from .director_plan import align_plan_to_words, validate_plan
from .director_provider import DirectorProvider


def create_director_plan(project_name: str, mode: str, transcript: dict[str, Any],
                         provider: DirectorProvider) -> dict[str, Any]:
    context = DirectorContextBuilder().build(project_name, mode, transcript)
    result = provider.create(context)
    duration = float(context["duration"])
    validate_plan(result.plan, project_name, duration, mode)
    aligned = align_plan_to_words(result.plan, transcript)
    validate_plan(aligned, project_name, duration, mode)
    aligned["metadata"] = {
        "provider": result.provider, "model": result.model,
        "invocation_count": result.invocation_count,
        "elapsed_seconds": result.elapsed_seconds,
    }
    validate_plan(aligned, project_name, duration, mode)
    return aligned
