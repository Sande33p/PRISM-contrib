"""Versioned HTTP boundary for an explicitly injected text-reference adapter.

This route does not configure a runtime, render prompts, or load model assets.
Cooperative cancellation is available through the adapter API; disconnecting an
HTTP client does not signal cancellation or establish that execution stopped.
"""

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, StrictInt, StrictStr
from src.backends.text_contracts import TextRequest
from src.backends.u2_reference import (
    LedgerCapacityError,
    RequestConflictError,
    RequestInProgressError,
    U2ReferenceAdapter,
)


class ReferenceTextRequest(BaseModel):
    """Strict JSON representation of contract-v1 pretokenized text input."""

    if hasattr(BaseModel, "model_validate"):
        model_config = {"strict": True, "extra": "forbid"}
    else:

        class Config:
            extra = "forbid"

    # Strict scalar types work with both Pydantic majors. The canonical
    # TextRequest owns all length/range checks, avoiding incompatible v1/v2
    # Field constraints and keeping one authoritative resource-limit contract.
    request_id: StrictStr
    prompt_ids: list[StrictInt]
    tokenizer_digest: StrictStr
    max_new_tokens: StrictInt = 128
    max_model_calls: StrictInt = 128
    block_size: StrictInt = 32
    steps_per_block: StrictInt = 32
    seed: StrictInt = 0
    contract_version: StrictInt = 1

    def to_contract(self) -> TextRequest:
        """Copy JSON token IDs into the canonical immutable request contract."""
        return TextRequest(
            request_id=self.request_id,
            prompt_ids=tuple(self.prompt_ids),
            tokenizer_digest=self.tokenizer_digest,
            max_new_tokens=self.max_new_tokens,
            max_model_calls=self.max_model_calls,
            block_size=self.block_size,
            steps_per_block=self.steps_per_block,
            seed=self.seed,
            contract_version=self.contract_version,
        )


def get_reference_adapter(request: Request) -> U2ReferenceAdapter:
    """Return this app's injected adapter, without creating a default runtime."""
    adapter = getattr(request.app.state, "reference_adapter", None)
    if adapter is None:
        raise HTTPException(status_code=503, detail={"code": "reference_adapter_unconfigured"})
    return adapter


reference_router = APIRouter()


@reference_router.post("/v1/backends/u2_reference/generate")
def generate_reference_text(
    request: ReferenceTextRequest,
    adapter: U2ReferenceAdapter = Depends(get_reference_adapter),  # noqa: B008
) -> dict[str, Any]:
    """Return the canonical committed-output receipt from the injected adapter.

    This synchronous handler runs in FastAPI's threadpool. Runtime failures are
    terminal adapter receipts; no fallback or retry is performed by this route.
    """
    try:
        result = adapter.generate(request.to_contract())
    except RequestConflictError:
        raise HTTPException(status_code=409, detail={"code": "request_conflict"}) from None
    except RequestInProgressError:
        raise HTTPException(status_code=409, detail={"code": "request_in_progress"}) from None
    except LedgerCapacityError:
        raise HTTPException(status_code=503, detail={"code": "reference_ledger_capacity"}) from None
    except ValueError:
        raise HTTPException(status_code=422, detail={"code": "invalid_reference_request"}) from None
    return asdict(result)
