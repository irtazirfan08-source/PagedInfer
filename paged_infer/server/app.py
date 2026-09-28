import json
import uuid
from typing import Iterator, List
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from paged_infer.memory.block_allocator import BlockAllocator
from paged_infer.engine.scheduler import Scheduler
from paged_infer.engine.sequence import Sequence
from paged_infer.engine.runner import ModelRunner

app = FastAPI(title="PagedInfer Serving Engine", version="0.1.0")

# Runtime state
allocator = BlockAllocator(num_blocks=32, block_size=16)
scheduler = Scheduler(block_allocator=allocator, max_batch_size=4)
runner = ModelRunner()


class GenerateRequest(BaseModel):
    prompt_tokens: List[int] = Field(..., min_length=1, description="List of prompt token IDs")
    max_new_tokens: int = Field(default=5, ge=1, le=64, description="Number of tokens to generate")


class GenerateResponse(BaseModel):
    seq_id: str
    output_tokens: List[int]
    num_generated: int


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "free_blocks": allocator.get_num_free_blocks(),
        "total_blocks": allocator.num_blocks,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    seq_id = f"req-{uuid.uuid4().hex[:6]}"
    sequence = Sequence(
        seq_id=seq_id,
        prompt_tokens=req.prompt_tokens,
        max_new_tokens=req.max_new_tokens,
    )
    scheduler.add_sequence(sequence)

    max_steps = req.max_new_tokens + 5
    steps = 0
    is_done = False

    while not is_done and steps < max_steps:
        steps += 1
        active_batch = scheduler.schedule()
        if not active_batch:
            break

        step_outputs = runner.step(active_batch)
        for seq in active_batch:
            token = step_outputs.get(seq.seq_id, 0)
            allocator.append_slot(seq.seq_id, seq.get_len())
            seq.append_token(token)

        finished_seqs = scheduler.post_step()
        for finished in finished_seqs:
            if finished.seq_id == seq_id:
                is_done = True
                break

    return GenerateResponse(
        seq_id=sequence.seq_id,
        output_tokens=sequence.output_tokens,
        num_generated=len(sequence.output_tokens),
    )


def stream_token_generator(seq_id: str, max_new_tokens: int) -> Iterator[str]:
    """Yields per-token SSE chunks as iterations complete in the engine."""
    max_steps = max_new_tokens + 5
    steps = 0
    is_done = False

    while not is_done and steps < max_steps:
        steps += 1
        active_batch = scheduler.schedule()
        if not active_batch:
            break

        step_outputs = runner.step(active_batch)
        emitted_token = None

        for seq in active_batch:
            token = step_outputs.get(seq.seq_id, 0)
            allocator.append_slot(seq.seq_id, seq.get_len())
            seq.append_token(token)
            if seq.seq_id == seq_id:
                emitted_token = token

        if emitted_token is not None:
            chunk = json.dumps({"seq_id": seq_id, "token_id": emitted_token})
            yield f"data: {chunk}\n\n"

        finished_seqs = scheduler.post_step()
        for finished in finished_seqs:
            if finished.seq_id == seq_id:
                is_done = True
                break

    yield "data: [DONE]\n\n"


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    seq_id = f"stream-{uuid.uuid4().hex[:6]}"
    sequence = Sequence(
        seq_id=seq_id,
        prompt_tokens=req.prompt_tokens,
        max_new_tokens=req.max_new_tokens,
    )
    scheduler.add_sequence(sequence)

    return StreamingResponse(
        stream_token_generator(seq_id, req.max_new_tokens),
        media_type="text/event-stream",
    )