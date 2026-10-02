"""Run the exported FIER /rank demo using a single Uvicorn worker."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--dynamic-batching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="merge concurrent requests into short-lived inference batches",
    )
    parser.add_argument("--max-batch-requests", type=int, default=8)
    parser.add_argument("--max-batch-candidates", type=int, default=256)
    parser.add_argument("--max-batch-wait-ms", type=float, default=2.0)
    parser.add_argument("--max-queue-size", type=int, default=1024)
    args = parser.parse_args()
    if args.torch_threads <= 0:
        parser.error("--torch-threads must be positive")
    if args.max_batch_requests <= 0:
        parser.error("--max-batch-requests must be positive")
    if args.max_batch_candidates <= 0:
        parser.error("--max-batch-candidates must be positive")
    if args.max_batch_wait_ms < 0.0:
        parser.error("--max-batch-wait-ms cannot be negative")
    if args.max_queue_size <= 0:
        parser.error("--max-queue-size must be positive")
    import torch
    import uvicorn
    from src.serving.api import create_app

    torch.set_num_threads(args.torch_threads)
    app = create_app(
        args.artifact_directory,
        device=args.device,
        dynamic_batching=args.dynamic_batching,
        max_batch_requests=args.max_batch_requests,
        max_batch_candidates=args.max_batch_candidates,
        max_batch_wait_ms=args.max_batch_wait_ms,
        max_queue_size=args.max_queue_size,
    )
    uvicorn.run(app, host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
