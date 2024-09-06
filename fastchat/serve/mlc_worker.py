"""
A model worker that executes the model based on LightLLM.

See documentations at docs/lightllm_integration.md
"""

import argparse
import asyncio
import json
import os
import time
import torch
import uvicorn

from transformers import AutoConfig

from typing import List

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse

from fastchat.serve.base_model_worker import BaseModelWorker
from fastchat.serve.model_worker import (
    logger,
    worker_id,
)
from fastchat.utils import get_context_length, is_partial_stop

from mlc_llm.serve import engine, engine_utils
from mlc_llm.serve.server import ServerContext
from mlc_llm.serve.server import ServerContext
from mlc_llm.protocol import error_protocol
from mlc_llm.interface.help import HELP
from mlc_llm.support.argparse import ArgumentParser
from mlc_llm.interface.serve import serve
from http import HTTPStatus
app = FastAPI()


class MLCWorker(BaseModelWorker):
    def __init__(
        self,
        controller_addr: str,
        worker_addr: str,
        worker_id: str,
        model_path: str,
        model_names: List[str],
        limit_worker_concurrency: int,
        no_register: bool,
        conv_template: str,
        llm_engine, 
        context_len,
    ):
        super().__init__(
            controller_addr,
            worker_addr,
            worker_id,
            model_path,
            model_names,
            limit_worker_concurrency,
            conv_template,
        )

        logger.info(
            f"Loading the model {self.model_names} on worker {worker_id}, worker type: LightLLM worker..."
        )
        self.tokenizer = llm_engine
        self.context_len = context_len

        self.is_first = True

        if not no_register:
            self.init_heart_beat()

    async def generate_stream(self, params):

        server_context: ServerContext = ServerContext.current()
        request_final_usage_include_extra = server_context.enable_debug
        # request_include_debug_config = server_context.enable_debug
        model = "qwen"
        # if not request_include_debug_config:
        #     request.debug_config = None
        print(f"{params=}")
        async_engine = server_context.get_engine(model)
        # if async_engine is None:
        #     return error_protocol.create_error_response(
        #         HTTPStatus.BAD_REQUEST, message=f'The requested model "{model}" is not served.'
        #     )
        request_id = params.pop("request_id")
        request = params.get("request", None)

        # Streaming response.
        # We manually get the first response from generator to
        # capture potential exceptions in this scope, rather then
        # the StreamingResponse scope.
        stream_generator = async_engine._handle_completion(  # pylint: disable=protected-access
            request, request_id, request_final_usage_include_extra=request_final_usage_include_extra
        )
        first_response = await anext(  # type: ignore  # pylint: disable=undefined-variable
            stream_generator
        )

        if isinstance(first_response, StopAsyncIteration):
            yield "data: [DONE]\n\n"
            return
        yield f"data: {first_response.model_dump_json(by_alias=True)}\n\n"
        
        async for response in stream_generator:
            yield f"data: {response.model_dump_json(by_alias=True)}\n\n"
        yield "data: [DONE]\n\n"
  

    async def generate(self, params):
        async for x in self.generate_stream(params):
            pass
        return json.loads(x[:-1].decode())


def release_worker_semaphore():
    worker.semaphore.release()


def acquire_worker_semaphore():
    if worker.semaphore is None:
        worker.semaphore = asyncio.Semaphore(worker.limit_worker_concurrency)
    return worker.semaphore.acquire()


def create_background_tasks(request_id):
    async def abort_request() -> None:
        await engine.abort(request_id)

    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_worker_semaphore)
    background_tasks.add_task(abort_request)
    return background_tasks


@app.post("/worker_generate_stream")
async def api_generate_stream(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = f"cmpl-{engine_utils.random_uuid()}"
    params["request_id"] = request_id
    params["request"] = request
    generator = worker.generate_stream(params)
    background_tasks = create_background_tasks(request_id)
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_generate")
async def api_generate(request: Request):
    params = await request.json()
    await acquire_worker_semaphore()
    request_id = f"cmpl-{engine_utils.random_uuid()}"
    params["request_id"] = request_id
    params["request"] = request
    output = await worker.generate(params)
    release_worker_semaphore()
    return JSONResponse(output)


@app.post("/worker_get_status")
async def api_get_status(request: Request):
    return worker.get_status()


@app.post("/count_token")
async def api_count_token(request: Request):
    params = await request.json()
    return worker.count_token(params)


@app.post("/worker_get_conv_template")
async def api_get_conv(request: Request):
    return worker.get_conv_template()


@app.post("/model_details")
async def api_model_details(request: Request):
    return {"context_length": worker.context_len}


import dataclasses
import json
from io import StringIO
from typing import Literal, Optional

@dataclasses.dataclass
class EngineConfigOverride:  # pylint: disable=too-many-instance-attributes
    """Arguments for overriding engine config."""

    # Overrides for EngineConfig (runtime)
    max_num_sequence: Optional[int] = None
    max_total_seq_length: Optional[int] = None
    prefill_chunk_size: Optional[int] = None
    max_history_size: Optional[int] = None
    gpu_memory_utilization: Optional[float] = None
    spec_draft_length: Optional[int] = None
    prefix_cache_mode: Optional[Literal["disable", "radix"]] = None
    prefix_cache_max_num_recycling_seqs: Optional[int] = None
    prefill_mode: Optional[Literal["chunked", "hybrid"]] = None
    context_window_size: Optional[int] = None
    sliding_window_size: Optional[int] = None
    attention_sink_size: Optional[int] = None
    tensor_parallel_shards: Optional[int] = None
    pipeline_parallel_stages: Optional[int] = None

    def __repr__(self) -> str:
        out = StringIO()
        print(f"max_num_sequence={self.max_num_sequence}", file=out, end="")
        print(f";max_total_seq_length={self.max_total_seq_length}", file=out, end="")
        print(f";prefill_chunk_size={self.prefill_chunk_size}", file=out, end="")
        print(f";max_history_size={self.max_history_size}", file=out, end="")
        print(f";gpu_memory_utilization={self.gpu_memory_utilization}", file=out, end="")
        print(f";spec_draft_length={self.spec_draft_length}", file=out, end="")
        print(f";prefix_cache_mode={self.prefix_cache_mode}", file=out, end="")
        print(
            f";prefix_cache_max_num_recycling_seqs={self.prefix_cache_max_num_recycling_seqs}",
            file=out,
            end="",
        )
        print(f";prefill_mode={self.prefill_mode}", file=out, end="")
        print(f";context_window_size={self.context_window_size}", file=out, end="")
        print(f";sliding_window_size={self.sliding_window_size}", file=out, end="")
        print(f";attention_sink_size={self.attention_sink_size}", file=out, end="")
        print(f";tensor_parallel_shards={self.tensor_parallel_shards}", file=out, end="")
        print(f";pipeline_parallel_stages={self.pipeline_parallel_stages}", file=out, end="")
        return out.getvalue().rstrip()

    @staticmethod
    def from_str(source: str) -> "EngineConfigOverride":
        """Parse engine config override values from a string."""
        parser = argparse.ArgumentParser(description="Engine config override values")

        parser.add_argument("--max_num_sequence", type=int, default=None)
        parser.add_argument("--max_total_seq_length", type=int, default=None)
        parser.add_argument("--prefill_chunk_size", type=int, default=None)
        parser.add_argument("--max_history_size", type=int, default=None)
        parser.add_argument("--gpu_memory_utilization", type=float, default=None)
        parser.add_argument("--spec_draft_length", type=int, default=None)
        parser.add_argument("--prefix_cache_mode", type=str, default="radix")
        parser.add_argument("--prefix_cache_max_num_recycling_seqs", type=int, default=None)
        parser.add_argument("--prefill_mode", type=str, default="hybrid")
        parser.add_argument("--context_window_size", type=int, default=None)
        parser.add_argument("--sliding_window_size", type=int, default=None)
        parser.add_argument("--attention_sink_size", type=int, default=None)
        parser.add_argument("--tensor_parallel_shards", type=int, default=None)
        parser.add_argument("--pipeline_parallel_stages", type=int, default=None)
        results = parser.parse_args([f"--{i}" for i in source.split(";") if i])
        return EngineConfigOverride(
            max_num_sequence=results.max_num_sequence,
            max_total_seq_length=results.max_total_seq_length,
            prefill_chunk_size=results.prefill_chunk_size,
            max_history_size=results.max_history_size,
            gpu_memory_utilization=results.gpu_memory_utilization,
            spec_draft_length=results.spec_draft_length,
            prefix_cache_mode=results.prefix_cache_mode,
            prefix_cache_max_num_recycling_seqs=results.prefix_cache_max_num_recycling_seqs,
            prefill_mode=results.prefill_mode,
            context_window_size=results.context_window_size,
            sliding_window_size=results.sliding_window_size,
            attention_sink_size=results.attention_sink_size,
            tensor_parallel_shards=results.tensor_parallel_shards,
            pipeline_parallel_stages=results.pipeline_parallel_stages,
        )


if __name__ == "__main__":

    parser = ArgumentParser("MLC LLM Serve CLI")

    # parser.add_argument(
    #     "model",
    #     type=str,
    #     help=HELP["model"] + " (required)",
    # )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=HELP["device_deploy"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--model-lib",
        type=str,
        default=None,
        help=HELP["model_lib"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["local", "interactive", "server"],
        default="local",
        help=HELP["mode_serve"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--enable-debug",
        action="store_true",
        help="whether we enable debug end points and debug config when accepting requests",
    )
    parser.add_argument(
        "--additional-models", type=str, nargs="*", help=HELP["additional_models_serve"]
    )
    parser.add_argument(
        "--speculative-mode",
        type=str,
        choices=["disable", "small_draft", "eagle", "medusa"],
        default="disable",
        help=HELP["speculative_mode_serve"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--prefix-cache-mode",
        type=str,
        choices=["disable", "radix"],
        default="radix",
        help=HELP["prefix_cache_mode_serve"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--prefill-mode",
        type=str,
        choices=["hybrid", "chunked"],
        default="hybrid",
        help=HELP["prefill_mode"] + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--overrides",
        type=EngineConfigOverride.from_str,
        default="",
        help=HELP["overrides_serve"],
    )
    parser.add_argument("--enable-tracing", action="store_true", help=HELP["enable_tracing_serve"])
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="host name" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port" + ' (default: "%(default)s")',
    )
    parser.add_argument("--allow-credentials", action="store_true", help="allow credentials")
    parser.add_argument(
        "--allow-origins",
        type=json.loads,
        default=["*"],
        help="allowed origins" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--allow-methods",
        type=json.loads,
        default=["*"],
        help="allowed methods" + ' (default: "%(default)s")',
    )
    parser.add_argument(
        "--allow-headers",
        type=json.loads,
        default=["*"],
        help="allowed headers" + ' (default: "%(default)s")',
    )
    # ----------------------
    parser.add_argument("--model-path", type=str, default="lmsys/vicuna-7b-v1.5")

    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--no-register", action="store_true")
    parser.add_argument("--limit_worker_concurrency", type=int, default=1024)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--gpus",
        type=str,
        default="3,4",
        help="A single GPU like 1 or multiple GPUs like 0,2",
    )
    args = parser.parse_args()
    if args.gpus:
        if len(args.gpus.split(",")) < args.num_gpus:
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.model_path:
        args.model = args.model_path
    print(f"###{args.model}")

    additional_models = []
    if args.additional_models is not None:
        for additional_model in args.additional_models:
            splits = additional_model.split(",", maxsplit=1)
            if len(splits) == 2:
                additional_models.append((splits[0], splits[1]))
            else:
                additional_models.append(splits[0])

    async_engine = engine.AsyncMLCEngine(
        model=args.model,
        device=args.device,
        model_lib=args.model_lib,
        mode=args.mode,
        engine_config=engine.EngineConfig(
            additional_models=additional_models,
            tensor_parallel_shards=args.overrides.tensor_parallel_shards,
            pipeline_parallel_stages=args.overrides.pipeline_parallel_stages,
            max_num_sequence=args.overrides.max_num_sequence,
            max_total_sequence_length=args.overrides.max_total_seq_length,
            max_single_sequence_length=args.overrides.context_window_size,
            prefill_chunk_size=args.overrides.prefill_chunk_size,
            sliding_window_size=args.overrides.sliding_window_size,
            attention_sink_size=args.overrides.attention_sink_size,
            max_history_size=args.overrides.max_history_size,
            gpu_memory_utilization=args.overrides.gpu_memory_utilization,
            speculative_mode=args.speculative_mode,
            spec_draft_length=args.overrides.spec_draft_length,
            prefix_cache_mode=args.prefix_cache_mode,
            prefix_cache_max_num_recycling_seqs=args.overrides.prefix_cache_max_num_recycling_seqs,
            prefill_mode=args.prefill_mode,
        ),
        enable_tracing=args.enable_tracing,
    )
    worker = MLCWorker(
        args.controller_address,
        args.worker_address,
        worker_id,
        args.model_dir,
        args.model,
        args.limit_worker_concurrency,
        args.no_register,
        async_engine,
        args.conv_template,
    )
    
    with ServerContext() as server_context:
        server_context.add_model(args.model, async_engine)
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
