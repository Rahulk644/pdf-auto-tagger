"""Modal app: Gemma 4 E4B served via vLLM — the production-grade path.

Why vLLM over transformers.generate():
  - Continuous batching: multiple chunks share the GPU efficiently instead
    of serializing / contending (the bug that timed out our bf16 run).
  - PagedAttention: efficient KV-cache memory management.
  - Optimized fused kernels for H100 (bf16 tensor cores, no per-layer
    Python dispatch overhead).
  Expected: ~50-100 tok/s vs transformers' ~5-10 tok/s on this model.

Same HTTP contract as modal_gemma_e4b.py so run_corpus_modal.py is
unchanged: POST {image_b64, prompt, max_tokens} -> {response, ...}.

DEPLOY:
  modal deploy modal_gemma_vllm.py
Then point the runner at the vLLM endpoint:
  MODAL_URL=https://aurax--prep-qa-gemma-vllm-vllmgemma-generate.modal.run \
  python run_corpus_modal.py Miramar
"""

import modal

app = modal.App("prep-qa-gemma-vllm")

# CUDA-devel base provides nvcc, which flashinfer needs to JIT-compile
# kernels at engine startup. debian_slim lacks it → engine crash.
vllm_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04",
        add_python="3.12",
    )
    .pip_install(
        "vllm>=0.7.0",          # recent — best chance of Gemma 4 support
        "fastapi[standard]",
        "pillow>=10.0.0",
        "hf-transfer>=0.1.0",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        # Force Triton attention (Gemma 4 heterogeneous head dims) and skip
        # the flashinfer sampler JIT path as a belt-and-suspenders measure.
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",
    })
)

hf_cache_vol = modal.Volume.from_name("hf-cache-gemma4-e4b", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache-gemma4", create_if_missing=True)

MODEL_ID = "google/gemma-4-E4B-it"
GPU = "H100"


def _extract_json_array(text: str) -> str:
    """Pull the final JSON array out of a (possibly thinking-prefixed) response.

    With enable_thinking on, Gemma emits a `<|channel>thought ... <channel|>`
    reasoning trace before the answer. That trace often *echoes the input*
    array, so a naive 'first [' grab returns the wrong thing. The real answer
    is always the LAST balanced [...] in the output, so we scan back from the
    final ']' and bracket-match to its opening '['. Falls back to the raw text
    if no balanced array is found (caller's parser then handles it / flags it).
    """
    end = text.rfind("]")
    if end == -1:
        return text
    depth = 0
    for i in range(end, -1, -1):
        c = text[i]
        if c == "]":
            depth += 1
        elif c == "[":
            depth -= 1
            if depth == 0:
                return text[i:end + 1]
    return text


@app.cls(
    image=vllm_image,
    gpu=GPU,
    secrets=[modal.Secret.from_name("huggingface-prep")],
    scaledown_window=300,
    timeout=1200,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    max_containers=20,           # ceiling, not a floor — only spins up as many as
                                 # the client drives (runner's PARALLEL env). Headroom
                                 # so a faster corpus run (PARALLEL=20) has containers.
)
@modal.concurrent(max_inputs=1)   # serialize per container: one tensor shape at a
                                  # time. Gemma 4's heterogeneous head dims force the
                                  # Triton attn backend, which JIT-compiles kernels for
                                  # each new shape. Concurrent multi-shape batches made
                                  # it compile mid-inference → hang. Parallelism comes
                                  # from container fan-out (max_containers) instead.
class VLLMGemma:
    @modal.enter()
    def load(self):
        import os
        from vllm import LLM, SamplingParams

        token = (os.environ.get("HF_TOKEN")
                 or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if token:
            os.environ["HF_TOKEN"] = token
            os.environ["HUGGING_FACE_HUB_TOKEN"] = token

        # bf16 — vLLM's batched decoding avoids the memory-bandwidth wall
        # that naive transformers.generate() hit. H100 has plenty of VRAM.
        self.llm = LLM(
            model=MODEL_ID,
            dtype="bfloat16",
            max_model_len=12288,           # thinking trace + JSON answer need room
                                           # (prompt ~2.5k + reasoning ~5k + answer)
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": 1},
            enforce_eager=False,           # use CUDA graphs for speed
            max_num_seqs=1,                # one sequence at a time — never batch two
                                           # different shapes into one forward pass
            trust_remote_code=True,
        )
        self.SamplingParams = SamplingParams

        # Warmup: force Triton to JIT-compile its attention / slot-mapping kernels
        # NOW (at startup), not during the first real request. Triton specializes
        # kernels on the head dims (fixed by the model), so compiling once with a
        # tiny image covers the shapes real requests reuse. Cached in-memory and on
        # the vllm_cache_vol, so this cost is paid once per cold container.
        try:
            from PIL import Image
            warm_img = Image.new("RGB", (64, 64), (255, 255, 255))
            warm_msgs = [{
                "role": "user",
                "content": [
                    {"type": "image_pil", "image_pil": warm_img},
                    {"type": "text", "text": "Reply with OK."},
                ],
            }]
            self.llm.chat(warm_msgs, SamplingParams(temperature=0.0, max_tokens=8))
            print("[vllm] Warmup complete — Triton attention kernels compiled")
        except Exception as e:
            print(f"[vllm] Warmup failed (non-fatal, first request will compile): {e}")

        print(f"[vllm] Loaded {MODEL_ID} on {GPU} (bf16, vLLM)")

    @modal.fastapi_endpoint(method="POST")
    def generate(self, payload: dict) -> dict:
        import base64
        import io
        import time

        from PIL import Image

        image_b64 = payload["image_b64"]
        prompt_text = payload["prompt"]
        # Thinking burns output tokens (a reasoning trace precedes the JSON), so
        # default the budget high. max_model_len is 8192 and prompts run ~2.1k
        # tokens incl. the image, leaving ~6k for output. Caller can override.
        max_tokens = int(payload.get("max_tokens", 8000))
        # Match the phone config that scored 13/13 on Miramar: temp=0.1, thinking on.
        temperature = float(payload.get("temperature", 0.1))
        enable_thinking = bool(payload.get("enable_thinking", True))

        img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image_pil", "image_pil": img},   # vLLM's PIL-image content type
                {"type": "text", "text": prompt_text},
            ],
        }]

        # temp=0.1 (not 0.0) so the thinking trace can branch; top_p 0.95 is the
        # standard sampling companion for low-temp reasoning.
        sp = self.SamplingParams(
            temperature=temperature, max_tokens=max_tokens, top_p=0.95,
        )

        # Enable Gemma's reasoning mode via the chat template flag. If this build's
        # template ignores the kwarg it's a harmless no-op (we verify with a smoke
        # test that a reasoning trace actually appears). If a vLLM build rejects the
        # kwarg outright, fall back to a plain call rather than 500 the request.
        t0 = time.time()
        try:
            outputs = self.llm.chat(
                messages, sp,
                chat_template_kwargs={"enable_thinking": enable_thinking},
            )
        except (TypeError, ValueError) as e:
            print(f"[vllm] enable_thinking kwarg rejected ({e}); plain chat", flush=True)
            outputs = self.llm.chat(messages, sp)
        elapsed = time.time() - t0

        out = outputs[0]
        raw_text = out.outputs[0].text
        finish_reason = out.outputs[0].finish_reason   # "stop" | "length" (truncated)
        # Strip the thinking trace server-side so the caller gets clean JSON.
        response = _extract_json_array(raw_text)
        thinking_chars = len(raw_text) - len(response)
        in_tokens = len(out.prompt_token_ids) if out.prompt_token_ids else 0
        out_tokens = len(out.outputs[0].token_ids)

        return {
            "response": response,
            "finish_reason": finish_reason,
            "thinking_chars": thinking_chars,
            "input_tokens": int(in_tokens),
            "output_tokens": int(out_tokens),
            "elapsed_s": round(elapsed, 2),
        }

    @modal.fastapi_endpoint(method="GET")
    def healthz(self) -> dict:
        return {"status": "ok", "model": MODEL_ID, "gpu": GPU, "runtime": "vllm"}
