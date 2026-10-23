"""Run text generation through the optional HuggingFace model adapter."""
from __future__ import annotations

import argparse

from model.inference import create_runner
from model.inference.types import TextGenerationRequest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=("qwen3", "qwen3.5"), default="qwen3")
    parser.add_argument("--model-id", help="HuggingFace id or local directory")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system-prompt")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:<index>")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="explicitly permit custom code from the model repository")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    model_id = args.model_id
    if model_id is None:
        if args.family == "qwen3":
            model_id = "Qwen/Qwen3-0.6B"
        else:
            raise ValueError("--model-id is required for the unverified Qwen3.5 family")
    if args.system_prompt is None:
        request = TextGenerationRequest(
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    else:
        request = TextGenerationRequest(
            messages=(
                {"role": "system", "content": args.system_prompt},
                {"role": "user", "content": args.prompt},
            ),
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
        )
    with create_runner(
        args.family,
        "text-generation",
        model_id=model_id,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    ) as runner:
        result = runner.predict(request)
    print(result.text)
    print(f"\n[{result.prompt_tokens} prompt tokens; {result.generated_tokens} generated tokens]")


if __name__ == "__main__":
    main()
