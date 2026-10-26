"""Unified command-line entrypoint for the project's model runners."""
from __future__ import annotations

import argparse
import sys


def _build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("dpa4", "text"),
        help="run DPA4 tools/inference or text-generation inference",
    )
    return parser


def run_text_generation(argv=None):
    """Run a registered text-generation adapter, such as Qwen3."""
    parser = argparse.ArgumentParser(description="Run text generation through a model adapter")
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
    args = parser.parse_args(argv)

    model_id = args.model_id
    if model_id is None:
        if args.family == "qwen3":
            model_id = "Qwen/Qwen3-0.6B"
        else:
            parser.error("--model-id is required for the unverified Qwen3.5 family")

    from runtime.inference import TextGenerationRequest, create_runner

    messages = None
    if args.system_prompt is not None:
        messages = (
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        )
    request = TextGenerationRequest(
        prompt=args.prompt if messages is None else None,
        messages=messages,
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
    return result


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if not args or args[0] in ("-h", "--help"):
        parser.print_help()
        return 0

    command = args.pop(0)
    if command == "dpa4":
        from runtime.dpa4.cli import main as dpa4_main

        return dpa4_main(args)
    if command == "text":
        return run_text_generation(args)
    parser.error(f"invalid choice: {command!r} (choose from 'dpa4', 'text')")


if __name__ == "__main__":
    main()
