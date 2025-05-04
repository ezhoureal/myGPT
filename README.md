# myGPT Project

This project re-implements many aspects of LLM systems for learning purposes and is still a work in progress. Inspired by (nanoGPT)[https://github.com/karpathy/nanoGPT]

## Installation

This project is managed by [uv](https://docs.astral.sh/uv/). To install the required dependencies, run:

```
uv sync
source .venv/bin/activate
```

## Usage

You can run `llm_core/inference.py` to generate text. Modify the `PROMPT` variable in the script to change the input text.

run `tokenizer/tokenizer.py` to train and encode text to tokens

## Modules

### llm_core

### manual_grad

### Tokenizer
Uses Byte Pair Encoding (BPE) tokenizer. The initial python implementation is under directory `tokenizer_py`. A Rust rewrite is in progress under `src` for better performance.

## Testing
to run the test suite, run `pytest` and `cargo test` in the root directory

## License

This project is licensed under the MIT License.