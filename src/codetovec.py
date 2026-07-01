from typing import Literal

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


class CodeToVec:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    device: Literal["cpu", "gpu"]

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-base-code",
        device: Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            device
        )
        self.device = device

    def _cleanse_python_code(
        self,
        source: str,
        remove_comments: bool,
        remove_docstrings: bool,
        remove_blank_lines: bool,
    ) -> str:
        import io
        import tokenize

        try:
            io_obj = io.StringIO(source)
            tokens = list(tokenize.generate_tokens(io_obj.readline))

            out_tokens = []
            docstring_indices = set()

            if remove_docstrings:
                for i, tok in enumerate(tokens):
                    if tok.type == tokenize.STRING:
                        prev_idx = i - 1
                        while prev_idx >= 0 and tokens[prev_idx].type in (
                            tokenize.NL,
                            tokenize.COMMENT,
                        ):
                            prev_idx -= 1
                        next_idx = i + 1
                        while next_idx < len(tokens) and tokens[next_idx].type in (
                            tokenize.NL,
                            tokenize.COMMENT,
                        ):
                            next_idx += 1

                        prev_is_start = prev_idx < 0 or tokens[prev_idx].type in (
                            tokenize.INDENT,
                            tokenize.DEDENT,
                            tokenize.NEWLINE,
                        )
                        next_is_end = next_idx >= len(tokens) or tokens[
                            next_idx
                        ].type in (
                            tokenize.NEWLINE,
                            tokenize.ENDMARKER,
                        )

                        if prev_is_start and next_is_end:
                            docstring_indices.add(i)

            for i, tok in enumerate(tokens):
                if remove_comments and tok.type == tokenize.COMMENT:
                    continue
                if i in docstring_indices:
                    continue
                out_tokens.append(tok)

            result = tokenize.untokenize(out_tokens)

            if remove_blank_lines:
                result = "\n".join(line for line in result.splitlines() if line.strip())

            return result
        except tokenize.TokenError:
            return source
        except Exception:
            return source

    def execute(
        self,
        text: str,
        remove_comments: bool = False,
        remove_docstrings: bool = False,
        remove_blank_lines: bool = False,
    ) -> list[float]:
        if remove_comments or remove_docstrings or remove_blank_lines:
            text = self._cleanse_python_code(
                text, remove_comments, remove_docstrings, remove_blank_lines
            )

        # トークナイズ処理
        encoded_text = self.tokenizer(text, return_tensors="pt").to(self.device)

        # ベクトル抽出
        with torch.no_grad():
            model_output = self.model(**encoded_text)

            # 先頭トークン[CLS]の抽出
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = F.normalize(embeddings, p=2, dim=1)

            return (
                embeddings.flatten().tolist()
                if self.device == "gpu"
                else embeddings.flatten().cpu().tolist()
            )
