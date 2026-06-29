from typing import Literal

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, TokenizersBackend


class CodeToVec:
    tokenizer: TokenizersBackend
    model: PreTrainedModel
    device: Literal["cpu", "gpu"]

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v2-code-base",
        device: Literal["cpu", "gpu"] = "cpu",
    ) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(
            device
        )
        self.device = device

    def execute(self, text: str) -> list[float]:
        # トークナイズ処理
        encoded_text = self.tokenizer(text, return_tensors="pt").to(self.device)

        # ベクトル抽出
        with torch.no_grad():
            model_output = self.model(**encoded_text)

            # 先頭トークンのベクトルを取得し、L2正規化を行う
            # _attention_mask = encoded_text.get("attention_mask")

            # 先頭トークン[CLS]の抽出
            embeddings = model_output.last_hidden_state[:, 0, :]
            embeddings = F.normalize(embeddings, p=2, dim=1)

            return (
                embeddings.flatten().tolist()
                if self.device == "gpu"
                else embeddings.flatten().cpu().tolist()
            )
