from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Projection(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=5 ** 0.5)
        self.lora_a.data = self.lora_a.data.to(base.weight.device)
        self.lora_b.data = self.lora_b.data.to(base.weight.device)
        self.scaling = alpha / rank
        object.__setattr__(self, "_transient_v", None)

    def forward(self, x):
        output = self.base(x)
        update = F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype))
        output = output + self.scaling * update.to(output.dtype)
        perturbation = object.__getattribute__(self, "_transient_v")
        if perturbation is not None:
            output = output + F.linear(x.float(), perturbation.float()).to(output.dtype)
        return output


class BinaryHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 256), nn.GELU(), nn.Linear(256, 1)
        )
        self.to(torch.float32)

    def forward(self, hidden):
        return self.net(hidden.float()).squeeze(-1)


class Auditor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone.config.use_cache = False
        if not hasattr(backbone, "model") or not hasattr(backbone.model, "layers"):
            raise ValueError("The backbone must expose model.layers with query/value projections.")
        self.head = BinaryHead(backbone.config.hidden_size)
        layers = backbone.model.layers
        if not layers:
            raise ValueError("The backbone contains no Transformer blocks.")
        self.layer_id = max(0, min(len(layers) - 1, int(round(0.5 * len(layers)))))
        for block in layers[self.layer_id + 1:]:
            for name, module in list(block.named_modules()):
                if name.rsplit(".", 1)[-1] not in ("q_proj", "v_proj"):
                    continue
                if not isinstance(module, nn.Linear):
                    raise TypeError("Query and value projections must be linear modules.")
                parent = block
                components = name.split(".")
                for component in components[:-1]:
                    parent = getattr(parent, component)
                setattr(parent, components[-1], Projection(module))
        if not self.projections():
            raise ValueError("No suffix query/value projections were selected.")
        self._reference_scales = {}
        self.eval()

    @classmethod
    def from_pretrained(cls, model_name, revision=None, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM

        backbone = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, dtype=dtype, attn_implementation="sdpa"
        )
        return cls(backbone)

    def forward(self, input_ids, attention_mask):
        output = self.backbone.model(input_ids=input_ids, attention_mask=attention_mask)
        positions = attention_mask.sum(dim=1) - 1
        rows = torch.arange(input_ids.shape[0], device=input_ids.device)
        return self.head(output.last_hidden_state[rows, positions])

    def projections(self):
        return {name: module for name, module in self.named_modules() if isinstance(module, Projection)}

    def reference_scales(self):
        if not self._reference_scales:
            with torch.no_grad():
                for name, module in self.projections().items():
                    scale = module.base.weight.detach().float().norm()
                    if not torch.isfinite(scale) or scale <= 0:
                        raise ValueError("Frozen projection norms must be finite and positive.")
                    self._reference_scales[name] = float(scale.clamp_min(1e-12))
        return dict(self._reference_scales)

    def trainable_state_dict(self):
        names = {name for name, parameter in self.named_parameters() if parameter.requires_grad}
        return {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items() if name in names}

    def load_trainable_state_dict(self, state):
        expected = {name: parameter for name, parameter in self.named_parameters() if parameter.requires_grad}
        if set(state) != set(expected):
            raise ValueError("Checkpoint tensors do not match the auditor's adapters and prediction head.")
        for name, tensor in state.items():
            if tensor.shape != expected[name].shape or tensor.dtype != torch.float32:
                raise ValueError(f"Invalid checkpoint tensor: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite checkpoint tensor: {name}")
        self.load_state_dict(state, strict=False)
