# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Classic BERT encoder for sequence classification (MiniLM-L6 warm start).

This mirrors HuggingFace's ``BertModel``/``BertForSequenceClassification`` closely enough to
round-trip state dicts. Notable architectural points handled here:

- Learned absolute position embeddings (positions ``0..T-1``) plus a token-type embedding table.
  At runtime every token is type 0 (single-segment classification), but the table must exist so
  checkpoints load; only row 0 is added to the embedding sum.
- Post-LN residual blocks: LayerNorm runs AFTER each residual add (``attention.output.LayerNorm``
  and ``output.LayerNorm``), unlike the pre-LN layout of most decoder models in this repo.
- Separate biased q/k/v projections (``attention.self.{query,key,value}``) and biases everywhere.
- The classifier is HF's layout: ``bert.pooler`` (dense + tanh on the CLS token) -> dropout ->
  ``classifier``. The reference checkpoint ``nreimers/MiniLM-L6-H384-uncased`` is a bare
  ``BertModel`` whose state dict DOES include ``pooler.dense.{weight,bias}``, so warm starts load
  the pooler too; only ``classifier.*`` is randomly initialized.

Dropout is applied only at the classifier head (HF's ``hidden_dropout_prob``); the internal
embedding/attention/hidden dropouts are omitted (they are identity at eval and this model is
fine-tuned briefly, mirroring the ModernBERT classifier port which has no internal dropout).
"""

import dataclasses
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax.numpy as jnp
import jax.random as jrandom
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call
from haliax.state_dict import (
    ModuleWithStateDictSerialization,
    StateDict,
    from_torch_compatible_state_dict,
    to_torch_compatible_state_dict,
)

from levanter.compat.hf_checkpoints import HFCheckpointConverter, HFCompatConfig
from levanter.layers.attention import AttentionBackend, AttentionMask, dot_product_attention
from levanter.models.classification import ClassificationExample, register_classifier_arch
from levanter.models.lm_model import LmConfig
from levanter.utils.activation import ActivationFunctionEnum
from levanter.utils.flop_utils import lm_flops_per_token
from levanter.utils.logging import silence_transformer_nag

silence_transformer_nag()
from transformers import BertConfig as HfBertConfig  # noqa: E402
from transformers import PretrainedConfig as HfConfig  # noqa: E402


MINILM_L6_CHECKPOINT = "nreimers/MiniLM-L6-H384-uncased"


@LmConfig.register_subclass("bert")
@dataclass(frozen=True)
class BertConfig(HFCompatConfig):
    """Config for classic BERT. Defaults match ``nreimers/MiniLM-L6-H384-uncased``."""

    max_seq_len: int = 512
    hidden_dim: int = 384
    intermediate_dim: int = 1536
    num_layers: int = 6
    num_heads: int = 12

    layer_norm_epsilon: float = 1e-12
    type_vocab_size: int = 2
    activation_function: ActivationFunctionEnum = ActivationFunctionEnum.gelu
    hidden_dropout_prob: float = 0.1  # applied at the classifier head only
    num_labels: int = 2
    initializer_range: float = 0.02
    pad_token_id: int = 0  # [PAD] = 0 for BERT WordPiece

    attn_backend: AttentionBackend = AttentionBackend.VANILLA
    upcast_attn: bool = False

    reference_checkpoint: str = MINILM_L6_CHECKPOINT
    tokenizer: Optional[str] = MINILM_L6_CHECKPOINT

    def __post_init__(self):
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim={self.hidden_dim} not divisible by num_heads={self.num_heads}")

    @property
    def Embed(self) -> Axis:
        return Axis("embed", self.hidden_dim)

    @property
    def Label(self) -> Axis:
        return Axis("label", self.num_labels)

    @property
    def Heads(self) -> Axis:
        return Axis("heads", self.num_heads)

    @property
    def HeadSize(self) -> Axis:
        return Axis("head_size", self.hidden_dim // self.num_heads)

    @property
    def Mlp(self) -> Axis:
        return Axis("mlp", self.intermediate_dim)

    @property
    def TokenType(self) -> Axis:
        return Axis("token_type", self.type_vocab_size)

    def mk_LayerNorm(self, axis: Axis) -> hnn.LayerNorm:
        return hnn.LayerNorm.init(axis, eps=self.layer_norm_epsilon, use_weight=True, use_bias=True)

    @property
    def model_type(self) -> Type["BertForSequenceClassification"]:  # pyrefly: ignore[bad-override]
        return BertForSequenceClassification

    def hf_checkpoint_converter(  # pyrefly: ignore[bad-override]
        self, ref_checkpoint: Optional[str] = None
    ) -> HFCheckpointConverter["BertConfig"]:
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=self.reference_checkpoint if ref_checkpoint is None else ref_checkpoint,
            trust_remote_code=False,
            tokenizer=self.tokenizer if self.tokenizer is not None else ref_checkpoint,
            HfConfigClass=HfBertConfig,
        )

    @classmethod
    def from_hf_config(cls, hf_config: HfConfig) -> "BertConfig":
        return BertConfig(
            max_seq_len=hf_config.max_position_embeddings,
            hidden_dim=hf_config.hidden_size,
            intermediate_dim=hf_config.intermediate_size,
            num_layers=hf_config.num_hidden_layers,
            num_heads=hf_config.num_attention_heads,
            layer_norm_epsilon=hf_config.layer_norm_eps,
            type_vocab_size=hf_config.type_vocab_size,
            activation_function=ActivationFunctionEnum(hf_config.hidden_act),
            hidden_dropout_prob=hf_config.hidden_dropout_prob,
            initializer_range=hf_config.initializer_range,
            pad_token_id=hf_config.pad_token_id,
            num_labels=getattr(hf_config, "num_labels", 2),
        )

    def to_hf_config(self, vocab_size: int, config_overrides: Optional[Dict] = None) -> HfBertConfig:
        if config_overrides is None:
            config_overrides = {}
        return HfBertConfig(
            vocab_size=vocab_size,
            hidden_size=self.hidden_dim,
            intermediate_size=self.intermediate_dim,
            num_hidden_layers=self.num_layers,
            num_attention_heads=self.num_heads,
            max_position_embeddings=self.max_seq_len,
            layer_norm_eps=self.layer_norm_epsilon,
            type_vocab_size=self.type_vocab_size,
            hidden_act=self.activation_function.value,
            hidden_dropout_prob=self.hidden_dropout_prob,
            initializer_range=self.initializer_range,
            pad_token_id=self.pad_token_id,
            num_labels=self.num_labels,
            architectures=["BertForSequenceClassification"],
            **config_overrides,
        )

    def flops_per_token(self, vocab_size: int, context_length: int) -> Optional[float]:
        return lm_flops_per_token(
            hidden_dim=self.hidden_dim,
            intermediate_dim=self.intermediate_dim,
            num_layers=self.num_layers,
            num_kv_heads=self.num_heads,
            num_heads=self.num_heads,
            seq_len=context_length,
            vocab_size=vocab_size,
            glu=False,
        )


class BertEmbeddings(ModuleWithStateDictSerialization):
    """Word + learned-position + token-type embeddings, then LayerNorm (HF ``BertEmbeddings``).

    All tokens are type 0 at runtime, so only row 0 of ``token_type_embeddings`` enters the sum,
    but the full table is a parameter so checkpoints round-trip.
    """

    word_embeddings: hnn.Embedding
    position_embeddings: hnn.Embedding
    token_type_embeddings: hnn.Embedding
    LayerNorm: hnn.LayerNorm

    @staticmethod
    def init(Vocab: Axis, config: BertConfig, *, key) -> "BertEmbeddings":
        k_word, k_pos, k_type = jrandom.split(key, 3)
        word = hnn.Embedding.init(Vocab, config.Embed, key=k_word, init_scale=config.initializer_range)
        position = hnn.Embedding.init(config.max_Pos, config.Embed, key=k_pos, init_scale=config.initializer_range)
        token_type = hnn.Embedding.init(
            config.TokenType, config.Embed, key=k_type, init_scale=config.initializer_range
        )
        return BertEmbeddings(word, position, token_type, config.mk_LayerNorm(config.Embed))

    @property
    def Vocab(self) -> Axis:
        return self.word_embeddings.Vocab

    @named_call
    def embed(self, input_ids: NamedArray) -> NamedArray:
        Pos = input_ids.resolve_axis("position")
        pos_ids = hax.arange(Pos, dtype=jnp.int32)
        x = self.word_embeddings(input_ids) + self.position_embeddings.embed(pos_ids)
        x = x + self.token_type_embeddings.weight["token_type", 0]
        return self.LayerNorm(x)


class BertSelfAttention(ModuleWithStateDictSerialization):
    """HF ``BertSelfAttention``: separate biased q/k/v projections + scaled dot-product attention."""

    config: BertConfig = eqx.field(static=True)
    query: hnn.Linear
    key: hnn.Linear
    value: hnn.Linear

    @staticmethod
    def init(config: BertConfig, *, key) -> "BertSelfAttention":
        k_q, k_k, k_v = jrandom.split(key, 3)
        Out = (config.Heads, config.HeadSize)
        query = hnn.Linear.init(In=config.Embed, Out=Out, key=k_q, use_bias=True, out_first=True)
        key_proj = hnn.Linear.init(In=config.Embed, Out=Out, key=k_k, use_bias=True, out_first=True)
        value = hnn.Linear.init(In=config.Embed, Out=Out, key=k_v, use_bias=True, out_first=True)
        return BertSelfAttention(config, query, key_proj, value)

    @named_call
    def __call__(self, x: NamedArray, mask: AttentionMask | NamedArray | None) -> NamedArray:
        q = self.query(x)
        k = self.key(x).rename({"position": "key_position"})
        v = self.value(x).rename({"position": "key_position"})
        attn_output = dot_product_attention(
            "position",
            "key_position",
            "head_size",
            q,
            k,
            v,
            mask,
            attention_dtype=jnp.float32 if self.config.upcast_attn else x.dtype,
            attn_backend=self.config.attn_backend,
        )
        return attn_output.astype(x.dtype)


class BertSelfOutput(ModuleWithStateDictSerialization):
    """HF ``BertSelfOutput``: output projection + post-LN over the residual sum."""

    dense: hnn.Linear
    LayerNorm: hnn.LayerNorm

    @staticmethod
    def init(config: BertConfig, *, key) -> "BertSelfOutput":
        dense = hnn.Linear.init(
            In=(config.Heads, config.HeadSize), Out=config.Embed, key=key, use_bias=True, out_first=True
        )
        return BertSelfOutput(dense, config.mk_LayerNorm(config.Embed))

    @named_call
    def __call__(self, attn_output: NamedArray, residual: NamedArray) -> NamedArray:
        return self.LayerNorm(self.dense(attn_output) + residual)


class BertAttention(ModuleWithStateDictSerialization):
    self_attn: BertSelfAttention
    output: BertSelfOutput

    @staticmethod
    def init(config: BertConfig, *, key) -> "BertAttention":
        k_self, k_out = jrandom.split(key, 2)
        return BertAttention(BertSelfAttention.init(config, key=k_self), BertSelfOutput.init(config, key=k_out))

    def _state_dict_key_map(self) -> Dict[str, Optional[str]]:
        # HF names this submodule "self", which is not a legal dataclass field name.
        return {"self_attn": "self"}

    @named_call
    def __call__(self, x: NamedArray, mask: AttentionMask | NamedArray | None) -> NamedArray:
        return self.output(self.self_attn(x, mask), x)


class BertIntermediate(ModuleWithStateDictSerialization):
    dense: hnn.Linear
    act: Callable = eqx.field(static=True)

    @staticmethod
    def init(config: BertConfig, *, key) -> "BertIntermediate":
        dense = hnn.Linear.init(In=config.Embed, Out=config.Mlp, key=key, use_bias=True, out_first=True)
        return BertIntermediate(dense, config.activation_function.to_fn())

    @named_call
    def __call__(self, x: NamedArray) -> NamedArray:
        return self.act(self.dense(x))


class BertOutput(ModuleWithStateDictSerialization):
    """HF ``BertOutput``: MLP down-projection + post-LN over the residual sum."""

    dense: hnn.Linear
    LayerNorm: hnn.LayerNorm

    @staticmethod
    def init(config: BertConfig, *, key) -> "BertOutput":
        dense = hnn.Linear.init(In=config.Mlp, Out=config.Embed, key=key, use_bias=True, out_first=True)
        return BertOutput(dense, config.mk_LayerNorm(config.Embed))

    @named_call
    def __call__(self, x: NamedArray, residual: NamedArray) -> NamedArray:
        return self.LayerNorm(self.dense(x) + residual)


class BertLayer(ModuleWithStateDictSerialization):
    attention: BertAttention
    intermediate: BertIntermediate
    output: BertOutput

    @staticmethod
    def init(config: BertConfig, *, key) -> "BertLayer":
        k_attn, k_inter, k_out = jrandom.split(key, 3)
        return BertLayer(
            BertAttention.init(config, key=k_attn),
            BertIntermediate.init(config, key=k_inter),
            BertOutput.init(config, key=k_out),
        )

    @named_call
    def __call__(self, x: NamedArray, mask: AttentionMask | NamedArray | None) -> NamedArray:
        attn_out = self.attention(x, mask)
        return self.output(self.intermediate(attn_out), attn_out)


class BertEncoder(ModuleWithStateDictSerialization):
    layer: list  # list[BertLayer]; serializes as layer.{i} to match HF encoder.layer.{i}

    @staticmethod
    def init(config: BertConfig, *, key) -> "BertEncoder":
        keys = jrandom.split(key, config.num_layers)
        return BertEncoder([BertLayer.init(config, key=k) for k in keys])

    @named_call
    def __call__(self, x: NamedArray, mask: AttentionMask | NamedArray | None) -> NamedArray:
        for layer in self.layer:
            x = layer(x, mask)
        return x


class BertPooler(ModuleWithStateDictSerialization):
    """HF ``BertPooler``: dense + tanh on the CLS (first) token."""

    dense: hnn.Linear

    @staticmethod
    def init(config: BertConfig, *, key) -> "BertPooler":
        dense = hnn.Linear.init(
            In=config.Embed, Out=config.Embed.alias("pooled_embed"), key=key, use_bias=True, out_first=True
        )
        return BertPooler(dense)

    @named_call
    def __call__(self, hidden: NamedArray) -> NamedArray:
        cls_token = hidden["position", 0]
        return hax.tanh(self.dense(cls_token)).rename({"pooled_embed": "embed"})


class BertModel(ModuleWithStateDictSerialization):
    """The bare BERT encoder (HF ``BertModel``): embeddings -> post-LN transformer -> pooler."""

    config: BertConfig = eqx.field(static=True)
    embeddings: BertEmbeddings
    encoder: BertEncoder
    pooler: BertPooler

    @staticmethod
    def init(Vocab: Axis, config: BertConfig, *, key) -> "BertModel":
        k_emb, k_enc, k_pool = jrandom.split(key, 3)
        return BertModel(
            config,
            BertEmbeddings.init(Vocab, config, key=k_emb),
            BertEncoder.init(config, key=k_enc),
            BertPooler.init(config, key=k_pool),
        )

    @property
    def Vocab(self) -> Axis:
        return self.embeddings.Vocab

    @named_call
    def __call__(self, input_ids: NamedArray, attn_mask: AttentionMask | NamedArray | None = None) -> NamedArray:
        return self.encoder(self.embeddings.embed(input_ids), attn_mask)


class BertForSequenceClassification(ModuleWithStateDictSerialization):
    """BERT encoder + HF classification head: ``bert.pooler`` (CLS dense+tanh) -> dropout -> ``classifier``.

    State-dict field names (``bert`` / ``classifier``) match HF ``BertForSequenceClassification``
    so fine-tuned HF checkpoints round-trip.
    """

    bert: BertModel
    dropout: hnn.Dropout
    classifier: hnn.Linear

    @property
    def config(self) -> BertConfig:
        return self.bert.config

    @property
    def Vocab(self) -> Axis:
        return self.bert.Vocab

    @property
    def Label(self) -> Axis:
        return self.classifier.Out if isinstance(self.classifier.Out, Axis) else self.config.Label

    @classmethod
    def init(cls, Vocab: Axis, config: BertConfig, *, key) -> "BertForSequenceClassification":
        k_bert, k_cls = jrandom.split(key, 2)
        bert = BertModel.init(Vocab, config, key=k_bert)
        dropout = hnn.Dropout(pdrop=config.hidden_dropout_prob)
        classifier = hnn.Linear.init(In=config.Embed, Out=config.Label, key=k_cls, use_bias=True, out_first=True)
        return BertForSequenceClassification(bert, dropout, classifier)

    @named_call
    def __call__(
        self,
        input_ids: NamedArray,
        attn_mask: AttentionMask | NamedArray | None = None,
        *,
        key=None,
    ) -> NamedArray:
        hidden = self.bert(input_ids, attn_mask)
        pooled = self.bert.pooler(hidden)
        pooled = self.dropout(pooled, key=key, inference=key is None)
        return self.classifier(pooled)

    def compute_loss(
        self,
        example: ClassificationExample,
        *,
        key=None,
        reduction: Optional[hax.ReductionFunction] = hax.mean,  # pyrefly: ignore[bad-function-definition]
        reduction_axis: Optional[hax.AxisSelection] = None,
    ) -> NamedArray:
        logits = self(example.tokens, example.attn_mask, key=key).astype(jnp.float32)
        target = hax.nn.one_hot(example.label, self.Label, dtype=logits.dtype)
        return hax.nn.cross_entropy_loss(logits, self.Label, target, reduction, reduction_axis=reduction_axis)

    def resize_vocab(self, new_size: int, key=None) -> "BertForSequenceClassification":
        new_embeddings = dataclasses.replace(
            self.bert.embeddings,
            word_embeddings=self.bert.embeddings.word_embeddings.resize_embeddings(new_size, key=key),
        )
        return dataclasses.replace(  # pyrefly: ignore[bad-specialization]
            self, bert=dataclasses.replace(self.bert, embeddings=new_embeddings)
        )


def load_hf_bert_classifier(
    config: BertConfig,
    ref: str,
    *,
    axis_mapping=None,
    dtype=None,
) -> BertForSequenceClassification:
    """Load a fine-tuned HF ``BertForSequenceClassification`` checkpoint into Levanter.

    ``HFCheckpointConverter.load_pretrained`` forces the template to ``config.model_type`` via the
    HF config's ``model_type``, so this loads the classifier template directly from the state dict
    (all keys, including ``bert.pooler`` and ``classifier``, must be present).
    """
    converter = config.hf_checkpoint_converter(ref_checkpoint=ref)
    hf_config = converter.hf_config_from_hf_checkpoint(ref)
    Vocab = Axis("vocab", hf_config.vocab_size)
    state_dict = converter.load_state_dict(ref, dtype=dtype)

    def _load(template):
        model = from_torch_compatible_state_dict(template, state_dict)
        return hax.shard_with_axis_mapping(model, axis_mapping) if axis_mapping is not None else model

    if axis_mapping is not None:
        _load = hax.named_jit(_load, axis_resources=axis_mapping, out_axis_resources=axis_mapping)
    template = eqx.filter_eval_shape(BertForSequenceClassification.init, Vocab, config, key=jrandom.PRNGKey(0))
    return _load(template)


def _bert_prefixed(state_dict: StateDict) -> StateDict:
    """Normalize a checkpoint to ``bert.``-prefixed keys (bare ``BertModel`` checkpoints are unprefixed)."""
    if any(k.startswith("bert.") for k in state_dict):
        return state_dict
    return {f"bert.{k}": v for k, v in state_dict.items()}


def _build_bert_classifier(
    config: BertConfig, Vocab: Axis, *, key, warm_start: bool, axis_mapping=None, compute_dtype=None
) -> BertForSequenceClassification:
    """Build the classifier; with ``warm_start`` load ``bert.*`` weights from the reference checkpoint.

    The reference (``nreimers/MiniLM-L6-H384-uncased``) is a bare ``BertModel`` state dict WITH
    pooler weights but no classifier head, so this partial-loads: any template key present in the
    checkpoint (same shape) is taken from the checkpoint; the rest — in practice ``classifier.*``,
    plus non-parameter buffers like ``embeddings.position_ids`` on the checkpoint side — keep their
    fresh random/template values.
    """
    model = BertForSequenceClassification.init(Vocab, config, key=key)
    if not warm_start:
        return model

    converter = config.hf_checkpoint_converter()
    ckpt = _bert_prefixed(converter.load_state_dict(config.reference_checkpoint, dtype=compute_dtype))

    embed_key = "bert.embeddings.word_embeddings.weight"
    if embed_key not in ckpt:
        raise ValueError(f"{config.reference_checkpoint} does not look like a BERT checkpoint (no {embed_key})")
    if ckpt[embed_key].shape[0] != Vocab.size:
        raise ValueError(
            f"vocab mismatch: checkpoint {config.reference_checkpoint} has {ckpt[embed_key].shape[0]} "
            f"tokens but Vocab has {Vocab.size}"
        )

    template_sd = to_torch_compatible_state_dict(model)
    merged = {
        **template_sd,
        **{k: v for k, v in ckpt.items() if k in template_sd and template_sd[k].shape == tuple(v.shape)},
    }
    model = from_torch_compatible_state_dict(model, merged)
    if axis_mapping is not None:
        model = hax.shard_with_axis_mapping(model, axis_mapping)
    return model


def _save_bert_classifier(config: BertConfig, model, path: str) -> None:
    config.hf_checkpoint_converter().save_pretrained(model, path, save_tokenizer=True)


register_classifier_arch(BertConfig, build=_build_bert_classifier, save=_save_bert_classifier)


__all__ = [
    "BertConfig",
    "BertEncoder",
    "BertForSequenceClassification",
    "BertModel",
    "BertPooler",
    "load_hf_bert_classifier",
]
