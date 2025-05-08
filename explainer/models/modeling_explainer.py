import math  # 导入数学库，用于数学运算，如平方根等
import warnings  # 导入警告模块，用于发出运行时警告
from typing import List, Optional, Tuple, Union  # 导入类型提示，用于标注函数参数和返回值类型

import torch  # 导入 PyTorch 主库，用于张量运算和深度学习
import torch.nn.functional as F  # 导入函数式接口，包含激活函数、损失函数等
import torch.utils.checkpoint  # 导入检查点模块，用于节省显存的前向/反向计算
from torch import nn  # 导入神经网络模块，包含各种层和容器
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入常用损失函数

from transformers.activations import ACT2FN  # 导入激活函数映射表
from transformers.cache_utils import Cache, DynamicCache, StaticCache  # 导入缓存相关工具
from transformers.modeling_attn_mask_utils import AttentionMaskConverter  # 导入注意力掩码转换器
from transformers.modeling_outputs import (  # 导入多种模型输出类型
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel  # 导入预训练模型基类
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS  # 导入注册的所有 LayerNorm 层列表
from transformers.utils import (  # 导入多种实用工具
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.models.llama.configuration_llama import LlamaConfig  # 导入 Llama 模型配置类


# 如果支持 flash attention，则导入相关函数
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func  # Fast attention 实现
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # 用于处理填充和索引  # noqa

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

_CONFIG_FOR_DOC = "LlamaConfig"  # 文档中引用的配置名称，用于自动生成文档


def _get_unpad_data(attention_mask):
    # 计算每条序列的真实长度（去除 padding 后）
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    # 获取所有非零位置的索引
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    # 批次中最大序列长度
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    # 计算累积序列长度，用于 batch 索引
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,               # 非 padding 元素的扁平索引
        cu_seqlens,            # 每条序列累积起始索引
        max_seqlen_in_batch,   # 最大真实序列长度
    )


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm 等价于 T5LayerNorm，使用 RMS 归一化
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 可训练权重参数
        self.variance_epsilon = eps  # 防止除零的小常数

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype  # 保存输入数据类型
        hidden_states = hidden_states.to(torch.float32)  # 转为 float32 计算精度更好
        # 计算方差：沿最后一个维度求平均
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 标准化：除以方差平方根
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 恢复原始 dtype 并乘以可训练权重
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)  # 将 LlamaRMSNorm 注册到全局 LayerNorm 列表中


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor  # 位置缩放因子
        self.dim = dim  # 头部维度大小
        self.max_position_embeddings = max_position_embeddings  # 最大位置长度
        self.base = base  # 频率基数

        # 计算反频率张量：1 / base^(i/dim)
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64)
                          .float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 注册 buffer，不参与梯度更新

        # 缓存最大序列的 cos/sin
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        t = t / self.scaling_factor  # 按缩放因子缩放位置
        freqs = torch.outer(t, self.inv_freq)  # 外积得到位置频率矩阵
        emb = torch.cat((freqs, freqs), dim=-1)  # 重复以匹配维度
        # 缓存 cos 和 sin 值
        self.register_buffer("_cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False)
        self.register_buffer("_sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False)

    @property
    def sin_cached(self):
        # 访问 sin 缓存时发出一次警告，提示即将弃用
        logger.warning_once(
            "The sin_cached attribute will be removed in 4.39. ... Use the forward method of RoPE from now on instead."
        )
        return self._sin_cached

    @property
    def cos_cached(self):
        # 访问 cos 缓存时发出一次警告
        logger.warning_once(
            "The cos_cached attribute will be removed in 4.39. ... Use the forward method of RoPE from now on instead."
        )
        return self._cos_cached

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [batch_size, num_heads, seq_len, head_dim]
        # 扩展 inv_freq 以匹配 batch 和 seq 长度
        inv_freq_expanded = self.inv_freq[None, :, None]\
            .float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # 强制使用 float32，以避免 bfloat16 在长上下文中精度丢失
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # 计算频率并拼接
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        # 返回与输入同 dtype 的 cos 和 sin
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """在线性缩放基础上扩展 RoPE。作者：/u/kaiokendev"""
    def forward(self, x, position_ids):
        # 对位置 ids 应用线性缩放
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """在线性缩放基础上扩展 RoPE，并在序列过长时动态调整。作者：/u/bloc97, /u/emozilla"""
    def forward(self, x, position_ids):
        # 如果序列长度超过缓存，则动态重算 inv_freq
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64)
                        .float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
        cos, sin = super().forward(x, position_ids)
        return cos, sin


def rotate_half(x):
    """将张量最后一维一分为二并旋转：(-x2, x1)"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """
    将 RoPE 应用到 query 和 key 张量上。
    cos/sin: [batch, seq_len, head_dim]
    unsqueeze_dim: 扩展维度以匹配 q/k 的形状
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 对 q 应用旋转嵌入
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 对 k 应用旋转嵌入
    return q_embed, k_embed  # 返回旋转后的 q 和 k


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size  # 隐藏层大小
        self.intermediate_size = config.intermediate_size  # 中间层大小
        # 三个线性映射层：gate, up, down
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # 激活函数

    def forward(self, x):
        # 支持并行预训练头切分
        if self.config.pretraining_tp > 1:
            slice = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice, dim=1)
            # 分片线性变换并拼接
            gate_proj = torch.cat(
                [F.linear(x, gate_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1
            )
            up_proj = torch.cat([F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)], dim=-1)
            # 激活与相乘后再分片
            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
            # 下投影并求和
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            # 常规模式：先 gate，再激活 * up，最后 down
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj  # 返回 MLP 输出


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    重复 key/value 张量以匹配 attention head 数。
    输入： (batch, kv_heads, seqlen, head_dim)
    输出： (batch, attn_heads, seqlen, head_dim)
    """
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states  # 无需重复
    # 插入新维度后扩展，再 reshape 回去
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class LlamaAttention(nn.Module):
    """多头自注意力机制，来自《Attention Is All You Need》论文"""
    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config  # 模型配置
        self.layer_idx = layer_idx  # 层索引（用于缓存）
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended..."
            )
        # 注意力相关超参数
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True  # 因果注意力

        # 确保 hidden_size 可被 head_dim 整除
        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got {self.hidden_size} and {self.num_heads})."
            )

        # 定义 Q/K/V/O 全连接层
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)

        self._init_rope()  # 初始化 RoPE 对象

    def _init_rope(self):
        # 根据配置选择不同的 RoPE 实现
        if self.config.rope_scaling is None:
            self.rotary_emb = LlamaRotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入隐藏状态，形状 [batch, seq_len, hidden_size]
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,  # 位置索引
        past_key_value: Optional[Cache] = None,  # 过去缓存，用于加速生成
        output_attentions: bool = False,  # 是否返回注意力权重
        use_cache: bool = False,  # 是否更新缓存
        cache_position: Optional[torch.LongTensor] = None,  # 缓存位置索引
        user_embed: Optional[torch.Tensor] = None,  # 用户嵌入（自定义）
        item_embed: Optional[torch.Tensor] = None,  # 项目嵌入（自定义）
        user_embed_pos: Optional[torch.Tensor] = None,  # 用户嵌入位置
        item_embed_pos: Optional[torch.Tensor] = None,  # 项目嵌入位置
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()  # 获取 batch 大小和序列长度

        # 根据 pretraining_tp 切分 Q/K/V
        if self.config.pretraining_tp > 1:
            key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)
            # 分片线性运算并拼接
            query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
            query_states = torch.cat(query_states, dim=-1)
            key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
            key_states = torch.cat(key_states, dim=-1)
            value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
            value_states = torch.cat(value_states, dim=-1)
        else:
            # 常规映射
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        # （可选）在指定位置注入用户/项目嵌入
        if query_states.shape[1] != 1:
            # 具体实现略，见原代码

        # 重塑为多头形状并转置
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 获取或更新 past_key_value 缓存
        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)  # 计算 RoPE cos/sin
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 重复 key/value 以匹配多头
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 计算注意力权重：Q @ K^T / sqrt(d)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # 应用注意力掩码
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # Softmax 并丢弃
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 注意力输出
        attn_output = torch.matmul(attn_weights, value_states)
        # 校验输出形状
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is {attn_output.size()}"
            )
        # 转置回原形状并投影
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        # 根据 pretraining_tp 合并输出
        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
            o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
            attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
        else:
            attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None  # 不返回注意力权重时置空

        return attn_output, attn_weights, past_key_value  # 返回注意力输出、权重和缓存

class LlamaFlashAttention2(LlamaAttention):
    """
    Llama flash attention 模块。继承自 `LlamaAttention`，保留原有权重不变。
    唯一需要修改的是 forward 方法：正确调用 flash attention 公共 API，
    并处理包含 padding token 的输入。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO：当 RoCm 下的 Flash Attention 版本低于 2.1 时，应移除此处理逻辑。
        # flash_attn<2.1 会生成左上对齐的 causal mask，而我们需要右下对齐（flash_attn>=2.1 的默认）。
        # 该属性用于区分两种行为，参见：https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0
        # 注意：flash_attn<2.1 且 q_seqlen != k_seqlen（除了 q_seqlen == 1）会生成错误的 mask。
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        user_embed: Optional[torch.Tensor] = None,
        item_embed: Optional[torch.Tensor] = None,
        user_embed_pos: Optional[torch.Tensor] = None,
        item_embed_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        output_attentions = False  # flash attention 暂不返回 attention 权重

        bsz, q_len, _ = hidden_states.size()  # batch_size, 序列长度, 隐藏维度

        # 线性映射得到 Q/K/V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 如果序列长度大于 1，可选注入 user/item 嵌入
        if query_states.shape[1] != 1:
            # 取出 user_embed_pos/item_embed_pos，向对应位置加上额外嵌入
            query_states[torch.arange(user_embed_pos.shape[0]), user_embed_pos[:,0], :] += user_embed
            query_states[torch.arange(item_embed_pos.shape[0]), item_embed_pos[:,0], :] += item_embed
            key_states[torch.arange(user_embed_pos.shape[0]), user_embed_pos[:,0], :] += user_embed
            key_states[torch.arange(item_embed_pos.shape[0]), item_embed_pos[:,0], :] += item_embed
            value_states[torch.arange(user_embed_pos.shape[0]), user_embed_pos[:,0], :] += user_embed
            value_states[torch.arange(item_embed_pos.shape[0]), item_embed_pos[:,0], :] += item_embed

        # 重塑为多头：[batch, heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 计算 RoPE 的 cos/sin，并应用于 Q/K
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 获取或更新缓存
        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # flash attention 要求输入形状 [batch, seq_len, heads, head_dim]
        # 需要多次转置，后续可优化 KV 缓存布局以避免大量转置
        query_states = query_states.transpose(1, 2)
        key_states   = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # 根据训练模式设置 dropout
        dropout_rate = self.attention_dropout if self.training else 0.0

        # 如果之前有强制将 LayerNorm/embedding upcast 到 float32，则这里可能需要 cast 回原 dtype
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"Hidden states is float32, will cast back to {target_dtype} for flash attention."
            )
            query_states = query_states.to(target_dtype)
            key_states   = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # 调用 flash attention 计算注意力
        attn_output = self._flash_attention_forward(
            query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
        )

        # 恢复形状并投影
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        # 不返回 attention 权重
        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states, key_states, value_states,
        attention_mask, query_length,
        dropout=0.0, softmax_scale=None
    ):
        """
        调用 Flash Attention API：
        - 若存在 padding token，先 unpad，再调用 flash_attn_varlen_func，
          最后 pad 回原 shape
        - 否则直接调用 flash_attn_func
        """
        # 根据版本决定 causal mask 对齐方式
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            causal = self.is_causal and query_length != 1

        # 若有 attention_mask（含 padding），先 unpad
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            (query_states, key_states, value_states,
             indices_q, cu_seq_lens, max_seq_lens) = self._upad_input(
                query_states, key_states, value_states,
                attention_mask, query_length
            )
            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_q, max_k = max_seq_lens

            # 变长 flash attention
            attn_output_unpad = flash_attn_varlen_func(
                query_states, key_states, value_states,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_q,
                max_seqlen_k=max_k,
                dropout_p=dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )
            # pad 回原本 shape
            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            # 固定长度 flash attention
            attn_output = flash_attn_func(
                query_states, key_states, value_states,
                dropout, softmax_scale=softmax_scale, causal=causal
            )
        return attn_output

    def _upad_input(
        self, query_layer, key_layer, value_layer,
        attention_mask, query_length
    ):
        """
        处理 unpad：
        - 调用 _get_unpad_data 得到 indices, cu_seqlens, max_seqlen
        - 对 key/value/query 进行 reshape & index_first_axis
        - 返回处理后的各项以及 cu_seqlens, max_seqlens
        """
        # 获取 key/value 的 unpad 信息
        indices_k, cu_seqlens_k, max_k = _get_unpad_data(attention_mask)
        batch_size, kv_seq_len, num_kv_heads, head_dim = key_layer.shape

        # 将 key/value flatten 到 axis0，然后按 indices_k 提取有效行
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_kv_heads, head_dim),
            indices_k
        )
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_kv_heads, head_dim),
            indices_k
        )

        # 处理 query：若长度相同或为 1，则 special case；否则调用 unpad_input
        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, self.num_heads, head_dim),
                indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_q = max_k
            indices_q = indices_k
        elif query_length == 1:
            # 单位置查询：cu_seqlens 为 0,1,2,...; indices 为开始索引
            cu_seqlens_q = torch.arange(batch_size+1, dtype=torch.int32, device=query_layer.device)
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
            max_q = 1
        else:
            # 右填充场景：截取最后 query_length，再 unpad
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_q = unpad_input(
                query_layer, attention_mask
            )

        return (
            query_layer, value_layer, value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_q, max_k),
        )


class LlamaSdpaAttention(LlamaAttention):
    """
    使用 torch.nn.functional.scaled_dot_product_attention 的注意力实现。
    继承自 LlamaAttention，仅修改 forward 以适配 SDPA API。
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        user_embed: Optional[torch.Tensor] = None,
        item_embed: Optional[torch.Tensor] = None,
        user_embed_pos: Optional[torch.Tensor] = None,
        item_embed_pos: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        # 若请求返回 attentions，则回退到手动实现
        if output_attentions:
            logger.warning_once(
                "SDPA 不支持 output_attentions=True，将回退到手动实现。"
            )
            return super().forward(
                hidden_states, attention_mask, position_ids,
                past_key_value, output_attentions,
                use_cache, cache_position
            )

        bsz, q_len, _ = hidden_states.size()
        # 计算 Q/K/V
        query_states = self.q_proj(hidden_states)
        key_states   = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # 可选注入 user/item 嵌入（同前）
        if query_states.shape[1] != 1:
            query_states[torch.arange(user_embed_pos.shape[0]), user_embed_pos[:,0], :] += user_embed
            # ... 前面逻辑同上省略

        # 重塑并应用 RoPE（同上）
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states   = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 更新缓存（同前）
        past_key_value = getattr(self, "past_key_value", past_key_value)
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 重复 KV 以匹配头数
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 构造 causal_mask
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, : key_states.shape[-2]]

        # SDPA 在 GPU 上需要保证 contiguity
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states   = key_states.contiguous()
            value_states = value_states.contiguous()

        # 调用 scaled_dot_product_attention
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        # 恢复形状并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


# 根据配置选择不同注意力类型的映射
LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 根据 config._attn_implementation 选择注意力实现
        self.self_attn = LLAMA_ATTENTION_CLASSES[
            config._attn_implementation
        ](config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)  # 前馈网络
        # 前后归一化层
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        user_embed: Optional[torch.Tensor] = None,
        item_embed: Optional[torch.Tensor] = None,
        user_embed_pos: Optional[torch.Tensor] = None,
        item_embed_pos: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Llama 解码器层：
        - 输入层归一化 → 自注意力 → 残差连接
        - 残差前归一化 → MLP → 残差连接
        """
        # 兼容旧参数名
        if "padding_mask" in kwargs:
            warnings.warn("`padding_mask` 已弃用，请使用 `attention_mask`。")

        # 残差连接保存
        residual = hidden_states

        # 输入层归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 自注意力
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            user_embed=user_embed,
            item_embed=item_embed,
            user_embed_pos=user_embed_pos,
            item_embed_pos=item_embed_pos,
            **kwargs,
        )
        hidden_states = residual + hidden_states  # 残差相加

        # MLP 前归一化 & 前馈
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states  # 残差相加

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs  # 返回隐藏状态、可选的 attention 权重 和 缓存


class LlamaPreTrainedModel(PreTrainedModel):
    # 指定该模型使用的配置类为 LlamaConfig
    config_class = LlamaConfig
    # 基础模型前缀，用于保存/加载时定位子模块
    base_model_prefix = "model"
    # 支持梯度检查点以节省显存
    supports_gradient_checkpointing = True
    # 不对这些模块拆分并行化
    _no_split_modules = ["LlamaDecoderLayer"]
    # 跳过这些键的设备放置（例如 cache）
    _skip_keys_device_placement = ["past_key_values"]
    # 支持 Flash Attention v2
    _supports_flash_attn_2 = True
    # 支持 SDPA 注意力
    _supports_sdpa = True
    # 支持自定义 Cache 类
    _supports_cache_class = True

    def _init_weights(self, module):
        # 初始化权重方法，根据模块类型选择初始化策略
        std = self.config.initializer_range  # 获取初始化标准差
        if isinstance(module, nn.Linear):
            # 对线性层权重做正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                # 偏置初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层权重做正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                # 将 padding 索引处的嵌入置零
                module.weight.data[module.padding_idx].zero_()

    def _setup_cache(self, cache_cls, max_batch_size, max_cache_len: Optional[int] = None):
        # 为所有解码层的 self_attn 设置 past_key_value 缓存
        if self.config._attn_implementation == "flash_attention_2" and cache_cls == StaticCache:
            # Flash Attention v2 不兼容 StaticCache
            raise ValueError(
                "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2`"
            )

        for layer in self.model.layers:
            # 获取该层所在设备
            device = layer.input_layernorm.weight.device
            # 如果预量化 dtype 存在则使用，否则使用 o_proj 的权重 dtype
            if hasattr(self.config, "_pre_quantization_dtype"):
                dtype = self.config._pre_quantization_dtype
            else:
                dtype = layer.self_attn.o_proj.weight.dtype
            # 为每一层创建 cache 实例
            layer.self_attn.past_key_value = cache_cls(
                self.config, max_batch_size, max_cache_len,
                device=device, dtype=dtype
            )

    def _reset_cache(self):
        # 清空所有层的 past_key_value
        for layer in self.model.layers:
            layer.self_attn.past_key_value = None


# 输入参数文档字符串，用于装饰 forward 方法
LLAMA_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            输入 token 的 ID，padding 会被忽略。
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            注意力掩码，1 表示保留，0 表示屏蔽 padding。
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            位置 ID，用于位置嵌入，范围 [0, config.n_positions-1]。
        past_key_values (`Cache` or legacy tuple, *optional*):
            预计算的 KV 缓存，用于加速生成。
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            直接传入嵌入表示，替代 input_ids。
        use_cache (`bool`, *optional*):
            是否返回并更新 past_key_values。
        output_attentions (`bool`, *optional*):
            是否返回注意力权重。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。
        return_dict (`bool`, *optional*):
            是否返回 ModelOutput 对象。
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            用于更新静态 cache 的位置索引，不受 padding 影响。
"""


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAMA_START_DOCSTRING,
)
class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer 解码器，由 config.num_hidden_layers 个 LlamaDecoderLayer 组成
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        # 保存 pad_token_id 与词表大小
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Token 嵌入层
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        # 解码器层列表
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        # 最后归一化层
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 梯度检查点开关
        self.gradient_checkpointing = False

        # 权重初始化及后处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入嵌入层
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # 设置新的输入嵌入层
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        user_embed: Optional[torch.Tensor] = None,
        item_embed: Optional[torch.Tensor] = None,
        user_embed_pos: Optional[torch.Tensor] = None,
        item_embed_pos: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # 优先使用显式传入参数，否则使用配置中的默认值
        output_attentions    = output_attentions    if output_attentions    is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache            = use_cache            if use_cache            is not None else self.config.use_cache
        return_dict          = return_dict          if return_dict          is not None else self.config.use_return_dict

        # 确保只能指定 input_ids 或 inputs_embeds 其一
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        # 梯度检查点与缓存不兼容，自动关闭缓存
        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing.")
            use_cache = False

        # 若未传入嵌入，则通过嵌入层计算
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        # 若使用缓存，先将 legacy cache 转为 DynamicCache
        if use_cache:
            if not isinstance(past_key_values, StaticCache):
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                past_seen_tokens = past_key_values.get_seq_length()

        # 计算 cache_position：静态 cache 要求显式传入
        if cache_position is None:
            if isinstance(past_key_values, StaticCache):
                raise ValueError("cache_position is required when using StaticCache.")
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device
            )

        # 若未传入 position_ids，则使用 cache_position
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 构建因果掩码
        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        # 初始化隐藏状态为嵌入
        hidden_states = inputs_embeds

        # 准备收集输出
        all_hidden_states = () if output_hidden_states else None
        all_self_attns   = () if output_attentions    else None
        next_decoder_cache = None

        # 按层迭代解码
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # 如果启用梯度检查点，则通过 checkpoint 调用
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    user_embed,
                    item_embed,
                    user_embed_pos,
                    item_embed_pos,
                )
            else:
                # 正常调用解码层 forward
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    user_embed=user_embed,
                    item_embed=item_embed,
                    user_embed_pos=user_embed_pos,
                    item_embed_pos=item_embed_pos,
                )

            hidden_states = layer_outputs[0]  # 更新隐藏状态

            # 更新缓存
            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        # 最后归一化
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 将 Cache 转回 legacy 或保持 Cache 对象
        next_cache = None
        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache) else next_decoder_cache
            )

        # 根据 return_dict 决定输出格式
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(self, attention_mask, input_tensor, cache_position):
        # 根据不同的注意力实现构建因果掩码
        if self.config._attn_implementation == "flash_attention_2":
            # Flash Attention v2: 若有 0，则用原 mask；否则不需要 mask
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # 其余实现：手动构建上三角矩阵
        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min  # 用于 mask 填充值
        sequence_length = input_tensor.shape[1]

        # 判断静态或动态 cache，以确定目标长度
        if hasattr(getattr(self.layers[0], "self_attn", {}), "past_key_value"):
            target_length = self.config.max_position_embeddings
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[-1] + 1
            )

        # 先填充极小值
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device
        )
        if sequence_length != 1:
            # 上三角部分置为极小值
            causal_mask = torch.triu(causal_mask, diagonal=1)
        # 应用 cache_position，使新 tokens 位置可见
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(
            input_tensor.shape[0], 1, -1, -1
        )

        # 若提供 attention_mask，则在 padding 处保留极小值
        if attention_mask is not None:
            causal_mask = causal_mask.clone()
            if attention_mask.dim() == 2:
                mask_length = attention_mask.shape[-1]
                # padding 与 causal_mask 对齐后填充
                padding_mask = (
                    causal_mask[..., :mask_length].eq(0.0)
                    & attention_mask[:, None, None, :].eq(0.0)
                )
                causal_mask[..., :mask_length] = causal_mask[..., :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
            elif attention_mask.dim() == 4:
                # 向后兼容 4D mask
                if attention_mask.shape[-2] < cache_position[0] + sequence_length:
                    offset = cache_position[0]
                else:
                    offset = 0
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask.eq(0.0)).to(dtype=dtype) * min_dtype
                causal_mask[
                    : mask_shape[0], : mask_shape[1],
                    offset : mask_shape[2] + offset, : mask_shape[3]
                ] = mask_slice

        # SDPA 特殊处理：在 CUDA 下需要解除对部分全 mask 行的屏蔽
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
        ):
            is_tracing = (
                torch.jit.is_tracing()
                or isinstance(input_tensor, torch.fx.Proxy)
                or (hasattr(torch, "_dynamo") and torch._dynamo.is_compiling())
            )
            if not is_tracing and torch.any(attention_mask != 1):
                causal_mask = AttentionMaskConverter._unmask_unattended(
                    causal_mask, min_dtype
                )

        return causal_mask  # 返回最终因果掩码



class LlamaForCausalLM(LlamaPreTrainedModel):
    # 将 lm_head.weight 与输入嵌入层的权重绑定，通常共享同一矩阵
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # 初始化主体 LlamaModel（Transformer 解码器结构）
        self.model = LlamaModel(config)
        # 存储词表大小，用于构建输出头
        self.vocab_size = config.vocab_size
        # 定义语言模型头：从隐藏状态映射到词汇表 logits
        # bias=False 因为通常与输入嵌入矩阵共享偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 在构造完成后执行权重初始化和必要的后处理
        self.post_init()

    # 返回输入嵌入层实例，方便外部访问或修改
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置新的输入嵌入层
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 返回输出层（lm_head），用于获取或替换
    def get_output_embeddings(self):
        return self.lm_head

    # 设置新的输出嵌入层（替换 lm_head）
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 替换下游解码器部分（整个 LlamaModel）
    def set_decoder(self, decoder):
        self.model = decoder

    # 获取当前的解码器部分
    def get_decoder(self):
        return self.model

    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,            # 输入 token ID
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码
        position_ids: Optional[torch.LongTensor] = None,# 位置 ID
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 上一步生成的 KV 缓存
        inputs_embeds: Optional[torch.FloatTensor] = None,         # 直接传入的嵌入
        labels: Optional[torch.LongTensor] = None,   # 训练时用于计算损失的标签
        use_cache: Optional[bool] = None,            # 是否返回并更新缓存
        output_attentions: Optional[bool] = None,    # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None, # 是否返回所有中间隐藏状态
        return_dict: Optional[bool] = None,          # 是否返回 ModelOutput 对象
        cache_position: Optional[torch.LongTensor] = None,  # 缓存位置 ID
        user_embed: Optional[torch.Tensor] = None,      # 用户嵌入（特定场景扩展）
        item_embed: Optional[torch.Tensor] = None,      # 物品嵌入
        user_embed_pos: Optional[torch.Tensor] = None,  # 用户嵌入在序列中的位置
        item_embed_pos: Optional[torch.Tensor] = None,  # 物品嵌入在序列中的位置
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        前向传播：可同时支持训练（计算 loss）和推理（生成 logits + 缓存）
        """
        # 若显式传入选项，则使用之，否则采用配置中的默认值
        output_attentions    = output_attentions    if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict          = return_dict          if return_dict is not None else self.config.use_return_dict

        # 调用内部的 LlamaModel 完成前向，获得隐藏状态和可选缓存
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            user_embed=user_embed,
            item_embed=item_embed,
            user_embed_pos=user_embed_pos,
            item_embed_pos=item_embed_pos,
        )

        # 从 outputs 中提取最后一层隐藏状态张量
        hidden_states = outputs[0]

        # 若在多卡或 tensor parallel 场景下，按分片计算 lm_head
        if self.config.pretraining_tp > 1:
            # 将 lm_head.weight 切分成若干 slice
            lm_head_slices = self.lm_head.weight.split(
                self.vocab_size // self.config.pretraining_tp, dim=0
            )
            # 对每个 slice 单独线性映射后拼接
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            # 常规模式：直接将隐藏状态通过 lm_head 输出 logits
            logits = self.lm_head(hidden_states)

        # 确保 logits 为 float32，以防数值不稳定
        logits = logits.float()

        loss = None
        # 如果提供了 labels，则计算交叉熵损失
        if labels is not None:
            # 预测第 i 个 token 时，应匹配第 i+1 个 label
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平为 2D 张量供交叉熵损失
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        # 如果 return_dict=False，则以 tuple 形式返回 (loss?, logits, past_key_values?, hiddens?, attentions?)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 否则返回 CausalLMOutputWithPast 对象
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None,
        inputs_embeds=None, cache_position=None, **kwargs
    ):
        """
        生成时调用：为下一步生成准备好 input_ids、position_ids、attention_mask、past_key_values 等。
        处理缓存裁剪、位置 ID 生成与截断等细节。
        """
        # 检查是否存在静态 cache
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(
                getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None
            )
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            # 如果是 Cache 类，则从 cache_position 或 get_seq_length 得到已生成长度
            if isinstance(past_key_values, Cache):
                past_length = (
                    cache_position[0] if cache_position is not None
                    else past_key_values.get_seq_length()
                )
                max_cache_len = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None else None
                )
                cache_length = (
                    past_length if max_cache_len is None else torch.min(max_cache_len, past_length)
                )
            else:
                # legacy tuple 格式
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_len = None

            # 根据已经使用的缓存长度，截断 input_ids 与 attention_mask
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length):]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            if (
                max_cache_len is not None and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_len
            ):
                attention_mask = attention_mask[:, -max_cache_len:]

        # 自动生成 position_ids：累积 attention_mask 后减 1
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1]:]

        # 如果直接传入了 inputs_embeds 且没有缓存，则只用 inputs_embeds
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        # 计算新的 cache_position 范围
        input_length = (
            position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_length, past_length + input_length, device=input_ids.device
            )
        else:
            cache_position = cache_position[-input_length:]

        # 如果是静态 cache，则不要再传入 past_key_values
        if has_static_cache:
            past_key_values = None

        # 最终返回一张字典，供下一步 generation 调用 forward
        model_inputs.update({
            "position_ids": position_ids,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "user_embed": kwargs.get("user_embed"),
            "item_embed": kwargs.get("item_embed"),
            "user_embed_pos": kwargs.get("user_embed_pos"),
            "item_embed_pos": kwargs.get("item_embed_pos"),
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """
        束搜索时调用：根据 beam_idx 对 past_key_values 中的各层 KV 缓存重新索引排序
        """
        reordered = ()
        for layer_past in past_key_values:
            # 对每个 tensor 按 beam_idx 重排 batch 维
            reordered += (
                tuple(
                    state.index_select(0, beam_idx.to(state.device))
                    for state in layer_past
                ),
            )
        return reordered

