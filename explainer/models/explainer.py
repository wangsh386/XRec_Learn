import torch                                # PyTorch 主库，提供张量（Tensor）及自动求导等功能
import torch.nn as nn                       # 神经网络模块，包含各种层（Layer）和容器（Module）
import torch.nn.functional as F             # 函数式接口，包括激活函数、损失函数等
from transformers import LlamaTokenizer    # HuggingFace Transformers 中的 Llama 分词器
from models.modeling_explainer import LlamaForCausalLM  # 自定义或第三方库中的 Llama 因果语言模型

# ---------------------------------------------
# 定义一个“参数化白化层”（Parametric Whitening Layer）
# ---------------------------------------------
class PWLayer(nn.Module):
    """Single Parametric Whitening Layer
    输入：x，先做 dropout，再减去可学习的偏置（bias），最后线性映射到目标维度。
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()
        # dropout 层：训练时随机丢弃部分特征，减少过拟合
        self.dropout = nn.Dropout(p=dropout)
        # bias 向量：初始化为 0，可学习参数，用于中心化（去掉均值）
        self.bias = nn.Parameter(torch.zeros(input_size), requires_grad=True)
        # 线性层：不带 bias，将 input_size 投影到 output_size
        self.lin = nn.Linear(input_size, output_size, bias=False)
        # 初始化子模块权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """对 Linear 层的权重进行正态分布初始化"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        """
        前向计算：
        1. dropout(x)
        2. 减去 bias 向量：centered = dropout(x) - bias
        3. 线性映射：lin(centered)
        """
        centered = self.dropout(x) - self.bias  # 中心化
        return self.lin(centered)              # 投影到 output_size
    
# ---------------------------------------------
# 定义一个 MoE（Mixture of Experts）增强的适配器层
# ---------------------------------------------
class MoEAdaptorLayer(nn.Module):
    """MoE-enhanced Adaptor
    使用多个 PWLayer 作为专家，并通过可学习的 gating 网络加权融合专家输出。
    """
    def __init__(self, n_exps=8, layers=[64, 4096], dropout=0.2, noise=True):
        super(MoEAdaptorLayer, self).__init__()
        self.n_exps = n_exps            # 专家数量
        self.noisy_gating = noise       # 是否启用噪声 gating

        # 创建 n_exps 个专家，每个专家是一个 PWLayer
        self.experts = nn.ModuleList([
            PWLayer(layers[0], layers[1], dropout)
            for _ in range(n_exps)
        ])
        # gating 权重矩阵：clean_logits = x @ w_gate
        self.w_gate = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        # 噪声权重矩阵：raw_noise = x @ w_noise
        self.w_noise = nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        """
        计算 gating 分布：
        - clean_logits = x @ w_gate
        - 若训练且 noisy_gating=True，则加噪声：noisy_logits = clean_logits + N(0, softplus(x@w_noise))
        - gates = softmax(logits)
        """
        clean_logits = x @ self.w_gate  # (batch_size, n_exps)
        if self.noisy_gating and train:
            raw_noise = x @ self.w_noise
            noise_std = F.softplus(raw_noise) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_std
            logits = noisy_logits
        else:
            logits = clean_logits
        gates = F.softmax(logits, dim=-1)  # 归一化到 [0,1]
        return gates

    def forward(self, x):
        """
        前向流程：
        1. 计算 gates
        2. 每个专家输出 expert_i(x)
        3. 按 gates 加权并求和
        """
        gates = self.noisy_top_k_gating(x, self.training)  # (B, n_exps)

        # 收集所有专家的输出，shape 列表 [(B, output_dim)]
        expert_outputs = [expert(x).unsqueeze(-2) for expert in self.experts]
        # 拼接: (B, n_exps, output_dim)
        expert_outputs = torch.cat(expert_outputs, dim=-2)

        # gates: (B, n_exps) -> (B, n_exps, 1)
        weighted = gates.unsqueeze(-1) * expert_outputs
        # sum over experts dim -> (B, output_dim)
        return weighted.sum(dim=-2)

# ---------------------------------------------
# 整合适配器与 Llama 模型，定义 Explainer 模块
# ---------------------------------------------
class Explainer(torch.nn.Module):
    def __init__(self, token_size=4096, user_embed_size=64, item_embed_size=64):
        super(Explainer, self).__init__()
        # 可选：登录 Hugging Face Hub（私有模型）
        from huggingface_hub import login
        login()

        # 加载 Llama-2-7b 量化模型，节省显存
        model_name = "meta-llama/Llama-2-7b-chat-hf"
        self.model = LlamaForCausalLM.from_pretrained(model_name, load_in_8bit=True)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)

        # 添加特殊 token，用作占位和控制
        special_tokens = {
            "additional_special_tokens": ["<USER_EMBED>", "<ITEM_EMBED>", "<EXPLAIN_POS>"]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.tokenizer.pad_token = "<pad>"
        # 调整模型嵌入维度匹配新词表长度
        self.model.resize_token_embeddings(len(self.tokenizer))

        # 冻结原模型参数，只训练下游适配器
        for param in self.model.parameters():
            param.requires_grad = False

        # 用户/物品嵌入转换：从小维度映射到 token_size
        self.user_embedding_converter = MoEAdaptorLayer(
            n_exps=8, layers=[user_embed_size, token_size], dropout=0.2, noise=True
        )
        self.item_embedding_converter = MoEAdaptorLayer(
            n_exps=8, layers=[item_embed_size, token_size], dropout=0.2, noise=True
        )

    def forward(self, user_embedding, item_embedding, input_text):
        """
        训练 / 推理 前向：
        1. 转换 user/item 嵌入并 half()
        2. 分词得到 input_ids
        3. 获取原始 inputs_embeds = embedding_layer(input_ids)
        4. 定位并替换特殊 token 的 embedding
        5. 调用 model，返回 logits 和解释位置
        """
        # 1. 转换并量化到 half
        converted_user = self.user_embedding_converter(user_embedding).half()
        converted_item = self.item_embedding_converter(item_embedding).half()

        # 2. 分词：文本 -> input_ids
        # 例如 input_text = ["推荐理由：<USER_EMBED><ITEM_EMBED><EXPLAIN_POS>请解释"]
        tokenized = self.tokenizer(input_text, padding=True, return_tensors="pt")
        input_ids = tokenized["input_ids"]  # shape: (B, seq_len)

        # 3. lookup 原始 embedding
        # 形状: (B, seq_len, hidden_size)
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # 4. 定位特殊 token ID
        uid = self.tokenizer.convert_tokens_to_ids("<USER_EMBED>")
        iid = self.tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")
        pid = self.tokenizer.convert_tokens_to_ids("<EXPLAIN_POS>")

        # 定位它们在序列中的位置 (batch_index, seq_index)
        coords_u = (input_ids == uid).nonzero()  # e.g., tensor([[0, 1]])
        coords_i = (input_ids == iid).nonzero()  # e.g., tensor([[0, 2]])
        coords_p = (input_ids == pid).nonzero()  # e.g., tensor([[0, 3]])
        # 仅取 seq_index
        pos_u = coords_u[:, 1]  # tensor([1])
        pos_i = coords_i[:, 1]  # tensor([2])
        pos_p = coords_p[:, 1]  # tensor([3])

        # 5. 用转换后的嵌入覆盖原 embedding
        batch_idx = torch.arange(input_ids.size(0))  # tensor([0,...,B-1])
        inputs_embeds[batch_idx, pos_u, :] = converted_user
        inputs_embeds[batch_idx, pos_i, :] = converted_item

        # 调用模型，传入 embedding 和位置信息
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            user_embed=converted_user,
            item_embed=converted_item,
            user_embed_pos=coords_u[:, 1:],
            item_embed_pos=coords_i[:, 1:]
        )
        return input_ids, outputs, pos_p

    def loss(self, input_ids, outputs, explain_pos, device):
        """
        计算交叉熵损失，仅对 <EXPLAIN_POS> 之后的 token 监督：
        - mask 解释位置之前的所有 input_ids = -100 (忽略)
        - shift_logits & shift_labels -> CrossEntropyLoss
        """
        seq_len = input_ids.size(1)
        interval = torch.arange(seq_len).to(device)  # [0..seq_len-1]
        mask = interval[None, :] < explain_pos[:, None]
        input_ids[mask] = -100  # CrossEntropyLoss 忽略 -100

        logits = outputs.logits  # (B, seq_len, vocab_size)
        # 右移
        shift_labels = input_ids[:, 1:].contiguous()
        shift_logits = logits[:, :-1, :].contiguous()
        # 展平
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        shift_labels = shift_labels.view(-1)
        return nn.CrossEntropyLoss()(shift_logits, shift_labels)

    def generate(self, user_embedding, item_embedding, input_text):
        """
        文本生成：与 forward 类似，但调用 model.generate()
        """
        converted_user = self.user_embedding_converter(user_embedding).half()
        converted_item = self.item_embedding_converter(item_embedding).half()
        tokenized = self.tokenizer(input_text, padding=True, return_tensors="pt")
        input_ids = tokenized["input_ids"]
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        uid = self.tokenizer.convert_tokens_to_ids("<USER_EMBED>")
        iid = self.tokenizer.convert_tokens_to_ids("<ITEM_EMBED>")
        coords_u = (input_ids == uid).nonzero()
        coords_i = (input_ids == iid).nonzero()
        pos_u = coords_u[:, 1]
        pos_i = coords_i[:, 1]
        batch_idx = torch.arange(input_ids.size(0))
        inputs_embeds[batch_idx, pos_u, :] = converted_user
        inputs_embeds[batch_idx, pos_i, :] = converted_item

        # 调用生成接口
        generated_ids = self.model.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=128,
            user_embed=converted_user,
            item_embed=converted_item,
            user_embed_pos=coords_u[:, 1:],
            item_embed_pos=coords_i[:, 1:]
        )
        # 解码为文本
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
