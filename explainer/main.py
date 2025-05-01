# 导入必要的库
import os           # 操作系统接口，用于文件路径、目录操作等
import pickle       # 用于序列化和反序列化Python对象（保存/加载列表、字典等）
import json         # 处理JSON格式数据
import torch        # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块

# 从项目中导入自定义模块
from models.explainer import Explainer   # 自定义模型：解释生成器
from utils.data_handler import DataHandler  # 数据处理工具类
from utils.parse import args              # 参数解析模块（命令行参数）

# 设置设备为GPU（如果可用）或CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")  # 输出当前使用的设备信息

# 定义主类 XRec（Explainable Recommender System）
class XRec:
    def __init__(self):
        # 打印当前使用的数据集名称
        print(f"dataset: {args.dataset}")

        # 初始化模型Explainer，并将其移动到指定设备上
        self.model = Explainer().to(device)

        # 初始化数据处理器
        self.data_handler = DataHandler()

        # 加载训练、验证、测试数据集
        self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()

        # 用户嵌入转换器和物品嵌入转换器的保存路径
        self.user_embedding_converter_path = f"./data/{args.dataset}/user_converter.pkl"
        self.item_embedding_converter_path = f"./data/{args.dataset}/item_converter.pkl"

        # 测试预测结果与参考文本的保存路径
        self.tst_predictions_path = f"./data/{args.dataset}/tst_predictions.pkl"
        self.tst_references_path = f"./data/{args.dataset}/tst_references.pkl"

    # 训练函数
    def train(self):
        # 使用Adam优化器，学习率由参数指定
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        # 开始训练循环，遍历所有epoch
        for epoch in range(args.epochs):
            total_loss = 0  # 累计损失值
            self.model.train()  # 设置模型为训练模式

            # 遍历训练数据中的每一个batch
            for i, batch in enumerate(self.trn_loader):
                user_embed, item_embed, input_text = batch  # 解包batch数据

                # 将用户和物品嵌入移动到指定设备
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                # 前向传播获取输入ID、输出序列和解释位置
                input_ids, outputs, explain_pos_position = self.model.forward(user_embed, item_embed, input_text)

                input_ids = input_ids.to(device)  # 输入ID也移到设备上
                explain_pos_position = explain_pos_position.to(device)  # 解释位置也移到设备上

                optimizer.zero_grad()  # 清除梯度

                # 计算损失
                loss = self.model.loss(input_ids, outputs, explain_pos_position, device)

                loss.backward()     # 反向传播计算梯度
                optimizer.step()    # 更新参数

                total_loss += loss.item()  # 累加损失

                # 每100步打印一次训练信息
                if i % 100 == 0 and i != 0:
                    print(
                        f"Epoch [{epoch}/{args.epochs}], Step [{i}/{len(self.trn_loader)}], Loss: {loss.item()}"
                    )
                    print(f"Generated Explanation: {outputs[0]}")  # 打印第一个样本的生成解释

            # 每个epoch结束时打印总损失
            print(f"Epoch [{epoch}/{args.epochs}], Loss: {total_loss}")

            # 保存用户和物品嵌入转换器的状态
            torch.save(
                self.model.user_embedding_converter.state_dict(),
                self.user_embedding_converter_path,
            )
            torch.save(
                self.model.item_embedding_converter.state_dict(),
                self.item_embedding_converter_path,
            )

            # 打印保存路径
            print(f"Saved model to {self.user_embedding_converter_path}")
            print(f"Saved model to {self.item_embedding_converter_path}")

    # 评估函数（生成解释）
    def evaluate(self):
        loader = self.tst_loader               # 使用测试数据加载器
        predictions_path = self.tst_predictions_path  # 预测结果保存路径
        references_path = self.tst_references_path    # 参考答案保存路径

        # 加载训练好的用户和物品嵌入转换器状态
        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_embedding_converter_path)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_embedding_converter_path)
        )

        self.model.eval()  # 设置模型为评估模式

        predictions = []   # 存储模型生成的解释
        references = []    # 存储真实解释

        with torch.no_grad():  # 不计算梯度
            for i, batch in enumerate(loader):
                # 解包batch数据
                user_embed, item_embed, input_text, explain = batch

                # 移动到指定设备
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                # 生成解释
                outputs = self.model.generate(user_embed, item_embed, input_text)

                # 后处理：去除生成解释中不必要的部分（如"[..."之后的内容）
                end_idx = outputs[0].find("[")
                if end_idx != -1:
                    outputs[0] = outputs[0][:end_idx]

                # 保存预测和参考解释
                predictions.append(outputs[0])
                references.append(explain[0])

                # 每10步打印一次进度
                if i % 10 == 0 and i != 0:
                    print(f"Step [{i}/{len(loader)}]")
                    print(f"Generated Explanation: {outputs[0]}")

        # 将预测和参考解释保存为pickle文件
        with open(predictions_path, "wb") as file:
            pickle.dump(predictions, file)
        with open(references_path, "wb") as file:
            pickle.dump(references, file)

        # 打印保存路径
        print(f"Saved predictions to {predictions_path}")
        print(f"Saved references to {references_path}")

# 主程序入口
def main():
    sample = XRec()  # 创建XRec实例

    # 根据参数决定运行模式：微调模型 或 生成解释
    if args.mode == "finetune":
        print("Finetune model...")  # 微调模型
        sample.train()
    elif args.mode == "generate":
        print("Generating explanations...")  # 生成解释
        sample.evaluate()

# 如果是直接运行该脚本，则执行main函数
if __name__ == "__main__":
    main()
