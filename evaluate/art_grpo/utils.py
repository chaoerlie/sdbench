import torch

def normalize_advantage(rewards):
    """
    对一组图像的奖励值进行标准化（z-score），作为 GRPO 中的 Advantage。

    Args:
        rewards (torch.Tensor): shape [G], 奖励值

    Returns:
        torch.Tensor: shape [G], 标准化 advantage
    """
    return (rewards - rewards.mean()) / (rewards.std() + 1e-8)


def grpo_loss(logits, advantages):
    """
    GRPO 损失函数：L = -sum( log_prob * advantage )

    Args:
        logits (torch.Tensor): 策略网络输出分数，shape: [G]
        advantages (torch.Tensor): 标准化优势，shape: [G]

    Returns:
        torch.Tensor: scalar loss
    """
    return -torch.mean(logits * advantages.detach())
