import torch
import torch.nn as nn
import torch.nn.functional as F


class LogSoftmaxLoss(nn.Module):

    def __init__(self, **kwargs):
        super(LogSoftmaxLoss, self).__init__()

        self.temperature = kwargs["temperature"]
        self.dist = kwargs.get("dist", "dot_product")

    def forward(self, audio_embeds, text_embeds, item_batch):
        """
        :param audio_embeds: tensor, (N, E).
        :param text_embeds: tensor, (N, E).
        :param item_batch: list of audio-text infos.
        :return:
        """
        N = audio_embeds.size(0)

        loss = torch.tensor(0., device=audio_embeds.device, requires_grad=True)

        for i in range(N):
            # Anchor audio-text pair
            A_i, T_i = audio_embeds[i], text_embeds[i]

            # Negative + Anchor audio-text pairs
            sample_indexes = [j for j in range(N) if item_batch[j]["fid"] != item_batch[i]["fid"]]
            sample_indexes.append(i)

            S_ai = score(audio_embeds[sample_indexes], T_i, self.dist) / self.temperature  # (N')
            S_it = score(A_i, text_embeds[sample_indexes], self.dist) / self.temperature  # (N')

            target = torch.as_tensor([j == i for j in sample_indexes], dtype=torch.float,
                                     device=audio_embeds.device)  # (N')

            # Log softmax loss (i.e., InfoNCE Loss, NT-Xent Loss, Multi-class N-pair Loss, Categorical CE Loss)
            L_ai = F.cross_entropy(S_ai, target)
            L_it = F.cross_entropy(S_it, target)

            loss = loss + L_ai + L_it

        loss = loss / N

        return loss


def score(audio_embed, text_embed, dist):
    """
    :param audio_embed: tensor, (E,) or (N, E).
    :param text_embed: tensor, (E,) or (N, E).
    """

    if dist == "dot_product":
        return torch.matmul(audio_embed, text_embed.t())

    elif dist == "cosine_similarity":
        return F.cosine_similarity(audio_embed, text_embed, -1, 1e-8)
