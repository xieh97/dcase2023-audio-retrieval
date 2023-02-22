import torch.nn as nn
import torch.nn.functional as F

from models import audio_encoders, text_encoders


class DualEncoderModel(nn.Module):

    def __init__(self, *args, **kwargs):
        super(DualEncoderModel, self).__init__()

        self.out_norm = kwargs.get("out_norm", None)
        self.audio_enc = getattr(audio_encoders, args[0], None)(**kwargs["audio_enc"])
        self.text_enc = getattr(text_encoders, args[1], None)(**kwargs["text_enc"])

        # Load pretrained weights for audio encoder
        if kwargs["audio_enc"]["init"] == "prior":
            self.audio_enc.load_state_dict(kwargs["audio_enc"]["weight"])

            # Freeze or fine-tune pretrained weights
            if kwargs["audio_enc"]["name"] == "CNN14Encoder":
                for param in self.audio_enc.bn0.parameters():
                    param.requires_grad = kwargs["audio_enc"].get("trainable", False)

                for param in self.audio_enc.cnn.parameters():
                    param.requires_grad = kwargs["audio_enc"].get("trainable", False)

                for param in self.audio_enc.fc.parameters():
                    param.requires_grad = kwargs["audio_enc"].get("trainable", False)

    def audio_branch(self, audio):
        audio_embeds = self.audio_enc(audio)

        if self.out_norm == "L2":
            audio_embeds = F.normalize(audio_embeds, p=2.0, dim=-1)

        return audio_embeds

    def text_branch(self, text):
        text_embeds = self.text_enc(text)

        if self.out_norm == "L2":
            text_embeds = F.normalize(text_embeds, p=2.0, dim=-1)

        return text_embeds

    def forward(self, audio, text):
        """
        :param audio: tensor, (batch_size, time_steps, Mel_bands).
        :param text: tensor, (batch_size, len_padded_text).
        """
        audio_embeds = self.audio_branch(audio)
        text_embeds = self.text_branch(text)

        # audio_embeds: [N, E]    text_embeds: [N, E]
        return audio_embeds, text_embeds
