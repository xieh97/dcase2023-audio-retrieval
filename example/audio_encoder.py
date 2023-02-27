import librosa
import numpy as np
import torch
import torch.nn.functional as F

from models import audio_encoders

y, sr = librosa.load("example/example.wav", sr=None, mono=True)

win_len_secs = 0.040
hop_len_secs = 0.020
n_mels = 64
log_offset = np.spacing(1)

win_len = int(round(sr * win_len_secs))
hop_len = int(round(sr * hop_len_secs))
n_fft = 2 ** int(np.ceil(np.log(win_len) / np.log(2.0)))
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_len,
                                                 win_length=win_len, n_mels=n_mels)
log_mel = np.log(mel_spectrogram + log_offset)
log_mel = np.vstack(log_mel).transpose()  # [Time, Mel]

print("Audio log-mel:", log_mel.shape)

# %%

# Initiate CNN14 model
cnn14_encoder = audio_encoders.CNN14Encoder(out_dim=300)

# Load pretrained parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
state_dict = torch.load("example/audio_encoder.pth", map_location=device)
cnn14_encoder.load_state_dict(state_dict)
cnn14_encoder.eval()

# Construct input tensor
input_vec = torch.as_tensor(log_mel)
input_vec = torch.unsqueeze(input_vec, dim=0)

# Generate output embedding
output_embed = cnn14_encoder(input_vec)
output_embed = F.normalize(output_embed, p=2.0, dim=-1)

print("Audio embedding:", output_embed.shape)
