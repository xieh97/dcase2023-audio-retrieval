import torch

from models import audio_encoders

# new_key : old_key
keymap = {
    "bn0.weight": "bn0.weight",
    "bn0.bias": "bn0.bias",
    "bn0.running_mean": "bn0.running_mean",
    "bn0.running_var": "bn0.running_var",
    "bn0.num_batches_tracked": "bn0.num_batches_tracked",

    "cnn.0.weight": "conv_block1.conv1.weight",
    "cnn.1.weight": "conv_block1.bn1.weight",
    "cnn.1.bias": "conv_block1.bn1.bias",
    "cnn.1.running_mean": "conv_block1.bn1.running_mean",
    "cnn.1.running_var": "conv_block1.bn1.running_var",
    "cnn.1.num_batches_tracked": "conv_block1.bn1.num_batches_tracked",
    "cnn.3.weight": "conv_block1.conv2.weight",
    "cnn.4.weight": "conv_block1.bn2.weight",
    "cnn.4.bias": "conv_block1.bn2.bias",
    "cnn.4.running_mean": "conv_block1.bn2.running_mean",
    "cnn.4.running_var": "conv_block1.bn2.running_var",
    "cnn.4.num_batches_tracked": "conv_block1.bn2.num_batches_tracked",

    "cnn.8.weight": "conv_block2.conv1.weight",
    "cnn.9.weight": "conv_block2.bn1.weight",
    "cnn.9.bias": "conv_block2.bn1.bias",
    "cnn.9.running_mean": "conv_block2.bn1.running_mean",
    "cnn.9.running_var": "conv_block2.bn1.running_var",
    "cnn.9.num_batches_tracked": "conv_block2.bn1.num_batches_tracked",
    "cnn.11.weight": "conv_block2.conv2.weight",
    "cnn.12.weight": "conv_block2.bn2.weight",
    "cnn.12.bias": "conv_block2.bn2.bias",
    "cnn.12.running_mean": "conv_block2.bn2.running_mean",
    "cnn.12.running_var": "conv_block2.bn2.running_var",
    "cnn.12.num_batches_tracked": "conv_block2.bn2.num_batches_tracked",

    "cnn.16.weight": "conv_block3.conv1.weight",
    "cnn.17.weight": "conv_block3.bn1.weight",
    "cnn.17.bias": "conv_block3.bn1.bias",
    "cnn.17.running_mean": "conv_block3.bn1.running_mean",
    "cnn.17.running_var": "conv_block3.bn1.running_var",
    "cnn.17.num_batches_tracked": "conv_block3.bn1.num_batches_tracked",
    "cnn.19.weight": "conv_block3.conv2.weight",
    "cnn.20.weight": "conv_block3.bn2.weight",
    "cnn.20.bias": "conv_block3.bn2.bias",
    "cnn.20.running_mean": "conv_block3.bn2.running_mean",
    "cnn.20.running_var": "conv_block3.bn2.running_var",
    "cnn.20.num_batches_tracked": "conv_block3.bn2.num_batches_tracked",

    "cnn.24.weight": "conv_block4.conv1.weight",
    "cnn.25.weight": "conv_block4.bn1.weight",
    "cnn.25.bias": "conv_block4.bn1.bias",
    "cnn.25.running_mean": "conv_block4.bn1.running_mean",
    "cnn.25.running_var": "conv_block4.bn1.running_var",
    "cnn.25.num_batches_tracked": "conv_block4.bn1.num_batches_tracked",
    "cnn.27.weight": "conv_block4.conv2.weight",
    "cnn.28.weight": "conv_block4.bn2.weight",
    "cnn.28.bias": "conv_block4.bn2.bias",
    "cnn.28.running_mean": "conv_block4.bn2.running_mean",
    "cnn.28.running_var": "conv_block4.bn2.running_var",
    "cnn.28.num_batches_tracked": "conv_block4.bn2.num_batches_tracked",

    "cnn.32.weight": "conv_block5.conv1.weight",
    "cnn.33.weight": "conv_block5.bn1.weight",
    "cnn.33.bias": "conv_block5.bn1.bias",
    "cnn.33.running_mean": "conv_block5.bn1.running_mean",
    "cnn.33.running_var": "conv_block5.bn1.running_var",
    "cnn.33.num_batches_tracked": "conv_block5.bn1.num_batches_tracked",
    "cnn.35.weight": "conv_block5.conv2.weight",
    "cnn.36.weight": "conv_block5.bn2.weight",
    "cnn.36.bias": "conv_block5.bn2.bias",
    "cnn.36.running_mean": "conv_block5.bn2.running_mean",
    "cnn.36.running_var": "conv_block5.bn2.running_var",
    "cnn.36.num_batches_tracked": "conv_block5.bn2.num_batches_tracked",

    "cnn.40.weight": "conv_block6.conv1.weight",
    "cnn.41.weight": "conv_block6.bn1.weight",
    "cnn.41.bias": "conv_block6.bn1.bias",
    "cnn.41.running_mean": "conv_block6.bn1.running_mean",
    "cnn.41.running_var": "conv_block6.bn1.running_var",
    "cnn.41.num_batches_tracked": "conv_block6.bn1.num_batches_tracked",
    "cnn.43.weight": "conv_block6.conv2.weight",
    "cnn.44.weight": "conv_block6.bn2.weight",
    "cnn.44.bias": "conv_block6.bn2.bias",
    "cnn.44.running_mean": "conv_block6.bn2.running_mean",
    "cnn.44.running_var": "conv_block6.bn2.running_var",
    "cnn.44.num_batches_tracked": "conv_block6.bn2.num_batches_tracked",

    "fc.1.weight": "fc1.weight",
    "fc.1.bias": "fc1.bias",
}

# Parameters of pretrained CNN14
cnn14_params = torch.load("YOUR_DATA_PATH/pretrained_models/Cnn14_mAP=0.431.pth")["model"]

# Transfer pretrained parameters
cnn14_encoder = audio_encoders.CNN14Encoder(out_dim=300)
state_dict = cnn14_encoder.state_dict().copy()
for key in keymap:
    state_dict[key] = cnn14_params[keymap[key]]
cnn14_encoder.load_state_dict(state_dict)

# Save transferred CNN14
torch.save(cnn14_encoder.state_dict(), "YOUR_DATA_PATH/pretrained_models/CNN14_300.pth")
