import torch
import torch.utils.data as tud
from torch.utils.data._utils.collate import default_collate as torch_default_collate

import torchvision.transforms as tr

import matplotlib.pyplot as plt

from squirrel.driver import MessagepackDriver
from squirrel_datasets_core.driver import TorchvisionDriver


def get_dataloader_eval(batch_size: int) -> tud.DataLoader:
    """Dataloader to load evaluation/test dataset."""

    url = "./squirrel_middlebury_patched"  # path to unzipped data folder containing *.gz files
    # Get iterator from driver
    driver = MessagepackDriver(url)
    it = driver.get_iter()

    #############################
    ## YOUR PREPROCESSING HERE ##
    preprocess = tr.Compose([
        lambda x: x
    ])
    #############################

    dataset = (
        it
        .map(preprocess)
        .batched(batch_size, torch_default_collate, drop_last_if_not_full=False)
        .to_torch_iterable()
    )
    return tud.DataLoader(dataset, shuffle=None, batch_size=None)


def get_dataloader_train(batch_size: int, shuffe_size: int = 100, num_workers: int = 0) -> tud.DataLoader:
    """Dataloader to Sintel training data."""
    # Path to folder containing the `Sintel` folder previously donwloaded.
    url = "./"

    driver = TorchvisionDriver("SintelStereo", url=url)
    it = driver.get_iter()

    preprocess = tr.Compose([
        lambda x: (tr.ToTensor()(x[0]), tr.ToTensor()(x[1]))
    ])

    dataset = (
        it
        .shuffle(shuffe_size)
        .split_by_worker_pytorch()
        #############################################################
        ### YOUR PREPROCESSING, COLLATING, AUGMENTATION, ETC. HERE ##
        #############################################################
        .map(preprocess)
        .batched(batch_size, torch_default_collate, drop_last_if_not_full=True)
        .to_torch_iterable()
    )
    return tud.DataLoader(dataset, shuffle=None, batch_size=None, num_workers=num_workers)

batch_size = 1
# dl_eval = get_dataloader_eval(batch_size)
dl_train = get_dataloader_train(batch_size)

for i, d in enumerate(dl_train):

    # img_l = d["img_l"][0] # 32, 3, 360, 360
    # img_r = d["img_r"][0] # 32, 3, 360, 360
    img_l = d[0][0]  # 32, 3, 360, 360
    img_r = d[1][0]  # 32, 3, 360, 360
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(img_l.permute(1, 2, 0).numpy())
    ax[1].imshow(img_r.permute(1, 2, 0).numpy())
    ax[0].set_title(f"{img_l.shape}, {img_l.dtype}, {img_l.min()}, {img_l.max()}")
    fig.tight_layout()

    plt.savefig(f"./data_train/{i}.png")

    if i == 100:
        break

exit()
# Your Code
from typing import Dict, List, Optional, Set, Tuple, Union
from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTModel, ViTConfig, ViTPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPooling, BaseModelOutput
from torch.nn import CosineSimilarity
cos = CosineSimilarity(dim=1, eps=1e-6)

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')


# inputs = torch.randn(3, 360, 360)
# model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
# model = ViTModel.from_pretrained('google/vit-base-patch16-224')

## Baseline Evaluation

def get_eval_performance(dl_eval, model):
    model.eval()
    left_embeddings = torch.zeros(306, 768)
    right_embeddings = torch.zeros(306, 768)

    # # Calculate all embeddings
    for i, d in enumerate(dl_eval):
        # print(i)
        bs = d["img_l"].shape[0]
        img_l = torch.unbind(d["img_l"]) # 2, 3, 360, 360
        inputs_l = feature_extractor(images=img_l, return_tensors="pt")
        inputs_l = inputs_l["pixel_values"].cuda()
        outputs_l = model(inputs_l)
        img_r = torch.unbind(d["img_r"])  # 32, 3, 360, 360
        inputs_r = feature_extractor(images=img_r, return_tensors="pt")
        inputs_r = inputs_r["pixel_values"].cuda()
        outputs_r = model(inputs_r)

        left_embeddings[i * batch_size:i * batch_size + bs, :] = outputs_l.pooler_output.data.cpu()
        right_embeddings[i * batch_size:i * batch_size + bs, :] = outputs_r.pooler_output.data.cpu()

    match_matrix = torch.zeros(306, 306)
    accuracy = []
    for i in range(306):
        for j in range(306):
            match_matrix[i][j] = cos(left_embeddings[i:i+1], right_embeddings[j:j+1])
        match = torch.argmax(match_matrix[i:i+1], dim=1)
        if match == i:
            accuracy.append(1.)
        else:
            accuracy.append(0.)

    return torch.tensor(accuracy), match_matrix


## Adapter Fine Tuning
from torch import nn

ViTBase = ViTModel.from_pretrained('google/vit-base-patch16-224')
for param in ViTBase.parameters():
    if param.requires_grad:
        param.requires_grad = False

adapter_config = dict()
adapter_config["hidden_size"] = ViTBase.config.hidden_size
adapter_config["bottleneck_dim"] = 8
adapter_config["activation"] = nn.GELU

class Adapter(nn.Module):
    def __init__(self, adapter_config: dict):
        super(Adapter, self).__init__()
        self.down_project = nn.Linear(adapter_config["hidden_size"], adapter_config["bottleneck_dim"])
        self.activation = adapter_config["activation"]()
        self.up_project = nn.Linear(adapter_config["bottleneck_dim"], adapter_config["hidden_size"])
        self._init_params()

    def _init_params(self):
        for param in self.down_project.parameters():
            # torch.nn.init.constant_(param, 0.)
            torch.nn.init.normal_(param, 0., 1e-5)
        for param in self.up_project.parameters():
            # torch.nn.init.constant_(param, 0.)
            torch.nn.init.normal_(param, 0., 1e-5)

    def forward(self, hidden_states):
        # hidden_states = (B, T, D)
        outputs = self.down_project(hidden_states)
        outputs = self.activation(outputs)
        outputs = self.up_project(outputs)

        adapter_outputs = outputs + hidden_states
        return adapter_outputs

class ViTWithAdapterOutput(nn.Module):
    def __init__(self, ViTBaseOutput: nn.Module) -> None:
        super().__init__()
        self.ViTBaseOutput = ViTBaseOutput
        self.dense = ViTBaseOutput.dense
        self.dropout = ViTBaseOutput.dropout
        self.adapter = Adapter(adapter_config)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Second adapter of the ViTWithAdapter layer implemented here
        hidden_states = self.adapter(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViTWithAdapterLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, ViTBaseLayer: nn.Module) -> None:
        super().__init__()
        self.ViTBaseLayer = ViTBaseLayer
        self.chunk_size_feed_forward = ViTBaseLayer.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ViTBaseLayer.attention
        self.intermediate = ViTBaseLayer.intermediate
        self.output = ViTWithAdapterOutput(ViTBaseLayer.output)
        self.layernorm_before = ViTBaseLayer.layernorm_before
        self.layernorm_after = ViTBaseLayer.layernorm_after
        self.adapter = Adapter(adapter_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # First adapter of the ViTWithAdapter layer implemented here
        adapter_output = self.adapter(attention_output)

        # first residual connection
        hidden_states = adapter_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTWithAdapterEncoder(nn.Module):
    def __init__(self, ViTBaseEncoder: nn.Module) -> None:
        super().__init__()
        self.ViTBaseEncoder = ViTBaseEncoder
        self.layer = nn.ModuleList([ViTWithAdapterLayer(layer) for layer in ViTBaseEncoder.layer])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class ViTWithAdapter(nn.Module):
    def __init__(self, ViTBase: ViTPreTrainedModel):
        super(ViTWithAdapter, self).__init__()
        self.ViTBase = ViTBase
        self.encoder = ViTWithAdapterEncoder(ViTBase.encoder)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.ViTBase.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.ViTBase.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.ViTBase.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.ViTBase.get_head_mask(head_mask, self.ViTBase.config.num_hidden_layers)

        embedding_output = self.ViTBase.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        sequence_output = self.ViTBase.layernorm(sequence_output)
        pooled_output = self.ViTBase.pooler(sequence_output) if self.ViTBase.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

## Baseline performance reproduction
# accuracy, match_matrix = get_eval_performance(dl_eval, model)
# print(accuracy.mean())
# plt.imshow(match_matrix)

## Training code
epochs = 3
batch_size = 16
lr = 3e-4

dl_train = get_dataloader_train(batch_size)

ViTBase.train()
for param in ViTBase.parameters():
    if param.requires_grad:
        param.requires_grad = False
model = ViTWithAdapter(ViTBase)
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=len(dl_train), epochs=epochs,
                                                pct_start=0.1, anneal_strategy='linear', div_factor=10., final_div_factor=100.)
loss_fn = nn.CosineEmbeddingLoss()


# Training loop
for epoch in range(epochs):
    for i, d in enumerate(dl_train):
        optimizer.zero_grad()
        img_l = torch.unbind(d["img_l"])  # B, 3, 360, 360
        inputs_l = feature_extractor(images=img_l, return_tensors="pt")
        inputs_l = inputs_l["pixel_values"].cuda() # B, 197, 224

        img_r = torch.unbind(d["img_r"])  # B, 3, 360, 360
        inputs_r = feature_extractor(images=img_r, return_tensors="pt")
        inputs_r = inputs_r["pixel_values"].cuda() # B, 197, 224

        outputs_l = model(inputs_l)
        outputs_r = model(inputs_r)

        loss = loss_fn(outputs_l.pooler_output, outputs_r.pooler_output, torch.tensor([1]))
        loss.backward()
        optimizer.step()
        # scheduler.step()


mask = 2*torch.eye(batch_size)-1.

def contrastive_loss(outputs_l, outputs_r, mask, batch_size):
    loss = torch.tensor(0.).cuda()
    for b in range(batch_size):
        x = loss_fn(torch.tile(outputs_l[b:b + 1], [batch_size, 1]), outputs_r, mask[b])
        y = loss_fn(torch.tile(outputs_r[b:b + 1], [batch_size, 1]), outputs_l, mask[b])
        loss += (x[b] + (torch.sum(x[:b]) + torch.sum(x[b + 1:])) / (batch_size - 1)) / 4.
        loss += (y[b] + (torch.sum(y[:b]) + torch.sum(y[b + 1:])) / (batch_size - 1)) / 4.
    return loss/batch_size





