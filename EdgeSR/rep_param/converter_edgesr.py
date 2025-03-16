''' converter '''
import math
import argparse
import os
from typing import OrderedDict
import torch
from basicsr.archs.edgesr_arch import EdgeSR,EdgeSRForInference
import torch.nn.functional as F


def get_info(state: OrderedDict) -> dict:

    return {
        'num_in_ch': 1,
        'num_out_ch': 1,
        'num_feat': 32,
        'num_block': 6,
        'num_residual_feat': 1,
        'upscale': 2,
    }


def model_allclose(model1: EdgeSR, model2: EdgeSRForInference, num_in_ch: int) -> bool:
    # Randomly generate input
    input = torch.rand(4, num_in_ch, 64,64 )
    out1 = torch.clamp(model1(input)[0], min=0, max=1)
    out2 = model2(input)    # No clip operation added for model2 because the logic is already embedded.
    print(torch.max((out1 - out2).abs()).item() * 255)

    return torch.max((out1 - out2).abs()).item() * 255 < 0.01


def convert(state: OrderedDict) -> OrderedDict:
    # Get the hyperparameters of the model according to the checkpoint
    info = get_info(state)
    # Create model objects based on hyperparameters and load checkpoints.
    model1 = EdgeSR(**info)
    model1.eval()
    # state = filter_state_dict(state, model1)
    print(model1)  # 输出模型结构，确保它正确初始化

    model1.load_state_dict(state)
    model2 = EdgeSRForInference(**info)

    # for conv_first
    if hasattr(model1.conv_first, 'rep_params'):
        weight, bias = model1.conv_first.rep_params()
    else:
        weight, bias = model1.conv_first.weight.data, model1.conv_first.bias.data
    model2.conv_first.weight.data = torch.cat([weight, model1.conv_residual_first.weight.data])
    model2.conv_first.bias.data = torch.cat([bias, model1.conv_residual_first.bias.data])

    # for backbones
    for (backbone_conv, residual_conv, add_residual_conv,add_expand_conv,add_convs,add_test,
         layer2) in zip(model1.backbone_convs, model1.residual_convs,
                        model1.add_residual_convs,model1.add_expand_convs,model1.add_convs,
                        model1.add_test,model2.backbone_convs):


        #myconv_rep
        backbone_conv.conv.weight,backbone_conv.conv.bias = backbone_conv.conv.rep_params()
        # backbone_conv.conv.weight, backbone_conv.conv.bias = backbone_conv.conv.conv3x3.weight,backbone_conv.conv.conv3x3.bias
        w4, b4 = backbone_conv.sk.weight.data, backbone_conv.sk.bias.data
        device = backbone_conv.expand.weight.get_device()
        if device < 0:
            device = None
        w = F.conv2d(backbone_conv.expand.weight.flip(2, 3).permute(1, 0, 2, 3), backbone_conv.conv.weight, padding=2,
                     stride=1).flip(2, 3).permute(1, 0, 2, 3)
        b = ((backbone_conv.conv.weight * backbone_conv.expand.bias.reshape(1, -1, 1, 1)).sum((1, 2, 3))
             + backbone_conv.conv.bias)
        w = F.conv2d(w.flip(2, 3).permute(1, 0, 2, 3), backbone_conv.reduce.weight, padding=0,
                     stride=1).flip(2, 3).permute(1, 0,2,3)
        b = (backbone_conv.reduce.weight * b.reshape(1, -1, 1, 1)).sum((1, 2, 3)) + backbone_conv.reduce.bias
        s = backbone_conv.scale_attn.scale_attn(backbone_conv.scale_attn.tensor)
        w4 = w4*(s.permute(1,0,2,3))
        b4 = b4*(s.view(-1))
        k4_w = F.pad(w4, [1, 1, 1, 1])
        weight_r = w + k4_w
        bias_r = b + b4

        # weight_r, bias_r = backbone_conv.eval_conv.weight.data, backbone_conv.eval_conv.bias.data
        # weight_r, bias_r = backbone_conv.weight.data, backbone_conv.bias.data
        weight_y,bias_y = add_convs.weight.data,add_convs.bias.data
        weight_b1,bias_b1 = add_expand_conv.weight.data,add_expand_conv.bias.data
        weight_b2, bias_b2 = add_residual_conv.weight.data, add_residual_conv.bias.data
        weight_g,bias_g = residual_conv.weight.data,residual_conv.bias.data
        weight_t,bias_t = add_test.weight.data,add_test.bias.data

        weight_1 = F.conv2d(input=weight_b1.permute(1,0,2,3),weight=weight_y).permute(1, 0, 2, 3)
        bias_1 = F.conv2d(input=bias_b1.view(1,-1,1,1), weight=weight_y, bias=bias_y).view(-1)
        # bias_1 = (weight_y * bias_b1.reshape(1,-1,1,1).sum(1,2,3)) + bias_y
        weight_2 = F.conv2d(input=weight_b2.permute(1,0,2,3),weight=weight_y).permute(1, 0, 2, 3)
        bias_2 = F.conv2d(input=bias_b2.view(1,-1,1,1), weight=weight_y, bias=bias_y).view(-1)

        weight_rep = weight_r + weight_1
        bias_rep = bias_r + bias_1

        weight = torch.cat([weight_rep, weight_2], dim=1)
        bias = bias_rep + bias_2

        weight_ = torch.cat([torch.zeros(weight_g.shape[0], weight.shape[1] - weight_g.shape[1], 3, 3), weight_g], dim=1)
        bias_ = bias_g

        layer2.weight.data = torch.cat([weight, weight_])
        layer2.bias.data = torch.cat([bias, bias_])

    # for conv_last and conv_clip
    if hasattr(model1.conv_last, 'rep_params'):
        weight, bias = model1.conv_last.rep_params()
    else:
        weight, bias = model1.conv_last.weight.data, model1.conv_last.bias.data
    model2.conv_last.weight.data[:, :, :, :] = 0
    model2.conv_last.bias.data[:] = 1
    model2.conv_last.weight.data[:weight.shape[0], :, :, :] = -torch.cat([weight, model1.conv_residual_last.weight.data], dim=1)
    model2.conv_last.bias.data[:weight.shape[0]] = 1 - (bias + model1.conv_residual_last.bias.data)
    model2.conv_clip.weight.data[:, :, :, :] = 0
    model2.conv_clip.bias.data[:] = 1
    for i in range(model2.conv_clip.weight.shape[0]):
        model2.conv_clip.weight.data[i, i , 0, 0] = -1
        model2.conv_clip.bias.data[i] = 1

    # Verify that the network output before and after the transformation is the same (due to floating point numbers, etc., it cannot be exactly the same).
    assert model_allclose(model1, model2, info['num_in_ch'])

    return model2.state_dict()




def get_reparameterization_state(state: OrderedDict):
    ''' get reparameterization state '''
    info = get_info(state)

    # 打印 state_dict 的键
    print("Keys in state_dict:", state.keys())

    # 打印模型的 state_dict 的键
    model1 = EdgeSR(**info)  # 初始化模型
    print("Keys in EdgeSR model:", model1.state_dict().keys())

    model1.load_state_dict(state)
    model2 = EdgeSRForInference(**info)

    # reparameterization for backbones
    for layer1, layer2 in zip(model1.backbone, model2.backbone):
        weight, bias = layer1.rep_params()
        layer2[0].weight.data = weight
        layer2[0].bias.data = bias
        if hasattr(layer1.act, 'weight'):
            layer2[1].weight.data = layer1.act.weight.data

    assert model_allclose(model1, model2, info['num_in_ch'])

    return model2.state_dict()


def main():
    ''' main '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default=r"C:\Users\acer\Desktop\v4r\super resolution\SRFormer-main\net_g_best1.pth", help='ECBSR model state path.')
    parser.add_argument('--output', type=str, default=r"C:\Users\acer\Desktop\v4r\super resolution\SRFormer-main\new_experiments\EdgeSR\pretrained_repmodel\EdgeSR_M6C32_x2.pth", help='ECBSR for inference output path.')
    args = parser.parse_args()

    if os.path.split(args.output)[0] != '':
        os.makedirs(os.path.split(args.output)[0], exist_ok=True)

    state = torch.load(args.input)
    state = state['params']
    converted_state = convert(state)
    torch.save(converted_state, args.output)
    # reparameterization_state = get_reparameterization_state(state)
    # torch.save(reparameterization_state, args.output)


if __name__ == '__main__':
    main()
