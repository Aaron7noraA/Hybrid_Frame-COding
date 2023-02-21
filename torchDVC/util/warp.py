import torch
Backward_tensorGrid = [{} for i in range(8)]
Backward_tensorGrid_cpu = {}

def torch_warp(tensorInput, tensorFlow):
    if tensorInput.device == torch.device('cpu'):
        if str(tensorFlow.size()) not in Backward_tensorGrid_cpu:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid_cpu[str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cpu()

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid_cpu[str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)
    else:
        device_id = tensorInput.device.index
        if str(tensorFlow.size()) not in Backward_tensorGrid[device_id]:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid[device_id][str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cuda().to(device_id)

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid[device_id][str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)