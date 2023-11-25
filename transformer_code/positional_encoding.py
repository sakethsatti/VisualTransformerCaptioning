import torch

def get_angles(pos, i, d_model):
    angle_rates = 1 / torch.pow(10000,
                                 (2 * (i//2)) / torch.float32(d_model))
    return pos * angle_rates

def positional_encoding_1d(position, d_model):
    pos = torch.arange(position).unsqueeze(1)
    i = torch.arange(d_model).unsqueeze(0)

    angle_rads = get_angles(pos, i, d_model)

    angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads.unsqueeze(0)
    return pos_encoding.float()

def positional_encoding_2d(row, col, d_model):
    assert d_model % 2 == 0

    row_pos = torch.repeat(torch.arange(row), col).unsqueeze(1)
    col_pos = torch.repeat(torch.arange(col).unsqueeze(0),
                            row, dim=0).reshape(-1, 1)

    angle_rads_row = get_angles(row_pos,
                                torch.arange(d_model // 2).unsqueeze(0),
                                d_model // 2)
    angle_rads_col = get_angles(col_pos,
                                torch.arange(d_model // 2).unsqueeze(0),
                                d_model // 2)

    angle_rads_row[:, 0::2] = torch.sin(angle_rads_row[:, 0::2])
    angle_rads_row[:, 1::2] = torch.cos(angle_rads_row[:, 1::2])
    angle_rads_col[:, 0::2] = torch.sin(angle_rads_col[:, 0::2])
    angle_rads_col[:, 1::2] = torch.cos(angle_rads_col[:, 1::2])

    pos_encoding = torch.cat([angle_rads_row, angle_rads_col],
                             dim=1).unsqueeze(0)
    return pos_encoding.float()