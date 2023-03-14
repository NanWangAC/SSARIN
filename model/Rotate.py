import os
import numpy as np
import torch

def reverse(arr, left, right):
    while left < right:
        tmp = arr[:, :, right].clone()
        arr[:, :, right] = arr[:, :, left]
        arr[:, :, left] = tmp.clone()
        left += 1
        right -= 1
    return arr


def rightShif(arr, arr_len, stride):
    stride %= arr_len
    arr = reverse(arr, 0, arr_len - stride - 1)
    arr = reverse(arr, arr_len - stride, arr_len - 1)
    arr = reverse(arr, 0, arr_len - 1)
    return arr


def circleRotate(mat, row_begin, col_begin, row_end, col_end, stride):
    mat_rst = mat.clone()

    top = mat[:, :, row_begin, col_begin:col_end]
    right = mat[:, :, row_begin:row_end, col_end]
    bottom = mat[:, :, row_end, col_begin + 1:col_end + 1]
    bottom = torch.flip(bottom, [2])
    left = mat[:, :, row_begin + 1:row_end + 1, col_begin]
    left = torch.flip(left, [2])

    arr = torch.cat((top, right), axis=2)
    arr = torch.cat((arr, bottom), axis=2)
    arr = torch.cat((arr, left), axis=2)
    arr = rightShif(arr, arr[0, 0, :].shape[0], stride)

    top_rst = arr[:, :, 0 * (col_end - col_begin):1 * (col_end - col_begin)]
    right_rst = arr[:, :, 1 * (col_end - col_begin):2 * (col_end - col_begin)]
    bottom_rst = arr[:, :, 2 * (col_end - col_begin):3 * (col_end - col_begin)]
    bottom_rst = torch.flip(bottom_rst, [2])
    left_rst = arr[:, :, 3 * (col_end - col_begin):4 * (col_end - col_begin)]
    left_rst = torch.flip(left_rst, [2])

    mat_rst[:, :, row_begin, col_begin:col_end] = top_rst
    mat_rst[:, :, row_begin:row_end, col_end] = right_rst
    mat_rst[:, :, row_end, col_begin + 1:col_end + 1] = bottom_rst
    mat_rst[:, :, row_begin + 1:row_end + 1, col_begin] = left_rst
    return mat_rst


def rotate(mat, stride):
    _, _, w, h = mat.shape
    row_begin = 0
    col_begin = 0
    col_end = w - 1
    row_end = h - 1

    mat_rst = mat.clone()
    while row_begin < row_end:
        mat_rst = circleRotate(mat_rst, row_begin, col_begin, col_end, row_end, stride * ((row_end - row_begin) // 2))
        row_begin = row_begin + 1
        col_begin = col_begin + 1
        col_end = col_end - 1
        row_end = row_end - 1
    return mat_rst