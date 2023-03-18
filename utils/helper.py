import torch
import yaml
import os


def save_dict_to_yaml(dict_value, save_path):
    with open(save_path, 'w', encoding='utf-8') as file:
        yaml.dump(dict_value, file, sort_keys=False)


def save_checkpoint(model, cfg, log_path, epoch_id):
    save_dict_to_yaml(cfg, os.path.join(log_path, 'config.yaml'))
    torch.save(model.module.state_dict(), os.path.join(log_path, f'ckpt_epoch_{epoch_id}.pt'))


def format_print_dict(print_dict):
    print_str = ''
    len_print_dict = len(print_dict.keys())
    for i, k in enumerate(print_dict.keys()):
        print_str += f'{k}: '
        if isinstance(print_dict[k], int):
            if i != len_print_dict - 1:
                print_str += f'{print_dict[k]}, '
            else:
                print_str += f'{print_dict[k]}'
        elif isinstance(print_dict[k], float):
            if i != len_print_dict - 1:
                print_str += f'{print_dict[k] :.5f}, '
            else:
                print_str += f'{print_dict[k] :.5f}'
        elif isinstance(print_dict[k], str):
            if i != len_print_dict - 1:
                print_str += f'{print_dict[k]}, '
            else:
                print_str += f'{print_dict[k]}'
        elif isinstance(print_dict[k], list):
            print_str += '['
            v_list = print_dict[k]
            lth = len(v_list)
            for i in range(lth):
                if i == lth - 1:
                    print_str += f'{print_dict[k][i] :.5f} '
                    print_str += ']'
                else:
                    print_str += f'{print_dict[k][i] :.5f}, '

    return print_str
