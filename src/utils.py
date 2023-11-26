import shutil
import os
def print_model_info(model):
    # print('Architecture: ', model)
    # print('parameters count: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # print('param_dict: ', param_dict)

    for pn, p in param_dict.items():
        # if p.dim() < 2:
        print(pn, p.dim())


def delete_files_in_directory(directory_path):
    try:
     os.mkdir(directory_path)
    except OSError as error:
     print("Error occurred while mkdir path.", error)

    try:
     files = os.listdir(directory_path)
     for file in files:
        file_path = os.path.join(directory_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            shutil.rmtree(file_path)
     print(f"{directory_path} All files deleted successfully.")
    except OSError as error:
     print("Error occurred while deleting files.", error)