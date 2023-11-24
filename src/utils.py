def print_model_info(model):
    # print('Architecture: ', model)
    # print('parameters count: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    param_dict = {pn: p for pn, p in model.named_parameters()}
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # print('param_dict: ', param_dict)

    for pn, p in param_dict.items():
        # if p.dim() < 2:
        print(pn, p.dim())


