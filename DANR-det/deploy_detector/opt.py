class Config:
    # paths
    model_path = "./deploy_detector/model_resneSt50_v1.0.pth"
    num_layers = 50

    # inference modes (will boost performance but sacrifice speed)
    gpus = [3]
    flip_test = False     # support testing on both original image and mirrored image
    test_scales = [1.0]   # support multi-scale testing

    # thresholds
    vis_thresh = 0.2    # threshold of obj confidence, above which to visualize
    censor_thresh = 0.2 # threshold of obj confidence, above which to censor
    K = 40              # max number of objects to decode

    # network specific
    input_h = 512
    input_w = 512
    down_ratio = 4
    output_h = input_h // down_ratio
    output_w = input_w // down_ratio
    input_res = max(input_h, input_w)
    output_res = max(output_h, output_w)

    # dataset specific
    num_classes = 27
    mean = [0.408, 0.447, 0.470]  # bgr, opencv image default, if use torch.transform, need rgb for both image and mean order
    std = [0.289, 0.274, 0.278]
    class_name = [
        '__background__',
        'screwdriver', 'drill (tool)', 'bow and arrow', 'chainsaw', 'scissors', 'dagger', 'hammer', 'knife',
        'missile', 'kitchen knife', 'nail (construction)', 'rifle', 'shotgun', 'sword', 'tripod', 'torch', 'tool',
        'weapon', 'axe', 'bomb', 'chisel', 'snail', 'stool', 'handgun', 'grinder', 'ruler', 'binoculars']
    pos_ids = [3, 6, 9, 12, 13, 14, 18, 20, 24] # remove kitchen knife from weapon set

    # task head
    cat_spec_wh =True if num_layers == 34 else False
    heads = {'hm': num_classes,
             'wh': 2 * num_classes if cat_spec_wh else 2,
             'reg': 2,
             }
    head_conv = 64


opt = Config()

if __name__ == "__main__":
    print(opt.heads)

