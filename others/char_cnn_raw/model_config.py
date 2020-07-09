# config for model

class ModelConfig(object):
    numFilters = 256
    filterSizes = [4, 4, 4]
    # 该列表中子列表的三个元素分别是卷积核的数量，卷积核的高度，池化的尺寸
    convLayers = [[256, 7, 4],
                  [256, 7, 4],
                  [256, 3, 4]]
    #                   [256, 3, None],
    #                   [256, 3, None],
    #                   [256, 3, 3]]
    fcLayers = [512]
    dropoutKeepProb = 0.5

    epsilon = 1e-3  # BN层中防止分母为0而加入的极小值
    decay = 0.999  # BN层中用来计算滑动平均的值
    l2RegLambda = 0.0
