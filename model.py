import segmentation_models as sm

def network(CLASSES, BACKBONE, arch):
    # define network parameters
    n_classes = 1 if len(CLASSES) == 1 else (len(CLASSES) + 1)  # case for binary and multiclass segmentation
    activation = 'sigmoid' if n_classes == 1 else 'softmax'

    #create model
    if arch == 'FPN':
    	model = sm.FPN(BACKBONE, classes=n_classes, activation=activation)
    if arch == 'Unet':
    	model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
    return model, n_classes