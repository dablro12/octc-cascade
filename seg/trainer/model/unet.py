import segmentation_models_pytorch as smp
def U_Net(width = 224, height = 224):
    model = smp.Unet(
        encoder_name= "resnet34",
        encoder_weights= "imagenet",
        encoder_depth= 5,
        decoder_channels= (512, 256, 128, 64, 32),
        in_channels = 1,
        activation= None,
        classes = 1
        )
    return model
