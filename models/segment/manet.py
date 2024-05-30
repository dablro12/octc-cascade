import segmentation_models_pytorch as smp 
def manet(width = 224, height =224):
    model = smp.MAnet(
        encoder_name= "resnet34",
        encoder_weights= "imagenet",
        encoder_depth= 5,
        decoder_channels= (512, 256, 128, 64, 32),
        in_channels = 1,
        classes = 1
        )
    return model
