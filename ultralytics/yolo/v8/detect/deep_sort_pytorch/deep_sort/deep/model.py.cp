import timm

model_name = 'vit_base_patch16_224'
vit_model = timm.create_model(model_name, pretrained=True)
