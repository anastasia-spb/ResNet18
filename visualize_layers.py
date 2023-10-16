from typing import Type, Union, Optional, List, Any
import wandb

from PIL import Image
from main import preprocess_input
from ResNet18 import resnet18


def log_intermediate_representations(batch):
    wandb.login()
    run = wandb.init(project="ResNet18 Demo", mode="online")

    model = resnet18()
    model.eval()


    def store_intermediate_states(module, output, tag: str = ''):
        layer_images = []
        for o in output[0][0]:
            layer_img = wandb.Image(Image.fromarray((o.detach().cpu().numpy()*255).astype('uint8')))
            layer_images.append([layer_img])
        table = wandb.Table(columns=["Intermediate States"], data=layer_images)
        run.log({'ResNet18_'+tag: table})

    pre_layer1 = model._modules.get('layer1')
    _ = pre_layer1.register_forward_pre_hook(lambda module, output: store_intermediate_states(module, output, tag='layer1'))

    pre_layer2 = model._modules.get('layer2')
    _ = pre_layer2.register_forward_pre_hook(lambda module, output: store_intermediate_states(module, output, tag='layer2'))

    pre_layer3 = model._modules.get('layer3')
    _ = pre_layer3.register_forward_pre_hook(lambda module, output: store_intermediate_states(module, output, tag='layer3'))

    pre_layer4 = model._modules.get('layer4')
    _ = pre_layer4.register_forward_pre_hook(lambda module, output: store_intermediate_states(module, output, tag='layer4'))
    # after forward
    _ = pre_layer4.register_forward_pre_hook(lambda module, output: store_intermediate_states(module, output, tag='layer4_post'))

    _ = model(batch)

    wandb.finish()



if __name__ == "__main__":
    _, batch = preprocess_input(img_path = './dog.jpg')
    batch = batch[None, :, :, :]
    log_intermediate_representations(batch)
