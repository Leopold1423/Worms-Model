import torch
from tools.common_tools import *
from models.vgg import VGG
base_dir = os.path.dirname(os.path.abspath(__file__))

#device
device_count = torch.cuda.device_count()
print("\ndevice_count: {}".format(device_count))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

#model
from models.resnet import resnet56
from models.ghost_net import GhostModule
model = VGG("VGG16")
model = replace_conv(model, GhostModule, arc='vgg16',
                             pretrain=False, cheap_pretrian=False, point_conv=False)
model.eval()

#load
vgg_checkpoint_path = os.path.join(base_dir,"..", "results", "gvgg16-1","checkpoint_best.pkl")
check_p = torch.load(vgg_checkpoint_path, map_location="cpu")
pretrain_dict = check_p["model_state_dict"]
print(check_p['epoch'],check_p['best_acc'])
state_dict_cpu = state_dict_to_cpu(pretrain_dict)
model.load_state_dict(state_dict_cpu)
print("load state dict from :{} done~~".format(vgg_checkpoint_path))

# train
if False:
    model = model.to(device)
    from torch.utils.data import DataLoader
    from tools.cifar10_dataset import CifarDataset
    from tools.model_trainer import ModelTrainer
    from config.config import cfg
    test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "cifar10_test")
    valid_data = CifarDataset(data_dir=test_dir, transform=cfg.transforms_valid)
    valid_loader = DataLoader(dataset=valid_data, batch_size=128, num_workers=cfg.workers)
    #loss
    loss_f = nn.CrossEntropyLoss().to(device)
    #train
    loss_valid, acc_valid, mat_valid = ModelTrainer.valid(valid_loader, model, loss_f, device)
    print("Valid Acc:{:.2%} Valid loss:{:.4f} ".format(acc_valid, loss_valid))
    print(mat_valid)

# save
else:
    save_path = os.path.join(base_dir,"..", "pt_file","vgg.pt")
    input_tensor = torch.rand(1,3,32,32)
    script_model = torch.jit.trace(model,input_tensor)
    script_model.save(save_path)
    print("saved")




