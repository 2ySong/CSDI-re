import argparse
import torch
import datetime
import json
import yaml
import os

# from main_model import CSDI_Forecasting
from test_fenghe import CSDI_Forecasting
from dataset_forecasting import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base_forecasting.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
# parser.add_argument('--device', default='cuda:0', help='Device for Attack')#注意服务器上用cuda
parser.add_argument(
    "--device",
    default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    help="Device for Attack",
)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=10)

args = parser.parse_args()
print(args)


# 操作
path = "./config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

if args.datatype == "electricity":
    target_dim = 370

config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./models_save/forecasting_" + args.datatype + "_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)

train_loader, valid_loader, test_loader, scaler, mean_scaler, lengths = get_dataloader(
    datatype=args.datatype,
    device=args.device,
    batch_size=config["train"]["batch_size"],
)
# 将部分配置数据写入最后的JSON文件
with open(foldername + "/result_" + str(args.nsample) + "samples.json", "w") as f:
    json.dump(
        {
            "epochs": config["train"]["epochs"],
            "batch_size": config["train"]["batch_size"],
            "learning rate": 1.0e-3,
            "lengths": lengths,
        },
        f,
        indent=4,
    )
model = CSDI_Forecasting(config, args.device, target_dim).to(args.device)


print(model)

if args.modelfolder == "":
    train(
        model,
        config["train"],
        train_loader,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    model.load_state_dict(
        torch.load("./models_save/" + args.modelfolder + "/model.pth")
    )
model.target_dim = target_dim

# 评估过程，并生成样本
evaluate(
    model,
    test_loader,
    nsample=args.nsample,
    scaler=scaler,
    mean_scaler=mean_scaler,
    foldername=foldername,
)
