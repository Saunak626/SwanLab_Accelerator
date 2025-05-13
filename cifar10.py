import torch
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
from torchvision.models import resnet18, ResNet18_Weights

#=====accelerate/swanlab======
from accelerate import Accelerator
from accelerate.logging import get_logger
import swanlab
from swanlab.integration.accelerate import SwanLabTracker
#=====================

def main():
    # hyperparameters
    config = {
        "num_epoch": 5,
        "batch_num": 16,
        "learning_rate": 1e-3,
    }

    # Download the raw CIFAR-10 data.
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    BATCH_SIZE = config["batch_num"]
    my_training_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    my_testing_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    # Using resnet18 model, make simple changes to fit the data set
    my_model = resnet18(weights=ResNet18_Weights.DEFAULT)
    my_model.conv1 = torch.nn.Conv2d(my_model.conv1.in_channels, my_model.conv1.out_channels, 3, 1, 1)
    my_model.maxpool = torch.nn.Identity()
    my_model.fc = torch.nn.Linear(my_model.fc.in_features, 10)

    # Criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    my_optimizer = torch.optim.SGD(my_model.parameters(), lr=config["learning_rate"], momentum=0.9)

#=====accelerate/swanlab======
    # 创建SwanLab日志记录器
    tracker = SwanLabTracker("CIFAR10_TRAING")
    # 传入Accelerator
    accelerator = Accelerator(log_with=tracker)
    # 初始化所有日志记录器
    accelerator.init_trackers("CIFAR10_TRAING", config=config)
    my_model, my_optimizer, my_training_dataloader, my_testing_dataloader = accelerator.prepare(
        my_model, my_optimizer, my_training_dataloader, my_testing_dataloader
    )
    device = accelerator.device
    my_model.to(device)

    # Get logger
    logger = get_logger(__name__)

    print(f"Number of processes: {accelerator.num_processes}")
    # 打印每个进程使用的设备信息
    print(f"Process {accelerator.process_index} is using device: {accelerator.device} (Name: {torch.cuda.get_device_name(accelerator.device)})")
#=====================

    # Begin training

    for epoch in range(config["num_epoch"]):
        # train model
        if accelerator.is_local_main_process:
            print(f"begin epoch {epoch} training...")
        step = 0
        for stp, data in enumerate(my_training_dataloader):
            my_optimizer.zero_grad()
            inputs, targets = data
            outputs = my_model(inputs)
            loss = criterion(outputs, targets)
            accelerator.backward(loss)
            my_optimizer.step()
            accelerator.log({"training_loss": loss, "epoch_num": epoch})
            
            # 检测主进程，每100个batch打印一次
            if accelerator.is_local_main_process and stp % 100 == 0:
                print(f"train epoch {epoch} [{stp}/{len(my_training_dataloader)}] | train loss {loss}")

        # eval model
        if accelerator.is_local_main_process:
            print(f"begin epoch {epoch} evaluating...")
        with torch.no_grad():
            total_acc_num = 0
            for stp, (inputs, targets) in enumerate(my_testing_dataloader):
                predictions = my_model(inputs)
                predictions = torch.argmax(predictions, dim=-1)
                # Gather all predictions and targets
                all_predictions, all_targets = accelerator.gather_for_metrics((predictions, targets))
                acc_num = (all_predictions.long() == all_targets.long()).sum()
                total_acc_num += acc_num
                if accelerator.is_local_main_process:
                    print(f"eval epoch {epoch} [{stp}/{len(my_testing_dataloader)}] | val acc {acc_num/len(all_targets)}")

            accelerator.log({"eval acc": total_acc_num / len(my_testing_dataloader.dataset)})

    accelerator.wait_for_everyone()
    accelerator.save_model(my_model, "cifar_cls.pth")

    accelerator.end_training()


if __name__ == "__main__":
    main()