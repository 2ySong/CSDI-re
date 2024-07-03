import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import json
from accelerate import Accelerator


# 训练部分
def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=20,  # 验证间隔，即每多少个 epoch 进行一次验证。原20
    foldername="",
):
    accelerator = Accelerator()
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()  # 设置模型为训练模式
        """
        tqdm 是一个进度条库
            train_loader 是一个数据加载器。
            mininterval 和 maxinterval 控制进度条更新的最小和最大间隔时间。
        """
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:

            """
            enumerate 用于获取批次索引和对应的数据。
            """
            batchs = {}
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                # loss.backward()
                accelerator.backward(loss)
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                # 如果当前批次数达到配置中指定的最大值，则停止当前轮次的训练。
                if batch_no >= config["itr_per_epoch"]:
                    break
            lr_scheduler.step()  # 在每轮训练结束后，更新学习率调度器以调整学习率。

        #  ((epoch_no + 1) % valid_epoch_interval == 0)：
        # 确保当前的 epoch 编号加 1 后除以验证间隔的余数为 0。这样可以确保每隔 valid_epoch_interval 个 epoch 进行一次验证。
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0  # 在验证过程中禁用梯度计算以节省内存和计算资源
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)

                        avg_loss_valid += loss.item()

                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            # 如果当前的平均验证损失比最佳验证损失更低，更新最佳验证损失并打印更新信息。
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


# 计算给定分位数的分位数损失
def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


# 计算分位数损失的分母，用于标准化
def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


# 计算 CRPS（连续排名概率评分）指标，评估预测分布与真实分布的差异。
def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


# 计算总和上的 CRPS 指标。
def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):
    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1), quantiles[i], dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


"""
评估函数
    model: 要评估的模型。
    test_loader: 测试数据加载器。
    nsample: 每个输入生成的样本数。
    scaler: 目标值缩放因子。
    mean_scaler: 目标值均值缩放因子。
    foldername: 保存评估结果的文件夹路径（可选）。
函数逻辑：
    1. 设置模型为评估模式。
    2. 遍历每个测试批次数据，生成预测结果：
    3. 调用模型的 evaluate 方法生成样本。
    4. 计算 MSE 和 MAE 损失。
    5. 更新进度条。
    6. 将所有生成的样本和真实值保存到 pickle 文件中。
    7. 计算并保存 CRPS 和 CRPS_sum 指标。
    8. 打印评估结果（RMSE、MAE、CRPS、CRPS_sum）。
"""


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                # print(output) 选择输出整个模型，（建议不要）
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler**2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            # 样本保存到这
            with open(
                foldername + "/generated_outputs_" + str(nsample) + "samples.pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )
            CRPS_sum = calc_quantile_CRPS_sum(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            # 将CRPS这种值存入到json文件中
            with open(
                foldername + "/result_" + str(nsample) + "samples.json", "r"
            ) as f:
                result = json.load(f)
            result.update(
                {
                    "RMSE": np.sqrt(mse_total / evalpoints_total),
                    "MAE": mae_total / evalpoints_total,
                    "CRPS": CRPS,
                    "CRPS_sum": CRPS_sum,
                }
            )
            # 将更新后的字典写回JSON文件
            with open(
                foldername + "/result_" + str(nsample) + "samples.json", "w"
            ) as f:
                json.dump(result, f, indent=4)

            print("RMSE:", np.sqrt(mse_total / evalpoints_total))
            print("MAE:", mae_total / evalpoints_total)
            print("CRPS:", CRPS)
            print("CRPS_sum:", CRPS_sum)
