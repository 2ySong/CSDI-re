# csdi论文复现仓库
## 项目概述
本仓库旨在复现csdi论文中提出的时序预测方法。与timegrad等方法不同，本项目专注于直接生成未来一整段时序值。该方法在时序预测领域具有广泛的应用前景，尤其适用于需要预测连续时间序列的场景。

## 主要功能
时序预测模型：实现csdi论文中描述的时序预测模型，包括模型构建、训练和预测过程。

数据处理：提供对时序数据进行预处理和后处理的工具，确保数据格式与模型输入要求相匹配。

模型评估：包括常用的时序预测评估指标，如均方误差（MSE）、均方根误差（RMSE）等，用于评估模型性能。

结果可视化：提供可视化工具，展示预测结果与实际数据的对比，便于直观分析模型性能。

## 使用说明
1. 环境准备

- 确保你的环境中安装了以下依赖库：

- numpy：用于数值计算。

- pandas：用于数据处理。

- matplotlib：用于结果可视化。

- tensorflow 或 pytorch（根据项目实现选择）：用于深度学习模型构建和训练。

- 可以使用pip命令安装这些库：

    ```python
    pip install numpy pandas matplotlib tensorflow
    ```
    或者
    ```python
    pip install numpy pandas matplotlib pytorch
    ```
2. 数据准备
将你的时序数据存储在CSV或其他支持的格式中，并确保数据满足模型输入要求。具体的数据格式和处理步骤请参考data_processing.py文件中的说明。

3. 模型训练
运行train.py脚本进行模型训练。你可以通过修改脚本中的参数来调整模型结构、训练轮次、学习率等超参数。
    ```python
    python train.py --epochs 100 --learning_rate 0.001
    ```

4. 模型预测
训练完成后，运行predict.py脚本进行模型预测。你需要提供测试数据集作为输入，并指定训练好的模型权重文件路径。

    ```python
    python predict.py --test_data test_data.csv --model_weights model_weights.h5
    ```

5. 结果评估与可视化
预测完成后，predict.py脚本将输出评估指标和预测结果。同时，你可以使用visualize.py脚本将预测结果与实际数据进行可视化对比。

    ```python
    python visualize.py --predictions predictions.csv --actual_data actual_data.csv
    ```

## 注意事项

在进行模型训练和预测时，请确保数据集的路径和格式正确无误。

根据你的数据集和任务需求，可能需要调整模型结构和超参数。

在使用可视化工具时，请确保安装了相应的图形库（如matplotlib）。

贡献与反馈
如果你在使用本仓库过程中发现任何问题或有任何建议，欢迎通过GitHub的issue功能进行反馈。我们也欢迎你对项目进行贡献，包括但不限于代码修复、功能增强和文档完善等。