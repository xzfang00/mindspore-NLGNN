import mindspore
import utils
import mindspore.nn as nn
from mindspore import Tensor
import numpy as np
import mindspore.context as context
from model import NLGNN
from mindspore.nn.optim import Adam

if __name__ == '__main__':
    print('-------------------------------------------------------' * 3)
    dataset_name = 'texas'
    heter_dataset = ['chameleon', 'cornell', 'actor', 'squirrel', 'texas', 'wisconsin']  # 异配图
    homo_dataset = ['cora', 'citeseer', 'pubmed']  # 同配图
    print(f"此时数据集是{dataset_name}")
    lr = 0.01
    weight_decay = 5e-3
    max_epoch = 500
    patience = 50  # 如果超过这么多轮次没有提升，直接停
    num_hidden = [92, 48, 16, 8]
    le_list = ['mlp', 'gcn', 'gat']
    le = le_list[0]
    print(f"此时模型是{le}")
    device = 'GPU'  # 'CPU'
    context.set_context(mode=context.GRAPH_MODE, device_target=device)
    print("当前使用的设备是:", context.get_context("device_target"))
    window_size = 5

    split_by_label_flag = True  #划分数据集是否按照标签划分
    if dataset_name in ['chameleon', 'cornell', 'texas']:
        split_by_label_flag = False

    if dataset_name in heter_dataset:
        dataset, num_features, num_classes, edge_index = utils.load_heter_data(dataset_name)
    elif dataset_name in homo_dataset:
        dataset, num_features, num_classes, edge_index = utils.load_homo_data(dataset_name)
    else:
        print("我们现在没有{} dataset.".format(dataset_name))
    # 输出当前dataset的信息
    utils.graph_info(dataset, None)
    # 设置全局随机种子
    mindspore.set_seed(123)
    train_index, val_index, test_index = utils.split_data(dataset, 0.5, 0.3, 0.2, split_by_label_flag)
    utils.graph_info(dataset, train_index)
    # 得到dataset的x和label，x.shape为(num_node，num_feature)
    x, label = utils.getxANDlable(dataset)
    x = x.astype(np.float32)
    x = Tensor(x)
    # 得到该dataset，对应的GraphField对象
    graphFiled, in_degree, out_degree = utils.getGraphFiled(x, edge_index=edge_index)  # x 是float32
    net = NLGNN(le, window_size, num_features, num_hidden, num_classes, 0.5, graphFiled.get_graph(), in_degree, out_degree, x)
    optimizer = Adam(net.trainable_params(), learning_rate=lr, weight_decay=weight_decay)
    # 权重衰减(Weight Decay)是一种正则化技术，用于防止神经网络过拟合。
    # 它具体来说，权重衰减会使得模型的参数更新变得更加缓慢，从而减少了模型对训练数据的过度拟合。
    # 系数越大，权重衰减的效果就越明显
    los = []  # 用来存储测试集准确率
    counter = 0
    min_loss = 100.0
    max_acc = 0.0

    train_index_t = Tensor(train_index)
    val_index_t = Tensor(val_index)
    test_index_t = Tensor(test_index)
    loss_fn = nn.CrossEntropyLoss()
    label = Tensor(label)

    def forward_fn(index, label):
        result = net()
        loss = loss_fn(result[index], label[index])
        return loss, result
    for epoch in range(max_epoch):

        net.set_train()
        grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
        (loss, result), grads = grad_fn(train_index_t, label)
        optimizer(grads)

        # 训练集正确率与损失
        loss_train = loss
        train_correct = 0
        train_correct += (result[train_index_t].argmax(1) == label[train_index_t]).asnumpy().sum()
        correct_train = train_correct / len(train_index)
        #  验证集准确率与损失
        loss_val = loss_fn(result[val_index_t], label[val_index_t])
        val_correct = 0
        val_correct += (result[val_index_t].argmax(1) == label[val_index_t]).asnumpy().sum()
        correct_val = val_correct / len(val_index)
        # 测试准确率
        test_correct = 0
        test_correct += (result[test_index_t].argmax(1) == label[test_index_t]).asnumpy().sum()
        correct_test = test_correct / len(test_index)
        los.append(correct_test)

        if epoch % 10 == 0:  # 每10轮一输出
            print(f"轮次{epoch} 训练集损失{loss_train} 训练集正确率{correct_train*100:.2f}% 验证集损失{loss_val} 验证集准确率{correct_val*100:.2f}%")
        # 判断是否长时间没更新
        if loss_val < min_loss and max_acc < correct_val:
            min_loss = loss_val
            max_acc = correct_val
            counter = 0  # 计数器
        if counter > patience:
            print("由于模型长时间没有更新，所以终止")
            break
        counter += 1

    print(f"测试效果最好的一次准确率{max(los)*100:.2f}%")

