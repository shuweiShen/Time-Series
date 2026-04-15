from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single

warnings.filterwarnings('ignore')


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):  # 需要进行修改，增加重叠大小参数，与data_provider同时进行修改
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def criterion(self):
        def composite_loss(x1, x2, x1_pred, x2_pred, y1_true, y2_true):

            base_loss1 = nn.MSELoss(reduction='none')(x1_pred, y1_true).mean(dim=[1, 2])  # (B,)
            base_loss2 = nn.MSELoss(reduction='none')(x2_pred, y2_true).mean(dim=[1, 2])  # (B,)
            base_loss = base_loss1 + base_loss2  # 每个实例的基础损失 (B,)



            X_union = torch.cat([x1, x2], dim=1)  # (B, T1+T2, D)，按时间维度拼接

            sigma = torch.var(X_union, dim=[1, 2], unbiased=False) + 1e-8  # 每个实例的方差 (B,)
            mu = torch.mean(X_union, dim=[1, 2]) + 1e-8  # 每个实例的均值 (B,)
            c_i = sigma / mu


            omega_alpha = getattr(self.args, 'omega_alpha', 1.0)  # 论文超参数α，默认1.0
            omega_beta = getattr(self.args, 'omega_beta', 0.0)  # 论文超参数β，默认0.0
            omega_i = torch.sigmoid(omega_alpha * c_i + omega_beta)  # 论文公式8：每个实例的权重 (B,)


            shift = self.args.shift
            pred_len = self.args.pred_len
            overlap_len = pred_len - shift
            if overlap_len <= 0:
                raise ValueError(f"偏移量{shift}需小于预测长度{pred_len}！请调整args.shift")

            x1_overlap_pred = x1_pred[:, shift:, :]
            x2_overlap_pred = x2_pred[:, :overlap_len, :]
            y_overlap_true = y1_true[:, shift:, :]


            x1_overlap_loss = nn.MSELoss(reduction='none')(x1_overlap_pred, y_overlap_true).mean(dim=[1, 2])  # (B,)
            x2_overlap_loss = nn.MSELoss(reduction='none')(x2_overlap_pred, y_overlap_true).mean(dim=[1, 2])  # (B,)
            margin = getattr(self.args, 'margin', 0.01)
            # 每个实例的精度约束（非负）
            short_term_constraint = torch.max(
                torch.tensor(0.0, device=self.device),
                x2_overlap_loss - x1_overlap_loss + margin
            )  # (B,)


            standard_consistency = nn.MSELoss(reduction='none')(x1_overlap_pred, x2_overlap_pred).mean(
                dim=[1, 2])  # (B,)

            x1_error = torch.abs(x1_overlap_pred - y_overlap_true).mean(dim=[1, 2])  # 每个实例的误差 (B,)
            x2_error = torch.abs(x2_overlap_pred - y_overlap_true).mean(dim=[1, 2])  # 每个实例的误差 (B,)
            error_ratio = x2_error / (x1_error + 1e-8)  # 误差比 (B,)
            error_ratio_weight = torch.sigmoid(1.0 - error_ratio)  # 误差权重 (B,)
            # 最终一致性损失（每个实例级）
            consistency_constraint = standard_consistency * error_ratio_weight  # 去掉原来的complexity_weight，完全对齐论文

            per_instance_loss = base_loss + omega_i * short_term_constraint + (1 - omega_i) * consistency_constraint
            total_loss = per_instance_loss.mean()  # 对batch求平均，对应论文的1/B求和



            return total_loss

        return composite_loss

    def test_vali(self, test_data, test_loader, criterion):

        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # 接收4元素：batch_x, batch_y, batch_x_mark, batch_y_mark
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input（与原逻辑一致）
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # 模型预测
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # 提取有效预测结果
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 计算基础损失（如MSE，无需复合约束）
                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            # 输入改为样本对：(batch_x1, batch_x2, batch_y1, batch_y2, ...)
            for i, (batch_x1, batch_x2, batch_y1, batch_y2, batch_x1_mark, batch_x2_mark, batch_y1_mark,
                    batch_y2_mark) in enumerate(vali_loader):
                # 数据转设备
                batch_x1 = batch_x1.float().to(self.device)
                batch_x2 = batch_x2.float().to(self.device)
                batch_y1 = batch_y1.float().to(self.device)
                batch_y2 = batch_y2.float().to(self.device)
                batch_x1_mark = batch_x1_mark.float().to(self.device)
                batch_x2_mark = batch_x2_mark.float().to(self.device)
                batch_y1_mark = batch_y1_mark.float().to(self.device)
                batch_y2_mark = batch_y2_mark.float().to(self.device)

                # 生成decoder输入
                dec_inp1 = torch.zeros_like(batch_y1[:, -self.args.pred_len:, :]).float()
                dec_inp1 = torch.cat([batch_y1[:, :self.args.label_len, :], dec_inp1], dim=1).float().to(self.device)
                dec_inp2 = torch.zeros_like(batch_y2[:, -self.args.pred_len:, :]).float()
                dec_inp2 = torch.cat([batch_y2[:, :self.args.label_len, :], dec_inp2], dim=1).float().to(self.device)

                # 模型预测
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs1 = self.model(batch_x1, batch_x1_mark, dec_inp1, batch_y1_mark)
                        outputs2 = self.model(batch_x2, batch_x2_mark, dec_inp2, batch_y2_mark)
                else:
                    outputs1 = self.model(batch_x1, batch_x1_mark, dec_inp1, batch_y1_mark)
                    outputs2 = self.model(batch_x2, batch_x2_mark, dec_inp2, batch_y2_mark)

                # 提取有效预测结果
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs1 = outputs1[:, -self.args.pred_len:, f_dim:]
                outputs2 = outputs2[:, -self.args.pred_len:, f_dim:]
                batch_y1 = batch_y1[:, -self.args.pred_len:, f_dim:].to(self.device)
                batch_y2 = batch_y2[:, -self.args.pred_len:, f_dim:].to(self.device)

                # 计算复合损失
                loss = criterion(outputs1, outputs2, batch_y1, batch_y2)
                total_loss.append(loss.item())  # 注意：原代码用np.average，需将loss转为item()

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self.criterion()
        base_criterion = self._select_criterion()  # 基础MSE
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x1, batch_x2, batch_y1, batch_y2, batch_x1_mark, batch_x2_mark, batch_y1_mark, batch_y2_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                # 数据转设备
                batch_x1 = batch_x1.float().to(self.device)
                batch_x2 = batch_x2.float().to(self.device)
                batch_y1 = batch_y1.float().to(self.device)
                batch_y2 = batch_y2.float().to(self.device)
                batch_x1_mark = batch_x1_mark.float().to(self.device)
                batch_x2_mark = batch_x2_mark.float().to(self.device)
                batch_y1_mark = batch_y1_mark.float().to(self.device)
                batch_y2_mark = batch_y2_mark.float().to(self.device)

                # --------------------------
                # 生成两个样本的decoder输入（与原逻辑一致）
                # --------------------------
                # x1的decoder输入
                dec_inp1 = torch.zeros_like(batch_y1[:, -self.args.pred_len:, :]).float()
                dec_inp1 = torch.cat([batch_y1[:, :self.args.label_len, :], dec_inp1], dim=1).float().to(self.device)
                # x2的decoder输入
                dec_inp2 = torch.zeros_like(batch_y2[:, -self.args.pred_len:, :]).float()
                dec_inp2 = torch.cat([batch_y2[:, :self.args.label_len, :], dec_inp2], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        # x1预测
                        outputs1 = self.model(batch_x1, batch_x1_mark, dec_inp1, batch_y1_mark)
                        # x2预测
                        outputs2 = self.model(batch_x2, batch_x2_mark, dec_inp2, batch_y2_mark)
                else:
                    outputs1 = self.model(batch_x1, batch_x1_mark, dec_inp1, batch_y1_mark)
                    outputs2 = self.model(batch_x2, batch_x2_mark, dec_inp2, batch_y2_mark)

                # --------------------------
                # 提取有效预测结果（与原逻辑一致，取最后pred_len步）
                # --------------------------
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs1 = outputs1[:, -self.args.pred_len:, f_dim:]  # x1的预测结果
                outputs2 = outputs2[:, -self.args.pred_len:, f_dim:]  # x2的预测结果
                batch_y1 = batch_y1[:, -self.args.pred_len:, f_dim:].to(self.device)  # x1的真实标签
                batch_y2 = batch_y2[:, -self.args.pred_len:, f_dim:].to(self.device)  # x2的真实标签

                # --------------------------
                # 计算复合损失
                # --------------------------
                loss = criterion(outputs1, outputs2, batch_y1, batch_y2)
                train_loss.append(loss.item())

                # --------------------------
                # 反向传播与优化（与原逻辑一致）
                # --------------------------
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.test_vali(test_data, test_loader, base_criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
