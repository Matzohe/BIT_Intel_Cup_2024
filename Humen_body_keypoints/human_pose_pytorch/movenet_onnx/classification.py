import torch.nn as nn
import torch
import torch.nn.functional as F
from Config import ConfigSet, set_seed
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
# from scheduled import ScheduledOptim


class ResidualNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, input_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.activation(self.linear1(self.linear2(x) + x))
        return x


class Conv1dResidual(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.cnn = nn.Conv1d(input_channel, output_channel, kernel_size=3, padding=1)
        self.residual_nn = nn.Conv1d(input_channel, input_channel, kernel_size=3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.gelu(self.cnn(self.gelu(self.residual_nn(x) + x)))
        return x


class Classification(nn.Module):
    def __init__(self, cnn_input, cnn_output, layer_input, layer_output):
        super(Classification, self).__init__()
        self.model = nn.Sequential()
        before_input = [2, 4]
        before_output = [4, 2]
        self.before_process = nn.Sequential()
        for i in range(2):
            self.before_process.add_module("Before{}".format(i), ResidualNet(before_input[i], before_output[i]))
        self.cnn_model = nn.Sequential()
        for i in range(len(cnn_input)):
            self.cnn_model.add_module("Conv1d_{}".format(i), Conv1dResidual(cnn_input[i], cnn_output[i]))

        self.layer_norm = nn.LayerNorm(17, eps=1e-6)

        for i in range(len(layer_input)):
            self.model.add_module("residual{}".format(i), ResidualNet(layer_input[i], layer_output[i]))

    def forward(self, x):

        x = x[:, :, :, :2].reshape(-1, 2, 17)
        x = self.layer_norm(x)
        # output = self.cnn_model(x).view(-1, 17)
        output = x.reshape(-1, 34)
        return F.softmax(self.model(output), dim=-1)
        # return F.softmax(output.view(output.shape[0], -1), dim=-1)


def make_training_data(data_path):
    data_input = torch.load(data_path)
    new_list = []
    for key, values in data_input.items():
        for each in values:
            new_list.append((each, key))
    return new_list


class KeypointDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = make_training_data(data_path)

    def __getitem__(self, item):
        return self.data[item][0], self.data[item][1]

    def __len__(self):
        return len(self.data)


def training_process(config):
    device = config["device"]
    train_data_root = config["train_root"]
    test_data_root = config["test_root"]
    train_dataset = KeypointDataset(train_data_root)
    test_dataset = KeypointDataset(test_data_root)
    training_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    epoch_num = config["epoch_num"]
    model = Classification(config["cnn_input"], config["cnn_output"],
                           config["layer_input"], config["layer_output"]).to(device)
    optimizer = optim.Adam([{"params": model.parameters()}], lr=config['lr'])
    # scheduled_optimizer = ScheduledOptim(optimizer, init_lr=config["lr"], all_steps=10000, start_steps=1000)
    loss_function = nn.CrossEntropyLoss()
    loss_list = []
    acc_list = []
    for i in range(epoch_num):
        all_loss = 0
        acc_num = 0
        all_num = 0
        for x, y in tqdm(training_dataloader, desc="Training epoch{}".format(i), total=len(training_dataloader)):
            # scheduled_optimizer.zero_grad()
            optimizer.zero_grad()
            output = model(x.to(device))
            loss = loss_function(output, torch.tensor(y, device=device))
            all_loss += loss.detach().item()
            loss.backward()
            # scheduled_optimizer.step_and_update_lr()
            optimizer.step()
        for x, y in tqdm(test_dataloader, desc="Test epoch{}".format(i), total=len(test_dataloader)):
            with torch.no_grad():
                output = torch.argmax(model(x.to(device)), dim=-1).view(-1)
                for j in range(len(output)):
                    all_num += 1
                    if output[j] == y[j]:
                        acc_num += 1
        loss_list.append(all_loss)
        acc_list.append(acc_num / all_num)

    plt.plot([k for k in range(len(loss_list))], loss_list, color='g')
    plt.show()
    plt.close()
    plt.plot([k for k in range(len(acc_list))], acc_list, color='g')
    plt.show()
    torch.save(model.state_dict(), r"G:\Module_Parameter\humen_pose\after_movenet\0\model_param.pt")


if __name__ == "__main__":
    set_seed(3456)
    Config = ConfigSet(train_root=r"M:\data package\human_pose_detection\train_keypoint_dict.pt",
                       test_root=r"M:\data package\human_pose_detection\test_keypoint_dict.pt",
                       cnn_input=[2, 4, 8, 16, 8, 4, 2],
                       cnn_output=[4, 8, 16, 8, 4, 2, 1],
                       layer_input=[34, 64, 128, 256, 256, 128, 64, 32, 16],
                       layer_output=[64, 128, 256, 256, 128, 64, 32, 16, 5],
                       epoch_num=93,
                       lr=0.00002,
                       device="cuda" if torch.cuda.is_available() else "cpu",
                       )
    training_process(Config)

