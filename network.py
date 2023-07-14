import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import mean_squared_error
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MyDataset(Dataset):
    def __init__(self, data, targets, seq_length):
        self.data = data
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return self.data.shape[0] - self.seq_length

    def __getitem__(self, index):
        return (self.data[index:index+self.seq_length], self.targets[index+self.seq_length])
    
def get_data_loader(df_list,all_features):
    # 初始化一个列表来存储每个项目的 Dataset 和 DataLoader
    datasets = []
    train_loaders = []
    val_loaders = []
    test_loaders = []

    # 对每个 DataFrame 单独进行处理
    for df in df_list:
        # 删除具有特定索引的行
        df = df.drop('2021-10-raw', errors='ignore')
        for feature in all_features:
            if feature not in df.columns:
                df[feature] = 0
        # 将数据分为特征和目标变量
        X = df.drop('openrank', axis=1).values
        y = df['openrank'].values

        # 根据时间划分数据集
        train_data = df[df.index < '2022-10']
        train_X=train_data.drop('openrank', axis=1).values
        train_y=train_data['openrank'].values
        val_data = df[(df.index >= '2022-04') & (df.index < '2023-01')]
        val_X=val_data.drop('openrank', axis=1).values
        val_y=val_data['openrank'].values
        test_data = df[df.index >= '2022-07']
        test_X=test_data.drop('openrank', axis=1).values
        test_y=test_data['openrank'].values
        # 创建 Dataset
        train_dataset = MyDataset(train_X,train_y, seq_length=6)
        val_dataset = MyDataset(val_X,val_y, seq_length=6)
        test_dataset = MyDataset(test_X,test_y, seq_length=6)

        # 创建 DataLoader
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        train_loaders.append(train_loader)
        val_loaders.append(val_loader)
        test_loaders.append(test_loader)
    return train_loaders,val_loaders,test_loader

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])
        return out

def train_model(model, train_loaders, val_loaders, criterion, optimizer, num_epochs=100,early_stop=3,PATH = './test_state_dict.pth'):
    best_val_mse = float('inf')  # 保存最佳验证集均方误差
    best_model_state_dict = None  # 保存最佳模型参数
    stop_count = 0  # 连续增加的计数器
    for epoch in range(num_epochs):
        for i, (train_loader, val_loader) in enumerate(zip(train_loaders, val_loaders)):
            for data, targets in train_loader:
                data = data.float().to(device)
                targets = targets.float().unsqueeze(1).to(device)

                # 前向传播
                outputs = model(data)
                loss = criterion(outputs, targets)

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 在验证集上计算 MSE
        val_predictions = []
        val_targets = []
        for data, targets in val_loader:
            data = data.float().to(device)
            targets = targets.float().unsqueeze(1).to(device)
            outputs = model(data)
            val_predictions.extend(outputs.detach().cpu().numpy())
            val_targets.extend(targets.detach().cpu().numpy())
        val_mse = mean_squared_error(val_targets, val_predictions)

        print ('Epoch [{}/{}], Loss: {:.4f}, Val MSE: {:.4f}' 
                .format(epoch+1, num_epochs, loss.item(), val_mse))
            
            # 检查是否连续增加
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            stop_count = 0
                # 保存最佳模型参数
            best_model_state_dict = model.state_dict()
        else:
            stop_count += 1
            if stop_count >= early_stop:
                print("Validation MSE increased for 3 consecutive epochs. Training stopped.")
                break
        
        if stop_count >= early_stop:
            break

    # 返回在验证集上表现最好的模型参数
    model.load_state_dict(best_model_state_dict)
    
    torch.save(model.state_dict(), PATH)