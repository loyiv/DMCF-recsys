import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

user_latent_df = pd.read_csv("user_latent_features.csv")  # 用户隐特征
movie_latent_df = pd.read_csv("movie_latent_features.csv")  # 电影隐特征

# 将用户和电影的隐特征数据转换为 PyTorch 张量
user_latent_tensor = torch.tensor(user_latent_df.drop(columns=["UserID"]).values, dtype=torch.float)
movie_latent_tensor = torch.tensor(movie_latent_df.drop(columns=["MovieID"]).values, dtype=torch.float)


num_users = user_latent_tensor.shape[0]
num_movies = movie_latent_tensor.shape[0]

user_latent_expanded = user_latent_tensor.unsqueeze(1).expand(num_users, num_movies, -1)
movie_latent_expanded = movie_latent_tensor.unsqueeze(0).expand(num_users, num_movies, -1)


combined_latent = torch.cat([user_latent_expanded, movie_latent_expanded], dim=2)
print("Combined latent shape:", combined_latent.shape)

# 定义 Decoder 模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一层：输入维度为 input_dim
        self.fc2 = nn.Linear(hidden_dim, output_dim)   # 第二层：输出维度为 output_dim

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.mlp = MLP(input_dim, hidden_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.mlp(x)
        x = self.tanh(x) * 2 + 3  # 将输出缩放到 [1, 5] 区间
        return x

# 计算拼接后隐特征的维度
combined_dim = user_latent_tensor.shape[1] + movie_latent_tensor.shape[1]
hidden_dim = 64
output_dim = 1

# 初始化 Decoder 模型
decoder = Decoder(combined_dim, hidden_dim, output_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
decoder = decoder.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    decoder = nn.DataParallel(decoder)

combined_latent_flattened = combined_latent.view(-1, combined_dim).to(device)


predicted_ratings = decoder(combined_latent_flattened)
print("Initial predicted ratings shape:", predicted_ratings.shape)  


user_ids = np.tile(user_latent_df["UserID"].values, num_movies)
movie_ids = np.tile(movie_latent_df["MovieID"].values, num_users)
predicted_ratings_df = pd.DataFrame(predicted_ratings.detach().cpu().numpy(), columns=["PredictedRating"])
predicted_ratings_df.insert(0, "UserID", user_ids)
predicted_ratings_df.insert(1, "MovieID", movie_ids)
predicted_ratings_df['idx'] = range(len(predicted_ratings_df))


min_rating = predicted_ratings_df['PredictedRating'].min()
max_rating = predicted_ratings_df['PredictedRating'].max()
predicted_ratings_df['ScaledRating'] = (predicted_ratings_df['PredictedRating'] - min_rating) * 6 / (max_rating - min_rating)


predicted_ratings_df.to_csv("predicted_ratings3.csv", index=False)
print("初步预测评分已保存到 predicted_ratings3.csv")


actual_ratings_df = pd.read_excel("ratings.xlsx")
actual_ratings_df = actual_ratings_df.drop(columns=["Timestamp"])
actual_ratings_df.columns = ["UserID", "MovieID", "Rating"]


merged_df = pd.merge(predicted_ratings_df, actual_ratings_df, on=["UserID", "MovieID"], how="inner")


train_indices = merged_df['idx'].values
training_input = combined_latent_flattened[train_indices]
actual_ratings = torch.tensor(merged_df["Rating"].values, dtype=torch.float32).to(device)


train_input, val_input, train_labels, val_labels = train_test_split(
    training_input.cpu(), actual_ratings.cpu(), test_size=0.2, random_state=42
)

criterion = nn.MSELoss()
optimizer = optim.SGD(decoder.parameters(), lr=0.01, weight_decay=0.01)


num_epochs = 100
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    decoder.train()
    pred_train = decoder(train_input.to(device))
    loss_train = criterion(pred_train.squeeze(), train_labels.to(device))
    
    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()

    # 计算验证集损失
    decoder.eval()
    with torch.no_grad():
        pred_val = decoder(val_input.to(device))
        loss_val = criterion(pred_val.squeeze(), val_labels.to(device))
    
    # 保存训练损失和验证损失
    train_losses.append(loss_train.item())
    val_losses.append(loss_val.item())
    
    # 每个 epoch 打印一次训练和验证损失
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss_train.item():.4f}, Val Loss: {loss_val.item():.4f}")


decoder.eval()
with torch.no_grad():
    predicted_ratings_full = decoder(training_input.to(device))
    test_loss = criterion(predicted_ratings_full.squeeze(), actual_ratings)
print(f"Final Test MSE Loss: {test_loss.item():.4f}")


merged_df['PredictedRating'] = predicted_ratings_full.squeeze().cpu().numpy()
merged_df.to_csv("final_predicted_ratings1.csv", index=False)
print("最终预测评分已保存到 final_predicted_ratings1.csv")


plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Train and Validation Loss Curve")
plt.legend()
plt.savefig("Training_Validation_loss_curve.png")
print("训练和验证损失曲线已保存为 Training_Validation_loss_curve.png")
plt.show()
