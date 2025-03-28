import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


user_df = pd.read_excel("users.xlsx")  # 用户数据：UserID, Gender, Age, Occupation
movies_df = pd.read_excel("movies.xlsx")  # 电影数据：MovieID, Title, Genres
ratings_df = pd.read_excel("ratings.xlsx")  # 评分数据：UserID, MovieID, Rating



def gender_to_onehot(gender):
    return np.array([1, 0]) if gender == "F" else np.array([0, 1])


user_df["gender_onehot"] = user_df["Gender"].apply(gender_to_onehot)


age_mapping = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}


def age_to_onehot(age):
    onehot = np.zeros(7)
    idx = age_mapping.get(age, 0)
    onehot[idx] = 1
    return onehot


user_df["age_onehot"] = user_df["Age"].apply(age_to_onehot)

# 2.3 Occupation 保证为整数
user_df["Occupation"] = user_df["Occupation"].astype(int)



def split_genres(genres_str):
    return genres_str.split("|")


movies_df["genre_list"] = movies_df["Genres"].apply(split_genres)

# 构建电影类型映射字典
all_genres = set()
for genres in movies_df["genre_list"]:
    for g in genres:
        all_genres.add(g)
all_genres = sorted(list(all_genres))
genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
num_genres = len(genre_to_idx)  # 例如 18


# 为每部电影生成类型ID列表（不做填充，直接存放原始列表）
def map_genres_to_ids(genre_list):
    return [genre_to_idx[g] for g in genre_list]


movies_df["genre_ids"] = movies_df["genre_list"].apply(map_genres_to_ids)


movieid_to_genres = dict(zip(movies_df["MovieID"], movies_df["genre_ids"]))


liked_threshold = 4 


from collections import Counter

user_genre_counts = {}  # key: UserID, value: list of genre IDs（可能有重复）
for _, row in ratings_df.iterrows():
    user_id = row["UserID"]
    movie_id = row["MovieID"]
    rating = row["Rating"]
    if rating >= liked_threshold and movie_id in movieid_to_genres:
        if user_id not in user_genre_counts:
            user_genre_counts[user_id] = []
        user_genre_counts[user_id].extend(movieid_to_genres[movie_id])


# 对每个用户，统计出现频率最高的类型，并选择最多 top_n 个（top_n取7，即固定长度7）
def get_top_genres(genre_list, top_n=7):
    if not genre_list:
        return [-1] * top_n  # 若没有喜好，则填充 -1
    counts = Counter(genre_list)
    # 按出现频率降序排序，选取前 top_n 个
    top = [genre for genre, _ in counts.most_common(top_n)]
    # 如不足 top_n 个，填充 -1
    if len(top) < top_n:
        top = top + [-1] * (top_n - len(top))
    return top


# 将每个用户的喜好类型列表放入 user_df，新列命名为 "genre_ids_pref"
user_df["genre_ids_pref"] = user_df["UserID"].apply(lambda uid: get_top_genres(user_genre_counts.get(uid, []), top_n=7))


gender_data = np.stack(user_df["gender_onehot"].values)  # [num_users, 2]
age_data = np.stack(user_df["age_onehot"].values)  # [num_users, 7]
occupation_data = user_df["Occupation"].values.astype(np.int64)  # [num_users]
genre_pref_data = np.stack(user_df["genre_ids_pref"].values)  # [num_users, 7]

gender_tensor = torch.tensor(gender_data, dtype=torch.float)
age_tensor = torch.tensor(age_data, dtype=torch.float)
occupation_tensor = torch.tensor(occupation_data, dtype=torch.long)
genre_pref_tensor = torch.tensor(genre_pref_data, dtype=torch.long)

print("Gender tensor shape:", gender_tensor.shape)
print("Age tensor shape:", age_tensor.shape)
print("Occupation tensor shape:", occupation_tensor.shape)
print("Genre preference tensor shape:", genre_pref_tensor.shape)



class UserEncoder(nn.Module):
    def __init__(self, age_onehot_dim, occupation_num, occupation_embed_dim,
                 num_genres, genre_embed_dim, max_genres, output_dim):
        """
        参数:
          age_onehot_dim: 年龄 one-hot 的维度（7）
          occupation_num: 职业类别总数（如21）
          occupation_embed_dim: 职业 Embedding 维度（如8）
          num_genres: 电影类型总数（如18）
          genre_embed_dim: 电影类型 Embedding 维度（如8）
          max_genres: 用户喜好类型的固定长度（7）
          output_dim: 最终用户隐特征维度（如32）
        """
        super(UserEncoder, self).__init__()
        # 对职业使用 Embedding
        self.occupation_embed = nn.Embedding(occupation_num, occupation_embed_dim)
        # 对电影类型喜好使用 Embedding
        self.genre_embed = nn.Embedding(num_genres, genre_embed_dim)
        self.max_genres = max_genres

        # 拼接后输入全连接层的维度：
        # 性别: 2, 年龄: 7, 职业 Embedding: occupation_embed_dim, 电影类型喜好: genre_embed_dim
        input_dim = 2 + age_onehot_dim + occupation_embed_dim + genre_embed_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, gender_onehot, age_onehot, occupation_id, genre_ids):
        """
        参数:
          gender_onehot: Tensor, [batch_size, 2] (one-hot编码)
          age_onehot: Tensor, [batch_size, 7] (one-hot编码)
          occupation_id: LongTensor, [batch_size]
          genre_ids: LongTensor, [batch_size, max_genres]; 无效值为 -1
        输出:
          用户隐特征, [batch_size, output_dim]
        """
        # 职业嵌入
        occ_emb = self.occupation_embed(occupation_id)  # [batch_size, occupation_embed_dim]

        # 电影类型喜好部分：
        # 由于 genre_ids 中可能有 -1（填充值），先构造 mask：有效位置 (>= 0)
        mask = (genre_ids >= 0).float()  # [batch_size, max_genres]
        # 为了使用 Embedding 层，先将 -1 替换为 0（Embedding后再通过 mask 忽略不合法位置）
        genre_ids_clamped = genre_ids.clone()
        genre_ids_clamped[genre_ids_clamped < 0] = 0  # 替换所有 -1 为 0
        genre_embs = self.genre_embed(genre_ids_clamped)  # [batch_size, max_genres, genre_embed_dim]
        # 计算每个用户有效的类型数量
        counts = mask.sum(dim=1, keepdim=True)  # [batch_size, 1]
        counts = counts + 1e-8  # 防止除以0
        # 每个有效类型的权重 = 7 / (有效数量)
        weights = mask * (7.0 / counts)  # [batch_size, max_genres]
        # 加权求和：将 weights 扩展至最后一维
        weighted_genre_emb = (genre_embs * weights.unsqueeze(-1)).sum(dim=1)  # [batch_size, genre_embed_dim]

        # 拼接所有特征
        x = torch.cat([gender_onehot, age_onehot, occ_emb, weighted_genre_emb], dim=1)
        user_latent = F.relu(self.fc(x))
        return user_latent


#############################
# 7. 初始化用户编码器并计算用户隐特征
#############################
# 参数设置
age_onehot_dim = 7
occupation_num = 21  # 职业 0~20 共21类
occupation_embed_dim = 8
num_genres_total = num_genres  # 根据电影数据中所有类型数量
genre_embed_dim = 8
max_genres = 7  # 固定为7
output_dim = 32

user_encoder = UserEncoder(age_onehot_dim, occupation_num, occupation_embed_dim,
                           num_genres_total, genre_embed_dim, max_genres, output_dim)

# 假设我们对所有用户进行批量处理
user_latent_features = user_encoder(gender_tensor, age_tensor, occupation_tensor, genre_pref_tensor)
print("User latent features shape:", user_latent_features.shape)  # 期望：[num_users, output_dim]

#############################
# 8. 保存用户隐特征到 CSV 文件
#############################
latent_np = user_latent_features.detach().cpu().numpy()
latent_df = pd.DataFrame(latent_np)
latent_df.to_csv("user_latent_features.csv", index=False)
print("用户隐特征已保存到 user_latent_features.csv")