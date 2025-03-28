import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


user_df = pd.read_excel("users.xlsx")      
movies_df = pd.read_excel("movies.xlsx")    
ratings_df = pd.read_excel("ratings.xlsx")  


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


user_df["Occupation"] = user_df["Occupation"].astype(int)


def split_genres(genres_str):
    return genres_str.split("|")

movies_df["genre_list"] = movies_df["Genres"].apply(split_genres)

all_genres = set()
for genres in movies_df["genre_list"]:
    for g in genres:
        all_genres.add(g)
all_genres = sorted(list(all_genres))
genre_to_idx = {genre: idx for idx, genre in enumerate(all_genres)}
num_genres = len(genre_to_idx) 
def map_genres_to_ids(genre_list):
    return [genre_to_idx[g] for g in genre_list]

movies_df["genre_ids"] = movies_df["genre_list"].apply(map_genres_to_ids)


def pad_genre_list(genre_list, max_len=7, pad_value=-1):
    if len(genre_list) < max_len:
        return genre_list + [pad_value] * (max_len - len(genre_list))
    else:
        return genre_list[:max_len]

movies_df["genre_ids_padded"] = movies_df["genre_ids"].apply(lambda x: pad_genre_list(x, max_len=7))


movieid_to_genres = dict(zip(movies_df["MovieID"], movies_df["genre_ids"]))

liked_threshold = 4  # 评分>=4视为喜欢

user_genre_counts = {} 
for _, row in ratings_df.iterrows():
    user_id = row["UserID"]
    movie_id = row["MovieID"]
    rating = row["Rating"]
    if rating >= liked_threshold and movie_id in movieid_to_genres:
        if user_id not in user_genre_counts:
            user_genre_counts[user_id] = []
        user_genre_counts[user_id].extend(movieid_to_genres[movie_id])

def get_top_genres(genre_list, top_n=7):
    if not genre_list:
        return [-1] * top_n  
    counts = Counter(genre_list)
    top = [genre for genre, _ in counts.most_common(top_n)]
    if len(top) < top_n:
        top = top + [-1] * (top_n - len(top))
    return top

user_df["genre_ids_pref"] = user_df["UserID"].apply(lambda uid: get_top_genres(user_genre_counts.get(uid, []), top_n=7))



gender_data = np.stack(user_df["gender_onehot"].values)
age_data = np.stack(user_df["age_onehot"].values)
occupation_data = user_df["Occupation"].values.astype(np.int64)
genre_pref_data = np.stack(user_df["genre_ids_pref"].values)

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
        super(UserEncoder, self).__init__()
        self.occupation_embed = nn.Embedding(occupation_num, occupation_embed_dim)
        self.genre_embed = nn.Embedding(num_genres, genre_embed_dim)
        self.max_genres = max_genres
        input_dim = 2 + age_onehot_dim + occupation_embed_dim + genre_embed_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, gender_onehot, age_onehot, occupation_id, genre_ids):
        occ_emb = self.occupation_embed(occupation_id)
        mask = (genre_ids >= 0).float()
        genre_ids_clamped = genre_ids.clone()
        genre_ids_clamped[genre_ids_clamped < 0] = 0
        genre_embs = self.genre_embed(genre_ids_clamped)
        counts = mask.sum(dim=1, keepdim=True) + 1e-8
        weights = mask * (7.0 / counts)
        weighted_genre_emb = (genre_embs * weights.unsqueeze(-1)).sum(dim=1)
        x = torch.cat([gender_onehot, age_onehot, occ_emb, weighted_genre_emb], dim=1)
        user_latent = F.relu(self.fc(x))
        return user_latent


class MovieEncoder(nn.Module):
    def __init__(self, occupation_num, occupation_embed_dim,
                 num_genres, genre_embed_dim, max_genres, output_dim):
        """
        参数:
          occupation_num: 职业类别总数（如21）
          occupation_embed_dim: 职业 Embedding 维度（如8）
          num_genres: 电影类型总数（如18）
          genre_embed_dim: 电影类型 Embedding 维度（如8）
          max_genres: 电影类型的固定长度（7）
          output_dim: 最终电影隐特征维度（如32）
        """
        super(MovieEncoder, self).__init__()
        self.occupation_embed = nn.Embedding(occupation_num, occupation_embed_dim)
        self.genre_embed = nn.Embedding(num_genres, genre_embed_dim)
        self.max_genres = max_genres
        input_dim = genre_embed_dim + occupation_embed_dim
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, genre_ids, occupation_id):
        # 为避免负数导致索引错误，将 genre_ids 中的 -1 替换为 0
        genre_ids_clamped = genre_ids.clone()
        genre_ids_clamped[genre_ids_clamped < 0] = 0
        genre_embs = self.genre_embed(genre_ids_clamped)  # [batch_size, max_genres, genre_embed_dim]
        mask = (genre_ids >= 0).float()
        counts = mask.sum(dim=1, keepdim=True) + 1e-8
        weights = mask * (7.0 / counts)
        weighted_genre_emb = (genre_embs * weights.unsqueeze(-1)).sum(dim=1)
        occ_emb = self.occupation_embed(occupation_id)
        x = torch.cat([weighted_genre_emb, occ_emb], dim=1)
        movie_latent = F.relu(self.fc(x))
        return movie_latent


def get_movie_occupation(movie_id):
    movie_ratings = ratings_df[ratings_df["MovieID"] == movie_id]
    if movie_ratings.empty:
        return 0
    max_rating = movie_ratings["Rating"].max()
    best_rows = movie_ratings[movie_ratings["Rating"] == max_rating]
    best_user_id = best_rows.iloc[0]["UserID"]
    occ = user_df[user_df["UserID"] == best_user_id]["Occupation"]
    if occ.empty:
        return 0
    return int(occ.iloc[0])

movies_df["occupation_for_movie"] = movies_df["MovieID"].apply(get_movie_occupation)


genre_data = np.stack(movies_df["genre_ids_padded"].values)  # 使用填充后的版本
occupation_data_movie = movies_df["occupation_for_movie"].values.astype(np.int64)

genre_tensor = torch.tensor(genre_data, dtype=torch.long)
occupation_tensor_movie = torch.tensor(occupation_data_movie, dtype=torch.long)

print("Genre tensor shape:", genre_tensor.shape)
print("Movie occupation tensor shape:", occupation_tensor_movie.shape)


occupation_num = 21
occupation_embed_dim = 8
num_genres_total = num_genres
genre_embed_dim = 8
max_genres = 7
output_dim = 32

movie_encoder = MovieEncoder(occupation_num, occupation_embed_dim,
                             num_genres_total, genre_embed_dim, max_genres, output_dim)

movie_latent_features = movie_encoder(genre_tensor, occupation_tensor_movie)
print("Movie latent features shape:", movie_latent_features.shape)  # 期望：[num_movies, output_dim]

#############################
# 保存电影隐特征到 CSV 文件
#############################
movie_latent_features_df = pd.DataFrame(movie_latent_features.detach().cpu().numpy())
movie_latent_features_df.insert(0, "MovieID", movies_df["MovieID"])
movie_latent_features_df.to_csv("movie_latent_features.csv", index=False)
print("Movie latent features saved to movie_latent_features.csv")