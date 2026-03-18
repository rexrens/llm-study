import pandas as pd
from urllib import request
from gensim.models import Word2Vec

def print_playlist(playlists):
    # 打印第一个和第二个播放列表作为示例
    print('playlist #1:\n', playlists[0], '\0')
    print('playlist #2:\n', playlists[1], '\0')

def load_play_list():
    # 从指定URL读取播放列表数据
    data = request.urlopen("https://storage.googleapis.com/maps-premium/dataset/yes_complete/train.txt")

    # 解码并分割文本行，跳过前两行（可能是标题或元数据）
    lines = data.read().decode('utf-8').split('\n')[2:]

    # 将每一行分割成歌曲ID列表，并过滤掉长度小于等于1的播放列表
    return [s.rstrip().split() for s in lines if len(s.split()) > 1]

def load_songs():
    # 从指定URL读取歌曲映射信息
    songs_file = request.urlopen("https://storage.googleapis.com/maps-premium/dataset/yes_complete/song_hash.txt")
    songs_file = songs_file.read().decode('utf-8').split('\n')
    songs = [s.rstrip().split('\t') for s in songs_file]

    # 创建DataFrame存储歌曲信息，并设置'id'列为索引
    songs_df = pd.DataFrame(data=songs, columns=['id', 'title', 'artist'])
    songs_df = songs_df.set_index('id')
    return songs_df

# 查找与特定歌曲最相似的10首歌曲
def print_recommendations(m, song_id):
    import numpy as np

    sim_songs = m.wv.most_similar(positive=[str(song_id)], topn=10)
    s = np.array(sim_songs)[:,0]
    return songs_df.iloc[s]

if __name__ == '__main__':

    playlists = load_play_list()

    print_playlist(playlists)

    songs_df = load_songs()

    # 使用Word2Vec模型对播放列表进行训练
    model = Word2Vec(sentences=playlists, vector_size=100, window=20, negative=50, min_count=1, workers=4)

    song_id = 2172
    # 输出指定歌曲的信息
    print(songs_df.iloc[song_id])

    print(print_recommendations(m= model, song_id=song_id))

