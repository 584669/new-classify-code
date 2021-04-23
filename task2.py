from task1 import *
from matplotlib import pyplot as plt
#数据读取与数据分析
train_df = pd.read_csv(data_path+'/train_set.csv', sep='\t', nrows=100)
print(train_df.head())
#句子长度分析
train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
print(train_df['text_len'].describe())

_ = plt.hist(train_df['text_len'], bins=200)
plt.xlabel('Text char count')
plt.title("Histogram of char count")
plt.show()
#新闻类别分布
train_df['label'].value_counts().plot(kind='bar')
plt.title('News class count')
plt.xlabel("category")
plt.show()
#字符分布统计
from collections import Counter
all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:d[1], reverse = True)

print(len(word_count))

print(word_count[0])

print(word_count[-1])

#

train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
all_lines = ' '.join(list(train_df['text_unique']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d:int(d[1]), reverse = True)

print(word_count[0])

print(word_count[1])

print(word_count[2])


# 数据分析的结论
# 通过上述分析我们可以得出以下结论：
#
# 赛题中每个新闻包含的字符个数平均为1000个，还有一些新闻字符较长；
# 赛题中新闻类别分布不均匀，科技类新闻样本量接近4w，星座类新闻样本量不到1k；
# 赛题总共包括7000-8000个字符；
# 通过数据分析，我们还可以得出以下结论：
#
# 每个新闻平均字符个数较多，可能需要截断；
#
# 由于类别不均衡，会严重影响模型的精度；