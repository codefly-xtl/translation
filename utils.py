import torch


def no_peace(char, pre_char):
    return char in set(',.!?') and pre_char != ' '


def process_data():
    # 加载数据
    with open('./data/fra.txt', encoding='utf-8') as f:
        raw_data = f.read()
    # 对数据进行处理:变小写,在标点符号前插入空格
    raw_data = raw_data.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    out = [' ' + char if i > 0 and no_peace(char, raw_data[i - 1]) else char for i, char in enumerate(raw_data)]
    data = ''.join(out)
    return data


def get_sentence(data):
    source = []
    target = []
    for line in data.split('\n'):
        parts = line.split('\t')
        if len(parts) == 2:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))
    # source 例: source = [['i am person'],['i like you']]
    return source, target


class Vocab:
    def __init__(self, sentence, min_freq=0, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        self.all_words = [word for words in sentence for word in words]
        self.word_preq = self.get_word_preq()
        self.index_to_word = ['<unk>'] + reserved_tokens
        self.word_to_index = {word: index for index, word in enumerate(self.index_to_word)}
        for word, freq in self.word_preq:
            if freq < min_freq:
                break
            self.index_to_word.append(word)
            self.word_to_index[word] = len(self.word_to_index)

    # 统计词频
    def get_word_preq(self):
        word_preq = {}
        for word in self.all_words:
            if word not in word_preq:
                word_preq[word] = 1
            else:
                word_preq[word] += 1
        # 排序
        word_preq = sorted(word_preq.items(), key=lambda x: x[1], reverse=True)
        return word_preq

    def __len__(self):
        return len(self.index_to_word)

    def to_word(self, indexs):
        return [self.index_to_word[i] for i in indexs]

    def prase(self, raw_data):
        raw_data = raw_data.replace('\u202f', ' ').replace('\xa0', ' ').lower()
        out = [' ' + char if i > 0 and no_peace(char, raw_data[i - 1]) else char for i, char in enumerate(raw_data)]
        data = ''.join(out)
        source = []
        for line in data.split('\n'):
            source.append(line.split(' '))
        data = []
        for i in range(len(source)):
            source_sentence = source[i]
            source_word = truncate_or_pad(source_sentence, 20)
            source_index = self.to_index(source_word)
            data.append(source_index)
        return torch.stack(data)

    def to_index(self, words):
        output = []
        for word in words:
            if word not in self.index_to_word:
                output.append(self.word_to_index['<unk>'])
            else:
                output.append(self.word_to_index[word])
        return torch.tensor(output)


def truncate_or_pad(line, num_steps):
    # 例: line = ['i','am','person']
    # 超出后进行截断
    if len(line) > num_steps:
        return line[:num_steps]
    # 没有超出就pad
    line.append('<eos>')
    for i in range(num_steps - len(line)):
        line.append('<pad>')
    return line


def get_train_iter(batch_size, num_steps):
    data = process_data()
    # source_sentences 例: source_sentences = [['i am person'],['i like you']]
    source_sentences, target_sentences = get_sentence(data)
    source_Vocab = Vocab(source_sentences, min_freq=0, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    target_Vocab = Vocab(target_sentences, min_freq=0, reserved_tokens=['<pad>', '<bos>', '<eos>'])
    database = []
    batch_num = len(source_Vocab) // batch_size
    # 每一个batch放在database里面
    for i in range(batch_num):
        source_batch = []
        source_len_batch = []
        target_batch = []
        target_len_batch = []
        for j in range(batch_size):
            # 获取一个句子以及翻译
            source_sentence = source_sentences[i * batch_size + j]
            target_sentence = target_sentences[i * batch_size + j] + ['<eos>']
            source_valid_len = len(source_sentence)
            target_valid_len = len(target_sentence)
            # 将句子变为单词列表,超过num_steps的截断,不够num_steps的补齐
            source_word = truncate_or_pad(source_sentence, num_steps)
            target_word = truncate_or_pad(target_sentence, num_steps)
            # 获取单词对应的标号
            source_index = source_Vocab.to_index(source_word)
            target_index = target_Vocab.to_index(target_word)
            # 存放起来
            source_batch.append(source_index)
            source_len_batch.append(source_valid_len)
            target_batch.append(target_index)
            target_len_batch.append(target_valid_len)
        source_batch_tensor = torch.stack(source_batch)
        target_batch_tensor = torch.stack(target_batch)
        source_len_batch_tensor = torch.tensor(source_len_batch)
        target_len_batch_tensor = torch.tensor(target_len_batch)
        database.append((source_batch_tensor, source_len_batch_tensor, target_batch_tensor, target_len_batch_tensor))
    return database, source_Vocab, target_Vocab
