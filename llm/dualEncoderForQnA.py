"""
    Note: such a model can only be trained for small tasks
"""
from datasets import load_dataset
from numpy import mean
from sklearn.metrics.pairwise import cosine_similarity
from torch import arange, argmax, clamp, long, nn, no_grad, sum, tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, BertModel, BertTokenizer

###
tokenizer = BertTokenizer.from_pretrained('google-bert/bert-base-uncased')
model = BertModel.from_pretrained('google-bert/bert-base-uncased')
# Function to get BERT embeddings
def get_bert_embeddings(sentence, word):
    inputs = tokenizer(sentence, return_tensors='pt')
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    word_tokens = tokenizer.tokenize(sentence)
    word_index = word_tokens.index(word)
    word_embedding = last_hidden_states[0, word_index+1, :]  # +1 to account for [CLS] token
    return word_embedding.detach().numpy()
print(' --> test how similar does BERT find the word "bat"')
sentence1 = "The bat flew out of the cave at night."
sentence2 = "He swung the bat and hit a home run."
word = "bat"
bert_embedding1 = get_bert_embeddings(sentence1, word)
bert_embedding2 = get_bert_embeddings(sentence2, word)
print(cosine_similarity([bert_embedding1], [bert_embedding2])[0][0])

###
from sentence_transformers import CrossEncoder
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, 
                     default_activation_function=nn.Sigmoid())
question = "Where is the capital of India?"
answers = [
    "New Delhi is the capital of India.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
    "Where is the capital of India?" # <-- note the question itself is one of the answers
]
scores = model.predict([(question, answers[0]),
                        (question, answers[1]),
                        (question, answers[2]),
                        (question, answers[3])])
print(scores)
most_relevant_idx = argmax(tensor(scores)).item()
print(f"The most relevant passage is: {answers[most_relevant_idx]}")

###
def get_sentence_embedding(sentence):
    encoded_input = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    attention_mask = encoded_input['attention_mask']   # to indicate which tokens are valid and which are padding
    # Get the model output (without the specific classification head)
    with no_grad(): output = model(**encoded_input)
    token_embeddings = output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    # mean pooling operation, considering the BERT input_mask and padding
    sentence_embedding = sum(token_embeddings * input_mask_expanded, 1) / clamp(input_mask_expanded.sum(1), min=1e-9)
    return sentence_embedding.flatten().tolist()
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, output_embed_dim):
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead=8, batch_first=True),
            num_layers=3,
            norm=nn.LayerNorm([embed_dim]),
            enable_nested_tensor=False
        )
        self.projection = nn.Linear(embed_dim, output_embed_dim)
    def forward(self, tokenizer_output):
        x = self.embedding_layer(tokenizer_output['input_ids'])
        x = self.encoder(x, src_key_padding_mask=tokenizer_output['attention_mask'].logical_not())
        cls_embed = x[:,0,:]
        return self.projection(cls_embed)
def train(dataset, num_epochs=10):
    embed_size = 512
    output_embed_size = 128
    max_seq_len = 64
    batch_size = 32
    n_iters = len(dataset) // batch_size + 1
    # define the question/answer encoders
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    question_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)
    answer_encoder = Encoder(tokenizer.vocab_size, embed_size, output_embed_size)
    # define the dataloader, optimizer and loss function    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)    
    optimizer = Adam(list(question_encoder.parameters()) + list(answer_encoder.parameters()), lr=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        running_loss = []
        for idx, data_batch in enumerate(dataloader):
            # Tokenize the question/answer pairs (each is a batc of 32 questions and 32 answers)
            question, answer = data_batch
            question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
            answer_tok = tokenizer(answer, padding=True, truncation=True, return_tensors='pt', max_length=max_seq_len)
            if idx == 0 and epoch == 0:
                print(question_tok['input_ids'].shape, answer_tok['input_ids'].shape)
            # Compute the embeddings: the output is of dim = 32 x 128
            question_embed = question_encoder(question_tok)
            answer_embed = answer_encoder(answer_tok)
            if idx == 0 and epoch == 0:
                print(question_embed.shape, answer_embed.shape)
            # Compute similarity scores: a 32x32 matrix
            # row[N] reflects similarity between question[N] and answers[0...31]
            similarity_scores = question_embed @ answer_embed.T
            if idx == 0 and epoch == 0:
                print(similarity_scores.shape)
            # we want to maximize the values in the diagonal
            target = arange(question_embed.shape[0], dtype=long)
            loss = loss_fn(similarity_scores, target)
            running_loss += [loss.item()]
            if idx == n_iters-1:
                print(f"Epoch {epoch}, loss = ", mean(running_loss))
            optimizer.zero_grad()    # reset optimizer so gradients are all-zero
            loss.backward()
            optimizer.step()
    return question_encoder, answer_encoder
class MyDataset(Dataset):
    def __init__(self, datapath):
        self.data = pd.read_csv(datapath, sep="\t", nrows=300)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data.iloc[idx]['questions'], self.data.iloc[idx]['answers']
dataset = MyDataset('./shared_data/nq_sample.tsv')
dataset.data.head(5)
qe, ae = train(dataset, num_epochs=5)

question = 'What is the tallest mountain in the world?'
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
question_tok = tokenizer(question, padding=True, truncation=True, return_tensors='pt', max_length=64)
question_emb = qe(question_tok)[0]
print(question_tok)
print(question_emb[:5])

from pandas import DataFrame
sts_dataset = load_dataset("mteb/stsbenchmark-sts")
sts = DataFrame({'sent1': sts_dataset['test']['sentence1'], 
                    'sent2': sts_dataset['test']['sentence2'],
                    'score': [x/5 for x in sts_dataset['test']['score']]})
sts.head(10)
