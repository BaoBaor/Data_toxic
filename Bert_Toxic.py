from transformers import BertTokenizer,BertModel
import torch
import torch.nn as nn
import pandas as pd
class BERTDataset:
    def __init__(self, texts, labels, max_len = 128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.num_examples = len(self.texts)
    
    def __len__(self):
        return self.num_examples
    
    def _getitem_(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        tokenizer_text = self.tokenizer(
            text, 
            add_special_tokens= True, 
            padding ="max_len", 
            max_length=self.max_len
        )
        ids = tokenizer_text["input_ids"]
        mask = tokenizer_text["attention_mask"]
        token_type_ids = tokenizer_text["token_type_ids"]
        
        return {"ids": torch.tensor(ids, dtype=torch.long),
                "mask": torch.tensor(mask, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype = torch.long),
                "target": torch.tensor(label, dtype=torch.float),
                }
class ToxicModel(nn.Module):
    """A simple bert model for training a 2 class classification process"""
    def __init__(self,num_labels):
        super(ToxicModel,self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased", return_dict = False)
        self.droptout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.droptout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits
def train(model, train_dataset, valid_dataset, epochs =1):
    """train a model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.Parameter())
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range (epochs):
        model.train()
        train_loss = 0
        for batch in train_dataset:
            optimizer.zero_grad()
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            target = batch["target"].to(device)
            logits = model(ids, token_type_ids, mask)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch["ids"].size(0)
            
        train_loss = train_loss / len(train_dataset)
        model.eval()
        valid_loss = 0
        for batch in valid_dataset:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            target = batch["target"].to(device)
            logits = model(ids, token_type_ids, mask)
            loss = criterion(logits, target)
            valid_loss += loss.item() * batch["ids"].size(0)
            
        valid_loss = valid_loss / len(valid_dataset)
        print(
            f"Epoch {epoch+1}/{epochs}.."
            f"Train loss: {train_loss:.3f}.."
            f"Validation loss: {valid_loss:.3f}"
        )
        
if __name__ == "__main__":
    df_train = pd.read_csv("D:/PV/data_toxic/data/train.csv")
    # df_valid = pd.read_csv("")
    train_dataset = BERTDataset(
        df_train.comment_text.values, 
        df_train.is_toxic.values)
    valid_dataset = BERTDataset(
        df_valid
    )