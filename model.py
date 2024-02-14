import torch
import config
from transformers import BertModel, BertPreTrainedModel, BertTokenizer

# Bert Pretrained model with final classifier 
class SentimentMultilabel(BertPreTrainedModel):
    def __init__(self, num_labels, conf):
        super(SentimentMultilabel, self).__init__(conf)
        self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL)
        self.drop = torch.nn.Dropout(0.4)
        self.classifier = torch.nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.drop(pooled_output)
        output = self.classifier(output)
        return output

# Bert Large Pretrained model with final classifier 
class SentimentMultilabelLarge(BertPreTrainedModel):
    def __init__(self, num_labels, conf):
        super(SentimentMultilabelLarge, self).__init__(conf)
        self.bert = BertModel.from_pretrained(config.PRE_TRAINED_MODEL)
        self.drop = torch.nn.Dropout(0.4)
        self.classifier = torch.nn.Linear(1024, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        _, pooled_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        output = self.drop(pooled_output)
        output = self.classifier(output)
        return output