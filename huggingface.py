from transformers import BertTokenizer, BertForSequenceClassification, pipeline

finbert_tone = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone',num_labels=3)
tokenizer_tone = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')

nlp_tone = pipeline("text-classification", model=finbert_tone, tokenizer=tokenizer_tone)

sentences = []

input_vals = 'We experienced our largest financial loss to date.|By reducing the amount of materials used to make our products, we reduce the emissions from transporting and processing these materials, and limit the amount of scrap generated along the way.|We expect next year to be a bellwhether year.' #@param {type:"string"}
if '|' in input_vals:
  sentences = input_vals.split('|')
else:
  sentences = [input_vals]
  
results = nlp_tone(sentences)
print(*zip(sentences, results), sep='\n')