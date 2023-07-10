from transformers import MarianMTModel, MarianTokenizer
import pandas as pd
import torch
#Get GPU
device_name = torch.cuda.get_device_name()
n_gpu = torch.cuda.device_count()
print(f"Found device: {device_name}, n_gpu: {n_gpu}")
device = torch.device("cuda")

# Get the name of the first model
first_model_name = 'Helsinki-NLP/opus-mt-en-fr'
# Get the tokenizer
first_model_tkn = MarianTokenizer.from_pretrained(first_model_name)
# Load the pretrained model based on the name
first_model = MarianMTModel.from_pretrained(first_model_name)
first_model = first_model.to(device)
# Get the name of the second model
second_model_name = 'Helsinki-NLP/opus-mt-fr-en'
# Get the tokenizer
second_model_tkn = MarianTokenizer.from_pretrained(second_model_name)
# Load the pretrained model based on the name
second_model = MarianMTModel.from_pretrained(second_model_name)
second_model = second_model.to(device)


noiseLevel = 0.1 #Change amount of noise generated here (Percentage of words in input, not chars)
queries_eval = pd.read_csv('queries.eval.tsv', sep='\t', header = None)
queries_train = pd.read_csv('queries.train.tsv', sep='\t', header = None)
queries_dev = pd.read_csv('queries.dev.tsv', sep='\t', header = None)
collection = pd.read_csv('collection.tsv', sep='\t', header = None)

#HELPER FUNCTIONS
#Add language token in front of original texts i.e. "">>fr<< what is prescribed to treat thyroid storm"
def format_batch_texts(language_code, batch_texts):

  formated_bach = [">>{}<< {}".format(language_code, text) for text in batch_texts]

  return formated_bach

def perform_translation(batch_texts, model, tokenizer, language="fr"):
    # Prepare the text data into appropriate format for the model
    formated_batch_texts = format_batch_texts(language, batch_texts)
    # Generate translation using model
    translated = model.generate(**tokenizer(formated_batch_texts, return_tensors="pt", padding=True).to(device))

    # Convert the generated tokens indices back into text
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated]

    return translated_texts
def combine_texts(original_texts, back_translated_batch):

  return set(original_texts + back_translated_batch)

def perform_back_translation_with_augmentation(batch_texts, original_language="en", temporary_language="fr"):

  # Translate from Original to Temporary Language
  tmp_translated_batch = perform_translation(batch_texts, first_model, first_model_tkn, temporary_language)

  # Translate Back to English
  back_translated_batch = perform_translation(tmp_translated_batch, second_model, second_model_tkn, original_language)

  # Return The Final Result
  #return combine_texts(temp, back_translated_batch)
  return back_translated_batch

def generateNoise(outputfileName, inputData):
  inputData = inputData[1].tolist()
  # Execute the function for Data Augmentation
  final_augmented = perform_back_translation_with_augmentation(inputData)
  # Find the name of the column by index
  n = queries_eval.columns[1]
  # Drop that column
  queries_eval.drop(n, axis = 1, inplace = True)
  # Put whatever series you want in its place
  queries_eval[n] = final_augmented
  inputData.to_csv(outputfileName, sep = '\t', header = None, index = False)

#Perform Back Translation
evalCopy = queries_eval.copy()
generateNoise("queries.eval.BackTranslation.tsv", evalCopy)
del evalCopy

trainCopy = queries_train.copy()
generateNoise("queries.train.BackTranslation.tsv", trainCopy)
del trainCopy

devCopy = queries_dev.copy()
generateNoise("queries.dev.BackTranslation.tsv", devCopy)
del devCopy

collectionCopy = collection.copy()
generateNoise("collection.BackTranslation.tsv", collectionCopy)
del collectionCopy
