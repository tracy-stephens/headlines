from transformers import BertTokenizer, TFBertModel

TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased')

def addWord(word, pos, ner):
    """
    Convert a word into a word token and add supplied NER and POS labels. Note that the word can be
    tokenized to two or more tokens. Correspondingly, we add - for now - custom 'X' tokens to the labels in order to
    maintain the 1:1 mappings between word tokens and labels.

    arguments: word, pos label, ner label
    returns: dictionary with tokens and labels
    """
    # the dataset contains various '"""' combinations which we choose to truncate to '"', etc.
    if word == '""""':
        word = '"'
    elif word == '``':
        word = '`'

    tokens = TOKENIZER.tokenize(word)
    tokenLength = len(tokens)  # find number of tokens corresponfing to word to later add 'X' tokens to labels

    addDict = dict()

    addDict['wordToken'] = tokens
    addDict['posToken'] = [pos] + ['posX'] * (tokenLength - 1)
    addDict['nerToken'] = [ner] + ['nerX'] * (tokenLength - 1)
    addDict['tokenLength'] = tokenLength

    return addDict


if __name__ == "__main__":

    #print(addWord('protest', 'VB', 'O'))



