import pandas as pd 
import pickle 
import config
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# remove null labels 
def remove_empty(text):
    return [lab for lab in text if lab != '']

# remove extra spaces from labels
def remove_space(text):
    return [lab.strip() for lab in text]

# to replace duplicate labels with correct labels
def replace_label(df, src, trg):
    def replace(texts):
        return [lab if lab != src else trg for lab in texts]
    
    df['target'] = df['target'].map(replace)

# to get all noisy labels that don't have any sentiments 
def get_noisy_labels(df):
    noisy_labels = []
    for label,count in Counter(df.target.explode()).items():
        if count < 5:
            if 'positive' not in label.split():
                if 'negative' not in label.split():
                    noisy_labels.append(label)

    return noisy_labels

# to remove nosiy labels from the dataframe
def remove_noisy_labels(df):
    print("Removing noisy labels...")
    df = df.drop([384]) # outlier with 14 labels 
    noisy_labels = get_noisy_labels(df)
    for i in range(len(df)):
        for nLabel in noisy_labels:
            if nLabel in df.iloc[i,1]:
                df.iloc[i,1].remove(nLabel)
    
    # to remove datapoints that doesn't have any labels 
    df = df[df["target"].str.len() != 0]

    return df 

# combine labels that have very low frequency to a single label based on threshold
def combine_labels(df,min_samples = 100):
    print("Combining labels...")
    label_counts = df.target.explode().value_counts()
    label_names = label_counts.index
    
    fewer_labels = []
    for i,label in enumerate(label_names):
        if label_counts[i] < min_samples:
            fewer_labels.append(label)
    
    def replace_fewer(labels):
        fewers = []
        for label in labels:
            sentiment = label.split(' ')[-1]
            if label in fewer_labels:
                fewers.append(' '.join(['extra',sentiment]))
            else:
                fewers.append(label)
                
        return fewers 
    
    df['target'] = df['target'].map(replace_fewer)  

    return df

# undesample very frequent labels 
def undersample_labels(df, labels, frac):
    udf = df[df.target.apply(lambda x: x == labels)]
    indexesToDrop = udf.index.values
    underSampleLabel = udf.sample(frac = frac)
    df = df.drop(indexesToDrop)
    df = df.append(underSampleLabel)
    return df

# encode labels for training
def encode_labels(df):
    print("Encoding Labels...")
    le = MultiLabelBinarizer()
    df['encoded'] = le.fit_transform(df.target.tolist()).tolist()
    df = df[['text','encoded']]

    encoder = open('output/encoder.pkl', 'ab') 
    pickle.dump(le, encoder)                      
    encoder.close()
    
    return df 

# splitting the dataset and saving it 
def split_and_save(df, split_size = 0.2):
    df_train, df_test = train_test_split(df, test_size=split_size)
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    df_train.to_pickle('output/train.pkl')
    df_test.to_pickle('output/test.pkl')
    print("Preprocessed and Saved...")
    
    return True

# loading the dataframe and doing basic preprocessing
def loader(dfPath):
    df = pd.read_csv(dfPath,header = None)
    df = df.fillna('')
    
    # to get all the multi labels in one column
    columns = ['text']
    labels = []
    
    for idx in range(1, 15):
        name = 'label_' + str(idx)
        labels.append(name)
        columns.append(name)

    df.columns = columns

    df['target'] = df[labels].values.tolist()

    df['target'] = df['target'].map(remove_empty)
    df['target'] = df['target'].map(remove_space)

    df =  df[['text','target']]

    # replacing labels that are similar but have some spelling mistakes
    replace_label(df, 'advisor/agent service positive','advisoragent service positive')
    replace_label(df, 'advisor/agent service negative','advisoragent service negative')
    replace_label(df, 'tyre age/dot code negative','tyre agedot code negative')
    
    # removing noisy labels 
    df = remove_noisy_labels(df)
    # combining labels that have frequence less than 100
    df = combine_labels(df)
    # undersampling high frequency labels datapoints 
    df = undersample_labels(df, ['value for money positive'], 0.1)
    df = undersample_labels(df, ['garage service positive'], 0.2)
    df = undersample_labels(df, ['value for money positive','garage service positive'], 0.2)
    # encoding the labels for training
    df = encode_labels(df)

    return df 

if __name__ == '__main__':
    df = loader(config.INPUT_FILE)
    split_and_save(df)
    
