import pandas as pd


def preprocess():
    df = pd.read_csv('data/ner_data.csv', encoding="latin1").fillna(method="ffill")

    df.rename(columns = {'Sentence #' : 'Sentence'}, inplace = True)
    df['Sentence'] = df.Sentence.apply(lambda x : x.split(' ')[1])
    df['Sentence'] = df.Sentence.astype(int)
    df_train = df.loc[df.Sentence < int(df.Sentence.nunique()*0.8)]

    word = [' '.join(df_train.loc[df_train.Sentence == i].Word.values.tolist()) for i in range(df_train.Sentence.nunique())]
    tag = [' '.join(df_train.loc[df_train.Sentence == i].Tag.values.tolist()) for i in range(df_train.Sentence.nunique())]
    
    df_train = pd.DataFrame({'text' : word, 'labels' : tag}).iloc[1:,:].reset_index(drop = True)
    df_train.to_csv('data/preprocessed.csv', index = False)



if __name__ == '__main__':
    preprocess()
