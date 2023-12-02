import os
import time

from corpus_utils import read_corpus, move_empty_files
from nlp_utils import preprocessing, preprocessing_v2, no_spacing
from collections import Counter, OrderedDict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.base import clone
from evaluation_utils import compute_evaluation_measures, compute_means_std_eval_measures


if __name__ == '__main__': #garante que o código abaixo só será executado se o script for executado diretamente (não quando importado como um módulo).

    corpus_path = r'C:\Users\jessi\OneDrive\Documentos\UFV\UFV 2023-2\SIN 323 - IA\Trabalho\resume_corpus_master\resumes_corpus' #caminho para o corpus que será usado.

    vectorizer_options = ['tf_idf'] # opções para o vetorizador que será usado

    n_splits = 5 #número de divisões para a validação cruzada. 

    n_total = -1 #número total de exemplos a serem usados. Se for -1, todos os exemplos serão usados.

    max_features = 5000 #define o número máximo de recursos a serem usados pelo vetorizador.
   
    print('\nLoading Corpus\n') # mensagem indicando que o corpus está sendo carregado.

    corpus_df = read_corpus(corpus_path, num_examples=n_total) # lê o corpus do caminho especificado e armazena-o em um DataFrame

    print('\nPreProcessing Corpus\n') # mensagem indicando que o corpus está sendo pré-processado.

    corpus_df['resume_norm'] = corpus_df['resume'].apply(lambda t: preprocessing_v2(t)).astype(str) # função de pré-processamento a cada currículo no DataFrame e armazena o resultado em uma nova coluna
    corpus_df['label_unique'] = corpus_df['label'].apply(lambda l: l[0]).astype(str) # extrai a primeira etiqueta de cada lista de etiquetas no DataFrame e armazena o resultado em uma nova coluna.
    corpus_df['no_spacing'] = corpus_df['resume_norm'].apply(lambda t: no_spacing(t)).astype(str) #  remove os espaços de cada currículo no DataFrame e armazena o resultado em uma nova coluna.

    corpus_df_unique = corpus_df.drop_duplicates(subset='no_spacing') # remove as duplicatas do DataFrame com base na coluna ‘no_spacing’ e armazena o resultado em um novo DataFrame

    corpus_df_unique['resume_nlp'] = corpus_df_unique['resume'].apply(lambda t: preprocessing(t)).astype(str) #  função de pré-processamento a cada currículo no novo DataFrame e armazena o resultado em uma nova coluna

    resumes = corpus_df_unique['resume_nlp'].values # extrai os valores da coluna ‘resume_nlp’ e os armazena em uma variável
    labels = corpus_df_unique['label_unique'].values # extrai os valores da coluna ‘label_unique’ e os armazena em uma variável

    print(f'\nCorpus: {len(resumes)} -- {len(labels)}')  # número de currículos e etiquetas.

    print('\nExample:') # mensagem indicando que um exemplo será mostrado
    print(f'  Resume: {resumes[-1]}') #  imprime o último currículo
    print(f'  Label: {labels[-1]}') # mprime a última etiqueta

    counter_labels = Counter(labels) # número de ocorrências de cada etiqueta

    labels_distribution = OrderedDict(sorted(counter_labels.items())) # ordena as etiquetas e suas contagens em uma ordem específica e as armazena em um dicionário ordenado

    print(f'\nLabels distribution: {labels_distribution}') # distribuição das etiquetas
 
    for vectorizer_opt in vectorizer_options: #  loop para cada opção de vetorizador

        print(f'\n\nVectorizer: {vectorizer_opt}\n\n') # opção de vetorizador atual

        results_dir = f'../results/ml/{vectorizer_opt}' # define o diretório onde os resultados serão salvos

        os.makedirs(results_dir, exist_ok=True) # cria o diretório de resultados se ele não existir

        vectorizer = None #  inicializa a variável ‘vectorizer’ como None

        if vectorizer_opt == 'tf_idf': # verifica se a opção de vetorizador atual é ‘tf_idf’
            vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=max_features) # Se a opção de vetorizador for ‘tf_idf’, cria um vetorizador TF-IDF com o número máximo de recursos especificado
        elif vectorizer_opt == 'count': # verifica se a opção de vetorizador atual é ‘count’
            vectorizer = CountVectorizer(ngram_range=(1, 1), binary=False, max_features=max_features) # Se a opção de vetorizador for ‘count’, cria um vetorizador de contagem com o número máximo de recursos especificado
        else:
            vectorizer = CountVectorizer(ngram_range=(1, 1), binary=True, max_features=max_features) # Se a opção de vetorizador não for ‘tf_idf’ nem ‘count’, cria um vetorizador de contagem binária com o número máximo de recursos especificado

        label_encoder = LabelEncoder() # cria um codificador de etiquetas

        print(f'\nVectorizer Option: {vectorizer_opt}') # opção de vetorizador atual

        y_labels = label_encoder.fit_transform(labels) # ajusta o codificador de etiquetas aos rótulos e transforma os rótulos em números

        print(f'\nLabels Mappings: {label_encoder.classes_}') # classes do codificador de etiquetas

        classifiers = { # dicionário de classificadores que serão usados
            'logistic_regression': LogisticRegression(class_weight='balanced', max_iter=500),
            'knn': KNeighborsClassifier(),
            'decision_tree': DecisionTreeClassifier(class_weight='balanced'),
            'random_forest': RandomForestClassifier(class_weight='balanced'),
            'extra_trees_classifier': ExtraTreesClassifier(class_weight='balanced'),
            'xgboost': XGBClassifier(),
            'lgbm': LGBMClassifier(class_weight='balanced'),
            'svc': SVC(class_weight='balanced'),
            'cat_boost_classifier': CatBoostClassifier(verbose=False),
            'mlp_classifier': MLPClassifier()
        }

        print('\n\n------------Evaluations------------\n') # avaliações estão começando

        for clf_name, clf_base in classifiers.items(): # loop para cada classificador

            results_dict = { #  dicionário para armazenar os resultados
                'all_accuracy': [],
                'all_macro_avg_p': [],
                'all_macro_avg_r': [],
                'all_macro_avg_f1': [],
                'all_weighted_avg_p': [],
                'all_weighted_avg_r': [],
                'all_weighted_avg_f1': []
            }

            print(f'\n  Classifier: {clf_name}') # nome do classificador atual

            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) # cria um objeto de validação cruzada estratificada

            all_y_test = [] # inicializa uma lista para armazenar todos os rótulos de teste
            all_y_pred = [] # inicializa uma lista para armazenar todas as previsões

            for k, (train_idx, test_idx) in enumerate(skf.split(resumes, y_labels)): # loop para cada divisão de treino/teste

                classifier = clone(clf_base) # cria uma cópia do classificador base

                X_train = [resume for i, resume in enumerate(resumes) if i in train_idx] # cria o conjunto de treinamento
                X_test = [resume for i, resume in enumerate(resumes) if i in test_idx] # cria o conjunto de teste

                y_train = y_labels[train_idx] # cria os rótulos de treinamento
                y_test = y_labels[test_idx] #  cria os rótulos de teste

                X_train = vectorizer.fit_transform(X_train).toarray() # ajusta o vetorizador ao conjunto de treinamento e transforma o conjunto de treinamento em uma matriz esparsa
                X_test = vectorizer.transform(X_test).toarray() # transforma o conjunto de teste em uma matriz esparsa usando o vetorizador que foi ajustado ao conjunto de treinamento

                X_train, _, y_train, _ = train_test_split(
                    X_train, y_train, test_size=0.1, stratify=y_train, shuffle=True,
                    random_state=42) # divide o conjunto de treinamento e os rótulos de treinamento em conjuntos de treinamento e validação

                print(f'\n    Folder {k + 1} - {len(X_train)} - {len(X_test)}') # número do fold atual (k + 1) e os tamanhos dos conjuntos de treinamento e teste

                classifier.fit(X_train, y_train) # ajusta o classificador ao conjunto de treinamento

                y_pred = classifier.predict(X_test) # usa o classificador ajustado para fazer previsões no conjunto de teste

                all_y_test.extend(y_test) # adiciona os rótulos de teste à lista de todos os rótulos de teste
                all_y_pred.extend(y_pred) # adiciona as previsões à lista de todas as previsões

                compute_evaluation_measures(y_test, y_pred, results_dict) # calcula várias medidas de avaliação (como precisão, recall, F1 score) para as previsões e atualiza o dicionário de resultados

            compute_means_std_eval_measures(clf_name, all_y_test, all_y_pred, results_dict, results_dir) # calcula a média e o desvio padrão das medidas de avaliação para todos os folds e salva os resultados no diretório de resultados

        time.sleep(60) #  pausa de 60 segundos antes de continuar com o próximo loop
