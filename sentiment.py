import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random
random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.linear_model import LogisticRegression
nltk.download("stopwords")          # Download the stop words from nltk


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data>S <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])



def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)
    
    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)



def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir+"train-pos.txt", "r") as f:
        for i,line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_pos.append(words)
    with open(path_to_dir+"train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            train_neg.append(words)
    with open(path_to_dir+"test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_pos.append(words)
    with open(path_to_dir+"test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w)>=3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg



def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))
    
    dictpos={}
    dictneg={}
    
    for line in train_pos:
        word_occured=[]
        for word in line:
            if word not in stopwords and word not in word_occured:
                word_occured.append(word)
                if word in dictpos:
                    dictpos[word]+=1
                else:
                    dictpos[word]=1
                   
    
    for line in train_neg:
        word_occured=[]
        for word in line:
            if word not in stopwords and word not in word_occured:
                word_occured.append(word)
                if word in dictneg:
                    dictneg[word]+=1
                else:
                    dictneg[word]=1
      
   
    
    pos_1_pct=0.01*len(train_pos)
    neg_1_pct=0.01*len(train_neg)
    
    feature=[]
    
    for word in dictpos:
        if dictpos[word]>=pos_1_pct:
            if (word not in dictneg) or (word in dictneg and dictpos[word]>=2*dictneg[word]):
                feature.append(word)
    
    for word in dictneg:
        if dictneg[word]>=neg_1_pct:
            if (word not in dictpos) or (word in dictpos and dictneg[word]>=2*dictpos[word]):
                feature.append(word)
    
    
    
    train_pos_vec=[]
    for line in train_pos:
        temp=[]
        for word in feature:
            if word in line:
                temp.append(1)
            else:
                temp.append(0)
        train_pos_vec.append(temp)
    
    train_neg_vec=[]
    for line in train_neg:
        temp=[]
        for word in feature:
            if word in line:
                temp.append(1)
            else:
                temp.append(0)
        train_neg_vec.append(temp)
    
    test_pos_vec=[]
    for line in test_pos:
        temp=[]
        for word in feature:
            if word in line:
                temp.append(1)
            else:
                temp.append(0)
        test_pos_vec.append(temp)
        
    test_neg_vec=[]
    for line in test_neg:
        temp=[]
        for word in feature:
            if word in line:
                temp.append(1)
            else:
                temp.append(0)
        test_neg_vec.append(temp)
            
    
    # Determine a list of words that will be used as features. 
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    
    labeled_train_pos=[]
    for i, line in enumerate(train_pos):
        labeled_train_pos.append(LabeledSentence(words=line, tags = ['TRAIN_POS_%s' %i]))
    
    labeled_train_neg=[]
    for i, line in enumerate(train_neg):
        labeled_train_neg.append(LabeledSentence(words=line, tags=['TRAIN_NEG_%s' %i]))
      
    labeled_test_pos=[]
    for i, line in enumerate(test_pos):
        labeled_test_pos.append(LabeledSentence(words=line, tags=['TEST_POS_%s' %i]))
     
    labeled_test_neg=[]
    for i, line in enumerate(test_neg):
        labeled_test_neg.append(LabeledSentence(words=line, tags=['TEST_NEG_%s' %i]))
        
    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run 
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    
    train_pos_vec=[]
    for i, line in enumerate(labeled_train_pos):
        docvec=model.docvecs['TRAIN_POS_%s' %i]
        train_pos_vec.append(docvec)
    
    train_neg_vec=[]
    for i, line in enumerate(labeled_train_neg):
        docvec=model.docvecs['TRAIN_NEG_%s' %i]
        train_neg_vec.append(docvec)
        
    test_pos_vec=[]
    for i, line in enumerate(labeled_test_pos):
        docvec=model.docvecs['TEST_POS_%s' %i]
        test_pos_vec.append(docvec)
        
    test_neg_vec=[]
    for i, line in enumerate(labeled_test_neg):
        docvec=model.docvecs['TEST_NEG_%s' %i]
        test_neg_vec.append(docvec)
        
    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec



def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X= train_pos_vec+train_neg_vec
    
    

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    
    nb_model=BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(X, Y)
    
    lr_model=LogisticRegression()
    lr_model.fit(X,Y)
    
    
    return nb_model, lr_model



def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"]*len(train_pos_vec) + ["neg"]*len(train_neg_vec)
    X= train_pos_vec+train_neg_vec
    

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    nb_model=GaussianNB()
    nb_model.fit(X, Y)
    
    lr_model=LogisticRegression()
    lr_model.fit(X,Y)
    
    return nb_model, lr_model



def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE
    
    X= test_pos_vec+test_neg_vec
    index_pos=len(test_pos_vec)
    
    predicted=model.predict(X)
    
    tp=0
    tn=0
    fp=0
    fn=0
    
    for i, label in enumerate(predicted):
        if i<=(index_pos-1) and label=='pos':
            tp+=1
        elif i<=(index_pos-1) and label=='neg':
            fn+=1
        elif i>(index_pos-1) and label=='pos':
            fp+=1
        else:
            tn+=1
    
    accuracy=(tp+tn)/float(tp+tn+fp+fn)
        
    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)



if __name__ == "__main__":
    main()
