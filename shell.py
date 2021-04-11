
from cmd import Cmd
from process import *
from joblib import dump, load
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import time

TRAIN_DIR = 'data/train-set'
TEST_DIR = 'data/test-set'

class MyPrompt(Cmd):
    prompt = 'spam_filter> '
    intro = "Welcome! Type ? to list commands"

    def do_exit(self, inp):
        '''exit the application. Shorthand: x q.'''
        print("Bye")
        return True

    def do_train(self, inp):
        '''train the machine learning model. Shorthand: a'''
        start = time.process_time()
        # Create a dictionary of words with its frequency
        create_dict(TRAIN_DIR)
        # Prepare feature vectors per training mail and its labels
        train_labels = np.zeros(700)
        train_labels[350:] = 1
        train_matrix = extract_features(TRAIN_DIR)

        # Training Naive
        model = MultinomialNB()
        model.fit(train_matrix, train_labels)
        dump(model, "model.joblib")
        print("Time taken to train model: " + str(time.process_time() - start) + "secs")

    def do_test(self, inp):
        '''test the machine learning model. Shorthand: b'''
        start = time.process_time()
        model = load("model.joblib")
        test_matrix = extract_features(TEST_DIR)
        test_labels = np.zeros(260)
        test_labels[130:] = 1

        result = model.predict(test_matrix)
        cm = confusion_matrix(test_labels, result)
        print(f'tn: {cm[0, 0]}, fp: {cm[0, 1]}\nfn: {cm[1, 0]}, tp: {cm[1, 1]}')
        print(classification_report(test_labels,result))
        print(accuracy_score(test_labels,result))

        print("Time taken to test model: " + str(time.process_time() - start))

    # def do_test_single(self, inp):
    #     '''test the machine learning model on a single test input. Shorthand: c'''
    #     start = time.process_time()
    #     model = load("model.joblib")
    #     email = input("Enter your test email:")
    #     print(email)
    #     matrix = extract_text_features(email)
    #     result = model.predict(matrix)
    #     print(result)

    def default(self, inp):
        if inp == 'x' or inp == 'q':
            return self.do_exit(inp)
        if inp == 'a':
            return self.do_train(inp)
        if inp == 'b':
            return self.do_test(inp)

MyPrompt().cmdloop()