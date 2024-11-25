import unittest
import torch
import timeout_decorator
import numpy as np
import re
from gradescope_utils.autograder_utils.decorators import weight
from collections import defaultdict

from principles_of_autonomy.grader import get_locals, compare_iterators


def replace_punctuation(text):
    """
    Replaces all punctuation in the text with spaces.
    """
    return re.sub(r'[^\w\s]', ' ', text)

# Function to clean SMS messages


def clean_data(messages):
    cleaned_messages = [replace_punctuation(msg).lower() for msg in messages]
    return cleaned_messages


class TestPSet10(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    def test_01(self):
        disgruntled_sms_messages = clean_data([
            "Why do companies keep trusting machines to make decisions about people?",
            "Another news story about an AI system failing. When will they learn?",
            "My job application was rejected by some algorithm. It didn't even give feedback!",
            "Machine learning is just glorified guesswork with no accountability.",
            "How can we trust something we don't fully understand to run our lives?",
            "I heard an AI misdiagnosed a patient. These systems shouldn't replace doctors!",
            "Data bias means these models are only as fair as their flawed training data.",
            "Machine learning isn't intelligence; it's pattern matching with hype.",
            "Remember when AI was supposed to solve everything? Still waiting.",
            "They call it innovation, but it's just shifting blame to algorithms.",
            "Why do we keep pretending that machine learning is neutral or objective?",
            "Another app collecting my data for 'AI research.' When does it stop?",
            "A robot wrote that article? No wonder it felt so lifeless and cold.",
            "Trusting machine learning to drive cars is like gambling with people's lives.",
            "So AI is deciding who gets a loan now? What could possibly go wrong?",
            "Why do people act like 'AI-powered' is a feature and not a red flag?",
            "An algorithm rejected my friend's health insurance claim. Machines lack empathy.",
            "Machine learning is just a fancy way to say 'we're not sure why it works.'",
            "Someone explain how using AI for surveillance is a good idea. I'll wait.",
            "If machine learning is so smart, why does it need so much hand-holding?"
        ])

        sample_counts = {'about': 2,
                         'make': 1,
                         'companies': 1,
                         'do': 3,
                         'keep': 2,
                         'decisions': 1,
                         'why': 5,
                         'people': 3,
                         'trusting': 2,
                         'machines': 2,
                         'to': 6,
                         'they': 2,
                         'news': 1,
                         'system': 1,
                         'learn': 1,
                         'will': 1,
                         'another': 2,
                         'when': 3,
                         'an': 3,
                         'ai': 7,
                         'failing': 1
                         }
        get_counts = get_locals(
            self.notebook_locals, ["get_counts"])
        word_counts = get_counts(disgruntled_sms_messages)
        assert len(
            word_counts) == 161, "Method does not produce the correct number of unique words"
        for k, v in sample_counts.items():
            if word_counts[k] != v:
                raise RuntimeError(
                    f"Word {k} was miscounted in sample (should be {v} was calculated as {word_counts[k]})")

    @weight(10)
    def test_02(self):
        disgruntled_sms_messages = clean_data([
            "Why do companies keep trusting machines to make decisions about people?",
            "Another news story about an AI system failing. When will they learn?",
            "My job application was rejected by some algorithm. It didn't even give feedback!",
            "Machine learning is just glorified guesswork with no accountability.",
            "How can we trust something we don't fully understand to run our lives?",
            "I heard an AI misdiagnosed a patient. These systems shouldn't replace doctors!",
            "Data bias means these models are only as fair as their flawed training data.",
            "Machine learning isn't intelligence; it's pattern matching with hype.",
            "Remember when AI was supposed to solve everything? Still waiting.",
            "They call it innovation, but it's just shifting blame to algorithms.",
            "Why do we keep pretending that machine learning is neutral or objective?",
            "Another app collecting my data for 'AI research.' When does it stop?",
            "A robot wrote that article? No wonder it felt so lifeless and cold.",
            "Trusting machine learning to drive cars is like gambling with people's lives.",
            "So AI is deciding who gets a loan now? What could possibly go wrong?",
            "Why do people act like 'AI-powered' is a feature and not a red flag?",
            "An algorithm rejected my friend's health insurance claim. Machines lack empathy.",
            "Machine learning is just a fancy way to say 'we're not sure why it works.'",
            "Someone explain how using AI for surveillance is a good idea. I'll wait.",
            "If machine learning is so smart, why does it need so much hand-holding?"
        ])

        get_log_probabilities = get_locals(
            self.notebook_locals, ["get_log_probabilities"])

        k = 0.8
        no_word_value = np.log(
            k) - np.log(len(disgruntled_sms_messages) + 2 * k)

        log_probs = get_log_probabilities(disgruntled_sms_messages, k)
        assert isinstance(
            log_probs, defaultdict), "return object should be a default dict"
        assert np.isclose(
            no_word_value, log_probs['NoWaYThisSiSAWorD']), "Default value for unknown word is incorrect"

        one_word_value = np.log(
            1 + k) - np.log(len(disgruntled_sms_messages) + 2 * k)
        assert np.isclose(one_word_value, log_probs['decisions'])

        six_word_value = np.log(
            6 + k) - np.log(len(disgruntled_sms_messages) + 2 * k)
        assert np.isclose(six_word_value, log_probs['to'])

    @weight(10)
    def test_03(self):
        disgruntled_sms_messages = clean_data([
            "Why do companies keep trusting machines to make decisions about people?",
            "Another news story about an AI system failing. When will they learn?",
            "My job application was rejected by some algorithm. It didn't even give feedback!",
            "Machine learning is just glorified guesswork with no accountability.",
            "How can we trust something we don't fully understand to run our lives?",
            "I heard an AI misdiagnosed a patient. These systems shouldn't replace doctors!",
            "Data bias means these models are only as fair as their flawed training data.",
            "Machine learning isn't intelligence; it's pattern matching with hype.",
            "Remember when AI was supposed to solve everything? Still waiting.",
            "They call it innovation, but it's just shifting blame to algorithms.",
            "Why do we keep pretending that machine learning is neutral or objective?",
            "Another app collecting my data for 'AI research.' When does it stop?",
            "A robot wrote that article? No wonder it felt so lifeless and cold.",
            "Trusting machine learning to drive cars is like gambling with people's lives.",
            "So AI is deciding who gets a loan now? What could possibly go wrong?",
            "Why do people act like 'AI-powered' is a feature and not a red flag?",
            "An algorithm rejected my friend's health insurance claim. Machines lack empathy.",
            "Machine learning is just a fancy way to say 'we're not sure why it works.'",
            "Someone explain how using AI for surveillance is a good idea. I'll wait.",
            "If machine learning is so smart, why does it need so much hand-holding?"
        ])
        pro_ai_messages = clean_data([
            "AI is revolutionizing healthcare by helping doctors catch diseases early.",
            "Machine learning has made language translation so much more accessible!",
            "Self-driving cars could save countless lives by reducing human error.",
            "AI helps businesses run more efficiently, saving time and resources.",
            "Thanks to AI, we can analyze massive datasets and discover insights faster than ever."
        ])
        learn_distributions = get_locals(
            self.notebook_locals, ["learn_distributions"])
        k = 0.9
        log_probs, log_priors = learn_distributions(
            [disgruntled_sms_messages, pro_ai_messages], k)

        true_log_priors = [-0.2231435513142097, -1.6094379124341003]
        assert np.allclose(log_priors, true_log_priors,
                           atol=1e-6), "Calculated log priors are incorrect"
        assert len(log_probs) == 2
        assert np.all([isinstance(x, defaultdict) for x in log_probs]
                      ), "items in log_probs should be defaultdicts"

        no_word_value = np.log(
            k) - np.log(len(disgruntled_sms_messages) + 2 * k)
        assert np.isclose(
            no_word_value, log_probs[0]['NoWaYThisSiSAWorD']), "Default value for unknown word is incorrect"
        no_word_value = np.log(
            k) - np.log(len(pro_ai_messages) + 2 * k)
        assert np.isclose(
            no_word_value, log_probs[1]['NoWaYThisSiSAWorD']), "Default value for unknown word is incorrect"

        one_word_value = np.log(
            1 + k) - np.log(len(disgruntled_sms_messages) + 2 * k)
        assert np.isclose(
            one_word_value, log_probs[0]['decisions']), "log probs in spam collection are incorrect"
        one_word_value = np.log(
            1 + k) - np.log(len(pro_ai_messages) + 2 * k)
        assert np.isclose(
            one_word_value, log_probs[1]['catch']), "log probs in ham collection are incorrect"

        six_word_value = np.log(
            6 + k) - np.log(len(disgruntled_sms_messages) + 2 * k)
        assert np.isclose(
            six_word_value, log_probs[0]['to']), "log probs in spam collection are incorrect"
        three_word_value = np.log(
            3 + k) - np.log(len(pro_ai_messages) + 2 * k)
        assert np.isclose(
            three_word_value, log_probs[1]['ai']), "log probs in ham collection are incorrect"

    @weight(10)
    def test_04(self):
        professor_messages = clean_data([
            "Let's analyze this dataset using statistical methods.",
            "Remember to cite your sources in APA format.",
            "Office hours are tomorrow if you have questions about the assignment.",
            "This theory revolutionized the way we understand human behavior.",
            "Have you completed the readings for next week's lecture?",
            "We'll need to verify these results with a peer-reviewed study.",
            "Please submit your papers by the deadline for full credit.",
            "Today's topic is the intersection of ethics and machine learning.",
            "You should critically evaluate all information you encounter.",
            "Don't forget, your research proposal is due next Monday."
        ])
        snowboarder_messages = clean_data([
            "That drop was insane—I thought I was gonna fly!",
            "Fresh powder on the slopes today, let's hit it early!",
            "Dude, did you see that backflip? Totally nailed it!",
            "The steeper, the better—who wants to cruise anyway?",
            "I'm chasing that rush you only get at the edge of control.",
            "That was a gnarly crash, but I'm going back up!",
            "The snow conditions are perfect for some cliff jumps.",
            "Carving through that tree line felt like surfing a wave!",
            "Let's find the highest peak and shred down from there.",
            "Avalanche warnings? Adds to the thrill, let's go!"
        ])
        classify_message, learn_distributions = get_locals(
            self.notebook_locals, ["classify_message", "learn_distributions"])
        k = 0.9
        learned_values = learn_distributions(
            [professor_messages, snowboarder_messages], k)
        # is_snowboarder = [True, False, True, False,
        #                   True, False, True, False, True, False]
        # mixed_test_set = clean_data([
        #     "The steeper, the better—who wants to cruise anyway?",  # Snowboarder
        #     "Today's topic is the intersection of ethics and machine learning.",  # Professor
        #     "Fresh powder on the slopes today, let's hit it early!",  # Snowboarder
        #     "Office hours are tomorrow if you have questions about the assignment.",  # Professor
        #     "Avalanche warnings? Adds to the thrill, let's go!",  # Snowboarder
        #     "You should critically evaluate all information you encounter.",  # Professor
        #     "That drop was insane—I thought I was gonna fly!",  # Snowboarder
        #     "Have you completed the readings for next week's lecture?",  # Professor
        #     "Let's find the highest peak and shred down from there.",  # Snowboarder
        #     "Please submit your papers by the deadline for full credit."  # Professor
        # ])
        mixed_test_set = clean_data([
            "We need to validate the hypothesis before drawing conclusions.",  # Professor
            "The untouched snow at the summit is calling my name!",  # Snowboarder
            "That double cork spin was unreal—I need to try it next!",  # Snowboarder
            "Don't forget to review the supplemental materials for deeper context.",  # Professor
            "I live for the feeling of freefall before the board catches the slope.",  # Snowboarder
            "The methodology section should clearly outline your experiment.",  # Professor
            "That run was pure adrenaline—I'm going up for another one!",  # Snowboarder
            "Your thesis needs a stronger argument to support this claim.",  # Professor
            "That ridge looks dangerous, but it'll be worth the rush.",  # Snowboarder
            "Collaborating on this research could lead to groundbreaking results."  # Professor
        ])
        is_snowboarder = [False, True, True, False,
                          True, False, True, False, True, False]

        for message, label in zip(mixed_test_set, is_snowboarder):
            student_ans = classify_message(message, *learned_values)
            # print(student_ans, label)
            if (student_ans == "spam" and label) or (student_ans == "ham" and not label):
                raise AssertionError(
                    "Classifier failed on test set where it should not ")

    @weight(5)
    def test_05(self):
        sigmoid, binary_cross_entropy, loss_gradient = get_locals(
            self.notebook_locals, ["sigmoid",
                                   "binary_cross_entropy", "loss_gradient"]
        )

        # sigmoid
        z = np.asarray([-5.232, 5785.23, 12.3, 9.99, 0, 0.0234])
        out = sigmoid(z)
        ans = [0.00531443,  1., 0.99999545, 0.99995415, 0.5, 0.50584973]
        assert np.allclose(out, ans)

        # bce
        y_true = np.asarray([0, 1, 1, 0, 1, 0, 0, 1])
        y_pred = np.asarray([0, 0.88, 1.0, 0.23, 0.999, 0.23, 0.58, 0.0001])
        out = binary_cross_entropy(y_true, y_pred)
        ans = 1.3411755424741487
        assert np.isclose(out, ans, atol=1e-6)

        # loss gradient
        train = np.asarray([[3.11793866e-01,  8.42014665e-01,  1.00000000e+00],
                            [1.18118168e+00, -1.84907492e+00,  1.00000000e+00],
                            [-6.64466581e-04,  5.35220735e-01,  1.00000000e+00],
                            [8.50135497e-01, -9.23584957e-01,  1.00000000e+00],
                            [2.11500691e+00, -1.42619726e+00,  1.00000000e+00],
                            [-1.00135059e-01,  5.67273652e-01,  1.00000000e+00],
                            [3.39257626e+00,  2.57794949e+00,  1.00000000e+00],
                            [1.53617010e+00,  1.48573267e+00,  1.00000000e+00],
                            [1.95195600e+00,  1.74980414e+00,  1.00000000e+00],
                            [1.44611645e+00,  1.40569959e+00,  1.00000000e+00],
                            [3.31511183e+00, -2.82957350e+00,  1.00000000e+00],
                            [2.94112842e+00, -3.66711592e+00,  1.00000000e+00],
                            [2.37456561e+00, -2.77239287e+00,  1.00000000e+00],
                            [1.67545039e+00,  1.61211627e+00,  1.00000000e+00],
                            [2.09837866e+00, -2.75413880e+00,  1.00000000e+00],
                            [1.48508593e+00, -1.29458688e+00,  1.00000000e+00],
                            [1.88190832e+00,  1.77526195e+00,  1.00000000e+00],
                            [2.01913253e+00,  1.85649457e+00,  1.00000000e+00],
                            [7.53271912e-01, -4.38671179e-01,  1.00000000e+00],
                            [2.54264126e+00,  2.24358113e+00,  1.00000000e+00]])
        labels = np.asarray([0, 1, 1, 0, 0, 1, 1, 1, 1, 1,
                            0, 0, 0, 1, 0, 0, 1, 1, 0, 1])
        y_pred = np.asarray([0.83537171, 0.96479656, 0.12684591, 0.52331132, 0.85061104, 0.53760925,
                            0.73004648, 0.78545965, 0.83659554, 0.66653243, 0.89773919, 0.64779123,
                            0.94463739, 0.62929838, 0.87242862, 0.59565536, 0.34434898, 0.81345749,
                            0.43238103, 0.12654566])

        ans = np.asarray([0.30880138, -0.9196375, 0.10807316])

        student_ans = loss_gradient(train, labels, y_pred)
        assert np.allclose(
            student_ans, ans, atol=1e-4), "Calculated weights are incorrect"

    @weight(25)
    def test_06(self):
        # all of the gradient descent steps
        batch_gradient_descent, stochastic_gradient_descent, mini_batch_gradient_descent = get_locals(
            self.notebook_locals, ["batch_gradient_descent", "stochastic_gradient_descent", "mini_batch_gradient_descent"])
        raise NotImplementedError()

    @ weight(5)
    def test_07(self):
        SimpleNN = get_locals(self.notebook_locals, ["SimpleNN"])
        inp = (12, 7, 43, 18)
        nn = SimpleNN(
            *inp
        )
        fake_input = torch.zeros(26, inp[0])
        out = nn(fake_input)
        n_params = np.sum([p.numel() for p in nn.parameters()])
        true_n_params = inp[0] * inp[1] + inp[1] + \
            inp[1] * inp[2] + inp[2] + inp[2] * inp[3] + inp[3]
        assert np.allclose(
            n_params, true_n_params), "Number of model parameters is incorrect (should be input_dim * output_dim + output_dim for each linear layer)"
        assert out.shape == (26, inp[-1])

    @ weight(10)
    def test_08(self):
        ans = ('a', 'b')
        mc_answer_set_1 = get_locals(self.notebook_locals, ["mc_answer_set_1"])
        assert compare_iterators(ans, mc_answer_set_1)

    @ weight(5)
    def test_09(self):
        early_stopping = get_locals(self.notebook_locals, ["early_stopping"])

        # test best_val_update loss and counter work
        val_loss = 10.0
        best_val_loss = 12.0
        counter = 7
        patience = 10
        post_best_val_loss, post_counter, stop_training = early_stopping(
            val_loss,
            best_val_loss,
            counter,
            patience
        )
        assert not stop_training, "Training should continue when patience is not exceeded"
        assert post_counter == 0, "Post counter should reset when better val loss is seen"
        assert np.allclose(post_best_val_loss,
                           val_loss), "Best val loss needs to be updated"

        # test counter expiration works
        val_loss = 13.0
        best_val_loss = 12.0
        counter = 10
        patience = 10
        post_best_val_loss, post_counter, stop_training = early_stopping(
            val_loss,
            best_val_loss,
            counter,
            patience
        )
        assert stop_training, "Training should stop when patience is exceeded (<=)"
        assert np.allclose(
            post_best_val_loss, best_val_loss), "Best val loss should be maintained if not exceeded"

    @ weight(10)
    def test_10(self):
        ans = dict(
            q1=dict(a=True, b=True, c=False),
            q2=dict(a=True, b=False, c=False),
            q3=dict(a=True, b=True, c=False),
            q4=dict(a=True, b=True, c=False, d=False),
            q5=dict(a=True, b=True, c=False),
        )
        mc_answer_set_2 = get_locals(self.notebook_locals, ["mc_answer_set_2"])
        for k, v in ans.items():
            student_ans_for_q = mc_answer_set_2[k]
            for part, part_ans in v.items():
                assert student_ans_for_q[part] == part_ans

    @ weight(5)
    @ timeout_decorator.timeout(1.0)
    def test_11(self):
        word = get_locals(self.notebook_locals, ['form_confirmation_word'])
        password_hash = hash("ParametersAllTheWayDown".lower())
        if hash(word.strip().lower()) == password_hash:
            return
        else:
            raise RuntimeError(f"Incorrect form word {word}")
