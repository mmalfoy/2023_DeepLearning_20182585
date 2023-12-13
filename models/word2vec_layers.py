import numpy as np

# Affine 계층
def affine_forward(input_matrix, weight_matrix):
    return np.dot(input_matrix, weight_matrix)

def affine_backward(input_matrix, weight_matrix, gradient):
    grad_input = np.dot(gradient, weight_matrix.T)
    grad_weight = np.dot(input_matrix.T, gradient)
    return grad_input, grad_weight

# 활성화 함수 (시그모이드 함수)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Word2Vec 모델 정의
class Word2Vec:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.rand(vocab_size, embedding_dim)

    def forward(self, target_index, context_index):
        target_embedding = self.embedding_matrix[target_index]
        context_embedding = self.embedding_matrix[context_index]
        prediction = np.dot(target_embedding, context_embedding)
        return prediction

    def backward(self, target_index, context_index, gradient, learning_rate):
        """역전파 메소드 수정: 학습률 인자 추가"""
        target_embedding = self.embedding_matrix[target_index]
        context_embedding = self.embedding_matrix[context_index]
        grad_target, grad_context = affine_backward(target_embedding, context_embedding, gradient)
        self.embedding_matrix[target_index] -= learning_rate * grad_target
        self.embedding_matrix[context_index] -= learning_rate * grad_context

    def compute_loss(self, prediction, label):
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        loss = -label * np.log(prediction) - (1 - label) * np.log(1 - prediction)
        return loss