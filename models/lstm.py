import numpy as np


def sigmoid(x):
    x = np.clip(x, -20, 20)  # x의 값을 -20과 20 사이로 제한
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


class LSTM:
    def __init__(self, Wx, Wh, b):
        '''

        Parameters
        ----------
        Wx: 입력 x에 대한 가중치 매개변수(4개분의 가중치가 담겨 있음)
        Wh: 은닉 상태 h에 대한 가장추 매개변수(4개분의 가중치가 담겨 있음)
        b: 편향（4개분의 편향이 담겨 있음）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape

        # 각 시퀀스 스텝에 대해 순전파 수행
        for t in range(x.shape[1]):
            xt = x[:, t, :]  # 현재 시퀀스 스텝의 입력
            combined = np.hstack((h_prev, xt))  # 현재 은닉 상태와 현재 입력 결합

            A = np.dot(combined, Wx.T) + np.dot(h_prev, Wh.T) + b

            f = sigmoid(A[:, :H])
            g = np.tanh(A[:, H:2 * H])
            i = sigmoid(A[:, 2 * H:3 * H])
            o = sigmoid(A[:, 3 * H:])

            c_next = f * c_prev + g * i
            h_next = o * np.tanh(c_next)

            # 다음 시퀀스 스텝을 위해 은닉 상태와 셀 상태 업데이트
            h_prev = h_next
            c_prev = c_next

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        # 초기 기울기
        dWx_total, dWh_total, db_total = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)
        print(f"dWx_total shape: {dWx_total.shape}")  # dWx_total의 형태 출력
        print(f"dWh_total shape: {dWh_total.shape}")  # dWh_total의 형태 출력
        print(f"db_total shape: {db_total.shape}")  # db_total의 형태 출력

        # 각 시퀀스 스텝에 대해 역전파 수행
        for t in reversed(range(x.shape[1])):
            xt = x[:, t, :]
            print(f"xt shape: {xt.shape}")  # xt의 형태 출력

            tanh_c_next = np.tanh(c_next)

            ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)


            di = ds * g
            df = ds * c_prev
            do = dh_next * tanh_c_next
            dg = ds * i

            di *= i * (1 - i)
            df *= f * (1 - f)
            do *= o * (1 - o)
            dg *= (1 - g ** 2)

            # dA를 2차원 배열로 변경
            dA = np.hstack([df, dg, di, do])  # dA의 크기가 (batch_size, 4 * hidden_dim)이 되도록 변경
            print(f"dA shape: {dA.shape}")  # dA의 형태 출력
            # 각 시퀀스 스텝에 대한 기울기 계산
            dWx = np.dot(xt.T, dA)
            print(f"dWx shape: {dWx.shape}")  # dWx의 형태 출력
            dWh = np.dot(h_prev.T, dA)
            print(f"dWh shape: {dWh.shape}")  # dWh의 형태 출력
            db = dA.sum(axis=0)
            print(f"db shape: {db.shape}")  # db의 형태 출력

            dx = np.dot(dA, Wx)
            print(f"xt shape: {xt.shape}")  # xt의 형태 출력
            dh_prev = np.dot(dA, Wh)
            dc_prev = f * ds  # 이전 셀 상태에 대한 기울기

            # 기울기 누적

            dWx_total += dWx
            print(f"dWx_total after update shape: {dWx_total.shape}")  # dWx_total 업데이트 후 형태 출력

            dWh_total += dWh
            print(f"dWh_total after update shape: {dWh_total.shape}")  # dWh_total 업데이트 후 형태 출력

            db_total += db
            print(f"db_total after update shape: {db_total.shape}")  # db_total 업데이트 후 형태 출력


        self.grads[0][...] = dWx_total
        self.grads[1][...] = dWh_total
        self.grads[2][...] = db_total

        return dx, dh_prev, dc_prev

    def reset_grads(self):
        # 모든 그라디언트를 0으로 초기화
        for grad in self.grads:
            grad.fill(0)
