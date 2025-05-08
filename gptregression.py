import math
from models.linreg.linreg import linreg

if __name__ == '__main__':
    # data of log_10 of parameters count in millions to gpt version number
    data = [
        [math.log(117, 10), 1],
        [math.log(1500, 10), 2],
        [math.log(175000, 10), 3],
        [math.log(1760000, 10), 4],
        [math.log(12000000, 10), 4.5],
    ]

    coefs = linreg(data=data)

    # parameter count to gpt version number
    def param_count_to_gpt_num(params):
        return -0.334 + 0.679 * math.log(params/(10**6), 10)

    params = 10**9
    version = param_count_to_gpt_num(params=params)
    print(f'My language model has {params} parameters, which is analagous to GPT-{version:.3f}')
