import argparse
from ppo import PPO
from network import MLP
from dm_control import suite

# hparams.json 은 hyperparameter를 조정할 수 있는 file이고,
# ppo.py 의 init에서 불러온다. (실험하면서 조정해야 하는 값이다.)

# 사용하려는 state는 다음과 같다. (센서값)
# [ax1, ay1, az1, adc1, pos, ax2, ay2, az2, adc2] (9-D)
# 사용하려는 action은 다음과 같다. (모터값)
# [motor1, motor2]



parser = argparse.ArgumentParser()
parser.add_argument('-mode', choices=['train', 'test'], type=str)
args = parser.parse_args()

def train(env):
    print('Training mode..')
    
    # 다른 실험 설정으로 실험하려면 hparams.json 수정
    ppo_model = PPO(env)
    
    ppo_model.learn(200_000_000) # 200,000,000 timestep 동안 training 
    # 이 timestep도 hyperparameter이므로 바꿔도 무방하고,
    # 충분히 training이 되었다고 판단하면 중간에 그만 두어도 좋다.
    # (어차피 10 episode마다 모델이 저장되므로)

def test(env):
    print('Testing mode..')
    
    # 새로운 MLP 를 만든다.
    actor = MLP(8, 2) # 입력차원 8 (state), 출력차원 2 (action)인 MLP
    
    try:
        actor.load_weights('./weights/actor')
    except Exception as _e:
        raise FileNotFoundError
    
    
    

def main():
    
    global args
    
    env = suite.load(domain_name='reindeer', task_name='walk')
    
    if args.mode == 'train':
        train(env)
    else:
        test(env)
    

if __name__ == "__main__":
    main()
