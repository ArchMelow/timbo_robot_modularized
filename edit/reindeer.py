'''
Reindeer domain in dm_control suite
Designed by Jaejin Lee
'''

import collections

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
import numpy as np

_DEFAULT_TIME_LIMIT = 3000
_CONTROL_TIMESTEP = .04

# global constants
_WALK_DIST_RANGE = 100 # actually 1
_ANGLE_DEVIATION = 15 # in degrees


SUITE = containers.TaggedTasks()

def get_model_and_assets():
  return common.read_model('reindeer.xml'), common.ASSETS

@SUITE.add()
def running(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
  physics = Physics.from_xml_string(*get_model_and_assets())
  task = Run()
  environment_kwargs = environment_kwargs or {}
  return control.Environment(
      physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
      **environment_kwargs)
  
@SUITE.add()
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Returns the 'walk' task"""
    physics = Physics.from_xml_string(*get_model_and_assets())
    task = Walk()
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics, task, time_limit=time_limit, control_timestep=_CONTROL_TIMESTEP,
        **environment_kwargs)
  

"""
inherits mujoco.Physics class
you can define additional physics here.
"""
class Physics(mujoco.Physics):
    
    def __init__(self, data):
        # init Physics instance with the base.Task __init__()
        super(Physics, self).__init__(data)
        # we need to know how much distance the agent has advanced forward
        self._prev_pos = self.named.data.xpos['modules', 'y']
        print(f'_PREV_POS init : {self._prev_pos}')
    
    # how module is standing upright w.r.t z-axis
    # 1 means same dir to the z-axis (global), -1 means opposite dir
    # returns : float in range [-1, 1] (dot product)
    def get_upright_angle(self):
        # [0 0 1] dot (frame matrix's z-axis) 
        return self.named.data.xmat['modules', 'zz']
    
    # packs imu data into dict. 
    # this will be input to the neural net
    def get_imu_data(self):
        return {'head' : 
                    {'ax' : self.named.data.sensordata[0],
                     'ay' : self.named.data.sensordata[1],
                     'az' : self.named.data.sensordata[2]
                     },
                'tail' :
                    {'ax' : self.named.data.sensordata[0],
                     'ay' : self.named.data.sensordata[1],
                     'az' : self.named.data.sensordata[2]
                     }
                }
    
    # 앞으로 가는 경우 modules의 y값이 감소 (리턴값 음수)
    # 뒤로 가는 경우 리턴값 양수
    # 가만히 있는 경우 0
    def get_distance(self):
        _cur_pos = self.named.data.xpos['modules', 'y']
        _cur_dist = _cur_pos - self._prev_pos
        self._prev_pos = _cur_pos # y좌표 업데이트
        return _cur_dist
    
    
def get_common_obs(physics):
    assert isinstance(physics, Physics)
    obs = collections.OrderedDict()
    obs['upright_angle'] = physics.get_upright_angle()
    obs['imu'] = physics.get_imu_data()
    obs['distance'] = physics.get_distance() # 얼마나 움직였는지
    return obs

# 각 센서값이 가지는 값의 범위를 잘 보고,
# 실제 활용에서 이 범위에 맞춰주어야 한다.
def return_obs_as_vector(physics):
    assert isinstance(physics, Physics)
    #print(physics.get_imu_data()['head'].values())
    ax1, ay1, az1 = list(physics.get_imu_data()['head'].values()) # 단위 g (9.8m/s^2)
    adc1 = physics.named.data.qpos['head_motor'] # [-1, 1] (실수 범위) -> -80 ~ 80도로 회전범위가 제한되어 있음.
    #d = physics.named.data['xpos','modules'] # 나중에 모델의 입력으로 들어갈 때, 오히려 방해가 될 수 있다.
    ax2, ay2, az2 = list(physics.get_imu_data()['tail'].values())
    adc2 = physics.named.data.qpos['tail_motor'] # [-1, 1] (실수 범위) -> -80 ~ 80도로 회전범위가 제한되어 있음.
    return np.array([ax1, ay1, az1, adc1, ax2, ay2, az2, adc2])
    
"""
walking task
"""
class Walk(base.Task):
  def __init__(self, random=None):
    
    # init Test instance with the base.Task __init__()
    super(Walk, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    physics.reset()
    
  # 현재 읽어들인 센서값을 관측값으로 리턴
  def get_observation(self, physics):
    return return_obs_as_vector(physics)

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    # 바뀔 수 있음
    # 바로 서있는 것에 가깝고, 앞으로 이동할수록 1에 가까운 reward
    # 아니라면 0에 가까운 reward
    
    assert isinstance(physics, Physics)
    
    # 한 step 마다 일정 범위만큼 걷는 것을 기대한다.
    # 걷기만 하면 되기 때문에 범위를 최대한 넓게 잡는다.

    cur_dist = round(physics.get_distance(), 2)
    # precision을 위해 100을 곱해 int로 변환
    cur_dist *= 100
    
    move_reward = rewards.tolerance(
        cur_dist, # 이전 step에서 이동한 거리
        bounds=(5, _WALK_DIST_RANGE), # 이 안에 들어오면 1
        margin=5, # 뒤로 움직였을 경우 0
        value_at_margin=0.2, #  reward를 0.2까지 줄여나간다. (그 이후로는 0)
        sigmoid="linear")
        
    
    # 서 있는 만큼의 reward
    # 일정 각도만큼을 곧게 서있는 상태(cos = 1)에서 허용하고, 
    # 반대로 서 있으면 (cos = 0) 0이 된다.
    cos_deviation = np.cos(np.deg2rad(_ANGLE_DEVIATION))
    
    upright_reward = rewards.tolerance(
        physics.get_upright_angle(), # [-1, 1]
        bounds=(cos_deviation, float('inf')), # (15, inf)
        margin=1+cos_deviation,
        sigmoid="linear",
        value_at_margin=0 # 오차범위를 벗어나면 reward가 0       
    )
    
    
    return upright_reward * move_reward # 조정 가능    

"""
running task
"""

class Run(base.Task):
  def __init__(self, random=None):
    
    # init Reindeer instance with the base.Task __init__()
    super(Run, self).__init__(random=random)

  def initialize_episode(self, physics):
    """Sets the state of the environment at the start of each episode."""
    physics.reset()
    
  
  def get_observation(self, physics):
    """Returns either the pure state or a set of egocentric features."""
    return 

  def get_reward(self, physics):
    """Returns a reward to the agent."""
    return 0
