from ctypes import *
import os, sys

dir = os.path.dirname(sys.modules["nim_agents"].__file__)
if sys.platform == "win32":
  libName = "nim_agents.dll"
elif sys.platform == "darwin":
  libName = "libnim_agents.dylib"
else:
  libName = "libnim_agents.so"
dll = cdll.LoadLibrary(os.path.join(dir, libName))

class NimAgentsError(Exception):
    pass

class SeqIterator(object):
    def __init__(self, seq):
        self.idx = 0
        self.seq = seq
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx < len(self.seq):
            self.idx += 1
            return self.seq[self.idx - 1]
        else:
            self.idx = 0
            raise StopIteration

def nim_agents_init_chook():
    dll.nim_agents_nim_agents_init_chook()

def start_measure():
    dll.nim_agents_start_measure()

def end_measure():
    dll.nim_agents_end_measure()

class RandomPolicy(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.nim_agents_random_policy_unref(self)

    def __init__(self, environment_config):
        result = dll.nim_agents_new_random_policy(environment_config.encode("utf8"))
        self.ref = result

    def step_batch(self, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions):
        dll.nim_agents_random_policy_step_batch(self, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

class ThinkyPolicy(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.nim_agents_thinky_policy_unref(self)

    def __init__(self, environment_config):
        result = dll.nim_agents_new_thinky_policy(environment_config.encode("utf8"))
        self.ref = result

    def step_batch(self, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions):
        dll.nim_agents_thinky_policy_step_batch(self, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

class RaceCarPolicy(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.nim_agents_race_car_policy_unref(self)

    def __init__(self, environment_config):
        result = dll.nim_agents_new_race_car_policy(environment_config.encode("utf8"))
        self.ref = result

    def step_batch(self, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions):
        dll.nim_agents_race_car_policy_step_batch(self, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

class LadybugPolicy(Structure):
    _fields_ = [("ref", c_ulonglong)]

    def __bool__(self):
        return self.ref != None

    def __eq__(self, obj):
        return self.ref == obj.ref

    def __del__(self):
        dll.nim_agents_ladybug_policy_unref(self)

    def __init__(self, environment_config):
        result = dll.nim_agents_new_ladybug_policy(environment_config.encode("utf8"))
        self.ref = result

    def step_batch(self, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions):
        dll.nim_agents_ladybug_policy_step_batch(self, agent_ids, num_agent_ids, num_agents, num_tokens, size_token, raw_observations, num_actions, raw_actions)

dll.nim_agents_nim_agents_init_chook.argtypes = []
dll.nim_agents_nim_agents_init_chook.restype = None

dll.nim_agents_start_measure.argtypes = []
dll.nim_agents_start_measure.restype = None

dll.nim_agents_end_measure.argtypes = []
dll.nim_agents_end_measure.restype = None

dll.nim_agents_random_policy_unref.argtypes = [RandomPolicy]
dll.nim_agents_random_policy_unref.restype = None

dll.nim_agents_new_random_policy.argtypes = [c_char_p]
dll.nim_agents_new_random_policy.restype = c_ulonglong

dll.nim_agents_random_policy_step_batch.argtypes = [RandomPolicy, c_void_p, c_longlong, c_longlong, c_longlong, c_longlong, c_void_p, c_longlong, c_void_p]
dll.nim_agents_random_policy_step_batch.restype = None

dll.nim_agents_thinky_policy_unref.argtypes = [ThinkyPolicy]
dll.nim_agents_thinky_policy_unref.restype = None

dll.nim_agents_new_thinky_policy.argtypes = [c_char_p]
dll.nim_agents_new_thinky_policy.restype = c_ulonglong

dll.nim_agents_thinky_policy_step_batch.argtypes = [ThinkyPolicy, c_void_p, c_longlong, c_longlong, c_longlong, c_longlong, c_void_p, c_longlong, c_void_p]
dll.nim_agents_thinky_policy_step_batch.restype = None

dll.nim_agents_race_car_policy_unref.argtypes = [RaceCarPolicy]
dll.nim_agents_race_car_policy_unref.restype = None

dll.nim_agents_new_race_car_policy.argtypes = [c_char_p]
dll.nim_agents_new_race_car_policy.restype = c_ulonglong

dll.nim_agents_race_car_policy_step_batch.argtypes = [RaceCarPolicy, c_void_p, c_longlong, c_longlong, c_longlong, c_longlong, c_void_p, c_longlong, c_void_p]
dll.nim_agents_race_car_policy_step_batch.restype = None

dll.nim_agents_ladybug_policy_unref.argtypes = [LadybugPolicy]
dll.nim_agents_ladybug_policy_unref.restype = None

dll.nim_agents_new_ladybug_policy.argtypes = [c_char_p]
dll.nim_agents_new_ladybug_policy.restype = c_ulonglong

dll.nim_agents_ladybug_policy_step_batch.argtypes = [LadybugPolicy, c_void_p, c_longlong, c_longlong, c_longlong, c_longlong, c_void_p, c_longlong, c_void_p]
dll.nim_agents_ladybug_policy_step_batch.restype = None

