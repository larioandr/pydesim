# Python Discrete-Event Simulator

This package provides a discrete-event simulator written in Python language.

## Changelog:

Latest:
- create connections with `set()` method;
- by default create reverse connections between modules;
- create unidirectional connections with `set(reverse=False)` method;
- add `connection` argument to `handle_message()` call;
- by default, `handle_message()` does not raise `NotImplementedError` exception. 

Version 0.1.3:

- do not pass parameters to the model constructor when inherited from `Model`, instead use `sim.params` to access simulation parameters;
- redesign connections and children API, implement them with different managers;
- connections are now represented in `_ModulesConnection`, which supports delays;
- connections provide `send(message)` call which works similar to OMNeT++ by causing `handle_message(message, from=None)` method of the peer module to be called after the delay;
- parent is filled when registering a child, removed it from the model constructor. 

Version 0.1.2:

- stop support for `initialize()` method for classes inherited from `Model`;
- classes inherited from `Model` **MUST** have `sim` as the first constructor argument and **MAY** provide keyword arguments in constructor being filled with parameters during model data instantiation. 
  
> NOTE: Initialization is expected to be implemented in `Model` user-defined subclass constructor. The reason for this is simplicity, and it is hard to generalize logics in cases when model parts being instantiated and initialized dynamically. However, children should initialize themselves with care in cases when they schedule some events at other modules, since the existence of these modules is completely the question of model implementation, not any algorithm at the simulator.