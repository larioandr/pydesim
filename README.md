# Python Discrete-Event Simulator

This package provides a discrete-event simulator written in Python language.

## Changelog:

Latest:

- do not pass parameters to the model constructor when inherited from `Model`, instead use `sim.params` to access simulation parameters.

Version 0.1.2:

- stop support for `initialize()` method for classes inherited from `Model`;
- classes inherited from `Model` **MUST** have `sim` as the first constructor argument and **MAY** provide keyword arguments in constructor being filled with parameters during model data instantiation. 
  
> NOTE: Initialization is expected to be implemented in `Model` user-defined subclass constructor. The reason for this is simplicity, and it is hard to generalize logics in cases when model parts being instantiated and initialized dynamically. However, children should initialize themselves with care in cases when they schedule some events at other modules, since the existence of these modules is completely the question of model implementation, not any algorithm at the simulator.