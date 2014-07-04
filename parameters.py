import random

class Parameters(object):
  '''
  This class provides methods for searching ranges of parameters to see how
  they affect performance.
  '''
  def __init__(self):
    '''
    Have to keep track of the names and valid values of each parameter
    defined by the user.
    '''
    # list of parameter names
    self.names = []

    # list of allowed parameter values
    self.allowed_values = []

    # list of current parameter values
    self.current_values = []

    # the number of possible combinations of parameter values for all parameters
    self.combinations = 1


  def define(self,name,allowed_values):
    '''
    This method allows users to define a parameter by providing its name and
    a list of values for the parameter.  
    '''
    if name not in self.names:
      self.names.append(name)
      self.allowed_values.append(allowed_values)
      self.combinations = self.combinations * len(allowed_values)
      self.current_values.append(allowed_values[0])
    else:
      print "Parameter: ", name," is already defined!"


  def getAllNames(self):
    '''
    This method returns the names of all defined parameters.
    '''
    return self.names


  def getValue(self,name):
    '''
    This method returns the current value of the parameter specified by name.
    '''
    assert(name in self.names)
    return self.current_values[self.names.index(name)]


  def getAllValues(self):
    '''
    This method returns the current values of all defined parameters.
    '''
    for i,parameter in enumerate(self.names):
      return self.current_values


  def generateRandomCombination(self):
    '''
    This method randomly selects a value for each parameter from its list of 
    allowed parameter values.
    '''
    for i,parameter in enumerate(self.names):
      self.current_values[i] = random.choice(self.allowed_values[i])


  def generateNextCombination(self):
    '''
    This method finds the next combination of parameters values using the 
    allowed value lists for each parameter.
    '''
    i = 0
    while i < len(self.names):
      # if current value is not the last in the list
      if self.current_values[i] != self.allowed_values[i][-1]:
        # change parameter to next value in allowed value list and return
        index = self.allowed_values[i].index(self.current_values[i])
        self.current_values[i] = self.allowed_values[i][index + 1]
        i = len(self.names)
      else:
        # change parameter to first value in allowed value list
        self.current_values[i] = self.allowed_values[i][0]
        # move next parameter to next value in its allowed value list
        i = i + 1


  # Make a list of possible parameter values.
  def linearRange(start,stop,step):
    pval = start
    plist = [pval]
    while pval < stop:
      pval = pval + step
      plist.append(pval)
    return plist





