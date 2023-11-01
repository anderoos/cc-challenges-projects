class School:
  def __init__(self, name, level, numberOfStudents):
    self.name = name
    self.level = level
    self.numberOfStudents = numberOfStudents
  def get_name(self):
    return self.name
  def get_level(self):
    return self.level
  def get_numberOfStudents(self):
    return self.numberOfStudents
  def set_numberOfStudents(self, new_numberOfStudents):
    self.numberOfStudents = new_numberOfStudents
### first str interpolation version, returns syntax error
  # def __repr__(self):
  #   print("A {level} school named {name} with {numberOfStudents} students.".format(level=self._level, name=self._name, str(numberOfStudents=self._numberOfStudents))

### f-str interpolation is more powerful and literal, introduced in Python 3.6, in comparison to %s and .format
  def __repr__(self):
    return f'A {self.level} school named {self.name} with {self.numberOfStudents} students.'

class PrimarySchool(School):
  def __init__(self, name, numberOfStudents, pickupPolicy):
    super().__init__(name, "Primary", numberOfStudents)
    self.pickupPolicy = pickupPolicy
  def get_pickupPolicy(self):
    return self.pickupPolicy
  def __repr__(self):
    primaryRepr = super().__repr__()
    return (primaryRepr + f'The pickup policy is {self.pickupPolicy}')

class HighSchool(School):
  def __init__(self, name, numberOfStudents, sportsTeams):
    super().__init__(name, "High", numberOfStudents)
    self.sportsTeams = sportsTeams
  def sportsTeams(self):
    self.sportsTeams = sportsTeams
  def get_sportsTeams():
    return self.sportsTeams
  def __repr__(self):
    highRepr = super().__repr__()
    return (highRepr + f' We have {self.sportsTeams}')
# test1 = School("The Academy", "high", 100)
# print(test1)
# print(test1.get_name())
# print(test1.get_level())
# print(test1.get_numberOfStudents())
# test1._numberOfStudents = 500
# print(test1.get_numberOfStudents())
# test2 = PrimarySchool("Codecademy", 200, 'Pickup allowed')
# print(test2)
# print(test2.get_pickupPolicy())
test3 = HighSchool("The Academy", 300, ["Football", "Tennis, Golf"])
print(test3)
