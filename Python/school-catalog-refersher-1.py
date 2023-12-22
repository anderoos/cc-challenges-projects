class School:
  def __init__(self, name, level, numberOfStudents):
    self.name = name
    self.level = level
    self.numberOfStudents = numberOfStudents

  def get_name(self):
    return self.name

  def get_level(self):
    return self.level

  def get_num_students(self):
    return self.numberOfStudents

  def set_num_students(self, new_number):
    self.numberOfStudents = new_number

  def __repr__(self):
    return f"A {self.level} school named {self.name} with {self.numberOfStudents} students. "

# mit = School("MIT", "Graduate", "2000")
# print(mit.get_num_students)


class PrimarySchool(School):
  def __init__(self, name, numberOfStudents, pickupPolicy):
    super().__init__(name, "Primary", numberOfStudents)
    self.pickupPolicy = pickupPolicy
    
  def get_pickupPolicy(self):
    return self.pickupPolicy
    
  def __repr__(self):
    primaryRepr = super().__repr__()
    return (primaryRepr + f'The pickup policy is {self.pickupPolicy}')

# primary_test = PrimarySchool("School of Arts and Tech", 200, "after 3pm.")
# print(primary_test.get_pickupPolicy)


class HighSchool(School):
  def __init__(self, name, numberOfStudents, sportsTeams):
    super().__init__(name, "Highschool", numberOfStudents)
    self.sportsTeams = sportsTeams
    
  def get_sportsTeams(self):
    return self.sportsTeams
    
  def add_sportsTeams(self, new_team):
    self.sportsTeams += f", {new_team}"
    
  def __repr__(self):
    highRepr = super().__repr__()
    return (highRepr + f'We also have {self.sportsTeams}')

high_test = HighSchool("Hunter Highschool", 500, "Baseball")
print(high_test.get_sportsTeams)
high_test.add_sportsTeams("Basketball")
print(high_test.get_sportsTeams)

