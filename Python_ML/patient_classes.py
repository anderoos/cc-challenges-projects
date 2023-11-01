class Patient:
  def __init__(self, name, age, sex, bmi, num_of_children, smoker):
    self.name = name
    self.age = age
    self.sex = sex
    self.bmi = bmi
    self.num_of_children = num_of_children
    self.smoker = smoker
    # add more parameters here
  def estimated_insurance_cost(self):
    estimated_cost = 250 * self.age - 128 * self.sex + 370 * self.bmi + 425 * self.num_of_children + 24000 * self.smoker - 12500
    statement = print(f'{self.name}\'s estimated insurance cost is {str(estimated_cost)} dollars.')
    return statement
  def update_age(self, new_age):
    self.age = new_age
    statement = print(f'{self.name} is now {self.age} years old.')
    self.estimated_insurance_cost()
    return statement
  def update_children(self, update_child):
    self.num_of_children = update_child
    if self.num_of_children == 0:
      word = 'children'
    elif self.num_of_children == 1:
      word = 'child'
    else:
      word = 'children'
    statement = print(f'{self.name} has {self.num_of_children} {word}.')
    return statement
  def patient_profile(self):
    patient_info = {
      'name' : self.name,
      'age' : self.age,
      'sex': self.sex,
      'bmi': self.bmi,
      'num_of_children': self.num_of_children}
    return print(patient_info)
    
patient1 = Patient('John Doe', 25, 1, 22.2, 0, 0)

patient1.patient_profile()