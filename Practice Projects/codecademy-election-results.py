import codecademylib
import numpy as np
from matplotlib import pyplot as plt

survey_responses = ['Ceballos', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos','Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos', 
'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos', 'Ceballos', 'Ceballos', 'Ceballos',
'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Ceballos',
'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Ceballos', 'Ceballos', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Kerrigan', 'Ceballos']

# CP 2
total_ceballos = sum([1 for nm in survey_responses if nm == "Ceballos"])
# CP 3
percentage_ceballos = float(total_ceballos)/len(survey_responses)
print(percentage_ceballos)
# CP4 & 5
possible_surveys = np.random.binomial(70, 0.54, 10000)/ 70.0
plt.hist(possible_surveys, range=(0, 1), bins = 20)
plt.show()
# CP 6
ceballos_loss_surveys = len(possible_surveys[possible_surveys < .50])/ float(len(possible_surveys))
print(ceballos_loss_surveys)
# CP 7
large_survey = np.random.binomial(7000.0, 0.54, 10000)/ 70.0
ceballos_loss_new = np.mean(large_survey < 50)
print(ceballos_loss_new)