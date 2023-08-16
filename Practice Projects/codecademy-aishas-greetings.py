# Write your code below: 
from contextlib import contextmanager

@contextmanager
def generic(card_type, sender_name, recipient):
  open_generic_card = open(card_type, 'r')
  open_new_card = open(f'{sender_name}_generic.txt', 'w')
  try:
    open_new_card.write(f"Dear {recipient} \n")
    open_new_card.write(open_generic_card.read())
    open_new_card.write(f"\nSincerely,{sender_name}")
    yield open_new_card
  finally:
    open_generic_card.close()
    open_new_card.close()

# with generic('thankyou_card.txt', 'Mwenda', 'Amanda') as order1:
#   print('Card Generated!\n--------------------------')

# with open("Mwenda_generic.txt", 'r') as order1_readable:
#   print(order1_readable.read())

class Personalized():
  def __init__(self, sender, receiver, mode='w'):
    self.sender = sender
    self.receiver = receiver
    self.mode = mode
    self.file = open(f'{self.sender}_personalized.txt', self.mode)
  def __enter__(self):
    self.file.write(f'Dear {self.receiver},\n')
    return self.file
  def __exit__(self, *exc):
    self.file.write(f"\nSincerely, \n{self.sender}")
    self.file.close()
  
# with Personalized('John', 'Michael') as order2:
#   order2.write("I am so proud of you! Being your friend for all these years has been nothing but a blessing. I don’t say it often but I just wanted to let you know that you inspire me and I love you! All the best. Always.")

# with open('John_personalized.txt', 'r') as order2_readable:
#   print('Card printing!\n-----------------------')
#   print(order2_readable.read())

with generic('happy_bday.txt', 'Josiah', 'Remy') as order3:
  with Personalized('Josiah', 'Esther') as order4:
    order4.write("Happy Birthday!! I love you to the moon and back. Even though you’re a pain sometimes, you’re a pain I can't live without. I am incredibly proud of you and grateful to have you as a sister. Cheers to 25!! You’re getting old!")

with open('Josiah_generic.txt', 'r') as order3_readable:
  with open('Josiah_personalized.txt', 'r') as order4_readable:
    print(order3_readable.read())
    print('----------------------')
    print(order4_readable.read())

