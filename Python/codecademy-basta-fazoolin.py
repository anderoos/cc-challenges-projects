brunch_items = {'pancakes': 7.50, 'waffles': 9.00, 'burger': 11.00, 'home fries': 4.50, 'coffee': 1.50, 'espresso': 3.00, 'tea': 1.00, 'mimosa': 10.50, 'orange juice': 3.50}
early_bird_items = {'salumeria plate': 8.00, 'salad and breadsticks (serves 2, no refills)': 14.00, 'pizza with quattro formaggi': 9.00, 'duck ragu': 17.50, 'mushroom ravioli (vegan)': 13.50, 'coffee': 1.50, 'espresso': 3.00}
dinner_items = {'crostini with eggplant caponata': 13.00, 'caesar salad': 16.00, 'pizza with quattro formaggi': 11.00, 'duck ragu': 19.50, 'mushroom ravioli (vegan)': 13.50, 'coffee': 2.00, 'espresso': 3.00}
kids_items = {'chicken nuggets': 6.50, 'fusilli with wild mushrooms': 12.00, 'apple juice': 3.00}
arepa_items = {'arepa pabellon': 7.00, 'pernil arepa': 8.50, 'guayanes arepa': 8.00, 'jamon arepa': 7.50}

class Menu:
  def __init__(self, name, items, start_time, end_time):
    self.name = name
    self.items = items
    self.start_time = start_time
    self.end_time = end_time
  def __repr__(self):
    return "{menu} menu is available from {start} to {end}".format(menu = self.name, start = self.start_time, end = self.end_time)
  def calculate_bill(self, purchased_items):
    total_bill = 0
    for item in purchased_items:
      if item in self.items:
        total_bill += self.items[item]
    return total_bill
  
class Franchise:
  def __init__(self, address, menus):
    self.address = address
    self.menus = menus
  def __repr__(self):
    return "Our address is in {a}".format(a = self.address)
  def available_menus(self, current_time):
    menus_for_current_time = []
    for menu in self.menus:
      if current_time >= menu.start_time and current_time < menu.end_time:
        menus_for_current_time.append(menu)
    return menus_for_current_time
  # def available_menus(self, time):
  #   available_menu = []
  #   for menu in self.menus:
  #     if time >= menu.start_time and time < menu.end_time:
  #       available_menu.append(menu)
  #   return(available_menu)

class Business:
  def __init__(self, name, franchises):
    self.name = name
    self.franchises = franchises
    
brunch = Menu("brunch", brunch_items, '1100', '1600')
early_bird = Menu("early bird", early_bird_items, '1500', '1800')
dinner = Menu("dinner", dinner_items, '1700', '2300')
kids = Menu("kids", kids_items, '1100', '2100')
arepas = Menu('Take a\' Arepa', arepa_items, '1000', '2000')
# print(brunch.calculate_bill(['pancakes', 'home fries', 'coffee']))
# print(early_bird.calculate_bill(['salumeria plate', 'vegan mushroom ravioli']))
flagship = Franchise('1232 West End Road', [brunch, early_bird, dinner, kids])
new_installment = Franchise('12 East Mulberry Street', [brunch, early_bird, dinner, kids])
# print(flagship)
# print(new_installment)
# print(flagship.available_menus('1200'))
# print(flagship.available_menus('1700'))
arepas_place = Franchise('189 Fitzgerald Avenue', arepas)
print(arepas_place)