import random, logging, sys
from datetime import datetime

# Creating logger for module
logger = logging.getLogger(__name__)
stream_handler = logging.StreamHandler(sys.stdout) # Stream handler
logging.basicConfig(filename = 'banking_log.log', level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s') # Logging configuration
logger.addHandler(stream_handler) # Initializing stream handler 

class BankAccount:
  def __init__(self):
    self.balance=100
    logger.info("Hello! Welcome to the ATM Depot!")
    
  def authenticate(self):
    while True:
      pin = int(input("Enter account pin: "))
      if pin != 1234:
        logger.error("Invalid pin. Try again.")
      else:
        return None
 
  def deposit(self):
    try:
      amount=float(input("Enter amount to be deposited: "))
      if amount < 0:
        print("Warning! You entered a negative number to deposit.")
      self.balance += amount
      print("Amount Deposited: ${amount}".format(amount=amount))
      logger.info("Transaction Info:")
      logger.info("Status: Successful")
      logger.info("Transaction #{number}".format(number=random.randint(10000, 1000000)))
    except ValueError:
      logger.error("You entered a non-number value to deposit.")
      logger.error("Transaction Info:")
      logger.error("Status: Failed")
      logger.error("Transaction #{number}".format(number=random.randint(10000, 1000000)))
 
  def withdraw(self):
    try:
      amount = float(input("Enter amount to be withdrawn: "))
      if self.balance >= amount:
        self.balance -= amount
        logger.info("You withdrew: ${amount}".format( amount=amount))
        logger.info("Transaction Info:")
        logger.info("Status: Successful")
        logger.info("Transaction #{number}".format(number=random.randint(10000, 1000000)))
      else:
        logger.error("Insufficient balance to complete withdraw.")
        logger.error("Transaction Info:")
        logger.error("Status: Failed")
        logger.error("Transaction #{number}".format(number=random.randint(10000, 1000000)))
    except ValueError:
      logger.error("You entered a non-number value to withdraw.")
      logger.error("Transaction Info:")
      logger.error("Status: Failed")
      logger.error("Transaction #{number}".format(number=random.randint(10000, 1000000)))
 
  def display(self):
    logger.info("Available Balance = ${balance}".format(balance=self.balance))
 
acct = BankAccount()
acct.authenticate()
acct.deposit()
acct.withdraw()
acct.display()