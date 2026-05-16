# Read and write the student marks from a file and calculate the average. 

def calculate_average_marks(file_name):
    total_marks = 0
    student_count = 0

    try:
        with open(file_name, "r") as file:
            for line in file:
                try:
                    marks = float(line.strip())
                    total_marks += marks
                    student_count += 1
                except ValueError:
                    print(f"Warning: '{line.strip()}' is not a valid number")
    except FileNotFoundError:
        print(f"Error: file not found")
        return None
    
    if student_count == 0:
        print("No valid marks found in the file.")
        return None
    
    average_marks = total_marks / student_count
    return average_marks
average = calculate_average_marks("example.txt")


# Ask user for username and password if pass less than 6 characters, raise an exception
class PasswordTooShortError(Exception):
    pass

def get_user_credentials():
    username = input("Enter your username: ")
    password = input("Enter your password: ")

    if len(password) < 6:
        raise PasswordTooShortError("Password must be at least 6 characters long.")
    
    return username, password
try:
    username, password = get_user_credentials()
    print(f"Username: {username}, Password: {password}")

except PasswordTooShortError as e:
    print(f"Error: {e}")

# Read transaction data from a file and calculate total amount
def calculate_total_amount(file_name):
    total_amount = 0.0
    try:
        with open(file_name, "r") as file:
            for line in file:
                try:
                    amount = float(line.strip())
                    total_amount += amount
                except ValueError:
                    print(f"Warning: '{line.strip()}' is not a valid number and will be skipped.")
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return None
    
    return total_amount
total = calculate_total_amount("transactions.txt")
if total is not None:
    print(f"Total amount: {total}") 


# Store username in a file and prevent duplicate usernames
def store_username(username, file_name="usernames.txt"):
    try:
        with open(file_name, "r") as file:
            existing_usernames = set(line.strip() for line in file)
    except FileNotFoundError:
        existing_usernames = set()

    if username in existing_usernames:
        print(f"Error: The username '{username}' already exists. Please choose a different username.")
        return False

    with open(file_name, "a") as file:
        file.write(username + "\n")
    
    print(f"Username '{username}' has been stored successfully.")
    return True
new_username = input("Enter a new username to store: ")
store_username(new_username)