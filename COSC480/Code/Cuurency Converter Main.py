"""
A currency converter program that take in real time currency rates
and helps to convert one currency to another. Also stores the history of 
currency data of all the entered currency in the program.

Author - Shahron Shaji
Id     - 51781227
"""

import tkinter as tk
from tkinter import Toplevel
from tkinter import messagebox
import requests
import matplotlib.pyplot as plt
import os
from datetime import datetime

def frames(window, color):
    """Creates the title frame"""
    
    title_frame = tk.Frame(window)
    title_frame.configure(bg = color)
    title_frame.pack()
    return title_frame

def to_upper(name):
    """Converts the string to uppercase"""
    return name.upper()

def home_window(window):
    """Pressing the button takes back to home"""
    window.destroy()
    gui_main_window()
    
def gui_geometry(width, height, window):
    """Makes the size of the window"""
    
    # Calculate the screen width and height
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    
    # Calculate the x and y coordinates for the window to be centered
    position_x = (screen_width / 2) - (width / 2)
    position_y = (screen_height / 2) - (height / 2)
    return f"{width}x{height}+{int(position_x)}+{int(position_y)}"

def get_currency_rates(currency):
    """Gets the real time currency rates"""
    
    #API Key
    api_key = '16e1aaba03db42c8b99fd06f27f529ec'
    
    #URL of the API
    api_url = 'https://api.currencyfreaks.com/v2.0/rates/latest?apikey=' + api_key
    
    #GET request to the API
    response = requests.get(api_url)
    
    #Checks for proper API conncetion
    if response.status_code == 200:
        #Converting the JSON format to Dictionary Format
        data = response.json()
        try:
            amount = data["rates"][currency]
        except Exception as e:
            error = str(e).strip("'")
            if error == "":
                error_message = "Blank Values in Entry Box"
            else:
                error_message = f"There is no currency named {error}"
            messagebox.showerror("ERROR", error_message)
        else:
            return amount
    else:
        messagebox.showerror("ERROR", f"Failed to retrieve data. Status code: {response.status_code}")
        
def time_stamp():
    """Returns the timestamp"""
    
    # Get the current date and time
    current_time = datetime.now()
    
    # Format the timestamp as a string
    timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
    return timestamp

def filename_txt(filename):
    """Adds the .txt to the filename"""
    filenametxt = f"{filename}.txt"
    return filenametxt

def read_file(filename):
    """Reads the file"""
    infile = open(filename,"r")
    data = infile.read()
    infile.close() 
    return data

def edit_file(filename, value, mode):
    """Write the file"""
    timestamp = time_stamp()
    filenametxt = filename_txt(filename)
    if mode == "w":
        file = open(filenametxt, mode)
    else:
        file = open(filenametxt, "a")
    file.write(f"{value:.2f},{timestamp} \n")
    file.close()    

def get_history_plot(filename):
    """Plots the history of stored data"""
    
    #Runs to check if the currency exists or not
    rates = get_currency_rates(filename)
    
    #Reads the File
    filename_str = f"{filename}.txt"
    if rates != None:
        try:
            data = read_file(filename_str)
            axes = plt.axes()
            lines = data.splitlines()
            infos = [line.split(",") for line in lines]
            value = []
            month = []
            for info in infos:
                value.append(float(info[0]))
                time = info[1]
                date_obj = datetime.strptime(time, "%Y-%m-%d %H:%M:%S ")
                month.append(date_obj.strftime("%b %d"))                
            axes.plot(month, value)
            axes.grid(True)
            axes.set_title(f"History of {filename}")
            axes.set_xlabel("Date")
            axes.set_ylabel(f"Value(USD as base currency)")       
            plt.show()
        except Exception as e:
            messagebox.showerror("ERROR", f"No data regarding {filename} found as of now. Please run the conversion to add history data.")
    
    
def gui_window_history(main_window):
    """Creates the GUI for plotting the History of Currency"""
    
    #Closes the Main Window
    main_window.destroy()
    
    #Window name and size
    history_window = tk.Tk()
    history_window.title("History of Currency")
    history_window.configure(bg="white")
    
    #Window dimension and position for geometry
    window_width = 400
    window_height = 350
    position = gui_geometry(window_width,window_height,history_window)
    
    # Set the window size and position
    history_window.geometry(position)
    
    #Title frame and label
    title_frame = frames(history_window, "white") 
    
    title_label = tk.Label(
        title_frame, 
        text="History of Currency",
        bg = "white",
        fg = "#8B4513",
        font = ("Verdana", 16, "bold")
    )
    title_label.grid(row = 0, padx = (0,25), pady=(50,0))
    
    #Currency Frame, Label and Entry
    currency_frame = frames(history_window, "white")
    
    currency_label = tk.Label(
        currency_frame, 
        text="Currency Name\n(Abbreviation):",
        bg = "white",
        fg = "black",
        font = ("Verdana", 10, "bold")
    )
    currency_label.grid(row = 0, padx = (0,25), pady=(50,0))
    
    currency_entry = tk.Entry(
        currency_frame,
        bg = "#FFF8E7",
        fg = "#8B4513",
        insertbackground = "#8B4513"        
    )
    
    #Command for getting the hsitory plot
    command_get_history = lambda: get_history_plot(
        to_upper(currency_entry.get())
    )
    
    #Get History Frame and button
    currency_entry.grid(row = 0, column = 1, padx = (0,25), pady=(50,0))
    get_history_frame = frames(history_window, "white")
    
    get_history_button = tk.Button(
        get_history_frame, 
        text="Get History", 
        command = command_get_history, 
        font = ("Helvetica", 10, "bold")
    )
    get_history_button.grid(row = 0, pady = (50,0))
    
    #Button to return Home
    return_home = lambda: home_window(history_window)
    home_button = tk.Button(
        get_history_frame, 
        text="Home", 
        command = return_home, 
        font = ("Helvetica", 10, "bold")
    )
    home_button.grid(row = 1, pady = (10,0))    

    history_window.mainloop()
    
def gui_result_window(result_string):
    """GUI for outputing the result"""
    
    #Defining the result window
    result_window = Toplevel()
    result_window.title("Conversion Result")
    result_window.configure(bg="#FFF8E7")
    
    #Window dimension and position for geometry
    window_width = 300
    window_height = 40
    position = gui_geometry(window_width,window_height,result_window)    
    
    result_window.geometry(position)
    result_label = tk.Label(
        result_window, 
        text=result_string,
        bg = "#FFF8E7",
        fg = "navy",
        font = ("Helvetica", 10, "bold")
    )
    result_label.pack()    

def check_filename(name, value):
    """Checking if the file exists and doing the necessary"""
    if os.path.exists(f"{name}.txt"):
        edit_file(name, value, "a")
    else:
        edit_file(name, value, "w")   

def convert(base, target, amount):
    """Converts the base currency to target currency"""
    
    #Calculate the Rates
    base_rate = float(get_currency_rates(base))
    target_rate = float(get_currency_rates(target))
    
    #adding the current rates at the time to the files
    check_filename(base, base_rate)
    check_filename(target, target_rate)
    
    #Calculate the converted amount
    factor = target_rate/base_rate
    try:
        amount_float = float(amount)
    except ValueError as e:
        error_message = str(e).strip("'")
        messagebox.showerror("ERROR", error_message)
    else:        
        converted_amount = amount_float * factor
    
    result_string = f"1 {base} = {factor:.2f} {target}(apprx.)\n {amount} {base} = {converted_amount:.2f} {target}"
    gui_result_window(result_string)
        
    
def gui_window_convert(main_window):
    """Creates the GUI for Converting the currency"""
    
    #Closes the Main Window
    main_window.destroy()
    
    #Window name and Size
    window = tk.Tk()
    window.title("Currency Converter")
    window.configure(bg="white")
    
    #Window dimension and position for geometry
    window_width = 400
    window_height = 350
    position = gui_geometry(window_width,window_height,window)
    
    # Set the window size and position
    window.geometry(position)
    
    #Frame for Entry and Label
    entry_label_frame = frames(window, "white")
    
    #Label and Enrty for Base currency
    base_currency_label = tk.Label(
        entry_label_frame, 
        text="Base Currency:",
        bg = "white",
        fg = "#8B4513",
        font = ("Helvetica", 10, "bold")
    )
    base_currency_label.grid(row = 0, padx = (0,25), pady=(50,0))
    base_currency_entry = tk.Entry(
        entry_label_frame,
        bg = "#FFF8E7",
        fg = "#8B4513",
        insertbackground = "#8B4513"
    )
    base_currency_entry.grid(row = 0, column = 1, pady=(50,0))
    
    #Label and Enrty for Target currency
    target_currency_label = tk.Label(
        entry_label_frame, 
        text="Target Currency:",
        bg = "white",
        fg = "#8B4513",
        font = ("Helvetica", 10, "bold")
    )
    target_currency_label.grid(row = 1, padx = (0,25), pady=(50,0))
    target_currency_entry = tk.Entry(
        entry_label_frame,
        bg = "#FFF8E7",
        fg = "#8B4513",
        insertbackground = "#8B4513"        
    )
    target_currency_entry.grid(row = 1, column = 1, pady=(50,0))  
    
    ##Label and Enrty for amount to be converted
    amount_label = tk.Label(
        entry_label_frame, 
        text="Amount:", 
        bg = "white",
        fg = "#8B4513",
        font = ("Helvetica", 10, "bold")
    )
    amount_label.grid(row = 2, padx = (0,23), pady=(50,0))
    amount_entry = tk.Entry(
        entry_label_frame,
        bg = "#FFF8E7",
        fg = "#8B4513",
        insertbackground = "#8B4513"        
    )
    amount_entry.grid(row = 2, column = 1, pady=(50,0)) 
    
    #Command to call the API after clicking the button
    call_api = lambda: convert(
        to_upper(base_currency_entry.get()), 
        to_upper(target_currency_entry.get()), 
        amount_entry.get()
    )
    
    #Frame for Convert Button
    convert_button_frame = frames(window, "white")       
    
    #Button for converting
    convert_button = tk.Button(
        convert_button_frame, 
        text="Convert", 
        command = call_api, 
        font = ("Helvetica", 10, "bold")
    )
    convert_button.grid(pady=(50,0))          
    
    #Button to return Home
    return_home = lambda: home_window(window)
    home_button = tk.Button(
        convert_button_frame, 
        text="Home", 
        command = return_home, 
        font = ("Helvetica", 10, "bold")
    )
    home_button.grid(pady=(10,0))   
        
    window.mainloop()

def gui_main_window():
    """This is the base GUI window"""
    
    #Window name and Size
    main_window = tk.Tk()
    main_window.title("FORENCY EXCHANGE")
    main_window.configure(bg = "white")
    
    #Window dimension and position for geometry
    window_width = 600
    window_height = 300
    position = gui_geometry(window_width,window_height,main_window)
    
    # Set the window size and position
    main_window.geometry(position)
    
    #Creating a title frame
    title_frame = frames(main_window, "white")
    title = tk.Label(
        title_frame, 
        text="Forency Exchange", 
        fg = 'purple',
        bg ="white",
        font = ("Verdana", 18, "bold")
    )
    title.grid(pady=(50,0))
    
    # Creating a frame for the buttons
    button_frame = frames(main_window, "white")
    
    command_convert = lambda: gui_window_convert(main_window)
    # Create two buttons
    conversion_button = tk.Button(
        button_frame, 
        text="Currency Conversion", 
        command = command_convert
    )
    
    command_history = lambda: gui_window_history(main_window)
    history_button = tk.Button(
        button_frame, 
        text="History of Currency",
        command = command_history
    )
    
    # Place the buttons side by side with 100 pixels of separation
    conversion_button.grid(row=1, padx=(0, 100), pady=(100,0))
    history_button.grid(row=1, column=1, pady=(100,0)) 
    
    main_window.mainloop()
    
def main():
    """The Main function of the program"""
    gui_main_window()
    
main()