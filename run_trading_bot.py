import tkinter as tk
from trading_bot_gui import TradingBotGUI

if __name__ == "__main__":
    # Create the main window
    root = tk.Tk()
    
    # Create the application
    app = TradingBotGUI(root)
    
    # Start the main loop
    root.mainloop()