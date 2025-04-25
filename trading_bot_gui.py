import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
import subprocess
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk

# Import our trading bot modules
try:
    from ml_strategy import train_ml_model, add_features
    from strategies import get_strategy_list, get_strategy_class
    from stock_list import get_stock_names, get_yahoo_symbols, get_fyers_symbols
    from live_paper_trading import PaperTradingEngine, authenticate_fyers
except ImportError:
    pass  # We'll handle this in the GUI

class TradingBotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Bot with Machine Learning")
        self.root.geometry("1000x700")
        self.root.minsize(900, 600)
        
        # Set theme colors
        self.bg_color = "#f5f5f5"
        self.accent_color = "#4a6fa5"
        self.button_color = "#4a6fa5"
        self.text_color = "#333333"
        self.success_color = "#28a745"
        self.warning_color = "#ffc107"
        self.danger_color = "#dc3545"
        
        # Configure the root window
        self.root.configure(bg=self.bg_color)
        
        # Create style for ttk widgets
        self.style = ttk.Style()
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure("TLabel", background=self.bg_color, foreground=self.text_color)
        self.style.configure("TButton", background=self.button_color, foreground="white")
        self.style.configure("TNotebook", background=self.bg_color)
        self.style.configure("TNotebook.Tab", background=self.bg_color, foreground=self.text_color, padding=[10, 5])
        
        # Create main container
        self.main_frame = ttk.Frame(self.root, style="TFrame")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create header
        self.create_header()
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Create tabs
        self.data_tab = ttk.Frame(self.notebook, style="TFrame")
        self.train_tab = ttk.Frame(self.notebook, style="TFrame")
        self.backtest_tab = ttk.Frame(self.notebook, style="TFrame")
        self.compare_tab = ttk.Frame(self.notebook, style="TFrame")
        self.results_tab = ttk.Frame(self.notebook, style="TFrame")
        self.live_trading_tab = ttk.Frame(self.notebook, style="TFrame")
        
        self.notebook.add(self.data_tab, text="Data")
        self.notebook.add(self.train_tab, text="Train Model")
        self.notebook.add(self.backtest_tab, text="Backtest")
        self.notebook.add(self.compare_tab, text="Compare")
        self.notebook.add(self.results_tab, text="Results")
        self.notebook.add(self.live_trading_tab, text="Live Trading")
        
        # Setup each tab
        self.setup_data_tab()
        self.setup_train_tab()
        self.setup_backtest_tab()
        self.setup_compare_tab()
        self.setup_results_tab()
        self.setup_live_trading_tab()
        
        # Create footer
        self.create_footer()
        
        # Initialize variables
        self.data_file = None
        self.model_file = "ml_model.pkl"
        self.log_text = None
        self.current_process = None
        
        # Check for existing data files
        self.check_existing_files()
    
    def create_header(self):
        """Create the application header"""
        header_frame = ttk.Frame(self.main_frame, style="TFrame")
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Title
        title_label = ttk.Label(
            header_frame, 
            text="Trading Bot with Machine Learning", 
            font=("Arial", 16, "bold"),
            style="TLabel"
        )
        title_label.pack(side=tk.LEFT, padx=5)
        
        # System info
        system_info = f"System: i5 Processor, 8GB RAM"
        system_label = ttk.Label(
            header_frame, 
            text=system_info, 
            font=("Arial", 10),
            style="TLabel"
        )
        system_label.pack(side=tk.RIGHT, padx=5)
    
    def create_footer(self):
        """Create the application footer"""
        footer_frame = ttk.Frame(self.main_frame, style="TFrame")
        footer_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Status
        self.status_label = ttk.Label(
            footer_frame, 
            text="Ready", 
            font=("Arial", 10),
            style="TLabel"
        )
        self.status_label.pack(side=tk.LEFT, padx=5)
        
        # Version
        version_label = ttk.Label(
            footer_frame, 
            text="v1.0.0", 
            font=("Arial", 10),
            style="TLabel"
        )
        version_label.pack(side=tk.RIGHT, padx=5)
    
    def setup_data_tab(self):
        """Setup the Data tab"""
        # Create frames
        top_frame = ttk.Frame(self.data_tab, style="TFrame")
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        middle_frame = ttk.Frame(self.data_tab, style="TFrame")
        middle_frame.pack(fill=tk.X, padx=10, pady=10)
        
        bottom_frame = ttk.Frame(self.data_tab, style="TFrame")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Data source selection
        source_label = ttk.Label(top_frame, text="Data Source:", style="TLabel")
        source_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.data_source = tk.StringVar(value="yahoo")
        yahoo_radio = ttk.Radiobutton(top_frame, text="Yahoo Finance", variable=self.data_source, value="yahoo")
        fyers_radio = ttk.Radiobutton(top_frame, text="Fyers API", variable=self.data_source, value="fyers")
        local_radio = ttk.Radiobutton(top_frame, text="Local File", variable=self.data_source, value="local")
        
        yahoo_radio.grid(row=0, column=1, padx=5, pady=5)
        fyers_radio.grid(row=0, column=2, padx=5, pady=5)
        local_radio.grid(row=0, column=3, padx=5, pady=5)
        
        # Stock symbol
        symbol_label = ttk.Label(top_frame, text="Stock Symbol:", style="TLabel")
        symbol_label.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.symbol_entry = ttk.Entry(top_frame, width=15)
        self.symbol_entry.insert(0, "RELIANCE.NS")
        self.symbol_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Date range
        date_label = ttk.Label(top_frame, text="Date Range:", style="TLabel")
        date_label.grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        
        date_frame = ttk.Frame(top_frame, style="TFrame")
        date_frame.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        self.start_date_entry = ttk.Entry(date_frame, width=10)
        self.start_date_entry.insert(0, "2021-01-01")
        self.start_date_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(date_frame, text="to", style="TLabel").pack(side=tk.LEFT, padx=2)
        
        self.end_date_entry = ttk.Entry(date_frame, width=10)
        self.end_date_entry.insert(0, "2023-12-31")
        self.end_date_entry.pack(side=tk.LEFT, padx=2)
        
        # Buttons
        button_frame = ttk.Frame(middle_frame, style="TFrame")
        button_frame.pack(fill=tk.X)
        
        download_button = ttk.Button(
            button_frame, 
            text="Download Data", 
            command=self.download_data
        )
        download_button.pack(side=tk.LEFT, padx=5)
        
        browse_button = ttk.Button(
            button_frame, 
            text="Browse Local File", 
            command=self.browse_data_file
        )
        browse_button.pack(side=tk.LEFT, padx=5)
        
        view_button = ttk.Button(
            button_frame, 
            text="View Data", 
            command=self.view_data
        )
        view_button.pack(side=tk.LEFT, padx=5)
        
        # Data preview
        preview_label = ttk.Label(bottom_frame, text="Data Preview:", style="TLabel")
        preview_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.data_preview = scrolledtext.ScrolledText(bottom_frame, height=15)
        self.data_preview.pack(fill=tk.BOTH, expand=True)
    
    def setup_train_tab(self):
        """Setup the Train Model tab"""
        # Create frames
        top_frame = ttk.Frame(self.train_tab, style="TFrame")
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        middle_frame = ttk.Frame(self.train_tab, style="TFrame")
        middle_frame.pack(fill=tk.X, padx=10, pady=10)
        
        bottom_frame = ttk.Frame(self.train_tab, style="TFrame")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Model parameters
        params_frame = ttk.LabelFrame(top_frame, text="Model Parameters", style="TFrame")
        params_frame.pack(fill=tk.X, pady=5)
        
        # Target days
        ttk.Label(params_frame, text="Prediction Target (days):", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.target_days = tk.IntVar(value=1)
        target_1 = ttk.Radiobutton(params_frame, text="1 Day", variable=self.target_days, value=1)
        target_3 = ttk.Radiobutton(params_frame, text="3 Days", variable=self.target_days, value=3)
        target_5 = ttk.Radiobutton(params_frame, text="5 Days", variable=self.target_days, value=5)
        
        target_1.grid(row=0, column=1, padx=5, pady=5)
        target_3.grid(row=0, column=2, padx=5, pady=5)
        target_5.grid(row=0, column=3, padx=5, pady=5)
        
        # Test size
        ttk.Label(params_frame, text="Test Size:", style="TLabel").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.test_size = tk.DoubleVar(value=0.2)
        test_size_entry = ttk.Entry(params_frame, textvariable=self.test_size, width=5)
        test_size_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Random Forest parameters
        ttk.Label(params_frame, text="Estimators:", style="TLabel").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.n_estimators = tk.IntVar(value=100)
        estimators_entry = ttk.Entry(params_frame, textvariable=self.n_estimators, width=5)
        estimators_entry.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(params_frame, text="Max Depth:", style="TLabel").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.max_depth = tk.IntVar(value=10)
        max_depth_entry = ttk.Entry(params_frame, textvariable=self.max_depth, width=5)
        max_depth_entry.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Resource usage controls
        resource_frame = ttk.LabelFrame(top_frame, text="Resource Management", style="TFrame")
        resource_frame.pack(fill=tk.X, pady=5)
        
        # CPU usage
        ttk.Label(resource_frame, text="CPU Usage:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.cpu_usage = tk.StringVar(value="auto")
        cpu_auto = ttk.Radiobutton(resource_frame, text="Auto (50%)", variable=self.cpu_usage, value="auto")
        cpu_low = ttk.Radiobutton(resource_frame, text="Low (25%)", variable=self.cpu_usage, value="low")
        cpu_high = ttk.Radiobutton(resource_frame, text="High (75%)", variable=self.cpu_usage, value="high")
        cpu_max = ttk.Radiobutton(resource_frame, text="Maximum", variable=self.cpu_usage, value="max")
        
        cpu_auto.grid(row=0, column=1, padx=5, pady=5)
        cpu_low.grid(row=0, column=2, padx=5, pady=5)
        cpu_high.grid(row=0, column=3, padx=5, pady=5)
        cpu_max.grid(row=0, column=4, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(middle_frame, style="TFrame")
        button_frame.pack(fill=tk.X)
        
        train_button = ttk.Button(
            button_frame, 
            text="Train Model", 
            command=self.train_model
        )
        train_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        progress_frame = ttk.Frame(middle_frame, style="TFrame")
        progress_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(progress_frame, text="Progress:", style="TLabel").pack(side=tk.LEFT, padx=5)
        self.train_progress = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
        self.train_progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.progress_label = ttk.Label(progress_frame, text="0%", style="TLabel")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        # Training log
        log_label = ttk.Label(bottom_frame, text="Training Log:", style="TLabel")
        log_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.train_log = scrolledtext.ScrolledText(bottom_frame, height=15)
        self.train_log.pack(fill=tk.BOTH, expand=True)
    
    def setup_backtest_tab(self):
        """Setup the Backtest tab"""
        # Create frames
        top_frame = ttk.Frame(self.backtest_tab, style="TFrame")
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        middle_frame = ttk.Frame(self.backtest_tab, style="TFrame")
        middle_frame.pack(fill=tk.X, padx=10, pady=10)
        
        bottom_frame = ttk.Frame(self.backtest_tab, style="TFrame")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Stock selection
        stock_frame = ttk.Frame(top_frame, style="TFrame")
        stock_frame.grid(row=0, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(stock_frame, text="Stock:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Get stock names or use a default list if import fails
        try:
            stock_list = get_stock_names()
        except:
            stock_list = ["Reliance Industries", "TCS", "HDFC Bank", "Infosys", "ICICI Bank"]
        
        self.selected_stock = tk.StringVar(value=stock_list[0])
        stock_dropdown = ttk.Combobox(stock_frame, textvariable=self.selected_stock, values=stock_list, width=30)
        stock_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Strategy selection
        strategy_frame = ttk.Frame(top_frame, style="TFrame")
        strategy_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W+tk.E)
        
        ttk.Label(strategy_frame, text="Strategy:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Get strategy list or use default if import fails
        try:
            strategy_list = get_strategy_list()
        except:
            strategy_list = ["SMA Crossover", "RSI", "MACD", "Bollinger Bands", "Dual MA"]
        
        self.selected_strategy = tk.StringVar(value=strategy_list[0])
        strategy_dropdown = ttk.Combobox(strategy_frame, textvariable=self.selected_strategy, values=strategy_list, width=30)
        strategy_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Also keep the old strategy radio buttons for backward compatibility
        self.strategy = tk.StringVar(value="ml")
        
        # Strategy parameters
        params_frame = ttk.LabelFrame(top_frame, text="Strategy Parameters", style="TFrame")
        params_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # Create parameter frames for each strategy
        
        # ML parameters
        self.ml_frame = ttk.Frame(params_frame, style="TFrame")
        
        ttk.Label(self.ml_frame, text="Prediction Threshold:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.prediction_threshold = tk.DoubleVar(value=0.6)
        threshold_entry = ttk.Entry(self.ml_frame, textvariable=self.prediction_threshold, width=5)
        threshold_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # SMA parameters
        self.sma_frame = ttk.Frame(params_frame, style="TFrame")
        
        ttk.Label(self.sma_frame, text="Fast Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.sma_fast = tk.IntVar(value=10)
        fast_entry = ttk.Entry(self.sma_frame, textvariable=self.sma_fast, width=5)
        fast_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.sma_frame, text="Slow Period:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.sma_slow = tk.IntVar(value=30)
        slow_entry = ttk.Entry(self.sma_frame, textvariable=self.sma_slow, width=5)
        slow_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # RSI parameters
        self.rsi_frame = ttk.Frame(params_frame, style="TFrame")
        
        ttk.Label(self.rsi_frame, text="RSI Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.rsi_period = tk.IntVar(value=14)
        rsi_period_entry = ttk.Entry(self.rsi_frame, textvariable=self.rsi_period, width=5)
        rsi_period_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.rsi_frame, text="Oversold:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.rsi_oversold = tk.IntVar(value=30)
        rsi_oversold_entry = ttk.Entry(self.rsi_frame, textvariable=self.rsi_oversold, width=5)
        rsi_oversold_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.rsi_frame, text="Overbought:", style="TLabel").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.rsi_overbought = tk.IntVar(value=70)
        rsi_overbought_entry = ttk.Entry(self.rsi_frame, textvariable=self.rsi_overbought, width=5)
        rsi_overbought_entry.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        
        # MACD parameters
        self.macd_frame = ttk.Frame(params_frame, style="TFrame")
        
        ttk.Label(self.macd_frame, text="Fast Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.macd_fast = tk.IntVar(value=12)
        macd_fast_entry = ttk.Entry(self.macd_frame, textvariable=self.macd_fast, width=5)
        macd_fast_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.macd_frame, text="Slow Period:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.macd_slow = tk.IntVar(value=26)
        macd_slow_entry = ttk.Entry(self.macd_frame, textvariable=self.macd_slow, width=5)
        macd_slow_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.macd_frame, text="Signal Period:", style="TLabel").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.macd_signal = tk.IntVar(value=9)
        macd_signal_entry = ttk.Entry(self.macd_frame, textvariable=self.macd_signal, width=5)
        macd_signal_entry.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        
        # Bollinger Bands parameters
        self.bb_frame = ttk.Frame(params_frame, style="TFrame")
        
        ttk.Label(self.bb_frame, text="Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.bb_period = tk.IntVar(value=20)
        bb_period_entry = ttk.Entry(self.bb_frame, textvariable=self.bb_period, width=5)
        bb_period_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.bb_frame, text="Std Dev:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.bb_devfactor = tk.DoubleVar(value=2.0)
        bb_devfactor_entry = ttk.Entry(self.bb_frame, textvariable=self.bb_devfactor, width=5)
        bb_devfactor_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Dual MA parameters
        self.dualma_frame = ttk.Frame(params_frame, style="TFrame")
        
        ttk.Label(self.dualma_frame, text="MA1 Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.ma1_period = tk.IntVar(value=10)
        ma1_period_entry = ttk.Entry(self.dualma_frame, textvariable=self.ma1_period, width=5)
        ma1_period_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.dualma_frame, text="MA1 Type:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.ma1_type = tk.StringVar(value="SMA")
        ma1_type_dropdown = ttk.Combobox(self.dualma_frame, textvariable=self.ma1_type, values=["SMA", "EMA"], width=5)
        ma1_type_dropdown.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.dualma_frame, text="MA2 Period:", style="TLabel").grid(row=0, column=4, padx=5, pady=5, sticky=tk.W)
        self.ma2_period = tk.IntVar(value=30)
        ma2_period_entry = ttk.Entry(self.dualma_frame, textvariable=self.ma2_period, width=5)
        ma2_period_entry.grid(row=0, column=5, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.dualma_frame, text="MA2 Type:", style="TLabel").grid(row=0, column=6, padx=5, pady=5, sticky=tk.W)
        self.ma2_type = tk.StringVar(value="EMA")
        ma2_type_dropdown = ttk.Combobox(self.dualma_frame, textvariable=self.ma2_type, values=["SMA", "EMA"], width=5)
        ma2_type_dropdown.grid(row=0, column=7, padx=5, pady=5, sticky=tk.W)
        
        # Common parameters
        common_frame = ttk.Frame(params_frame, style="TFrame")
        common_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(common_frame, text="Initial Cash:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.initial_cash = tk.DoubleVar(value=100000.0)
        cash_entry = ttk.Entry(common_frame, textvariable=self.initial_cash, width=10)
        cash_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(common_frame, text="Commission (%):", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.commission = tk.DoubleVar(value=0.1)
        commission_entry = ttk.Entry(common_frame, textvariable=self.commission, width=5)
        commission_entry.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Update visible parameters based on strategy selection
        self.selected_strategy.trace_add("write", self.update_strategy_params)
        self.update_strategy_params()
        
        # Buttons
        button_frame = ttk.Frame(middle_frame, style="TFrame")
        button_frame.pack(fill=tk.X)
        
        backtest_button = ttk.Button(
            button_frame, 
            text="Run Backtest", 
            command=self.run_backtest
        )
        backtest_button.pack(side=tk.LEFT, padx=5)
        
        # Backtest log
        log_label = ttk.Label(bottom_frame, text="Backtest Log:", style="TLabel")
        log_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.backtest_log = scrolledtext.ScrolledText(bottom_frame, height=15)
        self.backtest_log.pack(fill=tk.BOTH, expand=True)
    
    def setup_compare_tab(self):
        """Setup the Compare tab"""
        # Create frames
        top_frame = ttk.Frame(self.compare_tab, style="TFrame")
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        bottom_frame = ttk.Frame(self.compare_tab, style="TFrame")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Buttons
        compare_button = ttk.Button(
            top_frame, 
            text="Compare Strategies", 
            command=self.compare_strategies
        )
        compare_button.pack(side=tk.LEFT, padx=5)
        
        # Comparison results
        self.comparison_frame = ttk.Frame(bottom_frame, style="TFrame")
        self.comparison_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initially show a message
        self.comparison_message = ttk.Label(
            self.comparison_frame, 
            text="Click 'Compare Strategies' to run comparison", 
            style="TLabel"
        )
        self.comparison_message.pack(expand=True)
    
    def setup_results_tab(self):
        """Setup the Results tab"""
        # Create frames
        top_frame = ttk.Frame(self.results_tab, style="TFrame")
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        bottom_frame = ttk.Frame(self.results_tab, style="TFrame")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results selection
        results_label = ttk.Label(top_frame, text="Select Results:", style="TLabel")
        results_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.results_type = tk.StringVar(value="ml")
        ml_radio = ttk.Radiobutton(top_frame, text="Machine Learning", variable=self.results_type, value="ml", command=self.load_results)
        sma_radio = ttk.Radiobutton(top_frame, text="SMA Crossover", variable=self.results_type, value="sma", command=self.load_results)
        comparison_radio = ttk.Radiobutton(top_frame, text="Comparison", variable=self.results_type, value="comparison", command=self.load_results)
        
        ml_radio.grid(row=0, column=1, padx=5, pady=5)
        sma_radio.grid(row=0, column=2, padx=5, pady=5)
        comparison_radio.grid(row=0, column=3, padx=5, pady=5)
        
        # Results display
        self.results_frame = ttk.Frame(bottom_frame, style="TFrame")
        self.results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initially show a message
        self.results_message = ttk.Label(
            self.results_frame, 
            text="Select a result type to view", 
            style="TLabel"
        )
        self.results_message.pack(expand=True)
    
    def update_strategy_params(self, *args):
        """Update visible strategy parameters based on selection"""
        # Hide all parameter frames
        for frame in [self.ml_frame, self.sma_frame, self.rsi_frame, 
                     self.macd_frame, self.bb_frame, self.dualma_frame]:
            try:
                frame.pack_forget()
            except:
                pass
        
        # Show the appropriate frame based on the selected strategy
        strategy = self.selected_strategy.get()
        
        if strategy == "Machine Learning" or strategy == "ML Strategy":
            self.ml_frame.pack(fill=tk.X, pady=5)
        elif strategy == "SMA Crossover":
            self.sma_frame.pack(fill=tk.X, pady=5)
        elif strategy == "RSI":
            self.rsi_frame.pack(fill=tk.X, pady=5)
        elif strategy == "MACD":
            self.macd_frame.pack(fill=tk.X, pady=5)
        elif strategy == "Bollinger Bands":
            self.bb_frame.pack(fill=tk.X, pady=5)
        elif strategy == "Dual MA":
            self.dualma_frame.pack(fill=tk.X, pady=5)
        else:
            # Default to SMA for unknown strategies
            self.sma_frame.pack(fill=tk.X, pady=5)
            
        # Also handle the old radio button selection for backward compatibility
        if self.strategy.get() == "ml":
            self.ml_frame.pack(fill=tk.X, pady=5)
        elif self.strategy.get() == "sma" and strategy not in ["SMA Crossover", "RSI", "MACD", "Bollinger Bands", "Dual MA"]:
            self.sma_frame.pack(fill=tk.X, pady=5)
    
    def check_existing_files(self):
        """Check for existing data and model files"""
        # Check for data files
        data_files = [f for f in os.listdir() if f.endswith('.csv')]
        if data_files:
            self.data_file = data_files[0]
            self.update_status(f"Found data file: {self.data_file}")
        
        # Check for model file
        if os.path.exists(self.model_file):
            self.update_status(f"Found model file: {self.model_file}")
    
    def update_status(self, message):
        """Update the status bar message"""
        self.status_label.config(text=message)
        self.root.update_idletasks()
    
    def redirect_output(self, text_widget):
        """Redirect stdout and stderr to the specified text widget"""
        self.log_text = text_widget
        sys.stdout = TextRedirector(text_widget, "stdout")
        sys.stderr = TextRedirector(text_widget, "stderr")
    
    def restore_output(self):
        """Restore stdout and stderr to their original values"""
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    
    def download_data(self):
        """Download data from the selected source"""
        source = self.data_source.get()
        symbol = self.symbol_entry.get()
        start_date = self.start_date_entry.get()
        end_date = self.end_date_entry.get()
        
        if not symbol:
            messagebox.showerror("Error", "Please enter a stock symbol")
            return
        
        self.update_status(f"Downloading data for {symbol}...")
        
        # Clear the data preview
        self.data_preview.delete(1.0, tk.END)
        
        # Redirect output to data preview
        self.redirect_output(self.data_preview)
        
        try:
            if source == "yahoo":
                # Run the download_data.py script
                self.current_process = subprocess.Popen(
                    [sys.executable, "download_data.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Read output in a separate thread
                threading.Thread(target=self.read_process_output, args=(self.data_preview,)).start()
                
            elif source == "fyers":
                # Format the symbol for Fyers API (e.g., RELIANCE.NS -> NSE:RELIANCE-EQ)
                fyers_symbol = symbol
                if ".NS" in symbol:
                    fyers_symbol = f"NSE:{symbol.replace('.NS', '-EQ')}"
                elif ".BSE" in symbol:
                    fyers_symbol = f"BSE:{symbol.replace('.BSE', '-EQ')}"
                
                # Run the improved Fyers data script with parameters
                self.data_preview.insert(tk.END, f"Downloading data from Fyers API for {fyers_symbol}...\n")
                self.data_preview.insert(tk.END, f"Date range: {start_date} to {end_date}\n\n")
                
                # Check if the improved script exists, otherwise use the original
                fyers_script = "get_fyers_data_improved.py" if os.path.exists("get_fyers_data_improved.py") else "get_fyers_data.py"
                
                self.current_process = subprocess.Popen(
                    [
                        sys.executable, 
                        fyers_script,
                        "--symbol", fyers_symbol,
                        "--start_date", start_date,
                        "--end_date", end_date,
                        "--resolution", "D"  # Daily data
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Read output in a separate thread
                threading.Thread(target=self.read_process_output, args=(self.data_preview,)).start()
                
            else:  # Local file - nothing to download
                self.data_preview.insert(tk.END, "Please use 'Browse Local File' to select a data file.")
        
        except Exception as e:
            self.data_preview.insert(tk.END, f"Error: {str(e)}")
        
        # Restore output
        self.restore_output()
    
    def read_process_output(self, text_widget):
        """Read output from a subprocess and update the text widget"""
        if not self.current_process:
            return
        
        for line in iter(self.current_process.stdout.readline, ''):
            if line:
                text_widget.insert(tk.END, line)
                text_widget.see(tk.END)
                self.root.update_idletasks()
        
        # Process completed
        self.current_process.wait()
        
        if self.current_process.returncode == 0:
            # Look for CSV files
            data_files = [f for f in os.listdir() if f.endswith('.csv')]
            if data_files:
                self.data_file = data_files[0]
                text_widget.insert(tk.END, f"\nData download completed. File: {self.data_file}\n")
                self.update_status(f"Data downloaded: {self.data_file}")
            else:
                text_widget.insert(tk.END, "\nData download completed but no CSV file found.\n")
                self.update_status("Data download completed but no file found")
        else:
            text_widget.insert(tk.END, f"\nError during data download. Return code: {self.current_process.returncode}\n")
            self.update_status("Error during data download")
        
        self.current_process = None
    
    def browse_data_file(self):
        """Browse for a local data file"""
        file_path = filedialog.askopenfilename(
            title="Select Data File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        
        if file_path:
            self.data_file = file_path
            self.update_status(f"Selected data file: {os.path.basename(self.data_file)}")
            self.view_data()
    
    def view_data(self):
        """View the selected data file"""
        if not self.data_file or not os.path.exists(self.data_file):
            messagebox.showerror("Error", "No data file selected or file does not exist")
            return
        
        self.update_status(f"Loading data from {self.data_file}...")
        
        # Clear the data preview
        self.data_preview.delete(1.0, tk.END)
        
        try:
            # Load the data
            df = pd.read_csv(self.data_file)
            
            # Display basic info
            self.data_preview.insert(tk.END, f"File: {self.data_file}\n")
            self.data_preview.insert(tk.END, f"Shape: {df.shape}\n\n")
            
            # Display head
            self.data_preview.insert(tk.END, "First 5 rows:\n")
            self.data_preview.insert(tk.END, df.head().to_string())
            
            # Display tail
            self.data_preview.insert(tk.END, "\n\nLast 5 rows:\n")
            self.data_preview.insert(tk.END, df.tail().to_string())
            
            # Display summary statistics
            self.data_preview.insert(tk.END, "\n\nSummary Statistics:\n")
            self.data_preview.insert(tk.END, df.describe().to_string())
            
            self.update_status(f"Data loaded: {self.data_file}")
        
        except Exception as e:
            self.data_preview.insert(tk.END, f"Error loading data: {str(e)}")
            self.update_status("Error loading data")
    
    def train_model(self):
        """Train the machine learning model"""
        if not self.data_file or not os.path.exists(self.data_file):
            messagebox.showerror("Error", "No data file selected or file does not exist")
            return
        
        self.update_status("Training model...")
        
        # Clear the training log
        self.train_log.delete(1.0, tk.END)
        
        # Redirect output to training log
        self.redirect_output(self.train_log)
        
        # Start training in a separate thread
        threading.Thread(target=self._train_model_thread).start()
    
    def _train_model_thread(self):
        """Thread function for model training with resource management"""
        try:
            # Import here to avoid circular imports
            from ml_strategy import train_ml_model
            import psutil
            
            # Reset progress bar
            self.root.after(0, lambda: self.train_progress.config(value=0))
            self.root.after(0, lambda: self.progress_label.config(text="0%"))
            
            # Get parameters
            target_days = self.target_days.get()
            test_size = self.test_size.get()
            n_estimators = self.n_estimators.get()
            max_depth = self.max_depth.get()
            
            # Determine CPU usage based on selection
            cpu_usage = self.cpu_usage.get()
            cpu_count = psutil.cpu_count(logical=True)
            
            if cpu_usage == "low":
                n_jobs = max(1, int(cpu_count * 0.25))  # 25% of cores
            elif cpu_usage == "auto":
                n_jobs = max(1, int(cpu_count * 0.5))   # 50% of cores
            elif cpu_usage == "high":
                n_jobs = max(1, int(cpu_count * 0.75))  # 75% of cores
            elif cpu_usage == "max":
                n_jobs = -1  # All cores
            else:
                n_jobs = None  # Auto-determine
            
            # Progress callback function
            def update_progress(progress):
                self.root.after(0, lambda: self.train_progress.config(value=progress))
                self.root.after(0, lambda: self.progress_label.config(text=f"{progress}%"))
            
            # Log training parameters
            self.train_log.insert(tk.END, f"Training model with the following parameters:\n")
            self.train_log.insert(tk.END, f"- Target days: {target_days}\n")
            self.train_log.insert(tk.END, f"- Test size: {test_size}\n")
            self.train_log.insert(tk.END, f"- Estimators: {n_estimators}\n")
            self.train_log.insert(tk.END, f"- Max depth: {max_depth}\n")
            self.train_log.insert(tk.END, f"- CPU usage: {cpu_usage} ({n_jobs} cores)\n")
            self.train_log.insert(tk.END, f"- Data file: {self.data_file}\n\n")
            
            # Train the model with resource constraints
            model, scaler, feature_names = train_ml_model(
                data_file=self.data_file,
                target_days=target_days,
                test_size=test_size,
                save_path=self.model_file,
                n_estimators=n_estimators,
                max_depth=max_depth,
                n_jobs=n_jobs,
                progress_callback=update_progress
            )
            
            if model is not None:
                self.train_log.insert(tk.END, "\nModel training completed successfully.\n")
                self.update_status("Model training completed")
                
                # Set progress to 100% to ensure it's complete
                self.root.after(0, lambda: self.train_progress.config(value=100))
                self.root.after(0, lambda: self.progress_label.config(text="100%"))
            else:
                self.train_log.insert(tk.END, "\nModel training failed.\n")
                self.update_status("Model training failed")
        
        except Exception as e:
            self.train_log.insert(tk.END, f"Error during model training: {str(e)}\n")
            self.update_status("Error during model training")
        
        # Restore output
        self.restore_output()
    
    def run_backtest(self):
        """Run a backtest with the selected strategy"""
        if not self.data_file or not os.path.exists(self.data_file):
            messagebox.showerror("Error", "No data file selected or file does not exist")
            return
        
        strategy = self.strategy.get()
        
        if strategy == "ml" and not os.path.exists(self.model_file):
            messagebox.showerror("Error", "ML model file not found. Please train the model first.")
            return
        
        self.update_status(f"Running {strategy} backtest...")
        
        # Clear the backtest log
        self.backtest_log.delete(1.0, tk.END)
        
        # Redirect output to backtest log
        self.redirect_output(self.backtest_log)
        
        # Start backtest in a separate thread
        threading.Thread(target=self._run_backtest_thread).start()
    
    def _run_backtest_thread(self):
        """Thread function for running backtest"""
        try:
            strategy = self.strategy.get()
            
            if strategy == "ml":
                # Run ML backtest
                self.backtest_log.insert(tk.END, "Running Machine Learning backtest...\n")
                
                # Run the backtest_ml.py script
                self.current_process = subprocess.Popen(
                    [sys.executable, "backtest_ml.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Read output
                for line in iter(self.current_process.stdout.readline, ''):
                    if line:
                        self.backtest_log.insert(tk.END, line)
                        self.backtest_log.see(tk.END)
                        self.root.update_idletasks()
                
                # Process completed
                self.current_process.wait()
                
                if self.current_process.returncode == 0:
                    self.backtest_log.insert(tk.END, "\nML backtest completed successfully.\n")
                    self.update_status("ML backtest completed")
                else:
                    self.backtest_log.insert(tk.END, f"\nError during ML backtest. Return code: {self.current_process.returncode}\n")
                    self.update_status("Error during ML backtest")
            
            else:
                # Run SMA backtest
                self.backtest_log.insert(tk.END, "Running SMA Crossover backtest...\n")
                
                # Run the backtest.py script
                self.current_process = subprocess.Popen(
                    [sys.executable, "backtest.py"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                
                # Read output
                for line in iter(self.current_process.stdout.readline, ''):
                    if line:
                        self.backtest_log.insert(tk.END, line)
                        self.backtest_log.see(tk.END)
                        self.root.update_idletasks()
                
                # Process completed
                self.current_process.wait()
                
                if self.current_process.returncode == 0:
                    self.backtest_log.insert(tk.END, "\nSMA backtest completed successfully.\n")
                    self.update_status("SMA backtest completed")
                else:
                    self.backtest_log.insert(tk.END, f"\nError during SMA backtest. Return code: {self.current_process.returncode}\n")
                    self.update_status("Error during SMA backtest")
        
        except Exception as e:
            self.backtest_log.insert(tk.END, f"Error during backtest: {str(e)}\n")
            self.update_status("Error during backtest")
        
        # Restore output
        self.restore_output()
        self.current_process = None
    
    def compare_strategies(self):
        """Compare the SMA and ML strategies"""
        if not self.data_file or not os.path.exists(self.data_file):
            messagebox.showerror("Error", "No data file selected or file does not exist")
            return
        
        if not os.path.exists(self.model_file):
            messagebox.showerror("Error", "ML model file not found. Please train the model first.")
            return
        
        self.update_status("Comparing strategies...")
        
        # Clear the comparison frame
        for widget in self.comparison_frame.winfo_children():
            widget.destroy()
        
        # Create a text widget for logs
        comparison_log = scrolledtext.ScrolledText(self.comparison_frame, height=10)
        comparison_log.pack(fill=tk.X, expand=False, pady=(0, 10))
        
        # Redirect output to comparison log
        self.redirect_output(comparison_log)
        
        # Start comparison in a separate thread
        threading.Thread(target=self._compare_strategies_thread, args=(comparison_log,)).start()
    
    def _compare_strategies_thread(self, log_widget):
        """Thread function for strategy comparison"""
        try:
            log_widget.insert(tk.END, "Comparing strategies...\n")
            
            # Run the compare_strategies.py script
            self.current_process = subprocess.Popen(
                [sys.executable, "compare_strategies.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Read output
            for line in iter(self.current_process.stdout.readline, ''):
                if line:
                    log_widget.insert(tk.END, line)
                    log_widget.see(tk.END)
                    self.root.update_idletasks()
            
            # Process completed
            self.current_process.wait()
            
            if self.current_process.returncode == 0:
                log_widget.insert(tk.END, "\nStrategy comparison completed successfully.\n")
                self.update_status("Strategy comparison completed")
                
                # Check if comparison image exists
                if os.path.exists("strategy_comparison.png"):
                    # Display the comparison image
                    self.display_comparison_image()
            else:
                log_widget.insert(tk.END, f"\nError during strategy comparison. Return code: {self.current_process.returncode}\n")
                self.update_status("Error during strategy comparison")
        
        except Exception as e:
            log_widget.insert(tk.END, f"Error during strategy comparison: {str(e)}\n")
            self.update_status("Error during strategy comparison")
        
        # Restore output
        self.restore_output()
        self.current_process = None
    
    def display_comparison_image(self):
        """Display the strategy comparison image"""
        try:
            # Create a frame for the image
            image_frame = ttk.Frame(self.comparison_frame)
            image_frame.pack(fill=tk.BOTH, expand=True)
            
            # Load the image
            img = Image.open("strategy_comparison.png")
            
            # Resize the image to fit the frame
            img = self.resize_image(img, 800, 600)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Create a label to display the image
            image_label = ttk.Label(image_frame, image=photo)
            image_label.image = photo  # Keep a reference to prevent garbage collection
            image_label.pack(fill=tk.BOTH, expand=True)
        
        except Exception as e:
            print(f"Error displaying comparison image: {str(e)}")
    
    def resize_image(self, img, max_width, max_height):
        """Resize an image while maintaining aspect ratio"""
        width, height = img.size
        
        # Calculate aspect ratio
        aspect_ratio = width / height
        
        # Calculate new dimensions
        if width > max_width or height > max_height:
            if width / max_width > height / max_height:
                # Width is the limiting factor
                new_width = max_width
                new_height = int(new_width / aspect_ratio)
            else:
                # Height is the limiting factor
                new_height = max_height
                new_width = int(new_height * aspect_ratio)
            
            # Resize the image
            img = img.resize((new_width, new_height), Image.LANCZOS)
        
        return img
    
    def setup_live_trading_tab(self):
        """Setup the Live Trading tab with two modes: machine-driven and user-selected"""
        # Create frames
        top_frame = ttk.Frame(self.live_trading_tab, style="TFrame")
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        middle_frame = ttk.Frame(self.live_trading_tab, style="TFrame")
        middle_frame.pack(fill=tk.X, padx=10, pady=10)
        
        bottom_frame = ttk.Frame(self.live_trading_tab, style="TFrame")
        bottom_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Mode selection
        mode_frame = ttk.LabelFrame(top_frame, text="Trading Mode", style="TFrame")
        mode_frame.pack(fill=tk.X, pady=5)
        
        self.trading_mode = tk.StringVar(value="auto")
        auto_radio = ttk.Radiobutton(
            mode_frame, 
            text="Automated (Machine-Driven)", 
            variable=self.trading_mode, 
            value="auto",
            command=self.update_trading_mode
        )
        user_radio = ttk.Radiobutton(
            mode_frame, 
            text="User-Selected Strategy", 
            variable=self.trading_mode, 
            value="user",
            command=self.update_trading_mode
        )
        
        auto_radio.grid(row=0, column=0, padx=20, pady=5, sticky=tk.W)
        user_radio.grid(row=0, column=1, padx=20, pady=5, sticky=tk.W)
        
        # Stock selection
        stock_frame = ttk.Frame(top_frame, style="TFrame")
        stock_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(stock_frame, text="Stock Symbol:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Get Fyers symbols or use a default list if import fails
        try:
            stock_list = get_fyers_symbols()
        except:
            stock_list = ["NSE:RELIANCE-EQ", "NSE:TCS-EQ", "NSE:HDFCBANK-EQ", "NSE:INFY-EQ", "NSE:ICICIBANK-EQ"]
        
        self.live_symbol = tk.StringVar(value=stock_list[0])
        symbol_dropdown = ttk.Combobox(stock_frame, textvariable=self.live_symbol, values=stock_list, width=30)
        symbol_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Strategy selection (for user mode)
        self.strategy_frame = ttk.LabelFrame(middle_frame, text="Strategy Selection", style="TFrame")
        self.strategy_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(self.strategy_frame, text="Strategy:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        # Get strategy list or use default if import fails
        try:
            strategy_list = get_strategy_list()
        except:
            strategy_list = ["SMA Crossover", "RSI", "MACD", "Bollinger Bands", "Dual MA"]
        
        self.live_strategy = tk.StringVar(value=strategy_list[0])
        strategy_dropdown = ttk.Combobox(self.strategy_frame, textvariable=self.live_strategy, values=strategy_list, width=30)
        strategy_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Strategy parameters (customizable indicators)
        self.params_frame = ttk.LabelFrame(self.strategy_frame, text="Strategy Parameters", style="TFrame")
        self.params_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W+tk.E)
        
        # SMA Crossover parameters (default)
        ttk.Label(self.params_frame, text="Fast SMA Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.sma_fast_period = tk.IntVar(value=10)
        ttk.Entry(self.params_frame, textvariable=self.sma_fast_period, width=5).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        ttk.Label(self.params_frame, text="Slow SMA Period:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.sma_slow_period = tk.IntVar(value=30)
        ttk.Entry(self.params_frame, textvariable=self.sma_slow_period, width=5).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Update parameters when strategy changes
        strategy_dropdown.bind("<<ComboboxSelected>>", self.update_strategy_params)
        
        # Trading controls
        controls_frame = ttk.LabelFrame(middle_frame, text="Trading Controls", style="TFrame")
        controls_frame.pack(fill=tk.X, pady=10)
        
        # Initial capital
        ttk.Label(controls_frame, text="Initial Capital (₹):", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.initial_capital = tk.DoubleVar(value=100000.0)
        ttk.Entry(controls_frame, textvariable=self.initial_capital, width=10).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Commission
        ttk.Label(controls_frame, text="Commission (%):", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.commission_rate = tk.DoubleVar(value=0.1)  # 0.1%
        ttk.Entry(controls_frame, textvariable=self.commission_rate, width=5).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Start/Stop buttons
        button_frame = ttk.Frame(controls_frame, style="TFrame")
        button_frame.grid(row=1, column=0, columnspan=4, padx=5, pady=5)
        
        self.start_button = ttk.Button(
            button_frame, 
            text="Start Live Paper Trading", 
            command=self.start_live_trading
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(
            button_frame, 
            text="Stop Trading", 
            command=self.stop_live_trading,
            state=tk.DISABLED
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Manual trading buttons (for user mode)
        self.manual_frame = ttk.Frame(controls_frame, style="TFrame")
        self.manual_frame.grid(row=2, column=0, columnspan=4, padx=5, pady=5)
        
        self.buy_button = ttk.Button(
            self.manual_frame, 
            text="Buy", 
            command=self.manual_buy,
            state=tk.DISABLED
        )
        self.buy_button.pack(side=tk.LEFT, padx=5)
        
        self.sell_button = ttk.Button(
            self.manual_frame, 
            text="Sell", 
            command=self.manual_sell,
            state=tk.DISABLED
        )
        self.sell_button.pack(side=tk.LEFT, padx=5)
        
        # Quantity for manual trades
        ttk.Label(self.manual_frame, text="Quantity:", style="TLabel").pack(side=tk.LEFT, padx=5)
        self.trade_quantity = tk.IntVar(value=0)  # 0 means use available funds
        ttk.Entry(self.manual_frame, textvariable=self.trade_quantity, width=5).pack(side=tk.LEFT, padx=5)
        
        # Trading log and status
        log_label = ttk.Label(bottom_frame, text="Trading Log:", style="TLabel")
        log_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.trading_log = scrolledtext.ScrolledText(bottom_frame, height=15)
        self.trading_log.pack(fill=tk.BOTH, expand=True)
        
        # Portfolio status frame
        status_frame = ttk.LabelFrame(bottom_frame, text="Portfolio Status", style="TFrame")
        status_frame.pack(fill=tk.X, pady=5)
        
        # Portfolio value
        ttk.Label(status_frame, text="Portfolio Value:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.portfolio_value_label = ttk.Label(status_frame, text="₹100,000.00", style="TLabel")
        self.portfolio_value_label.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Current position
        ttk.Label(status_frame, text="Current Position:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
        self.position_label = ttk.Label(status_frame, text="0 shares", style="TLabel")
        self.position_label.grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
        
        # P/L
        ttk.Label(status_frame, text="Profit/Loss:", style="TLabel").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.pl_label = ttk.Label(status_frame, text="₹0.00 (0.00%)", style="TLabel")
        self.pl_label.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Last price
        ttk.Label(status_frame, text="Last Price:", style="TLabel").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
        self.price_label = ttk.Label(status_frame, text="₹0.00", style="TLabel")
        self.price_label.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
        
        # Initialize trading engine variable
        self.trading_engine = None
        self.update_thread = None
        self.is_trading = False
        
        # Initial UI update based on mode
        self.update_trading_mode()
    
    def update_trading_mode(self):
        """Update UI based on selected trading mode"""
        mode = self.trading_mode.get()
        
        if mode == "auto":
            # Machine-driven mode
            self.manual_frame.grid_remove()  # Hide manual trading controls
            self.params_frame.grid_remove()  # Hide strategy parameters
        else:
            # User-selected mode
            self.manual_frame.grid()  # Show manual trading controls
            self.params_frame.grid()  # Show strategy parameters
    
    def update_strategy_params(self, event=None):
        """Update parameter fields based on selected strategy"""
        strategy = self.live_strategy.get()
        
        # Clear existing parameter widgets
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        # Add parameters based on strategy
        if strategy == "SMA Crossover":
            ttk.Label(self.params_frame, text="Fast SMA Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            self.sma_fast_period = tk.IntVar(value=10)
            ttk.Entry(self.params_frame, textvariable=self.sma_fast_period, width=5).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            
            ttk.Label(self.params_frame, text="Slow SMA Period:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
            self.sma_slow_period = tk.IntVar(value=30)
            ttk.Entry(self.params_frame, textvariable=self.sma_slow_period, width=5).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
            
        elif strategy == "RSI":
            ttk.Label(self.params_frame, text="RSI Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            self.rsi_period = tk.IntVar(value=14)
            ttk.Entry(self.params_frame, textvariable=self.rsi_period, width=5).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            
            ttk.Label(self.params_frame, text="Oversold Level:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
            self.rsi_oversold = tk.IntVar(value=30)
            ttk.Entry(self.params_frame, textvariable=self.rsi_oversold, width=5).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
            
            ttk.Label(self.params_frame, text="Overbought Level:", style="TLabel").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            self.rsi_overbought = tk.IntVar(value=70)
            ttk.Entry(self.params_frame, textvariable=self.rsi_overbought, width=5).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
            
        elif strategy == "MACD":
            ttk.Label(self.params_frame, text="Fast Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            self.macd_fast = tk.IntVar(value=12)
            ttk.Entry(self.params_frame, textvariable=self.macd_fast, width=5).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            
            ttk.Label(self.params_frame, text="Slow Period:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
            self.macd_slow = tk.IntVar(value=26)
            ttk.Entry(self.params_frame, textvariable=self.macd_slow, width=5).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
            
            ttk.Label(self.params_frame, text="Signal Period:", style="TLabel").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            self.macd_signal = tk.IntVar(value=9)
            ttk.Entry(self.params_frame, textvariable=self.macd_signal, width=5).grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
            
        elif strategy == "Bollinger Bands":
            ttk.Label(self.params_frame, text="Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            self.bb_period = tk.IntVar(value=20)
            ttk.Entry(self.params_frame, textvariable=self.bb_period, width=5).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            
            ttk.Label(self.params_frame, text="Std Dev Factor:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
            self.bb_devfactor = tk.DoubleVar(value=2.0)
            ttk.Entry(self.params_frame, textvariable=self.bb_devfactor, width=5).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
            
        elif strategy == "Dual MA":
            ttk.Label(self.params_frame, text="MA1 Period:", style="TLabel").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
            self.ma1_period = tk.IntVar(value=10)
            ttk.Entry(self.params_frame, textvariable=self.ma1_period, width=5).grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
            
            ttk.Label(self.params_frame, text="MA2 Period:", style="TLabel").grid(row=0, column=2, padx=5, pady=5, sticky=tk.W)
            self.ma2_period = tk.IntVar(value=30)
            ttk.Entry(self.params_frame, textvariable=self.ma2_period, width=5).grid(row=0, column=3, padx=5, pady=5, sticky=tk.W)
            
            ttk.Label(self.params_frame, text="MA1 Type:", style="TLabel").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
            self.ma1_type = tk.StringVar(value="SMA")
            ma1_dropdown = ttk.Combobox(self.params_frame, textvariable=self.ma1_type, values=["SMA", "EMA"], width=5)
            ma1_dropdown.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)
            
            ttk.Label(self.params_frame, text="MA2 Type:", style="TLabel").grid(row=1, column=2, padx=5, pady=5, sticky=tk.W)
            self.ma2_type = tk.StringVar(value="EMA")
            ma2_dropdown = ttk.Combobox(self.params_frame, textvariable=self.ma2_type, values=["SMA", "EMA"], width=5)
            ma2_dropdown.grid(row=1, column=3, padx=5, pady=5, sticky=tk.W)
    
    def get_strategy_params(self):
        """Get the parameters for the selected strategy"""
        strategy = self.live_strategy.get()
        params = {'ticker': self.live_symbol.get()}
        
        if strategy == "SMA Crossover":
            params.update({
                'sma_fast_period': self.sma_fast_period.get(),
                'sma_slow_period': self.sma_slow_period.get(),
                'order_percentage': 0.95
            })
        elif strategy == "RSI":
            params.update({
                'rsi_period': self.rsi_period.get(),
                'rsi_oversold': self.rsi_oversold.get(),
                'rsi_overbought': self.rsi_overbought.get(),
                'order_percentage': 0.95
            })
        elif strategy == "MACD":
            params.update({
                'macd_fast': self.macd_fast.get(),
                'macd_slow': self.macd_slow.get(),
                'macd_signal': self.macd_signal.get(),
                'order_percentage': 0.95
            })
        elif strategy == "Bollinger Bands":
            params.update({
                'bb_period': self.bb_period.get(),
                'bb_devfactor': self.bb_devfactor.get(),
                'order_percentage': 0.95
            })
        elif strategy == "Dual MA":
            params.update({
                'ma1_period': self.ma1_period.get(),
                'ma2_period': self.ma2_period.get(),
                'ma1_type': self.ma1_type.get(),
                'ma2_type': self.ma2_type.get(),
                'order_percentage': 0.95
            })
        
        return params
    
    def start_live_trading(self):
        """Start live paper trading"""
        # Get parameters
        symbol = self.live_symbol.get()
        strategy = self.live_strategy.get()
        mode = self.trading_mode.get()
        initial_capital = self.initial_capital.get()
        commission = self.commission_rate.get() / 100.0  # Convert from percentage
        
        # Log the start
        self.log_to_trading("Starting live paper trading...")
        self.log_to_trading(f"Symbol: {symbol}")
        self.log_to_trading(f"Strategy: {strategy}")
        self.log_to_trading(f"Mode: {'Automated (Machine-Driven)' if mode == 'auto' else 'User-Selected Strategy'}")
        self.log_to_trading(f"Initial Capital: ₹{initial_capital:,.2f}")
        self.log_to_trading(f"Commission: {self.commission_rate.get()}%")
        
        # Get strategy parameters
        strategy_params = self.get_strategy_params()
        param_str = ", ".join([f"{k}: {v}" for k, v in strategy_params.items() if k != 'ticker'])
        self.log_to_trading(f"Strategy Parameters: {param_str}")
        
        # Authenticate with Fyers API
        self.log_to_trading("Authenticating with Fyers API...")
        
        try:
            # Fyers API credentials (should be stored securely in a real app)
            client_id = "NSRJ65D4YS-100"
            secret_key = "NMLEEFZHP0"
            redirect_uri = "http://127.0.0.1/"
            
            # Authenticate
            fyers_client = authenticate_fyers(client_id, secret_key, redirect_uri)
            
            if not fyers_client:
                self.log_to_trading("Authentication failed. Cannot start trading.", error=True)
                return
            
            self.log_to_trading("Authentication successful.")
            
            # Create trading engine
            self.trading_engine = PaperTradingEngine(
                fyers_client=fyers_client,
                symbol=symbol,
                strategy_name=strategy,
                strategy_params=strategy_params,
                initial_capital=initial_capital,
                commission=commission,
                mode=mode
            )
            
            # Start the engine
            if self.trading_engine.start():
                self.log_to_trading("Paper trading engine started successfully.")
                self.is_trading = True
                
                # Update UI
                self.start_button.config(state=tk.DISABLED)
                self.stop_button.config(state=tk.NORMAL)
                
                if mode == "user":
                    self.buy_button.config(state=tk.NORMAL)
                    self.sell_button.config(state=tk.NORMAL)
                
                # Start update thread
                self.start_update_thread()
            else:
                self.log_to_trading("Failed to start paper trading engine.", error=True)
                self.trading_engine = None
        
        except Exception as e:
            self.log_to_trading(f"Error starting live trading: {e}", error=True)
            self.trading_engine = None
    
    def stop_live_trading(self):
        """Stop live paper trading"""
        if not self.trading_engine or not self.is_trading:
            return
        
        self.log_to_trading("Stopping live paper trading...")
        
        try:
            # Stop the engine
            self.trading_engine.stop()
            self.is_trading = False
            
            # Get final summary
            summary = self.trading_engine.get_portfolio_summary()
            
            # Log final results
            self.log_to_trading("Trading stopped. Final results:")
            self.log_to_trading(f"Final Portfolio Value: ₹{summary['portfolio_value']:,.2f}")
            self.log_to_trading(f"Total Return: ₹{summary['absolute_return']:,.2f} ({summary['percent_return']:.2f}%)")
            self.log_to_trading(f"Total Trades: {summary['total_trades']} (Buy: {summary['buy_trades']}, Sell: {summary['sell_trades']})")
            self.log_to_trading(f"Realized P/L: ₹{summary['realized_pl']:,.2f}")
            
            # Save state
            self.trading_engine.save_state()
            self.log_to_trading("Trading state saved.")
            
            # Update UI
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.buy_button.config(state=tk.DISABLED)
            self.sell_button.config(state=tk.DISABLED)
            
            # Stop update thread
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=1)
            
            self.trading_engine = None
            
        except Exception as e:
            self.log_to_trading(f"Error stopping live trading: {e}", error=True)
    
    def manual_buy(self):
        """Execute a manual buy order"""
        if not self.trading_engine or not self.is_trading:
            return
        
        try:
            quantity = self.trade_quantity.get()
            if quantity <= 0:
                quantity = None  # Use available funds
            
            self.log_to_trading(f"Placing manual BUY order for {quantity if quantity else 'available funds'}")
            
            # Execute the order
            if self.trading_engine.place_buy_order(quantity):
                self.log_to_trading("Buy order placed successfully.")
            else:
                self.log_to_trading("Failed to place buy order.", error=True)
        
        except Exception as e:
            self.log_to_trading(f"Error placing buy order: {e}", error=True)
    
    def manual_sell(self):
        """Execute a manual sell order"""
        if not self.trading_engine or not self.is_trading:
            return
        
        try:
            quantity = self.trade_quantity.get()
            if quantity <= 0:
                quantity = None  # Sell entire position
            
            self.log_to_trading(f"Placing manual SELL order for {quantity if quantity else 'entire position'}")
            
            # Execute the order
            if self.trading_engine.place_sell_order(quantity):
                self.log_to_trading("Sell order placed successfully.")
            else:
                self.log_to_trading("Failed to place sell order.", error=True)
        
        except Exception as e:
            self.log_to_trading(f"Error placing sell order: {e}", error=True)
    
    def start_update_thread(self):
        """Start a thread to update the UI with trading status"""
        def update_worker():
            while self.is_trading and self.trading_engine:
                try:
                    # Get current status
                    summary = self.trading_engine.get_portfolio_summary()
                    
                    # Update UI with status
                    self.update_trading_status(summary)
                    
                    # Sleep for a bit
                    time.sleep(2)
                    
                except Exception as e:
                    print(f"Error in update thread: {e}")
                    time.sleep(5)
        
        self.update_thread = threading.Thread(target=update_worker)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def update_trading_status(self, summary):
        """Update the UI with current trading status"""
        # Update portfolio value
        self.portfolio_value_label.config(text=f"₹{summary['portfolio_value']:,.2f}")
        
        # Update position
        if summary['position'] > 0:
            self.position_label.config(text=f"{summary['position']} shares @ ₹{summary['entry_price']:.2f}")
        else:
            self.position_label.config(text="0 shares")
        
        # Update P/L
        if summary['percent_return'] >= 0:
            self.pl_label.config(text=f"₹{summary['absolute_return']:,.2f} (+{summary['percent_return']:.2f}%)", foreground="green")
        else:
            self.pl_label.config(text=f"₹{summary['absolute_return']:,.2f} ({summary['percent_return']:.2f}%)", foreground="red")
        
        # Update price
        self.price_label.config(text=f"₹{summary['current_price']:.2f}")
    
    def log_to_trading(self, message, error=False):
        """Add a message to the trading log"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if error:
            log_message = f"[{timestamp}] ERROR: {message}\n"
            self.trading_log.insert(tk.END, log_message, "error")
            self.trading_log.tag_config("error", foreground="red")
        else:
            log_message = f"[{timestamp}] {message}\n"
            self.trading_log.insert(tk.END, log_message)
        
        self.trading_log.see(tk.END)  # Scroll to the end
        
        # Also update status bar
        self.status_label.config(text=message)

    def load_results(self):
        """Load and display results based on selection"""
        results_type = self.results_type.get()
        
        # Clear the results frame
        for widget in self.results_frame.winfo_children():
            widget.destroy()
        
        if results_type == "ml":
            # Check if ML backtest results exist
            if os.path.exists("ml_backtest_results.png"):
                self.display_result_image("ml_backtest_results.png")
            else:
                ttk.Label(self.results_frame, text="ML backtest results not found. Run ML backtest first.", style="TLabel").pack(expand=True)
        
        elif results_type == "sma":
            # Check if SMA backtest results exist
            if os.path.exists("backtest_results.png"):
                self.display_result_image("backtest_results.png")
            else:
                ttk.Label(self.results_frame, text="SMA backtest results not found. Run SMA backtest first.", style="TLabel").pack(expand=True)
        
        elif results_type == "comparison":
            # Check if comparison results exist
            if os.path.exists("strategy_comparison.png"):
                self.display_result_image("strategy_comparison.png")
            else:
                ttk.Label(self.results_frame, text="Comparison results not found. Run strategy comparison first.", style="TLabel").pack(expand=True)
    
    def display_result_image(self, image_path):
        """Display a result image"""
        try:
            # Create a frame for the image
            image_frame = ttk.Frame(self.results_frame)
            image_frame.pack(fill=tk.BOTH, expand=True)
            
            # Load the image
            img = Image.open(image_path)
            
            # Resize the image to fit the frame
            img = self.resize_image(img, 800, 600)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)
            
            # Create a label to display the image
            image_label = ttk.Label(image_frame, image=photo)
            image_label.image = photo  # Keep a reference to prevent garbage collection
            image_label.pack(fill=tk.BOTH, expand=True)
        
        except Exception as e:
            print(f"Error displaying result image: {str(e)}")
            ttk.Label(self.results_frame, text=f"Error displaying image: {str(e)}", style="TLabel").pack(expand=True)


class TextRedirector:
    """Redirects stdout/stderr to a text widget"""
    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag
    
    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)
    
    def flush(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    app = TradingBotGUI(root)
    root.mainloop()