================================================================================
              AI-POWERED MUTUAL FUND ADVISOR - SETUP GUIDE FOR SIRAJ
================================================================================

Project: AI-Powered Mutual Fund Investment Advisor
Created by: Your Team
Model: Google Gemma-2B with LoRA Fine-tuning
Data: 20 years of historical data from 200+ mutual funds

This guide will help you set up and run this project on your laptop from scratch.

================================================================================
                        TABLE OF CONTENTS
================================================================================

1. PROJECT OVERVIEW
2. WHAT YOU'LL NEED (PREREQUISITES)
3. STEP-BY-STEP INSTALLATION GUIDE
4. HOW TO RUN THE PROJECT
5. UNDERSTANDING THE PROJECT STRUCTURE
6. TROUBLESHOOTING COMMON ISSUES
7. FREQUENTLY ASKED QUESTIONS


================================================================================
                        1. PROJECT OVERVIEW
================================================================================

This is an AI-powered mutual fund advisor that:
- Analyzes 20 years of historical data from 200+ Indian mutual funds
- Uses Google's Gemma-2B AI model to generate personalized recommendations
- Provides 3 investment strategies based on risk tolerance (Low, Moderate, High)
- Offers a user-friendly web interface to get investment advice

WHAT IT DOES:
You enter your:
- Age
- Investment amount
- Investment horizon (how long you want to invest)
- Risk tolerance (Low, Moderate, High)
- Financial goal (Wealth Creation, Retirement, etc.)

It gives you:
- Personalized fund recommendations
- Expected returns
- Portfolio allocation percentages
- SIP (Systematic Investment Plan) suggestions
- Tax optimization strategies


================================================================================
                   2. WHAT YOU'LL NEED (PREREQUISITES)
================================================================================

HARDWARE REQUIREMENTS:
----------------------
MINIMUM:
- 8GB RAM (16GB recommended)
- 10GB free disk space (20GB recommended)
- 2 CPU cores (4+ recommended)
- Stable internet connection (for initial setup only)

OPTIMAL (for faster training):
- 16GB+ RAM
- 20GB+ free disk space
- NVIDIA GPU with 8GB+ VRAM (optional, speeds up training)

SOFTWARE REQUIREMENTS:
---------------------
1. Python 3.11 or higher
   - Download from: https://www.python.org/downloads/
   - IMPORTANT: During installation, check "Add Python to PATH"

2. Git (optional, if cloning from repository)
   - Download from: https://git-scm.com/downloads

3. Web Browser (Chrome, Firefox, Edge - any modern browser)

4. Hugging Face Account (FREE)
   - Create account at: https://huggingface.co/join
   - We need this to download the AI model


================================================================================
              3. STEP-BY-STEP INSTALLATION GUIDE
================================================================================

STEP 1: VERIFY PYTHON INSTALLATION
-----------------------------------
1. Open Command Prompt (Windows) or Terminal (Mac/Linux)
   - Windows: Press Win+R, type "cmd", press Enter
   - Mac: Press Cmd+Space, type "terminal", press Enter
   - Linux: Press Ctrl+Alt+T

2. Check Python version:
   Type: python --version
   OR: python3 --version

   You should see something like: "Python 3.11.x" or "Python 3.12.x"

   If you see "command not found" or version is less than 3.11:
   - Download and install Python 3.11+ from https://www.python.org/downloads/
   - Remember to check "Add Python to PATH" during installation


STEP 2: EXTRACT PROJECT FILES
------------------------------
1. Extract the project folder to a location on your laptop
   Recommended: C:\MF_Advisor_Project (Windows)
             OR ~/MF_Advisor_Project (Mac/Linux)

2. Navigate to the project directory:
   Windows: cd "C:\MF_Advisor_Project"
   Mac/Linux: cd ~/MF_Advisor_Project

   OR just right-click the folder and select "Open in Terminal"


STEP 3: CREATE VIRTUAL ENVIRONMENT
-----------------------------------
A virtual environment keeps project dependencies isolated.

1. Create virtual environment:
   Type: python -m venv venv

   Wait for it to complete (takes 1-2 minutes)

2. Activate the virtual environment:

   Windows:
   Type: venv\Scripts\activate

   Mac/Linux:
   Type: source venv/bin/activate

   You should see (venv) appear at the start of your command prompt.
   This means the virtual environment is active.

NOTE: You need to activate the virtual environment EVERY TIME you work on
      this project in a new terminal/command prompt window.


STEP 4: GET HUGGING FACE TOKEN
-------------------------------
We need this to download the Gemma-2B AI model.

1. Go to: https://huggingface.co/join and create a FREE account

2. After logging in, go to: https://huggingface.co/settings/tokens

3. Click "New token"
   - Name: "MF_Advisor"
   - Type: Select "Read"
   - Click "Generate token"

4. COPY the token (it looks like: hf_xxxxxxxxxxxxxxxxxxxx)

5. Go to: https://huggingface.co/google/gemma-2b
   Click "Agree and access repository" to accept the license

6. Create a file named ".env" in the project folder:

   Windows (in project folder):
   - Create new text file
   - Rename it to ".env" (no .txt extension)
   - Edit the file and add this line:
     HUGGINGFACE_HUB_TOKEN=hf_your_token_here
     (Replace "hf_your_token_here" with your actual token)

   Mac/Linux (in terminal, in project folder):
   Type: echo "HUGGINGFACE_HUB_TOKEN=hf_your_token_here" > .env
         (Replace "hf_your_token_here" with your actual token)


STEP 5: INSTALL DEPENDENCIES
-----------------------------
This installs all required Python packages.

1. Make sure virtual environment is activated (you see "(venv)" in prompt)

2. Type: pip install -r requirements.txt

   This will take 5-15 minutes depending on your internet speed.
   You'll see many packages being downloaded and installed.

3. Verify installation:
   Type: python -c "import transformers, torch, fastapi; print('All imports successful!')"

   If you see "All imports successful!" - you're good!
   If you see errors, check Troubleshooting section below.


================================================================================
                      4. HOW TO RUN THE PROJECT
================================================================================

You have TWO options:

┌────────────────────────────────────────────────────────────────────────┐
│              OPTION 1: AUTOMATED SETUP (RECOMMENDED)                   │
│              Total Time: 2-5 hours (mostly automated)                  │
└────────────────────────────────────────────────────────────────────────┘

This is the EASIEST way. One command does everything for you!

STEP 1: Make sure you're in the project folder with venv activated
        You should see (venv) in your command prompt

STEP 2: Run the automated setup:
        Type: python setup_and_train_complete.py

WHAT THIS DOES (automatically):
- Installs any missing dependencies
- Downloads 20 years of historical mutual fund data (15-45 minutes)
- Generates 25,000 training examples (5-10 minutes)
- Trains the AI model (1-4 hours on CPU, 30-60 min on GPU)
- Sets up everything ready to use

You can grab lunch/coffee while this runs! It will show progress messages.

STEP 3: After setup completes (when you see "Setup completed successfully!"):
        Type: python run_server.py

STEP 4: Open your web browser and go to: http://localhost:8000

DONE! You should see the Mutual Fund Advisor interface.


┌────────────────────────────────────────────────────────────────────────┐
│              OPTION 2: QUICK SETUP (FOR TESTING/DEMO)                  │
│              Total Time: 10-15 minutes                                 │
└────────────────────────────────────────────────────────────────────────┘

If you want to test the system quickly without waiting hours:

STEP 1: Type: python setup_and_train_complete.py --quick

This uses a smaller model and less data - much faster but less accurate.

STEP 2: Type: python run_server.py

STEP 3: Open browser: http://localhost:8000


┌────────────────────────────────────────────────────────────────────────┐
│              OPTION 3: MANUAL STEP-BY-STEP                             │
│              (Only if you want full control)                           │
└────────────────────────────────────────────────────────────────────────┘

If automated setup fails or you want to do steps manually:

STEP A: Collect Historical Data (15-45 minutes)
Type: python -c "import asyncio; from src.enhanced_20_year_collector import Enhanced20YearCollector; collector = Enhanced20YearCollector(); asyncio.run(collector.collect_20_year_data(max_funds=200))"

This creates: data/historical_20y.db (a database file with all fund data)

STEP B: Train the AI Model (1-4 hours)
Type: python train_enhanced_model.py

This creates: models/enhanced_20y_advisor/ (the trained AI model)

STEP C: Start the Server
Type: python run_server.py

STEP D: Open browser: http://localhost:8000


================================================================================
              5. UNDERSTANDING THE PROJECT STRUCTURE
================================================================================

Here's what each folder/file does:

MF_Advisor_Project/
│
├── api/                          API server code
│   └── app.py                    Main FastAPI application
│
├── backtesting/                  Strategy testing tools
│   └── mf_backtest_engine.py     Backtest framework
│
├── data/                         Data storage
│   └── historical_20y.db         Database with 20 years of fund data
│
├── evaluation/                   Model evaluation tools
│   └── model_evaluator.py        Evaluates prediction accuracy
│
├── logs/                         Log files (check here if errors occur)
│
├── models/                       Trained AI models
│   └── enhanced_20y_advisor/     Main production model
│
├── reports/                      Generated reports (backtesting, etc.)
│
├── src/                          Source code
│   ├── enhanced_20_year_collector.py    Data collection script
│   ├── generate_gemma_training_data.py  Training data generator
│   ├── inference_stable.py              AI recommendation engine
│   └── config.py                        Configuration settings
│
├── web/                          Web interface
│   └── index.html                User interface
│
├── venv/                         Virtual environment (don't modify)
│
├── .env                          Environment variables (YOUR HUGGING FACE TOKEN)
├── requirements.txt              List of required Python packages
├── setup_and_train_complete.py  Automated setup script
├── train_enhanced_model.py       Model training script
└── run_server.py                 Start the web server


KEY FILES YOU'LL USE:
- setup_and_train_complete.py    Run this first to set everything up
- run_server.py                   Run this to start the web interface
- .env                            Contains your Hugging Face token


================================================================================
                  6. TROUBLESHOOTING COMMON ISSUES
================================================================================

PROBLEM 1: "python: command not found" or "Python is not recognized"
---------------------------------------------------------------------
SOLUTION:
- Python is not installed or not in PATH
- Download and install Python 3.11+ from https://www.python.org/downloads/
- During installation, CHECK "Add Python to PATH"
- After installation, restart your terminal/command prompt


PROBLEM 2: "No module named 'transformers'" or other import errors
--------------------------------------------------------------------
SOLUTION:
- Virtual environment not activated OR dependencies not installed
- Make sure you see (venv) in your command prompt
- If not, activate it: venv\Scripts\activate (Windows) or source venv/bin/activate (Mac/Linux)
- Then run: pip install -r requirements.txt


PROBLEM 3: "Database not found: data/historical_20y.db"
--------------------------------------------------------
SOLUTION:
- Data collection hasn't run yet
- Run: python setup_and_train_complete.py
- OR manually collect data (see Option 3, Step A above)


PROBLEM 4: "Token is invalid" or "401 Unauthorized" from Hugging Face
-----------------------------------------------------------------------
SOLUTION:
- Your Hugging Face token is wrong or not set
- Check .env file exists in project folder
- Check it contains: HUGGINGFACE_HUB_TOKEN=hf_your_actual_token
- Get new token from: https://huggingface.co/settings/tokens
- Accept Gemma-2B license: https://huggingface.co/google/gemma-2b


PROBLEM 5: Training is too slow or "Out of Memory"
---------------------------------------------------
SOLUTION A: Use Quick Training
- Type: python quick_train.py
- This uses a smaller model (5-10 minutes instead of hours)

SOLUTION B: Reduce memory usage
- Open train_enhanced_model.py in a text editor
- Find line with "per_device_train_batch_size = 2"
- Change to "per_device_train_batch_size = 1"
- Save and try again

SOLUTION C: Close other applications
- Close web browsers, video players, games, etc.
- This frees up RAM for training


PROBLEM 6: "Address already in use" when running server
--------------------------------------------------------
SOLUTION:
- Port 8000 is being used by another program

Windows:
- Find process using port 8000: netstat -ano | findstr :8000
- Kill it: taskkill /PID <number> /F
  (Replace <number> with the number you saw)

Mac/Linux:
- Find and kill process: lsof -ti:8000 | xargs kill

OR change the port:
- Edit run_server.py
- Change "port=8000" to "port=8001"
- Then access: http://localhost:8001


PROBLEM 7: Setup is taking too long
------------------------------------
SOLUTION:
- Use quick setup: python setup_and_train_complete.py --quick
- OR reduce data: python setup_and_train_complete.py --max-funds 50
  (Collects only 50 funds instead of 200)


PROBLEM 8: Web page shows "Cannot connect to server"
-----------------------------------------------------
SOLUTION:
- Server is not running
- Go to terminal, make sure you ran: python run_server.py
- Server should show: "Uvicorn running on http://0.0.0.0:8000"
- Then try accessing: http://localhost:8000 in browser


================================================================================
                   7. FREQUENTLY ASKED QUESTIONS
================================================================================

Q1: How long does the complete setup take?
A: First time: 2-5 hours (data collection + training)
   Quick mode: 10-15 minutes
   Once done, starting server takes only 5-10 seconds

Q2: Do I need internet after initial setup?
A: No! After setup, everything works offline

Q3: Do I need a GPU?
A: No, but GPU makes training MUCH faster (4 hours → 30 minutes)
   The system works perfectly fine on CPU

Q4: Can I use this on Mac/Linux?
A: Yes! All commands work the same, just use "/" instead of "\"
   Example: source venv/bin/activate instead of venv\Scripts\activate

Q5: How do I use the system after it's set up?
A:
1. Open terminal in project folder
2. Activate venv: venv\Scripts\activate
3. Start server: python run_server.py
4. Open browser: http://localhost:8000
5. Fill in the form and click "Get Recommendations"

Q6: Can I run this on Windows 11?
A: Yes! Works on Windows 10, 11, Mac, and Linux

Q7: What if I don't have Hugging Face token?
A: You MUST create a free account and get a token
   It's required to download the Gemma-2B model
   Takes only 2 minutes: https://huggingface.co/join

Q8: Is this safe to use for real investments?
A: This is an EDUCATIONAL PROJECT for learning AI/ML
   Always consult with SEBI-registered financial advisors for real investments
   Don't make investment decisions based solely on this tool

Q9: Can I modify the code?
A: Yes! It's a learning project
   Key files to explore:
   - src/inference_stable.py (recommendation logic)
   - api/app.py (API endpoints)
   - web/index.html (user interface)

Q10: How do I stop the server?
A: Press Ctrl+C in the terminal where server is running

Q11: How do I update the data?
A: Run data collection again:
   python -c "import asyncio; from src.enhanced_20_year_collector import Enhanced20YearCollector; collector = Enhanced20YearCollector(); asyncio.run(collector.collect_20_year_data())"

Q12: What if I get stuck?
A:
1. Check logs/ folder for error messages
2. Check this README's troubleshooting section
3. Try quick setup: python setup_and_train_complete.py --quick
4. Google the error message
5. Ask ChatGPT or Claude for help with the error


================================================================================
                      QUICK START CHEAT SHEET
================================================================================

FIRST TIME SETUP:
-----------------
1. Install Python 3.11+
2. Extract project folder
3. Open terminal in project folder
4. python -m venv venv
5. venv\Scripts\activate        (Windows)
   OR source venv/bin/activate  (Mac/Linux)
6. pip install -r requirements.txt
7. Create .env file with Hugging Face token
8. python setup_and_train_complete.py
9. python run_server.py
10. Open: http://localhost:8000


EVERY TIME YOU WANT TO USE IT:
-------------------------------
1. Open terminal in project folder
2. venv\Scripts\activate        (Windows)
   OR source venv/bin/activate  (Mac/Linux)
3. python run_server.py
4. Open: http://localhost:8000


QUICK TEST (Skip long training):
---------------------------------
1-7. (Same as first time setup)
8. python setup_and_train_complete.py --quick
9. python run_server.py
10. Open: http://localhost:8000


================================================================================
                         SYSTEM REQUIREMENTS SUMMARY
================================================================================

OPERATING SYSTEM:
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 20.04+, or equivalent)

MINIMUM SPECS:
- Python 3.11 or higher
- 8GB RAM
- 10GB free disk space
- 2 CPU cores
- Internet (for setup only)

RECOMMENDED SPECS:
- Python 3.11+
- 16GB RAM
- 20GB free disk space
- 4+ CPU cores
- GPU (optional, speeds up training)

TIME INVESTMENT:
- Setup time: 15 minutes (manual steps) + 2-5 hours (automated training)
- OR: 15 minutes total (quick mode)
- Daily use: 5 seconds (start server) + instant (get recommendations)


================================================================================
                          WHAT TO EXPECT
================================================================================

DURING SETUP (Automated):
You'll see messages like:
- "Installing dependencies..."
- "Collecting data from 200 funds..."
- "Downloaded NAV data for [Fund Name]..."
- "Generating training data..."
- "Training model - Epoch 1/4..."
- "Setup completed successfully!"

This is normal! Just wait for it to complete.

WHEN SERVER IS RUNNING:
You'll see:
- "INFO:     Uvicorn running on http://0.0.0.0:8000"
- "INFO:     Started server process"

This means it's working! Open http://localhost:8000

USING THE WEB INTERFACE:
1. Fill in your details (age, amount, etc.)
2. Click "Get Recommendations"
3. Wait 5-10 seconds
4. See 3 strategies: Low Risk, Moderate Risk, High Risk
5. Each shows:
   - Fund names
   - Allocation percentages
   - Expected returns
   - SIP amounts
   - Rationale


================================================================================
                          IMPORTANT NOTES
================================================================================

1. ALWAYS activate virtual environment before running any command
   You should see (venv) at the start of your prompt

2. The first setup takes time (2-5 hours) - this is NORMAL
   Once done, you never have to do it again
   Starting the system daily takes only 5 seconds

3. Keep your Hugging Face token PRIVATE
   Don't share your .env file with anyone

4. Close the server properly (Ctrl+C) before closing terminal
   Don't just close the terminal window

5. If you move the project folder, you'll need to create a new venv

6. The database (data/historical_20y.db) and model (models/enhanced_20y_advisor/)
   are the most important files - don't delete them!

7. You can safely delete and recreate the venv folder if needed
   Just run: python -m venv venv again and reinstall dependencies


================================================================================
                       VERIFICATION CHECKLIST
================================================================================

After setup, verify everything is working:

□ Python version 3.11+ installed
  Check: python --version

□ Virtual environment created
  Check: Folder "venv" exists in project directory

□ Dependencies installed
  Check: python -c "import transformers, torch, fastapi; print('OK')"

□ Hugging Face token set
  Check: .env file exists with HUGGINGFACE_HUB_TOKEN

□ Data collected
  Check: File "data/historical_20y.db" exists and is > 50MB

□ Model trained
  Check: Folder "models/enhanced_20y_advisor" exists with files inside

□ Server starts successfully
  Check: python run_server.py shows "Uvicorn running on http://0.0.0.0:8000"

□ Web interface accessible
  Check: http://localhost:8000 opens in browser and shows form

□ Recommendations work
  Check: Fill form, click button, get 3 strategies displayed


================================================================================
                         GETTING HELP
================================================================================

If you're stuck:

1. Read the troubleshooting section above
2. Check the logs/ folder for error details
3. Try the quick setup mode
4. Search the error message on Google
5. Ask an AI assistant (ChatGPT, Claude) with the error message
6. Check PROJECT_SETUP_GUIDE.txt for more details
7. Check PROJECT_EXECUTION_GUIDE.txt for advanced options


================================================================================
                          FINAL TIPS
================================================================================

1. Be patient during first setup - it's downloading GBs of data and models
2. Use quick mode (--quick flag) if you just want to test/demo
3. Once set up, the system is very fast to start and use
4. The AI gives suggestions, not financial advice - always consult professionals
5. Have fun exploring the code and learning!


================================================================================
                        SUCCESS INDICATORS
================================================================================

YOU'VE SUCCESSFULLY SET UP THE PROJECT WHEN:

✓ You can run: python run_server.py without errors
✓ Browser shows the investment advisor form at http://localhost:8000
✓ You can enter investment details and get 3 strategy recommendations
✓ Each strategy shows multiple mutual funds with allocations
✓ Expected returns are calculated for each strategy
✓ No error messages appear in terminal or browser


================================================================================
                   COMMANDS SUMMARY REFERENCE
================================================================================

CREATE VIRTUAL ENVIRONMENT:
  python -m venv venv

ACTIVATE VIRTUAL ENVIRONMENT:
  Windows: venv\Scripts\activate
  Mac/Linux: source venv/bin/activate

INSTALL DEPENDENCIES:
  pip install -r requirements.txt

AUTOMATED SETUP (Full):
  python setup_and_train_complete.py

AUTOMATED SETUP (Quick):
  python setup_and_train_complete.py --quick

START SERVER:
  python run_server.py

ACCESS WEB INTERFACE:
  http://localhost:8000

STOP SERVER:
  Press Ctrl+C in terminal

DEACTIVATE VIRTUAL ENVIRONMENT:
  deactivate


================================================================================
                         PROJECT CREDITS
================================================================================

This project uses:
- Google Gemma-2B (AI Model)
- FastAPI (Web Framework)
- Transformers + PEFT (AI Training)
- LoRA (Efficient Fine-tuning)
- Historical data from MFAPI.in and Yahoo Finance
- Python 3.11+


================================================================================
                            GOOD LUCK!
================================================================================

You're all set to run the AI-Powered Mutual Fund Advisor!

If you followed this guide, you should have:
✓ A working virtual environment
✓ All dependencies installed
✓ Historical data collected
✓ AI model trained
✓ Web server running
✓ Recommendations working

Enjoy exploring the project!

For questions or issues, refer to the troubleshooting section.

Happy Investing (and Learning)!

================================================================================
                         END OF SETUP GUIDE
================================================================================
Last Updated: November 2024
Version: 1.0 (Siraj Edition)
