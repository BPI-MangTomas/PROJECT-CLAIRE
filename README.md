# PROJECT-CLAIRE
An Intelligent Banking Assistant Designed for Secure, Inclusive, and Real-Time Customer Support in the Philippine Financial Sector
> **IMPORTANT:** These instructions are tested **only on Windows**.

## Prerequisites
Before you start, make sure you have the following installed:

### Required Software
- **Visual Studio Build Tools** with C++ support
- **Python** (latest version)
- **Node.js 18 or higher** with npm

### Check Your Installation
Verify Node.js and npm are installed:
```bash
node --version
npm --version
```

If you don't have Node.js or need to upgrade:
- Download from: https://nodejs.org/
- Choose the LTS (Long Term Support) version

## Installation Steps

### 1. Install Visual Studio Build Tools
- Download: https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022  
- Run the installer and select **Desktop development with C++** workload  
- Include these components: MSVC v143 - VS 2022 C++ x64/x86 build tools, Windows 11 SDK, CMake tools for Windows

### 2. Install Python
- Download the latest Python version from https://www.python.org/downloads/  
- **Make sure to check "Add Python to PATH"** during installation

### 3. Restart Your Device
**Restart your device** to ensure environment changes take effect.

### 4. Download and Extract Project
Download and extract the CLAIRE-PROJECT to your desired path:  
Example extraction path: `C:\Users\<YourUsername>\Desktop\PROJECT-CLAIRE`

### 5. Download Data Models
Download the pretrained data models (4 files) from Google Drive:

📂 CLAIRE Models Folder: https://drive.google.com/drive/folders/1df87MOvmy20DeR7NsWgTm7ascED4fEaw?usp=drive_link

After downloading:
   - Extract the downloaded models folder
   - Move it into your project’s /backend directory
   - Overwrite the existing models folder if prompted

Final path should look like:

**PROJECT-CLAIRE/**
   - backend/
      - models/   <-- place models here
      - start_claire.py
   - frontend/



## Project Setup

### 6. Navigate to Project Directory
Open **Command Prompt** (`cmd.exe`) and navigate to the project directory:
```cmd
cd C:\Users\<YourUsername>\Desktop\PROJECT-CLAIRE
```

### 7. Set Up Backend Environment
Create and activate a Python virtual environment:
```cmd
python -m venv project-claire-env
project-claire-env\Scripts\activate
python -m pip install --upgrade pip
```

Install Python dependencies:
```cmd
pip install -r requirements.txt
```

### 8. Set Up Frontend Environment
Navigate to the frontend directory and install dependencies:
```cmd
cd frontend
npm install
```

This will:
- Read the package.json file
- Download and install all required packages (React, Vite, and development tools)
- Create a node_modules folder with all dependencies

## Running the Application

### 9. Start the Backend Service
From the project root directory, start the backend:
```cmd
cd backend
python start_claire.py
```
Keep this terminal window open - the backend service needs to stay running.

### 10. Start the Frontend Development Server
Open a **new Command Prompt window** and navigate to the frontend directory:
```cmd
cd C:\Users\<YourUsername>\Desktop\PROJECT-CLAIRE\frontend
npm run dev
```

This will:
- Start the Vite development server
- Open your app (usually at http://localhost:5173)
- Enable hot reload (changes appear instantly)


## Usage
1. Ensure both services are running:
   - Backend service (Step 8)
   - Frontend development server (Step 9)
2. Open your browser and navigate to the frontend URL (typically http://localhost:5173)
3. The frontend will communicate with the backend API to provide full functionality

## Troubleshooting
- **Python errors:** Ensure Python is properly added to your PATH
- **Node.js commands don't work:** Restart your command prompt after Node.js installation
- **Connection issues:** Make sure both backend and frontend servers are running simultaneously
- **Port conflicts:** Check that the default ports (backend and frontend) are not being used by other applications
- **Dependencies issues:** Try deleting `node_modules` folder and running `npm install` again
- **Models not found:** Verify that the /backend/models folder exists and contains the downloaded files

## Development Notes
- The backend runs the API service that handles data processing
- The frontend provides the user interface and communicates with the backend
- Both services must be running for the application to work properly
- Changes to frontend code will automatically refresh in the browser
- Backend changes may require restarting the backend service
- Pretrained models must be present in /backend/models for full functionality



