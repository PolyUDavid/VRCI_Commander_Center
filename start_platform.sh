#!/bin/bash
# VRCI Platform Startup Script
# Author: VRCI Research Team
# Contact: admin@gy4k.com
# Version: 1.0.0

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         VRCI Platform Startup Script v1.0.0               ║${NC}"
echo -e "${BLUE}║    Decentralized Vehicle-Road-Cloud Integration           ║${NC}"
echo -e "${BLUE}║                                                            ║${NC}"
echo -e "${BLUE}║    Contact: admin@gy4k.com                                 ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
port_in_use() {
    lsof -i :"$1" >/dev/null 2>&1
}

# Function to kill process on port
kill_port() {
    local port=$1
    if port_in_use "$port"; then
        echo -e "${YELLOW}⚠ Port $port is in use, attempting to free it...${NC}"
        lsof -ti :"$port" | xargs kill -9 2>/dev/null || true
        sleep 2
    fi
}

# Check Python version
echo -e "${BLUE}[1/7] Checking Python version...${NC}"
if ! command_exists python3; then
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.11+${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo -e "${GREEN}✓ Found Python $PYTHON_VERSION${NC}"

# Check virtual environment
echo -e "${BLUE}[2/7] Checking virtual environment...${NC}"
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${BLUE}[3/7] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Check dependencies
echo -e "${BLUE}[4/7] Checking dependencies...${NC}"
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠ PyTorch not found. Installing dependencies...${NC}"
    pip install -q --upgrade pip
    pip install -q -r requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${GREEN}✓ Dependencies already installed${NC}"
fi

# Check CUDA availability
echo -e "${BLUE}[5/7] Checking GPU availability...${NC}"
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✓ CUDA available: $GPU_NAME${NC}"
else
    echo -e "${YELLOW}⚠ CUDA not available. Running on CPU (slower performance)${NC}"
fi

# Check model files
echo -e "${BLUE}[6/7] Checking model files...${NC}"
MODEL_COUNT=$(ls backend/models/*.pth 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 5 ]; then
    echo -e "${GREEN}✓ All 5 model files found${NC}"
elif [ "$MODEL_COUNT" -gt 0 ]; then
    echo -e "${YELLOW}⚠ Only $MODEL_COUNT/5 model files found${NC}"
    echo -e "${YELLOW}  Some features may not work. Contact admin@gy4k.com for model files.${NC}"
else
    echo -e "${YELLOW}⚠ No model files found${NC}"
    echo -e "${YELLOW}  Platform will run in limited mode. Contact admin@gy4k.com for model files.${NC}"
fi

# Free ports if in use
echo -e "${BLUE}[7/7] Preparing ports...${NC}"
kill_port 8001
kill_port 8080
echo -e "${GREEN}✓ Ports 8001 and 8080 are ready${NC}"

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All checks passed! Starting VRCI Platform...${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""

# Start backend API server
echo -e "${BLUE}Starting Backend API Server (port 8001)...${NC}"
cd backend
python api_server_ai.py > ../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo -e "${YELLOW}Waiting for backend to initialize...${NC}"
sleep 5

# Check if backend is running
if ! port_in_use 8001; then
    echo -e "${RED}✗ Backend failed to start. Check logs/backend.log${NC}"
    kill $BACKEND_PID 2>/dev/null || true
    exit 1
fi
echo -e "${GREEN}✓ Backend API Server running (PID: $BACKEND_PID)${NC}"

# Start frontend server
echo -e "${BLUE}Starting Frontend Server (port 8080)...${NC}"
cd frontend
python -m http.server 8080 > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to start
sleep 2

# Check if frontend is running
if ! port_in_use 8080; then
    echo -e "${RED}✗ Frontend failed to start. Check logs/frontend.log${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 1
fi
echo -e "${GREEN}✓ Frontend Server running (PID: $FRONTEND_PID)${NC}"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                  VRCI Platform Ready!                      ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  Dashboard:  ${BLUE}http://localhost:8080/dashboard_ultimate.html${GREEN}  ║${NC}"
echo -e "${GREEN}║  API Docs:   ${BLUE}http://localhost:8001/docs${GREEN}                     ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  Backend PID:  $BACKEND_PID                                     ║${NC}"
echo -e "${GREEN}║  Frontend PID: $FRONTEND_PID                                     ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  Logs:                                                     ║${NC}"
echo -e "${GREEN}║    Backend:  logs/backend.log                              ║${NC}"
echo -e "${GREEN}║    Frontend: logs/frontend.log                             ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  To stop:    ./stop_platform.sh                            ║${NC}"
echo -e "${GREEN}║              or press Ctrl+C                               ║${NC}"
echo -e "${GREEN}║                                                            ║${NC}"
echo -e "${GREEN}║  Support:    admin@gy4k.com                                ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Save PIDs for stop script
echo "$BACKEND_PID" > .backend.pid
echo "$FRONTEND_PID" > .frontend.pid

# Open browser (optional)
if command_exists open; then
    echo -e "${BLUE}Opening dashboard in browser...${NC}"
    sleep 2
    open "http://localhost:8080/dashboard_ultimate.html" 2>/dev/null || true
elif command_exists xdg-open; then
    echo -e "${BLUE}Opening dashboard in browser...${NC}"
    sleep 2
    xdg-open "http://localhost:8080/dashboard_ultimate.html" 2>/dev/null || true
fi

# Wait for user interrupt
echo -e "${YELLOW}Press Ctrl+C to stop the platform${NC}"
trap "echo ''; echo -e '${YELLOW}Shutting down...${NC}'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm -f .backend.pid .frontend.pid; echo -e '${GREEN}✓ Platform stopped${NC}'; exit 0" INT

# Keep script running
wait
