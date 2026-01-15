#!/bin/bash
# VRCI Platform Stop Script
# Contact: admin@gy4k.com

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Stopping VRCI Platform...${NC}"

# Read PIDs
if [ -f ".backend.pid" ]; then
    BACKEND_PID=$(cat .backend.pid)
    kill $BACKEND_PID 2>/dev/null && echo -e "${GREEN}✓ Backend stopped (PID: $BACKEND_PID)${NC}" || echo -e "${YELLOW}⚠ Backend not running${NC}"
    rm -f .backend.pid
fi

if [ -f ".frontend.pid" ]; then
    FRONTEND_PID=$(cat .frontend.pid)
    kill $FRONTEND_PID 2>/dev/null && echo -e "${GREEN}✓ Frontend stopped (PID: $FRONTEND_PID)${NC}" || echo -e "${YELLOW}⚠ Frontend not running${NC}"
    rm -f .frontend.pid
fi

# Force kill any remaining processes on ports
lsof -ti :8001 | xargs kill -9 2>/dev/null
lsof -ti :8080 | xargs kill -9 2>/dev/null

echo -e "${GREEN}✓ Platform stopped${NC}"
