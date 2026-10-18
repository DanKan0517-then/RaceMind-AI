"""Run the RaceMind AI web server locally:  python main.py  ->  http://localhost:7860"""

import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "7860")), reload=False)
