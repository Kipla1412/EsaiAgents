import os
import uvicorn

os.environ["CONFIG"] = r"D:\backend\txtai\src\python\config.yml" 
if __name__ == "__main__":
    uvicorn.run("txtai.api.application:app", host="127.0.0.1", port=8000, reload=True)
