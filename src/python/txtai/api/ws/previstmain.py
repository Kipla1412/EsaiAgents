# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# from .previsit import register_medical_ws
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse
# import os
# app = FastAPI()


# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
# @app.get("/download/pdf/{patient_id}")
# async def download_pdf(patient_id: str):

#     pdf_path = f"reports/{patient_id}_soap_report.pdf"

#     if not os.path.exists(pdf_path):
#         raise HTTPException(status_code=404, detail="PDF not found")

#     return FileResponse(
#         pdf_path,
#         media_type="application/pdf",
#         filename=f"{patient_id}_soap_report.pdf"
#     )


# register_medical_ws(app)
