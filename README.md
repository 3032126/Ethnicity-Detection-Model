﻿# Ethnicity-Detection-Model
MongoDB : https://www.mongodb.com/try/download/community
NodeJS : https://nodejs.org/en/learn/getting-started/how-to-install-nodejs
Server AI (python)
ลิงค์ model copyไฟล์ "trained_model_e100_b64.keras" 
https://drive.google.com/drive/folders/19VzJYWxqV1cDQgrY8sSFwTKcjF0I8xVx

สร้าง folder "app" เก็บเอา model และ app.py มาวาง

install
pip install fastapi uvicorn tensorflow pillow

run
uvicorn app:app --reload

server จะรันด้วย port 8000
 
========================================================================

Server Web

วิธีรัน
รันตัว server Ai ให้ได้ก่อน เพราะว่าตัวนี้จะต่อ API ไปหา server Ai ด้วย
ลง mongodb ในเครื่องก่อน + create DB ชื่อ webDB
รันคำสั่ง npm start
ลองเล่นได้เลย
