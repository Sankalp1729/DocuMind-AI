FROM node:20-alpine

WORKDIR /app/frontend/.streamlit

COPY frontend/.streamlit/package*.json ./

RUN npm install

COPY frontend/.streamlit ./

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0", "--port", "5173"]