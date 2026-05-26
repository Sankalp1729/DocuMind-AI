FROM node:20-alpine AS build

WORKDIR /app/frontend/.streamlit

COPY frontend/.streamlit/package*.json ./

RUN npm install

COPY frontend/.streamlit ./

RUN npm run build

FROM node:20-alpine AS runtime

WORKDIR /app/frontend/.streamlit

RUN npm install -g serve

COPY --from=build /app/frontend/.streamlit/dist ./dist

EXPOSE 5173

CMD ["serve", "-s", "dist", "-l", "5173"]