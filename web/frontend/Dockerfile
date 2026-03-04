FROM node:20-alpine

WORKDIR /app

COPY web/frontend/package.json .

RUN npm install

COPY web/frontend/ .

EXPOSE 3000

CMD ["npm", "run", "dev"]
