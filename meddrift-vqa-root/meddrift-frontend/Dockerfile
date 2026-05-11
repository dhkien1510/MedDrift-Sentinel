# Sử dụng Node 20 thay vì 18
FROM node:20-alpine 
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
# Vite cần mở cổng này để Docker mapping được
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host"]