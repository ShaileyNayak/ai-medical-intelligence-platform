# Frontend — multi-stage Vite build served by nginx
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .

# Prefer absolute backend origin for Vercel / split deploys.
# Leave unset in Compose so build falls back to http://localhost:8000
# (or set VITE_API_URL / VITE_API_BASE_URL explicitly).
ARG VITE_API_URL=
ARG VITE_API_BASE_URL=
ENV VITE_API_URL=$VITE_API_URL
ENV VITE_API_BASE_URL=$VITE_API_BASE_URL

RUN npm run build

FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://127.0.0.1/ >/dev/null || exit 1

CMD ["nginx", "-g", "daemon off;"]
