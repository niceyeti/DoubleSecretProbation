FROM golang:1.17-alpine

WORKDIR ./

COPY go.mod ./
COPY go.sum ./
RUN go mod download

COPY *.go ./
RUN go build -o /server

EXPOSE 8080
CMD ["./server"]
