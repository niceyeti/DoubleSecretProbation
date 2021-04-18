package main

import (
	"io"
	"log"
	"net/http"
)

type myHandler struct{}

func (*myHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	io.WriteString(w, "URL: "+r.URL.String())
}

func Tmp(w http.ResponseWriter, r *http.Request) {
	io.WriteString(w, "version 2")
}


func main(){
	mux := http.NewServeMux()

	// Register routes and register handlers in this form.
	mux.Handle("/",&myHandler{})
	
	mux.HandleFunc("/tmp", Tmp)
	
	//http.ListenAndServe uses the default server structure.
	err := http.ListenAndServe(":8080", mux)
	if err != nil {
		log.Fatal(err)
	}
}