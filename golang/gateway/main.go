
/*
An http api with authentication, roles, and db backend.

Design: the http server consists of a series of middlewares for handling requests.package main
The first stage is authorization: is the requester known and permitted at all.
The second stage is authorization: does the requester have access to the desired resource.
Finally, proxy the request to the db and return results, like Fluent in c#.

The point is to monkey around with Go: security, tools/testing, concurrency and other goodies.
*/


package main


import (
	"encoding/json"
	"log"
	"net/http"
	//"os"
	//"path/filepath"
	"time"
	"github.com/gorilla/mux"
)


func main() {
	router := mux.NewRouter()

	router.HandleFunc("/json/endpoint", func(w http.ResponseWriter, r *http.Request) {
		// an example API handler
		json.NewEncoder(w).Encode(map[string]bool{"yup": true})
	})

	srv := &http.Server{
		Handler: router,
		Addr:    "127.0.0.1:8000",
		WriteTimeout: 15 * time.Second,
		ReadTimeout:  15 * time.Second,
	}

	log.Fatal(srv.ListenAndServe())
}


