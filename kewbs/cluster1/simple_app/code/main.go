package main

/*
For git and shiggles: a very basic web api to build as a 'webapp' for some helm deployment tutorials.
*/

import (
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"os/exec"

	//"net/http/test"

	"github.com/gorilla/mux"
)

func EchoHandler(w http.ResponseWriter, r *http.Request) {
	b, err := ioutil.ReadAll(r.Body)
	fmt.Println("Echoing: " + string(b))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	w.Header().Set("Content-Type", "text/plain")

	b = append(b, "\n"...)
	w.Write(b)

	w.WriteHeader(http.StatusOK)
}

func FortuneHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Fortune hit")
	w.Header().Set("Content-Type", "text/plain")
	cmd := exec.Command("fortune")
	cmd.Stdout = w // this is efficient, but should determine any command errors before writing. this will write arbitrary failure data when non-200 responses are returned.
	if err := cmd.Run(); err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.WriteHeader(http.StatusOK)
}

func HealthHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Health hit")
	/*
		Optional introspective health checks could go here...
	*/
	w.WriteHeader(http.StatusOK)
}

func RootHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Println("Hit root")
	w.Header().Set("Content-Type", "text/html")
	http.FileServer(http.Dir("./static")).ServeHTTP(w, r)
}

func main() {
	fmt.Println("starting super awesome app...")

	r := mux.NewRouter()
	r.HandleFunc("/", RootHandler).Methods("GET")
	r.HandleFunc("/health", EchoHandler).Methods("GET")
	r.HandleFunc("/fortune", FortuneHandler).Methods("GET")
	r.HandleFunc("/echo", EchoHandler).Methods("POST")
	http.Handle("/", r)
	log.Fatal(http.ListenAndServe(":80", r))
}
