/*
An obnoxious and utterly irrelevant interview test of engineering knowledge,
and yet a fun programming problem from Cracking the Code Interview:
- given a dictionary of words D, and two words of equal length, w1 and w2
- find the series of one-letter transformations by which w1 becomes w2
- constraint: each intervening word must be in D
Example: 
	w1=DAMP w2=LIKE: DAMP -> LAMP -> LIMP -> LIME -> LIKE

Observations:
	- each edit could be a single letter substitution, deletion, or insertion
	- do deletion/substitution offer any advantage? maybe, but they must occur in pairs to preserve length
	- given w1 and w2, the set of all edits is:
		- replace one letter in w1 with one in w2 (maintaining positions)
		- if the new word is in D, add it to edits
		This gives a data structure mapping words to single-substitution edits closer to w2:
			[w'] -> w'' such that |w'|==|w''| and dist(w'',w2) < dist(w',w2)
			edits : map[string][]string
			*edits can be built incrementally, rather than computing all single-edit word peers

Solutions:
1) Word distance incorporating dictionary lookups?
2) encoding?
3) linear pass?
4) precompute a data structure of all word pairs that are one-edit distance from one another?
	* A graph whose nodes are all w's, and undirected edges indicate one-edit peers
	* reachability means a walk on this graph from w1 to w2
5) priority queue of nearest neighbors or similar scheme? tries?

Brute force: given word w', find w'' s.t. w'' is one letter nearer to w2, by iterating all words in D.
		Until w1 == w2:
		* for all letters in w2, evaluate one-letter changes to w1, and look them up in D
			* (this is the hyper-simlified version, no insert/delete)
DP:
	Dist(w1,w2) = 
		one edit to w1 nearer to w2
		one edit from 2 back nearer to w2

	  L I K E
	D
	A
	M
	P

	D,L:
		is D->L in D vocab?
			* yes: 1 substitution (optimal), add to queue with penalty ps
			* no: 
				* is deletion of D in vocab?
					* yes: add to queue with penalty pd
					* no: blocked
				* is insertion of X after D in vocab?
					* yes: add to queue with penalty pi
					* no: 
			* if none succeed, proceed to other letters
*/

package main 

import (
	"fmt"
	"log"
	"net/http"
	"io"
	"strings"
	"container/list"
)

const termListAddr = "https://raw.githubusercontent.com/dwyl/english-words/master/words.txt"

// Downloads the word list stored at: 
// The words are returned in lowercase.
// The words are unicode code points.
func buildDict() map[string]bool {
	fmt.Println("Downloading word list...")
	resp, err := http.Get(termListAddr)
	if err != nil {
		log.Fatalf("Could not download term list from: "+termListAddr)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	words := strings.Split(string(body), "\n")
	lexicon := make(map[string]bool, len(words))
	for i := range words {
		word := strings.ToLower(words[i])
		lexicon[word] = true
	}

	return lexicon
}

func transform(w1, w2 string, lexicon map[string]bool) []string {
	if len(w1) == 0 || len(w2) == 0 {
		return nil
	}
	if len(w1) != len(w2) {
		return nil
	}
	w1 = strings.ToLower(w1)
	w2 = strings.ToLower(w2)

	// backTrack tracks the path by which a certain word sequence 
	backTrack := make(map[string]string)
	visited := make(map[string]bool)
	q := list.New()
	q.PushBack(w1)
	backTrack[w1] = w1
	visited[w1] = true

	for q.Len() > 0 {
		prev := q.Front().Value.(string)
		q.Remove(q.Front())

		// For each letter in word, edit-toward w2 and check if result is in vocabulary. If so, push it back in the queue.
		buff := []byte(prev)
		for i := range prev {
			buff[i] = w2[i]
			next := string(buff)
			buff[i] = prev[i]
			fmt.Println(i, " ", prev, " -> ", next)

			if next == w2 {
				fmt.Println("Path found!")
				result := []string{next}
				next = prev
				for next != "" && next != w1 {
					result = append(result, next)
					next = backTrack[next]
					fmt.Println(next)
				}
				result = append(result, w1)
				return result
			}

			if _, ok := lexicon[next]; ok && !visited[next]{
				q.PushBack(next)
				backTrack[next] = prev
				visited[next] = true
			}
		}
	}

	fmt.Println("No path found")
	return nil
}

func main() {
	lex := buildDict()
	result := transform("damp", "like", lex)
	fmt.Println(result)
}
