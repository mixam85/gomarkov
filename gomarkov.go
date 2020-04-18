package gomarkov

import (
	"encoding/json"
	"errors"
	"fmt"
	wr "github.com/mroth/weightedrand"
	"math/rand"
	"sync"
	"time"
)

//Tokens are wrapped around a sequence of words to maintain the
//start and end transition counts
const (
	StartToken = "$"
	EndToken   = "^"
)

//Chain is a markov chain instance
type Chain struct {
	Order           int
	statePool       *spool
	frequencyMat    map[int]sparseArray
	frequencyMatSum map[int]int
	weightedrandMat map[int]wr.Chooser
	lock            *sync.RWMutex
}

type chainJSON struct {
	Order    int                 `json:"int"`
	SpoolMap map[string]int      `json:"spool_map"`
	FreqMat  map[int]sparseArray `json:"freq_mat"`
}

//MarshalJSON ...
func (chain Chain) MarshalJSON() ([]byte, error) {
	obj := chainJSON{
		chain.Order,
		chain.statePool.stringMap,
		chain.frequencyMat,
	}
	return json.Marshal(obj)
}

//UnmarshalJSON ...
func (chain *Chain) UnmarshalJSON(b []byte) error {
	var obj chainJSON
	err := json.Unmarshal(b, &obj)
	if err != nil {
		return err
	}
	chain.Order = obj.Order
	intMap := make(map[int]string)
	for k, v := range obj.SpoolMap {
		intMap[v] = k
	}
	chain.statePool = &spool{
		stringMap: obj.SpoolMap,
		intMap:    intMap,
	}
	chain.frequencyMat = obj.FreqMat
	chain.frequencyMatSum = make(map[int]int, 0)
	chain.weightedrandMat = make(map[int]wr.Chooser, 0)
	for i, mat := range chain.frequencyMat {
		chain.frequencyMatSum[i] = mat.sum()
		wrmat := make([]wr.Choice, 0)
		for j, v := range mat {
			wrmat = append(wrmat, wr.Choice{Item: j, Weight: uint(v)})
		}
		chain.weightedrandMat[i] = wr.NewChooser(wrmat...)
	}
	chain.lock = new(sync.RWMutex)
	return nil
}

//NewChain creates an instance of Chain
func NewChain(order int) *Chain {
	chain := Chain{Order: order}
	chain.statePool = &spool{
		stringMap: make(map[string]int),
		intMap:    make(map[int]string),
	}
	chain.frequencyMat = make(map[int]sparseArray, 0)
	chain.frequencyMatSum = make(map[int]int, 0)
	chain.lock = new(sync.RWMutex)
	return &chain
}

//Add adds the transition counts to the chain for a given sequence of words
func (chain *Chain) Add(input []string) {
	startTokens := array(StartToken, chain.Order)
	endTokens := array(EndToken, chain.Order)
	tokens := make([]string, 0)
	tokens = append(tokens, startTokens...)
	tokens = append(tokens, input...)
	tokens = append(tokens, endTokens...)
	pairs := MakePairs(tokens, chain.Order)
	for i := 0; i < len(pairs); i++ {
		pair := pairs[i]
		currentIndex := chain.statePool.add(pair.CurrentState.key())
		nextIndex := chain.statePool.add(pair.NextState)
		chain.lock.Lock()
		if chain.frequencyMat[currentIndex] == nil {
			chain.frequencyMat[currentIndex] = make(sparseArray, 0)
		}
		chain.frequencyMat[currentIndex][nextIndex]++
		chain.frequencyMatSum[currentIndex]++
		chain.lock.Unlock()
	}
}

//TransitionProbability returns the transition probability between two states
func (chain *Chain) TransitionProbability(next string, current NGram) (float64, error) {
	if len(current) != chain.Order {
		return 0, errors.New("N-gram length does not match chain order")
	}
	currentIndex, currentExists := chain.statePool.get(current.key())
	nextIndex, nextExists := chain.statePool.get(next)
	if !currentExists || !nextExists {
		return 0, nil
	}
	arr := chain.frequencyMat[currentIndex]
	sum := float64(chain.frequencyMatSum[currentIndex])
	freq := float64(arr[nextIndex])
	return freq / sum, nil
}

//Generate generates new text based on an initial seed of words
func (chain *Chain) Generate(current NGram) (string, error) {
	if len(current) != chain.Order {
		return "", errors.New("N-gram length does not match chain order")
	}
	currentIndex, currentExists := chain.statePool.get(current.key())
	if !currentExists {
		return "", fmt.Errorf("Unknown ngram %v", current)
	}
	if chain.weightedrandMat != nil {
		i := chain.weightedrandMat[currentIndex].Pick().(int)
		return chain.statePool.intMap[i], nil
	} else {
		arr := chain.frequencyMat[currentIndex]
		sum := chain.frequencyMatSum[currentIndex]
		randN := rand.Intn(sum)
		for i, freq := range arr {
			randN -= freq
			if randN <= 0 {
				return chain.statePool.intMap[i], nil
			}
		}
	}
	return "", nil
}

func init() {
	rand.Seed(time.Now().UnixNano())
}
